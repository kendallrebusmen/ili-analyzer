"""Score pipeline stages against synthetic ground truth (spec section 8).

Each stage adds a section to ``outputs/validation_metrics.json``. Only meaningful on
synthetic data, where ``ground_truth*.csv`` exists.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAX_LISTED = 20


def _run_joints(std: pd.DataFrame, gw_to_true: dict) -> list[tuple[str, int, int]]:
    """(gw_id, true start weld, true end weld) for each joint of a run."""
    welds = std[std.feature_type == "girth_weld"].sort_values("log_dist_ft", kind="stable")
    ids = [str(g) for g in welds.us_gw_id]
    unknown = [g for g in ids if g not in gw_to_true]
    if unknown:
        raise ValueError(f"{len(unknown)} weld id(s) of run {std.run_id.iloc[0]} are not in the truth file "
                         f"(first: {unknown[0]}); is it from another dataset?")
    true = [gw_to_true[g] for g in ids]
    return [(ids[k], true[k], true[k + 1]) for k in range(len(ids) - 1)]


def _truth_groups(prev_j, curr_j, weld_truth: pd.DataFrame) -> list[dict]:
    """Joint groups between consecutive welds present in both runs, with the true match type."""
    common = sorted({s for _, s, _ in prev_j} | {prev_j[-1][2]} if prev_j else set())
    common = sorted(set(common) & ({s for _, s, _ in curr_j} | ({curr_j[-1][2]} if curr_j else set())))
    replaced = set(weld_truth.loc[weld_truth.in_replacement.astype(bool), "true_weld_idx"].astype(int))
    edges = [-float("inf"), *common, float("inf")]
    groups = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = [g for g, s, _ in prev_j if lo <= s < hi]
        c = [g for g, s, _ in curr_j if lo <= s < hi]
        if not p and not c:
            continue
        in_rep = any(lo < w < hi for w in replaced)
        shape = (len(p), len(c))
        kind = {(1, 1): "1:1", (2, 1): "2:1", (1, 2): "1:2"}.get(shape) if not in_rep and lo != -float("inf") \
            and hi != float("inf") else None
        groups.append({"type": kind or "gap", "prev": p, "curr": c, "in_replacement": in_rep})
    return groups


def score_alignment(weld_map: pd.DataFrame, prev_std: pd.DataFrame, curr_std: pd.DataFrame,
                    weld_truth: pd.DataFrame) -> dict:
    """Joint-level accuracy of a weld map against ``ground_truth_welds``.

    Truth groups are the joints between consecutive welds reported in both runs: one prev
    and one curr joint is 1:1, two and one is 2:1, one and two is 1:2; any other shape, or
    a group containing a replaced weld, is all removed/inserted. A group is correct when the
    weld map reproduces exactly its joints and type and each of its joints is in exactly one row,
    so duplicated or contradictory rows count against the map. A weld id that is not a joint of
    its run is a ``ValueError``.
    """
    missing = sorted({"true_weld_idx", "prev_gw_id", "curr_gw_id", "in_replacement"} - set(weld_truth.columns))
    if missing:
        raise ValueError(f"weld truth is missing columns {missing}; expected ground_truth_welds.csv")
    wt = weld_truth.copy()
    prev_map = {str(g): int(t) for g, t in zip(wt.prev_gw_id, wt.true_weld_idx) if pd.notna(g)}
    curr_map = {str(g): int(t) for g, t in zip(wt.curr_gw_id, wt.true_weld_idx) if pd.notna(g)}
    prev_j, curr_j = _run_joints(prev_std, prev_map), _run_joints(curr_std, curr_map)
    groups = _truth_groups(prev_j, curr_j, wt)

    known = {"prev": {g for g, _, _ in prev_j}, "curr": {g for g, _, _ in curr_j}}
    rows: list[tuple[str, frozenset, frozenset]] = []
    used: dict[tuple[str, str], list[int]] = {}
    for k, r in enumerate(weld_map.itertuples(index=False)):
        sides = {}
        for run in ("prev", "curr"):
            sides[run] = frozenset(str(v) for v in (getattr(r, f"{run}_gw_id"), getattr(r, f"{run}_gw_id_2"))
                                   if pd.notna(v))
            for g in sides[run]:
                if g not in known[run]:
                    raise ValueError(f"weld map row {k} names {run} joint {g}, which is not in that run")
                used.setdefault((run, g), []).append(k)
        rows.append((r.match_type, sides["prev"], sides["curr"]))

    def rows_of(g) -> set[int] | None:
        """The rows holding the group's joints, or None if any joint is in no row or several."""
        hits = [used.get(key, []) for key in [("prev", p) for p in g["prev"]] + [("curr", c) for c in g["curr"]]]
        return None if any(len(h) != 1 for h in hits) else {h[0] for h in hits}

    correct_joints, wrong = 0, []
    for g in groups:
        hit = rows_of(g)
        if hit is None:
            ok = False
        elif g["type"] == "gap":
            ok = all((t, bool(p), bool(c)) in {("removed", True, False), ("inserted", False, True)}
                     for t, p, c in (rows[k] for k in hit))
        else:
            ok = len(hit) == 1 and rows[hit.pop()] == (g["type"], frozenset(g["prev"]), frozenset(g["curr"]))
        g["correct"] = ok
        if ok:
            correct_joints += len(g["prev"]) + len(g["curr"])
        elif len(wrong) < MAX_LISTED:
            wrong.append({"type": g["type"], "prev": g["prev"], "curr": g["curr"]})
    n_joints = len(prev_j) + len(curr_j)
    missed = [g for g in groups if g["type"] in ("2:1", "1:2")]
    replaced = [g for g in groups if g["in_replacement"]]
    return {
        "joint_accuracy": correct_joints / n_joints if n_joints else 1.0,
        "n_joints": n_joints,
        "n_groups": len(groups),
        "n_wrong_groups": sum(not g["correct"] for g in groups),
        "wrong_groups": wrong,
        "n_missed_weld_groups": len(missed),
        "missed_welds_handled": all(g["correct"] for g in missed),
        "replaced_section_flagged": bool(replaced) and all(g["correct"] for g in replaced),
    }


def write_metrics(section: str, metrics: dict, path: Path) -> Path:
    """Merge ``{section: metrics}`` into the JSON file at ``path``."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
