"""Score pipeline stages against synthetic ground truth (spec section 8).

Each stage adds a section to ``outputs/validation_metrics.json``. Only meaningful on
synthetic data, where ``ground_truth*.csv`` exists.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ili.schema import ANOMALY_TYPES

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


# Spec section 9, M4 acceptance: anomaly matching precision and recall against ground truth.
MIN_MATCH_PRECISION = 0.98
MIN_MATCH_RECALL = 0.95
_TRUTH_LINK_CLASSES = ("matched_1to1", "matched_cluster")
_MATCHED_TYPES = ("matched_1to1", "matched_cluster")
_REPORTED_STATUSES = ("reported", "reported_in_cluster")
# Ids each side of a match-table row may hold: (min, max) per run; None is unbounded.
_ROW_SHAPES = {
    "matched_1to1": ((1, 1), (1, 1)),
    "matched_cluster": ((1, None), (1, None)),
    "new": ((0, 0), (1, None)),
    "missing": ((1, None), (0, 0)),
}


def _ids(value) -> list[str]:
    """The feature ids of one match-table cell (``;``-joined for clusters, None when absent)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    return [i for i in str(value).split(";") if i]


def _truth_classes(truth: pd.DataFrame, run: str) -> dict[str, str]:
    """{feature id: truth_class} for one run; one vendor feature id must have one class."""
    classes: dict[str, str] = {}
    for fid, cls in zip(truth[f"{run}_feature_id"], truth.truth_class):
        if pd.isna(fid):
            continue
        fid = str(fid)
        if classes.setdefault(fid, cls) != cls:
            raise ValueError(f"{run} feature {fid} has conflicting truth classes {classes[fid]!r} and {cls!r}")
    return classes


def _fits(n: int, bounds: tuple) -> bool:
    lo, hi = bounds
    return n >= lo and (hi is None or n <= hi)


def _link_list(links) -> list[dict]:
    return [{"prev_feature_id": p, "curr_feature_id": c} for p, c in sorted(links)[:MAX_LISTED]]


def score_matching(match_table: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Link-level precision and recall of a match table against ``ground_truth.csv``.

    Truth links are the distinct (prev, curr) feature-id pairs of anomaly truth rows classed
    matched_1to1 or matched_cluster with both ids reported; a vendor cluster report yields one
    link per true pit it holds. Predicted links are every prev id x every curr id of each
    matched row. ``confusion`` counts, per run, each reported feature id of the table by its
    truth class and its row's match type. A table id absent from the truth (truth from another
    dataset), conflicting truth classes for one id, or missing columns are a ``ValueError``.

    The table must also account for every reported anomaly exactly once. Each run's reported ids
    are its anomaly truth rows with ``{run}_status`` reported or reported_in_cluster, and:
    ``n_unreported`` counts reported ids absent from the table, ``n_duplicated`` ids in more than
    one row, ``n_wrong_side`` rows whose ids do not fit their match type (matched_1to1: one id each
    side; matched_cluster: both sides non-empty; new: current run only; missing: previous run only;
    any other type), and ``n_duplicate_links`` links produced by more than one row. ``passes``
    needs the precision and recall targets, all four counts 0, and at least one truth link.
    """
    for df, cols, what in ((truth, ("feature_type", "prev_feature_id", "curr_feature_id", "truth_class",
                                    "prev_status", "curr_status"),
                            "truth (expected ground_truth.csv)"),
                           (match_table, ("match_type", "prev_feature_id", "curr_feature_id"), "match table")):
        missing = sorted(set(cols) - set(df.columns))
        if missing:
            raise ValueError(f"{what} is missing columns {missing}")

    classes = {run: _truth_classes(truth, run) for run in ("prev", "curr")}
    linked = truth[truth.truth_class.isin(_TRUTH_LINK_CLASSES) & truth.feature_type.isin(ANOMALY_TYPES)
                   & truth.prev_feature_id.notna() & truth.curr_feature_id.notna()]
    true_links = {(str(p), str(c)) for p, c in zip(linked.prev_feature_id, linked.curr_feature_id)}

    anomalies = truth[truth.feature_type.isin(ANOMALY_TYPES)]
    reported = {run: {str(i) for i, st in zip(anomalies[f"{run}_feature_id"], anomalies[f"{run}_status"])
                      if pd.notna(i) and st in _REPORTED_STATUSES} for run in ("prev", "curr")}

    pred_links: set[tuple[str, str]] = set()
    link_rows: dict[tuple[str, str], int] = {}
    seen: dict[str, dict[str, int]] = {"prev": {}, "curr": {}}
    n_wrong_side = 0
    confusion: dict[str, dict[str, dict[str, int]]] = {"prev": {}, "curr": {}}
    for k, (mtype, prev_cell, curr_cell) in enumerate(
            zip(match_table.match_type, match_table.prev_feature_id, match_table.curr_feature_id)):
        ids = {"prev": _ids(prev_cell), "curr": _ids(curr_cell)}
        for run, fids in ids.items():
            for fid in fids:
                if fid not in classes[run]:
                    raise ValueError(f"match table row {k} names {run} feature {fid!r}, which is not in the "
                                     f"truth; is it from another dataset?")
                cell = confusion[run].setdefault(classes[run][fid], {})
                cell[mtype] = cell.get(mtype, 0) + 1
                seen[run][fid] = seen[run].get(fid, 0) + 1
        shape = _ROW_SHAPES.get(mtype)
        if shape is None or not (_fits(len(ids["prev"]), shape[0]) and _fits(len(ids["curr"]), shape[1])):
            n_wrong_side += 1
        if mtype in _MATCHED_TYPES:
            row_links = {(p, c) for p in ids["prev"] for c in ids["curr"]}
            pred_links |= row_links
            for link in row_links:
                link_rows[link] = link_rows.get(link, 0) + 1

    counts = {
        "n_unreported": sum(len(reported[run] - set(seen[run])) for run in ("prev", "curr")),
        "n_duplicated": sum(n > 1 for run in ("prev", "curr") for n in seen[run].values()),
        "n_wrong_side": n_wrong_side,
        "n_duplicate_links": sum(n > 1 for n in link_rows.values()),
    }
    correct = pred_links & true_links
    precision = len(correct) / len(pred_links) if pred_links else 1.0
    recall = len(correct) / len(true_links) if true_links else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "n_pred_links": len(pred_links),
        "n_true_links": len(true_links),
        "n_correct_links": len(correct),
        "false_links": _link_list(pred_links - true_links),
        "missed_links": _link_list(true_links - pred_links),
        **counts,
        "confusion": confusion,
        "passes": bool(precision >= MIN_MATCH_PRECISION and recall >= MIN_MATCH_RECALL
                       and not any(counts.values()) and true_links),
    }


# Spec section 6.5, M5 acceptance: the governing rate is >= the true rate for at least this share of
# features. PLACEHOLDER target.
MIN_GROWTH_CONSERVATISM = 0.90
_GROWTH_HISTORY_CLASSES = ("matched", "new", "no_history")
_GROWTH_TRUTH_CLASSES = ("matched_1to1", "matched_cluster", "new")
_GROWTH_FEATURE_TYPE = "metal_loss"


def _true_growth_by_id(truth: pd.DataFrame) -> dict[str, float]:
    """{current feature id: max growth_mpy of the true features it reports} (NaN rates ignored)."""
    reported = truth[truth.curr_feature_id.notna()]
    rates = pd.to_numeric(reported.growth_mpy, errors="raise").astype(float)
    return {str(k): float(v) for k, v in rates.groupby(reported.curr_feature_id.astype(str)).max().items()}


def _errors(est: np.ndarray, true: np.ndarray) -> dict:
    """Bias (mean est - true) and RMSE in mpy; None for no rows."""
    err = est - true
    return {"bias_mpy": float(np.mean(err)) if len(err) else None,
            "rmse_mpy": float(np.sqrt(np.mean(err ** 2))) if len(err) else None,
            "n": int(len(err))}


def score_growth(features: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Estimated versus true growth of a growth table against ``ground_truth.csv`` (spec 6.5).

    A row's true rate is the max ``growth_mpy`` over the truth rows whose ``curr_feature_id`` is one
    of the row's ``;``-split current ids: one id for 1:1 and new rows, and for a cluster row (or a
    vendor cluster report holding several true pits) the fastest pit, which governs the envelope.
    ``raw`` scores ``raw_mpy`` on ``matched`` history rows; ``governing`` scores ``governing_mpy`` on
    every row that has one. ``conservatism_rate`` is the share of all rows with governing >= true, so
    a row without a governing rate (``n_no_governing``) counts as not conservative; it is also given
    per history class. ``non_conservative`` lists the largest shortfalls.

    The table must score every current feature exactly once, so dropping or repeating rows cannot move
    the rate: ``n_truth_not_scored`` counts the truth's current-run metal_loss ids (``curr_status``
    reported or reported_in_cluster, ``truth_class`` matched_1to1, matched_cluster or new) in no row,
    and ``n_duplicated`` the current ids in more than one row or twice in one cell. ``passes`` needs the
    conservatism target and both counts 0. A current id absent from the truth, one whose truth rows
    have no growth, a ``history_class`` outside matched/new/no_history, or missing columns are a
    ``ValueError``.
    """
    for df, cols, what in ((truth, ("feature_type", "curr_feature_id", "curr_status", "truth_class",
                                    "growth_mpy"), "truth (expected ground_truth.csv)"),
                           (features, ("match_type", "curr_feature_id", "history_class", "raw_mpy",
                                       "governing_mpy"), "growth features")):
        missing = sorted(set(cols) - set(df.columns))
        if missing:
            raise ValueError(f"{what} is missing columns {missing}")

    unknown = sorted({str(c) for c in features.history_class} - set(_GROWTH_HISTORY_CLASSES))
    if unknown:  # an unknown class would silently drop rows from the raw score
        raise ValueError(f"growth features have history_class {unknown}; expected one of "
                         f"{list(_GROWTH_HISTORY_CLASSES)}")
    by_id = _true_growth_by_id(truth)
    true = np.empty(len(features))
    seen: dict[str, int] = {}
    for k, cell in enumerate(features.curr_feature_id):
        ids = _ids(cell)
        for fid in ids:
            seen[fid] = seen.get(fid, 0) + 1
        if not ids:
            raise ValueError(f"growth feature row {k} has no current feature id")
        for fid in ids:
            if fid not in by_id:
                raise ValueError(f"growth feature row {k} names current feature {fid!r}, which is not in the "
                                 f"truth; is it from another dataset?")
        rates = [by_id[fid] for fid in ids if not np.isnan(by_id[fid])]
        if not rates:
            raise ValueError(f"growth feature row {k} ({cell}) has no true growth_mpy in the truth")
        true[k] = max(rates)

    expected = truth[(truth.feature_type == _GROWTH_FEATURE_TYPE) & truth.curr_status.isin(_REPORTED_STATUSES)
                     & truth.truth_class.isin(_GROWTH_TRUTH_CLASSES) & truth.curr_feature_id.notna()]
    not_scored = sorted({str(i) for i in expected.curr_feature_id} - set(seen))
    duplicated = sorted(fid for fid, n in seen.items() if n > 1)

    cls = features.history_class.to_numpy(object)
    raw = features.raw_mpy.to_numpy(float)
    gov = features.governing_mpy.to_numpy(float)
    has_gov = ~np.isnan(gov)
    matched = (cls == "matched") & ~np.isnan(raw)
    ok = has_gov & (gov >= true)  # NaN governing is not conservative
    by_class = {}
    for c in _GROWTH_HISTORY_CLASSES:
        rows = cls == c
        n = int(rows.sum())
        by_class[c] = {"rate": float(ok[rows].mean()) if n else None, "n": n}
    rate = float(ok.mean()) if len(ok) else None

    short = np.where(~ok)[0]
    short = short[np.argsort(-np.nan_to_num(true[short] - gov[short], nan=np.inf), kind="stable")]
    listed = [{"curr_feature_id": str(features.curr_feature_id.iloc[k]), "history_class": str(cls[k]),
               "governing_mpy": None if np.isnan(gov[k]) else float(gov[k]), "true_mpy": float(true[k])}
              for k in short[:MAX_LISTED]]
    return {
        "n_features": int(len(features)),
        "raw": _errors(raw[matched], true[matched]),
        "governing": _errors(gov[has_gov], true[has_gov]),
        "conservatism_rate": rate,
        "conservatism_by_class": by_class,
        "n_no_governing": int((~has_gov).sum()),
        "n_non_conservative": int(len(short)),
        "non_conservative": listed,
        "n_truth_not_scored": len(not_scored),
        "truth_not_scored": not_scored[:MAX_LISTED],
        "n_duplicated": len(duplicated),
        "duplicated": duplicated[:MAX_LISTED],
        "passes": bool(rate is not None and rate >= MIN_GROWTH_CONSERVATISM and not not_scored and not duplicated),
    }


def write_metrics(section: str, metrics: dict, path: Path) -> Path:
    """Merge ``{section: metrics}`` into the JSON file at ``path``."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
