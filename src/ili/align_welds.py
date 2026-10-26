"""Girth weld alignment between two ILI runs (spec section 6.3).

Maps each joint of the current run to its joint in the previous run:
1. Joints come from each run's girth-weld rows (log order).
2. Fixed features (valves, tees, fittings) are paired between runs by position, and a
   current -> previous log map is fitted to the pairs. Paired anchor joints of matching
   length are forced 1:1 and split the line into sections. Joints beyond the other run's
   first or last weld have no counterpart.
3. Within a section, current-run lengths are drift-scaled (line-wide scale, then the
   median ratio of the section's confident matches), and a banded Needleman-Wunsch
   alignment over the joint-length sequences chooses 1:1, 2:1, 1:2, inserted, or removed
   matches. Gaps are cheaper inside spans listed in repairs_between_runs.csv, and free at
   the line's open ends.

Every tuning value comes from ``settings.alignment``.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT, AlignmentCfg, load_settings
from ili.ingest import load_standardized
from ili.schema import ANCHOR_TYPES
from ili.validation import score_alignment, write_metrics

JOINT_COLUMNS = ["joint_idx", "gw_id", "start_ft", "end_ft", "length_ft", "anchor_type", "anchor_ft"]
_ANCHOR_VALUES = {t.value for t in ANCHOR_TYPES}


def build_joints(std: pd.DataFrame) -> pd.DataFrame:
    """One row per joint [weld k, weld k+1) from the run's girth-weld rows, with the type and
    log distance of the first fixed feature (valve/tee/fitting) inside it, if any."""
    welds = std[std.feature_type == "girth_weld"].sort_values("log_dist_ft", kind="stable")
    logs = welds.log_dist_ft.to_numpy(dtype=float)
    ids = welds.us_gw_id.astype(object).to_numpy()
    n = max(len(logs) - 1, 0)
    joints = pd.DataFrame({
        "joint_idx": np.arange(n),
        "gw_id": ids[:n],
        "start_ft": logs[:n],
        "end_ft": logs[1:n + 1],
        "length_ft": np.diff(logs) if n else np.array([], dtype=float),
    })
    anchor_type, anchor_ft = [None] * n, np.full(n, np.nan)
    anchors = std[std.feature_type.isin(_ANCHOR_VALUES)].sort_values("log_dist_ft", kind="stable")
    for kind, x in zip(anchors.feature_type, anchors.log_dist_ft):
        k = int(np.searchsorted(logs, x, side="right") - 1)
        if 0 <= k < n and anchor_type[k] is None:
            anchor_type[k], anchor_ft[k] = str(kind), float(x)
    joints["anchor_type"] = pd.Series(anchor_type, dtype=object)
    joints["anchor_ft"] = anchor_ft
    joints = joints[JOINT_COLUMNS]
    joints.attrs["run_id"] = std.run_id.iloc[0] if "run_id" in std and len(std) else None
    return joints


def _span(joints: pd.DataFrame) -> tuple[float, float]:
    return float(joints.start_ft.iloc[0]), float(joints.end_ft.iloc[-1])


@dataclass(frozen=True)
class LogMap:
    """Linear map between the runs' log distances: prev = offset + scale * curr."""
    scale: float
    offset: float

    def to_prev(self, x):
        return self.offset + self.scale * x

    def to_curr(self, x):
        return (x - self.offset) / self.scale


def _endpoint_map(prev_j: pd.DataFrame, curr_j: pd.DataFrame) -> LogMap:
    """Map the runs' first and last welds onto each other (right when both cover the same welds)."""
    (p0, p1), (c0, c1) = _span(prev_j), _span(curr_j)
    scale = (p1 - p0) / (c1 - c0)
    return LogMap(scale, p0 - scale * c0)


def _fit_map(ppos: np.ndarray, cpos: np.ndarray, fallback: LogMap) -> LogMap:
    """Least-squares line through paired anchor positions; a single pair keeps the fallback scale."""
    if len(ppos) == 0:
        return fallback
    if len(ppos) == 1:
        return LogMap(fallback.scale, float(ppos[0] - fallback.scale * cpos[0]))
    scale, offset = np.polyfit(cpos, ppos, 1)
    return LogMap(float(scale), float(offset))


def _pair_anchor_lists(ppos, ptype, cpos, ctype, tol) -> list[tuple[int, int]]:
    """Order-preserving pairing of two anchor lists (curr positions already mapped onto prev):
    same type and |d| <= tol to pair, cost |d|; an unpaired anchor costs ``tol``."""
    n, m = len(ppos), len(cpos)
    cost = np.zeros((n + 1, m + 1))
    cost[:, 0] = np.arange(n + 1) * tol
    cost[0, :] = np.arange(m + 1) * tol
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(ppos[i - 1] - cpos[j - 1])
            pair = cost[i - 1, j - 1] + d if ptype[i - 1] == ctype[j - 1] and d <= tol else np.inf
            cost[i, j] = min(pair, cost[i - 1, j] + tol, cost[i, j - 1] + tol)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        d = abs(ppos[i - 1] - cpos[j - 1])
        if ptype[i - 1] == ctype[j - 1] and d <= tol and np.isclose(cost[i, j], cost[i - 1, j - 1] + d):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(cost[i, j], cost[i - 1, j] + tol):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _anchors(joints: pd.DataFrame) -> pd.DataFrame:
    return joints[joints.anchor_type.notna()]


def match_anchors(prev_j: pd.DataFrame, curr_j: pd.DataFrame, cfg: AlignmentCfg) -> list[tuple[int, int]]:
    """Pair anchor joints between runs, in line order, as (prev joint_idx, curr joint_idx).

    Anchors are paired by position (the feature's log distance) under a current -> previous log
    map. Two starting maps are tried: the runs' endpoints, which is right when both runs cover
    the same welds, and the offset most same-type anchor pairs agree on, which is right when
    one run starts or ends elsewhere. Each is refined by pairing, refitting the map to the pairs
    by least squares, and pairing again until the pairs stop changing. The result with the most
    pairs wins (then the smaller total residual). An anchor with no counterpart is skipped.
    """
    pa, ca = _anchors(prev_j), _anchors(curr_j)
    if pa.empty or ca.empty:
        return []
    ppos, cpos = pa.anchor_ft.to_numpy(), ca.anchor_ft.to_numpy()
    ptype, ctype = pa.anchor_type.to_numpy(), ca.anchor_type.to_numpy()
    tol = cfg.anchor_tol_ft

    starts = [_endpoint_map(prev_j, curr_j)]
    offsets = (ppos[:, None] - cpos[None, :])[ptype[:, None] == ctype[None, :]]
    if len(offsets):
        votes = (np.abs(offsets[:, None] - offsets[None, :]) <= tol).sum(axis=1)
        starts.append(LogMap(1.0, float(offsets[np.argmax(votes)])))

    best_key, best_pairs = None, []
    for start in starts:
        log_map, pairs = start, None
        for _ in range(len(ppos) + 1):  # each round can only settle the pairing further
            new = _pair_anchor_lists(ppos, ptype, log_map.to_prev(cpos), ctype, tol)
            if new == pairs:
                break
            pairs = new
            i, j = [a for a, _ in pairs], [b for _, b in pairs]
            log_map = _fit_map(ppos[i], cpos[j], start)
        resid = float(np.abs(ppos[i] - log_map.to_prev(cpos[j])).sum())
        key = (-len(pairs), resid)
        if best_key is None or key < best_key:
            best_key, best_pairs = key, pairs
    return [(int(pa.joint_idx.iloc[i]), int(ca.joint_idx.iloc[j])) for i, j in best_pairs]


# ---------------------------------------------------------------------------
# Section alignment (banded Needleman-Wunsch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    match_type: str  # "1:1" | "2:1" | "1:2" | "inserted" | "removed"
    prev_idx: tuple[int, ...]
    curr_idx: tuple[int, ...]
    mismatch: float  # |a - b| / tolerance; NaN for inserted/removed
    cost: float


def _tolerance(a: float, b: float, cfg: AlignmentCfg) -> float:
    return max(cfg.joint_length_tol_ft, cfg.joint_length_tol_pct / 100 * max(a, b))


def _mismatch(a: float, b: float, cfg: AlignmentCfg) -> float:
    return abs(a - b) / _tolerance(a, b, cfg)


def _gap_cost(repair: bool, cfg: AlignmentCfg) -> float:
    return cfg.gap_penalty_repair if repair else cfg.gap_penalty


# (match type, prev joints consumed, curr joints consumed), in tie-break order.
_MOVES = (("1:1", 1, 1), ("2:1", 2, 1), ("1:2", 1, 2), ("removed", 1, 0), ("inserted", 0, 1))


def align_section(prev_len: np.ndarray, curr_len: np.ndarray, prev_repair: np.ndarray,
                  curr_repair: np.ndarray, cfg: AlignmentCfg, free_start: bool = False,
                  free_end: bool = False) -> list[Step]:
    """Globally align two joint-length sequences (curr already drift-scaled).

    Costs are in units of the length tolerance: a 1:1 match costs its mismatch, a 2:1 or 1:2
    match its mismatch plus ``missed_weld_penalty``, and an inserted or removed joint
    ``gap_penalty`` (``gap_penalty_repair`` inside a repaired span). The search is banded: row i
    covers the columns between the start-aligned diagonal (j = i) and the end-aligned one
    (j = i - (n - m)), widened by ``band_joints`` each side. So a head or tail offset of any size
    is searched, the end is always reachable, and the band only limits how far a path may stray
    from both alignments in mid-section. Ties resolve in the order of ``_MOVES``.

    ``free_start``/``free_end`` make gaps before the first (after the last) match cost nothing,
    for the head and tail of a line, where the two runs need not begin or end at the same weld.
    Charged there, the unavoidable gaps would push the alignment into coincidental length
    matches instead.
    """
    n, m = len(prev_len), len(curr_len)
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0
    rows = np.arange(n + 1)
    j_lo = np.maximum(0, np.minimum(rows, rows - (n - m)) - cfg.band_joints)
    j_hi = np.minimum(m, np.maximum(rows, rows - (n - m)) + cfg.band_joints)

    for i in range(n + 1):
        for j in range(j_lo[i], j_hi[i] + 1):
            if i == 0 and j == 0:
                continue
            best, best_k = np.inf, -1
            for k, (kind, dp, dc) in enumerate(_MOVES):
                if i < dp or j < dc or not np.isfinite(cost[i - dp, j - dc]):
                    continue
                if kind == "removed":
                    free = (free_start and j == 0) or (free_end and j == m)
                    c = 0.0 if free else _gap_cost(prev_repair[i - 1], cfg)
                elif kind == "inserted":
                    free = (free_start and i == 0) or (free_end and i == n)
                    c = 0.0 if free else _gap_cost(curr_repair[j - 1], cfg)
                else:
                    c = _mismatch(prev_len[i - dp:i].sum(), curr_len[j - dc:j].sum(), cfg)
                    if dp + dc == 3:
                        c += cfg.missed_weld_penalty
                total = cost[i - dp, j - dc] + c
                if total < best:
                    best, best_k = total, k
            cost[i, j], move[i, j] = best, best_k

    steps: list[Step] = []
    i, j = n, m
    while i or j:
        kind, dp, dc = _MOVES[move[i, j]]
        pi, ci = tuple(range(i - dp, i)), tuple(range(j - dc, j))
        mis = np.nan if kind in ("removed", "inserted") else _mismatch(prev_len[i - dp:i].sum(),
                                                                      curr_len[j - dc:j].sum(), cfg)
        steps.append(Step(kind, pi, ci, float(mis), float(cost[i, j] - cost[i - dp, j - dc])))
        i, j = i - dp, j - dc
    return steps[::-1]


# ---------------------------------------------------------------------------
# Whole-line alignment
# ---------------------------------------------------------------------------

WELD_MAP_COLUMNS = [
    "section_id", "match_type", "prev_gw_id", "prev_gw_id_2", "curr_gw_id", "curr_gw_id_2",
    "prev_start_ft", "curr_start_ft", "prev_length_ft", "curr_length_ft", "curr_length_scaled_ft",
    "mismatch", "cost", "confidence", "in_repair", "is_anchor",
]
MATCH_TYPES = ["1:1", "2:1", "1:2", "inserted", "removed"]


@dataclass
class AlignmentResult:
    weld_map: pd.DataFrame
    changed_sections: pd.DataFrame
    summary: dict


def _confidence(kind: str, mismatch: float, in_repair: bool, cfg: AlignmentCfg) -> str:
    if kind in ("inserted", "removed"):
        return "high" if in_repair else "low"
    level = "high" if mismatch <= cfg.confidence_bands.high else (
        "medium" if mismatch <= cfg.confidence_bands.medium else "low")
    return "medium" if kind in ("2:1", "1:2") and level == "high" else level


def _in_spans(x: np.ndarray, spans: np.ndarray) -> np.ndarray:
    if len(spans) == 0 or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    return ((x[:, None] >= spans[None, :, 0]) & (x[:, None] <= spans[None, :, 1])).any(axis=1)


def align_welds(prev_std: pd.DataFrame, curr_std: pd.DataFrame, settings, repairs: pd.DataFrame | None = None
                ) -> AlignmentResult:
    cfg = settings.alignment
    pj, cj = build_joints(prev_std), build_joints(curr_std)
    if pj.empty or cj.empty:
        raise ValueError("each run needs at least two girth welds to align")
    if repairs is not None and not {"start_ft", "end_ft"} <= set(repairs.columns):
        raise ValueError("repairs need start_ft and end_ft columns (current-run log distance)")
    pairs = match_anchors(pj, cj, cfg)
    plen, clen = pj.length_ft.to_numpy(), cj.length_ft.to_numpy()

    # Current -> previous log map fitted to the paired anchors (the run endpoints if none);
    # between anchors drift is taken as linear.
    ppos = np.array([pj.anchor_ft[a] for a, _ in pairs])
    cpos = np.array([cj.anchor_ft[b] for _, b in pairs])
    log_map = _fit_map(ppos, cpos, _endpoint_map(pj, cj))

    def prev_to_curr(x: np.ndarray) -> np.ndarray:
        if len(pairs) < 2:
            return log_map.to_curr(x)
        lo = cpos[0] + (x - ppos[0]) / log_map.scale
        hi = cpos[-1] + (x - ppos[-1]) / log_map.scale
        return np.where(x < ppos[0], lo, np.where(x > ppos[-1], hi, np.interp(x, ppos, cpos)))

    spans = np.empty((0, 2)) if repairs is None or repairs.empty else         repairs[["start_ft", "end_ft"]].to_numpy(dtype=float)
    cmid = ((cj.start_ft + cj.end_ft) / 2).to_numpy()
    pmid = prev_to_curr(((pj.start_ft + pj.end_ft) / 2).to_numpy())  # in curr log distance
    curr_rep, prev_rep = _in_spans(cmid, spans), _in_spans(pmid, spans)

    # A joint lying beyond the other run's first or last weld (by more than the anchor position
    # tolerance) has no counterpart, so it is removed/inserted outright. Left to the section
    # alignment, lengths alone could pair it with a joint far from its position.
    tol = cfg.anchor_tol_ft
    (c0, c1), (p0, p1) = _span(cj), prev_to_curr(np.array(_span(pj)))
    p_lead, p_trail = pmid < c0 - tol, pmid > c1 + tol
    c_lead, c_trail = cmid < p0 - tol, cmid > p1 + tol

    # A paired anchor bounds a section (its joints forced 1:1) only when the two anchor joints
    # agree in length. Otherwise a weld beside the anchor was missed in one run, and the anchor
    # joint is left to the section alignment, where it can take part in a 2:1 or 1:2.
    bounds = [(a, b) for a, b in pairs
              if _mismatch(plen[a], clen[b] * log_map.scale, cfg) <= cfg.confidence_bands.medium]
    edges = [(-1, -1), *bounds, (len(pj), len(cj))]
    for a, b in bounds:  # a section boundary is matched, never beyond the other run
        p_lead[a] = p_trail[a] = c_lead[b] = c_trail[b] = False

    def align(k, p_idx, c_idx, scale):
        return align_section(plen[p_idx], clen[c_idx] * scale, prev_rep[p_idx], curr_rep[c_idx], cfg,
                             free_start=k == 0, free_end=k == len(edges) - 2)

    # Drift correction in two passes: align each section at the line-wide scale, then rescale it
    # by the median length ratio of its confident 1:1 matches and align again. A median follows
    # the section's own drift but ignores the one joint a local odometer jump or a missed weld
    # changes, which a span ratio would spread over every joint.
    sections = []
    for k, ((pa, ca), (pb, cb)) in enumerate(zip(edges[:-1], edges[1:])):
        p_idx, c_idx = np.arange(pa + 1, pb), np.arange(ca + 1, cb)
        p_idx = p_idx[~(p_lead | p_trail)[p_idx]]
        c_idx = c_idx[~(c_lead | c_trail)[c_idx]]
        steps = align(k, p_idx, c_idx, log_map.scale)
        ratios = [plen[p_idx[s.prev_idx[0]]] / clen[c_idx[s.curr_idx[0]]] for s in steps
                  if s.match_type == "1:1" and s.mismatch <= cfg.confidence_bands.high]
        scale = float(np.median(ratios)) if ratios else log_map.scale
        sections.append((p_idx, c_idx, scale, align(k, p_idx, c_idx, scale) if ratios else steps))

    rows = []

    def add(section_id, kind, pi, ci, mismatch, cost, scale, is_anchor):
        rep = bool(prev_rep[list(pi)].any() or curr_rep[list(ci)].any())
        c_len = float(cj.length_ft.to_numpy()[list(ci)].sum()) if ci else np.nan
        rows.append({
            "section_id": section_id,
            "match_type": kind,
            "prev_gw_id": pj.gw_id[pi[0]] if pi else None,
            "prev_gw_id_2": pj.gw_id[pi[1]] if len(pi) > 1 else None,
            "curr_gw_id": cj.gw_id[ci[0]] if ci else None,
            "curr_gw_id_2": cj.gw_id[ci[1]] if len(ci) > 1 else None,
            "prev_start_ft": float(pj.start_ft[pi[0]]) if pi else np.nan,
            "curr_start_ft": float(cj.start_ft[ci[0]]) if ci else np.nan,
            "prev_length_ft": float(pj.length_ft.to_numpy()[list(pi)].sum()) if pi else np.nan,
            "curr_length_ft": c_len,
            "curr_length_scaled_ft": c_len * scale,
            "mismatch": mismatch,
            "cost": cost,
            "confidence": _confidence(kind, mismatch, rep, cfg),
            "in_repair": rep,
            "is_anchor": is_anchor,
        })

    def add_unmatched(section_id, p_mask, c_mask, scale):
        for i in np.flatnonzero(p_mask):
            add(section_id, "removed", (int(i),), (), np.nan, _gap_cost(prev_rep[i], cfg), scale, False)
        for j in np.flatnonzero(c_mask):
            add(section_id, "inserted", (), (int(j),), np.nan, _gap_cost(curr_rep[j], cfg), scale, False)

    add_unmatched(0, p_lead, c_lead, sections[0][2])
    for k, (p_idx, c_idx, scale, steps) in enumerate(sections):
        for s in steps:
            add(k, s.match_type, tuple(int(p_idx[i]) for i in s.prev_idx),
                tuple(int(c_idx[j]) for j in s.curr_idx), s.mismatch, s.cost, scale, False)
        if k < len(bounds):
            a, b = bounds[k]
            mis = _mismatch(plen[a], clen[b] * log_map.scale, cfg)
            add(k + 1, "1:1", (a,), (b,), mis, mis, log_map.scale, True)
    add_unmatched(len(sections) - 1, p_trail, c_trail, sections[-1][2])

    weld_map = pd.DataFrame(rows, columns=WELD_MAP_COLUMNS)
    changed = _changed_sections(weld_map)
    summary = _summary(weld_map, pj, cj, changed, pairs, bounds, log_map, [sc for *_, sc, _ in sections])
    return AlignmentResult(weld_map, changed, summary)


def _joint_counts(wm: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    n_prev = wm.prev_gw_id.notna().astype(int) + wm.prev_gw_id_2.notna().astype(int)
    n_curr = wm.curr_gw_id.notna().astype(int) + wm.curr_gw_id_2.notna().astype(int)
    return n_prev, n_curr


def _changed_sections(wm: pd.DataFrame) -> pd.DataFrame:
    cols = ["section_id", "prev_start_ft", "prev_end_ft", "curr_start_ft", "curr_end_ft", "match_types",
            "n_prev_joints", "n_curr_joints", "overlaps_repair"]
    changed = (wm.match_type != "1:1").to_numpy()
    n_prev, n_curr = _joint_counts(wm)
    out, k = [], 0
    while k < len(wm):
        if not changed[k]:
            k += 1
            continue
        end = k
        while end + 1 < len(wm) and changed[end + 1]:
            end += 1
        g = wm.iloc[k:end + 1]
        out.append({
            "section_id": int(g.section_id.iloc[0]),
            "prev_start_ft": g.prev_start_ft.min(),
            "prev_end_ft": (g.prev_start_ft + g.prev_length_ft).max(),
            "curr_start_ft": g.curr_start_ft.min(),
            "curr_end_ft": (g.curr_start_ft + g.curr_length_ft).max(),
            "match_types": ";".join(t for t in MATCH_TYPES if t in set(g.match_type)),
            "n_prev_joints": int(n_prev.iloc[k:end + 1].sum()),
            "n_curr_joints": int(n_curr.iloc[k:end + 1].sum()),
            "overlaps_repair": bool(g.in_repair.any()),
        })
        k = end + 1
    return pd.DataFrame(out, columns=cols)


def _summary(wm, pj, cj, changed, pairs, bounds, log_map, scales) -> dict:
    n_prev, n_curr = _joint_counts(wm)
    joints = n_prev + n_curr
    matched = wm.match_type.isin(["1:1", "2:1", "1:2"])
    outside = ~wm.in_repair
    return {
        "prev_run_id": str(pj.attrs.get("run_id")),
        "curr_run_id": str(cj.attrs.get("run_id")),
        "n_prev_joints": int(len(pj)),
        "n_curr_joints": int(len(cj)),
        "n_anchor_pairs": len(pairs),
        "n_anchor_bounds": len(bounds),
        "n_sections": len(scales),
        "line_scale": log_map.scale,
        "section_scale_min": float(min(scales)),
        "section_scale_max": float(max(scales)),
        "counts_by_match_type": {t: int((wm.match_type == t).sum()) for t in MATCH_TYPES},
        "counts_by_confidence": {c: int((wm.confidence == c).sum()) for c in ("high", "medium", "low")},
        "alignment_rate": float(joints[matched].sum() / joints.sum()),
        "alignment_rate_outside_repairs": float(joints[matched & outside].sum() / max(joints[outside].sum(), 1)),
        "n_changed_sections": int(len(changed)),
        "n_changed_sections_in_repairs": int(changed.overlaps_repair.sum()) if len(changed) else 0,
    }


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------

DRAFT_LABEL = "DRAFT – for review by qualified personnel."


def comparison_dir(out_dir: Path, prev_run_id: str, curr_run_id: str) -> Path:
    return Path(out_dir) / f"{prev_run_id}_vs_{curr_run_id}"


def write_alignment(result: AlignmentResult, out_dir: Path) -> dict[str, Path]:
    s = result.summary
    folder = comparison_dir(out_dir, s["prev_run_id"], s["curr_run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "weld_map.csv": folder / "weld_map.csv",
        "changed_sections.csv": folder / "changed_sections.csv",
        "alignment_summary.json": folder / "alignment_summary.json",
    }
    result.weld_map.to_csv(paths["weld_map.csv"], index=False, lineterminator="\n")
    result.changed_sections.to_csv(paths["changed_sections.csv"], index=False, lineterminator="\n")
    paths["alignment_summary.json"].write_text(json.dumps({"label": DRAFT_LABEL, **s}, indent=2), encoding="utf-8")
    return paths


def load_weld_map(path: Path) -> pd.DataFrame:
    """Read a ``weld_map.csv`` back with ids as text (``"1"``, never ``1.0``; blank is None)
    and the flags as bool."""
    id_cols = ["prev_gw_id", "prev_gw_id_2", "curr_gw_id", "curr_gw_id_2"]
    wm = pd.read_csv(path, dtype={c: str for c in id_cols + ["match_type", "confidence"]},
                     keep_default_na=False, na_values={c: [""] for c in WELD_MAP_COLUMNS if c not in id_cols
                                                       and c not in ("match_type", "confidence")})
    for c in id_cols:
        wm[c] = wm[c].astype(object).where(wm[c] != "", None)
    for c in ("in_repair", "is_anchor"):
        wm[c] = wm[c].astype(str).str.lower().eq("true")
    return wm


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.align_welds",
                                 description="Align girth welds between two ILI runs. " + DRAFT_LABEL)
    ap.add_argument("--prev", type=Path, required=True, help="standardized CSV of the previous run")
    ap.add_argument("--curr", type=Path, required=True, help="standardized CSV of the current run")
    ap.add_argument("--repairs", type=Path, help="repairs_between_runs.csv (current-run log distance)")
    ap.add_argument("--truth", type=Path, help="ground_truth_welds.csv (synthetic data only) to score against")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs", help="output directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prev, curr = load_standardized(args.prev), load_standardized(args.curr)
        repairs = pd.read_csv(args.repairs) if args.repairs else None
        result = align_welds(prev, curr, load_settings(args.settings), repairs)
        paths = write_alignment(result, args.out)
        score = None
        if args.truth:
            truth = pd.read_csv(args.truth, dtype={"prev_gw_id": str, "curr_gw_id": str})
            score = score_alignment(result.weld_map, prev, curr, truth)
            write_metrics("weld_alignment", score, Path(args.out) / "validation_metrics.json")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    print(DRAFT_LABEL)
    print(f"{s['prev_run_id']} vs {s['curr_run_id']}: {s['n_prev_joints']} / {s['n_curr_joints']} joints, "
          f"{s['n_anchor_pairs']} anchor pairs, {s['n_sections']} sections")
    print(f"alignment rate {s['alignment_rate']:.4f} (outside repairs {s['alignment_rate_outside_repairs']:.4f})")
    print("matches: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_match_type"].items()))
    print(f"changed sections: {s['n_changed_sections']} ({s['n_changed_sections_in_repairs']} in repaired spans)")
    if len(result.changed_sections):
        print(result.changed_sections.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if score is not None:
        print(f"joint accuracy vs ground truth: {score['joint_accuracy']:.4f} "
              f"(replaced section flagged: {score['replaced_section_flagged']}, "
              f"missed welds handled: {score['missed_welds_handled']})")
    print(f"wrote {paths['weld_map.csv'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
