"""Girth weld alignment between two ILI runs (spec section 6.3).

Maps each joint of the current run to its joint in the previous run:
1. Joints come from each run's girth-weld rows (log order).
2. Fixed features (valves, tees, fittings) are paired between runs and their
   joints forced 1:1; they split the line into sections.
3. Within a section, current-run lengths are scaled to the previous run's span
   (removes odometer drift), then a banded Needleman-Wunsch alignment over the
   joint-length sequences chooses 1:1, 2:1, 1:2, inserted, or removed matches.
   Gaps are cheaper inside spans listed in repairs_between_runs.csv.

Every tuning value comes from ``settings.alignment``.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ili.config import AlignmentCfg
from ili.schema import ANCHOR_TYPES

JOINT_COLUMNS = ["joint_idx", "gw_id", "start_ft", "end_ft", "length_ft", "anchor_type"]
_ANCHOR_VALUES = {t.value for t in ANCHOR_TYPES}


def build_joints(std: pd.DataFrame) -> pd.DataFrame:
    """One row per joint [weld k, weld k+1) from the run's girth-weld rows, with the type of
    the first fixed feature (valve/tee/fitting) inside it, if any."""
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
    anchor_type = [None] * n
    anchors = std[std.feature_type.isin(_ANCHOR_VALUES)].sort_values("log_dist_ft", kind="stable")
    for kind, x in zip(anchors.feature_type, anchors.log_dist_ft):
        k = int(np.searchsorted(logs, x, side="right") - 1)
        if 0 <= k < n and anchor_type[k] is None:
            anchor_type[k] = str(kind)
    joints["anchor_type"] = pd.Series(anchor_type, dtype=object)
    return joints[JOINT_COLUMNS]


def _span(joints: pd.DataFrame) -> tuple[float, float]:
    return float(joints.start_ft.iloc[0]), float(joints.end_ft.iloc[-1])


def match_anchors(prev_j: pd.DataFrame, curr_j: pd.DataFrame, cfg: AlignmentCfg) -> list[tuple[int, int]]:
    """Pair anchor joints between runs, in line order.

    Current-run positions are scaled onto the previous run's span. A small alignment over
    the two anchor lists pairs same-type anchors within ``anchor_tol_ft``; an anchor with no
    counterpart (not reported in the other run) is skipped at a cost of ``anchor_tol_ft``.
    """
    pa = prev_j[prev_j.anchor_type.notna()]
    ca = curr_j[curr_j.anchor_type.notna()]
    if pa.empty or ca.empty:
        return []
    (p0, p1), (c0, c1) = _span(prev_j), _span(curr_j)
    scale = (p1 - p0) / (c1 - c0)
    ppos = ((pa.start_ft + pa.end_ft) / 2).to_numpy()
    cpos = p0 + ((ca.start_ft + ca.end_ft) / 2 - c0).to_numpy() * scale
    ptype, ctype = pa.anchor_type.to_numpy(), ca.anchor_type.to_numpy()
    n, m, skip = len(pa), len(ca), cfg.anchor_tol_ft

    cost = np.zeros((n + 1, m + 1))
    cost[:, 0] = np.arange(n + 1) * skip
    cost[0, :] = np.arange(m + 1) * skip
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(ppos[i - 1] - cpos[j - 1])
            pair = cost[i - 1, j - 1] + d if ptype[i - 1] == ctype[j - 1] and d <= skip else np.inf
            cost[i, j] = min(pair, cost[i - 1, j] + skip, cost[i, j - 1] + skip)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        d = abs(ppos[i - 1] - cpos[j - 1])
        if ptype[i - 1] == ctype[j - 1] and d <= skip and np.isclose(cost[i, j], cost[i - 1, j - 1] + d):
            pairs.append((int(pa.joint_idx.iloc[i - 1]), int(ca.joint_idx.iloc[j - 1])))
            i, j = i - 1, j - 1
        elif np.isclose(cost[i, j], cost[i - 1, j] + skip):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


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


# (match type, prev joints consumed, curr joints consumed), in tie-break order.
_MOVES = (("1:1", 1, 1), ("2:1", 2, 1), ("1:2", 1, 2), ("removed", 1, 0), ("inserted", 0, 1))


def align_section(prev_len: np.ndarray, curr_len: np.ndarray, prev_repair: np.ndarray,
                  curr_repair: np.ndarray, cfg: AlignmentCfg) -> list[Step]:
    """Globally align two joint-length sequences (curr already drift-scaled).

    Costs are in units of the length tolerance: a 1:1 match costs its mismatch, a 2:1 or 1:2
    match its mismatch plus ``missed_weld_penalty``, and an inserted or removed joint
    ``gap_penalty`` (``gap_penalty_repair`` inside a repaired span). The search is banded to
    ``band_joints`` around the diagonal. Ties resolve in the order of ``_MOVES``.
    """
    n, m = len(prev_len), len(curr_len)
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0

    def gap(repair: bool) -> float:
        return cfg.gap_penalty_repair if repair else cfg.gap_penalty

    for i in range(n + 1):
        if n and m:
            centre = round(i * m / n)
            j_lo, j_hi = max(0, centre - cfg.band_joints), min(m, centre + cfg.band_joints)
        else:
            j_lo, j_hi = 0, m
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                continue
            best, best_k = np.inf, -1
            for k, (kind, dp, dc) in enumerate(_MOVES):
                if i < dp or j < dc or not np.isfinite(cost[i - dp, j - dc]):
                    continue
                if kind == "removed":
                    c = gap(prev_repair[i - 1])
                elif kind == "inserted":
                    c = gap(curr_repair[j - 1])
                else:
                    c = _mismatch(prev_len[i - dp:i].sum(), curr_len[j - dc:j].sum(), cfg)
                    if dp + dc == 3:
                        c += cfg.missed_weld_penalty
                total = cost[i - dp, j - dc] + c
                if total < best:
                    best, best_k = total, k
            cost[i, j], move[i, j] = best, best_k

    if not np.isfinite(cost[n, m]):  # band too narrow for this section: search it in full
        return align_section(prev_len, curr_len, prev_repair, curr_repair,
                             cfg.model_copy(update={"band_joints": max(n, m)}))
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
    pairs = match_anchors(pj, cj, cfg)

    # Prev -> curr log mapping through the paired anchors (drift between anchors is linear).
    (p0, p1), (c0, c1) = _span(pj), _span(cj)
    global_scale = (p1 - p0) / (c1 - c0)
    pmid = np.array([(pj.start_ft[a] + pj.end_ft[a]) / 2 for a, _ in pairs])
    cmid = np.array([(cj.start_ft[b] + cj.end_ft[b]) / 2 for _, b in pairs])

    def prev_to_curr(x: np.ndarray) -> np.ndarray:
        if len(pairs) < 2:
            ref_p, ref_c = (pmid[0], cmid[0]) if pairs else (p0, c0)
            return ref_c + (x - ref_p) / global_scale
        inside = np.interp(x, pmid, cmid)
        lo = cmid[0] + (x - pmid[0]) / global_scale
        hi = cmid[-1] + (x - pmid[-1]) / global_scale
        return np.where(x < pmid[0], lo, np.where(x > pmid[-1], hi, inside))

    spans = np.empty((0, 2)) if repairs is None or repairs.empty else \
        repairs[["start_ft", "end_ft"]].to_numpy(dtype=float)
    curr_rep = _in_spans(((cj.start_ft + cj.end_ft) / 2).to_numpy(), spans)
    prev_rep = _in_spans(prev_to_curr(((pj.start_ft + pj.end_ft) / 2).to_numpy()), spans)

    # Sections between anchor pairs; head and tail borrow the nearest inner scale.
    bounds = [(-1, -1), *pairs, (len(pj), len(cj))]
    sections = []
    for (pa, ca), (pb, cb) in zip(bounds[:-1], bounds[1:]):
        p_idx, c_idx = np.arange(pa + 1, pb), np.arange(ca + 1, cb)
        inner = pa >= 0 and pb < len(pj)
        p_span = pj.length_ft.to_numpy()[p_idx].sum()
        c_span = cj.length_ft.to_numpy()[c_idx].sum()
        scale = p_span / c_span if inner and c_span > 0 and p_span > 0 else None
        sections.append((p_idx, c_idx, scale))
    inner_scales = [s for *_, s in sections if s is not None]
    for k, (p_idx, c_idx, scale) in enumerate(sections):
        if scale is None:
            near = [s for *_, s in (sections[k + 1:] if k == 0 else sections[:k][::-1]) if s is not None]
            sections[k] = (p_idx, c_idx, near[0] if near else (inner_scales[0] if inner_scales else global_scale))

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

    for k, (p_idx, c_idx, scale) in enumerate(sections):
        steps = align_section(pj.length_ft.to_numpy()[p_idx], cj.length_ft.to_numpy()[c_idx] * scale,
                              prev_rep[p_idx], curr_rep[c_idx], cfg)
        for s in steps:
            add(k, s.match_type, tuple(int(p_idx[i]) for i in s.prev_idx),
                tuple(int(c_idx[j]) for j in s.curr_idx), s.mismatch, s.cost, scale, False)
        if k < len(pairs):
            a, b = pairs[k]
            mis = _mismatch(pj.length_ft[a], cj.length_ft[b] * global_scale, cfg)
            add(k + 1, "1:1", (a,), (b,), mis, mis, global_scale, True)

    weld_map = pd.DataFrame(rows, columns=WELD_MAP_COLUMNS)
    changed = _changed_sections(weld_map)
    summary = _summary(weld_map, pj, cj, changed, pairs, sections)
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


def _summary(wm, pj, cj, changed, pairs, sections) -> dict:
    n_prev, n_curr = _joint_counts(wm)
    joints = n_prev + n_curr
    matched = wm.match_type.isin(["1:1", "2:1", "1:2"])
    outside = ~wm.in_repair
    scales = [s for *_, s in sections]
    return {
        "n_prev_joints": int(len(pj)),
        "n_curr_joints": int(len(cj)),
        "n_anchor_pairs": len(pairs),
        "n_sections": len(sections),
        "section_scale_min": float(min(scales)),
        "section_scale_max": float(max(scales)),
        "counts_by_match_type": {t: int((wm.match_type == t).sum()) for t in MATCH_TYPES},
        "counts_by_confidence": {c: int((wm.confidence == c).sum()) for c in ("high", "medium", "low")},
        "alignment_rate": float(joints[matched].sum() / joints.sum()),
        "alignment_rate_outside_repairs": float(joints[matched & outside].sum() / max(joints[outside].sum(), 1)),
        "n_changed_sections": int(len(changed)),
        "n_changed_sections_in_repairs": int(changed.overlaps_repair.sum()) if len(changed) else 0,
    }
