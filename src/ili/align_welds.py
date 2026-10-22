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
