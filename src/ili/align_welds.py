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
