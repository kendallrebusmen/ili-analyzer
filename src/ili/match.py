"""Anomaly matching between two ILI runs (spec section 6.4).

Placement and helpers: each anomaly is placed in the weld map by its log distance, positions
are carried between runs through the matched joints, and clock and time differences are
computed on the circular clock and in decimal years.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ili.units import CLOCK_MINUTES

ANOMALY_TYPES = ("metal_loss", "dent", "gouge")
_MATCHED = ("1:1", "2:1", "1:2")
DAYS_PER_YEAR = 365.25


def clock_diff_min(a, b) -> np.ndarray:
    """Circular clock difference in minutes (0 to CLOCK_MINUTES / 2); NaN where either is missing."""
    x, y = (pd.Series(v, dtype="Float64").to_numpy(dtype=float, na_value=np.nan) for v in (a, b))
    d = np.abs(x - y) % CLOCK_MINUTES
    return np.minimum(d, CLOCK_MINUTES - d)


def _run_date(std: pd.DataFrame):
    dates = std.run_date.dropna().unique()
    if len(dates) != 1 or std.run_date.isna().any():
        raise ValueError(f"each run needs exactly one run_date, found {list(std.run_date.unique())}")
    return dates[0]


def delta_t_years(prev_std: pd.DataFrame, curr_std: pd.DataFrame) -> float:
    days = (_run_date(curr_std) - _run_date(prev_std)).days
    if days <= 0:
        raise ValueError("the current run must be later than the previous run")
    return days / DAYS_PER_YEAR


class PositionMap:
    """Piecewise-linear map between the runs' log distances through matched joints."""

    def __init__(self, prev_knots: np.ndarray, curr_knots: np.ndarray):
        self._prev, self._curr = prev_knots, curr_knots

    @classmethod
    def from_weld_map(cls, wm: pd.DataFrame) -> PositionMap:
        m = wm[wm.match_type.isin(_MATCHED)]
        if m.empty:
            raise ValueError("the weld map has no matched joints")
        ps, cs = m.prev_start_ft.to_numpy(float), m.curr_start_ft.to_numpy(float)
        pe, ce = ps + m.prev_length_ft.to_numpy(float), cs + m.curr_length_ft.to_numpy(float)
        prev, curr = np.concatenate([ps, pe]), np.concatenate([cs, ce])
        order = np.lexsort((prev, curr))
        return cls(prev[order], curr[order])

    @staticmethod
    def _interp(x, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        slope = (dst[-1] - dst[0]) / (src[-1] - src[0])
        out = np.interp(x, src, dst)
        out = np.where(x < src[0], dst[0] + (x - src[0]) * slope, out)
        return np.where(x > src[-1], dst[-1] + (x - src[-1]) * slope, out)

    def to_prev(self, x_curr) -> np.ndarray:
        return self._interp(x_curr, self._curr, self._prev)

    def to_curr(self, x_prev) -> np.ndarray:
        order = np.argsort(self._prev, kind="stable")
        return self._interp(x_prev, self._prev[order], self._curr[order])


def place(std: pd.DataFrame, wm: pd.DataFrame, run: str, pmap: PositionMap) -> pd.DataFrame:
    """The run's anomalies with the weld-map row holding each one and both runs' positions.

    A row's own-run span is [start, start + length) in that run's columns; rows with no side in
    the run hold no features of it. ``edge_dist_ft`` is the distance to the nearer span end."""
    if run not in ("prev", "curr"):
        raise ValueError(f"run must be 'prev' or 'curr', got {run!r}")
    out = std[std.feature_type.isin(ANOMALY_TYPES)].reset_index(drop=True)
    x = out.log_dist_ft.to_numpy(dtype=float)
    start = wm[f"{run}_start_ft"].to_numpy(dtype=float)
    end = start + wm[f"{run}_length_ft"].to_numpy(dtype=float)
    rows = np.flatnonzero(~np.isnan(start) & ~np.isnan(end))
    rows = rows[np.argsort(start[rows], kind="stable")]
    k = np.searchsorted(start[rows], x, side="right") - 1
    cand = rows[np.maximum(k, 0)]
    hit = (k >= 0) & (x < end[cand])
    wm_row = np.where(hit, cand, -1)
    out["wm_row"] = wm_row
    out["in_matched_row"] = hit & wm.match_type.isin(_MATCHED).to_numpy()[cand]
    mapped = pmap.to_curr(x) if run == "prev" else pmap.to_prev(x)
    out["pos_prev_ft"], out["pos_curr_ft"] = (x, mapped) if run == "prev" else (mapped, x)
    out["edge_dist_ft"] = np.where(hit, np.minimum(x - start[cand], end[cand] - x), np.nan)
    return out
