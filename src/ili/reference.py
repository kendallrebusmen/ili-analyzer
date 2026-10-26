"""Reference tables keyed to current-run log distance: value lookup by position and segments.

Shared by the match step (pipe properties) and the growth step (segments, HCA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ili.config import Settings
from ili.units import FT_PER_MI

SEGMENT_COLUMNS = ("start_ft", "end_ft", "segment_id")


def _require_columns(df: pd.DataFrame, cols: tuple, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s) {missing}")


def lookup(table: pd.DataFrame, pos_ft, column: str, tol_ft: float, what: str = "reference table") -> np.ndarray:
    """``column`` at each current-run position: the row covering it (``[start_ft, end_ft)``), else the
    nearest row within ``tol_ft`` of the table. ValueError naming the first position that is not
    covered, and for an empty table, a missing column, or a NaN in ``column`` among the rows."""
    _require_columns(table, ("start_ft", "end_ft", column), what)
    if table.empty:
        raise ValueError(f"the {what} has no rows")
    table = table.sort_values("start_ft", kind="stable")
    start, end, val = (table[c].to_numpy(float) for c in ("start_ft", "end_ft", column))
    if np.isnan(val).any():
        raise ValueError(f"{what} row starting at {start[np.isnan(val)][0]:.3f} ft has no {column}")
    x = np.asarray(pos_ft, dtype=float)
    k = np.clip(np.searchsorted(start, x, side="right") - 1, 0, len(start) - 1)
    alt = np.minimum(k + 1, len(start) - 1)  # in a gap between rows the next row may be nearer
    d_k, d_alt = (np.maximum(np.maximum(start[j] - x, x - end[j]), 0.0) for j in (k, alt))
    use_alt = d_alt < d_k
    near, row = np.where(use_alt, d_alt, d_k), np.where(use_alt, alt, k)
    bad = ~(near <= tol_ft)
    if bad.any():
        raise ValueError(f"position {x[bad][0]:.3f} ft is outside the {what} by more than {tol_ft} ft")
    return val[row]


def segment_table(settings: Settings, segments: pd.DataFrame | None, line_start_ft: float,
                  line_end_ft: float) -> pd.DataFrame:
    """Segments for growth statistics, columns ``start_ft, end_ft, segment_id``: the supplied table
    for ``hca_segment`` / ``reference_segments``, or consecutive fixed-length segments."""
    g = settings.growth
    if g.segment_definition != "fixed_length":
        if segments is None:
            raise ValueError(f"segment_definition {g.segment_definition!r} needs a segments table, none was given")
        _require_columns(segments, SEGMENT_COLUMNS, "segments table")
        return segments[list(SEGMENT_COLUMNS)].reset_index(drop=True)
    length = g.fixed_segment_length_mi * FT_PER_MI
    if not length > 0:
        raise ValueError("fixed_segment_length_mi must be positive")
    starts = np.arange(line_start_ft, line_end_ft, length)
    ends = np.minimum(starts + length, line_end_ft)
    ids = [f"FIX-{i:03d}" for i in range(1, len(starts) + 1)]
    return pd.DataFrame({"start_ft": starts, "end_ft": ends, "segment_id": ids})


def assign_segment(seg: pd.DataFrame, pos_ft) -> np.ndarray:
    """Segment id (``[start_ft, end_ft)``) at each position, None outside every segment."""
    _require_columns(seg, SEGMENT_COLUMNS, "segments table")
    x = np.asarray(pos_ft, dtype=float)
    out = np.full(len(x), None, dtype=object)
    if seg.empty:
        return out
    seg = seg.sort_values("start_ft", kind="stable")
    start, end = seg.start_ft.to_numpy(float), seg.end_ft.to_numpy(float)
    ids = seg.segment_id.astype(str).to_numpy(object)
    k = np.clip(np.searchsorted(start, x, side="right") - 1, 0, len(start) - 1)
    inside = (x >= start[k]) & (x < end[k])
    out[inside] = ids[k[inside]]
    return out
