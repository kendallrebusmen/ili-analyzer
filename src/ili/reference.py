"""Reference tables keyed to current-run log distance: value lookup by position, segments, and the repairs log.

Shared by the match step (pipe properties), the growth step (segments, HCA), and every step that reads
``repairs_between_runs.csv`` (``load_repairs`` / ``normalize_repairs``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ili.config import Settings
from ili.units import FT_PER_MI

SEGMENT_COLUMNS = ("start_ft", "end_ft", "segment_id")
REPAIR_COLUMNS = ("start_ft", "end_ft", "repair_type")
REPAIR_TYPES = ("sleeve", "cutout", "recoat", "pipe_replacement")  # spec 5.2 repairs_between_runs vocabulary
_REPAIR_TYPE_SEPARATORS = re.compile(r"[\s-]+")  # a run of spaces or hyphens becomes one underscore


def _require_columns(df: pd.DataFrame, cols: tuple, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s) {missing}")


def _repair_type_key(value) -> str | None:
    """``repair_type`` stripped, casefolded, with spaces/hyphens as ``_``; None for a blank."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip().casefold()
    return _REPAIR_TYPE_SEPARATORS.sub("_", text) if text else None


def normalize_repairs(repairs: pd.DataFrame, what: str = "repairs table") -> pd.DataFrame:
    """A copy of a ``repairs_between_runs`` table with ``repair_type`` normalised (strip, casefold, spaces and
    hyphens to ``_``), so " PIPE REPLACEMENT " is ``pipe_replacement``. ValueError naming ``what`` (the file)
    for a missing ``start_ft``, ``end_ft`` or ``repair_type`` column, or for any type (blank included) outside
    ``REPAIR_TYPES`` (spec 5.2), listing the offending values."""
    _require_columns(repairs, REPAIR_COLUMNS, what)
    keys = [_repair_type_key(v) for v in repairs.repair_type]
    bad = list(dict.fromkeys("blank" if k is None else repr(v)
                             for v, k in zip(repairs.repair_type, keys) if k not in REPAIR_TYPES))
    if bad:
        raise ValueError(f"{what}: repair_type {', '.join(bad)} is not one of {', '.join(REPAIR_TYPES)} "
                         "(spec 5.2)")
    out = repairs.copy()
    out["repair_type"] = pd.Series(keys, index=repairs.index, dtype=object)
    return out


def load_repairs(path: Path) -> pd.DataFrame:
    """Read ``repairs_between_runs.csv`` and ``normalize_repairs`` it; every error names the file (an empty or
    malformed file included)."""
    try:
        repairs = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path}: unreadable CSV: {exc}") from None
    return normalize_repairs(repairs, str(path))


def lookup(table: pd.DataFrame, pos_ft, column: str, tol_ft: float, what: str = "reference table",
           allow_nan: bool = False) -> np.ndarray:
    """``column`` at each current-run position: the row covering it (``[start_ft, end_ft)``), else the
    nearest row within ``tol_ft`` of the table. ValueError naming the first position that is not
    covered, and for an empty table, a missing column, or (unless ``allow_nan``, which returns it as NaN)
    a NaN in ``column`` among the rows used."""
    _require_columns(table, ("start_ft", "end_ft", column), what.removesuffix(" table"))
    if table.empty:
        raise ValueError(f"the {what} has no rows")
    table = table.sort_values("start_ft", kind="stable")
    start, end, val = (table[c].to_numpy(float) for c in ("start_ft", "end_ft", column))
    x = np.asarray(pos_ft, dtype=float)
    row, near = _nearest_row(start, end, x)
    bad = ~(near <= tol_ft)
    if bad.any():
        raise ValueError(f"position {x[bad][0]:.3f} ft is outside the {what} by more than {tol_ft} ft")
    got = val[row]
    if not allow_nan and np.isnan(got).any():  # only the rows the positions resolve to matter
        r = row[np.isnan(got)][0]
        raise ValueError(f"{what.removesuffix(' table')} row starting at {start[r]:.3f} ft has no {column}")
    return got


def _nearest_row(start: np.ndarray, end: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by ``start``: the index of the row covering each position (``[start, end)``), else the
    nearest row, and the distance (ft) to it (0 inside a row, NaN for a NaN position)."""
    k = np.clip(np.searchsorted(start, x, side="right") - 1, 0, len(start) - 1)
    alt = np.minimum(k + 1, len(start) - 1)  # in a gap between rows the next row may be nearer
    d_k, d_alt = (np.maximum(np.maximum(start[j] - x, x - end[j]), 0.0) for j in (k, alt))
    use_alt = d_alt < d_k
    return np.where(use_alt, alt, k), np.where(use_alt, d_alt, d_k)


def covered(table: pd.DataFrame, pos_ft, tol_ft: float) -> np.ndarray:
    """True where ``lookup`` would find a row for the position: inside a row or within ``tol_ft`` of the table.
    False for a NaN position and for every position of an empty table."""
    x = np.asarray(pos_ft, dtype=float)
    _require_columns(table, ("start_ft", "end_ft"), "reference")
    if table.empty:
        return np.zeros(len(x), dtype=bool)
    table = table.sort_values("start_ft", kind="stable")
    _, near = _nearest_row(table.start_ft.to_numpy(float), table.end_ft.to_numpy(float), x)
    return near <= tol_ft  # False for NaN


def check_segments(seg: pd.DataFrame) -> None:
    """ValueError for a repeated segment id or overlapping segments (sorted by start, a start before the
    previous end): either would pool or double-count features in the per-segment statistics."""
    _require_columns(seg, SEGMENT_COLUMNS, "segments table")
    ids = seg.segment_id.astype(str)
    dup = ids[ids.duplicated()]
    if not dup.empty:
        raise ValueError(f"segments table repeats segment_id {dup.iloc[0]!r}; segment ids must be unique")
    s = seg.sort_values("start_ft", kind="stable")
    start, end = s.start_ft.to_numpy(float), s.end_ft.to_numpy(float)
    over = np.flatnonzero(start[1:] < end[:-1])
    if len(over):
        k = over[0]
        a, b = s.segment_id.astype(str).iloc[k], s.segment_id.astype(str).iloc[k + 1]
        raise ValueError(f"segments {a!r} and {b!r} overlap ({b!r} starts at {start[k + 1]:.3f} ft, before "
                         f"{a!r} ends at {end[k]:.3f} ft)")


def segment_table(settings: Settings, segments: pd.DataFrame | None, line_start_ft: float,
                  line_end_ft: float) -> pd.DataFrame:
    """Segments for growth statistics, columns ``start_ft, end_ft, segment_id``: the supplied table
    for ``hca_segment`` / ``reference_segments`` (ValueError for a repeated id or overlapping segments), or
    consecutive fixed-length segments."""
    g = settings.growth
    if g.segment_definition != "fixed_length":
        if segments is None:
            raise ValueError(f"segment_definition {g.segment_definition!r} needs a segments table, none was given")
        _require_columns(segments, SEGMENT_COLUMNS, "segments table")
        seg = segments[list(SEGMENT_COLUMNS)].reset_index(drop=True)
        check_segments(seg)
        return seg
    length = g.fixed_segment_length_mi * FT_PER_MI
    if not length > 0:
        raise ValueError("fixed_segment_length_mi must be positive")
    starts = np.arange(line_start_ft, line_end_ft, length)
    ends = np.append(starts[1:], line_end_ft)[:len(starts)]  # each ends exactly where the next starts
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
