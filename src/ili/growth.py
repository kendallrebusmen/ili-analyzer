"""Per-feature corrosion growth rates (spec section 6.5).

Vectorised methods, each accepting scalars or arrays (NaN in, NaN out). Depths are in inches,
rates in mpy (mils per year), and elapsed time in decimal years. Every tuning value comes from
``settings.tool`` and ``settings.growth``.

``compute_growth`` assembles them over the M4 match table: one row per current-run metal-loss
feature with every method's inputs and result as columns, the governing rate chosen per history
class by ``growth.governing``, length growth, per-segment diagnostics, and a summary.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import get_args

import numpy as np
import pandas as pd

from ili.config import GrowthMethod, Settings, ToolCfg
from ili.match import MILS_PER_IN
from ili.reference import _require_columns, assign_segment, lookup
from ili.units import sigma_from_tolerance

PCT_PER_FRACTION = 100.0


def depth_in(depth_pct, wt_in):
    """Depth in inches = depth_pct / 100 * wt_in."""
    return np.asarray(depth_pct, dtype=float) / PCT_PER_FRACTION * np.asarray(wt_in, dtype=float)


def raw_rate(d_prev_in, d_curr_in, dt_yr):
    """Raw growth in mpy = (d_curr - d_prev) * 1000 / dt. May be negative."""
    delta = np.asarray(d_curr_in, dtype=float) - np.asarray(d_prev_in, dtype=float)
    return delta * MILS_PER_IN / np.asarray(dt_yr, dtype=float)


def sigma_diff_pct(tool: ToolCfg) -> float:
    """Sigma of a depth difference, % of wall = sqrt(2) * sigma (same tool on both runs)."""
    return float(np.sqrt(2.0) * sigma_from_tolerance(tool.depth_tolerance_pct_wt, tool.depth_certainty))


def tolerance_adjusted_rate(d_prev_in, d_curr_in, wt_in, dt_yr, settings: Settings):
    """Upper-bound rate in mpy = max((dd + k * sigma_diff / 100 * wt) * 1000 / dt, min_rate_mpy)."""
    g = settings.growth
    margin = g.k_sigma * sigma_diff_pct(settings.tool) / PCT_PER_FRACTION * np.asarray(wt_in, dtype=float)
    delta = np.asarray(d_curr_in, dtype=float) - np.asarray(d_prev_in, dtype=float)
    rate = (delta + margin) * MILS_PER_IN / np.asarray(dt_yr, dtype=float)
    return np.where(np.isnan(rate), np.nan, np.maximum(rate, g.min_rate_mpy))


def new_feature_rate(d_curr_in, wt_in, dt_yr, settings: Settings):
    """Rate in mpy for a feature new in the current run = (d_curr - threshold_pct / 100 * wt) * 1000 / dt."""
    floor_in = settings.tool.reporting_threshold_pct_wt / PCT_PER_FRACTION * np.asarray(wt_in, dtype=float)
    return (np.asarray(d_curr_in, dtype=float) - floor_in) * MILS_PER_IN / np.asarray(dt_yr, dtype=float)


def full_life_rate(d_curr_in, curr_year, install_year):
    """Rate in mpy = d_curr * 1000 / (curr_year - install_year); NaN if the age is <= 0 or unknown."""
    age = np.asarray(curr_year, dtype=float) - np.asarray(install_year, dtype=float)
    d = np.asarray(d_curr_in, dtype=float)
    valid = age > 0  # False for NaN
    safe_age = np.where(valid, age, 1.0)
    return np.where(valid, d * MILS_PER_IN / safe_age, np.nan)


def default_rate(n: int, settings: Settings):
    """Array of n copies of default_rate_mpy (mpy), or NaN when it is null."""
    value = settings.growth.default_rate_mpy
    return np.full(n, np.nan if value is None else float(value))


# --- assembled growth table -----------------------------------------------------------------------------

METHODS = get_args(GrowthMethod)  # raw, tolerance_adjusted, segment_percentile, new_feature, full_life, default_rate
HISTORY_CLASSES = ("matched", "new", "no_history")
MATCHED_TYPES = ("matched_1to1", "matched_cluster")
NEW_TYPE = "new"
GROWTH_FEATURE_TYPE = "metal_loss"
PIPE_WHAT = "pipe properties table"

INPUT_COLUMNS = (
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_feature_type", "curr_log_dist_ft",
    "prev_wt_nom_in", "curr_wt_nom_in", "prev_depth_pct", "curr_depth_pct", "curr_length_in",
)
FEATURE_COLUMNS = [
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_log_dist_ft", "segment_id",
    "history_class",
    "wt_in", "d_prev_pct", "d_curr_pct", "d_prev_in", "d_curr_in", "dt_yr", "sigma_diff_pct", "k_sigma",
    "install_year",
    *(f"{m}_mpy" for m in METHODS),
    "governing_mpy", "governing_method", "needs_review", "length_in", "length_rate_ipy",
]
SEGMENT_STAT_COLUMNS = [
    "segment_id", "start_ft", "end_ft", "n_features", "n_matched", "raw_mean_mpy", "raw_median_mpy",
    "raw_p_mpy", "pct_negative_raw", "governing_max_mpy",
]


@dataclass
class GrowthResult:
    """One row per current-run metal-loss feature, per-segment diagnostics, and a JSON-ready summary."""

    features: pd.DataFrame
    segments: pd.DataFrame
    summary: dict


def growth_rows(match_table: pd.DataFrame) -> pd.DataFrame:
    """The match-table rows that get a growth rate: metal_loss with a current side, in table order."""
    _require_columns(match_table, INPUT_COLUMNS, "match table")
    keep = (match_table.match_type.isin(MATCHED_TYPES + (NEW_TYPE,))
            & (match_table.curr_feature_type == GROWTH_FEATURE_TYPE))
    return match_table[keep.to_numpy(bool)].reset_index(drop=True)


def wall_thickness(rows: pd.DataFrame, pipe: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Wall (in): curr_wt_nom_in, else prev_wt_nom_in, else the pipe table's wt_nom_in at the position,
    looked up only for the rows that need it."""
    wt = rows.curr_wt_nom_in.to_numpy(float).copy()
    gap = np.isnan(wt)
    wt[gap] = rows.prev_wt_nom_in.to_numpy(float)[gap]
    gap = np.isnan(wt)
    if gap.any():
        pos = rows.curr_log_dist_ft.to_numpy(float)[gap]
        wt[gap] = lookup(pipe, pos, "wt_nom_in", settings.alignment.anchor_tol_ft, what=PIPE_WHAT)
    return wt


def history_class(match_type, d_prev_pct, d_curr_pct, install_year, prev_year: int) -> np.ndarray:
    """``matched`` (1:1 or cluster with both depths), ``new`` (new, in pipe installed by the previous run),
    else ``no_history`` (new pipe, or a matched row missing a depth)."""
    mtype = np.asarray(match_type, dtype=object)
    both = ~np.isnan(np.asarray(d_prev_pct, dtype=float)) & ~np.isnan(np.asarray(d_curr_pct, dtype=float))
    matched = np.isin(mtype, MATCHED_TYPES) & both
    new = (mtype == NEW_TYPE) & (np.asarray(install_year, dtype=float) <= prev_year)
    return np.where(matched, "matched", np.where(new, "new", "no_history")).astype(object)


def segment_percentile(raw_mpy, cls, segment_id, seg_table: pd.DataFrame, settings: Settings):
    """Per-row rate (mpy) and the unfloored percentile by segment id. The ``segment_percentile``-th
    percentile (linear) of the matched raw rates in each segment, negatives included, floored at
    ``min_rate_mpy`` for the rows. NaN for a segment without matched rows and a row outside every segment."""
    g = settings.growth
    raw = np.asarray(raw_mpy, dtype=float)
    seg_ids = np.asarray(segment_id, dtype=object)
    usable = (np.asarray(cls, dtype=object) == "matched") & np.isfinite(raw)
    p_by_seg = {}
    for sid in seg_table.segment_id.astype(str):
        vals = raw[usable & (seg_ids == sid)]
        p_by_seg[sid] = float(np.percentile(vals, g.segment_percentile, method="linear")) if len(vals) else np.nan
    p_rows = np.array([np.nan if s is None else p_by_seg.get(s, np.nan) for s in seg_ids], dtype=float)
    floored = np.where(np.isnan(p_rows), np.nan, np.maximum(p_rows, g.min_rate_mpy))
    return floored, p_by_seg


def governing_rate(rates: pd.DataFrame, cls, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the max (NaN ignored) over the ``<method>_mpy`` columns ``growth.governing`` lists for its
    class, and the first listed method attaining it. A NaN rate has method None."""
    cls = np.asarray(cls, dtype=object)
    gov = np.full(len(rates), np.nan)
    method = np.full(len(rates), None, dtype=object)
    for c in HISTORY_CLASSES:
        rows = cls == c
        names = getattr(settings.growth.governing, c)
        cols = [rates[f"{m}_mpy"].to_numpy(float)[rows] for m in names]
        best = reduce(np.fmax, cols, np.full(int(rows.sum()), np.nan))
        which = np.full(len(best), None, dtype=object)
        for name, col in reversed(list(zip(names, cols))):  # earlier names overwrite: ties go in list order
            which[col == best] = name
        gov[rows], method[rows] = best, which
    return gov, method


def length_growth_rate(length_in, governing_mpy, d_curr_in, settings: Settings) -> np.ndarray:
    """Length growth (in/yr): ``none`` 0; ``fixed_rate`` length_rate_ipy; ``proportional``
    length * governing / 1000 / d_curr (aspect ratio kept), NaN where d_curr <= 0."""
    g = settings.growth
    length = np.asarray(length_in, dtype=float)
    if g.length_growth == "none":
        return np.zeros(len(length))
    if g.length_growth == "fixed_rate":
        return np.full(len(length), float(g.length_rate_ipy))
    d = np.asarray(d_curr_in, dtype=float)
    valid = d > 0  # False for NaN
    rate = length * np.asarray(governing_mpy, dtype=float) / MILS_PER_IN / np.where(valid, d, 1.0)
    return np.where(valid, rate, np.nan)


def _pct_negative(raw: np.ndarray) -> float:
    """Percent of the rates below zero; NaN for none."""
    return float(PCT_PER_FRACTION * np.mean(raw < 0)) if len(raw) else np.nan


def segment_stats(features: pd.DataFrame, seg_table: pd.DataFrame, p_by_seg: dict) -> pd.DataFrame:
    """Per-segment diagnostics of the matched raw rates (``raw_p_mpy`` unfloored) and the governing max."""
    out = []
    for seg in seg_table.itertuples(index=False):
        sid = str(seg.segment_id)
        in_seg = features[features.segment_id == sid]
        raw = in_seg.raw_mpy[in_seg.history_class == "matched"].to_numpy(float)
        gov = in_seg.governing_mpy.to_numpy(float)
        out.append({
            "segment_id": sid, "start_ft": float(seg.start_ft), "end_ft": float(seg.end_ft),
            "n_features": len(in_seg), "n_matched": len(raw),
            "raw_mean_mpy": float(np.mean(raw)) if len(raw) else np.nan,
            "raw_median_mpy": float(np.median(raw)) if len(raw) else np.nan,
            "raw_p_mpy": p_by_seg.get(sid, np.nan),
            "pct_negative_raw": _pct_negative(raw),
            "governing_max_mpy": float(np.nanmax(gov)) if np.isfinite(gov).any() else np.nan,
        })
    return pd.DataFrame(out, columns=SEGMENT_STAT_COLUMNS)


def raw_histogram(raw_mpy, bin_mpy: float) -> dict:
    """Counts of the raw rates in bins of ``bin_mpy`` with edges at its multiples, spanning their range
    (at least one bin); empty lists for no rates."""
    if not bin_mpy > 0:
        raise ValueError("diagnostic_bin_mpy must be positive")
    raw = np.asarray(raw_mpy, dtype=float)
    raw = raw[np.isfinite(raw)]
    if not len(raw):
        return {"bin_edges_mpy": [], "counts": []}
    lo, hi = int(np.floor(raw.min() / bin_mpy)), int(np.ceil(raw.max() / bin_mpy))
    lo -= int(lo * bin_mpy > raw.min())  # guard float rounding at the ends
    hi += int(hi * bin_mpy < raw.max())
    hi = max(hi, lo + 1)
    edges = np.arange(lo, hi + 1) * bin_mpy
    counts, _ = np.histogram(raw, bins=edges)
    return {"bin_edges_mpy": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _summary(features: pd.DataFrame, dt_yr: float, prev_year: int, curr_year: int, settings: Settings,
             prev_run_id: str | None, curr_run_id: str | None) -> dict:
    raw = features.raw_mpy[features.history_class == "matched"].to_numpy(float)
    by_class = features.history_class.value_counts()
    by_method = features.governing_method.value_counts()
    pct_neg = _pct_negative(raw)
    return {
        "prev_run_id": prev_run_id, "curr_run_id": curr_run_id,
        "delta_t_years": float(dt_yr), "prev_year": int(prev_year), "curr_year": int(curr_year),
        "n_features": len(features),
        "counts_by_history_class": {c: int(by_class.get(c, 0)) for c in HISTORY_CLASSES},
        "counts_by_governing_method": {m: int(by_method.get(m, 0)) for m in METHODS},
        "n_needs_review": int(features.needs_review.sum()),
        "pct_negative_raw": None if np.isnan(pct_neg) else pct_neg,
        "raw_histogram": raw_histogram(raw, settings.growth.diagnostic_bin_mpy),
    }


def compute_growth(match_table: pd.DataFrame, dt_yr: float, prev_year: int, curr_year: int, pipe: pd.DataFrame,
                   seg_table: pd.DataFrame, settings: Settings, *, prev_run_id: str | None = None,
                   curr_run_id: str | None = None) -> GrowthResult:
    """Every spec 6.5 method for each current-run metal-loss row of the M4 match table, the governing rate
    by ``growth.governing`` per history class, and length growth. ``prev_year``/``curr_year`` are the runs'
    calendar years; reference positions are ``curr_log_dist_ft``. A position outside the pipe table is a
    ValueError (reference coverage is a data error)."""
    if not dt_yr > 0:
        raise ValueError("dt_yr must be positive")
    g = settings.growth
    rows = growth_rows(match_table)
    n = len(rows)
    pos = rows.curr_log_dist_ft.to_numpy(float)
    wt = wall_thickness(rows, pipe, settings)
    d_prev_pct, d_curr_pct = rows.prev_depth_pct.to_numpy(float), rows.curr_depth_pct.to_numpy(float)
    d_prev, d_curr = depth_in(d_prev_pct, wt), depth_in(d_curr_pct, wt)
    install = lookup(pipe, pos, "install_year", settings.alignment.anchor_tol_ft, what=PIPE_WHAT)
    seg_id = assign_segment(seg_table, pos)
    cls = history_class(rows.match_type, d_prev_pct, d_curr_pct, install, prev_year)
    matched, new = cls == "matched", cls == "new"

    f = pd.DataFrame({
        "match_id": rows.match_id, "match_type": rows.match_type, "prev_feature_id": rows.prev_feature_id,
        "curr_feature_id": rows.curr_feature_id, "curr_log_dist_ft": pos, "segment_id": pd.Series(seg_id, dtype=object),
        "history_class": cls, "wt_in": wt, "d_prev_pct": d_prev_pct, "d_curr_pct": d_curr_pct,
        "d_prev_in": d_prev, "d_curr_in": d_curr, "dt_yr": np.full(n, float(dt_yr)),
        "sigma_diff_pct": np.full(n, sigma_diff_pct(settings.tool)), "k_sigma": np.full(n, float(g.k_sigma)),
        "install_year": install,
    })
    f["raw_mpy"] = np.where(matched, raw_rate(d_prev, d_curr, dt_yr), np.nan)
    f["tolerance_adjusted_mpy"] = np.where(matched, tolerance_adjusted_rate(d_prev, d_curr, wt, dt_yr, settings),
                                           np.nan)
    f["segment_percentile_mpy"], p_by_seg = segment_percentile(f.raw_mpy, cls, seg_id, seg_table, settings)
    f["new_feature_mpy"] = np.where(new, new_feature_rate(d_curr, wt, dt_yr, settings), np.nan)
    f["full_life_mpy"] = full_life_rate(d_curr, curr_year, install)
    f["default_rate_mpy"] = default_rate(n, settings)
    gov, method = governing_rate(f, cls, settings)
    f["governing_mpy"], f["governing_method"] = gov, pd.Series(method, dtype=object)  # object keeps None
    f["needs_review"] = np.isnan(f.governing_mpy.to_numpy(float))
    f["length_in"] = rows.curr_length_in.to_numpy(float)
    f["length_rate_ipy"] = length_growth_rate(f.length_in, f.governing_mpy, d_curr, settings)
    f = f[FEATURE_COLUMNS]
    summary = _summary(f, dt_yr, prev_year, curr_year, settings, prev_run_id, curr_run_id)
    return GrowthResult(f, segment_stats(f, seg_table, p_by_seg), summary)
