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

import argparse
import json
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd

from ili.align_welds import DRAFT_LABEL, comparison_dir
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT, GrowthMethod, Settings, ToolCfg, load_settings
from ili.ingest import load_standardized
from ili.match import MILS_PER_IN, delta_t_years, load_match_table, parse_flag_column
from ili.reference import _require_columns, assign_segment, check_segments, lookup, segment_table
from ili.schema import FeatureType
from ili.units import sigma_from_tolerance
from ili.validation import score_growth, write_metrics

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


def half_life_rate(d_curr_in, curr_year, install_year):
    """Rate in mpy = d_curr * 1000 / ((curr_year - install_year) / 2), i.e. twice the full-life rate;
    NaN if the age is <= 0 or unknown."""
    return 2.0 * full_life_rate(d_curr_in, curr_year, install_year)


def shrinking_fixed_rate(d_prev_in, d_curr_in, cls, settings: Settings):
    """``shrinking_rate_mpy`` where the row is ``matched`` and its depth dropped (d_curr < d_prev), else NaN;
    NaN everywhere when the setting is null."""
    d_prev, d_curr = np.asarray(d_prev_in, dtype=float), np.asarray(d_curr_in, dtype=float)
    value = settings.growth.shrinking_rate_mpy
    if value is None:
        return np.full(len(d_prev), np.nan)
    shrank = (np.asarray(cls, dtype=object) == "matched") & (d_curr < d_prev)  # False for NaN
    return np.where(shrank, float(value), np.nan)


def in_recoat(pos_ft, repairs: pd.DataFrame | None) -> np.ndarray:
    """True where the position (ft) lies inside any ``repair_type == "recoat"`` span, ends inclusive. All
    False when there is no repairs table or it has no ``repair_type`` column."""
    pos = np.asarray(pos_ft, dtype=float)
    out = np.zeros(len(pos), dtype=bool)
    if repairs is None or "repair_type" not in repairs.columns:
        return out
    missing = [c for c in ("start_ft", "end_ft") if c not in repairs.columns]
    if missing:
        raise ValueError(f"repairs table is missing column(s): {', '.join(missing)}")
    for span in repairs[repairs.repair_type == RECOAT_TYPE].itertuples(index=False):
        out |= (pos >= float(span.start_ft)) & (pos <= float(span.end_ft))
    return out


def default_rate(n: int, settings: Settings):
    """Array of n copies of default_rate_mpy (mpy), or NaN when it is null."""
    value = settings.growth.default_rate_mpy
    return np.full(n, np.nan if value is None else float(value))


# --- assembled growth table -----------------------------------------------------------------------------

METHODS = get_args(GrowthMethod)  # raw ... default_rate, half_life, joint_percentile, shrinking_fixed
RECOAT_TYPE = "recoat"
HISTORY_CLASSES = ("matched", "new", "no_history")
# Why a row needs review, in this order in ``review_reasons``: no governing rate; a method of its class's
# governing rule is NaN; governing below ``min_rate_mpy`` (not floored, so flagged); M4 flagged the match.
REVIEW_REASONS = ("no_governing", "incomplete_rule", "below_min_rate", "match_review")
REASON_SEP = ";"
MATCHED_TYPES = ("matched_1to1", "matched_cluster")
NEW_TYPE = "new"
GROWTH_FEATURE_TYPE = "metal_loss"
PIPE_WHAT = "pipe properties table"
MATCH_M11_COLUMNS = ("curr_us_gw_id", "curr_surface", "curr_ml_origin")  # the CLI needs them for joint rates

INPUT_COLUMNS = (
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_feature_type", "curr_log_dist_ft",
    "prev_wt_nom_in", "curr_wt_nom_in", "prev_depth_pct", "curr_depth_pct", "curr_length_in",
    "confidence", "implausible", "needs_review",
)
FEATURE_COLUMNS = [
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_log_dist_ft", "segment_id",
    "joint_id", "surface", "ml_origin", "in_recoat", "in_stats",
    "history_class", "confidence", "implausible", "match_needs_review",
    "wt_in", "d_prev_pct", "d_curr_pct", "d_prev_in", "d_curr_in", "dt_yr", "sigma_diff_pct", "k_sigma",
    "install_year",
    *(f"{m}_mpy" for m in METHODS),
    "governing_mpy", "governing_method", "needs_review", "review_reasons", "length_in", "length_rate_ipy",
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


def _none_for_nan(values: pd.Series, kind: type) -> pd.Series:
    """Object column of ``kind`` values with None (not NaN) where blank, as ``match.load_match_table`` gives."""
    return pd.Series([None if pd.isna(v) else kind(v) for v in values], dtype=object)


def _text_column(rows: pd.DataFrame, name: str) -> pd.Series:
    """Object column of the match-table text column ``name`` with None where blank or absent."""
    if name not in rows.columns:
        return pd.Series([None] * len(rows), dtype=object)
    return pd.Series([None if pd.isna(v) else str(v) for v in rows[name]], dtype=object)


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


def segment_percentile(raw_mpy, cls, segment_id, seg_table: pd.DataFrame, settings: Settings, in_stats=None):
    """Per-row rate (mpy) and the unfloored percentile by segment id. The ``segment_percentile``-th
    percentile (linear) of the matched raw rates in each segment, negatives included, floored at
    ``min_rate_mpy`` for the rows. Rows with ``in_stats`` False (recoated) stay out of the distribution but
    still get the rate. NaN for a segment without matched rows and a row outside every segment."""
    g = settings.growth
    raw = np.asarray(raw_mpy, dtype=float)
    seg_ids = np.asarray(segment_id, dtype=object)
    usable = (np.asarray(cls, dtype=object) == "matched") & np.isfinite(raw)
    if in_stats is not None:
        usable &= np.asarray(in_stats, dtype=bool)
    p_by_seg = {}
    for sid in seg_table.segment_id.astype(str):
        vals = raw[usable & (seg_ids == sid)]
        p_by_seg[sid] = float(np.percentile(vals, g.segment_percentile, method="linear")) if len(vals) else np.nan
    p_rows = np.array([np.nan if s is None else p_by_seg.get(s, np.nan) for s in seg_ids], dtype=float)
    floored = np.where(np.isnan(p_rows), np.nan, np.maximum(p_rows, g.min_rate_mpy))
    return floored, p_by_seg


def joint_percentile(raw_mpy, cls, joint_id, in_stats, settings: Settings):
    """Per-row rate (mpy) and the unfloored percentile by joint id. The ``joint_percentile``-th percentile
    (linear) of the raw rates of ``matched`` rows with ``in_stats`` True in the row's joint, floored at
    ``min_rate_mpy`` for the rows. NaN when the joint has fewer than ``joint_percentile_min_matches`` such
    rows or the row has no joint (None/NaN)."""
    g = settings.growth
    raw = np.asarray(raw_mpy, dtype=float)
    joints = np.array([None if pd.isna(j) else j for j in joint_id], dtype=object)
    usable = ((np.asarray(cls, dtype=object) == "matched") & np.isfinite(raw)
              & np.asarray(in_stats, dtype=bool) & np.array([j is not None for j in joints], dtype=bool))
    p_by_joint = {}
    for jid in set(joints[usable]):
        vals = raw[usable & (joints == jid)]
        if len(vals) >= g.joint_percentile_min_matches:
            p_by_joint[jid] = float(np.percentile(vals, g.joint_percentile, method="linear"))
    p_rows = np.array([np.nan if j is None else p_by_joint.get(j, np.nan) for j in joints], dtype=float)
    return np.where(np.isnan(p_rows), np.nan, np.maximum(p_rows, g.min_rate_mpy)), p_by_joint


def _list_governing(rates: pd.DataFrame, cls, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """The ``conservative`` rule: per row, the max (NaN ignored) over the ``<method>_mpy`` columns
    ``growth.governing`` lists for its class, and the first listed method attaining it."""
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


def _pit_to_pit_governing(rates: pd.DataFrame, cls, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """The ``pit_to_pit`` rule (spec 6.5): ``matched`` rows that did not shrink (d_curr >= d_prev) take ``raw``
    floored at ``min_rate_mpy``, those that shrank take ``shrinking_fixed``; ``new`` rows take
    ``joint_percentile`` when finite, else ``half_life``; ``no_history`` rows take ``half_life``."""
    n = len(rates)
    col = lambda name: rates[f"{name}_mpy"].to_numpy(float)  # noqa: E731
    matched, new = cls == "matched", cls == "new"
    shrank = matched & (rates.d_curr_in.to_numpy(float) < rates.d_prev_in.to_numpy(float))
    grew = matched & ~shrank
    raw_floored = np.fmax(col("raw"), settings.growth.min_rate_mpy)
    use_joint = new & np.isfinite(col("joint_percentile"))
    gov = np.full(n, np.nan)
    method = np.full(n, None, dtype=object)
    for mask, name, values in ((grew, "raw", np.where(np.isnan(col("raw")), np.nan, raw_floored)),
                               (shrank, "shrinking_fixed", col("shrinking_fixed")),
                               (use_joint, "joint_percentile", col("joint_percentile")),
                               (~matched & ~use_joint, "half_life", col("half_life"))):
        gov[mask] = values[mask]
        method[mask] = name
    method[np.isnan(gov)] = None
    return gov, method


def governing_rate(rates: pd.DataFrame, cls, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the governing rate (mpy) and the method that supplied it, by ``growth.governing_profile``:
    ``conservative`` maxes over the methods ``growth.governing`` lists for the row's class; ``pit_to_pit``
    follows the fixed spec 6.5 logic. A NaN rate has method None."""
    cls = np.asarray(cls, dtype=object)
    if settings.growth.governing_profile == "pit_to_pit":
        return _pit_to_pit_governing(rates, cls, settings)
    return _list_governing(rates, cls, settings)


def review_flags(rates: pd.DataFrame, cls, governing_mpy, match_review, settings: Settings) -> dict[str, np.ndarray]:
    """Per ``REVIEW_REASONS`` name, a bool array of the rows it applies to. ``incomplete_rule`` (``conservative``
    only; ``pit_to_pit`` falls back by design and ``no_governing`` catches a missing rate): any method
    ``growth.governing`` lists for the row's class is NaN, so the max was taken over fewer rates than the
    rule intends. ``below_min_rate``: governing < ``min_rate_mpy`` (governing itself is never floored).
    ``match_review``: the M4 match row's ``needs_review``, carried so the match evidence reaches M6."""
    cls = np.asarray(cls, dtype=object)
    gov = np.asarray(governing_mpy, dtype=float)
    incomplete = np.zeros(len(rates), dtype=bool)
    if settings.growth.governing_profile == "conservative":
        for c in HISTORY_CLASSES:
            rows = cls == c
            for m in getattr(settings.growth.governing, c):
                incomplete |= rows & np.isnan(rates[f"{m}_mpy"].to_numpy(float))
    return {"no_governing": np.isnan(gov), "incomplete_rule": incomplete,
            "below_min_rate": gov < settings.growth.min_rate_mpy,  # False for NaN
            "match_review": np.asarray(match_review, dtype=bool)}


def join_reasons(flags: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """``needs_review`` (any flag) and ``review_reasons`` (flag names joined by ``;`` in the order given,
    None when there are none)."""
    names = list(flags)
    reasons = np.full(n, None, dtype=object)
    for k in range(n):
        hit = [r for r in names if flags[r][k]]
        reasons[k] = REASON_SEP.join(hit) if hit else None
    return np.array([r is not None for r in reasons], dtype=bool), reasons


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
    g = settings.growth
    raw = features.raw_mpy[features.history_class == "matched"].to_numpy(float)
    by_class = features.history_class.value_counts()
    by_method = features.governing_method.value_counts()
    pct_neg = _pct_negative(raw)
    reasons = [r.split(REASON_SEP) if isinstance(r, str) else [] for r in features.review_reasons]
    return {
        "prev_run_id": prev_run_id, "curr_run_id": curr_run_id,
        "delta_t_years": float(dt_yr), "prev_year": int(prev_year), "curr_year": int(curr_year),
        "n_features": len(features),
        "governing_profile": g.governing_profile,
        "n_excluded_recoated": int((features.in_recoat & ~features.in_stats).sum()),
        "joint_percentile": float(g.joint_percentile),
        "joint_percentile_min_matches": int(g.joint_percentile_min_matches),
        "shrinking_rate_mpy": None if g.shrinking_rate_mpy is None else float(g.shrinking_rate_mpy),
        "counts_by_history_class": {c: int(by_class.get(c, 0)) for c in HISTORY_CLASSES},
        "counts_by_governing_method": {m: int(by_method.get(m, 0)) for m in METHODS},
        "n_needs_review": int(features.needs_review.sum()),
        "counts_by_review_reason": {r: sum(r in x for x in reasons) for r in REVIEW_REASONS},
        "pct_negative_raw": None if np.isnan(pct_neg) else pct_neg,
        "raw_histogram": raw_histogram(raw, settings.growth.diagnostic_bin_mpy),
    }


def compute_growth(match_table: pd.DataFrame, dt_yr: float, prev_year: int, curr_year: int, pipe: pd.DataFrame,
                   seg_table: pd.DataFrame, settings: Settings, *, prev_run_id: str | None = None,
                   curr_run_id: str | None = None, repairs: pd.DataFrame | None = None) -> GrowthResult:
    """Every spec 6.5 method for each current-run metal-loss row of the M4 match table, the governing rate
    by ``growth.governing`` per history class, and length growth. ``prev_year``/``curr_year`` are the runs'
    calendar years; reference positions are ``curr_log_dist_ft``. A position outside the pipe table is a
    ValueError (reference coverage is a data error). The governing rate is never floored; ``needs_review``
    and ``review_reasons`` flag a row whose rate is missing, incomplete under its rule, or below
    ``min_rate_mpy``, or whose M4 match row needs review (``REVIEW_REASONS``); M4's ``confidence``,
    ``implausible`` and ``needs_review`` (as ``match_needs_review``) are carried as columns. ``repairs``
    (``repairs_between_runs``, optional) marks rows inside ``recoat`` spans as ``in_recoat``; with
    ``exclude_recoated_from_stats`` they are ``in_stats`` False and stay out of the segment and joint
    percentile distributions (they still get the rates)."""
    if not dt_yr > 0:
        raise ValueError("dt_yr must be positive")
    g = settings.growth
    check_segments(seg_table)  # segment_table checks too; a table built by hand must not pool segments
    rows = growth_rows(match_table)
    n = len(rows)
    pos = rows.curr_log_dist_ft.to_numpy(float)
    wt = wall_thickness(rows, pipe, settings)
    d_prev_pct, d_curr_pct = rows.prev_depth_pct.to_numpy(float), rows.curr_depth_pct.to_numpy(float)
    d_prev, d_curr = depth_in(d_prev_pct, wt), depth_in(d_curr_pct, wt)
    # An unknown install year is NaN, not an error: full_life is NaN and the row is no_history, flagged.
    install = lookup(pipe, pos, "install_year", settings.alignment.anchor_tol_ft, what=PIPE_WHAT, allow_nan=True)
    seg_id = assign_segment(seg_table, pos)
    cls = history_class(rows.match_type, d_prev_pct, d_curr_pct, install, prev_year)
    matched, new = cls == "matched", cls == "new"
    match_review = np.array([not pd.isna(v) and bool(v) for v in rows.needs_review], dtype=bool)  # None: unflagged

    in_rc = in_recoat(pos, repairs)
    in_stats = ~(in_rc & bool(g.exclude_recoated_from_stats))
    joint_id = _text_column(rows, "curr_us_gw_id")

    f = pd.DataFrame({
        "match_id": rows.match_id, "match_type": rows.match_type, "prev_feature_id": rows.prev_feature_id,
        "curr_feature_id": rows.curr_feature_id, "curr_log_dist_ft": pos, "segment_id": pd.Series(seg_id, dtype=object),
        "joint_id": joint_id, "surface": _text_column(rows, "curr_surface"),
        "ml_origin": _text_column(rows, "curr_ml_origin"), "in_recoat": in_rc, "in_stats": in_stats,
        "history_class": cls, "confidence": _none_for_nan(rows.confidence, str),
        "implausible": _none_for_nan(rows.implausible, bool), "match_needs_review": match_review, "wt_in": wt, "d_prev_pct": d_prev_pct, "d_curr_pct": d_curr_pct,
        "d_prev_in": d_prev, "d_curr_in": d_curr, "dt_yr": np.full(n, float(dt_yr)),
        "sigma_diff_pct": np.full(n, sigma_diff_pct(settings.tool)), "k_sigma": np.full(n, float(g.k_sigma)),
        "install_year": install,
    })
    f["raw_mpy"] = np.where(matched, raw_rate(d_prev, d_curr, dt_yr), np.nan)
    f["tolerance_adjusted_mpy"] = np.where(matched, tolerance_adjusted_rate(d_prev, d_curr, wt, dt_yr, settings),
                                           np.nan)
    f["segment_percentile_mpy"], p_by_seg = segment_percentile(f.raw_mpy, cls, seg_id, seg_table, settings,
                                                                    in_stats)
    f["new_feature_mpy"] = np.where(new, new_feature_rate(d_curr, wt, dt_yr, settings), np.nan)
    f["full_life_mpy"] = full_life_rate(d_curr, curr_year, install)
    f["default_rate_mpy"] = default_rate(n, settings)
    f["half_life_mpy"] = half_life_rate(d_curr, curr_year, install)
    f["joint_percentile_mpy"], _ = joint_percentile(f.raw_mpy, cls, joint_id, in_stats, settings)
    f["shrinking_fixed_mpy"] = shrinking_fixed_rate(d_prev, d_curr, cls, settings)
    gov, method = governing_rate(f, cls, settings)
    f["governing_mpy"], f["governing_method"] = gov, pd.Series(method, dtype=object)  # object keeps None
    needs_review, reasons = join_reasons(review_flags(f, cls, gov, match_review, settings), n)
    f["needs_review"], f["review_reasons"] = needs_review, pd.Series(reasons, dtype=object)
    f["length_in"] = rows.curr_length_in.to_numpy(float)
    f["length_rate_ipy"] = length_growth_rate(f.length_in, f.governing_mpy, d_curr, settings)
    f = f[FEATURE_COLUMNS]
    summary = _summary(f, dt_yr, prev_year, curr_year, settings, prev_run_id, curr_run_id)
    return GrowthResult(f, segment_stats(f, seg_table, p_by_seg), summary)


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def write_growth(result: GrowthResult, out_dir: Path) -> dict[str, Path]:
    s = result.summary
    folder = comparison_dir(out_dir, s["prev_run_id"], s["curr_run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / name for name in ("growth.csv", "growth_segments.csv", "growth_summary.json")}
    result.features.to_csv(paths["growth.csv"], index=False, lineterminator="\n")
    result.segments.to_csv(paths["growth_segments.csv"], index=False, lineterminator="\n")
    paths["growth_summary.json"].write_text(json.dumps({"label": DRAFT_LABEL, **s}, indent=2), encoding="utf-8")
    return paths


_GROWTH_TEXT = ("match_id", "match_type", "prev_feature_id", "curr_feature_id", "segment_id", "history_class",
                "confidence", "governing_method", "review_reasons", "joint_id", "surface", "ml_origin")
_GROWTH_FLAGS = ("needs_review", "implausible", "match_needs_review", "in_recoat", "in_stats")


def load_growth_table(path: Path) -> pd.DataFrame:
    """Read a ``growth.csv`` back with its types (as ``match.load_match_table``): ids and labels stay text
    (empty is None), flags are True/False/None (``match.parse_flag_column``), the rest float."""
    header = pd.read_csv(path, nrows=0).columns
    text = [c for c in _GROWTH_TEXT if c in header]
    flags = [c for c in _GROWTH_FLAGS if c in header]
    df = pd.read_csv(path, dtype={c: str for c in text + flags}, keep_default_na=False,
                     na_values={c: [""] for c in header if c not in text + flags})
    for c in text:
        df[c] = df[c].astype(object).where(df[c] != "", None)
    for c in flags:
        df[c] = parse_flag_column(df[c], path, c)
    return df


def _run_year(std: pd.DataFrame) -> int:
    return std.run_date.dropna().iloc[0].year


def _line_extent(curr_std: pd.DataFrame) -> tuple[float, float]:
    """First and last girth-weld log distance of the current run (ft)."""
    x = curr_std.log_dist_ft[curr_std.feature_type == FeatureType.girth_weld.value].dropna()
    if x.empty:
        raise ValueError("the current run has no girth welds to give the line extent")
    return float(x.min()), float(x.max())


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.growth",
                                 description="Corrosion growth rates between two ILI runs. " + DRAFT_LABEL)
    ap.add_argument("--prev", type=Path, required=True, help="standardized CSV of the previous run")
    ap.add_argument("--curr", type=Path, required=True, help="standardized CSV of the current run")
    ap.add_argument("--matches", type=Path, required=True, help="match_table.csv from ili.match")
    ap.add_argument("--pipe", type=Path, required=True, help="pipe_properties.csv (current-run log distance)")
    ap.add_argument("--segments", type=Path, help="segments CSV (start_ft, end_ft, segment_id); "
                    "required unless segment_definition is fixed_length")
    ap.add_argument("--repairs", type=Path, help="repairs_between_runs.csv; rows inside recoat spans are "
                    "left out of the percentile distributions")
    ap.add_argument("--truth", type=Path, help="ground_truth.csv (synthetic data only) to score against")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs", help="output directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(args.settings)
        prev, curr = load_standardized(args.prev), load_standardized(args.curr)
        matches = load_match_table(args.matches)
        absent = [c for c in MATCH_M11_COLUMNS if c not in matches.columns]
        if absent:
            raise ValueError(f"match table {args.matches} lacks column(s) {', '.join(absent)}; "
                             "it predates M11, re-run ili.match")
        pipe = pd.read_csv(args.pipe)
        segments = pd.read_csv(args.segments) if args.segments else None
        repairs = pd.read_csv(args.repairs) if args.repairs else None
        seg_table = segment_table(settings, segments, *_line_extent(curr))
        result = compute_growth(matches, delta_t_years(prev, curr), _run_year(prev), _run_year(curr), pipe,
                                seg_table, settings, prev_run_id=str(prev.run_id.iloc[0]),
                                curr_run_id=str(curr.run_id.iloc[0]), repairs=repairs)
        paths = write_growth(result, args.out)
        score = None
        if args.truth:
            truth = pd.read_csv(args.truth, dtype={"prev_feature_id": str, "curr_feature_id": str})
            score = score_growth(result.features, truth)
            write_metrics("growth", score, Path(args.out) / "validation_metrics.json")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    print(DRAFT_LABEL)
    print(f"{s['prev_run_id']} vs {s['curr_run_id']}: {s['n_features']} metal-loss features, "
          f"delta t {s['delta_t_years']:.2f} years")
    print("history: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_history_class"].items()))
    print("governing method: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_governing_method"].items()))
    pct = s["pct_negative_raw"]
    reasons = ", ".join(f"{k} {v}" for k, v in s["counts_by_review_reason"].items())
    print(f"needs review: {s['n_needs_review']} ({reasons}); negative raw growth: "
          + ("n/a" if pct is None else f"{pct:.1f}%"))
    if score is not None:
        g, r = score["governing"], score["raw"]
        print(f"conservatism {score['conservatism_rate']:.4f} vs ground truth "
              f"({score['n_non_conservative']} non-conservative, {score['n_no_governing']} without a rate, "
              f"passes: {score['passes']})")
        print(f"governing bias {g['bias_mpy']:+.2f} mpy, RMSE {g['rmse_mpy']:.2f}; raw bias {r['bias_mpy']:+.2f}, RMSE {r['rmse_mpy']:.2f}")
    print(f"wrote {paths['growth.csv'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
