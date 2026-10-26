"""Per-feature corrosion growth rates (spec section 6.5).

Vectorised methods, each accepting scalars or arrays (NaN in, NaN out). Depths are in inches,
rates in mpy (mils per year), and elapsed time in decimal years. Every tuning value comes from
``settings.tool`` and ``settings.growth``.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import numpy as np

from ili.config import Settings, ToolCfg
from ili.match import MILS_PER_IN
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
