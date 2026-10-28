"""Remaining life: when a growing feature reaches each failure/repair criterion (spec section 6.7).

Each feature is projected forward in time τ (decimal years) from its assessed state:

    depth(τ) = d0 + max(g, 0) τ        L(τ) = L0 + max(gL, 0) τ        (inches; g, gL in in/yr)

Growth below zero is treated as zero: metal loss does not heal, and M5 already flags such rows.
Pressures come from ``ili.strength`` with the configured ``strength.method`` (d/t is clipped to
[0, 1] there, so a depth projected past the wall still gives finite, non-negative pressures).

Each criterion has a margin that is > 0 while not met and <= 0 once met:

    tau_80     0.80 t - depth(τ)                      (in)
    tau_burst  P_F(τ) - MOP                            (psig)
    tau_safe   P_safe(τ) - MOP = design_factor P_F(τ) - MOP   (psig)

With g, gL >= 0 every margin is non-increasing in τ (deeper and longer never raises P_F; the
Original B31G z = 20 and Modified B31G z = 50 jumps are both downward in P_F), so the time a
criterion is met is the single crossing τ* = inf{τ : margin(τ) <= 0}. It is found on the grid
0, step, ..., horizon_yr and refined with ``brentq``; the result is never later than τ* (see
``_first_crossing``). Times: 0 when met today, inf when not met by the horizon, NaN when the
margin cannot be evaluated. RL = min over the criteria (spec 6.7).

``compute_remaining_life`` assembles the per-feature table from M5 ``growth.csv`` rows and the reference
tables: strength at τ = 0, the criterion times, RL and its upper bound, the repair-by date, the next
assessment date of the feature's segment, the dig flag, review reasons appended to M5's, and per-segment
reinspection intervals.
Every uncertain case is resolved on the conservative side and flagged (see the function docstrings).

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ili.align_welds import DRAFT_LABEL, comparison_dir
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT, Settings, load_settings
from ili.growth import REASON_SEP, REVIEW_REASONS, load_growth_table
from ili.ingest import load_standardized
from ili.match import DAYS_PER_YEAR, MILS_PER_IN, parse_flag_column
from ili.reference import _require_columns, assign_segment, check_segments, lookup
from ili.strength import (WALL_LOSS_LIMIT, assessed_depth_in, assessed_length_in, burst_pressure,
                          exceeds_wall_loss_limit, failure_stress, safe_pressure)

CRITERIA = ("tau_80", "tau_burst", "tau_safe")  # spec 6.7; also the tie-break order for RL
BEYOND_HORIZON = "beyond_horizon"  # rl_governing when no criterion is met within horizon_yr

BRENTQ_XTOL_YR = 1e-6  # absolute tolerance of the crossing time, years (numerical, ~30 s)
# scipy's smallest allowed relative tolerance, passed explicitly so the final bracket width
# (xtol + rtol |x|) used for the conservative step-back below is known exactly.
BRENTQ_RTOL = 4 * np.finfo(float).eps


def projected_depth_length(tau, d0, L0, g, gL):
    """(depth(τ), L(τ)) in inches: d0 + max(g, 0) τ and L0 + max(gL, 0) τ; NaN in, NaN out."""
    tau = np.asarray(tau, dtype=float)
    depth = np.asarray(d0, dtype=float) + np.maximum(np.asarray(g, dtype=float), 0.0) * tau
    length = np.asarray(L0, dtype=float) + np.maximum(np.asarray(gL, dtype=float), 0.0) * tau
    return depth, length


def usable_inputs(L0, D, smys, mop):
    """(L0, D, SMYS, MOP) as float arrays with values that cannot be real replaced by NaN.

    An OD, SMYS or MOP <= 0 is missing data, not a value: a zero MOP would make the pressure criteria
    never met (silently optimistic). A negative length makes z = L^2 / (D t) fall and then rise as the
    length grows, so the margins would not be monotone. Each becomes NaN, so the criteria that need it
    are NaN (``rl_not_evaluated``) while ``tau_80``, which needs only depth and wall, is kept. A length
    of exactly 0 is kept (z = 0).
    """
    def positive(x):
        a = np.asarray(x, dtype=float)
        return np.where(a > 0, a, np.nan)

    length = np.asarray(L0, dtype=float)
    return np.where(length >= 0, length, np.nan), positive(D), positive(smys), positive(mop)


def criterion_margins(tau, d0, L0, g, gL, D, t, smys, mop, settings: Settings) -> dict[str, np.ndarray]:
    """Margin of each criterion at time(s) τ (years); > 0 not met, <= 0 met.

    ``tau_80``: 0.80 t - depth(τ) in inches. ``tau_burst``: P_F(τ) - MOP and ``tau_safe``:
    P_safe(τ) - MOP in psig, with P_F = 2 S_F t / D from the configured ``strength.method`` and
    P_safe = P_F x ``strength.design_factor``. D, t, d0, L0 in inches; g, gL in in/yr; SMYS psi;
    MOP psig. Vectorised over τ; NaN inputs give NaN margins, and so do the unusable values of
    ``usable_inputs`` (OD, SMYS or MOP <= 0, negative L0) for the pressure margins.
    """
    L0, D, smys, mop = usable_inputs(L0, D, smys, mop)
    depth, length = projected_depth_length(tau, d0, L0, g, gL)
    t_arr = np.asarray(t, dtype=float)
    _, _, _, s_f = failure_stress(settings.strength.method, D, t, depth, length, smys)
    p_f = burst_pressure(s_f, t, D)
    p_safe = safe_pressure(p_f, settings)
    return {
        "tau_80": WALL_LOSS_LIMIT * t_arr - depth,
        "tau_burst": p_f - mop,
        "tau_safe": p_safe - mop,
    }


def strength_at(tau, d0, L0, g, gL, D, t, smys, mop, settings: Settings) -> dict[str, np.ndarray]:
    """Projected geometry and pressures at time(s) τ (years), elementwise (spec 6.6).

    Keys: ``depth_in`` and ``length_in`` (``projected_depth_length``), ``burst_psig`` (P_F from the configured
    ``strength.method``), ``safe_psig`` (P_F x design factor), ``erf`` = MOP / P_safe and ``fpr`` = P_F / MOP.
    Units as ``criterion_margins``. NaN in, NaN out; OD, SMYS or MOP <= 0 and a negative L0 count as missing
    (``usable_inputs``); ERF and FPR are NaN where MOP or P_safe is not a finite positive number.
    """
    L0, D, smys, mop = usable_inputs(L0, D, smys, mop)
    depth, length = projected_depth_length(tau, d0, L0, g, gL)
    _, _, _, s_f = failure_stress(settings.strength.method, D, t, depth, length, smys)
    p_f = np.asarray(burst_pressure(s_f, t, D), dtype=float)
    p_safe = np.asarray(safe_pressure(p_f, settings), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(mop) & (mop > 0) & np.isfinite(p_safe) & (p_safe > 0)
        erf = np.where(valid, mop / np.where(valid, p_safe, 1.0), np.nan)
        fpr = np.where(valid, p_f / np.where(valid, mop, 1.0), np.nan)
    return {"depth_in": depth, "length_in": length, "burst_psig": p_f, "safe_psig": p_safe, "erf": erf, "fpr": fpr}


def time_grid(settings: Settings) -> np.ndarray:
    """Search grid 0, step, 2 step, ..., horizon_yr in years; the horizon itself is always the last point."""
    horizon = float(settings.remaining_life.horizon_yr)
    step = float(settings.remaining_life.time_step_yr)
    n = max(math.ceil(horizon / step), 0)
    return np.minimum(np.arange(n + 1) * step, horizon)


def _first_crossing(margin: Callable[[float], float], grid: np.ndarray, m_grid: np.ndarray) -> float:
    """First time the (non-increasing) margin is <= 0, never later than the true crossing τ*.

    - m(0) <= 0: 0.0. No grid point with m <= 0: inf.
    - Otherwise [a, b] is the first grid interval whose END margin is <= 0, so m(a) > 0 >= m(b)
      and a < τ* <= b. ``brentq`` keeps a sign-change bracket, so on a continuous margin it finds
      the root and on a jump (Original B31G z = 20) it shrinks onto the jump location. On return
      the estimate x and the other bracket end are less than xtol + rtol |x| apart with τ* between
      them. If m(x) <= 0, x is at or after τ*, so it is stepped back by that width (not below a):
      the result is then before τ* by at most ~1e-6 yr. If m(x) > 0, x is already before τ*.
    - A margin of exactly 0 can hold on a whole stretch (a plateau: P_F is constant once d/t is
      clipped at 1, or after the Original B31G z = 20 jump with no depth growth). There ``brentq``
      may stop anywhere on the plateau (it returns b at once when m(b) == 0), far after τ*. So when
      the END margin m(b) is exactly 0, or ``brentq`` returns a point whose margin is exactly 0,
      ``_bisect_met`` is used instead: it is never later than τ*.
    - If ``brentq`` raises or returns something unusable (cannot happen with m(a) > 0 >= m(b),
      but guarded), fall back to the interval START a. a is not met yet, so it is up to one grid
      step EARLY - the conservative side. The END b is the first grid point at which the
      criterion is met and can be up to one step LATE (optimistic), so it is never used.
    """
    if m_grid[0] <= 0:
        return 0.0
    hits = np.flatnonzero(m_grid <= 0)
    if hits.size == 0:
        return math.inf
    i = int(hits[0])
    a, b = float(grid[i - 1]), float(grid[i])
    if m_grid[i] == 0:
        return _bisect_met(margin, a, b)
    try:
        x = float(brentq(margin, a, b, xtol=BRENTQ_XTOL_YR, rtol=BRENTQ_RTOL))
        m_x = float(margin(x))
    except (ValueError, RuntimeError):
        return a
    if not (math.isfinite(x) and math.isfinite(m_x) and a <= x <= b):
        return a
    if m_x == 0:
        return _bisect_met(margin, a, b)
    if m_x < 0:
        x = max(a, x - (BRENTQ_XTOL_YR + BRENTQ_RTOL * abs(x)))
    return x


def _bisect_met(margin: Callable[[float], float], a: float, b: float) -> float:
    """Bisection on the predicate margin <= 0 over [a, b] with m(a) > 0 >= m(b), down to ``BRENTQ_XTOL_YR``.

    Returns the lower end lo, which always has m(lo) > 0: with a non-increasing margin lo is before the
    first time the criterion is met, so the result is never late (and early by at most the tolerance).
    A non-finite midpoint margin stops the search at the current lo (still not late).
    """
    lo, hi = a, b
    while hi - lo > BRENTQ_XTOL_YR:
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        m_mid = float(margin(mid))
        if m_mid <= 0:
            hi = mid
        elif m_mid > 0:
            lo = mid
        else:  # NaN
            break
    return lo


def time_to_criteria(d0, L0, g, gL, D, t, smys, mop, settings: Settings) -> dict[str, float]:
    """Years until each of ``CRITERIA`` is met for one feature (scalar inputs, units as ``criterion_margins``).

    - 0.0 when met at τ = 0. The τ = 0 margin is evaluated without g and gL (growth only acts for
      τ > 0), so a criterion failing today is 0 even when the rate is NaN; it is NaN only when that
      criterion's own τ = 0 inputs are (e.g. tau_burst with a NaN MOP, while tau_80 needs only d0, t).
    - Otherwise NaN when the margin is not finite anywhere on the grid (a NaN rate, length, MOP or
      pipe value, or an unusable one per ``usable_inputs``: OD, SMYS or MOP <= 0, negative length),
      inf when not met by ``remaining_life.horizon_yr``, else the first crossing from
      ``_first_crossing``, which is never later than the true crossing.
    - All NaN for t <= 0 (invalid wall). OD <= 0 makes only the pressure criteria NaN.

    The grid is evaluated in one vectorised call; ``brentq`` runs only on each criterion's first
    crossing interval.
    """
    if float(t) <= 0:
        return {c: math.nan for c in CRITERIA}
    today = criterion_margins(0.0, d0, L0, 0.0, 0.0, D, t, smys, mop, settings)
    grid = time_grid(settings)
    m_grid = criterion_margins(grid, d0, L0, g, gL, D, t, smys, mop, settings)
    times: dict[str, float] = {}
    for name in CRITERIA:
        m0 = float(today[name])
        if m0 <= 0:  # False for NaN
            times[name] = 0.0
            continue
        m = m_grid[name]
        if math.isnan(m0) or not np.isfinite(m).all():
            times[name] = math.nan
            continue

        def margin(tau, _name=name):
            return float(criterion_margins(tau, d0, L0, g, gL, D, t, smys, mop, settings)[_name])

        times[name] = _first_crossing(margin, grid, m)
    return times


def remaining_life(times: dict[str, float], settings: Settings) -> tuple[float, str | None]:
    """(rl_yr, rl_governing) from ``time_to_criteria`` output (spec 6.7: RL = min of the times).

    - Any criterion exactly 0 (failing today): (0.0, the first such in ``CRITERIA`` order), even if
      another criterion is NaN - a missing input can never hide a failure today.
    - Else any NaN (or negative) criterion: (NaN, None) - an RL is never claimed when a criterion
      cannot be evaluated, even if another criterion is finite.
    - All inf: (``remaining_life.horizon_yr``, ``"beyond_horizon"``).
    - Else the minimum finite time and its criterion; ties go to the ``CRITERIA`` order.

    ``settings`` supplies ``horizon_yr`` for the all-inf case.
    """
    values = [float(times[c]) for c in CRITERIA]
    for name, value in zip(CRITERIA, values):
        if value == 0:
            return 0.0, name
    if any(math.isnan(v) or v < 0 for v in values):
        return math.nan, None
    governing = None
    for name, value in zip(CRITERIA, values):
        if math.isfinite(value) and (governing is None or value < times[governing]):
            governing = name
    if governing is None:
        return float(settings.remaining_life.horizon_yr), BEYOND_HORIZON
    return float(times[governing]), governing


def failure_mode(tau_80, tau_burst) -> np.ndarray:
    """``leak`` where tau_80 <= tau_burst (the wall is lost first), ``burst`` where tau_80 > tau_burst; None
    (object array) where either time is NaN or both are inf (nothing to compare)."""
    a, b = np.asarray(tau_80, dtype=float), np.asarray(tau_burst, dtype=float)
    out = np.full(np.broadcast(a, b).shape, None, dtype=object)
    ok = ~(np.isnan(a) | np.isnan(b) | (np.isinf(a) & np.isinf(b)))
    out[ok & (a <= b)] = "leak"
    out[ok & (a > b)] = "burst"
    return out


def condition_date(run_date: date, tau_yr: float) -> date | None:
    """run date + floor(τ x 365.25) days (never a day late); None for NaN or inf."""
    if not math.isfinite(tau_yr):
        return None
    return run_date + timedelta(days=math.floor(tau_yr * DAYS_PER_YEAR))


def rl_upper_bound(times: dict[str, float], rl_yr: float) -> float | None:
    """A known upper bound on RL in years: ``rl_yr`` when it is finite, else the smallest finite criterion
    time (RL = min of the times, so RL <= any one of them even when another is NaN); None when no
    criterion time is finite. It drives the dig flag and intervals of a NaN-RL row, never its repair-by
    date (a bound is not an RL)."""
    if math.isfinite(rl_yr):
        return float(rl_yr)
    finite = [float(times[c]) for c in CRITERIA if math.isfinite(times[c])]
    return min(finite) if finite else None


def factored_bound(rl_factored_yr, rl_upper_bound_yr, rl_factor) -> np.ndarray:
    """The factored RL used for the dig flag and intervals: RL x rl_factor where RL is finite, else the
    upper bound x rl_factor (NaN when there is neither)."""
    rl_f = np.asarray(rl_factored_yr, dtype=float)
    return np.where(np.isfinite(rl_f), rl_f, np.asarray(rl_upper_bound_yr, dtype=float) * float(rl_factor))


# --- assembled remaining-life table -------------------------------------------------------------------

MONTHS_PER_YEAR = 12
# Why a row needs review, appended in this order after M5's ``review_reasons``: no RL could be evaluated
# (NaN, so no dig decision was made); assessed d/t >= 0.80 (spec 6.6); outside every segment, so the
# next assessment is the HCA default (the conservative choice) rather than the segment's own date.
RL_REVIEW_REASONS = ("rl_not_evaluated", "exceeds_80pct", "no_segment")
NEXT_ASSESSMENT_SOURCES = ("planned", "capped", "default_hca", "default_non_hca", "default_no_segment")
RL_GOVERNING = (*CRITERIA, BEYOND_HORIZON)

GROWTH_INPUT_COLUMNS = (
    "match_id", "curr_feature_id", "curr_log_dist_ft", "wt_in", "d_curr_pct", "length_in", "governing_mpy",
    "length_rate_ipy", "needs_review", "review_reasons",
)
RL_SEGMENT_COLUMNS = ("start_ft", "end_ft", "segment_id", "could_affect_hca")
NEXT_INSPECTION_COLUMNS = ("segment_id", "planned_next_assessment_date")
PIPE_WHAT, MOP_WHAT = "pipe properties table", "MOP profile table"

RL_FEATURE_COLUMNS = [
    "match_id", "curr_feature_id", "curr_log_dist_ft", "segment_id", "could_affect_hca",
    "od_in", "wt_in", "smys_psi", "mop_psig", "d_reported_pct", "d_assessed_in", "assessed_dt", "exceeds_80pct",
    "length_in", "length_assessed_in",
    "method", "z", "folias_m", "flow_stress_psi", "failure_stress_psi", "burst_psig", "safe_psig", "erf", "fpr",
    "g_inyr", "gl_inyr", "tau_80_yr", "tau_burst_yr", "tau_safe_yr", "tau_80_date", "tau_burst_date",
    "tau_safe_date", "rl_yr", "rl_governing", "failure_mode",
    "rl_upper_bound_yr", "rl_factor", "rl_factored_yr", "repair_by_date",
    "next_assessment_date", "next_assessment_source", "years_to_next_assessment",
    "depth_next_in", "depth_next_pct", "burst_next_psig", "safe_next_psig", "erf_next", "fpr_next",
    "dig_required_by_remaining_life",
    "needs_review", "review_reasons",
]
INTERVAL_COLUMNS = [
    "segment_id", "could_affect_hca", "n_features", "n_rl_not_evaluated", "min_rl_factored_yr",
    "regulatory_max_yr", "recommended_interval_yr", "governing_match_id", "needs_review",
]


@dataclass
class RLResult:
    """One row per growth feature, one interval row per segment, and a JSON-ready summary."""

    features: pd.DataFrame
    intervals: pd.DataFrame
    summary: dict


def add_months(day: date, months) -> date:
    """``day`` plus a whole number of calendar months; a day past the target month's end is moved back to
    that month's last day (earlier, the conservative side). ValueError for a fractional number of months."""
    m = float(months)
    if not m.is_integer():
        raise ValueError(f"{months} is not a whole number of months")
    k = day.month - 1 + int(m)
    year, month = day.year + k // MONTHS_PER_YEAR, k % MONTHS_PER_YEAR + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _as_date(value, what: str) -> date | None:
    """A ``datetime.date`` from a date, datetime/Timestamp or ISO string; None for a blank. ValueError
    naming ``what`` for anything else."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):  # NaT is a datetime: test first
        return None
    if isinstance(value, datetime):  # includes pd.Timestamp
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                pass
    raise ValueError(f"{what}: {value!r} is not an ISO date")


def _as_bool(value, what: str) -> bool:
    """True/False from a bool or a ``True``/``False`` string (any case); ValueError naming ``what`` otherwise."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{what}: could_affect_hca must be True or False, got {value!r}")


def _years_after(day: date, years, what: str) -> date:
    """``day`` plus ``years`` calendar years (as whole months)."""
    months = float(years) * MONTHS_PER_YEAR
    if not months.is_integer():
        raise ValueError(f"{what} of {years} years is not a whole number of months")
    return add_months(day, months)


def _planned_dates(next_inspection: pd.DataFrame) -> dict[str, date]:
    """segment_id -> planned date from ``next_inspection.csv``; a blank date is left out (the default is used).
    ValueError for a repeated segment id or an unparseable date, naming the segment."""
    _require_columns(next_inspection, NEXT_INSPECTION_COLUMNS, "next inspection table")
    ids = next_inspection.segment_id.astype(str)
    dup = ids[ids.duplicated()]
    if not dup.empty:
        raise ValueError(f"next inspection table repeats segment_id {dup.iloc[0]!r}")
    planned = {}
    for sid, value in zip(ids, next_inspection.planned_next_assessment_date):
        when = _as_date(value, f"next inspection date for segment {sid!r}")
        if when is not None:
            planned[sid] = when
    return planned


def next_assessment(could_affect_hca: bool | None, planned: date | None, run_date: date, settings: Settings,
                    segment_id: str | None = None) -> tuple[date, str]:
    """(date, source) of a segment's next assessment (spec 6.7, ``remaining_life`` settings).

    - ``planned``: the segment's date in ``next_inspection.csv``.
    - ``capped``: an HCA planned date later than run date + ``hca_max_interval_months``, cut to that limit.
    - ``default_hca`` / ``default_non_hca``: no planned date; run date + ``default_next_assessment_yr``
      calendar years, the HCA default also capped at the months limit.
    - ``default_no_segment`` (``could_affect_hca`` None, outside every segment): the capped HCA default.

    ValueError for a planned date on or before the run date (a data error), naming the segment.
    """
    rl = settings.remaining_life
    hca_cap = add_months(run_date, rl.hca_max_interval_months)
    hca_default = min(_years_after(run_date, rl.default_next_assessment_yr.hca, "the HCA default"), hca_cap)
    if could_affect_hca is None:
        return hca_default, "default_no_segment"
    if planned is not None:
        if planned <= run_date:
            raise ValueError(f"segment {segment_id!r}: planned next assessment {planned.isoformat()} is not "
                             f"after the current run date {run_date.isoformat()}")
        if could_affect_hca and planned > hca_cap:
            return hca_cap, "capped"
        return planned, "planned"
    if could_affect_hca:
        return hca_default, "default_hca"
    return _years_after(run_date, rl.default_next_assessment_yr.non_hca, "the non-HCA default"), "default_non_hca"


def repair_by(run_date: date, rl_factored_yr: float) -> date | None:
    """run date + RL x rl_factor x 365.25 days, rounded DOWN to whole days (never a day late); None for NaN."""
    if not math.isfinite(rl_factored_yr):
        return None
    return run_date + timedelta(days=math.floor(rl_factored_yr * DAYS_PER_YEAR))


def append_reasons(m5_reasons, m5_needs_review, m6_flags: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """M5's ``;``-joined reasons with the M6 reasons that apply appended in ``m6_flags`` order, a name
    already present not repeated (None when there are none); ``needs_review`` = M5's flag or any reason.
    An unknown M5 flag (None/NaN) counts as True: review is never dropped for missing information."""
    n = len(m5_needs_review)
    reasons = np.full(n, None, dtype=object)
    needs = np.zeros(n, dtype=bool)
    for k in range(n):
        old = m5_reasons[k]
        hit = [r for r in old.split(REASON_SEP) if r] if isinstance(old, str) else []
        hit += [r for r, flag in m6_flags.items() if flag[k] and r not in hit]
        reasons[k] = REASON_SEP.join(hit) if hit else None
        flag = m5_needs_review[k]
        needs[k] = bool(hit) or flag is None or pd.isna(flag) or bool(flag)
    return needs, reasons


def _segment_hca(segments: pd.DataFrame) -> dict[str, bool]:
    return {str(sid): _as_bool(v, f"segment {str(sid)!r}")
            for sid, v in zip(segments.segment_id, segments.could_affect_hca)}


def reinspection_intervals(features: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per segment, in table order (spec 6.7).

    - ``min_rl_factored_yr``: the minimum over its features of RL x rl_factor, using for a NaN-RL feature
      its ``rl_upper_bound_yr`` x rl_factor (``factored_bound``; the true value can only be smaller).
      A feature with neither is left out of the min. Every NaN-RL feature is counted in
      ``n_rl_not_evaluated``.
    - ``regulatory_max_yr``: hca_max_interval_months / 12 (HCA) or default_next_assessment_yr.non_hca.
    - ``recommended_interval_yr`` = min of the two; the regulatory max for a segment without features; NaN
      for a segment with features none of which has an RL or a bound (no interval can be supported).
    - ``governing_match_id``: the feature attaining the min (first in table order), None without one.
    - ``needs_review``: any feature's RL could not be evaluated (``n_rl_not_evaluated`` > 0), even when
      a bound gave the min.
    """
    rl = settings.remaining_life
    hca = _segment_hca(segments)
    out = []
    for sid in segments.segment_id.astype(str):
        in_seg = features[features.segment_id == sid]
        not_evaluated = ~np.isfinite(in_seg.rl_factored_yr.to_numpy(float))
        vals = factored_bound(in_seg.rl_factored_yr, in_seg.rl_upper_bound_yr, rl.rl_factor)
        finite = np.isfinite(vals)
        reg_max = (rl.hca_max_interval_months / MONTHS_PER_YEAR if hca[sid]
                   else float(rl.default_next_assessment_yr.non_hca))
        if finite.any():
            k = int(np.flatnonzero(vals == vals[finite].min())[0])
            min_rl, gov = float(vals[k]), str(in_seg.match_id.iloc[k])
            recommended = float(min(min_rl, reg_max))
        else:
            min_rl, gov = math.nan, None
            recommended = float(reg_max) if len(in_seg) == 0 else math.nan
        out.append({
            "segment_id": sid, "could_affect_hca": hca[sid], "n_features": len(in_seg),
            "n_rl_not_evaluated": int(not_evaluated.sum()), "min_rl_factored_yr": min_rl,
            "regulatory_max_yr": float(reg_max),
            "recommended_interval_yr": recommended, "governing_match_id": gov,
            "needs_review": bool(not_evaluated.any()),
        })
    table = pd.DataFrame(out, columns=INTERVAL_COLUMNS)
    # object columns keep None (pandas would infer a string dtype with NaN for a missing id)
    table["segment_id"] = pd.Series([r["segment_id"] for r in out], dtype=object)
    table["governing_match_id"] = pd.Series([r["governing_match_id"] for r in out], dtype=object)
    table["needs_review"] = table.needs_review.astype(bool)
    return table


def _finite_min(values: np.ndarray) -> float | None:
    """The minimum finite value as a float, None when there is none (JSON-ready)."""
    v = values[np.isfinite(values)]
    return float(v.min()) if len(v) else None


def _parameters(settings: Settings) -> dict:
    """Every setting the remaining-strength and remaining-life results depend on (traceability)."""
    st, tool, rl = settings.strength, settings.tool, settings.remaining_life
    return {
        "method": st.method, "design_factor": float(st.design_factor),
        "depth_basis": st.depth_basis, "length_basis": st.length_basis,
        "depth_tolerance_pct_wt": float(tool.depth_tolerance_pct_wt),
        "depth_certainty": float(tool.depth_certainty),
        "length_tolerance_in": float(tool.length_tolerance_in),
        "time_step_yr": float(rl.time_step_yr), "horizon_yr": float(rl.horizon_yr),
        "rl_factor": float(rl.rl_factor), "hca_max_interval_months": float(rl.hca_max_interval_months),
        "default_next_assessment_yr": {"hca": float(rl.default_next_assessment_yr.hca),
                                       "non_hca": float(rl.default_next_assessment_yr.non_hca)},
    }


def _summary(f: pd.DataFrame, intervals: pd.DataFrame, run_date: date, settings: Settings, prev_run_id,
             curr_run_id) -> dict:
    reasons = [r.split(REASON_SEP) if isinstance(r, str) else [] for r in f.review_reasons]
    return {
        "prev_run_id": prev_run_id, "curr_run_id": curr_run_id, "curr_run_date": run_date.isoformat(),
        **_parameters(settings),
        "n_features": len(f),
        "counts_by_rl_governing": {g: int((f.rl_governing == g).sum()) for g in RL_GOVERNING},
        "n_rl_not_evaluated": int(f.rl_yr.isna().sum()),
        "n_exceeds_80pct": int(f.exceeds_80pct.sum()),
        "n_dig_required": int(f.dig_required_by_remaining_life.sum()),
        "n_needs_review": int(f.needs_review.sum()),
        "counts_by_review_reason": {r: sum(r in x for x in reasons) for r in (*REVIEW_REASONS, *RL_REVIEW_REASONS)},
        "min_rl_yr": _finite_min(f.rl_yr.to_numpy(float)),
        "min_rl_factored_yr": _finite_min(f.rl_factored_yr.to_numpy(float)),
        "counts_by_next_assessment_source": {s: int((f.next_assessment_source == s).sum())
                                             for s in NEXT_ASSESSMENT_SOURCES},
        "n_segments_needs_review": int(intervals.needs_review.sum()),
    }


def compute_remaining_life(growth: pd.DataFrame, curr_run_date, pipe: pd.DataFrame, mop: pd.DataFrame,
                           segments: pd.DataFrame, next_inspection: pd.DataFrame, settings: Settings, *,
                           prev_run_id: str | None = None, curr_run_id: str | None = None) -> RLResult:
    """Remaining life of every M5 growth row (spec 6.6-6.7), in the same order, with per-segment intervals.

    - Reference values at ``curr_log_dist_ft`` (``alignment.anchor_tol_ft``): od_in and smys_psi from
      ``pipe``, mop_psig from ``mop``. A position outside a table is a ValueError; a NaN value there is
      kept, so that row's RL is NaN and flagged ``rl_not_evaluated`` (never a guessed value). A value
      <= 0 (or a negative length) is treated as NaN (``usable_inputs``) in every computed column.
    - Wall = growth ``wt_in``; assessed depth and length by ``strength.depth_basis`` / ``length_basis``.
      Strength columns are at τ = 0; ``exceeds_80pct`` is the unclipped assessed d/t >= 0.80.
    - g = governing_mpy / 1000 in/yr, gL = length_rate_ipy; ``time_to_criteria`` and ``remaining_life``.
      rl_factored = RL x rl_factor; ``repair_by_date`` from ``repair_by``.
    - Segment (``segments``: start_ft, end_ft, segment_id, could_affect_hca) by ``assign_segment``; the
      next assessment by ``next_assessment``; years to it = days / 365.25.
    - ``rl_upper_bound_yr`` (``rl_upper_bound``): RL when finite, else the smallest finite criterion
      time, NaN when there is none.
    - ``dig_required_by_remaining_life`` = rl_factored < years to next assessment; for a NaN RL,
      rl_upper_bound x rl_factor < years to next assessment (the true RL can only be smaller, so a dig
      the bound proves is never dropped); False for a NaN RL without a bound. A NaN-RL row is flagged
      ``rl_not_evaluated`` either way and its ``repair_by_date`` is None.
    - ``review_reasons``: M5's, then ``RL_REVIEW_REASONS`` that apply (``append_reasons``).
    """
    _require_columns(growth, GROWTH_INPUT_COLUMNS, "growth table")
    _require_columns(segments, RL_SEGMENT_COLUMNS, "segments table")
    check_segments(segments)
    run_date = _as_date(curr_run_date, "current run date")
    if run_date is None:
        raise ValueError("the current run date is missing")
    rl_cfg, method = settings.remaining_life, settings.strength.method
    g_in = growth.reset_index(drop=True)
    n = len(g_in)
    pos = g_in.curr_log_dist_ft.to_numpy(float)
    tol = settings.alignment.anchor_tol_ft
    if n:
        od = lookup(pipe, pos, "od_in", tol, what=PIPE_WHAT, allow_nan=True)
        smys = lookup(pipe, pos, "smys_psi", tol, what=PIPE_WHAT, allow_nan=True)
        mop_psig = lookup(mop, pos, "mop_psig", tol, what=MOP_WHAT, allow_nan=True)
    else:
        od = smys = mop_psig = np.zeros(0)
    wt = g_in.wt_in.to_numpy(float)
    d_pct = g_in.d_curr_pct.to_numpy(float)
    length = g_in.length_in.to_numpy(float)
    d_as = np.asarray(assessed_depth_in(d_pct, wt, settings), dtype=float)
    l_as = np.asarray(assessed_length_in(length, settings), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dt_ratio = d_as / wt
        exceeds = np.asarray(exceeds_wall_loss_limit(d_as, wt), dtype=bool)
    # strength from usable values only (an OD/SMYS <= 0 or negative length gives NaN, not a made-up stress);
    # the od_in, smys_psi, mop_psig and length columns keep the values as read
    l_ok, od_ok, smys_ok, _ = usable_inputs(l_as, od, smys, mop_psig)
    z, folias, flow, s_f = (np.asarray(a, dtype=float)
                            for a in failure_stress(method, od_ok, wt, d_as, l_ok, smys_ok))
    p_f = np.asarray(burst_pressure(s_f, wt, od_ok), dtype=float)
    p_safe = np.asarray(safe_pressure(p_f, settings), dtype=float)
    g_inyr = g_in.governing_mpy.to_numpy(float) / MILS_PER_IN
    gl_inyr = g_in.length_rate_ipy.to_numpy(float)
    now = strength_at(0.0, d_as, l_as, 0.0, 0.0, od, wt, smys, mop_psig, settings)

    taus = {c: np.full(n, math.nan) for c in CRITERIA}
    rl_yr = np.full(n, math.nan)
    rl_bound = np.full(n, math.nan)
    rl_gov = np.full(n, None, dtype=object)
    for k in range(n):
        times = time_to_criteria(d_as[k], l_as[k], g_inyr[k], gl_inyr[k], od[k], wt[k], smys[k], mop_psig[k],
                                 settings)
        for c in CRITERIA:
            taus[c][k] = times[c]
        rl_yr[k], rl_gov[k] = remaining_life(times, settings)
        bound = rl_upper_bound(times, rl_yr[k])
        rl_bound[k] = math.nan if bound is None else bound
    rl_factored = rl_yr * rl_cfg.rl_factor

    seg_id = assign_segment(segments, pos)
    hca = _segment_hca(segments)
    planned = _planned_dates(next_inspection)
    unknown = sorted(set(next_inspection.segment_id.astype(str)) - set(hca))  # blank dates included
    if unknown:
        raise ValueError(f"next inspection table names segment(s) {unknown} that are not in the segments table")
    per_seg = {sid: next_assessment(hca[sid], planned.get(sid), run_date, settings, sid) for sid in hca}
    outside = next_assessment(None, None, run_date, settings)
    nxt = [outside if s is None else per_seg[s] for s in seg_id]
    years_next = np.array([(d - run_date).days / DAYS_PER_YEAR for d, _ in nxt], dtype=float)
    # NaN RL: the factored upper bound decides (flagged rl_not_evaluated either way); no bound: no dig claim
    for_dig = factored_bound(rl_factored, rl_bound, rl_cfg.rl_factor)
    dig = np.isfinite(for_dig) & (for_dig < years_next)
    nxt_str = strength_at(years_next, d_as, l_as, g_inyr, gl_inyr, od, wt, smys, mop_psig, settings)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_next_pct = nxt_str["depth_in"] / wt * 100.0

    m6_flags = {"rl_not_evaluated": np.isnan(rl_yr), "exceeds_80pct": exceeds,
                "no_segment": np.array([s is None for s in seg_id], dtype=bool)}
    needs_review, reasons = append_reasons(list(g_in.review_reasons), list(g_in.needs_review), m6_flags)

    def obj(values) -> pd.Series:
        return pd.Series(list(values), dtype=object)

    f = pd.DataFrame({
        "match_id": obj(g_in.match_id), "curr_feature_id": obj(g_in.curr_feature_id),
        "curr_log_dist_ft": pos, "segment_id": obj(seg_id),
        "could_affect_hca": obj(None if s is None else hca[s] for s in seg_id),
        "od_in": od, "wt_in": wt, "smys_psi": smys, "mop_psig": mop_psig,
        "d_reported_pct": d_pct, "d_assessed_in": d_as, "assessed_dt": dt_ratio, "exceeds_80pct": exceeds,
        "length_in": length, "length_assessed_in": l_as,
        "method": obj([method] * n), "z": z, "folias_m": folias, "flow_stress_psi": flow,
        "failure_stress_psi": s_f, "burst_psig": p_f, "safe_psig": p_safe,
        "erf": now["erf"], "fpr": now["fpr"],
        "g_inyr": g_inyr, "gl_inyr": gl_inyr,
        "tau_80_yr": taus["tau_80"], "tau_burst_yr": taus["tau_burst"], "tau_safe_yr": taus["tau_safe"],
        "tau_80_date": obj(condition_date(run_date, x) for x in taus["tau_80"]),
        "tau_burst_date": obj(condition_date(run_date, x) for x in taus["tau_burst"]),
        "tau_safe_date": obj(condition_date(run_date, x) for x in taus["tau_safe"]),
        "rl_yr": rl_yr, "rl_governing": obj(rl_gov),
        "failure_mode": obj(failure_mode(taus["tau_80"], taus["tau_burst"])),
        "rl_upper_bound_yr": rl_bound,
        "rl_factor": np.full(n, float(rl_cfg.rl_factor)), "rl_factored_yr": rl_factored,
        "repair_by_date": obj(repair_by(run_date, x) for x in rl_factored),
        "next_assessment_date": obj(d for d, _ in nxt), "next_assessment_source": obj(s for _, s in nxt),
        "years_to_next_assessment": years_next,
        "depth_next_in": nxt_str["depth_in"], "depth_next_pct": depth_next_pct,
        "burst_next_psig": nxt_str["burst_psig"], "safe_next_psig": nxt_str["safe_psig"],
        "erf_next": nxt_str["erf"], "fpr_next": nxt_str["fpr"], "dig_required_by_remaining_life": dig,
        "needs_review": needs_review, "review_reasons": obj(reasons),
    })[RL_FEATURE_COLUMNS]
    intervals = reinspection_intervals(f, segments, settings)
    return RLResult(f, intervals, _summary(f, intervals, run_date, settings, prev_run_id, curr_run_id))


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------

OUTPUT_FILES = ("remaining_life.csv", "reinspection_intervals.csv", "remaining_life_summary.json")


def write_remaining_life(result: RLResult, out_dir: Path) -> dict[str, Path]:
    """Write the three outputs into ``comparison_dir`` (created); the summary JSON carries the DRAFT label."""
    s = result.summary
    folder = comparison_dir(out_dir, s["prev_run_id"], s["curr_run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / name for name in OUTPUT_FILES}
    result.features.to_csv(paths["remaining_life.csv"], index=False, lineterminator="\n")
    result.intervals.to_csv(paths["reinspection_intervals.csv"], index=False, lineterminator="\n")
    paths["remaining_life_summary.json"].write_text(
        json.dumps({"label": DRAFT_LABEL, **s}, indent=2, allow_nan=False), encoding="utf-8")
    return paths


_RL_TEXT = ("match_id", "curr_feature_id", "segment_id", "method", "rl_governing", "next_assessment_source",
            "review_reasons", "failure_mode")
_RL_FLAGS = ("could_affect_hca", "exceeds_80pct", "dig_required_by_remaining_life", "needs_review")
_RL_DATES = ("repair_by_date", "next_assessment_date", "tau_80_date", "tau_burst_date", "tau_safe_date")


def load_remaining_life_table(path: Path) -> pd.DataFrame:
    """Read a ``remaining_life.csv`` back with its types (as ``growth.load_growth_table``): ids and labels
    stay text (empty is None), flags are True/False/None (``match.parse_flag_column``: any case), dates are
    ``datetime.date`` or None, the rest float. ValueError for a flag that is not true, false or blank, or a
    date that is not ISO."""
    header = pd.read_csv(path, nrows=0).columns
    text = [c for c in (*_RL_TEXT, *_RL_DATES) if c in header]
    flags = [c for c in _RL_FLAGS if c in header]
    df = pd.read_csv(path, dtype={c: str for c in text + flags}, keep_default_na=False,
                     na_values={c: [""] for c in header if c not in text + flags})
    for c in text:
        df[c] = df[c].astype(object).where(df[c] != "", None)
    for c in flags:
        df[c] = parse_flag_column(df[c], path, c)
    for c in _RL_DATES:
        if c in header:
            df[c] = pd.Series([_as_date(v, f"{path}: {c}") for v in df[c]], dtype=object, index=df.index)
    return df


def _run_ids(growth_path: Path, curr_std: pd.DataFrame) -> tuple[str | None, str]:
    """(prev_run_id, curr_run_id): from ``growth_summary.json`` beside the growth table (M5 writes it there);
    without it (or without its ``curr_run_id``), the current standardized file's run id and no previous id,
    with a warning on stderr. ValueError when the file is not a JSON object."""
    summary = Path(growth_path).parent / "growth_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{summary}: expected a JSON object, got {type(data).__name__}")
        prev, curr = data.get("prev_run_id"), data.get("curr_run_id")
        if curr is not None:
            return (None if prev is None else str(prev)), str(curr)
        print(f"warning: {summary} has no curr_run_id; previous run id unknown", file=sys.stderr)
    else:
        print(f"warning: growth_summary.json not found beside {growth_path}; previous run id unknown",
              file=sys.stderr)
    return None, str(curr_std.run_id.iloc[0])


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.remaining_life",
                                 description="Remaining strength and remaining life. " + DRAFT_LABEL)
    ap.add_argument("--curr", type=Path, required=True, help="standardized CSV of the current run (run date)")
    ap.add_argument("--growth", type=Path, required=True, help="growth.csv from ili.growth")
    ap.add_argument("--pipe", type=Path, required=True, help="pipe_properties.csv (od_in, smys_psi)")
    ap.add_argument("--mop", type=Path, required=True, help="mop_profile.csv")
    ap.add_argument("--segments", type=Path, required=True,
                    help="hca_segments.csv (start_ft, end_ft, segment_id, could_affect_hca)")
    ap.add_argument("--next-inspection", type=Path, required=True, help="next_inspection.csv")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs", help="output directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # every input is read before anything is written, so a bad input leaves no partial outputs
        settings = load_settings(args.settings)
        curr = load_standardized(args.curr)
        growth = load_growth_table(args.growth)
        pipe, mop = pd.read_csv(args.pipe), pd.read_csv(args.mop)
        segments = pd.read_csv(args.segments, dtype={"segment_id": str})
        next_inspection = pd.read_csv(args.next_inspection, dtype={"segment_id": str})
        prev_id, curr_id = _run_ids(args.growth, curr)
        run_dates = curr.run_date.dropna()
        if run_dates.empty:
            raise ValueError(f"{args.curr}: the current run has no run date")
        result = compute_remaining_life(growth, run_dates.iloc[0], pipe, mop, segments, next_inspection, settings,
                                        prev_run_id=prev_id, curr_run_id=curr_id)
        paths = write_remaining_life(result, args.out)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    print(DRAFT_LABEL)
    print(f"{s['prev_run_id']} vs {s['curr_run_id']}: {s['n_features']} metal-loss features, "
          f"{s['method']} strength, RL factor {s['rl_factor']}, horizon {s['horizon_yr']:g} years")
    print("governing criterion: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_rl_governing"].items())
          + f", not evaluated {s['n_rl_not_evaluated']}")
    min_rl = s["min_rl_yr"]
    print(f"exceeds 80%: {s['n_exceeds_80pct']}; dig required: {s['n_dig_required']}; "
          f"min RL: " + ("n/a" if min_rl is None else f"{min_rl:.2f} years"))
    reasons = ", ".join(f"{k} {v}" for k, v in s["counts_by_review_reason"].items())
    print(f"needs review: {s['n_needs_review']} ({reasons})")
    print("reinspection intervals:")
    print(result.intervals.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"wrote {paths['remaining_life.csv'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
