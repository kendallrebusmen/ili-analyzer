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

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from ili.config import Settings
from ili.strength import WALL_LOSS_LIMIT, burst_pressure, failure_stress, safe_pressure

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


def criterion_margins(tau, d0, L0, g, gL, D, t, smys, mop, settings: Settings) -> dict[str, np.ndarray]:
    """Margin of each criterion at time(s) τ (years); > 0 not met, <= 0 met.

    ``tau_80``: 0.80 t - depth(τ) in inches. ``tau_burst``: P_F(τ) - MOP and ``tau_safe``:
    P_safe(τ) - MOP in psig, with P_F = 2 S_F t / D from the configured ``strength.method`` and
    P_safe = P_F x ``strength.design_factor``. D, t, d0, L0 in inches; g, gL in in/yr; SMYS psi;
    MOP psig. Vectorised over τ; NaN inputs give NaN margins.
    """
    depth, length = projected_depth_length(tau, d0, L0, g, gL)
    t_arr = np.asarray(t, dtype=float)
    _, _, _, s_f = failure_stress(settings.strength.method, D, t, depth, length, smys)
    p_f = burst_pressure(s_f, t, D)
    p_safe = safe_pressure(p_f, settings)
    mop = np.asarray(mop, dtype=float)
    return {
        "tau_80": WALL_LOSS_LIMIT * t_arr - depth,
        "tau_burst": p_f - mop,
        "tau_safe": p_safe - mop,
    }


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
    try:
        x = float(brentq(margin, a, b, xtol=BRENTQ_XTOL_YR, rtol=BRENTQ_RTOL))
        m_x = float(margin(x))
    except (ValueError, RuntimeError):
        return a
    if not (math.isfinite(x) and math.isfinite(m_x) and a <= x <= b):
        return a
    if m_x <= 0:
        x = max(a, x - (BRENTQ_XTOL_YR + BRENTQ_RTOL * abs(x)))
    return x


def time_to_criteria(d0, L0, g, gL, D, t, smys, mop, settings: Settings) -> dict[str, float]:
    """Years until each of ``CRITERIA`` is met for one feature (scalar inputs, units as ``criterion_margins``).

    0.0 when met at τ = 0; inf when not met by ``remaining_life.horizon_yr``; NaN when that
    criterion's margin is not finite anywhere on the grid (a NaN input, e.g. a NaN rate, length,
    MOP or pipe value) or D, t <= 0. Otherwise the first crossing from ``_first_crossing``, which
    is never later than the true crossing. The grid is evaluated in one vectorised call;
    ``brentq`` runs only on each criterion's first crossing interval.
    """
    if not (float(D) > 0 and float(t) > 0):
        return {c: math.nan for c in CRITERIA}
    grid = time_grid(settings)
    m_grid = criterion_margins(grid, d0, L0, g, gL, D, t, smys, mop, settings)
    times: dict[str, float] = {}
    for name in CRITERIA:
        m = m_grid[name]
        if not np.isfinite(m).all():
            times[name] = math.nan
            continue

        def margin(tau, _name=name):
            return float(criterion_margins(tau, d0, L0, g, gL, D, t, smys, mop, settings)[_name])

        times[name] = _first_crossing(margin, grid, m)
    return times


def remaining_life(times: dict[str, float], settings: Settings) -> tuple[float, str | None]:
    """(rl_yr, rl_governing) from ``time_to_criteria`` output (spec 6.7: RL = min of the times).

    - Any NaN (or negative) criterion: (NaN, None) - an RL is never claimed when a criterion
      cannot be evaluated, even if another criterion is finite.
    - All inf: (``remaining_life.horizon_yr``, ``"beyond_horizon"``).
    - Else the minimum finite time and its criterion; ties go to the ``CRITERIA`` order.

    ``settings`` supplies ``horizon_yr`` for the all-inf case.
    """
    values = [float(times[c]) for c in CRITERIA]
    if any(math.isnan(v) or v < 0 for v in values):
        return math.nan, None
    governing = None
    for name, value in zip(CRITERIA, values):
        if math.isfinite(value) and (governing is None or value < times[governing]):
            governing = name
    if governing is None:
        return float(settings.remaining_life.horizon_yr), BEYOND_HORIZON
    return float(times[governing]), governing
