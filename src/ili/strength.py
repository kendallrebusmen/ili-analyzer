"""Remaining strength of corroded pipe: Original and Modified B31G (spec section 6.6).

Pure, numpy-vectorised functions accepting scalars or arrays (NaN in, NaN out). Units: D, t, d and L
in inches, stresses in psi, pressures in psig. Every B31G number is a named module constant citing
the spec; every engineering setting (basis, tolerances, design factor) comes from ``settings``.

RSTRENG effective area is out of scope for v1; ``rstreng_effective_area`` is the interface stub.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ili.config import Settings
from ili.growth import PCT_PER_FRACTION

# Original B31G (spec 6.6)
ORIGINAL_FLOW_FACTOR = 1.1  # spec 6.6: S_flow = 1.1 SMYS
ORIGINAL_Z_LIMIT = 20  # spec 6.6: Folias form for z <= 20, S_flow (1 - d/t) for z > 20
ORIGINAL_FOLIAS_Z_COEF = 0.8  # spec 6.6: M = sqrt(1 + 0.8 z)
ORIGINAL_DEPTH_FACTOR = 2.0 / 3.0  # spec 6.6: S_F = S_flow [1 - (2/3)(d/t)] / [1 - (2/3)(d/t)/M]

# Modified B31G, 0.85 dL (spec 6.6)
MODIFIED_FLOW_ADDER_PSI = 10_000  # spec 6.6: S_flow = SMYS + 10,000 psi
MODIFIED_Z_LIMIT = 50  # spec 6.6: quadratic Folias for z <= 50, linear for z > 50
MODIFIED_FOLIAS_Z_COEF = 0.6275  # spec 6.6: M = sqrt(1 + 0.6275 z - 0.003375 z^2)
MODIFIED_FOLIAS_Z2_COEF = 0.003375  # spec 6.6 (same expression)
MODIFIED_FOLIAS_LINEAR_SLOPE = 0.032  # spec 6.6: M = 0.032 z + 3.3 for z > 50
MODIFIED_FOLIAS_LINEAR_INTERCEPT = 3.3  # spec 6.6 (same expression)
MODIFIED_DEPTH_FACTOR = 0.85  # spec 6.6: S_F = S_flow [1 - 0.85 (d/t)] / [1 - 0.85 (d/t)/M]

BURST_PRESSURE_FACTOR = 2.0  # spec 6.6: P_F = 2 S_F t / D
WALL_LOSS_LIMIT = 0.80  # spec 6.6: d/t >= 0.80 is flagged as exceeding 80 % of wall


def assessed_depth_in(depth_pct, wt_in, settings: Settings):
    """Assessed depth in inches (spec 6.6, ``strength.depth_basis``).

    ``reported``: depth_pct. ``reported_plus_tolerance``: depth_pct + tool.depth_tolerance_pct_wt
    (% of wall), capped at 100 % of wall. Converted to inches = pct / 100 * wt_in. A negative reported
    depth (a data error) is clipped to 0 first, so the tolerance is never partly cancelled by it.
    """
    pct = np.maximum(np.asarray(depth_pct, dtype=float), 0.0)  # NaN stays NaN
    if settings.strength.depth_basis == "reported_plus_tolerance":
        pct = np.minimum(pct + settings.tool.depth_tolerance_pct_wt, PCT_PER_FRACTION)
    return pct / PCT_PER_FRACTION * np.asarray(wt_in, dtype=float)


def assessed_length_in(length_in, settings: Settings):
    """Assessed length in inches (spec 6.6, ``strength.length_basis``).

    ``reported``: length_in. ``reported_plus_tolerance``: length_in + tool.length_tolerance_in.
    """
    length = np.asarray(length_in, dtype=float)
    if settings.strength.length_basis == "reported_plus_tolerance":
        length = length + settings.tool.length_tolerance_in
    return length


def z_param(L, D, t):
    """Length parameter z = L^2 / (D t), dimensionless (L, D, t in inches)."""
    return np.asarray(L, dtype=float) ** 2 / (np.asarray(D, dtype=float) * np.asarray(t, dtype=float))


def folias_original(z):
    """Original B31G Folias factor M = sqrt(1 + 0.8 z) for z <= 20; NaN for z > 20 (unused there)."""
    z = np.asarray(z, dtype=float)
    with np.errstate(invalid="ignore"):
        m = np.sqrt(1.0 + ORIGINAL_FOLIAS_Z_COEF * z)
        return np.where(z <= ORIGINAL_Z_LIMIT, m, np.nan)


def folias_modified(z):
    """Modified B31G Folias factor: sqrt(1 + 0.6275 z - 0.003375 z^2) for z <= 50, else 0.032 z + 3.3."""
    z = np.asarray(z, dtype=float)
    with np.errstate(invalid="ignore"):
        quad = np.sqrt(1.0 + MODIFIED_FOLIAS_Z_COEF * z - MODIFIED_FOLIAS_Z2_COEF * z**2)
        lin = MODIFIED_FOLIAS_LINEAR_SLOPE * z + MODIFIED_FOLIAS_LINEAR_INTERCEPT
        return np.where(np.isnan(z), np.nan, np.where(z <= MODIFIED_Z_LIMIT, quad, lin))


def _dt_ratio(d, t):
    """d/t clipped to [0, 1]: negative depths act as no loss, depths beyond the wall as through-wall."""
    return np.clip(np.asarray(d, dtype=float) / np.asarray(t, dtype=float), 0.0, 1.0)


def failure_original_b31g(D, t, d, L, smys):
    """Original B31G -> (z, M, S_flow, S_F), all arrays; stresses in psi.

    S_flow = 1.1 SMYS. z <= 20: M = sqrt(1 + 0.8 z), S_F = S_flow [1 - (2/3)(d/t)] / [1 - (2/3)(d/t)/M].
    z > 20: S_F = S_flow (1 - d/t) and M is NaN (unused). d/t is clipped to [0, 1].
    """
    z = z_param(L, D, t)
    r = _dt_ratio(d, t)
    flow = ORIGINAL_FLOW_FACTOR * np.asarray(smys, dtype=float)
    m = folias_original(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        folias = flow * (1.0 - ORIGINAL_DEPTH_FACTOR * r) / (1.0 - ORIGINAL_DEPTH_FACTOR * r / m)
        long_form = flow * (1.0 - r)
    s_f = np.where(z <= ORIGINAL_Z_LIMIT, folias, long_form)
    s_f = np.where(np.isnan(z) | np.isnan(r), np.nan, s_f)
    flow = np.broadcast_to(flow, s_f.shape).copy()
    return z, m, flow, s_f


def failure_modified_b31g(D, t, d, L, smys):
    """Modified B31G (0.85 dL) -> (z, M, S_flow, S_F), all arrays; stresses in psi.

    S_flow = SMYS + 10,000. M = sqrt(1 + 0.6275 z - 0.003375 z^2) for z <= 50, else 0.032 z + 3.3.
    S_F = S_flow [1 - 0.85 (d/t)] / [1 - 0.85 (d/t)/M]. d/t is clipped to [0, 1].
    """
    z = z_param(L, D, t)
    r = _dt_ratio(d, t)
    flow = np.asarray(smys, dtype=float) + MODIFIED_FLOW_ADDER_PSI
    m = folias_modified(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_f = flow * (1.0 - MODIFIED_DEPTH_FACTOR * r) / (1.0 - MODIFIED_DEPTH_FACTOR * r / m)
    s_f = np.where(np.isnan(z) | np.isnan(r), np.nan, s_f)
    flow = np.broadcast_to(flow, s_f.shape).copy()
    return z, m, flow, s_f


STRENGTH_METHODS: dict[str, Callable] = {
    "original_b31g": failure_original_b31g,
    "modified_b31g": failure_modified_b31g,
}


def failure_stress(method: str, D, t, d, L, smys):
    """Dispatch to the configured method -> (z, M, S_flow, S_F); ValueError for an unknown method."""
    try:
        fn = STRENGTH_METHODS[method]
    except KeyError:
        raise ValueError(f"unknown strength method {method!r}; expected one of {sorted(STRENGTH_METHODS)}") from None
    return fn(D, t, d, L, smys)


def burst_pressure(S_F, t, D):
    """Predicted burst pressure in psig = 2 S_F t / D."""
    return BURST_PRESSURE_FACTOR * np.asarray(S_F, dtype=float) * np.asarray(t, dtype=float) / np.asarray(D, dtype=float)


def safe_pressure(P_F, settings: Settings):
    """Safe operating pressure in psig = P_F * strength.design_factor."""
    return np.asarray(P_F, dtype=float) * settings.strength.design_factor


def exceeds_wall_loss_limit(d, t):
    """True where d/t >= 0.80 (spec 6.6), regardless of the pressure result; d/t is not clipped here."""
    return np.asarray(d, dtype=float) / np.asarray(t, dtype=float) >= WALL_LOSS_LIMIT


def rstreng_effective_area(profile, D, t, smys):
    """RSTRENG effective-area interface (not implemented in v1).

    ``profile`` will be a table of river-bottom depth readings along the defect: axial position
    (in) and depth (in) pairs, ordered by position, from a vendor-supplied detailed profile.
    It is not in ``STRENGTH_METHODS`` and cannot be selected in config.
    """
    raise NotImplementedError("RSTRENG is out of scope for v1 (spec §6.6)")
