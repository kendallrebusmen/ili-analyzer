"""Unit conversions, clock-position formats, and tolerance-to-sigma conversion.

Internal clock convention: ``clock_min`` is an integer 0-719, minutes clockwise
from 12:00 looking downstream. 12:00 is 0, 3:00 is 180, 11:59 is 719.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import norm

FT_PER_M = 1 / 0.3048
IN_PER_MM = 1 / 25.4

CLOCK_MINUTES = 720
_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def sigma_from_tolerance(tol: float, certainty: float) -> float:
    """Standard deviation for a two-sided +/-tol at the given certainty (normal).

    Example: +/-10 % of wall at 80 % certainty gives sigma of about 7.80 % of wall.
    """
    if not 0 < certainty < 1:
        raise ValueError(f"certainty must be in (0, 1), got {certainty}")
    return tol / norm.ppf((1 + certainty) / 2)


def wrap_clock_min(x):
    """Round to the nearest minute and wrap into 0-719. Accepts scalars or arrays."""
    wrapped = np.mod(np.rint(x), CLOCK_MINUTES).astype(int)
    return int(wrapped) if np.ndim(wrapped) == 0 else wrapped


def clock_min_to_hhmm(m: int) -> str:
    """12-hour clock-face string: 0 -> '12:00', 30 -> '12:30', 60 -> '1:00'."""
    m = wrap_clock_min(m)
    hours, minutes = divmod(m, 60)
    return f"{hours or 12}:{minutes:02d}"


def hhmm_to_clock_min(s: str) -> int:
    """Parse 'h:mm' on a 12-hour face. '12:xx' and '0:xx' both mean xx minutes."""
    match = _HHMM.match(str(s))
    if not match:
        raise ValueError(f"not an h:mm clock position: {s!r}")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 12 or minutes > 59:
        raise ValueError(f"clock position out of range: {s!r}")
    return (hours % 12) * 60 + minutes


def degrees_to_clock_min(deg):
    """Degrees clockwise from top (0-360) to clock minutes; 360 deg = 720 min."""
    return wrap_clock_min(np.asarray(deg, dtype=float) * 2)


def clock_min_to_degrees(m):
    return np.asarray(m, dtype=float) / 2 if np.ndim(m) else float(m) / 2


def decimal_hours_to_clock_min(h):
    """Decimal clock hours (e.g. 3.5 = 3:30) to clock minutes; 12.0 -> 0."""
    return wrap_clock_min(np.asarray(h, dtype=float) * 60)
