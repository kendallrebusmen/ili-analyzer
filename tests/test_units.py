import numpy as np
import pytest

from ili.units import (
    FT_PER_M,
    IN_PER_MM,
    clock_min_to_degrees,
    clock_min_to_hhmm,
    decimal_hours_to_clock_min,
    degrees_to_clock_min,
    hhmm_to_clock_min,
    sigma_from_tolerance,
    wrap_clock_min,
)


def test_sigma_from_tolerance():
    assert sigma_from_tolerance(10, 0.80) == pytest.approx(7.803, abs=1e-3)
    assert sigma_from_tolerance(0.75, 0.80) == pytest.approx(0.5852, abs=1e-4)
    assert sigma_from_tolerance(15, 0.80) == pytest.approx(11.70, abs=1e-2)


def test_unit_constants():
    assert 1 / FT_PER_M == pytest.approx(0.3048)
    assert 1 / IN_PER_MM == pytest.approx(25.4)


def test_wrap():
    assert wrap_clock_min(719.6) == 0
    assert wrap_clock_min(-15) == 705
    assert wrap_clock_min(735) == 15


def test_wrap_array():
    out = wrap_clock_min(np.array([719.6, -15.0, 735.0, 360.4]))
    assert out.tolist() == [0, 705, 15, 360]


def test_hhmm_roundtrip():
    assert clock_min_to_hhmm(0) == "12:00"
    assert clock_min_to_hhmm(30) == "12:30"
    assert clock_min_to_hhmm(60) == "1:00"
    assert clock_min_to_hhmm(719) == "11:59"
    assert all(hhmm_to_clock_min(clock_min_to_hhmm(m)) == m for m in range(720))


def test_degrees():
    assert degrees_to_clock_min(90) == 180
    assert degrees_to_clock_min(360) == 0
    assert degrees_to_clock_min(359.9) == 0
    assert clock_min_to_degrees(180) == 90.0


def test_decimal_hours():
    assert decimal_hours_to_clock_min(3.5) == 210
    assert decimal_hours_to_clock_min(12.0) == 0


def test_hhmm_rejects_garbage():
    for bad in ["13:00", "3:60", "abc", "3", "-1:00"]:
        with pytest.raises(ValueError):
            hhmm_to_clock_min(bad)


def test_hhmm_accepts_zero_hour_as_twelve():
    assert hhmm_to_clock_min("0:30") == 30
    assert hhmm_to_clock_min(" 03:15 ") == 195
