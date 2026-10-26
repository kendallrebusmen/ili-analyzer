import math

import numpy as np
import pytest

from ili import growth
from ili.config import load_settings


@pytest.fixture
def settings():
    return load_settings()


def _with_growth(settings, **kw):
    return settings.model_copy(update={"growth": settings.growth.model_copy(update=kw)})


def test_sigma_diff(settings):
    assert growth.sigma_diff_pct(settings.tool) == pytest.approx(11.035, abs=1e-3)


def test_depth_in():
    out = growth.depth_in(np.array([40.0, np.nan]), 0.25)
    assert out[0] == pytest.approx(0.1)
    assert math.isnan(out[1])


def test_raw_rate():
    assert growth.raw_rate(0.075, 0.1, 5.0) == pytest.approx(5.0)
    assert growth.raw_rate(0.075, 0.0625, 5.0) == pytest.approx(-2.5)
    out = growth.raw_rate(np.array([0.075, np.nan]), np.array([0.1, 0.1]), 5.0)
    assert out[0] == pytest.approx(5.0)
    assert math.isnan(out[1])


def test_tolerance_adjusted(settings):
    expected = 1.28 * 11.035 / 100 * 0.25 * 1000 / 5
    assert growth.tolerance_adjusted_rate(0.1, 0.1, 0.25, 5.0, settings) == pytest.approx(expected, abs=0.01)
    assert expected == pytest.approx(7.06, abs=0.01)
    assert growth.tolerance_adjusted_rate(0.2, 0.05, 0.25, 5.0, settings) == pytest.approx(1.0)
    out = growth.tolerance_adjusted_rate(np.array([0.1, np.nan]), np.array([0.1, 0.1]), 0.25, 5.0, settings)
    assert out[0] == pytest.approx(expected, abs=0.01)
    assert math.isnan(out[1])


def test_new_feature(settings):
    d = 0.18 * 0.25
    assert growth.new_feature_rate(d, 0.25, 5.0, settings) == pytest.approx(4.0)
    out = growth.new_feature_rate(np.array([d, np.nan]), 0.25, 5.0, settings)
    assert out[0] == pytest.approx(4.0)
    assert math.isnan(out[1])
    # below the reporting threshold the rate is negative, not floored (the growth table flags it)
    assert growth.new_feature_rate(0.06 * 0.25, 0.25, 5.0, settings) == pytest.approx(-2.0)


def test_full_life():
    d = 0.4 * 0.25
    assert growth.full_life_rate(d, 2026, 1975) == pytest.approx(1.961, abs=1e-3)
    out = growth.full_life_rate(d, 2026, np.array([2026.0, 2030.0, 2023.0, np.nan]))
    assert math.isnan(out[0]) and math.isnan(out[1]) and math.isnan(out[3])
    assert out[2] == pytest.approx(d * 1000 / 3)
    assert not np.isinf(out).any()


def test_default_rate_null_is_nan(settings):
    out = growth.default_rate(3, _with_growth(settings, default_rate_mpy=None))
    assert out.shape == (3,) and np.isnan(out).all()


def test_default_rate_set(settings):
    out = growth.default_rate(3, _with_growth(settings, default_rate_mpy=2.0))
    assert out.tolist() == [2.0, 2.0, 2.0]
