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


# --- M11: half-life, joint percentile, shrinking-fixed, recoat -----------------------------------------


def test_half_life_is_twice_full_life():
    out = growth.half_life_rate([0.1, 0.1], 2026, [1986, 2026])
    assert out[0] == pytest.approx(5.0) and math.isnan(out[1])
    assert math.isnan(growth.half_life_rate([0.1], 2026, [float("nan")])[0])


def test_joint_percentile_min_matches(settings):
    raw = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    cls = np.array(["matched"] * 4 + ["new"], dtype=object)
    joint = np.array(["J1"] * 4 + ["J1"], dtype=object)
    stats = np.ones(5, dtype=bool)
    s = _with_growth(settings, joint_percentile=90, joint_percentile_min_matches=3)
    rate, unfloored = growth.joint_percentile(raw, cls, joint, stats, s)
    assert rate == pytest.approx([3.7] * 5) and unfloored["J1"] == pytest.approx(3.7)
    s5 = _with_growth(settings, joint_percentile=90, joint_percentile_min_matches=5)
    assert np.isnan(growth.joint_percentile(raw, cls, joint, stats, s5)[0]).all()
    neg = _with_growth(settings, joint_percentile=0, joint_percentile_min_matches=1)
    low = growth.joint_percentile(np.array([-5.0, -2.0]), np.array(["matched"] * 2, dtype=object),
                                  np.array(["J", "J"], dtype=object), np.ones(2, dtype=bool), neg)[0]
    assert low == pytest.approx([settings.growth.min_rate_mpy] * 2)
    blank = growth.joint_percentile(raw, cls, np.array([None] * 5, dtype=object), stats, s)[0]
    assert np.isnan(blank).all()


def test_shrinking_fixed_only_when_depth_drops(settings):
    cls = np.array(["matched", "matched"], dtype=object)
    s = _with_growth(settings, shrinking_rate_mpy=6.0)
    out = growth.shrinking_fixed_rate([0.10, 0.10], [0.08, 0.12], cls, s)
    assert out[0] == 6.0 and math.isnan(out[1])
    off = _with_growth(settings, shrinking_rate_mpy=None)
    assert np.isnan(growth.shrinking_fixed_rate([0.10, 0.10], [0.08, 0.12], cls, off)).all()


def test_in_recoat():
    import pandas as pd
    rep = pd.DataFrame({"start_ft": [10.0, 50.0], "end_ft": [20.0, 60.0], "repair_type": ["recoat", "sleeve"]})
    assert growth.in_recoat([5.0, 10.0, 20.0, 55.0], rep).tolist() == [False, True, True, False]
    assert not growth.in_recoat([15.0], None).any()
    assert not growth.in_recoat([15.0], rep[["start_ft", "end_ft"]]).any()


def test_segment_percentile_in_stats_excludes_rows(settings):
    import pandas as pd
    seg = pd.DataFrame({"segment_id": ["S"], "start_ft": [0.0], "end_ft": [10.0]})
    raw = np.array([1.0, 2.0, 50.0])
    cls = np.array(["matched"] * 3, dtype=object)
    sid = np.array(["S"] * 3, dtype=object)
    with_all, _ = growth.segment_percentile(raw, cls, sid, seg, settings)
    kept, _ = growth.segment_percentile(raw, cls, sid, seg, settings, np.array([True, True, False]))
    assert kept[2] == kept[0] and kept[0] < with_all[0]
