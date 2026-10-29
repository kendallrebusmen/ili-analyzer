"""Growth options that reproduce the consultant's practice (M12 parity): the year basis, rounding of the
governing rate, the fixed rate for no-change matches, and the joint-percentile pool of assigned rates.
Every option defaults to the earlier behaviour."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from test_growth_table import _by_id, _row, _run, _with_growth

from ili import growth
from ili.config import load_settings
from ili.match import delta_t_years


@pytest.fixture
def settings():
    return load_settings()


def _pit(settings, **kw):
    base = dict(governing_profile="pit_to_pit", shrinking_rate_mpy=2.5, min_rate_mpy=0.0)
    return _with_growth(settings, **{**base, **kw})


def test_defaults_keep_earlier_behaviour(settings):
    g = settings.growth
    assert g.days_per_year == 365.25 and g.rate_decimals is None
    assert g.shrinking_applies_to == "decrease" and g.joint_percentile_pool == "raw"


def test_days_per_year():
    prev = pd.DataFrame({"run_date": [pd.Timestamp("2020-08-18")] * 2})
    curr = pd.DataFrame({"run_date": [pd.Timestamp("2026-01-19")] * 2})
    assert delta_t_years(prev, curr) == pytest.approx(1980 / 365.25)
    assert delta_t_years(prev, curr, days_per_year=365.0) == pytest.approx(1980 / 365.0)
    with pytest.raises(ValueError, match="days_per_year"):
        delta_t_years(prev, curr, days_per_year=0.0)


def test_shrinking_rate_for_no_change_when_configured(settings):
    cls = np.array(["matched"] * 3, dtype=object)
    prev, curr = [0.10, 0.10, 0.10], [0.08, 0.10, 0.12]
    out = growth.shrinking_fixed_rate(prev, curr, cls, _with_growth(settings, shrinking_rate_mpy=6.0))
    assert out[0] == 6.0 and math.isnan(out[1]) and math.isnan(out[2])
    s = _with_growth(settings, shrinking_rate_mpy=6.0, shrinking_applies_to="no_increase")
    out = growth.shrinking_fixed_rate(prev, curr, cls, s)
    assert out[0] == 6.0 and out[1] == 6.0 and math.isnan(out[2])


def test_pit_to_pit_zero_change(settings):
    rows = [_row("Z1", "matched_1to1", 100.0, 30.0, 30.0),   # no change
            _row("S1", "matched_1to1", 150.0, 30.0, 26.0),   # shrank
            _row("G1", "matched_1to1", 200.0, 30.0, 36.0)]   # raw 3.0
    f = _by_id(_run(rows, _pit(settings)))
    assert f.loc["Z1", "governing_method"] == "raw" and f.loc["Z1", "governing_mpy"] == pytest.approx(0.0)
    f = _by_id(_run(rows, _pit(settings, shrinking_applies_to="no_increase")))
    assert f.loc["Z1", "governing_method"] == "shrinking_fixed" and f.loc["Z1", "governing_mpy"] == 2.5
    assert f.loc["S1", "governing_method"] == "shrinking_fixed"
    assert f.loc["G1", "governing_method"] == "raw" and f.loc["G1", "governing_mpy"] == pytest.approx(3.0)


def _joint_rows():
    # Raw = 0.5 * delta pct (wt 0.25 in, dt 5 yr). J1 holds three matches and one new anomaly.
    rows = [_row("M1", "matched_1to1", 100.0, 30.0, 30.0),   # raw 0 (no change)
            _row("M2", "matched_1to1", 110.0, 30.0, 32.0),   # raw 1.0
            _row("M3", "matched_1to1", 120.0, 30.0, 26.0),   # raw -2.0 (shrank)
            _row("N1", "new", 130.0, d_curr=20.0)]
    for r in rows:
        r["curr_us_gw_id"] = "J1"
    return rows


def test_joint_pool_raw_is_the_default(settings):
    s = _pit(settings, joint_percentile=100, joint_percentile_min_matches=1)
    f = _by_id(_run(_joint_rows(), s))
    assert f.loc["N1", "governing_method"] == "joint_percentile"
    assert f.loc["N1", "governing_mpy"] == pytest.approx(1.0)  # max of raw [0, 1, -2]


def test_joint_pool_of_assigned_rates(settings):
    s = _pit(settings, joint_percentile=100, joint_percentile_min_matches=1, joint_percentile_pool="assigned",
             shrinking_applies_to="no_increase")
    f = _by_id(_run(_joint_rows(), s))
    # assigned: M1 2.5 (no change), M2 1.0 (raw), M3 2.5 (shrank)
    assert f.loc["N1", "joint_percentile_mpy"] == pytest.approx(2.5)
    assert f.loc["N1", "governing_mpy"] == pytest.approx(2.5)
    s50 = _pit(settings, joint_percentile=50, joint_percentile_min_matches=1, joint_percentile_pool="assigned")
    f = _by_id(_run(_joint_rows(), s50))
    # shrinking only for a decrease: M1 raw 0, M2 1.0, M3 2.5 -> median 1.0
    assert f.loc["N1", "governing_mpy"] == pytest.approx(1.0)


def test_joint_pool_assigned_needs_pit_to_pit(settings):
    with pytest.raises(ValueError, match="joint_percentile_pool"):
        _run(_joint_rows(), _with_growth(settings, joint_percentile_pool="assigned"))


def test_rate_decimals_rounds_the_governing_rate(settings):
    rows = [_row("G1", "matched_1to1", 100.0, 30.0, 36.62)]   # raw 3.31
    f = _by_id(_run(rows, _pit(settings)))
    assert f.loc["G1", "governing_mpy"] == pytest.approx(3.31)
    f = _by_id(_run(rows, _pit(settings, rate_decimals=1)))
    assert f.loc["G1", "governing_mpy"] == pytest.approx(3.3)
    assert f.loc["G1", "raw_mpy"] == pytest.approx(3.31)  # the method columns keep full precision


def test_rate_decimals_rounds_the_joint_pool(settings):
    rows = _joint_rows()
    rows[1]["curr_depth_pct"] = 32.13   # raw 1.065 -> 1.1 when rounded
    s = _pit(settings, joint_percentile=100, joint_percentile_min_matches=1, joint_percentile_pool="assigned",
             rate_decimals=1, shrinking_rate_mpy=None)
    f = _by_id(_run(rows, s))
    assert f.loc["N1", "governing_mpy"] == pytest.approx(1.1)
