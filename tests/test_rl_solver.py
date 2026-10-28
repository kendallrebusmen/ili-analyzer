import math

import numpy as np
import pytest

import ili.remaining_life as rl_mod
from ili.config import load_settings
from ili.remaining_life import CRITERIA, criterion_margins, remaining_life, time_to_criteria

D, T, SMYS = 12.75, 0.250, 52_000.0


def _settings(method="modified_b31g", **rl_updates):
    s = load_settings()
    return s.model_copy(update={
        "strength": s.strength.model_copy(update={"method": method}),
        "remaining_life": s.remaining_life.model_copy(update=rl_updates),
    })


def _ttc(d0, L0, g, gL, mop, settings=None, D=D, t=T, smys=SMYS):
    return time_to_criteria(d0, L0, g, gL, D, t, smys, mop, settings or _settings())


def test_criteria_names_and_settings():
    assert CRITERIA == ("tau_80", "tau_burst", "tau_safe")
    assert load_settings().remaining_life.time_step_yr == 0.05


def test_tau80_linear():
    times = _ttc(d0=0.10, L0=1.0, g=0.005, gL=0.0, mop=500.0)
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-4)
    assert times["tau_80"] <= 20.0  # never later than the true crossing
    assert remaining_life(times, _settings()) == (times["tau_80"], "tau_80")


def test_already_met_is_zero():
    times = _ttc(d0=0.85 * T, L0=1.0, g=0.001, gL=0.0, mop=500.0)
    assert times["tau_80"] == 0.0
    # Reference case (d 0.125, L 4): P_F 1771.5, P_safe = 0.72 * 1771.5 = 1275.4 < MOP 1300 today.
    times = _ttc(d0=0.125, L0=4.0, g=0.001, gL=0.0, mop=1300.0)
    assert times["tau_safe"] == 0.0
    assert times["tau_burst"] > 0.0
    assert remaining_life(times, _settings()) == (0.0, "tau_safe")


def test_beyond_horizon():
    s = _settings()
    times = _ttc(d0=0.05, L0=2.0, g=0.0, gL=0.0, mop=500.0, settings=s)
    assert all(times[c] == math.inf for c in CRITERIA)
    assert remaining_life(times, s) == (s.remaining_life.horizon_yr, "beyond_horizon")


def _modified_pf_closed_form(tau, d0, L0, g, gL):
    """P_F written out from spec 6.6 (Modified B31G), independent of ili.strength."""
    d, L = d0 + g * tau, L0 + gL * tau
    z = L**2 / (D * T)
    m = math.sqrt(1 + 0.6275 * z - 0.003375 * z**2) if z <= 50 else 0.032 * z + 3.3
    r = min(d / T, 1.0)
    sf = (SMYS + 10_000) * (1 - 0.85 * r) / (1 - 0.85 * r / m)
    return 2 * sf * T / D


def _bisect(f, lo, hi, tol=1e-10):
    assert f(lo) > 0 >= f(hi)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def test_burst_crossing_matches_direct_solve():
    d0, L0, g, gL, mop = 0.125, 4.0, 0.004, 0.05, 1200.0
    times = _ttc(d0, L0, g, gL, mop)
    expected = _bisect(lambda tau: _modified_pf_closed_form(tau, d0, L0, g, gL) - mop, 0.0, 30.0)
    assert 0 < expected < 30
    assert times["tau_burst"] == pytest.approx(expected, abs=1e-4)
    assert times["tau_burst"] <= expected + 1e-9
    safe = _bisect(lambda tau: 0.72 * _modified_pf_closed_form(tau, d0, L0, g, gL) - mop, 0.0, 30.0)
    assert times["tau_safe"] == pytest.approx(safe, abs=1e-4)
    assert times["tau_safe"] < times["tau_burst"]


@pytest.mark.parametrize("mop", [1400.0, 1200.0, 1130.0])
def test_original_b31g_jump_found(mop):
    # d/t 0.5 constant; L grows 6.0 -> through z = 20 at L = sqrt(20 D t) = 7.98436 in.
    # Just below z = 20, P_F ~ 1627 > MOP; just above, P_F ~ 1122 < MOP. The jump is the crossing.
    # MOP 1200 / 1130 put the post-jump side closer to zero, where raw brentq lands just AFTER the jump.
    s = _settings("original_b31g")
    L0, gL = 6.0, 0.2
    tau_jump = (math.sqrt(20 * D * T) - L0) / gL
    assert tau_jump == pytest.approx(9.9218, abs=1e-3)
    times = _ttc(d0=0.125, L0=L0, g=0.0, gL=gL, mop=mop, settings=s)
    assert times["tau_burst"] == pytest.approx(tau_jump, abs=1e-3)
    assert times["tau_burst"] <= tau_jump  # not after the true crossing
    m = criterion_margins(np.array([tau_jump - 1e-3, times["tau_burst"]]), 0.125, L0, 0.0, gL, D, T, SMYS, mop, s)
    assert m["tau_burst"][0] > 0  # still on the high side just before


@pytest.mark.parametrize("jump_on_grid", [False, True])
def test_original_b31g_jump_on_or_off_grid(jump_on_grid):
    s = _settings("original_b31g")
    L_jump = math.sqrt(20 * D * T)
    tau_jump = 10.0 if jump_on_grid else 10.025
    gL = 0.2
    L0 = L_jump - gL * tau_jump
    times = _ttc(d0=0.125, L0=L0, g=0.0, gL=gL, mop=1400.0, settings=s)
    assert tau_jump - 1e-3 <= times["tau_burst"] <= tau_jump


def test_negative_growth_treated_as_zero():
    s = _settings()
    neg = _ttc(d0=0.10, L0=2.0, g=-0.01, gL=-0.1, mop=500.0, settings=s)
    zero = _ttc(d0=0.10, L0=2.0, g=0.0, gL=0.0, mop=500.0, settings=s)
    assert neg == zero
    assert all(neg[c] == math.inf for c in CRITERIA)
    tau = np.array([0.0, 10.0, 30.0])
    m_neg = criterion_margins(tau, 0.10, 2.0, -0.01, -0.1, D, T, SMYS, 500.0, s)
    m_zero = criterion_margins(tau, 0.10, 2.0, 0.0, 0.0, D, T, SMYS, 500.0, s)
    for c in CRITERIA:
        np.testing.assert_array_equal(m_neg[c], m_zero[c])


@pytest.mark.parametrize("field", ["d0", "L0", "g", "gL", "D", "t", "smys", "mop"])
def test_nan_input_gives_nan_rl(field):
    kwargs = dict(d0=0.10, L0=2.0, g=0.005, gL=0.01, D=D, t=T, smys=SMYS, mop=500.0)
    kwargs[field] = math.nan
    s = _settings()
    times = time_to_criteria(**kwargs, settings=s)
    assert any(math.isnan(times[c]) for c in CRITERIA)
    rl, gov = remaining_life(times, s)
    assert math.isnan(rl) and gov is None


def test_nan_mop_leaves_tau80_but_not_rl():
    s = _settings()
    times = _ttc(d0=0.10, L0=2.0, g=0.005, gL=0.0, mop=math.nan, settings=s)
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-4)
    assert math.isnan(times["tau_burst"]) and math.isnan(times["tau_safe"])
    rl, gov = remaining_life(times, s)
    assert math.isnan(rl) and gov is None


def test_nonpositive_wall_gives_nan():
    kwargs = dict(d0=0.10, L0=2.0, g=0.005, gL=0.01, D=D, t=0.0, smys=SMYS, mop=500.0)
    times = time_to_criteria(**kwargs, settings=_settings())
    assert all(math.isnan(times[c]) for c in CRITERIA)


@pytest.mark.parametrize("field,value", [("mop", 0.0), ("mop", -100.0), ("D", 0.0), ("D", -12.75),
                                         ("smys", 0.0), ("smys", -52_000.0), ("L0", -3.0)])
def test_unusable_reference_value_is_missing(field, value):
    # An OD, SMYS or MOP <= 0 (or a negative length) is missing data: the pressure criteria are NaN, not
    # silently never met (MOP 0 used to give all inf -> beyond_horizon, unflagged). tau_80 is kept.
    s = _settings()
    kwargs = dict(d0=0.10, L0=2.0, g=0.005, gL=0.5, D=D, t=T, smys=SMYS, mop=500.0)
    kwargs[field] = value
    times = time_to_criteria(**kwargs, settings=s)
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-4)
    assert math.isnan(times["tau_burst"]) and math.isnan(times["tau_safe"])
    rl, gov = remaining_life(times, s)
    assert math.isnan(rl) and gov is None
    m = criterion_margins(np.array([0.0, 5.0]), **kwargs, settings=s)
    assert np.isnan(m["tau_burst"]).all() and np.isnan(m["tau_safe"]).all() and np.isfinite(m["tau_80"]).all()


def test_zero_length_is_usable():
    times = _ttc(d0=0.10, L0=0.0, g=0.005, gL=0.0, mop=500.0)
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-4)
    assert not math.isnan(times["tau_burst"]) and not math.isnan(times["tau_safe"])


def test_depth_through_wall_stays_finite():
    s = _settings()
    d0, L0, g, gL, mop = 0.10, 2.0, 0.05, 0.0, 300.0  # reaches d = t at tau = 3 yr
    tau = np.linspace(0.0, 30.0, 601)
    m = criterion_margins(tau, d0, L0, g, gL, D, T, SMYS, mop, s)
    for c in CRITERIA:
        assert np.isfinite(m[c]).all()
    assert (m["tau_burst"] + mop >= 0).all()  # P_F stays non-negative past d = t
    times = _ttc(d0, L0, g, gL, mop, settings=s)
    assert times["tau_80"] == pytest.approx(2.0, abs=1e-4)
    rl, gov = remaining_life(times, s)
    assert gov == "tau_80" and rl == times["tau_80"]


def test_ties_follow_criteria_order():
    s = _settings()
    assert remaining_life({"tau_80": 5.0, "tau_burst": 5.0, "tau_safe": 7.0}, s) == (5.0, "tau_80")
    assert remaining_life({"tau_80": math.inf, "tau_burst": 3.0, "tau_safe": 3.0}, s) == (3.0, "tau_burst")
    assert remaining_life({"tau_80": 0.0, "tau_burst": 0.0, "tau_safe": 0.0}, s) == (0.0, "tau_80")
    assert remaining_life({"tau_80": math.inf, "tau_burst": math.inf, "tau_safe": 4.0}, s) == (4.0, "tau_safe")


def test_horizon_not_multiple_of_step_is_included():
    # tau_80 = 1.01 yr; grid 0, 0.05, ..., 1.00, then the horizon 1.02 itself.
    s = _settings(horizon_yr=1.02)
    times = _ttc(d0=0.10, L0=1.0, g=0.1 / 1.01, gL=0.0, mop=500.0, settings=s)
    assert times["tau_80"] == pytest.approx(1.01, abs=1e-4)
    s = _settings(horizon_yr=1.0)
    assert _ttc(d0=0.10, L0=1.0, g=0.1 / 1.01, gL=0.0, mop=500.0, settings=s)["tau_80"] == math.inf


def test_brentq_failure_falls_back_to_interval_start(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("did not converge")

    monkeypatch.setattr(rl_mod, "brentq", broken)
    times = _ttc(d0=0.10, L0=1.0, g=0.1 / 20.02, gL=0.0, mop=500.0)
    # True crossing 20.02 yr lies in the grid interval [20.00, 20.05]: the fallback takes its START,
    # which is before the crossing (conservative), never the END, which is after it.
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-9)
    assert times["tau_80"] < 20.02


# --- exact-zero plateaus: the first time the margin reaches 0, not a later point on the plateau -----------


def test_plateau_after_original_jump_not_late():
    # Original B31G, no depth growth: after the z = 20 jump (tau 1.013) S_F is the constant long form.
    # With MOP equal to that P_safe the margin is exactly 0 from 1.013 on; brentq used to return the
    # grid END 1.05 (stepped back by 1e-6), later than the true crossing.
    s = _settings("original_b31g")
    L0 = math.sqrt(20 * D * T) - 1.013  # gL 1 in/yr -> z = 20 at tau = 1.013
    _, _, _, s_long = rl_mod.failure_stress("original_b31g", D, T, 0.125, 100.0, SMYS)
    mop = float(rl_mod.safe_pressure(rl_mod.burst_pressure(s_long, T, D), s))
    m = criterion_margins(np.array([1.0, 1.05, 2.0]), 0.125, L0, 0.0, 1.0, D, T, SMYS, mop, s)["tau_safe"]
    assert m[0] > 0 and m[1] == 0 and m[2] == 0  # the plateau the fix is about
    times = _ttc(d0=0.125, L0=L0, g=0.0, gL=1.0, mop=mop, settings=s)
    assert 1.013 - 1e-5 <= times["tau_safe"] <= 1.013
    assert criterion_margins(times["tau_safe"], 0.125, L0, 0.0, 1.0, D, T, SMYS, mop, s)["tau_safe"] > 0
    assert remaining_life(times, s) == (times["tau_safe"], "tau_safe")


def test_plateau_at_through_wall_not_late():
    # Modified B31G, gL 0: once depth reaches the wall d/t is clipped at 1 and P_F is constant. With MOP
    # equal to that P_F the burst margin is exactly 0 from tau = 0.15 / g = 1.5195 yr on.
    s = _settings()
    L0, d0, g = 2.0, 0.10, 0.10 / 1.013
    _, _, _, s_wall = rl_mod.failure_stress("modified_b31g", D, T, T, L0, SMYS)
    mop = float(rl_mod.burst_pressure(s_wall, T, D))
    true = (T - d0) / g
    times = _ttc(d0=d0, L0=L0, g=g, gL=0.0, mop=mop, settings=s)
    assert true - 1e-5 <= times["tau_burst"] <= true


def test_bisect_met_never_late_and_stops_on_nan():
    def step(tau):  # met (exactly 0) from 0.3 on
        return 1.0 if tau < 0.3 else 0.0

    x = rl_mod._bisect_met(step, 0.0, 1.0)
    assert 0.3 - rl_mod.BRENTQ_XTOL_YR <= x < 0.3
    assert rl_mod._bisect_met(lambda tau: math.nan, 0.0, 1.0) == 0.0


# --- a criterion met today is met regardless of missing growth or other inputs --------------------------


@pytest.mark.parametrize("field", ["g", "gL"])
def test_met_today_is_zero_despite_nan_growth(field):
    s = _settings()
    deep = dict(d0=0.85 * T, L0=1.0, g=0.001, gL=0.0, mop=500.0)
    deep[field] = math.nan
    times = _ttc(**deep, settings=s)
    assert times["tau_80"] == 0.0
    assert remaining_life(times, s) == (0.0, "tau_80")
    # Reference case at MOP 1300: P_safe 1275.4 < 1300 < P_F 1771.5. tau_burst needs both rates (NaN);
    # tau_80 needs only g (NaN for a NaN g; with g 0.001 it is 75 yr away, beyond the horizon);
    # tau_safe is met today.
    ref = dict(d0=0.125, L0=4.0, g=0.001, gL=0.0, mop=1300.0)
    ref[field] = math.nan
    times = _ttc(**ref, settings=s)
    assert times["tau_safe"] == 0.0 and math.isnan(times["tau_burst"])
    assert math.isnan(times["tau_80"]) if field == "g" else times["tau_80"] == math.inf
    assert remaining_life(times, s) == (0.0, "tau_safe")


def test_nan_mop_with_deep_feature_is_zero():
    s = _settings()
    times = _ttc(d0=0.80 * T, L0=1.0, g=0.005, gL=0.0, mop=math.nan, settings=s)
    assert times["tau_80"] == 0.0
    assert math.isnan(times["tau_burst"]) and math.isnan(times["tau_safe"])
    assert remaining_life(times, s) == (0.0, "tau_80")


@pytest.mark.parametrize("field", ["D", "smys"])
def test_nan_pipe_value_keeps_tau80(field):
    s = _settings()
    deep = dict(d0=0.85 * T, L0=1.0, g=0.001, gL=0.0, mop=500.0, **{field: math.nan})
    times = _ttc(**deep, settings=s)
    assert times["tau_80"] == 0.0 and math.isnan(times["tau_burst"]) and math.isnan(times["tau_safe"])
    assert remaining_life(times, s) == (0.0, "tau_80")
    healthy = dict(d0=0.10, L0=1.0, g=0.005, gL=0.0, mop=500.0, **{field: math.nan})
    times = _ttc(**healthy, settings=s)
    assert times["tau_80"] == pytest.approx(20.0, abs=1e-4)
    assert remaining_life(times, s)[1] is None


def test_remaining_life_zero_beats_nan():
    s = _settings()
    assert remaining_life({"tau_80": math.nan, "tau_burst": 0.0, "tau_safe": 0.0}, s) == (0.0, "tau_burst")
    assert remaining_life({"tau_80": math.nan, "tau_burst": math.nan, "tau_safe": 0.0}, s) == (0.0, "tau_safe")
    rl, gov = remaining_life({"tau_80": math.nan, "tau_burst": 5.0, "tau_safe": math.inf}, s)
    assert math.isnan(rl) and gov is None


# --- strength_at, failure_mode, condition_date (M11 task 8) ------------------------------------------


def _ref(settings, tau=0.0, g=0.0, mop=1000.0):
    return rl_mod.strength_at(tau, 0.125, 4.0, g, 0.0, 12.75, 0.25, 52_000.0, mop, settings)


def test_strength_at_reference_case():
    settings = load_settings()
    out = _ref(settings)
    assert float(out["burst_psig"]) == pytest.approx(1771.5, abs=1)
    p_f = float(out["burst_psig"])
    assert float(out["safe_psig"]) == pytest.approx(0.72 * p_f, rel=1e-9)
    assert float(out["erf"]) == pytest.approx(1000.0 / (0.72 * p_f), rel=1e-9)
    assert float(out["fpr"]) == pytest.approx(p_f / 1000.0, rel=1e-9)


def test_strength_at_projects_growth():
    settings = load_settings()
    now, later = _ref(settings), _ref(settings, tau=5.0, g=0.01)
    assert float(later["depth_in"]) == pytest.approx(0.125 + 0.05)
    assert float(later["burst_psig"]) < float(now["burst_psig"])


@pytest.mark.parametrize("mop", [0.0, -5.0, float("nan")])
def test_erf_blank_for_bad_mop(mop):
    out = _ref(load_settings(), mop=mop)
    assert np.isnan(out["erf"]) and np.isnan(out["fpr"])


def test_failure_mode():
    inf, nan = float("inf"), float("nan")
    got = rl_mod.failure_mode([3, 5, 6, inf, nan], [5, 5, 5, inf, 5])
    assert list(got) == ["leak", "leak", "burst", None, None]


def test_condition_dates():
    from datetime import date
    run = date(2026, 6, 15)
    assert rl_mod.condition_date(run, 0.0) == run
    assert rl_mod.condition_date(run, 1.0) == date(2027, 6, 15)
    assert rl_mod.condition_date(run, float("inf")) is None
    assert rl_mod.condition_date(run, float("nan")) is None
