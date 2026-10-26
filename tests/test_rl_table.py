import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import ili.remaining_life as rl_mod
from ili import growth, reference
from ili.align_welds import align_welds
from ili.config import load_settings
from ili.match import delta_t_years, match_anomalies
from ili.remaining_life import RLResult, add_months, compute_remaining_life
from ili.synth import SynthParams, generate

FEATURE_COLUMNS = [
    "match_id", "curr_feature_id", "curr_log_dist_ft", "segment_id", "could_affect_hca",
    "od_in", "wt_in", "smys_psi", "mop_psig", "d_reported_pct", "d_assessed_in", "assessed_dt", "exceeds_80pct",
    "length_in", "length_assessed_in",
    "method", "z", "folias_m", "flow_stress_psi", "failure_stress_psi", "burst_psig", "safe_psig",
    "g_inyr", "gl_inyr", "tau_80_yr", "tau_burst_yr", "tau_safe_yr", "rl_yr", "rl_governing",
    "rl_upper_bound_yr", "rl_factor", "rl_factored_yr", "repair_by_date",
    "next_assessment_date", "next_assessment_source", "years_to_next_assessment", "dig_required_by_remaining_life",
    "needs_review", "review_reasons",
]
INTERVAL_COLUMNS = [
    "segment_id", "could_affect_hca", "n_features", "n_rl_not_evaluated", "min_rl_factored_yr",
    "regulatory_max_yr", "recommended_interval_yr", "governing_match_id", "needs_review",
]
NAN, INF = float("nan"), float("inf")
RUN = date(2026, 3, 15)
YR = 365.25
PIPE = pd.DataFrame({"start_ft": [0.0], "end_ft": [10_000.0], "od_in": [12.75], "smys_psi": [52_000.0]})
MOP = pd.DataFrame({"start_ft": [0.0], "end_ft": [10_000.0], "mop_psig": [1000.0]})
# Table order is not position order; [4000, 10000) is outside every segment.
SEGS = pd.DataFrame({
    "start_ft": [1000.0, 0.0, 2000.0, 3000.0],
    "end_ft": [2000.0, 1000.0, 3000.0, 4000.0],
    "segment_id": ["SEG-02", "SEG-01", "SEG-03", "SEG-04"],
    "could_affect_hca": [False, True, True, True],
})
NEXT = pd.DataFrame({
    "segment_id": ["SEG-01", "SEG-04"],
    "planned_next_assessment_date": ["2029-06-30", "2033-03-15"],  # SEG-04: 7 yr out, beyond 68 months
})


@pytest.fixture
def settings():
    return load_settings()


def _g(mid, pos, d_curr=30.0, wt=0.25, length=2.0, gov=5.0, lr=0.0, needs_review=False, reasons=None):
    return {"match_id": mid, "curr_feature_id": f"B-{mid}", "curr_log_dist_ft": pos, "wt_in": wt,
            "d_curr_pct": d_curr, "length_in": length, "governing_mpy": gov, "length_rate_ipy": lr,
            "needs_review": needs_review, "review_reasons": reasons}


def _run(rows, settings, segs=SEGS, nxt=NEXT, run=RUN, **kw):
    return compute_remaining_life(pd.DataFrame(rows), run, PIPE, MOP, segs, nxt, settings, **kw)


def _by_id(res):
    return res.features.set_index("match_id")


@pytest.fixture
def fake_rl(monkeypatch):
    """Replace the solver: RL by governing rate (mpy) -> tau_80 = that RL, others inf; NaN RL -> all NaN;
    a dict value is returned as the criterion times."""
    table = {}

    def fake(d0, L0, g, gL, D, t, smys, mop, settings):
        rl = table[round(g * 1000, 9)]
        if isinstance(rl, dict):
            return rl
        if math.isnan(rl):
            return {c: NAN for c in rl_mod.CRITERIA}
        return {"tau_80": rl, "tau_burst": INF, "tau_safe": INF}

    monkeypatch.setattr(rl_mod, "time_to_criteria", fake)
    return table


# --- next assessment ---------------------------------------------------------------------------------


def test_add_months_clamps_to_month_end():
    assert add_months(date(2026, 3, 15), 68) == date(2031, 11, 15)
    assert add_months(date(2026, 8, 31), 68) == date(2032, 4, 30)
    assert add_months(date(2028, 2, 29), 60) == date(2033, 2, 28)
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    with pytest.raises(ValueError, match="whole number of months"):
        add_months(RUN, 6.5)


def test_next_assessment_sources(settings):
    rows = [_g("F1", 500.0), _g("F2", 1500.0), _g("F3", 2500.0), _g("F4", 3500.0), _g("F5", 5000.0)]
    f = _by_id(_run(rows, settings))
    want = {
        "F1": ("SEG-01", True, date(2029, 6, 30), "planned"),
        "F2": ("SEG-02", False, date(2036, 3, 15), "default_non_hca"),  # 10 calendar years
        "F3": ("SEG-03", True, date(2031, 3, 15), "default_hca"),  # 5 calendar years
        "F4": ("SEG-04", True, date(2031, 11, 15), "capped"),  # 2033-03-15 cut to run + 68 months
        "F5": (None, None, date(2031, 3, 15), "default_no_segment"),  # HCA default, flagged
    }
    for mid, (seg, hca, when, source) in want.items():
        row = f.loc[mid]
        assert row.segment_id == seg
        assert row.could_affect_hca is hca
        assert row.next_assessment_date == when
        assert row.next_assessment_source == source
        assert row.years_to_next_assessment == (when - RUN).days / YR
    assert f.loc["F3", "years_to_next_assessment"] == 1826 / YR
    assert f.loc["F4", "years_to_next_assessment"] == 2071 / YR
    assert f.loc["F5", "review_reasons"] == "no_segment" and f.loc["F5", "needs_review"]
    assert not f.loc[["F1", "F2", "F3", "F4"], "needs_review"].any()


def test_hca_default_capped_at_months_limit(settings):
    s = settings.model_copy(update={"remaining_life": settings.remaining_life.model_copy(
        update={"default_next_assessment_yr": settings.remaining_life.default_next_assessment_yr.model_copy(
            update={"hca": 6})})})
    f = _by_id(_run([_g("F3", 2500.0), _g("F5", 5000.0)], s))
    for mid, source in (("F3", "default_hca"), ("F5", "default_no_segment")):
        assert f.loc[mid, "next_assessment_date"] == date(2031, 11, 15)  # 72 months cut to 68
        assert f.loc[mid, "next_assessment_source"] == source


def test_planned_non_hca_not_capped_and_hca_at_limit_kept(settings):
    nxt = pd.DataFrame({"segment_id": ["SEG-02", "SEG-04"],
                        "planned_next_assessment_date": ["2038-01-01", "2031-11-15"]})
    f = _by_id(_run([_g("F2", 1500.0), _g("F4", 3500.0)], settings, nxt=nxt))
    assert (f.loc["F2", "next_assessment_date"], f.loc["F2", "next_assessment_source"]) == (date(2038, 1, 1), "planned")
    assert (f.loc["F4", "next_assessment_date"], f.loc["F4", "next_assessment_source"]) == (date(2031, 11, 15), "planned")


@pytest.mark.parametrize("when", ["2026-03-14", "2026-03-15"])
def test_planned_date_not_after_run_is_error(settings, when):
    nxt = pd.DataFrame({"segment_id": ["SEG-04"], "planned_next_assessment_date": [when]})
    with pytest.raises(ValueError, match="SEG-04"):
        _run([_g("F4", 3500.0)], settings, nxt=nxt)


def test_next_inspection_unknown_segment_is_error(settings):
    nxt = pd.DataFrame({"segment_id": ["SEG-01", "SEG-99"], "planned_next_assessment_date": ["2029-01-01"] * 2})
    with pytest.raises(ValueError, match="SEG-99"):
        _run([_g("F1", 500.0)], settings, nxt=nxt)


def test_bad_next_inspection_tables(settings):
    with pytest.raises(ValueError, match="SEG-01"):
        _run([_g("F1", 500.0)], settings, nxt=pd.DataFrame({"segment_id": ["SEG-01"],
                                                            "planned_next_assessment_date": ["next spring"]}))
    with pytest.raises(ValueError, match="repeats segment_id 'SEG-01'"):
        _run([_g("F1", 500.0)], settings, nxt=pd.DataFrame({"segment_id": ["SEG-01", "SEG-01"],
                                                            "planned_next_assessment_date": ["2029-01-01"] * 2}))
    for blank_value in (None, "", float("nan"), pd.NaT):
        blank = pd.DataFrame({"segment_id": ["SEG-01"], "planned_next_assessment_date": [blank_value]})
        f = _by_id(_run([_g("F1", 500.0)], settings, nxt=blank))
        assert f.loc["F1", "next_assessment_source"] == "default_hca", blank_value
        assert f.loc["F1", "next_assessment_date"] == date(2031, 3, 15)
    parsed = pd.DataFrame({"segment_id": ["SEG-01"], "planned_next_assessment_date": [pd.Timestamp("2029-06-30")]})
    assert _by_id(_run([_g("F1", 500.0)], settings, nxt=parsed)).loc["F1", "next_assessment_date"] == date(2029, 6, 30)


def test_could_affect_hca_must_be_boolean(settings):
    segs = SEGS.assign(could_affect_hca=[False, True, None, True])
    with pytest.raises(ValueError, match="SEG-03"):
        _run([_g("F1", 500.0)], settings, segs=segs)


# --- RL, repair-by, dig ------------------------------------------------------------------------------


def test_dig_flag_and_repair_by(settings, fake_rl):
    fake_rl.update({5.0: 8.0, 6.0: 12.0})
    f = _by_id(_run([_g("F3", 2500.0, gov=5.0), _g("G3", 2600.0, gov=6.0)], settings))
    row = f.loc["F3"]  # SEG-03 default_hca: 1826 days = 4.9993 yr
    assert (row.rl_yr, row.rl_governing, row.rl_factor, row.rl_factored_yr) == (8.0, "tau_80", 0.5, 4.0)
    assert row.dig_required_by_remaining_life
    assert row.repair_by_date == RUN + timedelta(days=4 * YR) == date(2030, 3, 15)
    assert f.loc["G3", "rl_factored_yr"] == 6.0 and not f.loc["G3", "dig_required_by_remaining_life"]
    assert f.loc["G3", "repair_by_date"] == RUN + timedelta(days=math.floor(6 * YR))


def test_dig_flag_boundary(settings, fake_rl):
    years = 1826 / YR  # SEG-03 default_hca
    fake_rl.update({5.0: 2 * years, 6.0: 2 * np.nextafter(years, 0.0), 7.0: NAN})
    f = _by_id(_run([_g("EQ", 2500.0, gov=5.0), _g("LT", 2600.0, gov=6.0), _g("NA", 2700.0, gov=7.0)], settings))
    assert f.loc["EQ", "rl_factored_yr"] == f.loc["EQ", "years_to_next_assessment"]
    assert not f.loc["EQ", "dig_required_by_remaining_life"]  # equal is not less: no dig
    assert f.loc["LT", "rl_factored_yr"] < f.loc["LT", "years_to_next_assessment"]
    assert f.loc["LT", "dig_required_by_remaining_life"]
    na = f.loc["NA"]
    assert math.isnan(na.rl_yr) and na.rl_governing is None and math.isnan(na.rl_factored_yr)
    assert math.isnan(na.rl_upper_bound_yr)  # no criterion finite: no bound, no dig claim
    assert na.repair_by_date is None and not na.dig_required_by_remaining_life
    assert na.review_reasons == "rl_not_evaluated" and na.needs_review


def test_repair_by_rounds_down_and_beyond_horizon(settings, fake_rl):
    fake_rl.update({5.0: 0.0, 6.0: 1.0 / YR * 2 * 0.999})  # 0.999 day factored
    res = _run([_g("Z", 500.0, gov=5.0), _g("D", 600.0, gov=6.0)], settings)
    f = _by_id(res)
    assert f.loc["Z", "repair_by_date"] == RUN and f.loc["Z", "dig_required_by_remaining_life"]
    assert f.loc["D", "repair_by_date"] == RUN  # never a day late
    fake_rl.update({7.0: INF})  # no criterion within the horizon
    h = _by_id(_run([_g("H", 2500.0, gov=7.0)], settings)).loc["H"]
    assert (h.rl_yr, h.rl_governing, h.rl_factored_yr) == (30.0, "beyond_horizon", 15.0)
    assert h.rl_upper_bound_yr == 30.0  # equals RL whenever RL is finite
    assert h.repair_by_date == RUN + timedelta(days=math.floor(15.0 * YR)) == RUN + timedelta(days=5478)
    assert not h.dig_required_by_remaining_life and h.review_reasons is None


def test_failing_today_with_nan_growth_digs_now(settings):
    # 60 % reported + 10 = 70 % of 0.25 in, L 4: P_F 1397, P_safe 1006 < MOP 1200 < P_F.
    mop = pd.DataFrame({"start_ft": [0.0], "end_ft": [10_000.0], "mop_psig": [1200.0]})
    rows = [_g("S", 2500.0, d_curr=60.0, length=4.0, gov=NAN, needs_review=True, reasons="no_governing")]
    f = compute_remaining_life(pd.DataFrame(rows), RUN, PIPE, mop, SEGS, NEXT, settings).features.iloc[0]
    assert f.safe_psig < 1200.0 < f.burst_psig
    assert math.isnan(f.tau_80_yr) and math.isnan(f.tau_burst_yr) and f.tau_safe_yr == 0.0
    assert (f.rl_yr, f.rl_governing, f.rl_factored_yr) == (0.0, "tau_safe", 0.0)
    assert f.dig_required_by_remaining_life and f.repair_by_date == RUN
    assert f.review_reasons == "no_governing" and f.needs_review


def test_nan_mop_upper_bound_forces_dig(settings):
    # 60 % reported + 10 = 70 % of 0.25 in = 0.175 in; g 25 mpy -> 0.80 t = 0.2 in at tau = 1 yr. With a NaN
    # MOP the pressure criteria are NaN, so RL is NaN, but RL <= tau_80 = 1 yr: 1 x 0.5 < 5 yr proves a dig.
    mop = MOP.assign(mop_psig=[NAN])
    rows = [_g("B", 2500.0, d_curr=60.0, gov=25.0)]
    f = compute_remaining_life(pd.DataFrame(rows), RUN, PIPE, mop, SEGS, NEXT, settings).features.iloc[0]
    assert f.tau_80_yr == pytest.approx(1.0, abs=1e-5) and math.isnan(f.tau_burst_yr)
    assert math.isnan(f.rl_yr) and f.rl_governing is None and math.isnan(f.rl_factored_yr)
    assert f.rl_upper_bound_yr == f.tau_80_yr == pytest.approx(1.0, abs=1e-5)
    assert f.dig_required_by_remaining_life
    assert f.repair_by_date is None  # a bound is not an RL
    assert f.review_reasons == "rl_not_evaluated" and f.needs_review


def test_upper_bound_rules(settings, fake_rl):
    years = 1826 / YR  # SEG-03 default_hca
    fake_rl.update({
        1.0: {"tau_80": 4.0, "tau_burst": NAN, "tau_safe": 9.0},  # bound 4 -> 2 < 5: dig
        2.0: {"tau_80": 2 * years, "tau_burst": NAN, "tau_safe": INF},  # bound x 0.5 == years: not less
        3.0: {"tau_80": INF, "tau_burst": NAN, "tau_safe": INF},  # nothing finite: no bound
    })
    rows = [_g("P", 2500.0, gov=1.0), _g("Q", 2600.0, gov=2.0), _g("R", 2700.0, gov=3.0)]
    f = _by_id(_run(rows, settings))
    assert f.loc["P", "rl_upper_bound_yr"] == 4.0 and f.loc["P", "dig_required_by_remaining_life"]
    assert f.loc["Q", "rl_upper_bound_yr"] == 2 * years and not f.loc["Q", "dig_required_by_remaining_life"]
    assert math.isnan(f.loc["R", "rl_upper_bound_yr"]) and not f.loc["R", "dig_required_by_remaining_life"]
    for mid in "PQR":
        assert math.isnan(f.loc[mid, "rl_yr"]) and f.loc[mid, "repair_by_date"] is None
        assert f.loc[mid, "review_reasons"] == "rl_not_evaluated" and f.loc[mid, "needs_review"]


def test_rl_upper_bound_function():
    s = load_settings()
    assert rl_mod.rl_upper_bound({"tau_80": 3.0, "tau_burst": NAN, "tau_safe": 1.5}, NAN) == 1.5
    assert rl_mod.rl_upper_bound({"tau_80": INF, "tau_burst": NAN, "tau_safe": INF}, NAN) is None
    assert rl_mod.rl_upper_bound({"tau_80": NAN, "tau_burst": NAN, "tau_safe": NAN}, NAN) is None
    times = {"tau_80": INF, "tau_burst": INF, "tau_safe": INF}
    assert rl_mod.rl_upper_bound(times, rl_mod.remaining_life(times, s)[0]) == s.remaining_life.horizon_yr


@pytest.mark.parametrize("column", ["od_in", "smys_psi"])
def test_nan_pipe_values(settings, column):
    pipe = PIPE.assign(**{column: [NAN]})
    rows = [_g("OK", 500.0, d_curr=30.0), _g("DEEP", 600.0, d_curr=75.0)]  # 40 % and 85 % assessed
    f = _by_id(compute_remaining_life(pd.DataFrame(rows), RUN, pipe, MOP, SEGS, NEXT, settings))
    assert math.isnan(f.loc["OK", "rl_yr"]) and f.loc["OK", "review_reasons"] == "rl_not_evaluated"
    assert not f.loc["OK", "dig_required_by_remaining_life"]
    deep = f.loc["DEEP"]  # tau_80 does not depend on OD or SMYS and is met today
    assert (deep.rl_yr, deep.rl_governing) == (0.0, "tau_80")
    assert deep.dig_required_by_remaining_life and deep.repair_by_date == RUN
    assert deep.review_reasons == "exceeds_80pct"


def test_strength_columns_at_tau0_reference_case(settings):
    # Spec 6.6 reference case: reported 40 % + 10 % tolerance = 50 % of 0.25 in = 0.125 in, L 4.0 in.
    f = _run([_g("R", 500.0, d_curr=40.0, length=4.0, gov=2.0)], settings).features.iloc[0]
    assert list(_run([_g("R", 500.0)], settings).features.columns) == FEATURE_COLUMNS
    assert (f.od_in, f.wt_in, f.smys_psi, f.mop_psig) == (12.75, 0.25, 52_000.0, 1000.0)
    assert f.d_reported_pct == 40.0 and f.d_assessed_in == pytest.approx(0.125, abs=1e-12)
    assert f.assessed_dt == pytest.approx(0.5) and not f.exceeds_80pct
    assert f.length_in == 4.0 and f.length_assessed_in == 4.0
    assert f.method == "modified_b31g"
    assert f.z == pytest.approx(5.0196, abs=1e-3) and f.folias_m == pytest.approx(2.0161, abs=1e-3)
    assert f.flow_stress_psi == 62_000.0
    assert f.failure_stress_psi == pytest.approx(45_173, abs=5) and f.burst_psig == pytest.approx(1771.5, abs=1)
    assert f.safe_psig == pytest.approx(0.72 * f.burst_psig)
    assert f.g_inyr == 0.002 and f.gl_inyr == 0.0
    times = rl_mod.time_to_criteria(f.d_assessed_in, 4.0, 0.002, 0.0, 12.75, 0.25, 52_000.0, 1000.0, settings)
    assert (f.tau_80_yr, f.tau_burst_yr, f.tau_safe_yr) == (times["tau_80"], times["tau_burst"], times["tau_safe"])
    assert (f.rl_yr, f.rl_governing) == rl_mod.remaining_life(times, settings)


def test_exceeds_80pct_uses_assessed_unclipped_dt(settings):
    rep = settings.model_copy(update={"strength": settings.strength.model_copy(update={"depth_basis": "reported"})})
    f = _by_id(_run([_g("A", 500.0, d_curr=79.9), _g("B", 600.0, d_curr=80.0), _g("C", 700.0, d_curr=120.0)], rep))
    assert list(f.exceeds_80pct) == [False, True, True]
    assert f.loc["C", "assessed_dt"] == pytest.approx(1.2)
    assert f.loc["B", "review_reasons"] == "exceeds_80pct"


def test_nan_reference_values_flag_not_crash(settings):
    mop = pd.DataFrame({"start_ft": [0.0, 1000.0], "end_ft": [1000.0, 10_000.0], "mop_psig": [NAN, 1000.0]})
    res = compute_remaining_life(pd.DataFrame([_g("M", 500.0), _g("L", 1500.0, length=NAN)]), RUN, PIPE, mop,
                                 SEGS, NEXT, settings)
    f = _by_id(res)
    for mid in ("M", "L"):
        assert math.isnan(f.loc[mid, "rl_yr"]) and f.loc[mid, "review_reasons"] == "rl_not_evaluated"
        assert f.loc[mid, "rl_upper_bound_yr"] == pytest.approx(20.0, abs=1e-5)  # tau_80: 0.1 in at 5 mpy
    # The bound decides the dig: M (SEG-01, planned ~3.3 yr out): 20 x 0.5 = 10 is not less, no dig;
    # L (SEG-02 non-HCA default, 3653 days = 10.0014 yr): 10 < 10.0014, dig.
    assert not f.loc["M", "dig_required_by_remaining_life"]
    assert f.loc["L", "dig_required_by_remaining_life"] and f.loc["L", "repair_by_date"] is None


@pytest.mark.parametrize("table,column", [("mop", "mop_psig"), ("pipe", "od_in"), ("pipe", "smys_psi")])
def test_nonpositive_reference_value_is_missing(settings, table, column):
    pipe, mop = PIPE, MOP
    if table == "mop":
        mop = MOP.assign(**{column: [0.0]})
    else:
        pipe = PIPE.assign(**{column: [0.0]})
    f = compute_remaining_life(pd.DataFrame([_g("Z", 500.0)]), RUN, pipe, mop, SEGS, NEXT, settings).features.iloc[0]
    assert f[column] == 0.0  # the input column keeps the value as read
    assert math.isnan(f.tau_burst_yr) and math.isnan(f.tau_safe_yr) and math.isfinite(f.tau_80_yr)
    assert math.isnan(f.rl_yr) and f.rl_governing is None
    assert f.review_reasons == "rl_not_evaluated" and f.needs_review
    if table == "pipe":  # no made-up strength from an OD or SMYS of 0
        assert math.isnan(f.failure_stress_psi) and math.isnan(f.burst_psig) and math.isnan(f.safe_psig)


def test_negative_length_is_missing(settings):
    f = _run([_g("N", 500.0, length=-2.0)], settings).features.iloc[0]
    assert f.length_in == -2.0 and math.isnan(f.z) and math.isnan(f.burst_psig)
    assert math.isnan(f.rl_yr) and f.review_reasons == "rl_not_evaluated"


def test_requires_columns(settings):
    with pytest.raises(ValueError, match="governing_mpy"):
        compute_remaining_life(pd.DataFrame([_g("A", 500.0)]).drop(columns="governing_mpy"), RUN, PIPE, MOP,
                               SEGS, NEXT, settings)
    with pytest.raises(ValueError, match="could_affect_hca"):
        _run([_g("A", 500.0)], settings, segs=SEGS.drop(columns="could_affect_hca"))


# --- review reasons ----------------------------------------------------------------------------------


def test_reasons_appended_to_m5(settings, fake_rl):
    # The real solver gives RL 0 to any feature at >= 80 % (tau_80 met today), so the fake solver
    # supplies a NaN RL alongside exceeds_80pct to check all three M6 reasons on one row.
    fake_rl.update({1.0: NAN, 2.0: NAN, 5.0: 8.0})
    rows = [
        # M5 reasons kept first; then rl_not_evaluated, exceeds_80pct, no_segment in that order.
        _g("A", 5000.0, d_curr=75.0, gov=1.0, needs_review=True, reasons="no_governing;match_review"),
        # an M6 name already present is not repeated
        _g("B", 500.0, d_curr=75.0, gov=2.0, needs_review=True, reasons="exceeds_80pct"),
        _g("C", 500.0),  # nothing
        _g("D", 600.0, needs_review=True, reasons=None),  # M5 needs_review carried
        _g("E", 700.0, needs_review=False, reasons="below_min_rate"),  # a reason means review
        _g("N", 800.0, needs_review=None), _g("Q", 900.0, needs_review=NAN),  # unknown M5 flag: review
    ]
    f = _by_id(_run(rows, settings))
    assert f.loc["A", "review_reasons"] == "no_governing;match_review;rl_not_evaluated;exceeds_80pct;no_segment"
    assert f.loc["B", "review_reasons"] == "exceeds_80pct;rl_not_evaluated"
    assert f.loc["C", "review_reasons"] is None and not f.loc["C", "needs_review"]
    assert f.loc["D", "review_reasons"] is None and f.loc["D", "needs_review"]
    assert f.loc["E", "review_reasons"] == "below_min_rate" and f.loc["E", "needs_review"]
    assert f.loc["A", "needs_review"] and f.loc["B", "needs_review"]
    assert f.loc["N", "needs_review"] and f.loc["Q", "needs_review"] and f.loc["N", "review_reasons"] is None


# --- intervals ---------------------------------------------------------------------------------------


def test_interval_min_uses_upper_bound(settings, fake_rl):
    # SEG-03 (HCA, default 5 yr): A has RL 8 (factored 4); B's RL is NaN but bounded by tau_80 2 yr
    # (factored 1), which must set the min; C has neither and is only counted.
    fake_rl.update({1.0: 8.0, 2.0: {"tau_80": 2.0, "tau_burst": NAN, "tau_safe": INF}, 3.0: NAN})
    rows = [_g("A", 2100.0, gov=1.0), _g("B", 2200.0, gov=2.0), _g("C", 2300.0, gov=3.0)]
    seg3 = _run(rows, settings).intervals.set_index("segment_id").loc["SEG-03"]
    assert (seg3.n_features, seg3.n_rl_not_evaluated) == (3, 2)
    assert (seg3.min_rl_factored_yr, seg3.recommended_interval_yr, seg3.governing_match_id) == (1.0, 1.0, "B")
    assert seg3.needs_review
    # only bounded NaN-RL rows: the bound still gives an interval, flagged
    fake_rl.update({4.0: {"tau_80": 30.0, "tau_burst": NAN, "tau_safe": INF}})
    only = _run([_g("D", 2100.0, gov=4.0), _g("E", 2200.0, gov=3.0)], settings).intervals
    only = only.set_index("segment_id").loc["SEG-03"]
    assert (only.min_rl_factored_yr, only.recommended_interval_yr, only.governing_match_id) == (15.0, 68 / 12, "D")
    assert only.n_rl_not_evaluated == 2 and only.needs_review


def test_intervals_min_and_cap(settings, fake_rl):
    segs = pd.DataFrame({
        "start_ft": [1000.0, 0.0, 2000.0, 3000.0], "end_ft": [2000.0, 1000.0, 3000.0, 4000.0],
        "segment_id": ["SEG-B", "SEG-A", "SEG-C", "SEG-D"], "could_affect_hca": [False, True, True, True],
    })
    fake_rl.update({1.0: 8.0, 2.0: 6.0, 3.0: NAN, 4.0: 6.0, 5.0: 30.0, 6.0: NAN, 7.0: 1.0})
    rows = [
        _g("A1", 100.0, gov=1.0), _g("A2", 200.0, gov=2.0), _g("A3", 300.0, gov=3.0), _g("A4", 400.0, gov=4.0),
        _g("B1", 1500.0, gov=5.0), _g("D1", 3500.0, gov=6.0), _g("X1", 5000.0, gov=7.0),  # X1 outside
    ]
    res = _run(rows, settings, segs=segs, nxt=pd.DataFrame(columns=["segment_id", "planned_next_assessment_date"]))
    iv = res.intervals
    assert list(iv.columns) == INTERVAL_COLUMNS
    assert list(iv.segment_id) == ["SEG-B", "SEG-A", "SEG-C", "SEG-D"]
    by = iv.set_index("segment_id")
    hca_max = 68 / 12
    a = by.loc["SEG-A"]  # min 3.0 (A2 and A4 tie: first in table order), NaN A3 excluded but counted
    assert (a.n_features, a.n_rl_not_evaluated, a.min_rl_factored_yr) == (4, 1, 3.0)
    assert (a.regulatory_max_yr, a.recommended_interval_yr, a.governing_match_id) == (hca_max, 3.0, "A2")
    assert a.needs_review  # mixed: min of the finite RLs, flagged for the unevaluated one
    b = by.loc["SEG-B"]  # beyond-horizon 30 x 0.5 = 15, capped at the non-HCA 10 yr
    assert (b.could_affect_hca, b.n_features, b.min_rl_factored_yr) == (False, 1, 15.0)
    assert (b.regulatory_max_yr, b.recommended_interval_yr, b.governing_match_id) == (10.0, 10.0, "B1")
    assert not b.needs_review
    c = by.loc["SEG-C"]  # empty: regulatory max
    assert (c.n_features, c.n_rl_not_evaluated, c.governing_match_id) == (0, 0, None)
    assert math.isnan(c.min_rl_factored_yr) and c.recommended_interval_yr == hca_max and not c.needs_review
    d = by.loc["SEG-D"]  # only a NaN-RL feature: no interval can be recommended, flagged
    assert (d.n_features, d.n_rl_not_evaluated, d.governing_match_id, d.regulatory_max_yr) == (1, 1, None, hca_max)
    assert math.isnan(d.min_rl_factored_yr) and math.isnan(d.recommended_interval_yr) and d.needs_review
    assert iv.needs_review.map(type).eq(bool).all() or iv.needs_review.dtype == bool


# --- summary -----------------------------------------------------------------------------------------


def test_summary_counts_and_json(settings, fake_rl):
    fake_rl.update({1.0: 8.0, 2.0: NAN, 3.0: 30.0})
    rows = [_g("A", 2500.0, gov=1.0), _g("B", 500.0, gov=2.0, d_curr=75.0), _g("C", 5000.0, gov=3.0)]
    res = _run(rows, settings, prev_run_id="RUN-A", curr_run_id="RUN-B")
    assert isinstance(res, RLResult)
    s = res.summary
    json.dumps(s, allow_nan=False)
    assert (s["prev_run_id"], s["curr_run_id"], s["method"], s["n_features"]) == ("RUN-A", "RUN-B", "modified_b31g", 3)
    assert s["curr_run_date"] == "2026-03-15"
    # fake: finite RL governed by tau_80; 30.0 is reported as tau_80 too (not all inf)
    assert s["counts_by_rl_governing"] == {"tau_80": 2, "tau_burst": 0, "tau_safe": 0, "beyond_horizon": 0}
    assert s["n_rl_not_evaluated"] == 1
    assert (s["n_exceeds_80pct"], s["n_dig_required"], s["n_needs_review"]) == (1, 1, 2)
    assert s["counts_by_review_reason"]["rl_not_evaluated"] == 1
    assert s["counts_by_review_reason"]["exceeds_80pct"] == 1
    assert s["counts_by_review_reason"]["no_segment"] == 1
    assert s["counts_by_review_reason"]["no_governing"] == 0
    assert s["min_rl_yr"] == 8.0 and s["min_rl_factored_yr"] == 4.0
    assert s["counts_by_next_assessment_source"] == {
        "planned": 1, "capped": 0, "default_hca": 1, "default_non_hca": 0, "default_no_segment": 1}
    assert s["n_segments_needs_review"] == 1  # SEG-01: B's RL could not be evaluated
    empty = compute_remaining_life(pd.DataFrame(columns=list(_g("x", 0.0))), RUN, PIPE, MOP, SEGS, NEXT,
                                   settings).summary
    json.dumps(empty, allow_nan=False)
    assert empty["min_rl_yr"] is None and empty["n_features"] == 0


# --- growth table round trip and synthetic acceptance ------------------------------------------------


@pytest.fixture(scope="module")
def synth_pipeline():
    s = load_settings()
    r = generate(SynthParams(line_miles=5))
    ref = r.reference
    wm = align_welds(r.prev_std, r.curr_std, s, ref["repairs_between_runs"]).weld_map
    mt = match_anomalies(r.prev_std, r.curr_std, wm, s, ref["pipe_properties"],
                         ref["repairs_between_runs"]).match_table
    seg = reference.segment_table(s, ref["hca_segments"], 0.0, float(r.curr_std.log_dist_ft.max()))
    dt = delta_t_years(r.prev_std, r.curr_std)
    g = growth.compute_growth(mt, dt, r.prev_std.run_date.iloc[0].year, r.curr_std.run_date.iloc[0].year,
                              ref["pipe_properties"], seg, s, prev_run_id="A", curr_run_id="B")
    return s, r, g


def test_growth_table_roundtrip(synth_pipeline, tmp_path):
    _, _, g = synth_pipeline
    path = tmp_path / "growth.csv"
    g.features.to_csv(path, index=False, lineterminator="\n")
    back = growth.load_growth_table(path)
    assert list(back.columns) == list(g.features.columns)
    for c in ("match_id", "match_type", "prev_feature_id", "curr_feature_id", "segment_id", "history_class",
              "confidence", "governing_method", "review_reasons"):
        assert all(v is None or isinstance(v, str) for v in back[c]), c
        assert [None if pd.isna(v) else str(v) for v in g.features[c]] == list(back[c]), c
    for c in ("needs_review", "implausible", "match_needs_review"):
        assert all(v is None or type(v) is bool for v in back[c]), c
        assert [None if v is None else bool(v) for v in g.features[c]] == list(back[c]), c
    assert back.needs_review.map(type).eq(bool).all() and back.match_needs_review.map(type).eq(bool).all()
    assert back.curr_feature_id.str.startswith("B-").all()
    for c in ("wt_in", "d_curr_pct", "governing_mpy", "length_in", "length_rate_ipy", "curr_log_dist_ft"):
        np.testing.assert_allclose(back[c].to_numpy(float), g.features[c].to_numpy(float), rtol=1e-12, equal_nan=True)


def test_growth_table_bad_flag_text(synth_pipeline, tmp_path):
    _, _, g = synth_pipeline
    path = tmp_path / "growth.csv"
    bad = g.features.astype({"needs_review": object})
    bad.loc[3, "needs_review"] = "yes"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="needs_review.*'yes'"):
        growth.load_growth_table(path)


def test_every_feature_has_rl_and_governing(synth_pipeline, tmp_path):
    s, r, g = synth_pipeline
    ref = r.reference
    curr_date = r.curr_std.run_date.iloc[0]
    res = compute_remaining_life(g.features, curr_date, ref["pipe_properties"], ref["mop_profile"],
                                 ref["hca_segments"], ref["next_inspection"], s, prev_run_id="A", curr_run_id="B")
    f = res.features
    assert len(f) == len(g.features) and list(f.match_id) == list(g.features.match_id)
    assert list(f.columns) == FEATURE_COLUMNS
    for row in f.itertuples():
        if math.isfinite(row.rl_yr):
            assert row.rl_governing in (*rl_mod.CRITERIA, rl_mod.BEYOND_HORIZON)
            assert row.repair_by_date is not None and row.rl_upper_bound_yr == row.rl_yr
        else:
            assert math.isnan(row.rl_yr) and row.rl_governing is None and row.repair_by_date is None
            assert "rl_not_evaluated" in row.review_reasons.split(";") and row.needs_review
            bounded = math.isfinite(row.rl_upper_bound_yr)
            assert row.dig_required_by_remaining_life == (
                bounded and row.rl_upper_bound_yr * row.rl_factor < row.years_to_next_assessment)
    # SEG-03 is deliberately missing from next_inspection: HCA default, 5 calendar years.
    seg3 = f[f.segment_id == "SEG-03"]
    assert len(seg3) and (seg3.next_assessment_source == "default_hca").all()
    assert (seg3.next_assessment_date == curr_date.replace(year=curr_date.year + 5)).all()
    assert list(res.intervals.segment_id) == list(ref["hca_segments"].segment_id)
    assert res.intervals.n_features.sum() == f.segment_id.notna().sum()
    json.dumps(res.summary, allow_nan=False)
    # same result from the CSV round trip
    path = tmp_path / "growth.csv"
    g.features.to_csv(path, index=False, lineterminator="\n")
    again = compute_remaining_life(growth.load_growth_table(path), curr_date, ref["pipe_properties"],
                                   ref["mop_profile"], ref["hca_segments"], ref["next_inspection"], s)
    pd.testing.assert_frame_equal(again.features, f)
