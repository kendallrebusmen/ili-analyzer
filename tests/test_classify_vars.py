"""Feature variables for repair classification (spec 6.8; M7 Task 3)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ili import growth, reference
from ili.align_welds import align_welds
from ili.classify import (VARIABLE_COLUMNS, assessed_depth_pct, dent_depth_pct_od, dent_metal_loss, feature_variables,
                          general_corrosion, near_foreign_crossing, near_girth_weld, seam_interaction,
                          top_of_pipe, variable_dicts, widespread_circumferential)
from ili.config import load_repair_criteria, load_settings
from ili.match import TABLE_COLUMNS, delta_t_years, match_anomalies
from ili.remaining_life import compute_remaining_life, load_remaining_life_table
from ili.rules import compile_condition
from ili.vocabulary import VARIABLES

NAN = float("nan")
COLUMNS = ["match_id", "curr_feature_id", "segment_id", *VARIABLES, "repair_by_date", "upstream_needs_review",
           "upstream_review_reasons"]
PIPE = pd.DataFrame({"start_ft": [0.0, 5000.0], "end_ft": [5000.0, 10_000.0], "od_in": [12.75, 10.75],
                     "smys_psi": [52_000.0, 52_000.0]})
# [0, 4000) HCA, [4000, 8000) non-HCA, [8000, 10000) outside every segment
SEGS = pd.DataFrame({"start_ft": [0.0, 4000.0], "end_ft": [4000.0, 8000.0], "segment_id": ["SEG-01", "SEG-02"],
                     "could_affect_hca": [True, False]})
WELDS = np.arange(0.0, 10_001.0, 40.0)  # a girth weld every 40 ft
CURR_STD = pd.DataFrame({"feature_type": ["girth_weld"] * len(WELDS) + ["valve"],
                         "log_dist_ft": [*WELDS, 1020.0]})  # a valve on top of M1 must not count as a weld
CROSSINGS = pd.DataFrame({"log_dist_ft": [2000.0], "description": ["foreign line"]})


@pytest.fixture
def settings():
    s = load_settings()
    assert s.tool.depth_tolerance_pct_wt == 10 and s.strength.depth_basis == "reported_plus_tolerance"
    return s


def with_cls(settings, **kw):
    return settings.model_copy(update={"classification": settings.classification.model_copy(update=kw)})


def mrow(mid, ftype="metal_loss", pos=1020.0, mtype="new", needs_review=False, **curr):
    """One match-table row (every TABLE_COLUMNS field), current side from ``curr`` (names without ``curr_``)."""
    r = {c: None for c in TABLE_COLUMNS}
    r.update(match_id=mid, match_type=mtype, needs_review=needs_review, n_prev=0, n_curr=1,
             curr_feature_id=f"B-{mid}")
    side = dict(feature_type=ftype, log_dist_ft=pos, wt_nom_in=0.25, depth_pct=NAN, dent_depth_in=NAN,
                length_in=2.0, width_in=1.0, clock_min=360, seam_interaction=False, dent_metal_loss=None)
    side.update(curr)
    r.update({f"curr_{k}": v for k, v in side.items()})
    return r


def missing_row(mid, pos=1020.0):
    r = {c: None for c in TABLE_COLUMNS}
    r.update(match_id=mid, match_type="missing", needs_review=False, prev_feature_id=f"A-{mid}",
             prev_feature_type="metal_loss", prev_log_dist_ft=pos, n_prev=1, n_curr=0)
    return r


def table(rows):
    df = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    df["curr_clock_min"] = df.curr_clock_min.astype("Int64")
    return df


def rlrow(mid, assessed_dt=0.40, wt=0.25, burst=1800.0, safe=1296.0, mop=1000.0, dig=False,
          repair_by=date(2040, 1, 1), needs_review=False, reasons=None):
    return {"match_id": mid, "wt_in": wt, "mop_psig": mop, "assessed_dt": assessed_dt, "burst_psig": burst,
            "safe_psig": safe, "dig_required_by_remaining_life": dig, "repair_by_date": repair_by,
            "needs_review": needs_review, "review_reasons": reasons}


RL_EMPTY = pd.DataFrame([rlrow("X")]).iloc[:0]


def run(rows, rl_rows=(), settings=None, crossings=CROSSINGS, segs=SEGS, curr_std=CURR_STD):
    rl = pd.DataFrame(list(rl_rows)) if rl_rows else RL_EMPTY
    return feature_variables(table(rows), rl, curr_std, PIPE, segs, crossings, settings)


def one(df, mid):
    return variable_dicts(df[df.match_id == mid])[0]


def full(df, mid):
    row = df[df.match_id == mid].iloc[0]
    return {c: (None if not isinstance(row[c], str) and pd.isna(row[c]) else row[c]) for c in df.columns}


# --- whole-table shape on the synthetic line ----------------------------------------------------------


@pytest.fixture(scope="module")
def synth_vars(tmp_path_factory):
    from ili.synth import SynthParams, generate
    s = load_settings()
    r = generate(SynthParams(line_miles=5))
    ref = r.reference
    wm = align_welds(r.prev_std, r.curr_std, s, ref["repairs_between_runs"]).weld_map
    mt = match_anomalies(r.prev_std, r.curr_std, wm, s, ref["pipe_properties"],
                         ref["repairs_between_runs"]).match_table
    seg = reference.segment_table(s, ref["hca_segments"], 0.0, float(r.curr_std.log_dist_ft.max()))
    dt = delta_t_years(r.prev_std, r.curr_std)
    g = growth.compute_growth(mt, dt, r.prev_std.run_date.iloc[0].year, r.curr_std.run_date.iloc[0].year,
                              ref["pipe_properties"], seg, s)
    rl = compute_remaining_life(g.features, r.curr_std.run_date.iloc[0], ref["pipe_properties"],
                                ref["mop_profile"], ref["hca_segments"], ref["next_inspection"], s).features
    # The synthetic generator makes no gouges: add one current-run gouge (a copy of a new metal-loss row).
    src = mt[(mt.match_type == "new") & (mt.curr_feature_type == "metal_loss")].iloc[[0]]
    gouge = src.assign(match_id="M99999", curr_feature_id="B-GOUGE", curr_feature_type="gouge")
    mt = pd.concat([mt, gouge], ignore_index=True)
    v = feature_variables(mt, rl, r.curr_std, ref["pipe_properties"], ref["hca_segments"],
                          ref["foreign_crossings"], s)
    path = tmp_path_factory.mktemp("rl") / "remaining_life.csv"
    rl.to_csv(path, index=False, lineterminator="\n")
    return s, r, mt, rl, v, path


def test_one_row_per_current_anomaly(synth_vars):
    s, r, mt, rl, v, _ = synth_vars
    assert list(v.columns) == COLUMNS == ["match_id", "curr_feature_id", "segment_id", *VARIABLE_COLUMNS,
                                          "repair_by_date", "upstream_needs_review", "upstream_review_reasons"]
    want = mt[mt.match_type.isin(["matched_1to1", "matched_cluster", "new"])
              & mt.curr_feature_type.isin(["metal_loss", "dent", "gouge"])]
    assert list(v.match_id) == list(want.match_id)  # table order
    assert list(v.curr_feature_id) == list(want.curr_feature_id)
    assert not set(v.match_id) & set(mt.match_id[mt.match_type == "missing"])
    kinds = v.feature_type.value_counts()
    assert kinds.get("dent", 0) > 0 and kinds.get("gouge", 0) == 1 and kinds.get("metal_loss", 0) > 0
    g = full(v, "M99999")
    assert g["feature_type"] == "gouge" and g["p_fail"] is None and g["upstream_review_reasons"] is None
    assert g["depth_pct_assessed"] == pytest.approx(min(mt.curr_depth_pct.iloc[-1] + 10.0, 100.0))
    assert set(v.feature_type[v.match_id.isin(mt.match_id[mt.match_type == "matched_cluster"])]) <= {"metal_loss"}
    # metal loss carries M6's numbers exactly; dents and gouges carry none
    rl_by = rl.set_index("match_id")
    ml = v[v.feature_type == "metal_loss"].set_index("match_id")
    assert set(ml.index) == set(rl_by.index)
    for c, src in (("p_fail", "burst_psig"), ("p_safe", "safe_psig"), ("mop_local", "mop_psig"), ("wt_in", "wt_in")):
        np.testing.assert_array_equal(ml[c].to_numpy(float), rl_by.loc[ml.index, src].to_numpy(float), err_msg=c)
    np.testing.assert_allclose(ml.depth_pct_assessed.to_numpy(float),
                               rl_by.loc[ml.index, "assessed_dt"].to_numpy(float) * 100, rtol=1e-12)
    assert list(ml.dig_required_by_remaining_life) == list(rl_by.loc[ml.index, "dig_required_by_remaining_life"])
    other = v[v.feature_type != "metal_loss"]
    for c in ("p_fail", "p_safe", "mop_local", "dig_required_by_remaining_life", "repair_by_date"):
        assert other[c].isna().all(), c
    # every variable record is evaluable by every rule condition without error
    crit = load_repair_criteria()
    conds = [compile_condition(rule.condition, set(VARIABLES)) for rule in crit.rules]
    for d in variable_dicts(v):
        for c in conds:
            assert c.evaluate(d) in (True, False, None)
    # null PLACEHOLDER settings surface as unknown, never as a guessed False
    for c in ("near_girth_weld", "near_foreign_crossing", "general_corrosion", "widespread_circumferential"):
        assert v[c].isna().all(), c
    assert (v.manual_immediate == False).all() and (v.field_crack_confirmed == False).all()  # noqa: E712


def test_remaining_life_csv_round_trip(synth_vars):
    s, r, mt, rl, v, path = synth_vars
    back = load_remaining_life_table(path)
    assert list(back.columns) == list(rl.columns)
    for c in ("needs_review", "dig_required_by_remaining_life", "exceeds_80pct"):
        assert all(type(x) is bool for x in back[c]), c
    assert all(x is None or isinstance(x, date) for x in back.repair_by_date)
    ref = r.reference
    again = feature_variables(mt, back, r.curr_std, ref["pipe_properties"], ref["hca_segments"],
                              ref["foreign_crossings"], s)
    pd.testing.assert_frame_equal(again, v)


# --- hand-built rows ------------------------------------------------------------------------------------


def test_dent_variables(settings):
    rows = [
        mrow("D1", "dent", clock_min=240, dent_depth_in=0.765, dent_metal_loss=False),  # 6 % of 12.75
        mrow("D2", "dent", clock_min=241, dent_depth_in=0.255, dent_metal_loss=True),
        mrow("D3", "dent", clock_min=480, dent_depth_in=0.3, dent_metal_loss=None, depth_pct=12.0),
        mrow("D4", "dent", clock_min=479, dent_depth_in=0.3, dent_metal_loss=None),
        mrow("D5", "dent", clock_min=None, dent_depth_in=NAN, dent_metal_loss=False),
        mrow("D6", "dent", pos=6020.0, clock_min=0, dent_depth_in=0.43, dent_metal_loss=False, depth_pct=0.0),
        mrow("D7", "dent", clock_min=719, dent_depth_in=0.3, dent_metal_loss=None, depth_pct=0.0),
    ]
    v = run(rows, settings=settings)
    d = {m: one(v, m) for m in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")}
    assert d["D1"]["dent_depth_pct_od"] == pytest.approx(6.0, abs=1e-12)
    assert d["D2"]["dent_depth_pct_od"] == pytest.approx(2.0, abs=1e-12)
    assert d["D6"]["od_in"] == 10.75 and d["D6"]["dent_depth_pct_od"] == pytest.approx(0.43 / 10.75 * 100)
    assert d["D5"]["dent_depth_pct_od"] is None and d["D5"]["dent_depth_in"] is None
    # top of pipe: clock >= 480 or <= 240 (spec 6.8)
    tops = {m: d[m]["top_of_pipe"] for m in d}
    assert tops == {"D1": True, "D2": False, "D3": True, "D4": False, "D5": None, "D6": True, "D7": True}
    assert d["D5"]["clock_min"] is None and d["D1"]["clock_min"] == 240
    # dent_metal_loss from either source; None only if both are unknown
    dml = {m: d[m]["dent_metal_loss"] for m in d}
    assert dml == {"D1": False, "D2": True, "D3": True, "D4": None, "D5": False, "D6": False, "D7": False}
    # a dent has no pressures; its depth_pct_assessed is the assessed metal-loss depth on it, if any
    for m in d:
        assert d[m]["p_fail"] is None and d[m]["p_safe"] is None and d[m]["mop_local"] is None
        assert d[m]["dig_required_by_remaining_life"] is None and d[m]["feature_type"] == "dent"
    assert d["D3"]["depth_pct_reported"] == 12.0 and d["D3"]["depth_pct_assessed"] == pytest.approx(22.0)
    assert d["D1"]["depth_pct_assessed"] is None
    assert not v.upstream_review_reasons.fillna("").str.contains("rl_missing").any()


def test_dent_depth_pct_od_is_rounded_as_the_dent_bins(settings):
    from ili.report_tables import dent_bins
    assert 0.645 / 10.75 * 100 > 6.0  # the bare division lands one ulp above the 6 % edge
    v = run([mrow("D6", "dent", pos=6020.0, clock_min=0, dent_depth_in=0.645, dent_metal_loss=False)],
            settings=settings)
    assert one(v, "D6")["dent_depth_pct_od"] == 6.0  # not "deeper than 6 %"
    assert dent_depth_pct_od([0.645, NAN, 0.3], [10.75, 10.75, 0.0]).tolist()[0] == 6.0
    assert np.isnan(dent_depth_pct_od([0.645, NAN, 0.3], [10.75, 10.75, 0.0])[1:]).all()
    std = pd.DataFrame({"run_id": ["R"], "feature_type": ["dent"], "dent_depth_in": [0.645], "log_dist_ft": [6020.0],
                        "length_in": [2.0], "clock_min": [0], "seam_interaction": [False]})
    bins = dent_bins(std, np.array([10.75]), settings).set_index("bin")["all"]
    assert bins["2-6"] == 1 and bins[">6"] == 0  # the same feature is in the 2-6 bin


@pytest.mark.parametrize("clock, top", [(0, True), (240, True), (241, False), (479, False), (480, True),
                                        (719, True), (360, False), (None, None), (pd.NA, None)])
def test_top_of_pipe_edges(clock, top):
    assert list(top_of_pipe(pd.array([clock], dtype="Int64"))) == [top]


@pytest.mark.parametrize("flag, depth, want", [
    (True, NAN, True), (True, 0.0, True), (True, 15.0, True),
    (False, 15.0, True), (None, 15.0, True),
    (False, 0.0, False), (False, NAN, False), (None, 0.0, False),
    (None, NAN, None),
])
def test_dent_metal_loss_truth_table(flag, depth, want):
    assert list(dent_metal_loss(pd.Series([flag], dtype=object), np.array([depth]))) == [want]


def test_gouge_and_metal_loss_depths(settings):
    rows = [
        mrow("G1", "gouge", depth_pct=20.0),
        mrow("G2", "gouge", depth_pct=95.0),  # capped at 100 % of wall
        mrow("G3", "gouge", depth_pct=NAN),
        mrow("M1", depth_pct=30.0),
        mrow("M2", depth_pct=30.0, wt_nom_in=0.30, dent_depth_in=0.3),  # stray dent depth: ignored
    ]
    v = run(rows, [rlrow("M1", assessed_dt=0.4123, wt=0.281)], settings=settings)
    g1, g2, g3, m1, m2 = (one(v, m) for m in ("G1", "G2", "G3", "M1", "M2"))
    assert g1["depth_pct_reported"] == 20.0 and g1["depth_pct_assessed"] == pytest.approx(30.0)
    assert g2["depth_pct_assessed"] == pytest.approx(100.0) and g3["depth_pct_assessed"] is None
    assert g1["p_fail"] is None and g1["feature_type"] == "gouge" and g1["wt_in"] == 0.25
    assert m1["depth_pct_assessed"] == pytest.approx(41.23) and m1["wt_in"] == 0.281  # from M6
    assert m2["depth_pct_assessed"] == pytest.approx(40.0) and m2["wt_in"] == 0.30  # M6 row missing
    for d in (g1, m1, m2):  # dent variables belong to dents only
        assert d["dent_metal_loss"] is None and d["dent_depth_in"] is None and d["dent_depth_pct_od"] is None
    assert assessed_depth_pct(np.array([20.0, -5.0, NAN]), settings).tolist()[:2] == pytest.approx([30.0, 10.0])
    reported = settings.model_copy(update={"strength": settings.strength.model_copy(update={"depth_basis": "reported"})})
    assert one(run(rows, settings=reported), "G1")["depth_pct_assessed"] == pytest.approx(20.0)


def test_metal_loss_missing_from_rl_gets_unknown_pressures(settings):
    rows = [mrow("M1", depth_pct=30.0), mrow("M2", depth_pct=35.0, needs_review=False), mrow("D1", "dent")]
    # an M6 row under a dent's match_id (not expected) is ignored: M6 is metal loss only
    v = run(rows, [rlrow("M1", dig=True, repair_by=date(2031, 5, 6)), rlrow("D1", needs_review=True, reasons="x")],
            settings=settings)
    m1, m2 = full(v, "M1"), full(v, "M2")
    assert (m1["p_fail"], m1["p_safe"], m1["mop_local"]) == (1800.0, 1296.0, 1000.0)
    assert m1["dig_required_by_remaining_life"] is True and m1["repair_by_date"] == date(2031, 5, 6)
    assert m1["upstream_needs_review"] is False and m1["upstream_review_reasons"] is None
    for c in ("p_fail", "p_safe", "mop_local", "dig_required_by_remaining_life", "repair_by_date"):
        assert m2[c] is None, c
    assert m2["upstream_needs_review"] is True and m2["upstream_review_reasons"] == "rl_missing"
    assert m2["depth_pct_assessed"] == pytest.approx(45.0)  # still has its depth
    d1 = full(v, "D1")
    assert d1["upstream_needs_review"] is False and d1["upstream_review_reasons"] is None  # dents need no RL
    assert d1["p_fail"] is None and d1["dig_required_by_remaining_life"] is None and d1["repair_by_date"] is None
    # IMM-2 / D180-4 are then not evaluable, never a silent pass
    imm2 = compile_condition("p_fail < mop_local", set(VARIABLES))
    assert imm2.evaluate(one(v, "M2")) is None and imm2.evaluate(one(v, "M1")) is False


def test_upstream_review_union(settings):
    rows = [mrow("M1", needs_review=True), mrow("M2", needs_review=None), mrow("M3", pos=9020.0),
            mrow("M4", needs_review=True, pos=9100.0)]
    rl = [rlrow("M1", needs_review=True, reasons="rate_below_min;match_review"),
          rlrow("M2", needs_review=NAN, reasons=None),
          rlrow("M3", needs_review=True, reasons="no_segment")]
    v = run(rows, rl, settings=settings)
    got = {m: (full(v, m)["upstream_needs_review"], full(v, m)["upstream_review_reasons"]) for m in v.match_id}
    assert got == {
        "M1": (True, "match_review;rate_below_min"),
        "M2": (True, "match_review"),
        "M3": (True, "no_segment"),
        "M4": (True, "match_review;no_segment;rl_missing"),
    }


def test_proximity_unknown_when_setting_null_and_exact_when_set(settings):
    # welds every 40 ft: a feature at 1020 ft is 20 ft (240 in) from both neighbours; length 2 in -> edge 239 in
    rows = [
        mrow("M1", pos=1020.0, length_in=2.0),
        mrow("M2", pos=1000.5, length_in=4.0),  # 6 in from the weld at 1000 ft; edge 4 in
        mrow("M3", pos=1000.1, length_in=4.0),  # centre 1.2 in from the weld: edge clamps to 0
        mrow("M4", pos=1020.0, length_in=NAN),
        mrow("C1", pos=2000.0 + 50.0),
        mrow("C2", pos=2000.0 - 50.01),
    ]
    nulls = run(rows, settings=settings)
    for c in ("near_girth_weld", "near_foreign_crossing"):
        assert nulls[c].isna().all(), c
    s = with_cls(settings, girth_weld_proximity_in=4.0, crossing_proximity_ft=50.0)
    v = run(rows, settings=s)
    gw = {m: one(v, m)["near_girth_weld"] for m in v.match_id}
    assert gw == {"M1": False, "M2": True, "M3": True, "M4": None, "C1": False, "C2": False}
    s239 = with_cls(settings, girth_weld_proximity_in=239.0)
    assert one(run(rows, settings=s239), "M1")["near_girth_weld"] is True
    s_just = with_cls(settings, girth_weld_proximity_in=238.999)
    assert one(run(rows, settings=s_just), "M1")["near_girth_weld"] is False
    fc = {m: one(v, m)["near_foreign_crossing"] for m in v.match_id}
    assert fc["C1"] is True and fc["C2"] is False and fc["M1"] is False
    # no crossings file: unknown (set or not); a crossings table with no rows: known False when set
    assert run(rows, settings=s, crossings=None).near_foreign_crossing.isna().all()
    assert run(rows, settings=settings, crossings=None).near_foreign_crossing.isna().all()
    no_rows = pd.DataFrame({"log_dist_ft": pd.Series([], dtype=float)})
    assert run(rows, settings=s, crossings=no_rows).near_foreign_crossing.map(lambda x: x is False).all()
    assert run(rows, settings=settings, crossings=no_rows).near_foreign_crossing.isna().all()
    # the helpers directly
    assert list(near_girth_weld(np.array([10.0]), np.array([2.0]), np.array([]), 6.0)) == [None]
    assert list(near_foreign_crossing(np.array([10.0]), np.array([]), 5.0)) == [False]
    assert list(near_foreign_crossing(np.array([10.0]), None, 5.0)) == [None]
    assert list(near_foreign_crossing(np.array([10.0]), np.array([15.0, 100.0]), 5.0)) == [True]


@pytest.mark.parametrize("flag, clock, seam_min, tol, want", [
    (True, 360, None, None, True),        # vendor flag decides
    (False, 360, None, None, None),       # seam position unconfigured: overlap unknown
    (None, 360, None, None, None),
    (False, 30, 0, 30, True),             # overlap at the edge of the window
    (False, 31, 0, 30, False),
    (False, 700, 0, 30, True),            # across 12 o'clock on the 720-min circle
    (False, 689, 0, 30, False),
    (None, 30, 0, 30, True),              # overlap decides an unknown vendor flag
    (None, 31, 0, 30, None),
    (False, None, 0, 30, None),           # clock missing: overlap unknown
    (True, None, 0, 30, True),
    (False, 360, 360, None, None),        # only half the seam settings: unknown
])
def test_seam_interaction_truth_table(flag, clock, seam_min, tol, want):
    got = seam_interaction(pd.Series([flag], dtype=object), pd.array([clock], dtype="Int64"), seam_min, tol)
    assert list(got) == [want]


def test_seam_interaction_sources(settings):
    rows = [mrow("M1", seam_interaction=True), mrow("M2", seam_interaction=False, clock_min=95),
            mrow("M3", seam_interaction=False, clock_min=400), mrow("M4", seam_interaction=None, clock_min=400),
            mrow("D1", "dent", seam_interaction=False, clock_min=90)]
    v = run(rows, settings=settings)
    assert [one(v, m)["seam_interaction"] for m in v.match_id] == [True, None, None, None, None]
    s = with_cls(settings, seam_clock_min=90.0, seam_clock_tol_min=5.0)
    v = run(rows, settings=s)
    assert [one(v, m)["seam_interaction"] for m in v.match_id] == [True, True, False, None, True]
    d180_8 = compile_condition("seam_interaction and feature_type == 'metal_loss'", set(VARIABLES))
    assert d180_8.evaluate(one(v, "M3")) is False and d180_8.evaluate(one(v, "M2")) is True


def test_size_flags(settings):
    rows = [mrow("M1", length_in=6.0, width_in=6.0), mrow("M2", length_in=6.0, width_in=5.99),
            mrow("M3", length_in=NAN, width_in=5.0), mrow("M4", length_in=NAN, width_in=7.0),
            mrow("M5", length_in=7.0, width_in=NAN)]
    v = run(rows, settings=settings)
    assert v.general_corrosion.isna().all() and v.widespread_circumferential.isna().all()
    s = with_cls(settings, general_corrosion_min_dim_in=6.0, widespread_circ_min_width_frac=0.15)
    v = run(rows, settings=s)
    gc = [one(v, m)["general_corrosion"] for m in v.match_id]
    assert gc == [True, False, False, None, None]  # a known short side decides; otherwise unknown
    circ = math.pi * 12.75  # 40.055 in
    wc = widespread_circumferential(np.array([0.15 * circ, 0.15 * circ - 1e-9, NAN]), np.array([12.75] * 3), 0.15)
    assert list(wc) == [True, False, None]
    assert list(widespread_circumferential(np.array([7.0]), np.array([NAN]), 0.15)) == [None]
    assert list(general_corrosion(np.array([1.0]), np.array([1.0]), None)) == [None]


def test_outside_segments_hca_assumed_and_flagged(settings):
    rows = [mrow("M1", pos=1020.0), mrow("M2", pos=4020.0), mrow("M3", pos=9020.0), mrow("D1", "dent", pos=9500.0)]
    v = run(rows, [rlrow("M1"), rlrow("M2"), rlrow("M3", needs_review=True, reasons="no_segment")], settings=settings)
    got = {m: (full(v, m)["segment_id"], full(v, m)["could_affect_hca"], full(v, m)["upstream_needs_review"],
               full(v, m)["upstream_review_reasons"]) for m in v.match_id}
    assert got == {"M1": ("SEG-01", True, False, None), "M2": ("SEG-02", False, False, None),
                   "M3": (None, True, True, "no_segment"), "D1": (None, True, True, "no_segment")}


def test_od_lookup_outside_pipe_table_is_error(settings):
    with pytest.raises(ValueError, match="pipe properties table"):
        run([mrow("M1", pos=10_500.0)], settings=settings)


def test_rows_and_order(settings):
    rows = [mrow("M1"), missing_row("M2"), mrow("M3", "valve"), mrow("M4", "gouge", mtype="matched_1to1"),
            mrow("M5", mtype="matched_cluster", depth_pct=40.0), mrow("M6", "dent", mtype="new")]
    v = run(rows, settings=settings)
    assert list(v.match_id) == ["M1", "M4", "M5", "M6"] and list(v.columns) == COLUMNS
    assert list(v.feature_type) == ["metal_loss", "gouge", "metal_loss", "dent"]
    empty = run([missing_row("M2")], settings=settings)
    assert list(empty.columns) == COLUMNS and empty.empty and variable_dicts(empty) == []


def test_duplicate_rl_match_id_is_error(settings):
    with pytest.raises(ValueError, match="M1"):
        run([mrow("M1")], [rlrow("M1"), rlrow("M1")], settings=settings)


def test_variable_dicts_plain_types(settings):
    s = with_cls(settings, girth_weld_proximity_in=6.0)
    rows = [mrow("M1", depth_pct=30.0, clock_min=None), mrow("D1", "dent", dent_depth_in=0.5, clock_min=100)]
    v = run(rows, [rlrow("M1", burst=np.float64(1700.0), dig=np.bool_(True))], settings=s)
    ds = variable_dicts(v)
    assert [list(d) for d in ds] == [list(VARIABLES)] * 2
    allowed = (bool, float, int, str, type(None))
    for d in ds:
        for k, x in d.items():
            assert type(x) in allowed, (k, type(x))
            assert not (isinstance(x, float) and math.isnan(x)), k
    m1, d1 = ds
    assert m1["p_fail"] == 1700.0 and type(m1["p_fail"]) is float
    assert m1["dig_required_by_remaining_life"] is True and m1["clock_min"] is None and m1["top_of_pipe"] is None
    assert d1["clock_min"] == 100 and d1["dent_depth_in"] == 0.5 and d1["p_fail"] is None
    assert m1["manual_immediate"] is False and m1["field_crack_confirmed"] is False
    # works with the evaluator, which must not be handed a pandas row
    assert compile_condition("top_of_pipe and dent_depth_pct_od > 3", set(VARIABLES)).evaluate(d1) is True


def test_bad_flag_text_is_error(settings):
    with pytest.raises(ValueError, match="seam_interaction.*'yes'"):
        run([mrow("M1", seam_interaction="yes")], settings=settings)
    assert list(seam_interaction(pd.Series(["True", "false", ""], dtype=object), pd.array([None] * 3, dtype="Int64"),
                                 None, None)) == [True, None, None]


def test_blank_reference_positions_are_errors(settings):
    s = with_cls(settings, crossing_proximity_ft=50.0, girth_weld_proximity_in=6.0)
    blank_crossing = pd.DataFrame({"log_dist_ft": [2000.0, NAN], "description": ["a", "b"]})
    for cfg in (s, settings):  # an incomplete reference table is an error even while the setting is null
        with pytest.raises(ValueError, match="foreign crossings table: row 1 has a blank log_dist_ft"):
            run([mrow("M1")], settings=cfg, crossings=blank_crossing)
        blank_weld = pd.DataFrame({"feature_type": ["girth_weld"] * 3, "log_dist_ft": [1000.0, NAN, 1040.0]})
        with pytest.raises(ValueError, match="current-run girth welds: row 1 has a blank log_dist_ft"):
            run([mrow("M1")], settings=cfg, curr_std=blank_weld)
    with pytest.raises(ValueError, match="foreign crossings table"):
        near_foreign_crossing(np.array([10.0]), np.array([NAN]), 5.0)
    with pytest.raises(ValueError, match="girth welds"):
        near_girth_weld(np.array([10.0]), np.array([2.0]), np.array([NAN]), None)


def test_near_girth_weld_outside_weld_span(settings):
    welds = pd.DataFrame({"feature_type": ["girth_weld"] * 3, "log_dist_ft": [100.0, 140.0, 1000.0]})
    rows = [
        mrow("U1", pos=99.8, length_in=2.0),    # before the first weld, edge 1.4 in: True
        mrow("U2", pos=90.0, length_in=2.0),    # before the first weld, 119 in: unknown (a weld may lie before)
        mrow("U3", pos=100.0, length_in=2.0),   # on the first weld: inside the span
        mrow("U4", pos=120.0, length_in=2.0),   # inside, 239 in: False
        mrow("U5", pos=1000.3, length_in=2.0),  # after the last weld, edge 2.6 in: True
        mrow("U6", pos=1100.0, length_in=2.0),  # after the last weld, far: unknown
        mrow("U7", pos=1000.3, length_in=NAN),  # length unknown
    ]
    v = run(rows, settings=with_cls(settings, girth_weld_proximity_in=6.0), curr_std=welds)
    got = {m: one(v, m)["near_girth_weld"] for m in v.match_id}
    assert got == {"U1": True, "U2": None, "U3": True, "U4": False, "U5": True, "U6": None, "U7": None}


def test_top_of_pipe_uses_the_reported_rounded_clock(settings):
    t = table([mrow("R1", "dent"), mrow("R2", "dent"), mrow("R3", "dent")])
    t["curr_clock_min"] = pd.Series([240.4, 479.6, 240.6])
    v = feature_variables(t, RL_EMPTY, CURR_STD, PIPE, SEGS, CROSSINGS, settings)
    got = [(d["clock_min"], d["top_of_pipe"]) for d in variable_dicts(v)]
    assert got == [(240, True), (480, True), (241, False)]
    s = with_cls(settings, seam_clock_min=0.0, seam_clock_tol_min=240.0)
    assert [d["seam_interaction"] for d in variable_dicts(
        feature_variables(t, RL_EMPTY, CURR_STD, PIPE, SEGS, CROSSINGS, s))] == [True, True, False]  # 240/480/241


def test_blank_feature_type_on_current_side_is_error(settings):
    with pytest.raises(ValueError, match="'U1'.*curr_feature_type"):
        run([mrow("M0"), mrow("U1", None, depth_pct=90.0)], settings=settings)
    assert len(run([mrow("M0"), missing_row("M2")], settings=settings)) == 1  # a missing row has no current side
