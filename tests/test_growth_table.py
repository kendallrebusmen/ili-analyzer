import math

import numpy as np
import pandas as pd
import pytest

from ili import growth, reference
from ili.align_welds import align_welds
from ili.config import load_settings
from ili.match import delta_t_years, match_anomalies
from ili.synth import SynthParams, generate

FEATURE_COLUMNS = [
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_log_dist_ft", "segment_id",
    "joint_id", "surface", "ml_origin", "in_recoat", "in_stats",
    "history_class", "confidence", "implausible", "match_needs_review",
    "wt_in", "d_prev_pct", "d_curr_pct", "d_prev_in", "d_curr_in", "dt_yr", "sigma_diff_pct", "k_sigma",
    "install_year",
    "raw_mpy", "tolerance_adjusted_mpy", "segment_percentile_mpy", "new_feature_mpy", "full_life_mpy",
    "default_rate_mpy", "half_life_mpy", "joint_percentile_mpy", "shrinking_fixed_mpy",
    "governing_mpy", "governing_method", "needs_review", "review_reasons", "length_in", "length_rate_ipy",
]
SEGMENT_COLUMNS = [
    "segment_id", "start_ft", "end_ft", "n_features", "n_matched", "raw_mean_mpy", "raw_median_mpy",
    "raw_p_mpy", "pct_negative_raw", "governing_max_mpy",
]
NAN = float("nan")
DT, PREV_YEAR, CURR_YEAR = 5.0, 2021, 2026
# Pipe: 1975 pipe except a section replaced in 2023 at [1000, 1100).
PIPE = pd.DataFrame({
    "start_ft": [0.0, 1000.0, 1100.0], "end_ft": [1000.0, 1100.0, 3000.0],
    "wt_nom_in": [0.25, 0.25, 0.25], "install_year": [1975, 2023, 1975],
})
# SEG-B has no matched rows; 1000-1100 and beyond 2000 ft are outside every segment.
SEGS = pd.DataFrame({
    "start_ft": [0.0, 500.0, 1100.0], "end_ft": [500.0, 1000.0, 2000.0], "segment_id": ["SEG-A", "SEG-B", "SEG-C"],
})
# Tolerance margin over 5 yr at 0.25 in wall: 1.28 * 11.035 / 100 * 0.25 * 1000 / 5.
MARGIN = 1.28 * growth.sigma_diff_pct(load_settings().tool) / 100 * 0.25 * 1000 / 5


@pytest.fixture
def settings():
    return load_settings()


def _with_growth(settings, **kw):
    return settings.model_copy(update={"growth": settings.growth.model_copy(update=kw)})


def _row(mid, mtype, pos, d_prev=NAN, d_curr=NAN, wt=0.25, prev_wt=0.25, length=1.0, ftype="metal_loss",
         confidence="high", implausible=False, needs_review=False):
    matched = mtype in ("matched_1to1", "matched_cluster")
    return {
        "match_id": mid, "match_type": mtype,
        "prev_feature_id": f"A-{mid}" if matched or mtype == "missing" else None,
        "curr_feature_id": None if mtype == "missing" else f"B-{mid}",
        "curr_feature_type": None if mtype == "missing" else ftype,
        "curr_log_dist_ft": NAN if mtype == "missing" else pos,
        "prev_wt_nom_in": prev_wt if matched or mtype == "missing" else NAN,
        "curr_wt_nom_in": wt, "prev_depth_pct": d_prev, "curr_depth_pct": d_curr, "curr_length_in": length,
        # M4 sets confidence and implausible on matched rows only
        "confidence": confidence if matched else None, "implausible": implausible if matched else None,
        "needs_review": needs_review,
    }


def _table(rows):
    return pd.DataFrame(rows)


def _run(rows, settings, pipe=PIPE, segs=SEGS):
    return growth.compute_growth(_table(rows), DT, PREV_YEAR, CURR_YEAR, pipe, segs, settings)


def _by_id(result):
    return result.features.set_index("match_id")


# Raw = (d_curr - d_prev) % * 0.25 in / 100 * 1000 / 5 = 0.5 * delta pct, so raw [-2, 0, 1, 3, 10] mpy is
# delta pct [-4, 0, 2, 6, 20] from 30 %.
SEG_A_MATCHED = [
    _row("M1", "matched_1to1", 100.0, 30.0, 26.0),
    _row("M2", "matched_1to1", 150.0, 30.0, 30.0),
    _row("M3", "matched_cluster", 200.0, 30.0, 32.0),
    _row("M4", "matched_1to1", 250.0, 30.0, 36.0),
    _row("M5", "matched_1to1", 300.0, 30.0, 50.0),
]


# --- synthetic pipeline -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_growth():
    s = load_settings()
    r = generate(SynthParams(line_miles=5))
    ref = r.reference
    wm = align_welds(r.prev_std, r.curr_std, s, ref["repairs_between_runs"]).weld_map
    mt = match_anomalies(r.prev_std, r.curr_std, wm, s, ref["pipe_properties"],
                         ref["repairs_between_runs"]).match_table
    seg = reference.segment_table(s, ref["hca_segments"], 0.0, float(r.curr_std.log_dist_ft.max()))
    dt = delta_t_years(r.prev_std, r.curr_std)
    res = growth.compute_growth(mt, dt, r.prev_std.run_date.iloc[0].year, r.curr_std.run_date.iloc[0].year,
                                ref["pipe_properties"], seg, s)
    return mt, res


def test_one_row_per_current_metal_loss(synth_growth):
    mt, res = synth_growth
    f = res.features
    want = mt[(mt.match_type != "missing") & (mt.curr_feature_type == "metal_loss")]
    assert list(f.columns) == FEATURE_COLUMNS
    assert f.match_id.tolist() == want.match_id.tolist()  # one row each, in table order
    assert len(f) == 494 + 6 + 54 + 10  # + 10 mill anomalies (M11)
    excluded = mt[(mt.match_type == "missing") | (mt.curr_feature_type != "metal_loss")]
    assert set(excluded.curr_feature_type.dropna()) == {"dent"}  # the synth has dents to exclude
    assert not set(excluded.match_id) & set(f.match_id)
    assert set(f.match_type) == {"matched_1to1", "matched_cluster", "new"}
    assert all(isinstance(i, str) for i in f.curr_feature_id)
    assert f.curr_feature_id.tolist() == want.curr_feature_id.tolist()
    assert list(res.segments.columns) == SEGMENT_COLUMNS
    assert res.segments.segment_id.tolist() == ["SEG-01", "SEG-02", "SEG-03", "SEG-04"]
    assert int(res.segments.n_features.sum()) == f.segment_id.notna().sum()
    s = res.summary
    assert sum(s["counts_by_history_class"].values()) == len(f)
    assert sum(s["counts_by_governing_method"].values()) + s["counts_by_review_reason"]["no_governing"] == len(f)
    assert s["n_needs_review"] == int(f.needs_review.sum()) == int(f.review_reasons.notna().sum())
    # M4's review evidence is carried and never lost
    m4 = mt.set_index("match_id").loc[f.match_id]
    assert f.confidence.tolist() == m4.confidence.tolist()
    assert f.implausible.tolist() == m4.implausible.tolist()
    assert f.match_needs_review.tolist() == m4.needs_review.astype(bool).tolist()
    assert f.match_needs_review.any() and f.implausible.eq(True).any()  # the synth exercises both
    assert f.needs_review[f.match_needs_review].all()
    assert f.review_reasons[f.match_needs_review].str.contains("match_review").all()
    assert s["counts_by_review_reason"]["match_review"] == int(f.match_needs_review.sum())
    assert sum(s["raw_histogram"]["counts"]) == s["counts_by_history_class"]["matched"]


# --- hand-built ----------------------------------------------------------------------------------------


def test_segment_percentile_hand_built(settings):
    rows = SEG_A_MATCHED + [
        _row("N1", "new", 400.0, d_curr=18.0),   # SEG-A: new_feature (18 - 10) % -> 4.0 mpy
        _row("N2", "new", 600.0, d_curr=18.0),   # SEG-B: no matched rows (Review Focus #5)
        _row("N3", "new", 2500.0, d_curr=18.0),  # outside every segment
    ]
    res = _run(rows, settings)
    f = _by_id(res)
    assert f.loc[["M1", "M2", "M3", "M4", "M5"], "raw_mpy"].tolist() == pytest.approx([-2, 0, 1, 3, 10])
    assert f.loc[["M1", "M5", "N1"], "segment_percentile_mpy"].tolist() == pytest.approx([7.2, 7.2, 7.2])
    assert f.loc["N1", "history_class"] == "new"
    assert f.loc["N1", "governing_mpy"] == pytest.approx(7.2)
    assert f.loc["N1", "governing_method"] == "segment_percentile"
    assert f.loc["N2", "segment_id"] == "SEG-B" and math.isnan(f.loc["N2", "segment_percentile_mpy"])
    assert f.loc["N3", "segment_id"] is None and math.isnan(f.loc["N3", "segment_percentile_mpy"])
    for mid in ("N2", "N3"):  # the NaN percentile drops out; new_feature governs, flagged as incomplete
        assert f.loc[mid, "governing_mpy"] == pytest.approx(4.0)
        assert f.loc[mid, "governing_method"] == "new_feature"
        assert f.loc[mid, "review_reasons"] == "incomplete_rule"

    # P90 is floored at min_rate_mpy
    floored = _run(SEG_A_MATCHED, _with_growth(settings, min_rate_mpy=8.0))
    assert floored.features.segment_percentile_mpy.tolist() == pytest.approx([8.0] * 5)

    seg = res.segments.set_index("segment_id")
    assert seg.loc["SEG-A", ["start_ft", "end_ft", "n_features", "n_matched"]].tolist() == [0.0, 500.0, 6, 5]
    assert seg.loc["SEG-A", "raw_mean_mpy"] == pytest.approx(2.4)
    assert seg.loc["SEG-A", "raw_median_mpy"] == pytest.approx(1.0)
    assert seg.loc["SEG-A", "raw_p_mpy"] == pytest.approx(7.2)
    assert seg.loc["SEG-A", "pct_negative_raw"] == pytest.approx(20.0)
    assert seg.loc["SEG-A", "governing_max_mpy"] == pytest.approx(10 + MARGIN)
    assert seg.loc["SEG-B", ["n_features", "n_matched"]].tolist() == [1, 0]
    assert all(math.isnan(seg.loc["SEG-B", c]) for c in ("raw_mean_mpy", "raw_median_mpy", "raw_p_mpy",
                                                          "pct_negative_raw"))
    assert seg.loc["SEG-B", "governing_max_mpy"] == pytest.approx(4.0)
    assert seg.loc["SEG-C", "n_features"] == 0 and math.isnan(seg.loc["SEG-C", "governing_max_mpy"])

    s = res.summary
    assert s["delta_t_years"] == DT
    assert s["counts_by_history_class"] == {"matched": 5, "new": 3, "no_history": 0}
    assert s["counts_by_governing_method"] == {
        "raw": 0, "tolerance_adjusted": 3, "segment_percentile": 3, "new_feature": 2, "full_life": 0,
        "default_rate": 0, "half_life": 0, "joint_percentile": 0, "shrinking_fixed": 0,
    }
    assert s["n_needs_review"] == 2
    assert s["counts_by_review_reason"] == {"no_governing": 0, "incomplete_rule": 2, "below_min_rate": 0,
                                            "match_review": 0}
    assert s["pct_negative_raw"] == pytest.approx(20.0)
    assert s["raw_histogram"]["bin_edges_mpy"] == pytest.approx([float(e) for e in range(-2, 11)])
    assert s["raw_histogram"]["counts"] == [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    assert all(type(e) is float for e in s["raw_histogram"]["bin_edges_mpy"])
    assert all(type(c) is int for c in s["raw_histogram"]["counts"])
    assert all(type(v) is int for v in s["counts_by_history_class"].values())


def test_summary_without_matched_rows(settings):
    s = _run([_row("N1", "new", 400.0, d_curr=18.0)], settings, segs=SEGS).summary
    assert s["raw_histogram"] == {"bin_edges_mpy": [], "counts": []}
    assert s["pct_negative_raw"] is None
    assert s["prev_run_id"] is None and s["curr_run_id"] is None
    ids = growth.compute_growth(_table([_row("N1", "new", 400.0, d_curr=18.0)]), DT, PREV_YEAR, CURR_YEAR,
                                PIPE, SEGS, settings, prev_run_id="2021_A", curr_run_id="2026_B").summary
    assert (ids["prev_run_id"], ids["curr_run_id"]) == ("2021_A", "2026_B")


def test_governing_rule_by_class(settings):
    rows = SEG_A_MATCHED + [
        _row("T1", "matched_1to1", 1200.0, 50.0, 20.0),        # SEG-C: raw -15; P90 and tol both floor to 1.0
        _row("X1", "matched_1to1", 2500.0, 30.0, 32.0),        # no segment: percentile NaN, tol governs
        _row("H1", "matched_1to1", 120.0, NAN, 40.0),          # matched without a prev depth -> no_history
        _row("N1", "new", 400.0, d_curr=18.0),
    ]
    f = _by_id(_run(rows, settings))
    assert f.loc[["M1", "M2", "M3", "M4", "M5", "T1", "X1"], "history_class"].eq("matched").all()
    # matched: max(segment_percentile 7.2, tolerance_adjusted raw + margin)
    assert f.loc[["M1", "M2", "M3", "M4", "M5"], "governing_method"].tolist() == [
        "segment_percentile", "segment_percentile", "tolerance_adjusted", "tolerance_adjusted",
        "tolerance_adjusted"]
    assert f.loc[["M1", "M2", "M3", "M4", "M5"], "governing_mpy"].tolist() == pytest.approx(
        [7.2, 7.2, 1 + MARGIN, 3 + MARGIN, 10 + MARGIN])
    assert f.loc["M1", "tolerance_adjusted_mpy"] == pytest.approx(-2 + MARGIN)
    # tie at the 1.0 floor goes to the first method in the list
    assert f.loc["T1", ["segment_percentile_mpy", "tolerance_adjusted_mpy", "governing_mpy"]].tolist() == [1.0] * 3
    assert f.loc["T1", "governing_method"] == "segment_percentile"
    # NaN percentile ignored
    assert math.isnan(f.loc["X1", "segment_percentile_mpy"])
    assert f.loc["X1", "governing_mpy"] == pytest.approx(1 + MARGIN)
    assert f.loc["X1", "governing_method"] == "tolerance_adjusted"
    # no_history: full_life = 0.1 in * 1000 / 51 yr; raw / tolerance / new_feature do not apply
    assert f.loc["H1", "history_class"] == "no_history"
    assert all(math.isnan(f.loc["H1", c]) for c in ("raw_mpy", "tolerance_adjusted_mpy", "new_feature_mpy",
                                                     "default_rate_mpy"))
    assert f.loc["H1", "segment_percentile_mpy"] == pytest.approx(7.2)  # computed, but not in the no_history rule
    assert f.loc["H1", "governing_mpy"] == pytest.approx(100 / 51)
    assert f.loc["H1", "governing_method"] == "full_life"
    # new: raw and tolerance do not apply
    assert f.loc["N1", "history_class"] == "new"
    assert math.isnan(f.loc["N1", "raw_mpy"]) and math.isnan(f.loc["N1", "tolerance_adjusted_mpy"])
    assert f.loc["N1", "new_feature_mpy"] == pytest.approx(4.0)
    assert f.loc["N1", "full_life_mpy"] == pytest.approx(0.045 * 1000 / 51)
    # X1 has no percentile and H1 no default rate (null): each rule was maxed over fewer rates than listed
    assert f.needs_review[f.needs_review].index.tolist() == ["X1", "H1"]
    assert f.loc[["X1", "H1"], "review_reasons"].tolist() == ["incomplete_rule", "incomplete_rule"]

    # the configured list order decides ties; default_rate joins the max when set
    rule = settings.growth.governing.model_copy(update={"matched": ["tolerance_adjusted", "segment_percentile"]})
    f2 = _by_id(_run(rows, _with_growth(settings, governing=rule, default_rate_mpy=2.0)))
    assert f2.loc["T1", "governing_method"] == "tolerance_adjusted"
    assert f2.loc["H1", "governing_mpy"] == 2.0 and f2.loc["H1", "governing_method"] == "default_rate"
    assert not f2.loc["H1", "needs_review"] and f2.loc["H1", "review_reasons"] is None
    assert f2.loc["N1", "default_rate_mpy"] == 2.0 and f2.loc["N1", "governing_method"] == "segment_percentile"


def test_missing_inputs_flag_review(settings):
    assert settings.growth.default_rate_mpy is None  # the shipped value (Review Focus #4)
    rows = [
        _row("E1", "matched_1to1", 100.0, NAN, NAN, wt=NAN, prev_wt=NAN),  # no wall on either side, no depth
        _row("E2", "new", 2500.0, d_curr=NAN, wt=NAN),                       # no depth, outside every segment
        _row("W1", "matched_1to1", 150.0, 30.0, 32.0, wt=NAN, prev_wt=0.3),  # wall falls back to prev
    ]
    res = _run(rows, settings)
    f = _by_id(res)
    assert f.loc["E1", "history_class"] == "no_history" and f.loc["E2", "history_class"] == "new"
    assert f.loc["E1", "wt_in"] == 0.25 and f.loc["E2", "wt_in"] == 0.25  # pipe-table fallback
    assert f.loc["W1", "wt_in"] == 0.3
    assert f.loc["W1", "d_curr_in"] == pytest.approx(0.32 * 0.3)
    for mid in ("E1", "E2"):
        assert math.isnan(f.loc[mid, "full_life_mpy"]) and math.isnan(f.loc[mid, "default_rate_mpy"])
        assert math.isnan(f.loc[mid, "governing_mpy"])
        assert f.loc[mid, "governing_method"] is None
        assert bool(f.loc[mid, "needs_review"]) is True
        assert f.loc[mid, "review_reasons"] == "no_governing;incomplete_rule"
    assert not f.loc["W1", "needs_review"] and f.loc["W1", "review_reasons"] is None
    assert res.summary["n_needs_review"] == 2
    assert f.needs_review.dtype == bool

    # The pipe wall is looked up only for rows that need it: a NaN wall elsewhere in the table is fine ...
    gappy = PIPE.assign(wt_nom_in=[0.25, 0.25, NAN])
    ok = _run([_row("W2", "matched_1to1", 1200.0, 30.0, 32.0)], settings, pipe=gappy)
    assert ok.features.wt_in.tolist() == [0.25]
    # ... but a row that needs it there is a data error
    with pytest.raises(ValueError, match="no wt_nom_in"):
        _run([_row("W3", "matched_1to1", 1200.0, 30.0, 32.0, wt=NAN, prev_wt=NAN)], settings, pipe=gappy)


def test_new_pipe_is_no_history(settings):
    rows = [_row("P1", "new", 1050.0, d_curr=30.0), _row("P2", "new", 900.0, d_curr=30.0)]
    f = _by_id(_run(rows, settings))
    assert f.loc["P1", "install_year"] == 2023
    assert f.loc["P1", "history_class"] == "no_history"
    assert math.isnan(f.loc["P1", "new_feature_mpy"])
    assert f.loc["P1", "full_life_mpy"] == pytest.approx(0.075 * 1000 / 3)  # 2026 - 2023
    assert f.loc["P1", "governing_mpy"] == pytest.approx(25.0)
    assert f.loc["P1", "governing_method"] == "full_life"
    # the old pipe next to it stays `new`
    assert f.loc["P2", "install_year"] == 1975 and f.loc["P2", "history_class"] == "new"
    assert f.loc["P2", "new_feature_mpy"] == pytest.approx(10.0)
    # install_year == prev_year is still `new`
    f3 = _by_id(growth.compute_growth(_table(rows), DT, 2023, CURR_YEAR, PIPE, SEGS, settings))
    assert f3.loc["P1", "history_class"] == "new"


def test_missing_install_year_is_no_history(settings):
    # A NaN install year is a gap in the pipe table, not a reason to abort the run: full_life is NaN and the
    # row cannot be `new` (that needs a known year), so it is no_history and, with no rate, flagged.
    pipe = PIPE.assign(install_year=[1975, 2023, NAN])
    rows = [_row("Y1", "new", 1500.0, d_curr=30.0), _row("Y2", "matched_1to1", 1600.0, 30.0, 32.0),
            _row("Y3", "new", 400.0, d_curr=30.0)]
    f = _by_id(_run(rows, settings, pipe=pipe))
    assert math.isnan(f.loc["Y1", "install_year"]) and math.isnan(f.loc["Y1", "full_life_mpy"])
    assert f.loc["Y1", "history_class"] == "no_history"
    assert math.isnan(f.loc["Y1", "governing_mpy"]) and bool(f.loc["Y1", "needs_review"]) is True
    assert f.loc["Y1", "review_reasons"] == "no_governing;incomplete_rule"
    assert f.loc["Y2", "review_reasons"] is None  # full_life is not in the matched rule
    assert f.loc["Y2", "history_class"] == "matched" and math.isnan(f.loc["Y2", "full_life_mpy"])
    assert f.loc["Y3", "history_class"] == "new" and f.loc["Y3", "install_year"] == 1975
    # coverage is still a data error
    with pytest.raises(ValueError, match="outside the pipe properties table"):
        _run([_row("Y4", "new", 9000.0, d_curr=30.0)], settings, pipe=pipe)


def test_review_reasons(settings):
    # Governing is not floored: a rate below min_rate_mpy, or one maxed over fewer methods than the class's
    # rule lists, is flagged so M6 cannot use it silently.
    rows = SEG_A_MATCHED + [
        _row("B1", "new", 2500.0, d_curr=6.0),            # outside every segment: new_feature (6 - 10) % -> -2.0
        _row("B2", "new", 600.0, d_curr=6.0),             # SEG-B, no matched rows: same
        _row("C1", "matched_1to1", 300.5, NAN, 40.0),     # prev depth missing: no_history, full_life 1.96
    ]
    f = _by_id(_run(rows, settings))
    for mid in ("B1", "B2"):
        assert f.loc[mid, "history_class"] == "new"
        assert f.loc[mid, "governing_mpy"] == pytest.approx(-2.0)  # not floored ...
        assert f.loc[mid, "governing_method"] == "new_feature"
        assert bool(f.loc[mid, "needs_review"]) is True             # ... but flagged
        assert f.loc[mid, "review_reasons"] == "incomplete_rule;below_min_rate"
    assert f.loc["C1", "history_class"] == "no_history"
    assert f.loc["C1", "governing_mpy"] == pytest.approx(100 / 51) and f.loc["C1", "governing_method"] == "full_life"
    assert bool(f.loc["C1", "needs_review"]) is True
    assert f.loc["C1", "review_reasons"] == "incomplete_rule"  # default_rate null
    for mid in ("M1", "M2", "M3", "M4", "M5"):  # normal matched rows
        assert not f.loc[mid, "needs_review"] and f.loc[mid, "review_reasons"] is None
    # a governing rate below the floor with a complete rule: below_min_rate alone
    low = _by_id(_run(SEG_A_MATCHED + [_row("B3", "new", 400.0, d_curr=6.0)], settings))
    assert low.loc["B3", "governing_mpy"] == pytest.approx(7.2)  # the segment percentile governs: no flag
    assert low.loc["B3", "review_reasons"] is None
    only_new = settings.growth.governing.model_copy(update={"new": ["new_feature"]})
    low = _by_id(_run(SEG_A_MATCHED + [_row("B3", "new", 400.0, d_curr=6.0)], _with_growth(settings, governing=only_new)))
    assert low.loc["B3", "review_reasons"] == "below_min_rate"


def test_match_review_carried(settings):
    rows = [
        _row("R1", "matched_1to1", 100.0, 30.0, 26.0, confidence="low", implausible=True, needs_review=True),
        _row("R2", "matched_1to1", 150.0, 30.0, 32.0, confidence="medium"),
        _row("R3", "matched_cluster", 200.0, 30.0, 34.0, confidence="medium", needs_review=True),
        _row("R4", "new", 2500.0, d_curr=6.0, needs_review=True),
        _row("R5", "new", 300.0, d_curr=20.0, needs_review=None),  # unset flag: not a match review
    ]
    f = _by_id(_run(rows, settings))
    assert list(f.reset_index().columns) == FEATURE_COLUMNS
    assert f.confidence.tolist() == ["low", "medium", "medium", None, None]
    assert f.implausible.tolist() == [True, False, False, None, None]
    assert f.match_needs_review.tolist() == [True, False, True, True, False]
    assert f.match_needs_review.dtype == bool
    assert f.needs_review.tolist() == [True, False, True, True, False]
    assert f.review_reasons.tolist() == ["match_review", None, "match_review",
                                         "incomplete_rule;below_min_rate;match_review", None]
    # the rates do not change; only the flags do
    assert f.loc["R1", "governing_mpy"] == pytest.approx(-2 + MARGIN)
    assert f.loc["R1", "governing_method"] == "tolerance_adjusted"
    with pytest.raises(ValueError, match="confidence"):
        _run([{k: v for k, v in rows[0].items() if k != "confidence"}], settings)


def test_length_growth_options(settings):
    rows = [
        _row("L1", "new", 1050.0, d_curr=30.0, length=3.0),  # no_history, governing 25 mpy, d 0.075 in
        _row("L2", "new", 1060.0, d_curr=0.0, length=2.0),   # d_curr 0 -> proportional NaN
    ]
    f = _run(rows, settings).features
    assert f.length_in.tolist() == [3.0, 2.0]
    assert f.length_rate_ipy.tolist() == [0.0, 0.0]
    fixed = _run(rows, _with_growth(settings, length_growth="fixed_rate", length_rate_ipy=0.05)).features
    assert fixed.length_rate_ipy.tolist() == [0.05, 0.05]
    prop = _run(rows, _with_growth(settings, length_growth="proportional")).features
    assert prop.length_rate_ipy[0] == pytest.approx(3.0 * 25.0 / 1000 / 0.075)  # 1.0 in/yr
    assert prop.length_rate_ipy[0] == pytest.approx(1.0)
    assert math.isnan(prop.length_rate_ipy[1])


def test_excludes_dents_and_missing(settings):
    rows = [_row("D1", "matched_1to1", 100.0, 30.0, 32.0, ftype="dent"), _row("G1", "new", 120.0, d_curr=20.0,
                                                                             ftype="gouge"),
            _row("X1", "missing", 0.0, d_prev=30.0), _row("K1", "new", 140.0, d_curr=20.0)]
    f = _run(rows, settings).features
    assert f.match_id.tolist() == ["K1"]


def test_rejects_duplicate_segment_ids(settings):
    dup = pd.DataFrame({"start_ft": [0.0, 1100.0], "end_ft": [1000.0, 2000.0], "segment_id": ["HCA-1", "HCA-1"]})
    with pytest.raises(ValueError, match="repeats segment_id"):
        _run(SEG_A_MATCHED, settings, segs=dup)


def test_requires_match_table_columns(settings):
    with pytest.raises(ValueError, match="curr_depth_pct"):
        growth.compute_growth(_table([_row("K1", "new", 140.0, d_curr=20.0)]).drop(columns="curr_depth_pct"),
                              DT, PREV_YEAR, CURR_YEAR, PIPE, SEGS, settings)


# --- M11: new rates and recoat exclusion --------------------------------------------------------------------


def _recoat_rows():
    rows = [_row(f"M{i}", "matched_1to1", 100.0 + 50 * i, 30.0, 30.0 + 2 * i) for i in range(1, 5)]
    rows.append(_row("R1", "matched_1to1", 400.0, 30.0, 80.0))  # raw 25 mpy, inside the recoat
    for r, j in zip(rows, ["J1"] * 5):
        r["curr_us_gw_id"] = j
    return rows


RECOAT = pd.DataFrame({"start_ft": [390.0], "end_ft": [410.0], "repair_type": ["recoat"]})


def _run_repairs(rows, settings, repairs):
    return growth.compute_growth(_table(rows), DT, PREV_YEAR, CURR_YEAR, PIPE, SEGS, settings, repairs=repairs)


def test_recoated_rows_left_out_of_distributions(settings):
    rows = _recoat_rows()
    on = _run_repairs(rows, _with_growth(settings, joint_percentile_min_matches=3), RECOAT)
    off = _run_repairs(rows, _with_growth(settings, joint_percentile_min_matches=3, exclude_recoated_from_stats=False),
                       RECOAT)
    none = _run_repairs(rows, _with_growth(settings, joint_percentile_min_matches=3), None)
    f_on, f_off = _by_id(on), _by_id(off)
    assert f_on.loc["R1", "in_recoat"] and not f_on.loc["R1", "in_stats"]
    assert f_off.loc["R1", "in_recoat"] and f_off.loc["R1", "in_stats"]
    assert f_on.loc["R1", "raw_mpy"] == pytest.approx(25.0)  # still gets its own rate
    assert f_on.loc["M1", "segment_percentile_mpy"] < f_off.loc["M1", "segment_percentile_mpy"]
    assert f_on.loc["M1", "joint_percentile_mpy"] < f_off.loc["M1", "joint_percentile_mpy"]
    assert f_on.loc["R1", "segment_percentile_mpy"] == f_on.loc["M1", "segment_percentile_mpy"]
    assert on.summary["n_excluded_recoated"] == 1 and off.summary["n_excluded_recoated"] == 0
    # no recoat in the repairs: nothing excluded, same numbers as exclusion off
    assert none.summary["n_excluded_recoated"] == 0
    assert _by_id(none).governing_mpy.tolist() == f_off.governing_mpy.tolist()


def test_no_recoat_leaves_governing_unchanged(settings):
    rows = SEG_A_MATCHED + [_row("N1", "new", 400.0, d_curr=18.0)]
    base = _by_id(_run(rows, settings))
    other = _by_id(_run_repairs(rows, settings, RECOAT.assign(repair_type="sleeve")))
    assert base.governing_mpy.tolist() == other.governing_mpy.tolist()
    assert base.governing_method.tolist() == other.governing_method.tolist()
    assert not base.in_recoat.any() and base.in_stats.all()


def test_new_columns_and_summary(settings):
    rows = _recoat_rows()
    for r in rows:
        r.update(curr_surface="external", curr_ml_origin="corrosion")
    res = _run(rows, settings)
    f = _by_id(res)
    assert f.loc["M1", "joint_id"] == "J1" and f.loc["M1", "surface"] == "external"
    assert f.loc["M1", "ml_origin"] == "corrosion"
    s = res.summary
    assert s["governing_profile"] == settings.growth.governing_profile
    assert s["joint_percentile"] == settings.growth.joint_percentile
    assert {"joint_percentile_min_matches", "shrinking_rate_mpy", "n_excluded_recoated"} <= set(s)


# --- M11: pit_to_pit governing profile ----------------------------------------------------------------------


def _pit(settings, **kw):
    return _with_growth(settings, governing_profile="pit_to_pit", shrinking_rate_mpy=2.5, **kw)


def test_pit_to_pit_matched_grew_and_shrank(settings):
    rows = [
        _row("G1", "matched_1to1", 100.0, 30.0, 30.8),   # raw 0.4 mpy, grew: floored at min_rate_mpy
        _row("G2", "matched_1to1", 150.0, 30.0, 36.0),   # raw 3.0
        _row("G3", "matched_1to1", 200.0, 30.0, 30.0),   # raw 0, no change counts as grew
        _row("S1", "matched_1to1", 250.0, 30.0, 26.0),   # raw -2.0, shrank
    ]
    f = _by_id(_run(rows, _pit(settings)))
    assert f.loc["G1", "governing_mpy"] == pytest.approx(settings.growth.min_rate_mpy) == pytest.approx(1.0)
    assert f.loc["G1", "governing_method"] == "raw"
    assert f.loc["G2", "governing_mpy"] == pytest.approx(3.0) and f.loc["G2", "governing_method"] == "raw"
    assert f.loc["G3", "governing_mpy"] == pytest.approx(1.0) and f.loc["G3", "governing_method"] == "raw"
    assert f.loc["S1", "governing_mpy"] == pytest.approx(2.5) and f.loc["S1", "governing_method"] == "shrinking_fixed"
    assert not f.needs_review.any()  # a by-design rule is not "incomplete"


def test_pit_to_pit_new_falls_back_to_half_life(settings):
    rows = [_row(f"M{i}", "matched_1to1", 100.0 + 50 * i, 30.0, 30.0 + 2 * i) for i in range(1, 4)]
    for r in rows:
        r["curr_us_gw_id"] = "J1"
    rows += [
        _row("N1", "new", 400.0, d_curr=20.0),             # no joint: half_life
        _row("N2", "new", 410.0, d_curr=20.0),             # joint J1 with 3 matches: joint_percentile
        _row("H1", "new", 1050.0, d_curr=20.0),            # new pipe (2023): no_history, half_life
    ]
    rows[4]["curr_us_gw_id"] = "J1"
    f = _by_id(_run(rows, _pit(settings, joint_percentile_min_matches=3)))
    assert f.loc["N1", "governing_method"] == "half_life"
    assert f.loc["N1", "governing_mpy"] == pytest.approx(f.loc["N1", "half_life_mpy"])
    assert f.loc["N2", "governing_method"] == "joint_percentile"
    assert f.loc["N2", "governing_mpy"] == pytest.approx(f.loc["N2", "joint_percentile_mpy"])
    assert f.loc["H1", "history_class"] == "no_history" and f.loc["H1", "governing_method"] == "half_life"
    assert f.loc["H1", "governing_mpy"] == pytest.approx(f.loc["H1", "half_life_mpy"])
    assert not f.needs_review.any()


def test_pit_to_pit_shrinking_null_is_flagged(settings):
    rows = [_row("S1", "matched_1to1", 250.0, 30.0, 26.0), _row("G1", "matched_1to1", 100.0, 30.0, 36.0)]
    f = _by_id(_run(rows, _with_growth(settings, governing_profile="pit_to_pit", shrinking_rate_mpy=None)))
    assert math.isnan(f.loc["S1", "governing_mpy"]) and f.loc["S1", "governing_method"] is None
    assert f.loc["S1", "review_reasons"] == "no_governing" and bool(f.loc["S1", "needs_review"])
    assert f.loc["G1", "governing_mpy"] == pytest.approx(3.0)


def test_conservative_default_unchanged(synth_growth, settings):
    mt, res = synth_growth
    assert settings.growth.governing_profile == "conservative"
    f = res.features
    # The default profile is the pre-M11 list rule: same values as the list rule applied directly.
    cls = f.history_class.to_numpy(object)
    gov, method = growth.governing_rate(f, cls, _with_growth(settings, governing_profile="conservative"))
    assert np.array_equal(f.governing_mpy.to_numpy(float), gov, equal_nan=True)
    assert f.governing_method.tolist() == list(method)
    # and it differs from pit_to_pit on the same data, so the profile is really dispatched
    other, _ = growth.governing_rate(f, cls, _pit(settings))
    assert not np.array_equal(gov, other, equal_nan=True)


# --- M11: recoated rows out of the summary statistics; growth summary by surface -----------------------------


def test_recoated_rows_left_out_of_summary_statistics(settings):
    rows = _recoat_rows()
    on = _run_repairs(rows, settings, RECOAT)
    off = _run_repairs(rows, _with_growth(settings, exclude_recoated_from_stats=False), RECOAT)
    none = _run_repairs(rows, settings, None)
    seg_on, seg_off = on.segments.set_index("segment_id"), off.segments.set_index("segment_id")
    # M1..M4 raw 1, 2, 3, 4 mpy (+R1 25 mpy at 400 ft, same segment); n_features and governing max cover all rows
    seg = seg_on.index[seg_on.n_features == seg_on.n_features.max()][0]
    assert seg_on.loc[seg, "n_matched"] == seg_off.loc[seg, "n_matched"] - 1
    assert seg_on.loc[seg, "raw_mean_mpy"] < seg_off.loc[seg, "raw_mean_mpy"]
    assert seg_on.loc[seg, "n_features"] == seg_off.loc[seg, "n_features"]
    assert seg_on.loc[seg, "governing_max_mpy"] == seg_off.loc[seg, "governing_max_mpy"]
    assert on.summary["raw_histogram"]["counts"] != off.summary["raw_histogram"]["counts"]
    assert sum(on.summary["raw_histogram"]["counts"]) == sum(off.summary["raw_histogram"]["counts"]) - 1
    assert on.summary["n_features"] == off.summary["n_features"]
    # no recoat in the repairs: statistics identical to exclusion off
    pd.testing.assert_frame_equal(none.segments, off.segments)
    assert none.summary["raw_histogram"] == off.summary["raw_histogram"]
    assert none.summary["pct_negative_raw"] == off.summary["pct_negative_raw"]


def _surface_features():
    cols = ["curr_feature_id", "surface", "ml_origin", "in_recoat", "history_class", "d_prev_pct", "d_curr_pct",
            "d_curr_in", "install_year", "half_life_mpy", "full_life_mpy", "raw_mpy"]
    return pd.DataFrame([
        ["E1", "EXT", "corrosion", False, "matched", 20.0, 30.0, 0.075, 1970, 2.0, 1.5, 4.0],
        ["E2", "EXT", "corrosion", True, "matched", 20.0, 50.0, 0.125, 1980, 3.0, 2.5, 10.0],
        ["E3", "EXT", None, False, "new", NAN, 40.0, 0.100, 1975, 2.5, 2.0, NAN],
        ["I1", "INT", "corrosion", False, "no_history", NAN, 25.0, 0.0625, 1990, 1.0, 0.5, NAN],
    ], columns=cols)


def test_growth_summary_by_surface_values():
    out = growth.growth_summary_by_surface(_surface_features(), 2026).set_index(["surface", "scope"])
    assert list(out.columns) == growth.SURFACE_SUMMARY_COLUMNS[2:]
    a = out.loc[("EXT", "all")]
    assert (a.n_features, a.n_matched, a.matched_share) == (3, 2, "2 of 3")
    assert (a.max_depth_pct, a.max_depth_in, a.deepest_curr_feature_id) == (50.0, 0.125, "E2")
    assert a.pipe_age_yr == 46 and a.deepest_half_life_mpy == 3.0 and a.deepest_full_life_mpy == 2.5
    assert a.max_raw_mpy == 10.0 and a.mean_raw_mpy == 7.0 and a.mean_depth_change_pct == 20.0  # (10 + 30) / 2
    x = out.loc[("EXT", "excluding_recoated")]
    assert (x.n_features, x.n_matched, x.matched_share) == (2, 1, "1 of 2")
    assert (x.max_depth_pct, x.deepest_curr_feature_id, x.pipe_age_yr) == (40.0, "E3", 51)
    assert (x.max_raw_mpy, x.mean_raw_mpy, x.mean_depth_change_pct) == (4.0, 4.0, 10.0)
    i = out.loc[("INT", "all")]
    assert (i.matched_share, i.max_depth_pct, i.deepest_curr_feature_id) == ("0 of 1", 25.0, "I1")
    assert math.isnan(i.max_raw_mpy) and math.isnan(i.mean_depth_change_pct)
    assert len(out) == 4


def test_manufacturing_left_out():
    f = _surface_features()
    f.loc[1, "ml_origin"] = "manufacturing"  # E2, the deepest
    ext = growth.growth_summary_by_surface(f, 2026).set_index(["surface", "scope"]).loc[("EXT", "all")]
    assert ext.n_features == 2 and ext.deepest_curr_feature_id == "E3" and ext.max_raw_mpy == 4.0


def test_empty_surface_row_blank():
    f = _surface_features()
    out = growth.growth_summary_by_surface(f[f.surface == "EXT"], 2026).set_index(["surface", "scope"])
    r = out.loc[("INT", "all")]
    assert r.n_features == 0 and r.matched_share == "0 of 0"
    assert r.isna()[["max_depth_pct", "max_depth_in", "deepest_curr_feature_id", "pipe_age_yr", "max_raw_mpy",
                     "mean_raw_mpy", "mean_depth_change_pct"]].all()


def test_compute_growth_carries_by_surface(settings):
    rows = _recoat_rows()
    for r in rows:
        r.update(curr_surface="EXT", curr_ml_origin="corrosion")
    res = _run_repairs(rows, settings, RECOAT)
    s = res.by_surface.set_index(["surface", "scope"])
    assert s.loc[("EXT", "all"), "n_features"] == 5 and s.loc[("EXT", "excluding_recoated"), "n_features"] == 4
