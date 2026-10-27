from datetime import date

import numpy as np
import pandas as pd
import pytest

from ili.align_welds import align_welds
from ili.config import load_settings
from ili.match import ANOMALY_TYPES, match_anomalies
from ili.synth import SynthParams, generate
from tests.test_match_place import _std, _wm

S = load_settings()

COLUMNS = (
    ["match_id", "match_type", "score", "confidence", "review_status", "needs_review", "missing_hint",
     "n_prev", "n_curr", "absorbed", "prev_feature_id", "curr_feature_id"]
    + [f"{s}_{c}" for s in ("prev", "curr") for c in (
        "feature_type", "log_dist_ft", "wm_row", "us_gw_id", "dist_from_us_gw_ft", "wt_nom_in", "depth_pct",
        "dent_depth_in", "length_in", "width_in", "clock_min", "surface", "seam_interaction", "dent_metal_loss",
        "vendor_cluster_id")]
    + ["curr_pos_in_prev_ft", "axial_diff_in", "clock_diff_min", "depth_change_pct", "implausible"]
)
TYPES = ["matched_1to1", "matched_cluster", "missing", "new"]


@pytest.fixture(scope="module")
def run():
    r = generate(SynthParams(line_miles=5))
    pipe, repairs = r.reference["pipe_properties"], r.reference["repairs_between_runs"]
    wm = align_welds(r.prev_std, r.curr_std, S, repairs).weld_map
    return r, wm, match_anomalies(r.prev_std, r.curr_std, wm, S, pipe, repairs)


def _split(ids: pd.Series) -> list:
    return [i for v in ids.dropna() for i in v.split(";")]


def _count(v) -> int:
    return 0 if v is None else len(v.split(";"))


def test_every_anomaly_in_exactly_one_row(run):
    r, _, res = run
    t = res.match_table
    for side, std in (("prev", r.prev_std), ("curr", r.curr_std)):
        got = _split(t[f"{side}_feature_id"])
        want = std[std.feature_type.isin(ANOMALY_TYPES)].feature_id
        assert len(got) == len(set(got)) == len(want)
        assert set(got) == set(want)
        assert set(std[~std.feature_type.isin(ANOMALY_TYPES)].feature_id).isdisjoint(got)
        assert t[f"{side}_feature_type"].dropna().isin(ANOMALY_TYPES).all()
    assert (t.n_prev == t.prev_feature_id.map(_count)).all() and (t.n_curr == t.curr_feature_id.map(_count)).all()
    assert set(t.match_type) == set(TYPES)
    assert (t.match_id == [f"M{i:05d}" for i in range(1, len(t) + 1)]).all()


def test_rows_in_prev_frame_position_order(run):
    t = run[2].match_table
    pos = t.prev_log_dist_ft.where(t.prev_log_dist_ft.notna(), t.curr_pos_in_prev_ft)
    assert pos.is_monotonic_increasing


def test_replaced_section_features_missing_in_repair(run):
    r, _, res = run
    gt = r.ground_truth
    truth = gt[(gt.curr_status == "removed_by_repair") & gt.prev_feature_id.notna()]
    assert len(truth) == 3
    t = res.match_table.set_index("prev_feature_id")
    for fid in truth.prev_feature_id:
        row = t.loc[fid]
        assert row.match_type == "missing" and row.missing_hint == "in_repaired_section"
        assert row.confidence is None and bool(row.needs_review) is False
    hinted = res.match_table[res.match_table.missing_hint == "in_repaired_section"]
    assert set(hinted.prev_feature_id) == set(truth.prev_feature_id)


def test_columns_and_draft_fields(run):
    r, _, res = run
    t, summary = res.match_table, res.summary
    assert list(t.columns) == COLUMNS
    assert (t.review_status == "pending").all()
    unmatched = t.match_type.isin(["missing", "new"])
    assert (t.confidence.isna() == unmatched).all()
    assert set(t.confidence[~unmatched]) <= {"high", "medium", "low"}
    assert (t.missing_hint.notna() == (t.match_type == "missing")).all()
    assert t.implausible[unmatched].isna().all() and t.score[unmatched].isna().all()
    assert t.implausible[~unmatched].map(lambda v: isinstance(v, (bool, np.bool_))).all()
    assert (t.needs_review == ((t.confidence == "low") | (t.missing_hint == "unexplained") | t.absorbed)).all()
    assert t.absorbed.dtype == bool and not t.absorbed[t.match_type != "matched_cluster"].any()
    assert (t.confidence[t.absorbed] != "high").all() and summary["n_absorbed"] == int(t.absorbed.sum())
    assert all(isinstance(v, (bool, np.bool_)) for v in t.needs_review)
    assert summary["n_needs_review"] == int(t.needs_review.sum())
    assert summary["counts_by_match_type"] == t.match_type.value_counts().reindex(TYPES, fill_value=0).to_dict()
    assert sum(summary["counts_by_hint"].values()) == summary["counts_by_match_type"]["missing"]
    assert sum(summary["counts_by_confidence"].values()) == len(t) - unmatched.sum()
    assert summary["n_prev_anomalies"] == t.n_prev.sum() and summary["n_curr_anomalies"] == t.n_curr.sum()
    assert summary["prev_run_id"] == r.prev_std.run_id.iloc[0] and summary["curr_run_id"] == r.curr_std.run_id.iloc[0]
    assert summary["delta_t_years"] > 0


# ---- hand-built case -------------------------------------------------------------------------

def _frame(rows, run_date):
    """Rows: (feature_type, log_dist_ft, feature_id, depth, clock, length, vendor_cluster_id)."""
    df = _std([(r[0], r[1]) for r in rows], run_date)
    df["feature_id"] = [r[2] for r in rows]
    df["run_id"] = f"RUN-{run_date.year}"
    cols = {"depth_pct": 3, "clock_min": 4, "length_in": 5, "vendor_cluster_id": 6}
    for k, i in cols.items():
        df[k] = [r[i] if len(r) > i else None for r in rows]
    df["depth_pct"] = df.depth_pct.astype(float)
    df["clock_min"] = pd.array(df.clock_min, dtype="Int64")
    df["width_in"], df["wt_nom_in"] = 0.3, 0.25
    return df


def _runs():
    wm = _wm(("1:1", 0.0, 1000.0, 0.0, 1000.0), ("removed", 1000.0, 100.0, np.nan, np.nan),
             ("1:1", 1100.0, 1000.0, 1000.0, 1000.0))
    wm["in_repair"] = [False, True, False]
    ml = "metal_loss"
    prev = _frame([
        ("valve", 50.0, "PV"),
        (ml, 100.0, "P0", 30.0, 100, 1.0),
        (ml, 300.0 + 1 / 12, "P1", 28.0, 200, 0.5),  # id order differs from log order
        (ml, 300.0, "P2", 30.0, 200, 0.5),
        (ml, 500.0, "Pthr", 15.0, 400, 1.0),
        (ml, 700.0, "Punx", 40.0, 500, 1.0),
        (ml, 800.0, "Pnan", None, 600, 1.0),
        (ml, 1050.0, "Prow", 40.0, 300, 1.0),
        (ml, 1500.0, "Pspan", 50.0, 300, 1.0),
    ], date(2021, 6, 15))
    curr = _frame([
        ("weld", 10.0, "CW"),
        (ml, 100.2, "C0", 35.0, 110, 1.0),
        (ml, 300.04, "C1", 32.0, 200, 2.0, "V1"),
        (ml, 900.0, "Cnew", 25.0, 50, 1.0),
    ], date(2026, 6, 15))
    pipe = pd.DataFrame({"start_ft": [0.0], "end_ft": [2000.0], "od_in": [12.75]})
    repairs = pd.DataFrame({"start_ft": [1390.0], "end_ft": [1410.0]})
    return prev, curr, wm, pipe, repairs


def test_hand_built_table():
    prev, curr, wm, pipe, repairs = _runs()
    res = match_anomalies(prev, curr, wm, S, pipe, repairs)
    t = res.match_table.set_index("match_id")
    assert list(t.match_type) == ["matched_1to1", "matched_cluster", "missing", "missing", "missing", "new",
                                  "missing", "missing"]
    assert list(t.prev_feature_id.fillna("-")) == ["P0", "P2;P1", "Pthr", "Punx", "Pnan", "-", "Prow", "Pspan"]
    assert list(t.curr_feature_id.fillna("-")) == ["C0", "C1", "-", "-", "-", "Cnew", "-", "-"]
    assert list(t.missing_hint.fillna("-")) == ["-", "-", "near_threshold", "unexplained", "unexplained", "-",
                                                "in_repaired_section", "in_repaired_section"]

    one = t.loc["M00001"]
    a, c = 2.4 / S.matching.axial_tol_in, 10 / S.matching.clock_tol_min
    assert (one.n_prev, one.n_curr) == (1, 1)
    assert (one.prev_log_dist_ft, one.curr_log_dist_ft) == (100.0, 100.2)
    assert one.curr_pos_in_prev_ft == pytest.approx(100.2)
    assert (one.prev_depth_pct, one.curr_depth_pct, one.prev_clock_min, one.curr_clock_min) == (30.0, 35.0, 100, 110)
    assert one.prev_wm_row == one.curr_wm_row == 0
    assert one.axial_diff_in == pytest.approx(2.4) and one.clock_diff_min == 10
    assert one.depth_change_pct == pytest.approx(5.0) and one.implausible is False
    assert one.score == pytest.approx(np.sqrt((a * a + c * c) / 2))
    assert one.confidence == "medium" and one.review_status == "pending" and bool(one.needs_review) is False

    cl = t.loc["M00002"]
    assert (cl.n_prev, cl.n_curr) == (2, 1)
    assert cl.prev_log_dist_ft == pytest.approx(300.0 + 0.5 / 12)
    assert cl.prev_length_in == pytest.approx(1.5) and cl.prev_depth_pct == 30.0 and cl.prev_clock_min == 200
    assert cl.curr_log_dist_ft == pytest.approx(300.04) and cl.curr_length_in == pytest.approx(2.0)
    assert (cl.curr_depth_pct, cl.curr_vendor_cluster_id) == (32.0, "V1")
    assert cl.axial_diff_in == 0 and cl.clock_diff_min == 0 and cl.score == pytest.approx(0) and cl.confidence == "high"
    assert cl.curr_pos_in_prev_ft == pytest.approx(300.04)

    miss = t.loc["M00003"]
    assert (miss.n_prev, miss.n_curr, miss.prev_depth_pct) == (1, 0, 15.0)
    assert miss.curr_feature_type is None and np.isnan(miss.curr_log_dist_ft) and pd.isna(miss.curr_wm_row)
    assert miss.confidence is None and np.isnan(miss.score) and miss.implausible is None
    assert bool(t.loc["M00003"].needs_review) is False and bool(t.loc["M00004"].needs_review) is True
    assert t.loc["M00007"].prev_wm_row == 1 and t.loc["M00008"].prev_wm_row == 2

    new = t.loc["M00006"]
    assert (new.n_prev, new.n_curr, new.curr_depth_pct, new.curr_log_dist_ft) == (0, 1, 25.0, 900.0)
    assert new.prev_feature_type is None and new.confidence is None and new.missing_hint is None
    assert new.curr_pos_in_prev_ft == pytest.approx(900.0)
    assert res.summary["counts_by_match_type"] == {"matched_1to1": 1, "matched_cluster": 1, "missing": 5, "new": 1}
    assert res.summary["counts_by_hint"] == {"in_repaired_section": 2, "near_threshold": 1, "unexplained": 2,
                                           "rejected_by_review": 0}
    assert res.summary["counts_by_confidence"] == {"high": 1, "medium": 1, "low": 0}
    assert res.summary["n_needs_review"] == 2


def test_repair_span_alone_gives_hint():
    prev, curr, wm, pipe, _ = _runs()
    t = match_anomalies(prev, curr, wm, S, pipe, None).match_table.set_index("prev_feature_id")
    assert t.loc["Pspan"].missing_hint == "unexplained"  # only the span made it a repair hint
    assert t.loc["Prow"].missing_hint == "in_repaired_section"  # the removed row alone does


# ---- reference errors ------------------------------------------------------------------------

def test_reference_errors():
    prev, curr, wm, pipe, repairs = _runs()
    with pytest.raises(ValueError, match="1400"):
        match_anomalies(prev, curr, wm, S, pipe.assign(end_ft=1000.0), repairs)
    with pytest.raises(ValueError, match="od_in"):
        match_anomalies(prev, curr, wm, S, pipe.drop(columns="od_in"), repairs)
    with pytest.raises(ValueError, match="start_ft"):
        match_anomalies(prev, curr, wm, S, pipe, repairs.rename(columns={"start_ft": "begin_ft"}))
    # a small overhang (within anchor_tol_ft) uses the nearest row
    ok = match_anomalies(prev, curr, wm, S, pipe.assign(end_ft=1350.0), repairs)
    assert len(ok.match_table) == 8


def test_empty_pipe_table_or_missing_od_is_value_error():
    prev, curr, wm, pipe, repairs = _runs()
    with pytest.raises(ValueError, match="no rows"):
        match_anomalies(prev, curr, wm, S, pipe.iloc[:0], repairs)
    with pytest.raises(ValueError, match="od_in"):
        match_anomalies(prev, curr, wm, S, pipe.assign(od_in=np.nan), repairs)


def test_weld_map_missing_columns_is_value_error():
    prev, curr, wm, pipe, repairs = _runs()
    with pytest.raises(ValueError, match=r"weld map.*'curr_length_ft'.*'in_repair'"):
        match_anomalies(prev, curr, wm.drop(columns=["in_repair", "curr_length_ft"]), S, pipe, repairs)


def test_od_only_needed_in_matched_rows():
    """Prev features in the removed row and past the last matched row map outside a pipe table that
    ends with the current run; they never use OD, so matching succeeds."""
    prev, curr, wm, pipe, repairs = _runs()
    prev = prev[prev.feature_id != "Pspan"]
    curr_end = pd.DataFrame({"start_ft": [0.0], "end_ft": [920.0], "od_in": [12.75]})
    wm = wm.iloc[:2]  # the run ends after the removed row: Prow (removed) has no OD
    t = match_anomalies(prev, curr, wm, S, curr_end, None).match_table.set_index("prev_feature_id")
    assert t.loc["Prow"].match_type == "missing" and t.loc["Prow"].missing_hint == "in_repaired_section"


# ---- runs of different extent ----------------------------------------------------------------

@pytest.fixture(scope="module")
def seed1():
    r = generate(SynthParams(line_miles=5, seed=1))
    welds = r.curr_std[r.curr_std.feature_type == "girth_weld"].log_dist_ft.sort_values().to_numpy()
    return r, welds


def _match_extent(r, curr, pipe, repairs):
    wm = align_welds(r.prev_std, curr, S, repairs).weld_map
    t = match_anomalies(r.prev_std, curr, wm, S, pipe, repairs).match_table
    for side, std in (("prev", r.prev_std), ("curr", curr)):
        got = _split(t[f"{side}_feature_id"])
        assert sorted(got) == sorted(std[std.feature_type.isin(ANOMALY_TYPES)].feature_id)
    return t


def test_current_run_starting_later_rezeroed(seed1):
    """The current run starts 60 joints downstream with its odometer re-zeroed (pipe and repairs in
    its frame): the previous run's first 60 joints map to negative current-run positions."""
    r, welds = seed1
    x0 = welds[60]
    curr = r.curr_std[r.curr_std.log_dist_ft >= x0].copy()
    curr["log_dist_ft"] -= x0
    pipe = r.reference["pipe_properties"].copy()
    pipe[["start_ft", "end_ft"]] -= x0
    pipe = pipe[pipe.end_ft > 0].copy()
    pipe.loc[pipe.index[0], "start_ft"] = 0.0
    repairs = r.reference["repairs_between_runs"].copy()
    repairs[["start_ft", "end_ft"]] -= x0
    t = _match_extent(r, curr, pipe, repairs)
    head = t[(t.match_type == "missing") & (t.prev_log_dist_ft < x0 - S.alignment.anchor_tol_ft)]
    assert len(head) > 0 and t.match_type.eq("matched_1to1").sum() > 400


def test_current_run_ending_early_with_pipe_table(seed1):
    r, welds = seed1
    xe = welds[-61]
    curr = r.curr_std[r.curr_std.log_dist_ft <= xe]
    pipe = r.reference["pipe_properties"]
    pipe = pipe[pipe.start_ft < xe].copy()
    pipe.loc[pipe.index[-1], "end_ft"] = xe
    t = _match_extent(r, curr, pipe, r.reference["repairs_between_runs"])
    tail = t[(t.match_type == "missing") & (t.prev_log_dist_ft > xe + S.alignment.anchor_tol_ft)]
    assert len(tail) > 0 and t.match_type.eq("matched_1to1").sum() > 400


def test_hand_built_summary_ids():
    prev, curr, wm, pipe, repairs = _runs()
    s = match_anomalies(prev, curr, wm, S, pipe, repairs).summary
    assert (s["prev_run_id"], s["curr_run_id"]) == ("RUN-2021", "RUN-2026")
    assert (s["n_prev_anomalies"], s["n_curr_anomalies"]) == (8, 3)
    assert s["delta_t_years"] == pytest.approx(1826 / 365.25)


# ---- degenerate tables: nothing unmatched, one side empty, both empty -------------------------

def _degenerate(keep_prev, keep_curr):
    prev, curr, wm, pipe, repairs = _runs()
    def keep(std, ids):  # the valve / weld rows stay, so each run keeps its run date
        return std[std.feature_id.isin(ids) | ~std.feature_type.isin(ANOMALY_TYPES)]
    prev, curr = keep(prev, keep_prev), keep(curr, keep_curr)
    return match_anomalies(prev, curr, wm, S, pipe, repairs)


@pytest.mark.parametrize("keep_prev, keep_curr, counts", [
    (["P0"], ["C0"], {"matched_1to1": 1, "matched_cluster": 0, "missing": 0, "new": 0}),
    (["P0", "P1", "P2"], ["C0", "C1"], {"matched_1to1": 1, "matched_cluster": 1, "missing": 0, "new": 0}),
    (["P0"], [], {"matched_1to1": 0, "matched_cluster": 0, "missing": 1, "new": 0}),
    ([], ["C0"], {"matched_1to1": 0, "matched_cluster": 0, "missing": 0, "new": 1}),
    ([], [], {"matched_1to1": 0, "matched_cluster": 0, "missing": 0, "new": 0}),
])
def test_degenerate_tables_keep_columns_and_dtypes(keep_prev, keep_curr, counts):
    ref = match_anomalies(*_runs()[:3], S, *_runs()[3:]).match_table
    res = _degenerate(keep_prev, keep_curr)
    t, s = res.match_table, res.summary
    assert list(t.columns) == COLUMNS
    assert t.dtypes.to_dict() == ref.dtypes.to_dict()
    assert s["counts_by_match_type"] == counts
    assert set(s["counts_by_confidence"]) == {"high", "medium", "low"}
    assert set(s["counts_by_hint"]) == {"in_repaired_section", "near_threshold", "unexplained", "rejected_by_review"}
    assert sum(s["counts_by_confidence"].values()) == counts["matched_1to1"] + counts["matched_cluster"]
    assert s["n_needs_review"] == int(t.needs_review.sum())
    assert (s["n_prev_anomalies"], s["n_curr_anomalies"]) == (len(keep_prev), len(keep_curr))


def test_absorbed_cluster_row_is_capped_at_medium_and_reviewed():
    """Two curr pits 2.4 in apart (separate interaction groups) both inside a prev vendor report's
    4 in box: the report absorbs the second one. Score 0 would be high; absorbed caps it at medium."""
    wm = _wm(("1:1", 0.0, 1000.0, 0.0, 1000.0))
    wm["in_repair"] = False
    prev = _frame([("valve", 10.0, "PV"), ("metal_loss", 100.0, "V", 40.0, 100, 4.0, "C1")], date(2021, 6, 15))
    curr = _frame([("valve", 10.0, "CV"), ("metal_loss", 100.0 - 0.1, "a", 38.0, 100, 0.3),
                   ("metal_loss", 100.0 + 0.1, "b", 41.0, 100, 0.3)], date(2026, 6, 15))
    pipe = pd.DataFrame({"start_ft": [0.0], "end_ft": [2000.0], "od_in": [12.75]})
    res = match_anomalies(prev, curr, wm, S, pipe)
    row = res.match_table.iloc[0]
    assert len(res.match_table) == 1 and row.match_type == "matched_cluster" and row.curr_feature_id == "a;b"
    assert row.score == pytest.approx(0.0) and bool(row.absorbed) is True
    assert row.confidence == "medium" and bool(row.needs_review) is True
    assert res.summary["n_absorbed"] == 1 and res.summary["n_needs_review"] == 1


# ---- the 1:1 pass must not take a pit that an opposite-run vendor report claims ----------------

def _p10(n2_clock):
    """Prev: vendor report V (4 x 2 in) and an individual pit n 2.4 in upstream. Curr: pits s1, s2
    inside V's box and n2 beside n. Before withholding, n took s1 one-to-one (0.6 in away)."""
    wm = _wm(("1:1", 0.0, 1000.0, 0.0, 1000.0))
    wm["in_repair"] = False
    ml = "metal_loss"
    prev = _frame([("valve", 10.0, "PV"), (ml, 50.0, "V", 40.0, 100, 4.0, "C1"), (ml, 49.80, "n", 30.0, 100, 0.5)],
                  date(2021, 6, 15))
    prev["width_in"] = [0.5, 2.0, 0.5]
    curr = _frame([("valve", 10.0, "CV"), (ml, 49.85, "s1", 40.0, 100, 0.5), (ml, 50.10, "s2", 38.0, 100, 0.5),
                   (ml, 49.78, "n2", 31.0, n2_clock, 0.5)], date(2026, 6, 15))
    curr["width_in"] = 0.5
    pipe = pd.DataFrame({"start_ft": [0.0], "end_ft": [2000.0], "od_in": [12.75]})
    t = match_anomalies(prev, curr, wm, S, pipe).match_table
    return {(r.match_type, r.prev_feature_id, r.curr_feature_id) for r in t.itertuples()}, t


def test_one_to_one_does_not_take_a_vendor_cluster_pit():
    """n2 at 144 min is outside V's interaction distance (circ edge gap 1.20 in) but a 1:1 candidate
    of n (44 min): n pairs with n2 and V takes both pits it holds."""
    rows, t = _p10(144)
    assert rows == {("matched_1to1", "n", "n2"), ("matched_cluster", "V", "s1;s2")}


def test_reviewer_p10_pits_stay_with_the_vendor_report():
    """As reported (n2 at 140 min): n2's circ edge gap to V's box is 0.975 in, within circ_spacing_in, so
    it is withheld too, and n interacts with V in its own run (0.15 in). All five go to the cluster
    pass and come out as one absorbed cluster for review, never s1 taken by n one-to-one."""
    rows, t = _p10(140)
    assert rows == {("matched_cluster", "n;V", "n2;s1;s2")}
    assert t.absorbed.iloc[0] and t.needs_review.iloc[0] and t.confidence.iloc[0] == "medium"
