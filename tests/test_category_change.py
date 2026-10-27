"""Previous-run classification and category changes (M8 Task 1)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from test_classify_cli import _classify_args
from test_classify_cli import _through_rl
from test_classify_vars import CROSSINGS, CURR_STD, PIPE, RL_EMPTY, SEGS, mrow, rlrow, table

from ili import classify, ingest
from ili.category_change import (RULE_CATEGORIES, category_changes, previous_pressures, previous_side)
from ili.classify import CLASSIFICATION_COLUMNS, feature_variables, load_classification_table
from ili.config import load_repair_criteria, load_settings
from ili.strength import assessed_depth_in

DATE = pd.Timestamp("2021-06-15").date()
CURR_DATE = pd.Timestamp("2026-06-15").date()
MOP = pd.DataFrame({"start_ft": [0.0], "end_ft": [10_000.0], "mop_psig": [100.0]})
OD = 12.75


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def criteria():
    return load_repair_criteria()


def pair(mid, ftype="metal_loss", mtype="matched_1to1", prev_ftype=None, prev=None, curr=None, pos=1020.0):
    """A match-table row with both sides: the current side from ``curr``, the previous side from ``prev``
    (dicts of field names without the side prefix)."""
    r = mrow(mid, ftype, pos=pos, mtype=mtype, **(curr or {}))
    side = dict(feature_type=prev_ftype or ftype, wt_nom_in=0.25, depth_pct=float("nan"),
                dent_depth_in=float("nan"), length_in=2.0, width_in=1.0, clock_min=360, seam_interaction=False,
                dent_metal_loss=None)
    side.update(prev or {})
    r.update({f"prev_{k}": v for k, v in side.items()}, prev_feature_id=f"A-{mid}", n_prev=1)
    return r


def mtable(rows):
    mt = table(rows)
    mt["prev_clock_min"] = mt.prev_clock_min.astype("Int64")
    return mt


def run(rows, rl_rows, settings, criteria):
    """(match table, M6 table, current classification, CategoryChangeResult) of hand-built rows."""
    mt = mtable(rows)
    rl = pd.DataFrame(rl_rows) if rl_rows else RL_EMPTY
    curr = classify.classify(feature_variables(mt, rl, CURR_STD, PIPE, SEGS, CROSSINGS, settings), criteria,
                             settings, CURR_DATE).table
    return curr, category_changes(mt, rl, curr, CURR_STD, PIPE, MOP, SEGS, CROSSINGS, settings, criteria, DATE)


def test_previous_side_swaps_columns_and_keeps_position():
    rows = mtable([
        pair("M1", curr=dict(depth_pct=80.0, length_in=5.0, clock_min=100), prev=dict(depth_pct=50.0, length_in=3.0,
                                                                                         clock_min=200)),
        pair("M2", "dent", mtype="matched_cluster", pos=2000.0, curr=dict(dent_depth_in=0.8),
             prev=dict(dent_depth_in=0.2, dent_metal_loss=True)),
        mrow("M3", mtype="new"),
        {**mrow("M4", mtype="missing"), "prev_feature_id": "A-M4"},
    ])
    rows.loc[0, ["prev_log_dist_ft", "curr_log_dist_ft"]] = [1000.0, 1020.0]
    out = previous_side(rows)
    assert list(out.match_id) == ["M1", "M2"]  # matched rows only, in order
    m1 = out.iloc[0]
    assert (m1.curr_depth_pct, m1.curr_length_in, m1.curr_clock_min, m1.curr_feature_type) == (
        50.0, 3.0, 200, "metal_loss")
    assert m1.curr_wt_nom_in == 0.25 and m1.curr_seam_interaction is False
    assert m1.curr_log_dist_ft == 1020.0  # reference files are keyed to current-run log distance
    assert (m1.orig_prev_feature_id, m1.orig_curr_feature_id) == ("A-M1", "B-M1")
    assert (m1.curr_feature_id, m1.match_type, m1.needs_review) == ("A-M1", "matched_1to1", False)
    m2 = out.iloc[1]
    assert m2.curr_dent_depth_in == 0.2 and m2.curr_dent_metal_loss is True and m2.match_type == "matched_cluster"
    assert out.curr_clock_min.dtype == "Int64"
    assert rows.loc[0, "curr_depth_pct"] == 80.0  # the input is not changed


def test_previous_pressures_match_strength_reference_case(settings):
    s = settings.model_copy(update={"strength": settings.strength.model_copy(
        update={"method": "modified_b31g", "depth_basis": "reported_plus_tolerance", "length_basis": "reported"})})
    reported = 0.125 / 0.250 * 100 - s.tool.depth_tolerance_pct_wt  # assessed 0.125 in
    assert float(assessed_depth_in(reported, 0.250, s)) == pytest.approx(0.125)
    pipe = pd.DataFrame({"start_ft": [0.0], "end_ft": [5000.0], "od_in": [12.75], "smys_psi": [52_000.0]})
    mt = mtable([pair("M1", curr=dict(depth_pct=99.0), prev=dict(depth_pct=reported, length_in=4.0))])
    rl = pd.DataFrame([rlrow("M1", wt=0.250)])
    out = previous_pressures(previous_side(mt), rl, pipe, MOP, s)
    assert list(out.columns) == list(classify.RL_INPUT_COLUMNS) and len(out) == 1
    r = out.iloc[0]
    assert r.burst_psig == pytest.approx(1771.5, rel=1e-3)
    assert r.safe_psig == pytest.approx(r.burst_psig * s.strength.design_factor)
    assert r.assessed_dt == pytest.approx(0.5) and r.wt_in == 0.250 and r.mop_psig == 100.0
    assert r.dig_required_by_remaining_life is False and r.needs_review is False
    assert r.repair_by_date is None and r.review_reasons is None


def test_previous_pressures_wall_falls_back_to_previous_wall_and_skips_dents(settings):
    mt = mtable([pair("M1", prev=dict(depth_pct=40.0, wt_nom_in=0.30)),
                pair("M2", "dent", prev=dict(dent_depth_in=0.2)),
                pair("M3", prev=dict(depth_pct=float("nan")))])
    out = previous_pressures(previous_side(mt), RL_EMPTY, PIPE, MOP, settings)
    assert list(out.match_id) == ["M1", "M3"]  # one row per metal-loss row
    assert out.wt_in.iloc[0] == 0.30  # no M6 row: the previous wall
    assert math.isnan(out.assessed_dt.iloc[1]) and math.isnan(out.burst_psig.iloc[1])  # no depth: unknown, not guessed


def test_prev_categories_use_prev_values(settings, criteria):
    rows = [
        pair("M1", curr=dict(depth_pct=75.0), prev=dict(depth_pct=60.0)),  # assessed 70 -> 85
        pair("M2", pos=1040.0, curr=dict(depth_pct=75.0), prev=dict(depth_pct=75.0)),  # 85 -> 85
        pair("M3", "dent", pos=1060.0, curr=dict(dent_depth_in=0.065 * OD, clock_min=0, dent_metal_loss=False),
             prev=dict(dent_depth_in=0.015 * OD, clock_min=0, dent_metal_loss=False)),  # 1.5 % -> 6.5 % OD, top
    ]
    rl = [rlrow("M1", assessed_dt=0.85, mop=100.0), rlrow("M2", assessed_dt=0.85, mop=100.0)]
    curr, result = run(rows, rl, settings, criteria)
    t = result.table.set_index("match_id")
    assert list(result.table.columns) == [
        "match_id", "match_type", "prev_feature_id", "curr_feature_id", "feature_type", "prev_depth_pct_assessed",
        "curr_depth_pct_assessed", "prev_p_fail", "curr_p_fail", "prev_rule_category", "prev_governing_rule",
        "prev_triggered_rules", "prev_rules_not_evaluable", "curr_rule_category", "curr_governing_category",
        "curr_governing_rule", "moved_more_severe"]
    assert list(result.table.match_id) == ["M1", "M2", "M3"]
    m1 = t.loc["M1"]
    assert (m1.prev_rule_category, m1.curr_rule_category, m1.moved_more_severe) == ("evaluate", "immediate", True)
    assert m1.prev_depth_pct_assessed == pytest.approx(70.0) and m1.curr_depth_pct_assessed == pytest.approx(85.0)
    assert m1.prev_governing_rule == "EVALUATE" and m1.curr_governing_rule == "IMM-1"
    assert (m1.prev_feature_id, m1.curr_feature_id, m1.match_type) == ("A-M1", "B-M1", "matched_1to1")
    assert m1.prev_p_fail > 100.0 and m1.curr_p_fail == 1800.0  # prev strength computed, curr from M6
    m2 = t.loc["M2"]
    assert (m2.prev_rule_category, m2.curr_rule_category, m2.moved_more_severe) == ("immediate", "immediate", False)
    m3 = t.loc["M3"]
    assert (m3.prev_rule_category, m3.curr_rule_category, m3.moved_more_severe) == ("evaluate", "immediate", True)
    assert m3.curr_governing_rule == "IMM-4" and m3.feature_type == "dent"
    assert math.isnan(m3.prev_p_fail)  # dents carry no pressures
    assert list(result.previous.columns) == CLASSIFICATION_COLUMNS
    assert list(result.previous.match_id) == ["M1", "M2", "M3"]
    assert result.previous.set_index("match_id").governing_category.to_dict() == {
        "M1": "evaluate", "M2": "immediate", "M3": "evaluate"}


def test_less_severe_now_is_not_moved(settings, criteria):
    rows = [pair("M1", curr=dict(depth_pct=20.0), prev=dict(depth_pct=75.0))]
    _, result = run(rows, [rlrow("M1", assessed_dt=0.30, mop=100.0)], settings, criteria)
    r = result.table.iloc[0]
    assert (r.prev_rule_category, r.curr_rule_category, r.moved_more_severe) == ("immediate", "evaluate", False)


def test_changed_feature_type_keeps_current_type(settings, criteria):
    rows = [pair("M1", "metal_loss", prev_ftype="dent", curr=dict(depth_pct=75.0), prev=dict(dent_depth_in=0.01))]
    _, result = run(rows, [rlrow("M1", assessed_dt=0.85, mop=100.0)], settings, criteria)
    r = result.table.iloc[0]
    assert r.feature_type == "metal_loss" and r.curr_rule_category == "immediate"
    assert r.prev_rule_category == "evaluate" and r.moved_more_severe


def test_imp_scheduled_compares_as_evaluate(settings, criteria):
    rows = [pair("M1", curr=dict(depth_pct=20.0), prev=dict(depth_pct=20.0))]
    rl = [rlrow("M1", assessed_dt=0.30, mop=100.0, dig=True)]
    curr, result = run(rows, rl, settings, criteria)
    assert curr.governing_category.iloc[0] == "imp_scheduled"
    r = result.table.iloc[0]
    assert (r.curr_governing_category, r.curr_rule_category, r.prev_rule_category) == ("imp_scheduled", "evaluate",
                                                                                      "evaluate")
    assert not r.moved_more_severe
    assert "imp_scheduled" not in RULE_CATEGORIES


def test_new_and_missing_rows_absent(settings, criteria):
    rows = [pair("M1", curr=dict(depth_pct=20.0), prev=dict(depth_pct=20.0)),
            mrow("M2", pos=1040.0, depth_pct=75.0),  # new
            {**mrow("M3", mtype="missing"), "prev_feature_id": "A-M3", "prev_feature_type": "metal_loss",
             "prev_log_dist_ft": 1060.0, "prev_depth_pct": 75.0}]
    rl = [rlrow("M1", assessed_dt=0.30, mop=100.0), rlrow("M2", assessed_dt=0.85, mop=100.0)]
    curr, result = run(rows, rl, settings, criteria)
    assert list(curr.match_id) == ["M1", "M2"]
    assert list(result.table.match_id) == ["M1"] and list(result.previous.match_id) == ["M1"]


def test_load_classification_table_round_trip(settings, criteria, tmp_path):
    rows = [pair("M1", curr=dict(depth_pct=75.0), prev=dict(depth_pct=60.0))]
    curr, _ = run(rows, [rlrow("M1", assessed_dt=0.85, mop=100.0)], settings, criteria)
    path = tmp_path / "classification.csv"
    curr.to_csv(path, index=False, lineterminator="\n")
    back = load_classification_table(path)
    assert list(back.columns) == CLASSIFICATION_COLUMNS
    r = back.iloc[0]
    assert r.match_id == "M1" and r.curr_feature_id == "B-M1" and r.segment_id == "SEG-01"
    assert r.could_affect_hca is True and r.requires_review in (True, False) and r.seam_interaction is None
    assert r.discovery_date == curr.discovery_date.iloc[0] and r.due_date == curr.due_date.iloc[0]
    assert back.clock_min.dtype == "Int64" and r.p_fail == 1800.0 and r.review_reasons == curr.review_reasons.iloc[0]
    assert r.governing_category == "immediate"
    with pytest.raises(ValueError, match="missing column"):
        path.write_text("match_id\nM1\n", encoding="utf-8")
        load_classification_table(path)


def test_synthetic_moves_exist(synth, tmp_path, capsys):
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    ref = data / "reference"
    settings, criteria = load_settings(), load_repair_criteria()
    prev_std = ingest.load_standardized(std / "2021_vendorA.csv")
    curr_std = ingest.load_standardized(std / "2026_vendorB.csv")
    from ili.match import load_match_table
    from ili.remaining_life import load_remaining_life_table
    mt = load_match_table(folder / "match_table.csv")
    rl = load_remaining_life_table(folder / "remaining_life.csv")
    classification = load_classification_table(folder / "classification.csv")
    result = category_changes(
        mt, rl, classification, curr_std, pd.read_csv(ref / "pipe_properties.csv"),
        pd.read_csv(ref / "mop_profile.csv"),
        pd.read_csv(ref / "hca_segments.csv", dtype={"segment_id": str}), pd.read_csv(ref / "foreign_crossings.csv"),
        settings, criteria, prev_std.run_date.dropna().iloc[0])
    t = result.table
    matched = classification[classification.match_id.isin(mt.match_id[mt.match_type.isin(
        ["matched_1to1", "matched_cluster"])])]
    assert len(t) == len(matched) > 0 and list(t.match_id) == list(matched.match_id)
    assert t.moved_more_severe.sum() >= 1
    rank = {c: k for k, c in enumerate(RULE_CATEGORIES)}
    moved = t[t.moved_more_severe]
    assert (moved.curr_rule_category.map(rank) < moved.prev_rule_category.map(rank)).all()
    assert set(t.curr_rule_category) | set(t.prev_rule_category) <= set(RULE_CATEGORIES)
