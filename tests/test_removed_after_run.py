"""Features removed from the pipeline after the inspection: the ``removed_after_run`` manual flag.

A removed feature keeps its rule results (traceability) but its governing category is ``removed``: it has no
due date, needs no review, is off the dig list and counts as previously repaired in the IMP class summary."""

import numpy as np
import pandas as pd
import pytest
from test_classify import base, go, rec
from test_report_tables import _cls, _imp, _inputs, _row
from test_review_flags import _categories, _flags

from ili import config
from ili.category_change import _moved_more_severe
from ili.classify import (CATEGORY_PRIORITY, GOVERNING_REMOVED, REMOVED_ACTION_NOTE, REMOVED_CATEGORY,
                          FEATURE_VARIABLE_COLUMNS)
from ili.config import load_repair_criteria, load_settings
from ili.report_data import dig_list
from ili.report_tables import imp_class_rows
from ili.review import FLAG_NAMES
from test_classify_vars import mrow

NAN = np.nan


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def criteria():
    return load_repair_criteria()


def test_flag_name_is_known():
    assert "removed_after_run" in FLAG_NAMES
    assert "removed_after_run" in FEATURE_VARIABLE_COLUMNS
    assert REMOVED_CATEGORY == "removed" and REMOVED_CATEGORY not in CATEGORY_PRIORITY


def test_removed_overrides_every_category(settings, criteria):
    rows = [base("M1", dig_required_by_remaining_life=True, removed_after_run=True),
            base("M2", manual_immediate=True, removed_after_run=True),
            base("M3", seam_interaction=None, removed_after_run=True),       # not evaluable: no worst case left
            base("M4", dig_required_by_remaining_life=True, removed_after_run=False)]
    res = go(rows, settings, criteria)
    for mid in ("M1", "M2", "M3"):
        r = rec(res, mid)
        assert r["governing_category"] == "removed" and r["governing_rule"] == GOVERNING_REMOVED
        assert r["due_date"] is None and r["due_date_basis"] is None
        assert r["worst_case_category"] == "removed" and r["worst_case_rule"] == GOVERNING_REMOVED
        assert r["worst_case_due_date"] is None
        assert r["requires_review"] is False and r["review_reasons"] is None
        assert r["action_note"] == REMOVED_ACTION_NOTE
    assert rec(res, "M2")["triggered_rules"] == "IMM-5"  # the rule results stay on record
    assert rec(res, "M4")["governing_category"] == "imp_scheduled"
    s = res.summary
    assert s["counts_by_category"]["removed"] == 3 and s["counts_by_category"]["imp_scheduled"] == 1
    assert sum(s["counts_by_category"].values()) == 4


def test_flag_reaches_feature_variables(settings, tmp_path):
    rows = [mrow("M1", pos=1020.0, depth_pct=10.0), mrow("M2", pos=1100.0, depth_pct=10.0)]
    flags = _flags(tmp_path / "a", ("B-M1", "removed_after_run", True))
    v, cat, _ = _categories(rows, flags, settings)
    assert list(v.removed_after_run) == [True, False]
    assert cat == {"M1": "removed", "M2": "evaluate"}
    v, cat, _ = _categories(rows, None, settings)
    assert list(v.removed_after_run) == [False, False] and cat["M1"] == "evaluate"
    cleared = _flags(tmp_path / "b", ("B-M1", "removed_after_run", True), ("B-M1", "removed_after_run", False))
    assert _categories(rows, cleared, settings)[1]["M1"] == "evaluate"


def test_removed_is_off_the_dig_list():
    rows = [_row("M1", "imp_scheduled"), _row("M2", "removed"), _row("M3", "60_day")]
    out = dig_list(_inputs(rows, rl=[1.0, 2.0, 3.0]))
    assert list(out.match_id) == ["M3", "M1"]


def test_removed_counts_as_previously_repaired():
    ids = ["M0", "M1", "M2"]
    cls = _cls([{"match_id": m, "curr_feature_id": f"C{m}", "governing_category": c, "feature_type": "metal_loss"}
                for m, c in zip(ids, ["removed", "evaluate", "imp_scheduled"])])
    match = pd.DataFrame({"match_id": ids, "curr_log_dist_ft": [100.0, 200.0, 300.0]})
    rl = pd.DataFrame({"match_id": ids, "rl_yr": [1.0, NAN, 2.0], "rl_upper_bound_yr": [NAN, NAN, NAN]})
    settings = config.load_settings()
    out = imp_class_rows(cls, match, rl, None, settings)
    assert out.imp_class.tolist() == ["YTC < 5 yrs", "None", "YTC < 5 yrs"]  # class from remaining life
    assert out.previously_repaired.tolist() == [True, None, None]  # no repairs table: only the removal is known
    rows = [("removed", 1.0, NAN), ("imp_scheduled", 2.0, NAN), ("evaluate", NAN, NAN)]
    summ = _imp(rows, None).set_index("imp_class")
    assert summ.loc["YTC < 5 yrs", "n_classified"] == 2
    assert summ.loc["YTC < 5 yrs", "n_previously_repaired"] == 1 and summ.loc["YTC < 5 yrs", "n_unrepaired"] == 1
    assert summ.loc["None", "n_previously_repaired"] == 0


def test_category_change_ignores_removed():
    assert _moved_more_severe("evaluate", "removed") is False
    assert _moved_more_severe("removed", "immediate") is False


def test_removed_features_table_carries_the_flag_note():
    from ili.report_data import REMOVED_COLUMNS, removed_features
    rows = [_row("M1", "removed", feature_type="metal_loss", depth_pct_reported=54.0, triggered_rules=None),
            _row("M2", "evaluate"), _row("M3", "removed", feature_type="dent")]
    rows[2]["curr_feature_id"] = "CM3a;CM3b"
    inputs = _inputs(rows)
    cols = ["flag_id", "curr_feature_id", "flag", "value", "reviewer", "decided_utc", "note"]
    inputs.flags = pd.DataFrame([
        ["F000001", "CM1", "removed_after_run", "true", "Pat", "2026-10-29T10:00:00Z", "first"],
        ["F000002", "CM1", "removed_after_run", "true", "Lee", "2026-10-29T11:00:00Z", "spool cut out"],
        ["F000003", "CM3b", "removed_after_run", "true", "Lee", "2026-10-29T11:00:00Z", "spool cut out"],
        ["F000004", "CM2", "manual_immediate", "true", "Lee", "2026-10-29T11:00:00Z", "other flag"]], columns=cols)
    out = removed_features(inputs)
    assert tuple(out.columns) == REMOVED_COLUMNS
    assert list(out.match_id) == ["M1", "M3"]
    assert list(out.removal_note) == ["spool cut out", "spool cut out"]  # the latest row decides
    assert list(out.flagged_by) == ["Lee", "Lee"] and list(out.flag_id) == ["F000002", "F000003"]
    assert out.curr_log_dist_ft.eq(100.0).all()
    none = _inputs([_row("M2", "evaluate")])
    none.flags = pd.DataFrame(columns=cols)
    empty = removed_features(none)
    assert empty.empty and tuple(empty.columns) == REMOVED_COLUMNS
