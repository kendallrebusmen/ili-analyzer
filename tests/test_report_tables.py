from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ili import config
from ili.classify import CLASSIFICATION_COLUMNS
from ili.report_tables import imp_class_summary
from ili.report_data import (DIG_LIST_CATEGORIES, DIG_LIST_COLUMNS, backtest, config_rows, criteria_rows, dig_list,
                             largest_growth, match_stats, pending_reviews)

NAN = np.nan


def _cls(rows):
    """A classification table from partial rows; every other column is missing."""
    df = pd.DataFrame(rows)
    for c in CLASSIFICATION_COLUMNS:
        if c not in df:
            df[c] = None
    return df


def _inputs(cls_rows, rl=None, growth=None, curr_std=None):
    """Hand-built inputs for ``dig_list``: one match-table, growth, remaining-life and curr_std row per id."""
    ids = [r["match_id"] for r in cls_rows]
    match = pd.DataFrame({
        "match_id": ids, "prev_feature_id": [f"P{i}" for i in ids], "curr_log_dist_ft": 100.0,
        "curr_us_gw_id": "G1", "curr_dist_from_us_gw_ft": 3.0, "prev_depth_pct": 20.0, "curr_depth_pct": 30.0,
        "prev_length_in": 2.0, "curr_length_in": 3.0})
    growth = pd.DataFrame({"match_id": ids, "governing_mpy": 5.0, "governing_method": "raw"})
    rl = pd.DataFrame({"match_id": ids, "rl_yr": rl if rl is not None else [1.0] * len(ids),
                       "rl_governing": "burst", "failure_mode": "leak", "erf": 0.8, "fpr": 1.25,
                       "burst_next_psig": 1500.0, "safe_next_psig": 1080.0, "erf_next": 0.9, "fpr_next": 1.5})
    if curr_std is None:
        curr_std = pd.DataFrame({"feature_id": [r["curr_feature_id"] for r in cls_rows], "log_dist_ft": 100.0,
                                 "lat": 29.5, "lon": -95.5})
    return SimpleNamespace(classification=_cls(cls_rows), match_table=match, growth=growth, remaining_life=rl,
                           curr_std=curr_std)


def _row(mid, category, rl=None, due=None, **kw):
    return {"match_id": mid, "curr_feature_id": f"C{mid}", "governing_category": category, "due_date": due, **kw}


def test_dig_list_scope_and_columns():
    cats = ("evaluate", "imp_scheduled", "180_day", "60_day", "immediate")
    inputs = _inputs([_row(f"M{i}", c) for i, c in enumerate(cats)])
    out = dig_list(inputs)
    assert tuple(out.columns) == DIG_LIST_COLUMNS
    assert list(out.governing_category) == list(DIG_LIST_CATEGORIES)
    assert "evaluate" not in set(out.governing_category)
    assert out.lat.eq(29.5).all() and out.prev_feature_id.iloc[0] == "PM4"
    assert out.governing_mpy.eq(5.0).all() and out.rl_governing.eq("burst").all()


def test_dig_list_ranking():
    rows = [_row("M1", "180_day", due=date(2027, 1, 1), feature_type="dent"),  # dent: no RL
            _row("M2", "180_day", due=date(2027, 3, 1)),  # RL NaN, latest due
            _row("M3", "180_day", due=date(2027, 1, 1)),  # RL 2.0
            _row("M4", "immediate", due=date(2027, 1, 1)),
            _row("M5", "180_day", due=date(2027, 1, 1)),  # RL 0.5
            _row("M6", "180_day", due=date(2026, 12, 1)),  # RL NaN, earlier due
            _row("M7", "60_day", due=date(2027, 1, 1))]
    inputs = _inputs(rows, rl=[NAN, NAN, 2.0, 9.0, 0.5, NAN, 9.0])
    out = dig_list(inputs)
    assert list(out.match_id) == ["M4", "M7", "M5", "M3", "M6", "M1", "M2"]
    assert out.rl_yr.isna().tolist() == [False, False, False, False, True, True, True]


def test_dig_list_due_date_none_sorts_last():
    rows = [_row("M1", "60_day", due=None), _row("M2", "60_day", due=date(2027, 1, 1))]
    out = dig_list(_inputs(rows, rl=[NAN, NAN]))
    assert list(out.match_id) == ["M2", "M1"]


def test_dig_list_cluster_takes_the_member_nearest_its_position():
    rows = [_row("M1", "60_day"), _row("M2", "180_day")]
    rows[0]["curr_feature_id"] = "B-1;B-2;B-3"  # a matched cluster: ";"-joined member ids
    rows[1]["curr_feature_id"] = "B-4"
    curr = pd.DataFrame({"feature_id": ["B-1", "B-2", "B-3", "B-4"], "log_dist_ft": [99.0, 100.2, 100.9, 500.0],
                         "lat": [29.1, 29.2, 29.3, 29.4], "lon": [-95.1, -95.2, -95.3, -95.4]})
    out = dig_list(_inputs(rows, curr_std=curr)).set_index("match_id")  # match table: curr_log_dist_ft 100
    assert (out.loc["M1", "lat"], out.loc["M1", "lon"]) == (29.2, -95.2)
    assert (out.loc["M2", "lat"], out.loc["M2", "lon"]) == (29.4, -95.4)
    assert out.loc["M1", "curr_feature_id"] == "B-1;B-2;B-3"


def test_pending_reviews_counts_needs_review_pending_only():
    mt = pd.DataFrame({
        "match_id": ["M1", "M2", "M3", "M4"],
        "needs_review": [True, True, False, True],
        "review_status": ["pending", "accepted", "pending", "pending"]})
    out = pending_reviews(mt)
    assert list(out.match_id) == ["M1", "M4"]


def test_match_stats_by_type_and_confidence():
    mt = pd.DataFrame({
        "match_type": ["matched", "matched", "matched", "new"],
        "confidence": ["high", "high", "low", None]})
    out = match_stats(mt)
    assert out.loc["matched", "high"] == 2 and out.loc["matched", "low"] == 1
    assert out.loc["matched", "all"] == 3 and out.loc["new", "all"] == 1


def test_largest_growth_orders_and_breaks_ties_by_match_id():
    g = pd.DataFrame({"match_id": ["M3", "M1", "M2", "M4"], "governing_mpy": [5.0, 5.0, 9.0, NAN]})
    out = largest_growth(SimpleNamespace(growth=g), 3)
    assert list(out.match_id) == ["M2", "M1", "M3"]


def _settings(tol=10.0):
    s = config.load_settings(config.DEFAULT_SETTINGS)
    return s.model_copy(update={"tool": s.tool.model_copy(update={"depth_tolerance_pct_wt": tol})})


def test_backtest_within_tolerance_edges():
    curr = pd.DataFrame({"feature_id": ["A", "B", "C"], "depth_pct": [50.0, 50.0, 50.0]})
    digs = pd.DataFrame({"feature_id": ["A", "B", "C", "Z"], "field_depth_pct": [40.01, 40.0, 39.99, 10.0],
                         "field_length_in": 1.0, "dig_date": "2026-01-01"})
    res = backtest(digs, curr, _settings(10.0))
    assert list(res.table.within_tolerance) == [True, True, False]
    assert res.n_within == 2 and abs(res.pct_within - 200 / 3) < 1e-9
    assert res.unmatched_ids == ["Z"]
    assert backtest(None, curr, _settings()) is None
    assert res.no_depth_ids == []


def test_backtest_excludes_digs_without_both_depths():
    curr = pd.DataFrame({"feature_id": ["A", "B", "C"], "depth_pct": [50.0, NAN, 50.0]})
    digs = pd.DataFrame({"feature_id": ["A", "B", "C"], "field_depth_pct": [45.0, 40.0, NAN],
                         "field_length_in": 1.0, "dig_date": "2026-01-01"})
    res = backtest(digs, curr, _settings(10.0))
    assert list(res.table.feature_id) == ["A"]  # B: no ILI depth, C: no field depth
    assert res.n_within == 1 and res.pct_within == 100.0
    assert res.no_depth_ids == ["B", "C"] and res.unmatched_ids == []


def test_config_rows_flags_placeholders():
    s = config.load_settings(config.DEFAULT_SETTINGS)
    rows = config_rows(s, config.DEFAULT_SETTINGS).set_index("key")
    assert rows.loc["tool.depth_tolerance_pct_wt", "placeholder"]
    assert not rows.loc["remaining_life.horizon_yr", "placeholder"]
    assert rows.loc["growth.governing.matched", "value"] == ["segment_percentile", "tolerance_adjusted"]
    assert rows.loc["matching.interaction_rule.axial_spacing_in", "placeholder"]  # flagged on the parent's line
    assert not rows.loc["classification.due_days.60_day", "placeholder"]
    assert rows.loc["growth.default_rate_mpy", "value"] is None

    def count(node):
        return sum(count(v) for v in node.values()) if isinstance(node, dict) and node else 1
    assert len(rows) == count(s.model_dump())


def test_criteria_rows_in_file_order():
    crit = config.load_repair_criteria(config.DEFAULT_CRITERIA)
    out = criteria_rows(crit)
    assert list(out.id) == [r.id for r in crit.rules]
    assert list(out.columns) == ["id", "category", "applies_to", "feature_types", "condition", "requires_review",
                                 "description"]
    assert out.feature_types.map(lambda v: isinstance(v, str)).all()


def test_dig_list_has_erf_fpr():
    out = dig_list(_inputs([_row("M1", "60_day")]))
    i = list(out.columns).index
    assert i("erf") == i("mop_local") + 1 and i("fpr") == i("erf") + 1
    assert i("failure_mode") == i("rl_governing") + 1
    assert i("burst_next_psig") > i("fpr") and i("fpr_next") == i("erf_next") + 1
    row = out.iloc[0]
    assert (row.erf, row.fpr, row.erf_next, row.fpr_next, row.failure_mode) == (0.8, 1.25, 0.9, 1.5, "leak")


def _imp(rows, repairs, positions=None):
    """``imp_class_summary`` over hand-built rows: ``rows`` are (category, rl_yr, rl_upper_bound_yr)."""
    ids = [f"M{i}" for i in range(len(rows))]
    cls = _cls([{"match_id": m, "curr_feature_id": f"C{m}", "governing_category": c} for m, (c, _, _) in zip(ids, rows)])
    pos = positions if positions is not None else [100.0] * len(rows)
    match = pd.DataFrame({"match_id": ids, "curr_log_dist_ft": pos})
    rl = pd.DataFrame({"match_id": ids, "rl_yr": [r[1] for r in rows], "rl_upper_bound_yr": [r[2] for r in rows]})
    return imp_class_summary(cls, match, rl, repairs, config.load_settings())


def test_imp_class_summary_rows():
    rows = [("immediate", 0.5, NAN), ("60_day", NAN, NAN), ("180_day", 2.0, NAN),
            ("imp_scheduled", 4.0, NAN),   # YTC by rl_yr
            ("evaluate", NAN, 3.0),        # no category, NaN RL, bound 3: YTC < 5
            ("evaluate", 6.0, 1.0),        # rl_yr wins over a lower bound: None
            ("evaluate", NAN, NAN),        # neither: None, not evaluated
            ("evaluate", np.inf, NAN)]     # no failure within the horizon: None, evaluated
    repairs = pd.DataFrame({"start_ft": [90.0], "end_ft": [110.0]})
    out = _imp(rows, repairs, positions=[100.0, 500.0, 500.0, 90.0, 110.0, 500.0, 500.0, 500.0])
    assert list(out.columns) == ["imp_class", "n_classified", "n_previously_repaired", "n_unrepaired",
                                 "n_rl_not_evaluated"]
    assert list(out.imp_class) == ["Immediate", "60-day", "180-day", "YTC < 5 yrs", "None"]
    assert list(out.n_classified) == [1, 1, 1, 2, 3]
    assert list(out.n_previously_repaired) == [1, 0, 0, 2, 0]  # span ends are inclusive
    assert list(out.n_unrepaired) == [0, 1, 1, 0, 3]
    assert list(out.n_rl_not_evaluated) == [0, 1, 0, 0, 1]
    assert out.n_classified.sum() == len(rows)


def test_imp_class_summary_no_repairs_file():
    out = _imp([("immediate", 0.5, NAN), ("evaluate", NAN, NAN)], None)
    assert list(out.n_classified) == [1, 0, 0, 0, 1]
    assert out.n_previously_repaired.isna().all() and out.n_unrepaired.isna().all()


def test_imp_class_label_follows_the_setting(tmp_path):
    settings = config.load_settings()
    settings.report.ytc_flag_years = 7.5
    cls = _cls([{"match_id": "M0", "curr_feature_id": "C0", "governing_category": "evaluate"}])
    out = imp_class_summary(cls, pd.DataFrame({"match_id": ["M0"], "curr_log_dist_ft": [1.0]}),
                            pd.DataFrame({"match_id": ["M0"], "rl_yr": [7.0], "rl_upper_bound_yr": [NAN]}),
                            None, settings)
    assert out.imp_class.iloc[3] == "YTC < 7.5 yrs" and out.n_classified.iloc[3] == 1
