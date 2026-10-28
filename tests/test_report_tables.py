from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from ili import config
from ili.classify import CLASSIFICATION_COLUMNS
from ili.report_tables import (_location_notes, anomaly_distribution, anomaly_distribution_table, depth_bins,
                               dent_bins, imp_class_summary, joint_ml_counts, largest_joints)
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


def _imp(rows, repairs, positions=None, types=None):
    """``imp_class_summary`` over hand-built rows: ``rows`` are (category, rl_yr, rl_upper_bound_yr)."""
    ids = [f"M{i}" for i in range(len(rows))]
    types = types or ["metal_loss"] * len(rows)
    cls = _cls([{"match_id": m, "curr_feature_id": f"C{m}", "governing_category": c, "feature_type": t}
                for m, (c, _, _), t in zip(ids, rows, types)])
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
    assert list(out.n_rl_not_evaluated) == [0, 1, 0, 0, 1]  # all metal loss here
    assert out.n_classified.sum() == len(rows)


def test_imp_class_not_evaluated_counts_metal_loss_only():
    rows = [("immediate", NAN, NAN), ("evaluate", NAN, NAN), ("evaluate", NAN, NAN), ("60_day", NAN, NAN)]
    out = _imp(rows, None, types=["metal_loss", "metal_loss", "dent", "gouge"])
    assert list(out.n_classified) == [1, 1, 0, 0, 2]
    assert list(out.n_rl_not_evaluated) == [1, 0, 0, 0, 1]  # the dent and the gouge are never counted


def test_imp_class_repairs_without_span_columns_is_error():
    with pytest.raises(ValueError, match="start_ft"):
        _imp([("immediate", 1.0, NAN)], pd.DataFrame({"repair_type": ["sleeve"]}))


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


# --- depth bins and dent bins (spec 6.10) ---------------------------------------------------------------

def _std(rows, run_id="R1"):
    """A standardized run from partial rows: a metal-loss feature at mid-joint unless said otherwise."""
    base = {"run_id": run_id, "feature_type": "metal_loss", "surface": "EXT", "ml_origin": "corrosion",
            "depth_pct": 15.0, "dent_depth_in": NAN, "length_in": 2.0, "clock_min": 360, "seam_interaction": False,
            "log_dist_ft": 50.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _count(table, location, label, column="total"):
    row = table[(table.location == location) & (table.bin == label)]
    assert len(row) == 1
    return int(row[column].iloc[0])


def test_depth_bin_edges():
    std = _std([{"depth_pct": d} for d in (9.99, 10.0, 79.99, 80.0)])
    out = depth_bins(std, config.load_settings())
    assert [_count(out, "all", b) for b in ("0-10", "10-20", "70-80", ">=80")] == [1, 1, 1, 1]
    assert out[out.location == "all"].total.sum() == 4
    assert list(out[out.location == "all"].bin) == ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70",
                                                    "70-80", ">=80"]


def test_depth_bin_columns():
    std = _std([{"surface": "EXT"}, {"surface": "INT", "ml_origin": "unknown"},
                {"surface": "EXT", "ml_origin": "manufacturing"}, {"surface": "MID"}, {"surface": "UNK"}])
    out = depth_bins(std, config.load_settings())
    row = out[(out.location == "all") & (out.bin == "10-20")].iloc[0]
    assert (row.ext_corrosion, row.int_corrosion, row.manufacturing, row.total) == (1, 1, 1, 5)
    assert row.run_id == "R1"


def test_depth_bins_only_count_metal_loss_with_a_depth():
    std = _std([{"depth_pct": NAN}, {"feature_type": "dent"}, {}])
    assert depth_bins(std, config.load_settings()).query("location == 'all'").total.sum() == 1


def test_girth_weld_table_not_evaluated_note():
    settings = config.load_settings()
    assert settings.classification.girth_weld_proximity_in is None
    welds = _std([{"feature_type": "girth_weld", "log_dist_ft": 0.0}, {"feature_type": "girth_weld", "log_dist_ft": 100.0}])
    std = pd.concat([welds, _std([{"log_dist_ft": 50.0}, {"log_dist_ft": 0.1}])], ignore_index=True)
    assert depth_bins(std, settings).query("location == 'girth_weld'").total.sum() == 0
    gw = depth_bins(std, settings).query("location == 'girth_weld'")
    assert gw.n_location_unknown.eq(2).all() and depth_bins(std, settings).query("location == 'all'").n_location_unknown.eq(0).all()
    notes = _location_notes(depth_bins(std, settings), dent_bins(std, np.full(len(std), 10.0), settings), settings)
    assert any(n.table == "depth" and n.location == "girth_weld"
               and "classification.girth_weld_proximity_in is not set" in n.text and "(2 " in n.text for n in notes)


def test_girth_weld_and_seam_columns_with_settings():
    settings = config.load_settings()
    settings.classification.girth_weld_proximity_in = 6.0
    settings.classification.seam_clock_min = 360.0
    settings.classification.seam_clock_tol_min = 30.0
    welds = _std([{"feature_type": "girth_weld", "log_dist_ft": 0.0}, {"feature_type": "girth_weld", "log_dist_ft": 100.0}])
    std = pd.concat([welds, _std([{"log_dist_ft": 0.2, "clock_min": 360}, {"log_dist_ft": 50.0, "clock_min": 100}])],
                    ignore_index=True)
    out = depth_bins(std, settings)
    assert out.query("location == 'girth_weld'").total.sum() == 1
    assert out.query("location == 'long_seam'").total.sum() == 1
    assert out.query("location == 'all'").total.sum() == 2
    assert depth_bins(std, settings).n_location_unknown.eq(0).all()
    assert _location_notes(depth_bins(std, settings), dent_bins(std, np.full(len(std), 10.0), settings), settings) == []


def test_dent_bin_edges():
    std = _std([{"feature_type": "dent", "dent_depth_in": d, "depth_pct": NAN} for d in (2.0, 2.01, 6.0, 6.01, 0.0)])
    out = dent_bins(std, np.full(5, 100.0), config.load_settings())
    assert list(out.bin) == ["0-2", "2-6", ">6"]
    assert list(out["all"]) == [2, 2, 1]
    assert list(out.columns) == ["run_id", "bin", "all", "girth_weld", "long_seam", "n_girth_weld_unknown",
                                 "n_long_seam_unknown"]


def test_dent_bin_edge_survives_float_division():
    """0.9 in on a 15 in OD is 6.000000000000001 % unrounded: it belongs in 2-6, not >6."""
    assert 0.9 / 15 * 100 > 6.0
    std = _std([{"feature_type": "dent", "dent_depth_in": 0.9, "depth_pct": NAN}])
    assert list(dent_bins(std, np.array([15.0]), config.load_settings())["all"]) == [0, 1, 0]


def test_depth_bin_edge_is_rounded():
    std = _std([{"depth_pct": 20.0 - 1e-12}])
    assert _count(depth_bins(std, config.load_settings()), "all", "20-30") == 1


def test_dent_bins_count_unknown_locations():
    std = _std([{"feature_type": "dent", "dent_depth_in": 0.1, "depth_pct": NAN}] * 2)
    out = dent_bins(std, np.full(2, 10.0), config.load_settings())
    assert out.n_girth_weld_unknown.eq(2).all() and out.girth_weld.sum() == 0


def test_dent_od_lookup_error_names_the_run():
    prev = _std([{"feature_type": "dent", "dent_depth_in": 0.2, "log_dist_ft": 120.0, "depth_pct": NAN}], "P")
    curr = _std([{"feature_type": "metal_loss"}], "C")
    weld_map = pd.DataFrame({"match_type": ["1:1"], "prev_start_ft": [0.0], "prev_length_ft": [200.0],
                             "curr_start_ft": [0.0], "curr_length_ft": [200.0]})
    pipe = pd.DataFrame({"start_ft": [5000.0], "end_ft": [6000.0], "od_in": [10.0]})
    inputs = SimpleNamespace(prev_std=prev, curr_std=curr, weld_map=weld_map, pipe=pipe,
                             settings=config.load_settings())
    with pytest.raises(ValueError, match="dent-bin OD lookup failed for run P"):
        anomaly_distribution(inputs)


def test_dent_bins_skip_unknown_od():
    std = _std([{"feature_type": "dent", "dent_depth_in": 0.1, "depth_pct": NAN}] * 2)
    out = dent_bins(std, np.array([10.0, NAN]), config.load_settings())
    assert out["all"].sum() == 1
    with pytest.raises(ValueError, match="od_in"):
        dent_bins(std, np.array([10.0]), config.load_settings())


def test_anomaly_distribution_maps_previous_dents_to_the_current_run():
    """Reference files are keyed to current-run distance: a previous-run dent at 120 ft is at 220 ft in the
    current run, where the OD is 5 in (not 10 in), so 0.2 in is 4 % OD."""
    settings = config.load_settings()
    prev = _std([{"feature_type": "dent", "dent_depth_in": 0.2, "log_dist_ft": 120.0, "depth_pct": NAN}], "P")
    curr = _std([{"feature_type": "dent", "dent_depth_in": 0.2, "log_dist_ft": 120.0, "depth_pct": NAN}], "C")
    weld_map = pd.DataFrame({"match_type": ["1:1", "1:1"], "prev_start_ft": [0.0, 100.0],
                             "prev_length_ft": [100.0, 100.0], "curr_start_ft": [0.0, 200.0],
                             "curr_length_ft": [100.0, 100.0]})
    pipe = pd.DataFrame({"start_ft": [0.0, 150.0], "end_ft": [150.0, 400.0], "od_in": [10.0, 5.0]})
    inputs = SimpleNamespace(prev_std=prev, curr_std=curr, weld_map=weld_map, pipe=pipe, settings=settings)
    result = anomaly_distribution(inputs)
    by_run = result.dent.set_index(["run_id", "bin"])["all"]
    assert by_run[("P", "2-6")] == 1 and by_run[("C", "0-2")] == 1
    assert len(result.depth) == 2 * 3 * 9 and len(result.dent) == 2 * 3
    table = anomaly_distribution_table(result)
    assert set(table.table) == {"depth", "dent"} and table.columns[0] == "table"


# --- per-joint metal-loss counts ----------------------------------------------------------------------

def _wm(*rows):
    """A weld map from (match_type, prev_gw_id, prev_gw_id_2, curr_gw_id, curr_gw_id_2, curr_start_ft) tuples, with
    the ``align_welds`` convention: an absent joint is None (never NaN), ``*_2`` is set only for 2:1 / 1:2."""
    cols = ["match_type", "prev_gw_id", "prev_gw_id_2", "curr_gw_id", "curr_gw_id_2", "curr_start_ft"]
    wm = pd.DataFrame(rows, columns=cols, dtype=object)
    wm["curr_start_ft"] = wm.curr_start_ft.astype(float)
    wm.insert(0, "section_id", range(len(wm)))
    return wm


def _ml(*joints):
    """A standardized run from (us_gw_id, surface) pairs, all metal loss; a dent row is added to a joint that
    has any, to show only metal loss is counted."""
    rows = [{"feature_type": "metal_loss", "us_gw_id": j, "surface": s} for j, s in joints]
    rows.append({"feature_type": "dent", "us_gw_id": joints[0][0], "surface": "EXT"})
    return pd.DataFrame(rows)


def test_joint_counts_one_to_one():
    wm = _wm(("1:1", "P1", None, "C1", None, 10.0), ("1:1", "P2", None, "C2", None, 50.0))
    prev = _ml(("P1", "EXT"), ("P1", "INT"), ("P2", "EXT"))
    curr = _ml(("C1", "EXT"), ("C1", "EXT"), ("C1", "MID"), ("C1", "UNK"), ("C1", None), ("C1", "INT"))
    out = joint_ml_counts(wm, prev, curr)
    one, two = out.iloc[0], out.iloc[1]
    assert (one.prev_ext, one.prev_int, one.prev_other, one.prev_total) == (1, 1, 0, 2)
    assert (one.curr_ext, one.curr_int, one.curr_other, one.curr_total) == (2, 1, 3, 6)
    assert one.change_total == 4 and one.prev_gw_ids == "P1" and one.curr_gw_ids == "C1"
    assert (two.prev_total, two.curr_total, two.change_total) == (1, 0, -1)  # a joint with no metal loss is 0
    counts = [f"{s}_{k}" for s in ("prev", "curr") for k in ("ext", "int", "other", "total")] + ["change_total"]
    assert all(str(out[c].dtype) == "Int64" for c in counts)


def test_joint_counts_two_to_one_sums_both_prev_joints():
    wm = _wm(("2:1", "P1", "P2", "C1", None, 10.0), ("1:2", "P3", None, "C2", "C3", 90.0))
    prev = _ml(("P1", "EXT"), ("P2", "EXT"), ("P2", "INT"), ("P3", "INT"))
    curr = _ml(("C1", "EXT"), ("C2", "INT"), ("C3", "INT"), ("C3", "MID"))
    out = joint_ml_counts(wm, prev, curr)
    first, second = out.iloc[0], out.iloc[1]
    assert first.prev_gw_ids == "P1;P2" and first.curr_gw_ids == "C1"
    assert (first.prev_ext, first.prev_int, first.prev_total, first.curr_total, first.change_total) == (2, 1, 3, 1, -2)
    assert second.curr_gw_ids == "C2;C3" and (second.curr_int, second.curr_other, second.curr_total) == (2, 1, 3)
    assert second.change_total == 2


def test_inserted_joint_prev_blank():
    wm = _wm(("inserted", None, None, "C1", None, 10.0), ("removed", "P2", None, None, None, np.nan))
    out = joint_ml_counts(wm, _ml(("P2", "EXT")), _ml(("C1", "EXT")))
    ins, rem = out.iloc[0], out.iloc[1]
    assert ins.prev_gw_ids == "" and all(pd.isna(ins[f"prev_{k}"]) for k in ("ext", "int", "other", "total"))
    assert ins.curr_total == 1 and pd.isna(ins.change_total)
    assert rem.curr_gw_ids == "" and pd.isna(rem.curr_total) and rem.prev_total == 1 and pd.isna(rem.change_total)


def test_largest_joints_rank_and_break_ties_by_position():
    wm = _wm(("1:1", "P1", None, "C1", None, 30.0), ("1:1", "P2", None, "C2", None, 20.0),
             ("1:1", "P3", None, "C3", None, 10.0), ("removed", "P4", None, None, None, np.nan))
    prev = _ml(("P1", "EXT"), ("P2", "EXT"), ("P3", "EXT"), ("P4", "EXT"))
    curr = _ml(("C1", "EXT"), ("C1", "EXT"), ("C2", "EXT"), ("C2", "EXT"), ("C3", "EXT"))
    by_count, by_change = largest_joints(joint_ml_counts(wm, prev, curr), 2)
    assert list(by_count.curr_gw_ids) == ["C2", "C1"]  # tie on 2: the earlier position first
    assert list(by_change.curr_gw_ids) == ["C2", "C1"] and len(by_change) == 2
