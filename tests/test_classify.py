"""Repair classification and due dates (spec 6.8; M7 Task 4)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ili import growth, reference
from ili.align_welds import align_welds
from ili.classify import (CATEGORY_PRIORITY, CLASSIFICATION_COLUMNS, FEATURE_VARIABLE_COLUMNS,
                          IMMEDIATE_ACTION_NOTE, ClassificationResult, classify, feature_variables)
from ili.config import load_repair_criteria, load_settings
from ili.match import delta_t_years, match_anomalies
from ili.remaining_life import compute_remaining_life

ASSESSED = date(2026, 6, 15)
DISCOVERED = date(2026, 7, 1)
REPAIR_BY = date(2029, 3, 1)
WARNING = ("Discovery date not entered: using the latest permissible date (assessment date + 180 days) per "
           "195.452(h)(2). Enter the actual discovery date.")


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    c = s.classification
    assert c.due_days == {"60_day": 60, "180_day": 180} and c.discovery_max_days == 180
    return s


@pytest.fixture(scope="module")
def criteria():
    return load_repair_criteria()


def base(mid, ftype="metal_loss", **kw):
    """One feature-variables row with every input known and no rule triggered (a quiet feature).

    Metal loss: assessed depth 30%, P_F 1800 / P_safe 1296 against MOP 1000, bottom of pipe (clock 360),
    no seam, weld, crossing, general or circumferential corrosion; no dig by remaining life.
    Dents and gouges carry no pressures, dig flag or repair-by date (as ``feature_variables`` gives them)."""
    r = {c: None for c in FEATURE_VARIABLE_COLUMNS}
    r.update(match_id=mid, curr_feature_id=f"B-{mid}", segment_id="SEG-01", feature_type=ftype,
             could_affect_hca=True, od_in=12.75, wt_in=0.25, clock_min=360, top_of_pipe=False,
             length_in=2.0, width_in=1.0, seam_interaction=False, near_girth_weld=False,
             near_foreign_crossing=False, general_corrosion=False, widespread_circumferential=False,
             manual_immediate=False, field_crack_confirmed=False, upstream_needs_review=False,
             upstream_review_reasons=None)
    if ftype == "metal_loss":
        r.update(depth_pct_reported=20.0, depth_pct_assessed=30.0, p_fail=1800.0, p_safe=1296.0,
                 mop_local=1000.0, dig_required_by_remaining_life=False, repair_by_date=REPAIR_BY)
    elif ftype == "dent":
        r.update(dent_depth_in=0.1, dent_depth_pct_od=0.1 / 12.75 * 100, dent_metal_loss=False)
    else:
        r.update(depth_pct_reported=1.0, depth_pct_assessed=11.0)
    r.update(kw)
    return r


def frame(rows):
    df = pd.DataFrame(rows, columns=FEATURE_VARIABLE_COLUMNS)
    df["clock_min"] = df.clock_min.astype("Int64")
    return df


def go(rows, settings, criteria, discovery=DISCOVERED, assessment=ASSESSED, sha=None):
    return classify(frame(rows), criteria, settings, assessment, discovery_date=discovery, criteria_sha256=sha)


def rec(result, mid):
    """One output row as plain Python values (numpy scalars unwrapped, NaN/NA as None)."""
    row = result.table[result.table.match_id == mid].iloc[0]
    out = {}
    for c in result.table.columns:
        x = row[c]
        x = x.item() if isinstance(x, np.generic) else x
        out[c] = None if not isinstance(x, (str, date)) and pd.isna(x) else x
    return out


def dent_pct(pct, od=12.75):
    return dict(dent_depth_in=pct / 100 * od, dent_depth_pct_od=pct)


# --- governing category -----------------------------------------------------------------------------


def test_priority_and_governing_rule(settings, criteria):
    rows = [
        base("A", p_fail=950.0, p_safe=684.0),  # IMM-2 and D180-4
        base("B", depth_pct_assessed=85.0, p_fail=950.0, p_safe=684.0),  # IMM-1, IMM-2, D180-4
        base("C", "dent", clock_min=0, top_of_pipe=True, **dent_pct(4.0)),  # D60-1 and D180-2
        base("D", p_safe=990.0, dig_required_by_remaining_life=True),  # D180-4 beats the dig flag
        base("E", dig_required_by_remaining_life=True),
        base("F"),
    ]
    res = go(rows, settings, criteria)
    assert isinstance(res, ClassificationResult)
    got = {m: (rec(res, m)["governing_category"], rec(res, m)["governing_rule"], rec(res, m)["triggered_rules"])
           for m in "ABCDEF"}
    assert got == {
        "A": ("immediate", "IMM-2", "IMM-2;D180-4"),
        "B": ("immediate", "IMM-1", "IMM-1;IMM-2;D180-4"),
        "C": ("60_day", "D60-1", "D60-1;D180-2"),
        "D": ("180_day", "D180-4", "D180-4"),
        "E": ("imp_scheduled", "REMAINING_LIFE", None),
        "F": ("evaluate", "EVALUATE", None),
    }
    for m in "ABCDEF":
        r = rec(res, m)
        assert r["rule_set"] == "hca" and r["rules_not_evaluable"] is None
        assert r["requires_review"] is False and r["review_reasons"] is None, m
    assert rec(res, "A")["action_note"] == IMMEDIATE_ACTION_NOTE == \
        "Requires temporary pressure reduction or shutdown until repaired."
    assert all(rec(res, m)["action_note"] is None for m in "CDEF")
    assert CATEGORY_PRIORITY == ("immediate", "60_day", "180_day", "imp_scheduled", "evaluate")


def test_imp_scheduled_and_evaluate_defaults(settings, criteria):
    rows = [base("E", dig_required_by_remaining_life=True), base("F"), base("D1", "dent"), base("G1", "gouge"),
            # dig flag unknown (metal loss missing from M6): never imp_scheduled by guess; evaluate, reviewed
            base("M", p_fail=None, p_safe=None, mop_local=None, dig_required_by_remaining_life=None,
                 repair_by_date=None, upstream_needs_review=True, upstream_review_reasons="rl_missing")]
    res = go(rows, settings, criteria)
    e, f, d1, g1, m = (rec(res, k) for k in ("E", "F", "D1", "G1", "M"))
    assert (e["governing_category"], e["governing_rule"], e["due_date"]) == ("imp_scheduled", "REMAINING_LIFE",
                                                                             REPAIR_BY)
    for r in (f, d1, g1):
        assert (r["governing_category"], r["governing_rule"], r["due_date"]) == ("evaluate", "EVALUATE", None)
        assert r["requires_review"] is False and r["rules_not_evaluable"] is None
    assert (m["governing_category"], m["governing_rule"], m["due_date"]) == ("evaluate", "EVALUATE", None)
    # Review Focus #5: unknown pressures leave IMM-2 and D180-4 not evaluable, flagged (never a silent pass)
    assert m["rules_not_evaluable"] == "IMM-2;D180-4;REMAINING_LIFE"
    assert m["requires_review"] is True and m["review_reasons"] == "rl_missing;rule_not_evaluable"


def test_unknown_that_cannot_change_outcome_does_not_flag(settings, criteria):
    # Review Focus #1: D180-6 is "depth > 50 and (crossing or circumferential or weld)"; depth 30 decides False.
    unknown = dict(near_foreign_crossing=None, widespread_circumferential=None, near_girth_weld=None)
    res = go([base("A", depth_pct_assessed=30.0, **unknown)], settings, criteria)
    r = rec(res, "A")
    assert r["rules_not_evaluable"] is None and r["triggered_rules"] is None
    assert r["requires_review"] is False and r["review_reasons"] is None
    assert r["governing_category"] == "evaluate"
    assert res.summary["n_not_evaluable_by_rule"] == {}


def test_unknown_that_could_trigger_flags(settings, criteria):
    unknown = dict(near_foreign_crossing=None, widespread_circumferential=None, near_girth_weld=None)
    res = go([base("A", depth_pct_assessed=60.0, **unknown)], settings, criteria)
    r = rec(res, "A")
    assert r["rules_not_evaluable"] == "D180-6" and r["triggered_rules"] is None
    assert r["requires_review"] is True and r["review_reasons"] == "rule_not_evaluable"
    # Unknown is never treated as triggered: the feature is not raised to 180_day by D180-6.
    assert r["governing_category"] == "evaluate" and r["due_date"] is None
    assert res.summary["n_not_evaluable_by_rule"] == {"D180-6": 1}


def test_not_evaluable_more_severe_rule_neither_raises_nor_lowers(settings, criteria):
    """IMM-2 (P_F < MOP) is unknown and D180-4 (P_safe < MOP) is True: the feature is 180_day, not immediate,
    with IMM-2 listed and review required. This is correct: an unknown is never treated as triggered (that
    would be a guessed value); the reviewer resolves IMM-2 and may raise the category."""
    res = go([base("A", p_fail=None, p_safe=900.0),
              base("B", depth_pct_assessed=90.0, p_fail=None)], settings, criteria)  # IMM-1 True, IMM-2 unknown
    a, b = rec(res, "A"), rec(res, "B")
    assert (a["governing_category"], a["governing_rule"], a["triggered_rules"]) == ("180_day", "D180-4", "D180-4")
    assert a["rules_not_evaluable"] == "IMM-2" and a["requires_review"] is True
    assert a["review_reasons"] == "rule_not_evaluable"
    assert a["due_date"] == DISCOVERED + timedelta(days=180) and a["action_note"] is None
    # nor does an unknown lower a known category
    assert (b["governing_category"], b["governing_rule"]) == ("immediate", "IMM-1")
    assert b["rules_not_evaluable"] == "IMM-2" and b["requires_review"] is True


def test_rule_review_and_upstream_reasons(settings, criteria):
    rows = [base("A", depth_pct_assessed=60.0, general_corrosion=True),  # D180-5 requires review
            base("B", upstream_needs_review=True, upstream_review_reasons="match_review;rule_review"),
            base("C", upstream_needs_review=True, upstream_review_reasons=None)]
    res = go(rows, settings, criteria)
    a, b, c = rec(res, "A"), rec(res, "B"), rec(res, "C")
    assert (a["governing_category"], a["governing_rule"]) == ("180_day", "D180-5")
    assert a["requires_review"] is True and a["review_reasons"] == "rule_review"
    assert b["requires_review"] is True and b["review_reasons"] == "match_review;rule_review"  # no repeat
    assert c["requires_review"] is True and c["review_reasons"] is None  # an upstream flag alone counts


def test_non_hca_uses_nh_rules_and_requires_review(settings, criteria):
    rows = [base("N", could_affect_hca=False, segment_id="SEG-02", depth_pct_assessed=85.0),
            base("ND", "dent", could_affect_hca=False, segment_id="SEG-02", clock_min=100, top_of_pipe=True,
                 dent_metal_loss=True, **dent_pct(1.0)),
            base("NQ", could_affect_hca=False, segment_id="SEG-02")]
    res = go(rows, settings, criteria)
    n, nd, nq = rec(res, "N"), rec(res, "ND"), rec(res, "NQ")
    assert n["rule_set"] == nd["rule_set"] == nq["rule_set"] == "non_hca"
    assert (n["governing_category"], n["governing_rule"], n["triggered_rules"]) == ("immediate", "NH-IMM-1",
                                                                                    "NH-IMM-1")
    assert (nd["governing_category"], nd["governing_rule"], nd["triggered_rules"]) == ("immediate", "NH-IMM-3",
                                                                                       "NH-IMM-3")
    for r in (n, nd):
        assert r["requires_review"] is True and r["review_reasons"] == "rule_review"
        assert r["due_date"] == DISCOVERED and r["action_note"] == IMMEDIATE_ACTION_NOTE
    assert nq["governing_category"] == "evaluate" and nq["triggered_rules"] is None
    # no HCA rule id ever fires or is listed for a non-HCA feature
    assert res.summary["counts_by_rule"] == {"NH-IMM-1": 1, "NH-IMM-3": 1}


def test_hca_unknown_and_outside_segment_use_hca_rules(settings, criteria):
    rows = [base("U", could_affect_hca=None, segment_id="SEG-X", depth_pct_assessed=85.0),
            base("O", segment_id=None, depth_pct_assessed=85.0, upstream_needs_review=True,
                 upstream_review_reasons="no_segment")]
    res = go(rows, settings, criteria)
    u, o = rec(res, "U"), rec(res, "O")
    assert u["rule_set"] == o["rule_set"] == "hca_assumed"
    assert u["governing_rule"] == o["governing_rule"] == "IMM-1"
    assert u["requires_review"] is True and u["review_reasons"] == "hca_unknown"
    assert o["requires_review"] is True and o["review_reasons"] == "no_segment"


def test_feature_type_filter(settings, criteria):
    # A gouge deeper than 12.5% triggers D180-9 only; the same depth on metal loss does not (no IMM-1 etc.).
    res = go([base("G", "gouge", depth_pct_assessed=12.6), base("G0", "gouge", depth_pct_assessed=12.5),
              base("D", "dent", depth_pct_assessed=90.0)], settings, criteria)
    assert (rec(res, "G")["governing_category"], rec(res, "G")["triggered_rules"]) == ("180_day", "D180-9")
    assert rec(res, "G0")["governing_category"] == "evaluate"
    assert rec(res, "D")["triggered_rules"] is None  # IMM-1 is metal loss only


# --- due dates ---------------------------------------------------------------------------------------


def _five(**kw):
    return [base("IM", depth_pct_assessed=85.0, **kw), base("D60", "dent", clock_min=0, top_of_pipe=True,
                                                            **dent_pct(4.0), **kw),
            base("D180", p_safe=990.0, **kw), base("IMP", dig_required_by_remaining_life=True, **kw),
            base("EV", **kw)]


def test_due_dates(settings, criteria):
    res = go(_five(), settings, criteria, discovery=date(2026, 7, 1))
    due = {m: rec(res, m)["due_date"] for m in ("IM", "D60", "D180", "IMP", "EV")}
    assert due == {"IM": date(2026, 7, 1), "D60": date(2026, 8, 30), "D180": date(2026, 12, 28),
                   "IMP": REPAIR_BY, "EV": None}
    t = res.table
    assert set(t.discovery_date) == {date(2026, 7, 1)} and set(t.discovery_date_source) == {"entered"}
    s = res.summary
    assert s["discovery_date"] == "2026-07-01" and s["discovery_date_source"] == "entered"
    assert s["discovery_warning"] is None

    # missing: assessment + 180 days, with the warning
    res = go(_five(), settings, criteria, discovery=None)
    due = {m: rec(res, m)["due_date"] for m in ("IM", "D60", "D180", "IMP", "EV")}
    assert due == {"IM": date(2026, 12, 12), "D60": date(2027, 2, 10), "D180": date(2027, 6, 10),
                   "IMP": REPAIR_BY, "EV": None}
    assert set(res.table.discovery_date) == {date(2026, 12, 12)}
    assert set(res.table.discovery_date_source) == {"latest_permissible"}
    s = res.summary
    assert s["discovery_date"] == "2026-12-12" and s["discovery_date_source"] == "latest_permissible"
    assert s["discovery_warning"] == WARNING


@pytest.mark.parametrize("discovery, ok", [
    (date(2026, 6, 14), False),  # the day before the assessment
    (date(2026, 6, 15), True),  # the assessment day
    (date(2026, 12, 12), True),  # assessment + 180 days: the limit itself
    (date(2026, 12, 13), False),  # assessment + 181 days
])
def test_discovery_date_bounds(settings, criteria, discovery, ok):
    if ok:
        res = go(_five(), settings, criteria, discovery=discovery)
        assert rec(res, "IM")["due_date"] == discovery
        assert rec(res, "D60")["due_date"] == discovery + timedelta(days=60)
    else:
        with pytest.raises(ValueError, match="discovery date"):
            go(_five(), settings, criteria, discovery=discovery)


def test_discovery_limit_and_due_days_come_from_settings(settings, criteria):
    c = settings.classification.model_copy(update={"discovery_max_days": 90,
                                                   "due_days": {"60_day": 30, "180_day": 120}})
    s = settings.model_copy(update={"classification": c})
    res = go(_five(), s, criteria, discovery=None)
    assert rec(res, "IM")["due_date"] == ASSESSED + timedelta(days=90)
    assert rec(res, "D60")["due_date"] == ASSESSED + timedelta(days=120)
    assert rec(res, "D180")["due_date"] == ASSESSED + timedelta(days=210)
    assert "assessment date + 90 days" in res.summary["discovery_warning"]
    with pytest.raises(ValueError, match="discovery date"):
        go(_five(), s, criteria, discovery=ASSESSED + timedelta(days=91))


def test_imp_scheduled_without_repair_by_date_flags(settings, criteria):
    res = go([base("IMP", dig_required_by_remaining_life=True, repair_by_date=None)], settings, criteria)
    r = rec(res, "IMP")
    assert (r["governing_category"], r["due_date"]) == ("imp_scheduled", None)
    assert r["requires_review"] is True and r["review_reasons"] == "due_date_unknown"


def test_timestamp_dates_are_accepted(settings, criteria):
    res = classify(frame(_five()), criteria, settings, pd.Timestamp(ASSESSED),
                   discovery_date=pd.Timestamp(DISCOVERED))
    assert rec(res, "IM")["due_date"] == DISCOVERED and type(rec(res, "IM")["due_date"]) is date


# --- table and summary -------------------------------------------------------------------------------


def test_columns_order_and_summary(settings, criteria):
    rows = _five(could_affect_hca=True) + [base("NE", depth_pct_assessed=60.0, near_foreign_crossing=None)]
    res = go(rows, settings, criteria, sha="ab" * 32)
    assert list(res.table.columns) == CLASSIFICATION_COLUMNS == [
        *FEATURE_VARIABLE_COLUMNS, "rule_set", "triggered_rules", "rules_not_evaluable", "governing_category",
        "governing_rule", "requires_review", "review_reasons", "discovery_date", "discovery_date_source",
        "due_date", "action_note"]
    assert list(res.table.match_id) == ["IM", "D60", "D180", "IMP", "EV", "NE"]
    s = res.summary
    assert s["criteria_version"] == criteria.version and s["criteria_sha256"] == "ab" * 32
    assert s["counts_by_category"] == {"immediate": 1, "60_day": 1, "180_day": 1, "imp_scheduled": 1,
                                       "evaluate": 2}
    assert list(s["counts_by_category"]) == list(CATEGORY_PRIORITY)
    assert s["counts_by_rule"] == {"IMM-1": 1, "D60-1": 1, "D180-2": 1, "D180-4": 1}
    assert s["n_not_evaluable_by_rule"] == {"D180-6": 1}
    assert s["n_requires_review"] == 1 and s["n_features"] == 6
    import json
    assert json.loads(json.dumps(s)) == s  # JSON-safe


def test_empty_table(settings, criteria):
    res = classify(frame([]), criteria, settings, ASSESSED)
    assert res.table.empty and list(res.table.columns) == CLASSIFICATION_COLUMNS
    assert res.summary["n_features"] == 0
    assert res.summary["counts_by_category"] == dict.fromkeys(CATEGORY_PRIORITY, 0)


# --- 5-mile synthetic line through M3-M6 ----------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(settings, criteria):
    from ili.synth import SynthParams, generate
    s = settings
    r = generate(SynthParams(line_miles=5))
    ref = r.reference
    wm = align_welds(r.prev_std, r.curr_std, s, ref["repairs_between_runs"]).weld_map
    mt = match_anomalies(r.prev_std, r.curr_std, wm, s, ref["pipe_properties"],
                         ref["repairs_between_runs"]).match_table
    seg = reference.segment_table(s, ref["hca_segments"], 0.0, float(r.curr_std.log_dist_ft.max()))
    dt = delta_t_years(r.prev_std, r.curr_std)
    g = growth.compute_growth(mt, dt, r.prev_std.run_date.iloc[0].year, r.curr_std.run_date.iloc[0].year,
                              ref["pipe_properties"], seg, s)
    run_date = r.curr_std.run_date.iloc[0]
    rl = compute_remaining_life(g.features, run_date, ref["pipe_properties"], ref["mop_profile"],
                                ref["hca_segments"], ref["next_inspection"], s).features
    v = feature_variables(mt, rl, r.curr_std, ref["pipe_properties"], ref["hca_segments"],
                          ref["foreign_crossings"], s)
    return r, v, classify(v, criteria, s, run_date)


def test_synthetic_severe_features_are_immediate(synth):
    r, v, res = synth
    t = res.table.set_index("curr_feature_id")
    severe = r.ground_truth[r.ground_truth.population == "severe"]
    assert len(severe) == 5 and (severe.depth_curr_true_pct > 80).all()
    for fid in severe.curr_feature_id:
        row = t.loc[fid]
        assert row.governing_category == "immediate", fid
        assert row.governing_rule in ("IMM-1", "NH-IMM-1"), fid
        assert row.action_note == IMMEDIATE_ACTION_NOTE and row.due_date == row.discovery_date

    # Every dent: a hand classification from its reported current-run values (clock, depth, OD, flags).
    od = r.params.od_in
    assert od >= 12.75 and set(r.reference["pipe_properties"].od_in) == {od}  # the % OD alternatives apply
    std = r.curr_std.set_index("feature_id")
    dents = res.table[res.table.feature_type == "dent"]
    assert len(dents) == r.params.n_dents
    seen = set()
    for _, row in dents.iterrows():
        f = std.loc[row.curr_feature_id]
        clock = round(float(f.clock_min))
        top = clock >= 480 or clock <= 240
        pct = float(f.dent_depth_in) / od * 100
        ml = bool(f.dent_metal_loss) or (not pd.isna(f.depth_pct) and f.depth_pct > 0)
        seam = bool(f.seam_interaction)
        # D180-1: (near weld [unknown: null setting] or seam) and deeper than 2%
        d180_1 = (True if seam else None) if pct > 2 else False
        if top and (ml or pct > 6):
            want = "immediate"
        elif (top and pct > 3) or (not top and ml):
            want = "60_day"
        elif (top and pct > 2) or (not top and pct > 6) or d180_1 is True:
            want = "180_day"
        else:
            want = "evaluate"
        seen.add(want)
        assert row.governing_category == want, (row.curr_feature_id, clock, pct, ml, seam)
        listed = row.rules_not_evaluable.split(";") if isinstance(row.rules_not_evaluable, str) else []
        assert any(x.endswith("D180-1") for x in listed) == (d180_1 is None), row.curr_feature_id
        if d180_1 is None:
            assert row.requires_review
    assert {"immediate", "60_day"} <= seen  # the synthetic dents reach the upper categories

    s = res.summary
    assert s["n_features"] == len(v) == sum(s["counts_by_category"].values())
    assert s["discovery_date_source"] == "latest_permissible" and s["discovery_warning"] == WARNING
