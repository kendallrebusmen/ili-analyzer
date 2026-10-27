"""Previous-run classification and category changes (spec section 6.10).

The previous run's anomalies are classified with the same rules as the current run's (M7), on the values
the previous run reported, so the report can say which features moved to a more severe category. Reusing
``classify.feature_variables`` and ``classify.classify`` unchanged, ``previous_side`` presents each matched
row's previous-run side as a current-side row, ``previous_pressures`` supplies the strength numbers M6 only
computed for the current run, and ``category_changes`` compares the two classifications by match_id.

Unknown stays None: a previous-run value that was not reported is never guessed, and a rule it would
decide is "not evaluable" exactly as in the current run.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ili.classify import CATEGORY_PRIORITY, RL_INPUT_COLUMNS, classify, feature_variables
from ili.config import RepairCriteria, Settings
from ili.reference import _require_columns, lookup
from ili.remaining_life import MOP_WHAT, PIPE_WHAT, usable_inputs
from ili.schema import FeatureType
from ili.strength import assessed_depth_in, assessed_length_in, burst_pressure, failure_stress, safe_pressure

# The categories the rules decide; ``imp_scheduled`` (a remaining-life dig, no rule) compares as evaluate.
RULE_CATEGORIES = ("immediate", "60_day", "180_day", "evaluate")
PREVIOUS_MATCH_TYPES = ("matched_1to1", "matched_cluster")
# Current-side columns that take the previous run's value (names without ``curr_``).
SWAPPED_FIELDS = ("feature_type", "feature_id", "wt_nom_in", "depth_pct", "dent_depth_in", "length_in", "width_in",
                  "clock_min", "seam_interaction", "dent_metal_loss")
CHANGE_COLUMNS = [
    "match_id", "match_type", "prev_feature_id", "curr_feature_id", "feature_type", "prev_depth_pct_assessed",
    "curr_depth_pct_assessed", "prev_p_fail", "curr_p_fail", "prev_rule_category", "prev_governing_rule",
    "prev_triggered_rules", "prev_rules_not_evaluable", "curr_rule_category", "curr_governing_category",
    "curr_governing_rule", "moved_more_severe",
]


@dataclass(frozen=True)
class CategoryChangeResult:
    table: pd.DataFrame  # CHANGE_COLUMNS, one row per matched anomaly row, match-table order
    previous: pd.DataFrame  # the previous run's classification table (classify.CLASSIFICATION_COLUMNS)


def previous_side(match_table: pd.DataFrame) -> pd.DataFrame:
    """The ``matched_1to1`` and ``matched_cluster`` rows of the match table, with each ``curr_*`` column of
    ``SWAPPED_FIELDS`` replaced by its ``prev_*`` column (a cluster's previous side is the envelope), so
    ``classify.feature_variables`` reads the previous run as if it were the current one. Kept as they are:
    ``match_id``, ``match_type``, ``needs_review`` and ``curr_log_dist_ft`` -- reference files are keyed to
    current-run log distance (spec 5.2). The original ids stay as ``orig_prev_feature_id`` and
    ``orig_curr_feature_id``."""
    _require_columns(match_table, ("match_type", "prev_feature_id", "curr_feature_id",
                                   *(f"{s}_{f}" for s in ("prev", "curr") for f in SWAPPED_FIELDS)), "match table")
    rows = match_table[match_table.match_type.isin(PREVIOUS_MATCH_TYPES)].reset_index(drop=True)
    out = rows.assign(orig_prev_feature_id=rows.prev_feature_id, orig_curr_feature_id=rows.curr_feature_id)
    for field in SWAPPED_FIELDS:
        out[f"curr_{field}"] = rows[f"prev_{field}"]
    return out


def previous_pressures(prev_rows: pd.DataFrame, remaining_life: pd.DataFrame, pipe: pd.DataFrame,
                       mop: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Strength of the previous run's metal-loss rows (``previous_side`` rows with ``curr_feature_type``
    metal_loss), as a remaining-life table for ``classify.feature_variables``: columns
    ``classify.RL_INPUT_COLUMNS``, one row per such row, in order.

    - ``wt_in``: M6's ``wt_in`` for the match_id, else ``curr_wt_nom_in`` (the previous wall here).
    - ``mop_psig`` is ``mop`` at ``curr_log_dist_ft``; ``od_in`` and ``smys_psi`` come from ``pipe`` there
      (a position they do not cover is a ValueError, as in M6).
    - ``assessed_dt`` = ``assessed_depth_in`` / ``wt_in``; ``burst_psig`` and ``safe_psig`` use
      ``strength.method`` on the assessed depth and ``assessed_length_in`` (an unusable OD, SMYS or length
      gives NaN, as in M6).
    - ``dig_required_by_remaining_life`` False, ``repair_by_date`` None, ``needs_review`` False,
      ``review_reasons`` None: no growth is projected from the previous run.
    """
    _require_columns(prev_rows, ("match_id", "curr_feature_type", "curr_log_dist_ft", "curr_wt_nom_in",
                                 "curr_depth_pct", "curr_length_in"), "previous-run rows")
    _require_columns(remaining_life, ("match_id", "wt_in"), "remaining-life table")
    rows = prev_rows[prev_rows.curr_feature_type.astype(str) == FeatureType.metal_loss.value].reset_index(drop=True)
    n = len(rows)
    pos = rows.curr_log_dist_ft.to_numpy(float)
    tol = settings.alignment.anchor_tol_ft
    if n:
        od = lookup(pipe, pos, "od_in", tol, what=PIPE_WHAT, allow_nan=True)
        smys = lookup(pipe, pos, "smys_psi", tol, what=PIPE_WHAT, allow_nan=True)
        mop_psig = lookup(mop, pos, "mop_psig", tol, what=MOP_WHAT, allow_nan=True)
    else:
        od = smys = mop_psig = np.zeros(0)
    m6_wt = dict(zip(remaining_life.match_id.astype(str), remaining_life.wt_in.astype(float)))
    wt = np.array([m6_wt.get(str(m), np.nan) for m in rows.match_id], dtype=float)
    wt = np.where(np.isnan(wt), rows.curr_wt_nom_in.to_numpy(float), wt)
    d_as = np.asarray(assessed_depth_in(rows.curr_depth_pct.to_numpy(float), wt, settings), dtype=float)
    l_as = np.asarray(assessed_length_in(rows.curr_length_in.to_numpy(float), settings), dtype=float)
    l_ok, od_ok, smys_ok, _ = usable_inputs(l_as, od, smys, mop_psig)
    with np.errstate(invalid="ignore", divide="ignore"):
        dt = d_as / wt
        s_f = np.asarray(failure_stress(settings.strength.method, od_ok, wt, d_as, l_ok, smys_ok)[3], dtype=float)
        p_f = np.asarray(burst_pressure(s_f, wt, od_ok), dtype=float)
    p_safe = np.asarray(safe_pressure(p_f, settings), dtype=float)
    out = pd.DataFrame({
        "match_id": rows.match_id.astype(object), "wt_in": wt, "mop_psig": mop_psig, "assessed_dt": dt,
        "burst_psig": p_f, "safe_psig": p_safe,
        "dig_required_by_remaining_life": pd.Series([False] * n, dtype=object),
        "repair_by_date": pd.Series([None] * n, dtype=object),
        "needs_review": pd.Series([False] * n, dtype=object),
        "review_reasons": pd.Series([None] * n, dtype=object),
    })
    return out[list(RL_INPUT_COLUMNS)]


def _rule_category(category: str) -> str:
    """The rule-decided category: ``imp_scheduled`` (no rule fired) compares as ``evaluate``."""
    return "evaluate" if category == "imp_scheduled" else category


def category_changes(match_table: pd.DataFrame, remaining_life: pd.DataFrame, classification: pd.DataFrame,
                     curr_std: pd.DataFrame, pipe: pd.DataFrame, mop: pd.DataFrame, segments: pd.DataFrame,
                     crossings: pd.DataFrame | None, settings: Settings, criteria: RepairCriteria,
                     prev_run_date) -> CategoryChangeResult:
    """Classify the previous run's matched anomalies and compare them with ``classification`` (M7's table of
    the current run). ``table`` has columns ``CHANGE_COLUMNS``: one row per matched anomaly row of
    ``classification`` (``matched_1to1`` / ``matched_cluster``) whose previous side is an anomaly too, in
    match-table order. New rows (no previous side) and missing rows (no current side) are absent.

    - The previous side: ``previous_side`` -> ``previous_pressures`` -> ``classify.feature_variables`` and
      ``classify.classify`` with ``prev_run_date`` as the assessment date (discovery: latest permissible);
      ``previous`` is that classification table. The same settings, criteria, segments and crossings apply.
    - ``feature_type`` is the current one (a dent that became metal loss keeps ``metal_loss``); the
      ``prev_*`` columns hold the previous run's values, the ``curr_*`` columns the classification's.
    - ``*_rule_category``: the governing category with ``imp_scheduled`` read as ``evaluate``.
      ``moved_more_severe`` is True when the current one ranks above the previous in ``CATEGORY_PRIORITY``.
    """
    prev_rows = previous_side(match_table)
    pressures = previous_pressures(prev_rows, remaining_life, pipe, mop, settings)
    prev_vars = feature_variables(prev_rows, pressures, curr_std, pipe, segments, crossings, settings)
    previous = classify(prev_vars, criteria, settings, prev_run_date).table
    prev_by_id = previous.set_index("match_id")
    orig = prev_rows.set_index("match_id")
    rank = {c: k for k, c in enumerate(CATEGORY_PRIORITY)}
    rows = []
    for c in classification.itertuples(index=False):
        if c.match_id not in prev_by_id.index:
            continue
        p = prev_by_id.loc[c.match_id]
        prev_cat, curr_cat = _rule_category(p.governing_category), _rule_category(c.governing_category)
        rows.append({
            "match_id": c.match_id, "match_type": orig.match_type[c.match_id],
            "prev_feature_id": orig.orig_prev_feature_id[c.match_id],
            "curr_feature_id": orig.orig_curr_feature_id[c.match_id], "feature_type": c.feature_type,
            "prev_depth_pct_assessed": p.depth_pct_assessed, "curr_depth_pct_assessed": c.depth_pct_assessed,
            "prev_p_fail": p.p_fail, "curr_p_fail": c.p_fail, "prev_rule_category": prev_cat,
            "prev_governing_rule": p.governing_rule, "prev_triggered_rules": p.triggered_rules,
            "prev_rules_not_evaluable": p.rules_not_evaluable, "curr_rule_category": curr_cat,
            "curr_governing_category": c.governing_category, "curr_governing_rule": c.governing_rule,
            "moved_more_severe": rank[curr_cat] < rank[prev_cat],
        })
    table = pd.DataFrame(rows, columns=CHANGE_COLUMNS)
    table["moved_more_severe"] = table.moved_more_severe.astype(bool)
    return CategoryChangeResult(table, previous)
