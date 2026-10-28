"""Repair classification (spec section 6.8): the feature variables of every current-run anomaly.

``feature_variables`` builds one record per current-run anomaly (metal loss, dent, gouge) of the M4
match table, holding a value for every name in ``ili.vocabulary.VARIABLES``. Those records are what the
repair-criteria conditions (``config/repair_criteria.yaml``, evaluated by ``ili.rules``) are tested on.

Unknown is None, never a guessed value. A variable whose input is missing, or whose defining setting in
``classification`` is a null PLACEHOLDER, is None, so a rule that depends on it is reported as not
evaluable instead of silently passing or failing. Each variable family has its own helper below; each
documents its rule, including where the result is decided from partial information (Kleene logic).

``manual_immediate`` and ``field_crack_confirmed`` are manual and field entries from the M9 flag log
(``review.effective_flags``): True when the latest entry for the feature (any member of a cluster) is True,
False otherwise (no evaluator flag, no confirmed crack is on record).

``classify`` then applies the repair-criteria rules to those records, picks each feature's governing
category (``CATEGORY_PRIORITY``) and computes its due date from the discovery date (195.452(h)(2)).

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ili.align_welds import DRAFT_LABEL, comparison_dir
from ili.config import (DEFAULT_SETTINGS, PROJECT_ROOT, RepairCriteria, Settings, load_repair_criteria,
                        load_settings)
from ili.growth import PCT_PER_FRACTION, REASON_SEP
from ili.ingest import load_standardized
from ili.match import clock_diff_min, load_match_table, parse_flag_column
from ili.reference import _require_columns, assign_segment, check_segments, lookup
from ili.remaining_life import (PIPE_WHAT, RL_SEGMENT_COLUMNS, _as_date, _run_ids, _segment_hca,
                                load_remaining_life_table)
from ili.review import FLAG_COLUMNS, FLAGS_FILE, effective_flags, latest_flag_ids, load_flags_hashed
from ili.rules import _kleene_and, _kleene_or, _normalize, compile_condition
from ili.schema import ANOMALY_TYPES, FeatureType
from ili.strength import assessed_depth_in
from ili.vocabulary import VARIABLES

IN_PER_FT = 12.0
# A percentage is rounded to this many decimals before it is compared with an edge or a rule threshold, so an
# exact ratio stays exact (0.645 in on a 10.75 in OD is 6 %, not 6.000000000000001 %); the report's bins use it.
PCT_DECIMALS = 9
# Match-table rows that have a current-run side (a "missing" row is a previous-run feature only).
CURRENT_SIDE_MATCH_TYPES = ("matched_1to1", "matched_cluster", "new")
# Spec 6.8: "top of pipe" means above the 4 and 8 o'clock positions, i.e. clock_min >= 480 or <= 240.
TOP_OF_PIPE_FROM_MIN = 480  # 8:00
TOP_OF_PIPE_TO_MIN = 240  # 4:00
# Why a row needs review upstream of classification, beside M6's own reasons: the M4 match row needs
# review; the feature is outside every segment (HCA rules assumed); a metal-loss row has no M6 row.
REVIEW_MATCH, REVIEW_NO_SEGMENT, REVIEW_RL_MISSING = "match_review", "no_segment", "rl_missing"

FLAG_STATUS_FILE = "flag_status.csv"
FLAG_STATUS_COLUMNS = ["flag_id", "curr_feature_id", "flag", "applied_status", "reason"]
VARIABLE_COLUMNS = tuple(VARIABLES)
FEATURE_VARIABLE_COLUMNS = [
    "match_id", "curr_feature_id", "segment_id", *VARIABLE_COLUMNS,
    "repair_by_date", "upstream_needs_review", "upstream_review_reasons",
]
MATCH_INPUT_COLUMNS = (
    "match_id", "match_type", "needs_review", "curr_feature_id", "curr_feature_type", "curr_log_dist_ft",
    "curr_wt_nom_in", "curr_depth_pct", "curr_dent_depth_in", "curr_length_in", "curr_width_in",
    "curr_clock_min", "curr_seam_interaction", "curr_dent_metal_loss",
)
RL_INPUT_COLUMNS = (
    "match_id", "wt_in", "mop_psig", "assessed_dt", "burst_psig", "safe_psig",
    "dig_required_by_remaining_life", "repair_by_date", "needs_review", "review_reasons",
)


# --- value helpers ------------------------------------------------------------------------------


def _floats(values) -> np.ndarray:
    """Float array; None, NaN and pandas NA become NaN."""
    return pd.Series(values, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)


def _flag(value, what: str) -> bool | None:
    """True/False from a bool or a ``True``/``False`` string; None for None/NaN/NA/blank (unknown).
    ValueError naming ``what`` for anything else."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"
        if not text:
            return None
    raise ValueError(f"{what}: expected True, False or blank, got {value!r}")


def _flags(values, what: str) -> list:
    return [_flag(v, what) for v in values]


def _objects(values: list) -> np.ndarray:
    """1-D object array holding ``values`` as they are (Python bools and None)."""
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _tri(known: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Object array: Python bool ``test`` where ``known``, None elsewhere."""
    out = np.full(len(known), None, dtype=object)
    for k in np.flatnonzero(known):
        out[k] = bool(test[k])
    return out


# --- variable families --------------------------------------------------------------------------


def assessed_depth_pct(depth_pct, settings: Settings) -> np.ndarray:
    """Tolerance-adjusted depth, percent of wall, on the ``strength.depth_basis`` M6 uses
    (``strength.assessed_depth_in``). As a percentage of wall it does not depend on the wall, so it is
    evaluated on a unit wall: a missing wall never makes a reported depth unknown. NaN in, NaN out."""
    return np.asarray(assessed_depth_in(_floats(depth_pct), 1.0, settings), dtype=float) * PCT_PER_FRACTION


def dent_depth_pct_od(dent_depth_in, od_in) -> np.ndarray:
    """Dent depth as a percentage of OD = dent_depth_in / od_in x 100, rounded to ``PCT_DECIMALS`` places (the
    same value the report's dent bins use); NaN where either is NaN or the OD is not positive."""
    depth, od = np.asarray(dent_depth_in, dtype=float), np.asarray(od_in, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(od > 0, depth / od * PCT_PER_FRACTION, np.nan).round(PCT_DECIMALS)


def top_of_pipe(clock_min) -> np.ndarray:
    """Spec 6.8, verbatim: ``clock_min >= 480 or clock_min <= 240``; None when the clock is missing."""
    c = _floats(clock_min)
    with np.errstate(invalid="ignore"):
        return _tri(~np.isnan(c), (c >= TOP_OF_PIPE_FROM_MIN) | (c <= TOP_OF_PIPE_TO_MIN))


def dent_metal_loss(flag, depth_pct) -> np.ndarray:
    """Any indication of metal loss, cracking or a stress riser on a dent: the vendor flag OR a reported
    metal-loss depth > 0 on the dent.

    True when either source says so. None only when BOTH are unknown (flag None and no depth). Otherwise
    False -- in particular a vendor flag of False with no depth reported is False (the vendor reported the
    dent and no indication on it), and an unknown flag with a reported depth of 0 is False."""
    d = _floats(depth_pct)
    out = np.full(len(d), None, dtype=object)
    for k, f in enumerate(_flags(flag, "dent_metal_loss")):
        from_depth = None if np.isnan(d[k]) else bool(d[k] > 0)
        if f is True or from_depth is True:
            out[k] = True
        elif f is None and from_depth is None:
            out[k] = None
        else:
            out[k] = False
    return out


def seam_interaction(flag, clock_min, seam_clock_min, seam_clock_tol_min) -> np.ndarray:
    """Vendor seam flag OR clock overlap with the longitudinal seam, by Kleene OR.

    Overlap: the circular clock difference to ``seam_clock_min`` (on the 720-minute face) is
    <= ``seam_clock_tol_min``. It is unknown when either setting is null or the clock is missing. So the
    result is True when the vendor flag is True or the overlap is True, False only when both are known
    False, and None otherwise -- in particular, a vendor flag of False with the seam position unconfigured
    is None (the overlap is unknown), which leaves D180-8 not evaluable for metal loss without a vendor
    True flag until the PLACEHOLDER ``seam_clock_min`` is set. This is intended: it surfaces the gap."""
    vendor = _flags(flag, "seam_interaction")
    n = len(vendor)
    if seam_clock_min is None or seam_clock_tol_min is None:
        overlap = np.full(n, None, dtype=object)
    else:
        diff = clock_diff_min(clock_min, np.full(n, float(seam_clock_min)))
        with np.errstate(invalid="ignore"):
            overlap = _tri(~np.isnan(diff), diff <= seam_clock_tol_min)
    return _objects([_kleene_or([v, o]) for v, o in zip(vendor, overlap)])


def reference_positions(values, what: str) -> np.ndarray:
    """Sorted log distances (ft) of a reference table's points. ValueError naming ``what`` and the row
    (0-based, in the order given) for a blank position: an incomplete reference table is a data error,
    never a point silently dropped (it could be the one next to a feature)."""
    v = _floats(values)
    bad = np.flatnonzero(np.isnan(v))
    if bad.size:
        raise ValueError(f"{what}: row {bad[0]} has a blank log_dist_ft")
    return np.sort(v)


def _nearest_gap(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    """|x - nearest point| for each x (points sorted, non-empty); NaN where x is NaN."""
    k = np.clip(np.searchsorted(points, np.nan_to_num(x)), 1, max(points.size - 1, 1))
    return np.minimum(np.abs(x - points[k - 1]), np.abs(x - points[np.minimum(k, points.size - 1)]))


def near_girth_weld(pos_ft, length_in, weld_pos_ft, proximity_in) -> np.ndarray:
    """Edge distance to the nearest current-run girth weld <= ``proximity_in`` (inches).

    Edge distance = max(|x - w| x 12 - length / 2, 0), with x the feature's and w the nearest weld's log
    distance (ft) and length the feature's axial length (in). Inside the run's weld span (first to last
    weld) the result is True or False. Outside it an unrecorded weld could lie beyond the last known one,
    so the result is True when within ``proximity_in`` of the nearest known weld and None otherwise.
    None when the setting is null, the position or length is unknown, or the run has no girth welds.
    ValueError for a blank weld position (``reference_positions``)."""
    x, length = _floats(pos_ft), _floats(length_in)
    welds = reference_positions(weld_pos_ft, "current-run girth welds")
    if proximity_in is None or welds.size == 0:
        return np.full(len(x), None, dtype=object)
    edge_in = np.maximum(_nearest_gap(x, welds) * IN_PER_FT - length / 2, 0.0)
    with np.errstate(invalid="ignore"):
        near = edge_in <= proximity_in
        inside = (x >= welds[0]) & (x <= welds[-1])
    known = ~np.isnan(x) & ~np.isnan(length) & (inside | near)
    return _tri(known, near)


def near_foreign_crossing(pos_ft, crossing_pos_ft, proximity_ft) -> np.ndarray:
    """|x - crossing| <= ``proximity_ft`` for any foreign crossing (log distances, ft).

    None when the setting is null, the position is unknown, or there is no crossings table
    (``crossing_pos_ft`` None: whether crossings exist is unknown). A crossings table with no rows gives
    False (known: there are none). ValueError for a blank crossing position (``reference_positions``)."""
    x = _floats(pos_ft)
    c = None if crossing_pos_ft is None else reference_positions(crossing_pos_ft, "foreign crossings table")
    if proximity_ft is None or c is None:
        return np.full(len(x), None, dtype=object)
    if c.size == 0:
        return _tri(~np.isnan(x), np.zeros(len(x), dtype=bool))
    with np.errstate(invalid="ignore"):
        return _tri(~np.isnan(x), _nearest_gap(x, c) <= proximity_ft)


def general_corrosion(length_in, width_in, min_dim_in) -> np.ndarray:
    """General corrosion (rather than pitting): length >= ``min_dim_in`` AND width >= it, by Kleene AND
    (a known side below it decides False even if the other side is unknown). None when the setting is
    null."""
    length, width = _floats(length_in), _floats(width_in)
    if min_dim_in is None:
        return np.full(len(length), None, dtype=object)
    with np.errstate(invalid="ignore"):
        sides = [_tri(~np.isnan(v), v >= min_dim_in) for v in (length, width)]
    return _objects([_kleene_and([a, b]) for a, b in zip(*sides)])


def widespread_circumferential(width_in, od_in, min_width_frac) -> np.ndarray:
    """Widespread circumferential corrosion: width / (pi x OD) >= ``min_width_frac``. None when the
    setting is null, or the width or a positive OD is unknown."""
    width, od = _floats(width_in), _floats(od_in)
    if min_width_frac is None:
        return np.full(len(width), None, dtype=object)
    known = ~np.isnan(width) & (od > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _tri(known, width / (np.pi * od) >= min_width_frac)


# --- assembly -----------------------------------------------------------------------------------


def _review(match_flag, rl_flag, rl_reasons, has_rl: bool, no_segment: bool, rl_missing: bool):
    """(needs_review, ;-joined reasons or None). An unknown M4 or M6 flag counts as needing review."""
    reasons = []
    needs = _flag(match_flag, "match needs_review") is not False
    if needs:
        reasons.append(REVIEW_MATCH)
    if has_rl:
        needs |= _flag(rl_flag, "remaining-life needs_review") is not False
        if isinstance(rl_reasons, str):
            reasons += [r for r in rl_reasons.split(REASON_SEP) if r]
    reasons += [REVIEW_NO_SEGMENT] * no_segment + [REVIEW_RL_MISSING] * rl_missing
    reasons = list(dict.fromkeys(reasons))
    return needs or bool(reasons), (REASON_SEP.join(reasons) if reasons else None)


def _rl_rows(rows: pd.DataFrame, remaining_life: pd.DataFrame, is_ml: np.ndarray) -> np.ndarray:
    """Row index into ``remaining_life`` for each metal-loss row by match_id, -1 when absent (or not
    metal loss). ValueError for a match_id repeated in the remaining-life table."""
    ids = remaining_life.match_id.astype(str)
    dup = ids[ids.duplicated()]
    if not dup.empty:
        raise ValueError(f"remaining-life table repeats match_id {dup.iloc[0]!r}")
    where = {m: j for j, m in enumerate(ids)}
    return np.array([where.get(str(m), -1) if ml else -1 for m, ml in zip(rows.match_id, is_ml)], dtype=int)


def _take(values, j: np.ndarray) -> np.ndarray:
    """``values[j]`` as floats, NaN where j < 0."""
    v = _floats(values)
    out = np.full(len(j), np.nan)
    has = j >= 0
    out[has] = v[j[has]]
    return out


def _manual_flag(flags: dict[tuple[str, str], bool] | None, feature_ids, flag: str) -> list[bool]:
    """Per row: True when ``flags`` holds True for ``flag`` on any ``;``-joined member of the id cell."""
    flags = flags or {}
    return [any(flags.get((member, flag), False) for member in str(cell).split(REASON_SEP)) for cell in feature_ids]


def flag_status(flags: pd.DataFrame, classified: set[str], known: set[str]) -> pd.DataFrame:
    """What became of every row of the flag log (``review.load_flags``), in log order: ``flag_id``,
    ``curr_feature_id``, ``flag``, ``applied_status`` and ``reason`` (None when applied).

    ``superseded``: a later row for the same feature and flag decides. ``not_applied``: ``unknown_feature``
    (the id is not in the current run, ``known``) or ``not_anomaly`` (in the run but not a classified
    current-run anomaly, ``classified``: ids of the feature-variables rows, cluster members split)."""
    latest = latest_flag_ids(flags)
    rows = []
    for r in flags.itertuples(index=False):
        if r.flag_id not in latest:
            status, reason = "superseded", None
        elif r.curr_feature_id not in known:
            status, reason = "not_applied", "unknown_feature"
        elif r.curr_feature_id not in classified:
            status, reason = "not_applied", "not_anomaly"
        else:
            status, reason = "applied", None
        rows.append((r.flag_id, r.curr_feature_id, r.flag, status, reason))
    return pd.DataFrame(rows, columns=FLAG_STATUS_COLUMNS, dtype=object)


def feature_variables(match_table: pd.DataFrame, remaining_life: pd.DataFrame, curr_std: pd.DataFrame,
                      pipe: pd.DataFrame, segments: pd.DataFrame, crossings: pd.DataFrame | None,
                      settings: Settings, flags: dict[tuple[str, str], bool] | None = None) -> pd.DataFrame:
    """One row per current-run anomaly of the M4 match table, columns ``FEATURE_VARIABLE_COLUMNS``.

    Rows: match types ``CURRENT_SIDE_MATCH_TYPES`` with ``curr_feature_type`` metal_loss, dent or gouge,
    in table order (a cluster row is metal loss with the envelope's dimensions). Values:

    - ``feature_type``, ``clock_min`` (rounded to the minute), ``length_in``, ``width_in``,
      ``depth_pct_reported``: the match table's current side. ``top_of_pipe`` and the seam overlap use the
      same rounded ``clock_min``. A current-side row with a blank feature type is a ValueError.
    - Reference points: a blank girth-weld or crossing position is a ValueError (``reference_positions``).
    - ``od_in``: the pipe table at ``curr_log_dist_ft`` (``reference.lookup``, ``alignment.anchor_tol_ft``;
      a position outside it is a ValueError, a blank OD is None).
    - Metal loss, from ``remaining_life`` (M6) by match_id: ``p_fail`` = burst_psig, ``p_safe`` =
      safe_psig, ``mop_local`` = mop_psig, ``dig_required_by_remaining_life``, ``repair_by_date``, and
      ``depth_pct_assessed`` = assessed_dt x 100 and ``wt_in``. A metal-loss row missing from M6 has None
      for the pressures, dig flag and repair-by date and is flagged ``rl_missing``.
    - ``wt_in`` otherwise (and for a blank M6 value): the match table's ``curr_wt_nom_in``.
      ``depth_pct_assessed`` otherwise: ``assessed_depth_pct`` of the reported depth (gouges; dents
      with a reported metal-loss depth; metal loss without an M6 value), None without a depth.
    - Dents only: ``dent_depth_in``, ``dent_depth_pct_od`` = dent_depth_in / od_in x 100 rounded to
      ``PCT_DECIMALS`` places (``dent_depth_pct_od``, as the report's dent bins),
      ``dent_metal_loss``. None for metal loss and gouges, as are the pressures for dents and gouges.
    - ``seam_interaction``, ``near_girth_weld`` (current-run girth welds of ``curr_std``),
      ``near_foreign_crossing`` (``crossings`` None: unknown; no rows: False), ``general_corrosion``,
      ``widespread_circumferential``: the helpers of the same names, with the ``classification`` settings.
    - ``could_affect_hca``: the segment's flag (``segments``: start_ft, end_ft, segment_id,
      could_affect_hca); outside every segment it is True (HCA rules, the conservative choice) and the
      row is flagged ``no_segment``.
    - ``manual_immediate``, ``field_crack_confirmed``: from ``flags`` (``review.effective_flags``: the latest
      value per (current feature id, flag)). A cluster row is True when any member's flag is True; False
      otherwise, including when no flag was set or ``flags`` is None.
    - ``upstream_needs_review``: M4 needs_review OR M6 needs_review (unknown counts as True) OR any
      reason; ``upstream_review_reasons``: ``match_review``, M6's reasons, ``no_segment``,
      ``rl_missing``, ``;``-joined without repeats, None when there are none.
    """
    _require_columns(match_table, MATCH_INPUT_COLUMNS, "match table")
    _require_columns(remaining_life, RL_INPUT_COLUMNS, "remaining-life table")
    _require_columns(segments, RL_SEGMENT_COLUMNS, "segments table")
    _require_columns(curr_std, ("feature_type", "log_dist_ft"), "current standardized run")
    if crossings is not None:
        _require_columns(crossings, ("log_dist_ft",), "foreign crossings table")
    check_segments(segments)
    cfg = settings.classification
    current = match_table.match_type.isin(CURRENT_SIDE_MATCH_TYPES).to_numpy(bool)
    blank = current & match_table.curr_feature_type.isna().to_numpy(bool)
    if blank.any():
        raise ValueError(f"match {match_table.match_id[blank].iloc[0]!r} has a current-run side but no "
                         f"curr_feature_type")
    keep = current & match_table.curr_feature_type.isin(ANOMALY_TYPES).to_numpy(bool)
    rows = match_table[keep].reset_index(drop=True)
    rl = remaining_life.reset_index(drop=True)
    n = len(rows)

    ftype = np.array([str(t) for t in rows.curr_feature_type], dtype=object)
    is_ml, is_dent = ftype == FeatureType.metal_loss.value, ftype == FeatureType.dent.value
    pos = _floats(rows.curr_log_dist_ft)
    od = (lookup(pipe, pos, "od_in", settings.alignment.anchor_tol_ft, what=PIPE_WHAT, allow_nan=True)
          if n else np.zeros(0))
    length, width = _floats(rows.curr_length_in), _floats(rows.curr_width_in)
    clock = np.round(_floats(rows.curr_clock_min))  # the reported clock_min; every clock rule uses it
    depth = _floats(rows.curr_depth_pct)

    j = _rl_rows(rows, rl, is_ml)
    has_rl = j >= 0
    rl_wt, rl_dt = _take(rl.wt_in, j), _take(rl.assessed_dt, j)
    wt = np.where(~np.isnan(rl_wt), rl_wt, _floats(rows.curr_wt_nom_in))
    depth_assessed = np.where(~np.isnan(rl_dt), rl_dt * PCT_PER_FRACTION, assessed_depth_pct(depth, settings))
    dig = np.full(n, None, dtype=object)
    repair_by_date = np.full(n, None, dtype=object)
    for k in np.flatnonzero(has_rl):
        dig[k] = _flag(rl.dig_required_by_remaining_life.iloc[j[k]], "dig_required_by_remaining_life")
        repair_by_date[k] = _as_date(rl.repair_by_date.iloc[j[k]], "remaining-life repair_by_date")

    dent_depth = np.where(is_dent, _floats(rows.curr_dent_depth_in), np.nan)
    dent_pct_od = dent_depth_pct_od(dent_depth, od)
    dml = dent_metal_loss(rows.curr_dent_metal_loss, depth)
    dml[~is_dent] = None

    welds = reference_positions(
        curr_std.log_dist_ft[curr_std.feature_type.astype(str) == FeatureType.girth_weld.value],
        "current-run girth welds")
    cross = None if crossings is None else reference_positions(crossings.log_dist_ft, "foreign crossings table")

    seg_id = assign_segment(segments, pos)
    hca = _segment_hca(segments)
    review = [_review(rows.needs_review.iloc[k], rl.needs_review.iloc[j[k]] if has_rl[k] else None,
                      rl.review_reasons.iloc[j[k]] if has_rl[k] else None, bool(has_rl[k]),
                      seg_id[k] is None, bool(is_ml[k] and not has_rl[k])) for k in range(n)]

    def obj(values) -> pd.Series:
        return pd.Series(list(values), dtype=object)

    out = pd.DataFrame({
        "match_id": obj(rows.match_id), "curr_feature_id": obj(rows.curr_feature_id), "segment_id": obj(seg_id),
        "feature_type": obj(ftype),
        "could_affect_hca": obj(True if s is None else hca[s] for s in seg_id),
        "od_in": od, "wt_in": wt, "clock_min": pd.array(clock, dtype="Int64"),
        "top_of_pipe": obj(top_of_pipe(clock)),
        "depth_pct_reported": depth, "depth_pct_assessed": depth_assessed, "length_in": length, "width_in": width,
        "p_fail": _take(rl.burst_psig, j), "p_safe": _take(rl.safe_psig, j), "mop_local": _take(rl.mop_psig, j),
        "dent_depth_in": dent_depth, "dent_depth_pct_od": dent_pct_od, "dent_metal_loss": obj(dml),
        "seam_interaction": obj(seam_interaction(rows.curr_seam_interaction, clock, cfg.seam_clock_min,
                                                 cfg.seam_clock_tol_min)),
        "near_girth_weld": obj(near_girth_weld(pos, length, welds, cfg.girth_weld_proximity_in)),
        "near_foreign_crossing": obj(near_foreign_crossing(pos, cross, cfg.crossing_proximity_ft)),
        "general_corrosion": obj(general_corrosion(length, width, cfg.general_corrosion_min_dim_in)),
        "widespread_circumferential": obj(widespread_circumferential(width, od, cfg.widespread_circ_min_width_frac)),
        "dig_required_by_remaining_life": obj(dig),
        "manual_immediate": obj(_manual_flag(flags, rows.curr_feature_id, "manual_immediate")),
        "field_crack_confirmed": obj(_manual_flag(flags, rows.curr_feature_id, "field_crack_confirmed")),
        "repair_by_date": obj(repair_by_date),
        "upstream_needs_review": obj(bool(r[0]) for r in review), "upstream_review_reasons": obj(r[1] for r in review),
    })
    return out[FEATURE_VARIABLE_COLUMNS]


def variable_dicts(variables: pd.DataFrame) -> list[dict]:
    """One plain dict per row of a ``feature_variables`` table, keyed by ``VARIABLES`` (in vocabulary order)
    and ready for ``rules.Condition.evaluate``: numpy and pandas scalars become bool, int, float or str,
    and NaN/NA become None. (Never hand the evaluator a pandas row.)"""
    cols = list(VARIABLE_COLUMNS)
    return [dict(zip(cols, (_normalize(v) for v in row)))
            for row in variables[cols].astype(object).itertuples(index=False, name=None)]


# --- classification and due dates (spec 6.8) ------------------------------------------------------

# Governing-category priority, most severe first. Every current anomaly gets at least ``evaluate``
# ((h)(4)(iv)); a metal-loss feature M6 flags by remaining life gets ``imp_scheduled``.
CATEGORY_PRIORITY = ("immediate", "60_day", "180_day", "imp_scheduled", "evaluate")
# Spec 6.8: immediate conditions require a temporary pressure reduction or shutdown until repaired.
IMMEDIATE_ACTION_NOTE = "Requires temporary pressure reduction or shutdown until repaired."
# Pseudo-rule ids for the categories no criteria-file rule decides.
GOVERNING_REMAINING_LIFE, GOVERNING_EVALUATE = "REMAINING_LIFE", "EVALUATE"
RULE_SET_HCA, RULE_SET_NON_HCA, RULE_SET_HCA_ASSUMED = "hca", "non_hca", "hca_assumed"
# Review reasons classification adds after the upstream ones, in this order.
REVIEW_RULE, REVIEW_NOT_EVALUABLE = "rule_review", "rule_not_evaluable"
# The worst case (a not-evaluable rule, or the other side of an unknown top-of-pipe split) outranks the
# governing category: the feature could be more severe than it is classified.
REVIEW_COULD_BE_MORE_SEVERE = "could_be_more_severe"
REVIEW_HCA_UNKNOWN, REVIEW_NH_PLACEHOLDER = "hca_unknown", "nh_placeholder"
REVIEW_DUE_DATE, REVIEW_OVERDUE = "due_date_unknown", "overdue_at_discovery"
DISCOVERY_ENTERED, DISCOVERY_LATEST = "entered", "latest_permissible"
# Why a due date is what it is: the category's regulatory timeline, or M6's remaining-life repair-by date.
BASIS_REGULATORY, BASIS_REMAINING_LIFE = "regulatory", "remaining_life"
DUE_DATE_BASES = (BASIS_REGULATORY, BASIS_REMAINING_LIFE)
DISCOVERY_WARNING = ("Discovery date not entered: using the latest permissible date (assessment date + {days} "
                     "days) per 195.452(h)(2). Enter the actual discovery date.")
CLASSIFICATION_COLUMNS = [
    *FEATURE_VARIABLE_COLUMNS, "rule_set", "triggered_rules", "rules_not_evaluable", "governing_category",
    "governing_rule", "requires_review", "review_reasons", "discovery_date", "discovery_date_source",
    "due_date", "due_date_basis", "worst_case_category", "worst_case_rule", "worst_case_due_date", "action_note",
]


_CLASSIFICATION_FLAGS = (
    "could_affect_hca", "top_of_pipe", "dent_metal_loss", "seam_interaction", "near_girth_weld",
    "near_foreign_crossing", "general_corrosion", "widespread_circumferential", "dig_required_by_remaining_life",
    "manual_immediate", "field_crack_confirmed", "upstream_needs_review", "requires_review",
)
_CLASSIFICATION_DATES = ("repair_by_date", "discovery_date", "due_date", "worst_case_due_date")
_CLASSIFICATION_FLOATS = (
    "od_in", "wt_in", "depth_pct_reported", "depth_pct_assessed", "length_in", "width_in", "p_fail", "p_safe",
    "mop_local", "dent_depth_in", "dent_depth_pct_od",
)


def load_classification_table(path: Path) -> pd.DataFrame:
    """Read a ``classification.csv`` back with its types (as ``remaining_life.load_remaining_life_table``):
    flags are True/False/None (``match.parse_flag_column``: any case), dates are ``datetime.date`` or
    None, ``clock_min`` is Int64, the measurements are float, and every other column (ids included)
    stays text, empty is None. ValueError for a missing column, a bad flag or a date that is not ISO."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in CLASSIFICATION_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    numeric = (*_CLASSIFICATION_FLOATS, "clock_min")
    as_text = [c for c in CLASSIFICATION_COLUMNS if c not in numeric]
    df = pd.read_csv(path, dtype={c: str for c in as_text}, keep_default_na=False,
                     na_values={c: [""] for c in numeric})
    for c in (c for c in as_text if c not in _CLASSIFICATION_FLAGS):
        df[c] = df[c].astype(object).where(df[c] != "", None)
    for c in _CLASSIFICATION_FLAGS:
        df[c] = parse_flag_column(df[c], path, c)
    for c in _CLASSIFICATION_DATES:
        df[c] = pd.Series([_as_date(v, f"{path}: {c}") for v in df[c]], dtype=object, index=df.index)
    df["clock_min"] = df.clock_min.astype("Int64")
    return df[CLASSIFICATION_COLUMNS]


@dataclass(frozen=True)
class ClassificationResult:
    table: pd.DataFrame  # CLASSIFICATION_COLUMNS, one row per feature-variables row, same order
    summary: dict  # JSON-safe


def resolve_discovery_date(assessment_date, discovery_date, max_days: int) -> tuple[date, str]:
    """(discovery date, source). An entered date (``entered``) must lie in [assessment date, assessment
    date + ``max_days``] (195.452(h)(2); the limit itself is allowed), else ValueError. Without one, the
    latest permissible date, assessment date + ``max_days`` (``latest_permissible``). Dates may be
    ``datetime.date``, datetime/Timestamp or ISO text; a blank discovery date counts as not entered."""
    assessed = _as_date(assessment_date, "assessment date")
    if assessed is None:
        raise ValueError("the assessment date is missing")
    latest = assessed + timedelta(days=int(max_days))
    entered = _as_date(discovery_date, "discovery date")
    if entered is None:
        return latest, DISCOVERY_LATEST
    if entered < assessed:
        raise ValueError(f"discovery date {entered.isoformat()} is before the assessment date "
                         f"{assessed.isoformat()}")
    if entered > latest:
        raise ValueError(f"discovery date {entered.isoformat()} is more than {max_days} days after the assessment "
                         f"date {assessed.isoformat()} (latest permissible {latest.isoformat()}, 195.452(h)(2))")
    return entered, DISCOVERY_ENTERED


def _rule_set(could_affect_hca, upstream_reasons: list[str]) -> tuple[str, str, list[str]]:
    """(rule_set, the ``applies_to`` value it uses, extra review reasons). An unknown HCA flag uses the
    HCA rules (the conservative choice) and is flagged ``hca_unknown``; a feature outside every segment
    (upstream ``no_segment``, HCA already assumed by ``feature_variables``) is ``hca_assumed``."""
    if could_affect_hca is None:
        return RULE_SET_HCA_ASSUMED, "hca", [REVIEW_HCA_UNKNOWN]
    if could_affect_hca:
        return (RULE_SET_HCA_ASSUMED if REVIEW_NO_SEGMENT in upstream_reasons else RULE_SET_HCA), "hca", []
    return RULE_SET_NON_HCA, "non_hca", []


def _due_date(category: str, discovered: date, due_days: dict, repair_by: date | None):
    """(due_date, due_date_basis). The regulatory date of the category -- ``immediate``: the discovery
    date; ``60_day`` / ``180_day``: discovery + ``due_days``; none for ``imp_scheduled`` and ``evaluate``
    -- or M6's remaining-life ``repair_by`` when that is known and earlier: the regulation is an outer
    limit and the engineering remaining life can only make it sooner (a tie is ``regulatory``).
    ``imp_scheduled`` is ``repair_by`` (None when unknown); ``evaluate`` is None (basis None)."""
    if category == "evaluate":
        return None, None
    if category == "immediate":
        regulatory = discovered
    elif category in ("60_day", "180_day"):
        regulatory = discovered + timedelta(days=int(due_days[category]))
    else:  # imp_scheduled
        return repair_by, (BASIS_REMAINING_LIFE if repair_by is not None else None)
    if repair_by is not None and repair_by < regulatory:
        return repair_by, BASIS_REMAINING_LIFE
    return regulatory, BASIS_REGULATORY


def _join(items: list[str]):
    return REASON_SEP.join(items) if items else None


def _evaluate(compiled, d: dict, applies: str) -> tuple[list, list[str]]:
    """(triggered rules, ids of not-evaluable rules), both in criteria-file order, of the rules that apply to
    the feature (``applies_to`` ``all`` or ``applies``, ``feature_types`` holding its type)."""
    triggered, unknown = [], []
    for rule, cond, types in compiled:
        if rule.applies_to not in ("all", applies) or d["feature_type"] not in types:
            continue
        value = cond.evaluate(d)
        if value is True:
            triggered.append(rule)
        elif value is None:
            unknown.append(rule.id)
    return triggered, unknown


def _outcome(triggered: list, default: tuple[str, str], rank: dict) -> tuple[str, str]:
    """(category, governing rule): the highest-priority triggered category, or ``default`` when that is
    higher; the rule is the first triggered one in file order within it, else the default's pseudo-rule."""
    category = min([r.category for r in triggered] + [default[0]], key=rank.__getitem__)
    return category, next((r.id for r in triggered if r.category == category), default[1])


def classify(variables: pd.DataFrame, criteria: RepairCriteria, settings: Settings, assessment_date,
             discovery_date=None, criteria_sha256: str | None = None, criteria_path: str | None = None,
             crossings_file_given: bool | None = None) -> ClassificationResult:
    """Classify every row of a ``feature_variables`` table (spec 6.8). Columns ``CLASSIFICATION_COLUMNS``.

    - **Rules.** A rule applies when its ``applies_to`` is ``all`` or matches the feature's rule set
      (``could_affect_hca`` True: ``hca``; False: ``non_hca``; None: the HCA rules, ``hca_assumed``,
      reason ``hca_unknown``; outside every segment, upstream ``no_segment``: ``hca_assumed``) and its
      ``feature_types`` include the feature's. Each applicable condition is evaluated (Kleene) on the
      plain dict of ``variable_dicts``: True is triggered (``triggered_rules``), None is not evaluable
      (``rules_not_evaluable``), both ``;``-joined in criteria-file order.
    - **Governing category.** The highest ``CATEGORY_PRIORITY`` category of a triggered rule, or the
      default -- ``imp_scheduled`` when ``dig_required_by_remaining_life`` is True, else ``evaluate`` --
      when that is higher. ``governing_rule``: the first triggered rule in file order within the governing
      category, else ``REMAINING_LIFE`` / ``EVALUATE``. A not-evaluable rule NEVER raises or lowers the
      category (an unknown is neither a trigger nor a pass): it is listed and requires review. A
      metal-loss feature with an unknown dig flag is never ``imp_scheduled`` by guess; ``REMAINING_LIFE``
      is listed as not evaluable.
    - **Unknown top of pipe** (blank clock). The rules are evaluated twice, with ``top_of_pipe`` True and
      False. The less severe outcome (ties: the bottom case) is a guaranteed floor and governs, with its
      rule and due date; its triggered rules are ``triggered_rules``, and every other rule the Kleene
      evaluation left unknown stays in ``rules_not_evaluable``. The more severe outcome feeds the worst case.
    - **Worst case** (every feature). ``worst_case_category``: the most severe of the governing category,
      the category of every not-evaluable rule (``REMAINING_LIFE``: ``imp_scheduled``) and the other side
      of a top-of-pipe split; ``worst_case_rule``: the governing rule when the worst case is no worse, else
      the first such rule in file order; ``worst_case_due_date``: ``_due_date`` of the worst category.
    - **Review.** ``review_reasons``: the upstream reasons, then ``rule_review`` (a triggered rule has
      ``requires_review``), ``rule_not_evaluable``, ``could_be_more_severe`` (the worst case outranks the
      governing category), ``hca_unknown``, ``nh_placeholder`` (every ``non_hca``
      feature, rule fired or not: the NH set is a PLACEHOLDER pending IMP criteria), ``due_date_unknown``,
      ``overdue_at_discovery`` (a due date before the discovery date, any category), without repeats
      (None when there are none). ``requires_review``: the upstream flag (unknown counts as True) or any
      reason.
    - **Dates.** ``resolve_discovery_date`` (``classification.discovery_max_days``). ``due_date`` and
      ``due_date_basis`` (``_due_date``): the earlier of the category's regulatory date (``immediate`` the
      discovery date; ``60_day`` / ``180_day`` discovery + ``classification.due_days``) and M6's known
      ``repair_by_date``; ``imp_scheduled`` is ``repair_by_date`` (None, flagged ``due_date_unknown``,
      when unknown); ``evaluate`` None. ``action_note``: ``IMMEDIATE_ACTION_NOTE`` for ``immediate``,
      else None.

    The summary (``_classification_summary``) records, for traceability, the criteria file's version and
    ``criteria_sha256`` / ``criteria_path`` (both given by the caller), the assessment and discovery dates,
    every ``classification`` setting, whether a crossings file was given (``crossings_file_given``, from the
    caller; None when unknown) and the depth basis of the assessed depths (``strength.depth_basis``,
    ``tool.depth_tolerance_pct_wt``).
    """
    _require_columns(variables, FEATURE_VARIABLE_COLUMNS, "feature variables")
    cfg = settings.classification
    discovered, source = resolve_discovery_date(assessment_date, discovery_date, cfg.discovery_max_days)
    names = set(VARIABLES)
    compiled = [(r, compile_condition(r.condition, names), frozenset(t.value for t in r.feature_types))
                for r in criteria.rules]  # once per run
    rank = {c: k for k, c in enumerate(CATEGORY_PRIORITY)}
    rule_category = {r.id: r.category for r in criteria.rules} | {GOVERNING_REMAINING_LIFE: "imp_scheduled"}
    rows = variables.reset_index(drop=True)
    out = {c: [] for c in CLASSIFICATION_COLUMNS[len(FEATURE_VARIABLE_COLUMNS):]}
    for k, d in enumerate(variable_dicts(rows)):
        upstream = rows.upstream_review_reasons.iloc[k]
        reasons = [r for r in upstream.split(REASON_SEP) if r] if isinstance(upstream, str) else []
        rule_set, applies, extra = _rule_set(d["could_affect_hca"], reasons)
        dig = d["dig_required_by_remaining_life"]
        default = ("imp_scheduled", GOVERNING_REMAINING_LIFE) if dig is True else ("evaluate", GOVERNING_EVALUATE)
        triggered, unknown = _evaluate(compiled, d, applies)  # Kleene, on the values as known
        other_case = None
        if d["top_of_pipe"] is None:
            # Unknown clock: top and bottom are complementary (IMM-3/D60-2, IMM-4/D180-3), so the feature is
            # at least the less severe of the two outcomes. Evaluate both; the floor case (ties: bottom)
            # governs, the other case feeds the worst case. A rule the floor case fires is triggered; every
            # other rule the Kleene evaluation left unknown stays not evaluable.
            cases = []
            for top in (False, True):
                case_triggered, _ = _evaluate(compiled, {**d, "top_of_pipe": top}, applies)
                cases.append((*_outcome(case_triggered, default, rank), case_triggered))
            floor = max(cases, key=lambda c: rank[c[0]])  # least severe; the first (bottom) on a tie
            other_case = cases[1] if floor is cases[0] else cases[0]
            fired = {r.id for r in floor[2]}
            triggered, unknown = floor[2], [i for i in unknown if i not in fired]
        if d["feature_type"] == FeatureType.metal_loss.value and dig is None:
            unknown.append(GOVERNING_REMAINING_LIFE)
        category, governing = _outcome(triggered, default, rank)

        # Worst case: the most severe of the governing category, every not-evaluable rule's category and the
        # other side of a top-of-pipe split. Its rule is the governing rule when it is no worse, else the
        # first in file order of that category.
        candidates = [(rule_category[i], i) for i in unknown]
        if other_case is not None:
            candidates.append(other_case[:2])
        worst = min([category] + [c for c, _ in candidates], key=rank.__getitem__)
        worst_rule = governing if worst == category else next(i for c, i in candidates if c == worst)

        repair_by = _as_date(rows.repair_by_date.iloc[k], "repair_by_date")
        due, basis = _due_date(category, discovered, cfg.due_days, repair_by)
        worst_due = due if worst == category else _due_date(worst, discovered, cfg.due_days, repair_by)[0]
        reasons += [REVIEW_RULE] * any(r.requires_review for r in triggered)
        reasons += [REVIEW_NOT_EVALUABLE] * bool(unknown)
        reasons += [REVIEW_COULD_BE_MORE_SEVERE] * (rank[worst] < rank[category]) + extra
        reasons += [REVIEW_NH_PLACEHOLDER] * (rule_set == RULE_SET_NON_HCA)
        reasons += [REVIEW_DUE_DATE] * (category == "imp_scheduled" and due is None)
        reasons += [REVIEW_OVERDUE] * (due is not None and due < discovered)
        reasons = list(dict.fromkeys(reasons))
        upstream_flag = _flag(rows.upstream_needs_review.iloc[k], "upstream_needs_review")
        for col, value in (
            ("rule_set", rule_set), ("triggered_rules", _join([r.id for r in triggered])),
            ("rules_not_evaluable", _join(unknown)), ("governing_category", category),
            ("governing_rule", governing), ("requires_review", upstream_flag is not False or bool(reasons)),
            ("review_reasons", _join(reasons)), ("discovery_date", discovered),
            ("discovery_date_source", source), ("due_date", due), ("due_date_basis", basis),
            ("worst_case_category", worst), ("worst_case_rule", worst_rule), ("worst_case_due_date", worst_due),
            ("action_note", IMMEDIATE_ACTION_NOTE if category == "immediate" else None),
        ):
            out[col].append(value)

    table = rows.copy()
    for col, values in out.items():
        table[col] = pd.Series(values, dtype=bool if col == "requires_review" else object, index=table.index)
    table = table[CLASSIFICATION_COLUMNS]
    return ClassificationResult(table, _classification_summary(
        table, criteria, settings, discovered, source, assessment=_as_date(assessment_date, "assessment date"),
        criteria_sha256=criteria_sha256, criteria_path=criteria_path, crossings_file_given=crossings_file_given))


def _count_ids(joined, order: list[str]) -> dict[str, int]:
    """Occurrences of each id in the ``;``-joined cells, in ``order``, only ids that occur."""
    hits = [x for cell in joined if isinstance(cell, str) for x in cell.split(REASON_SEP)]
    return {i: hits.count(i) for i in order if i in hits}


def _has_reason(joined, reason: str) -> np.ndarray:
    return np.array([isinstance(c, str) and reason in c.split(REASON_SEP) for c in joined], dtype=bool)


def _reason_counts(joined) -> dict[str, int]:
    """Features per review reason, every reason, most frequent first (ties by name)."""
    hits = [x for cell in joined if isinstance(cell, str) for x in cell.split(REASON_SEP) if x]
    counts = {r: hits.count(r) for r in set(hits)}
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _classification_summary(table: pd.DataFrame, criteria: RepairCriteria, settings: Settings, discovered: date,
                            source: str, *, assessment: date, criteria_sha256: str | None,
                            criteria_path: str | None, crossings_file_given: bool | None) -> dict:
    """JSON-safe summary: criteria identity, counts, dates, and every input setting (traceability)."""
    cfg = settings.classification
    ids = [r.id for r in criteria.rules]
    return {
        "criteria_version": criteria.version,
        "criteria_sha256": criteria_sha256,
        "criteria_path": criteria_path,
        "n_features": int(len(table)),
        "counts_by_category": {c: int((table.governing_category == c).sum()) for c in CATEGORY_PRIORITY},
        "counts_by_rule": _count_ids(table.triggered_rules, ids),
        "n_requires_review": int(table.requires_review.sum()),
        "counts_by_review_reason": _reason_counts(table.review_reasons),
        "could_be_more_severe_by_worst_category": {
            c: int(((table.worst_case_category == c).to_numpy(bool)
                    & _has_reason(table.review_reasons, REVIEW_COULD_BE_MORE_SEVERE)).sum())
            for c in CATEGORY_PRIORITY},
        "n_overdue_at_discovery": int(sum(d is not None and d < discovered for d in table.due_date)),
        "counts_by_due_date_basis": {b: int((table.due_date_basis == b).sum()) for b in DUE_DATE_BASES},
        "n_not_evaluable_by_rule": _count_ids(table.rules_not_evaluable, [*ids, GOVERNING_REMAINING_LIFE]),
        "assessment_date": assessment.isoformat(),
        "discovery_date": discovered.isoformat(),
        "discovery_date_source": source,
        "discovery_warning": (DISCOVERY_WARNING.format(days=cfg.discovery_max_days)
                              if source == DISCOVERY_LATEST else None),
        # every classification setting (criteria_file is criteria_path above)
        "girth_weld_proximity_in": _optional_float(cfg.girth_weld_proximity_in),
        "crossing_proximity_ft": _optional_float(cfg.crossing_proximity_ft),
        "seam_clock_min": _optional_float(cfg.seam_clock_min),
        "seam_clock_tol_min": _optional_float(cfg.seam_clock_tol_min),
        "general_corrosion_min_dim_in": _optional_float(cfg.general_corrosion_min_dim_in),
        "widespread_circ_min_width_frac": _optional_float(cfg.widespread_circ_min_width_frac),
        "due_days": {k: int(v) for k, v in cfg.due_days.items()},
        "discovery_max_days": int(cfg.discovery_max_days),
        "crossings_file_given": None if crossings_file_given is None else bool(crossings_file_given),
        # the basis of depth_pct_assessed (the same as M6)
        "depth_basis": settings.strength.depth_basis,
        "depth_tolerance_pct_wt": float(settings.tool.depth_tolerance_pct_wt),
    }


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------

OUTPUT_FILES = ("classification.csv", "classification_summary.json")
BANNER_RULE = "!" * 78


def write_classification(result: ClassificationResult, out_dir: Path) -> dict[str, Path]:
    """Write the two outputs into ``comparison_dir`` (created); the summary JSON carries the DRAFT label.
    The summary needs ``prev_run_id`` and ``curr_run_id`` (``main`` adds them). The JSON text is built
    before anything is written."""
    s = result.summary
    folder = comparison_dir(out_dir, s["prev_run_id"], s["curr_run_id"])
    text = json.dumps({"label": DRAFT_LABEL, **s}, indent=2, allow_nan=False)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / name for name in OUTPUT_FILES}
    result.table.to_csv(paths["classification.csv"], index=False, lineterminator="\n")
    paths["classification_summary.json"].write_text(text, encoding="utf-8")
    return paths


def write_flag_status(status: pd.DataFrame, folder: Path) -> Path:
    """``flag_status.csv`` (``FLAG_STATUS_COLUMNS``; a header only without a log) in the comparison dir,
    through a temporary file and ``os.replace`` so a failure leaves the old file."""
    path = Path(folder) / FLAG_STATUS_FILE
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        status.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.classify",
                                 description="Repair classification, 195.452(h)(4). " + DRAFT_LABEL)
    ap.add_argument("--curr", type=Path, required=True, help="standardized CSV of the current run (run date)")
    ap.add_argument("--matches", type=Path, required=True, help="match_table.csv from ili.match")
    ap.add_argument("--remaining-life", type=Path, required=True, help="remaining_life.csv from ili.remaining_life")
    ap.add_argument("--pipe", type=Path, required=True, help="pipe_properties.csv")
    ap.add_argument("--segments", type=Path, required=True, help="hca_segments.csv")
    ap.add_argument("--crossings", type=Path, default=None, help="foreign_crossings.csv (optional)")
    ap.add_argument("--criteria", type=Path, default=None,
                    help="repair criteria YAML (default: classification.criteria_file of the settings)")
    ap.add_argument("--flags", type=Path, default=None,
                    help=f"manual flag log (default: {FLAGS_FILE} in the comparison dir, when it exists)")
    ap.add_argument("--discovery-date", default=None, help="discovery date, YYYY-MM-DD (default: latest permissible)")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs", help="output directory")
    return ap


def _print_report(result: ClassificationResult, paths: dict[str, Path]) -> None:
    s, t = result.summary, result.table
    print(DRAFT_LABEL)
    if s["discovery_warning"]:
        print(BANNER_RULE)
        print(f"!!! WARNING: {s['discovery_warning']}")
        print(BANNER_RULE)
    print(f"{s['prev_run_id']} vs {s['curr_run_id']}: {s['n_features']} features, criteria {s['criteria_version']}, "
          f"discovery date {s['discovery_date']} ({s['discovery_date_source']})")
    print("by category: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_category"].items()))
    immediate = t[t.governing_category == "immediate"]
    print(f"immediate ({len(immediate)}):")
    for r in immediate.itertuples(index=False):
        print(f"  {r.match_id}  {r.curr_feature_id}  {r.governing_rule}  {r.action_note}")
    print(f"requires review: {s['n_requires_review']} ("
          + ", ".join(f"{k} {v}" for k, v in s["counts_by_review_reason"].items()) + ")")
    more = s["could_be_more_severe_by_worst_category"]
    print(f"could be more severe: {sum(more.values())} (worst case: "
          + ", ".join(f"{k} {v}" for k, v in more.items() if k != "evaluate") + ")")
    print(f"overdue at discovery: {s['n_overdue_at_discovery']}")
    print(f"flags: {s['flags_file'] or 'none'}"
          + (f" ({len(t[t.manual_immediate])} manual immediate, {len(t[t.field_crack_confirmed])} field crack "
             f"confirmed, {len(s['flags_not_applied'])} not applied)" if s["flags_file"] else ""))
    for f in s["flags_not_applied"]:
        print(f"  {f['flag_id']} {f['flag']} on {f['curr_feature_id']} not applied: {f['reason']}")
    print(f"wrote {paths['classification.csv'].parent}")


def criteria_path(given: Path | None, settings: Settings) -> Path:
    """The criteria file to use: ``--criteria`` as given, else the settings' ``classification.criteria_file``,
    resolved against ``config.PROJECT_ROOT`` when relative (so the run does not depend on the working
    directory)."""
    if given is not None:
        return Path(given)
    path = Path(settings.classification.criteria_file)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _flags_path(given: Path | None, folder: Path) -> Path | None:
    """The flag log to apply: ``--flags`` (which must exist), else the comparison dir's log if any."""
    if given is not None:
        if not given.is_file():
            raise ValueError(f"flags file {given} not found")
        return given
    default = folder / FLAGS_FILE
    return default if default.is_file() else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # every input is read before anything is written, so a bad input leaves no partial outputs
        settings = load_settings(args.settings)
        crit_path = criteria_path(args.criteria, settings)
        criteria = load_repair_criteria(crit_path)
        digest = hashlib.sha256(crit_path.read_bytes()).hexdigest()
        curr = load_standardized(args.curr)
        matches = load_match_table(args.matches)
        remaining = load_remaining_life_table(args.remaining_life)
        pipe = pd.read_csv(args.pipe)
        segments = pd.read_csv(args.segments, dtype={"segment_id": str})
        crossings = pd.read_csv(args.crossings) if args.crossings is not None else None
        if crossings is None and settings.classification.crossing_proximity_ft is not None:
            print("warning: classification.crossing_proximity_ft is set but no --crossings file was given; "
                  "near_foreign_crossing will be unknown for every feature", file=sys.stderr)
        prev_id, curr_id = _run_ids(args.remaining_life, curr)
        run_dates = curr.run_date.dropna()
        if run_dates.empty:
            raise ValueError(f"{args.curr}: the current run has no run date")
        assessed = run_dates.iloc[0]
        flags_path = _flags_path(args.flags, comparison_dir(args.out, prev_id, curr_id))
        flags, flags_digest = (load_flags_hashed(flags_path) if flags_path
                               else (pd.DataFrame(columns=FLAG_COLUMNS, dtype=object), None))
        variables = feature_variables(matches, remaining, curr, pipe, segments, crossings, settings,
                                      flags=effective_flags(flags))
        result = classify(variables, criteria, settings, assessed, args.discovery_date, criteria_sha256=digest,
                          criteria_path=str(crit_path), crossings_file_given=crossings is not None)
        classified = {m for cell in variables.curr_feature_id for m in str(cell).split(REASON_SEP)}
        status = flag_status(flags, classified, set(curr.feature_id.astype(str)))
        not_applied = status[status.applied_status == "not_applied"]
        flag_summary = {
            "flags_file": str(flags_path) if flags_path else None,
            "flags_sha256": flags_digest,  # of the bytes parsed above, not a second read
            "flags_not_applied": [{"flag_id": r.flag_id, "curr_feature_id": r.curr_feature_id, "flag": r.flag,
                                   "reason": r.reason} for r in not_applied.itertuples(index=False)]}
        result = replace(result, summary={"prev_run_id": prev_id, "curr_run_id": curr_id, **result.summary,
                                          **flag_summary})
        paths = write_classification(result, args.out)
        write_flag_status(status, paths["classification.csv"].parent)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(result, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
