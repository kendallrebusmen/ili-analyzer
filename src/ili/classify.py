"""Repair classification (spec section 6.8): the feature variables of every current-run anomaly.

``feature_variables`` builds one record per current-run anomaly (metal loss, dent, gouge) of the M4
match table, holding a value for every name in ``ili.vocabulary.VARIABLES``. Those records are what the
repair-criteria conditions (``config/repair_criteria.yaml``, evaluated by ``ili.rules``) are tested on.

Unknown is None, never a guessed value. A variable whose input is missing, or whose defining setting in
``classification`` is a null PLACEHOLDER, is None, so a rule that depends on it is reported as not
evaluable instead of silently passing or failing. Each variable family has its own helper below; each
documents its rule, including where the result is decided from partial information (Kleene logic).

``manual_immediate`` and ``field_crack_confirmed`` are manual and field entries that M9 will supply;
until then they are False for every feature (no evaluator flag, no confirmed crack is on record).

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ili.config import Settings
from ili.growth import PCT_PER_FRACTION, REASON_SEP
from ili.match import clock_diff_min
from ili.reference import _require_columns, assign_segment, check_segments, lookup
from ili.remaining_life import PIPE_WHAT, RL_SEGMENT_COLUMNS, _as_date, _segment_hca
from ili.rules import _kleene_and, _kleene_or, _normalize
from ili.schema import ANOMALY_TYPES, FeatureType
from ili.strength import assessed_depth_in
from ili.vocabulary import VARIABLES

IN_PER_FT = 12.0
# Match-table rows that have a current-run side (a "missing" row is a previous-run feature only).
CURRENT_SIDE_MATCH_TYPES = ("matched_1to1", "matched_cluster", "new")
# Spec 6.8: "top of pipe" means above the 4 and 8 o'clock positions, i.e. clock_min >= 480 or <= 240.
TOP_OF_PIPE_FROM_MIN = 480  # 8:00
TOP_OF_PIPE_TO_MIN = 240  # 4:00
# Why a row needs review upstream of classification, beside M6's own reasons: the M4 match row needs
# review; the feature is outside every segment (HCA rules assumed); a metal-loss row has no M6 row.
REVIEW_MATCH, REVIEW_NO_SEGMENT, REVIEW_RL_MISSING = "match_review", "no_segment", "rl_missing"

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


def near_girth_weld(pos_ft, length_in, weld_pos_ft, proximity_in) -> np.ndarray:
    """Edge distance to the nearest current-run girth weld <= ``proximity_in`` (inches).

    Edge distance = max(|x - w| x 12 - length / 2, 0), with x the feature's and w the nearest weld's log
    distance (ft) and length the feature's axial length (in). None when the setting is null, the
    position or length is unknown, or the run has no girth welds."""
    x, length = _floats(pos_ft), _floats(length_in)
    welds = np.sort(_floats(weld_pos_ft))
    welds = welds[~np.isnan(welds)]
    if proximity_in is None or welds.size == 0:
        return np.full(len(x), None, dtype=object)
    known = ~np.isnan(x) & ~np.isnan(length)
    k = np.clip(np.searchsorted(welds, np.nan_to_num(x)), 1, max(welds.size - 1, 1))
    gap_ft = np.minimum(np.abs(x - welds[k - 1]), np.abs(x - welds[np.minimum(k, welds.size - 1)]))
    edge_in = np.maximum(gap_ft * IN_PER_FT - length / 2, 0.0)
    with np.errstate(invalid="ignore"):
        return _tri(known, edge_in <= proximity_in)


def near_foreign_crossing(pos_ft, crossing_pos_ft, proximity_ft) -> np.ndarray:
    """|x - crossing| <= ``proximity_ft`` for any foreign crossing (log distances, ft). None when the
    setting is null or the position is unknown; False (known) when the setting is set and there are no
    crossings."""
    x = _floats(pos_ft)
    if proximity_ft is None:
        return np.full(len(x), None, dtype=object)
    c = _floats(crossing_pos_ft)
    c = np.sort(c[~np.isnan(c)])
    if c.size == 0:
        return _tri(~np.isnan(x), np.zeros(len(x), dtype=bool))
    k = np.clip(np.searchsorted(c, np.nan_to_num(x)), 1, max(c.size - 1, 1))
    gap = np.minimum(np.abs(x - c[k - 1]), np.abs(x - c[np.minimum(k, c.size - 1)]))
    with np.errstate(invalid="ignore"):
        return _tri(~np.isnan(x), gap <= proximity_ft)


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


def feature_variables(match_table: pd.DataFrame, remaining_life: pd.DataFrame, curr_std: pd.DataFrame,
                      pipe: pd.DataFrame, segments: pd.DataFrame, crossings: pd.DataFrame | None,
                      settings: Settings) -> pd.DataFrame:
    """One row per current-run anomaly of the M4 match table, columns ``FEATURE_VARIABLE_COLUMNS``.

    Rows: match types ``CURRENT_SIDE_MATCH_TYPES`` with ``curr_feature_type`` metal_loss, dent or gouge,
    in table order (a cluster row is metal loss with the envelope's dimensions). Values:

    - ``feature_type``, ``clock_min``, ``length_in``, ``width_in``, ``depth_pct_reported``: the match
      table's current side. ``top_of_pipe``: ``top_of_pipe``.
    - ``od_in``: the pipe table at ``curr_log_dist_ft`` (``reference.lookup``, ``alignment.anchor_tol_ft``;
      a position outside it is a ValueError, a blank OD is None).
    - Metal loss, from ``remaining_life`` (M6) by match_id: ``p_fail`` = burst_psig, ``p_safe`` =
      safe_psig, ``mop_local`` = mop_psig, ``dig_required_by_remaining_life``, ``repair_by_date``, and
      ``depth_pct_assessed`` = assessed_dt x 100 and ``wt_in``. A metal-loss row missing from M6 has None
      for the pressures, dig flag and repair-by date and is flagged ``rl_missing``.
    - ``wt_in`` otherwise (and for a blank M6 value): the match table's ``curr_wt_nom_in``.
      ``depth_pct_assessed`` otherwise: ``assessed_depth_pct`` of the reported depth (gouges; dents
      with a reported metal-loss depth; metal loss without an M6 value), None without a depth.
    - Dents only: ``dent_depth_in``, ``dent_depth_pct_od`` = dent_depth_in / od_in x 100,
      ``dent_metal_loss``. None for metal loss and gouges, as are the pressures for dents and gouges.
    - ``seam_interaction``, ``near_girth_weld`` (current-run girth welds of ``curr_std``),
      ``near_foreign_crossing`` (``crossings`` may be None: no crossings), ``general_corrosion``,
      ``widespread_circumferential``: the helpers of the same names, with the ``classification`` settings.
    - ``could_affect_hca``: the segment's flag (``segments``: start_ft, end_ft, segment_id,
      could_affect_hca); outside every segment it is True (HCA rules, the conservative choice) and the
      row is flagged ``no_segment``.
    - ``manual_immediate``, ``field_crack_confirmed``: False (M9 supplies them).
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
    keep = match_table.match_type.isin(CURRENT_SIDE_MATCH_TYPES) & match_table.curr_feature_type.isin(ANOMALY_TYPES)
    rows = match_table[keep.to_numpy(bool)].reset_index(drop=True)
    rl = remaining_life.reset_index(drop=True)
    n = len(rows)

    ftype = np.array([str(t) for t in rows.curr_feature_type], dtype=object)
    is_ml, is_dent = ftype == FeatureType.metal_loss.value, ftype == FeatureType.dent.value
    pos = _floats(rows.curr_log_dist_ft)
    od = (lookup(pipe, pos, "od_in", settings.alignment.anchor_tol_ft, what=PIPE_WHAT, allow_nan=True)
          if n else np.zeros(0))
    length, width, clock = _floats(rows.curr_length_in), _floats(rows.curr_width_in), rows.curr_clock_min
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
    with np.errstate(invalid="ignore", divide="ignore"):
        dent_pct_od = np.where(od > 0, dent_depth / od * PCT_PER_FRACTION, np.nan)
    dml = dent_metal_loss(rows.curr_dent_metal_loss, depth)
    dml[~is_dent] = None

    welds = curr_std.log_dist_ft[curr_std.feature_type.astype(str) == FeatureType.girth_weld.value]
    cross = crossings.log_dist_ft if crossings is not None else []

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
        "od_in": od, "wt_in": wt, "clock_min": pd.array(_floats(clock).round(), dtype="Int64"),
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
        "manual_immediate": obj([False] * n), "field_crack_confirmed": obj([False] * n),
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
