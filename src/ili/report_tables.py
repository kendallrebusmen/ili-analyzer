"""Report tables of the consultant's practice (spec 6.8 IMP class summary, 6.10 report). DRAFT.

The builders read tables the earlier stages wrote (classification, match table, remaining life, repairs log)
and recompute nothing a stage wrote: a class, a remaining life or a position is looked up, never derived
again. Counts are plain integers; a value that cannot be known stays None so the writers render an empty cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ili.classify import near_girth_weld, seam_interaction
from ili.config import Settings
from ili.match import PositionMap
from ili.reference import lookup
from ili.remaining_life import _as_bool
from ili.schema import FeatureType, MlOrigin, Surface

IMP_CLASS_COLUMNS = ("imp_class", "n_classified", "n_previously_repaired", "n_unrepaired", "n_rl_not_evaluated")
_CATEGORY_CLASSES = {"immediate": "Immediate", "60_day": "60-day", "180_day": "180-day"}  # spec 6.8
NO_CLASS = "None"
LOCATIONS = ("all", "girth_weld", "long_seam")
DEPTH_COLUMNS = ("run_id", "location", "bin", "ext_corrosion", "int_corrosion", "manufacturing", "total",
                 "n_location_unknown")
DENT_COLUMNS = ("run_id", "bin", *LOCATIONS, "n_girth_weld_unknown", "n_long_seam_unknown")
BIN_DECIMALS = 9  # a percentage is rounded before binning so 0.9 in on 15 in is exactly 6 (as report_data.backtest)
PCT_PER_FRACTION = 100.0
PIPE_WHAT = "pipe properties table"


def _previously_repaired(position_ft: np.ndarray, repairs: pd.DataFrame) -> np.ndarray:
    """True where a current-run position (ft; NaN is never repaired) lies in any repair span, ends inclusive."""
    missing = [c for c in ("start_ft", "end_ft") if c not in repairs.columns]
    if missing:
        raise ValueError(f"repairs table is missing column(s): {', '.join(missing)}")
    hit = np.zeros(len(position_ft), dtype=bool)
    for start, end in zip(repairs.start_ft.astype(float), repairs.end_ft.astype(float)):
        hit |= (position_ft >= min(start, end)) & (position_ft <= max(start, end))
    return hit


def imp_class_summary(classification: pd.DataFrame, match_table: pd.DataFrame, remaining_life: pd.DataFrame,
                      repairs: pd.DataFrame | None, settings: Settings) -> pd.DataFrame:
    """The IMP class summary (spec 6.8), one row per class: Immediate, 60-day and 180-day by
    ``governing_category``; ``YTC < N yrs`` for any other row whose years to condition (``rl_yr``, else
    ``rl_upper_bound_yr``) is below ``report.ytc_flag_years``; ``None`` for the rest. A row with neither
    remaining life nor bound falls in ``None``. ``n_rl_not_evaluated`` counts, per class, the metal-loss rows
    (``feature_type``) with neither; dents and gouges never have a remaining life and are not counted. A feature is previously repaired when its current-run position
    (``curr_log_dist_ft`` of the match table, by ``match_id``) is inside any repair span; without a repairs
    table ``n_previously_repaired`` and ``n_unrepaired`` are None. Summary view only; categories drive due dates."""
    ytc = settings.report.ytc_flag_years
    ytc_label = f"YTC < {ytc:g} yrs"
    cls = classification.reset_index(drop=True)
    rl = remaining_life.drop_duplicates("match_id").set_index("match_id")
    keys = cls.match_id.to_numpy()
    years = rl.rl_yr.astype(float).reindex(keys).to_numpy()
    bound = rl.rl_upper_bound_yr.astype(float).reindex(keys).to_numpy()
    years = np.where(np.isnan(years), bound, years)  # rl_yr (inf: no failure within the horizon), else the bound
    not_evaluated = np.isnan(years) & cls.feature_type.eq("metal_loss").to_numpy(bool)
    ytc_hit = years < ytc  # NaN and inf are never below
    imp = np.where(cls.governing_category.isin(_CATEGORY_CLASSES), cls.governing_category.map(_CATEGORY_CLASSES),
                   np.where(ytc_hit, ytc_label, NO_CLASS))
    if repairs is None:
        repaired = None
    else:
        position = (match_table.drop_duplicates("match_id").set_index("match_id").curr_log_dist_ft
                    .astype(float).reindex(keys).to_numpy())
        repaired = _previously_repaired(position, repairs)
    rows = []
    for name in (*_CATEGORY_CLASSES.values(), ytc_label, NO_CLASS):
        in_class = imp == name
        n = int(in_class.sum())
        n_repaired = None if repaired is None else int((in_class & repaired).sum())
        rows.append({"imp_class": name, "n_classified": n, "n_previously_repaired": n_repaired,
                     "n_unrepaired": None if n_repaired is None else n - n_repaired,
                     "n_rl_not_evaluated": int((in_class & not_evaluated).sum())})
    out = pd.DataFrame(rows, columns=list(IMP_CLASS_COLUMNS))
    for col in ("n_previously_repaired", "n_unrepaired"):
        out[col] = out[col].astype(object)
    return out


def _edge_labels(edges, open_label: str) -> list[str]:
    """``a-b`` for each pair of consecutive edges, then ``open_label`` with the last edge (``:g`` formatting)."""
    pairs = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    return [*pairs, open_label.format(edges[-1])]


def _locations(std: pd.DataFrame, settings: Settings) -> dict[str, np.ndarray]:
    """Per location, True / False / None for each row of ``std``: ``all`` is always True; ``girth_weld`` uses
    the run's own girth welds (``near_girth_weld``); ``long_seam`` is ``seam_interaction``."""
    cfg = settings.classification
    welds = std.log_dist_ft[std.feature_type.astype(str) == FeatureType.girth_weld.value]
    pos, length = std.log_dist_ft.to_numpy(float), std.length_in.to_numpy(float)
    return {
        "all": np.full(len(std), True, dtype=object),
        "girth_weld": near_girth_weld(pos, length, welds, cfg.girth_weld_proximity_in),
        "long_seam": seam_interaction(std.seam_interaction, std.clock_min.to_numpy(float), cfg.seam_clock_min,
                                      cfg.seam_clock_tol_min),
    }


def _is_true(flags: np.ndarray) -> np.ndarray:
    return np.array([f is not None and bool(f) for f in flags], dtype=bool)


def _is_unknown(flags: np.ndarray) -> np.ndarray:
    return np.array([f is None for f in flags], dtype=bool)


def _run_id(std: pd.DataFrame) -> str:
    return str(std.run_id.iloc[0]) if len(std) else ""


def depth_bins(std: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One run's metal loss counted by depth bin (spec 6.10): per location (``all``, ``girth_weld``,
    ``long_seam``) and bin (``report.depth_bins_pct_wt``, left-closed: ``[70, 80)``, last bin ``>=80``), the
    external-corrosion, internal-corrosion and manufacturing counts and the total. Corrosion means
    ``ml_origin`` is not manufacturing; ``total`` also holds MID and UNK surfaces. The depth is rounded to
    ``BIN_DECIMALS`` places before binning. A row without a depth is in no bin; a row whose location is
    unknown (None) is not counted in that location, and ``n_location_unknown`` (the same on every row of a
    location) is the number of binnable rows left out for that reason (0 for ``all``). Every bin of every
    location has a row, counts 0 where nothing falls in it."""
    edges = list(settings.report.depth_bins_pct_wt)
    labels = _edge_labels(edges, ">={:g}")
    ml = std.feature_type.astype(str).eq(FeatureType.metal_loss.value).to_numpy(bool)
    depth = std.depth_pct.to_numpy(float).round(BIN_DECIMALS)
    with np.errstate(invalid="ignore"):
        index = np.searchsorted(edges, depth, side="right") - 1  # left-closed; NaN is masked below
    binned = ml & ~np.isnan(depth) & (index >= 0)
    manufacturing = std.ml_origin.astype(str).eq(MlOrigin.manufacturing.value).to_numpy(bool)
    surface = std.surface.astype(str)
    ext, internal = surface.eq(Surface.EXT.value).to_numpy(bool), surface.eq(Surface.INT.value).to_numpy(bool)
    rows = []
    for location, flags in _locations(std, settings).items():
        keep = binned & _is_true(flags)
        n_unknown = int((binned & _is_unknown(flags)).sum())
        for k, label in enumerate(labels):
            hit = keep & (index == k)
            rows.append({"run_id": _run_id(std), "location": location, "bin": label,
                         "ext_corrosion": int((hit & ext & ~manufacturing).sum()),
                         "int_corrosion": int((hit & internal & ~manufacturing).sum()),
                         "manufacturing": int((hit & manufacturing).sum()), "total": int(hit.sum()),
                         "n_location_unknown": n_unknown})
    return pd.DataFrame(rows, columns=list(DEPTH_COLUMNS))


def dent_bins(std: pd.DataFrame, od_in: np.ndarray, settings: Settings) -> pd.DataFrame:
    """One run's dents counted by depth as a percentage of OD (``dent_depth_in`` / ``od_in`` x 100, rounded to
    ``BIN_DECIMALS`` places; spec 6.10), right-closed bins from ``report.dent_bins_pct_od`` (``[0, 2]``,
    ``(2, 6]``, ``>6``), with the ``all``, ``girth_weld`` and ``long_seam`` columns. ``od_in`` is aligned with
    the rows of ``std``. A dent without a depth or OD is in no bin; an unknown location is not counted, and
    ``n_girth_weld_unknown`` / ``n_long_seam_unknown`` (the same on every row) are the numbers of binnable
    dents left out of those columns for that reason. Every bin has a row."""
    edges = list(settings.report.dent_bins_pct_od)
    labels = _edge_labels(edges, ">{:g}")
    dent = std.feature_type.astype(str).eq(FeatureType.dent.value).to_numpy(bool)
    od = np.asarray(od_in, dtype=float)
    if len(od) != len(std):
        raise ValueError(f"od_in has {len(od)} values for {len(std)} rows")
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(od > 0, std.dent_depth_in.to_numpy(float) / od * PCT_PER_FRACTION, np.nan).round(BIN_DECIMALS)
        index = np.maximum(np.searchsorted(edges, pct, side="left") - 1, 0)  # right-closed; 0 is in the first bin
    binned = dent & ~np.isnan(pct) & (pct >= 0)
    flags = _locations(std, settings)
    unknown = {loc: int((binned & _is_unknown(flags[loc])).sum()) for loc in LOCATIONS}
    rows = []
    for k, label in enumerate(labels):
        row = {"run_id": _run_id(std), "bin": label}
        for location in LOCATIONS:
            row[location] = int((binned & _is_true(flags[location]) & (index == k)).sum())
        rows.append({**row, "n_girth_weld_unknown": unknown["girth_weld"],
                     "n_long_seam_unknown": unknown["long_seam"]})
    return pd.DataFrame(rows, columns=list(DENT_COLUMNS))


@dataclass(frozen=True)
class Note:
    """Why a location's counts leave rows out: ``table`` (``depth`` or ``dent``), ``location`` and the text."""
    table: str
    location: str
    text: str


@dataclass(frozen=True)
class AnomalyDistribution:
    """Both runs' depth-bin and dent-bin tables (rows stacked, previous run first) and the notes saying which
    locations could not be evaluated for some rows, and why."""
    depth: pd.DataFrame
    dent: pd.DataFrame
    notes: list[Note]


def _location_notes(result_depth: pd.DataFrame, result_dent: pd.DataFrame, settings: Settings) -> list[Note]:
    """One note per run, table and location with rows left out for an unknown location (count and reason);
    none when every binnable row is known."""
    cfg = settings.classification
    reasons = {
        "girth_weld": ("girth-weld proximity",
                       "classification.girth_weld_proximity_in is not set" if cfg.girth_weld_proximity_in is None
                       else "no girth welds, or a position or length is missing, or the feature is outside the "
                            "weld span"),
        "long_seam": ("long seam",
                      "classification.seam_clock_min / seam_clock_tol_min is not set and the vendor flag is not True"
                      if cfg.seam_clock_min is None or cfg.seam_clock_tol_min is None
                      else "the clock position is missing and the vendor flag is not True"),
    }
    counts = []  # (table, location, run, n, what)
    for run, rows in result_depth.groupby("run_id", sort=False):
        for loc in ("girth_weld", "long_seam"):
            counts.append(("depth", loc, run, int(rows[rows.location == loc].n_location_unknown.iloc[0]),
                           "metal-loss row(s)"))
    for run, rows in result_dent.groupby("run_id", sort=False):
        for loc in ("girth_weld", "long_seam"):
            counts.append(("dent", loc, run, int(rows[f"n_{loc}_unknown"].iloc[0]), "dent(s)"))
    return [Note(table, loc, f"{reasons[loc][0]}: {reasons[loc][1]} ({n} {what} of run {run} not evaluated)")
            for table, loc, run, n, what in counts if n]


def anomaly_distribution(inputs) -> AnomalyDistribution:
    """Depth bins and dent bins for the previous and the current run (spec 6.10). Dent ODs come from the pipe
    properties (``inputs.pipe``) at each dent's current-run position; the previous run's positions are first
    mapped onto the current run with the weld map, as reference files are keyed to current-run distance. A
    dent position the pipe table does not cover is a ValueError naming the run."""
    settings = inputs.settings
    tol = settings.alignment.anchor_tol_ft
    pmap = PositionMap.from_weld_map(inputs.weld_map)
    depth, dent = [], []
    for std, is_prev in ((inputs.prev_std, True), (inputs.curr_std, False)):
        pos = std.log_dist_ft.to_numpy(float)
        if is_prev:
            pos = pmap.to_curr(pos)
        is_dent = std.feature_type.astype(str).eq(FeatureType.dent.value).to_numpy(bool)
        od = np.full(len(std), np.nan)
        if is_dent.any():
            try:
                od[is_dent] = lookup(inputs.pipe, pos[is_dent], "od_in", tol, what=PIPE_WHAT, allow_nan=True)
            except ValueError as exc:
                raise ValueError(f"dent-bin OD lookup failed for run {_run_id(std)}: {exc}") from exc
        depth.append(depth_bins(std, settings))
        dent.append(dent_bins(std, od, settings))
    depth, dent = pd.concat(depth, ignore_index=True), pd.concat(dent, ignore_index=True)
    return AnomalyDistribution(depth, dent, _location_notes(depth, dent, settings))


def anomaly_distribution_table(result: AnomalyDistribution) -> pd.DataFrame:
    """The ``anomaly_distribution.csv`` rows: depth rows then dent rows, a ``table`` column (``depth`` or
    ``dent``) first; a column that does not belong to a table is blank."""
    columns = ["table", *dict.fromkeys([*DEPTH_COLUMNS, *DENT_COLUMNS])]
    out = pd.concat([result.depth.assign(table="depth"), result.dent.assign(table="dent")],
                    ignore_index=True)[columns]
    for col in columns:
        if col not in ("table", "run_id", "location", "bin"):
            out[col] = out[col].astype("Int64")
    return out


JOINT_SIDES = ("prev", "curr")
JOINT_COUNT_KINDS = ("ext", "int", "other", "total")
JOINT_COUNT_COLUMNS = ("section_id", "match_type", "prev_gw_ids", "curr_gw_ids", "curr_start_ft",
                       *(f"{side}_{kind}" for side in JOINT_SIDES for kind in JOINT_COUNT_KINDS), "change_total")
GW_ID_SEPARATOR = ";"


def _metal_loss_by_joint(std: pd.DataFrame) -> pd.DataFrame:
    """One run's metal-loss counts per ``us_gw_id``: columns ``ext``, ``int``, ``other`` (surface MID, UNK or
    blank) and ``total``. A row without an upstream girth weld is counted nowhere."""
    ml = std[std.feature_type.astype(str).eq(FeatureType.metal_loss.value) & std.us_gw_id.notna()]
    surface = ml.surface.astype(str)
    kind = np.where(surface.eq(Surface.EXT.value), "ext", np.where(surface.eq(Surface.INT.value), "int", "other"))
    counts = pd.crosstab(ml.us_gw_id.astype(str), pd.Series(kind, index=ml.index, name="kind"))
    return counts.reindex(columns=list(JOINT_COUNT_KINDS[:3]), fill_value=0).assign(total=lambda c: c.sum(axis=1))


def joint_ml_counts(weld_map: pd.DataFrame, prev_std: pd.DataFrame, curr_std: pd.DataFrame) -> pd.DataFrame:
    """Metal loss per joint for both runs (spec 6.10, per-joint counts), one row per weld-map row in its order.
    A row's counts on a side are the sums over that side's joint(s) (``*_gw_id`` and ``*_gw_id_2``, so a 2:1 or
    1:2 row sums both), taken from each run's metal loss by ``us_gw_id``; no anomaly matching is used. A joint
    with no metal loss counts 0; a side the row does not have (an inserted joint's previous side, a removed
    joint's current side) is blank (``Int64`` NA). ``change_total`` is ``curr_total - prev_total``, blank when
    either side is. ``prev_gw_ids`` / ``curr_gw_ids`` are ``;``-joined, "" for an absent side."""
    runs = {"prev": _metal_loss_by_joint(prev_std), "curr": _metal_loss_by_joint(curr_std)}
    out = pd.DataFrame({"section_id": weld_map.section_id.to_numpy(), "match_type": weld_map.match_type.to_numpy(),
                        "curr_start_ft": weld_map.curr_start_ft.astype(float).to_numpy()})
    ids = {side: [[str(v) for v in row if pd.notna(v)]
                  for row in weld_map[[f"{side}_gw_id", f"{side}_gw_id_2"]].itertuples(index=False)]
           for side in JOINT_SIDES}
    for side in JOINT_SIDES:
        out[f"{side}_gw_ids"] = [GW_ID_SEPARATOR.join(row) for row in ids[side]]
        for kind in JOINT_COUNT_KINDS:
            counts = runs[side][kind]
            out[f"{side}_{kind}"] = pd.array([int(counts.reindex(row, fill_value=0).sum()) if row else pd.NA
                                              for row in ids[side]], dtype="Int64")
    out["change_total"] = out.curr_total - out.prev_total
    return out[list(JOINT_COUNT_COLUMNS)]


def largest_joints(counts: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ``n`` joints with the largest ``curr_total`` and the ``n`` with the largest ``change_total``, ties
    broken by ``curr_start_ft`` (ascending). A row blank in the ranked column is not ranked."""
    def top(column: str) -> pd.DataFrame:
        ranked = counts[counts[column].notna()]
        return (ranked.sort_values([column, "curr_start_ft"], ascending=[False, True], kind="stable")
                .head(n).reset_index(drop=True))
    return top("curr_total"), top("change_total")


AOI_COLUMNS = ("rank", "start_ft", "end_ft", "surfaces", "origins", "max_depth_pct", "n_anomalies", "min_rl_yr",
               "hca", "description")
MIN_PER_HOUR = 60
HOURS_ON_FACE = 12


def _clock_text(clock_min: float) -> str:
    """``h:mm`` on the 12-hour face from minutes past 12 o'clock (0 and 360 give 12:00 and 6:00); "" if NaN."""
    if pd.isna(clock_min):
        return ""
    hour, minute = divmod(int(round(clock_min)) % (HOURS_ON_FACE * MIN_PER_HOUR), MIN_PER_HOUR)
    return f"{hour or HOURS_ON_FACE}:{minute:02d}"


def _repair_clauses(start_ft: float, end_ft: float, repairs: pd.DataFrame | None) -> list[str]:
    """One "overlaps <type> repair of <date>" clause per repair span touching [start_ft, end_ft] (ends
    inclusive), ordered by the repair's start; none when there is no repairs table."""
    if repairs is None or repairs.empty:
        return []
    lo, hi = repairs.start_ft.astype(float), repairs.end_ft.astype(float)
    hit = repairs[(np.minimum(lo, hi) <= end_ft) & (np.maximum(lo, hi) >= start_ft)]
    hit = hit.assign(_start=np.minimum(hit.start_ft.astype(float), hit.end_ft.astype(float))).sort_values(
        "_start", kind="stable")
    clauses = []
    for repair in hit.itertuples(index=False):
        kind = str(repair.repair_type).replace("_", " ")
        when = repair.repair_date
        clauses.append(f"overlaps {kind} repair" + ("" if pd.isna(when) else f" of {str(when)[:10]}"))
    return clauses


def anomalies_of_interest(inputs) -> pd.DataFrame:
    """Groups of current-run metal loss ranked by remaining life (spec 6.10, anomalies of interest).

    Metal loss sorted by ``log_dist_ft`` starts a new group where the gap to the previous feature exceeds
    ``report.aoi_group_ft`` (ft). A member's remaining life and HCA flag come from the ``remaining_life`` row
    whose ``curr_feature_id`` (``;``-split for a cluster) holds it; no row gives NaN. ``min_rl_yr`` is the
    finite minimum over members (NaN if none); ``hca`` is True if any member is, None if all are unknown, else
    False. Sorted by ``min_rl_yr`` ascending (NaN last), ``max_depth_pct`` descending, ``start_ft``; the first
    ``report.aoi_count`` rows keep ``rank`` 1... A row with no ``log_dist_ft`` is left out. The description names the deepest member, the depth range of
    the others and every repair span overlapping the group (``repairs`` may be None)."""
    std = inputs.curr_std
    ml = std[std.feature_type.astype(str).eq(FeatureType.metal_loss.value) & std.log_dist_ft.notna()].sort_values(
        "log_dist_ft", kind="stable").reset_index(drop=True)  # a row with no position cannot be placed
    rl_by_id: dict[str, tuple[float, object]] = {}
    for row in inputs.remaining_life.itertuples(index=False):
        for fid in str(row.curr_feature_id).split(GW_ID_SEPARATOR):
            rl_by_id[fid] = (float(row.rl_yr) if pd.notna(row.rl_yr) else np.nan, row.could_affect_hca)
    group = (ml.log_dist_ft.diff() > inputs.settings.report.aoi_group_ft).cumsum()
    rows = []
    for _, g in ml.groupby(group, sort=True):
        rl = np.array([rl_by_id.get(str(f), (np.nan, None))[0] for f in g.feature_id], dtype=float)
        flags = [rl_by_id.get(str(f), (np.nan, None))[1] for f in g.feature_id]
        known = [bool(f) for f in flags if pd.notna(f)]
        depth = g.depth_pct.astype(float)
        start, end = float(g.log_dist_ft.min()), float(g.log_dist_ft.max())
        deepest = g.loc[depth.fillna(-np.inf).idxmax()]
        clauses = []
        if pd.notna(deepest.depth_pct):
            clauses.append(f"Deepest {deepest.depth_pct:g}% WT"
                           + (f", {deepest.length_in:g} in long" if pd.notna(deepest.length_in) else "")
                           + (f", {deepest.width_in:g} in wide" if pd.notna(deepest.width_in) else "")
                           + (f" at {_clock_text(deepest.clock_min)}" if pd.notna(deepest.clock_min) else ""))
        others = depth.drop(deepest.name).dropna()
        if len(g) > 1 and not others.empty:
            clauses.append(f"others {others.min():g}-{others.max():g}% WT")
        clauses += _repair_clauses(start, end, inputs.repairs)
        rows.append({"start_ft": start, "end_ft": end,
                     "surfaces": ", ".join(sorted(set(g.surface.dropna().astype(str)))),
                     "origins": ", ".join(sorted(set(g.ml_origin.dropna().astype(str)))),
                     "max_depth_pct": float(depth.max()), "n_anomalies": len(g),
                     "min_rl_yr": float(np.nanmin(rl)) if np.isfinite(rl).any() else np.nan,
                     "hca": any(known) if known else None, "description": "; ".join(clauses)})
    out = pd.DataFrame(rows, columns=[c for c in AOI_COLUMNS if c != "rank"])
    out = out.sort_values(["min_rl_yr", "max_depth_pct", "start_ft"], ascending=[True, False, True],
                          na_position="last", kind="stable").head(inputs.settings.report.aoi_count)
    out.insert(0, "rank", range(1, len(out) + 1))
    out["hca"] = out["hca"].astype(object)
    return out.reset_index(drop=True)


PIPE_SUMMARY_WHAT = "pipe_properties.csv"
WALL_GRADE_COLUMNS = ("wt_nom_in", "grade", "length_ft")
LENGTH_CHECK_COLUMNS = ("run_id", "ili_length_ft", "reference_length_ft", "diff_ft", "diff_pct")


@dataclass(frozen=True)
class PipeSummary:
    """Pipe properties summarised for the report header. ``by_wall_grade`` has columns ``wt_nom_in``, ``grade``,
    ``length_ft``. The ``(min, max)`` pairs are None when no row can be evaluated: ``design_pressure_psig``
    (psig) and ``stress_at_mop_pct_smys`` (percent of SMYS). ``install_years`` and ``coatings`` are sorted unique
    values; ``hca_length_ft`` is the length (ft) of the segments that could affect an HCA and ``hca_pct`` its
    share of the pipe length (percent; None for a pipe of no length)."""
    by_wall_grade: pd.DataFrame
    design_pressure_psig: tuple[float, float] | None
    stress_at_mop_pct_smys: tuple[float, float] | None
    install_years: list
    coatings: list[str]
    hca_length_ft: float
    hca_pct: float | None


def _range(values: np.ndarray) -> tuple[float, float] | None:
    values = values[np.isfinite(values)]
    return (float(values.min()), float(values.max())) if len(values) else None


def _sorted_unique(values: pd.Series) -> list:
    """Sorted unique non-blank values; a whole number (install year) as an int."""
    return sorted({int(v) if isinstance(v, (float, np.floating)) and float(v).is_integer() else v
                   for v in values.dropna() if str(v).strip() != ""})


def pipe_summary(pipe: pd.DataFrame, mop: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> PipeSummary:
    """The pipe summary of the report header (spec 6.10): length by wall thickness and grade; the range of
    design pressure 2 SMYS t F / D (psig; F is ``strength.design_factor``) over the pipe rows; the range of
    stress at MOP, MOP D / (2 t SMYS) x 100 (% SMYS), over the intervals between the union of the pipe and MOP
    breakpoints (an interval that either table does not cover, or whose value is NaN, is skipped, not an
    error); sorted install years and coatings; and the length of the segments with ``could_affect_hca``
    (True or the text ``True``). A missing pipe column is a ValueError naming the file."""
    need = ("start_ft", "end_ft", "od_in", "wt_nom_in", "grade", "smys_psi", "install_year", "coating_type")
    missing = [c for c in need if c not in pipe.columns]
    if missing:
        raise ValueError(f"{PIPE_SUMMARY_WHAT} is missing column(s): {', '.join(missing)}")
    rows = pipe.assign(length_ft=pipe.end_ft.astype(float) - pipe.start_ft.astype(float))
    by = (rows.groupby(["wt_nom_in", "grade"], sort=True, dropna=False).length_ft.sum().reset_index()
          [list(WALL_GRADE_COLUMNS)])
    od, wt, smys = (rows[c].to_numpy(float) for c in ("od_in", "wt_nom_in", "smys_psi"))
    design = 2.0 * smys * wt * settings.strength.design_factor / od

    start, end = pipe.start_ft.to_numpy(float), pipe.end_ft.to_numpy(float)
    m_start, m_end = mop.start_ft.to_numpy(float), mop.end_ft.to_numpy(float)
    edges = np.unique(np.concatenate([start, end, m_start, m_end]))
    stress = []
    for x in (edges[:-1] + edges[1:]) / 2.0:  # an interval's midpoint is inside exactly one row per table
        p_hit = np.flatnonzero((start <= x) & (x < end))
        m_hit = np.flatnonzero((m_start <= x) & (x < m_end))
        if len(p_hit) and len(m_hit):
            i, j = p_hit[0], m_hit[0]
            stress.append(float(mop.mop_psig.iloc[j]) * od[i] / (2.0 * wt[i] * smys[i]) * PCT_PER_FRACTION)

    hca = np.array([_as_bool(v, "segments") for v in segments.could_affect_hca], dtype=bool)
    hca_ft = float((segments.end_ft.to_numpy(float) - segments.start_ft.to_numpy(float))[hca].sum())
    total = float(rows.length_ft.sum())
    return PipeSummary(by, _range(design), _range(np.array(stress, dtype=float)),
                       _sorted_unique(pipe.install_year), [str(c) for c in _sorted_unique(pipe.coating_type)],
                       hca_ft, hca_ft / total * PCT_PER_FRACTION if total > 0 else None)


def length_check(prev_std: pd.DataFrame, curr_std: pd.DataFrame, pipe: pd.DataFrame) -> pd.DataFrame:
    """Each run's surveyed length against the pipe table's, one row per run (previous first): ``ili_length_ft``
    is the largest minus the smallest ``log_dist_ft`` of the run's girth welds, ``reference_length_ft`` the
    largest ``end_ft`` minus the smallest ``start_ft`` of ``pipe``, ``diff_ft`` their difference (ILI minus
    reference) and ``diff_pct`` that as a percent of the reference. NaN where a run has no girth weld."""
    reference = float(pipe.end_ft.astype(float).max() - pipe.start_ft.astype(float).min())
    rows = []
    for std in (prev_std, curr_std):
        welds = std.log_dist_ft[std.feature_type.astype(str) == FeatureType.girth_weld.value].astype(float)
        ili = float(welds.max() - welds.min()) if len(welds) else np.nan
        diff = ili - reference
        rows.append({"run_id": _run_id(std), "ili_length_ft": ili, "reference_length_ft": reference,
                     "diff_ft": diff, "diff_pct": diff / reference * PCT_PER_FRACTION if reference else np.nan})
    return pd.DataFrame(rows, columns=list(LENGTH_CHECK_COLUMNS))
