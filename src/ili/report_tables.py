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
