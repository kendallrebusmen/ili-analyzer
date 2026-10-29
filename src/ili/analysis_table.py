"""Analysis table: one row per classified anomaly, in the consultant's "Classified_Anomalies" layout.

``build_analysis_table`` joins what the stages already wrote (classification, growth, remaining life, match
table, the standardized tallies, the pipe table, the settings); nothing engineering is recomputed. Formatting
only: clock text, degrees, readable labels, D/S weld distance (joint length - U/S distance) and the change in
depth (current - previous). A value the inputs do not hold is blank (NaN), never 0; dents and gouges have no
growth, strength or remaining life, so those cells are blank.

DRAFT - for review by qualified personnel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ili.app_columns import CATEGORY_WORDS
from ili.reference import _nearest_row
from ili.report_data import ReportInputs, _lookup, _nearest_member
from ili.report_tables import _clock_text, imp_class_rows

ANALYSIS_TABLE_FILE = "analysis_table.csv"
ANALYSIS_XLSX_FILE = "classified_anomalies.xlsx"
SHEET_NAME = "Classified anomalies"

ID, GROWTH, DENTS, SIZE, STRENGTH, PROJECTION, LIFE, NEMI = (
    "Identity & location", "Growth", "Dents", "Size & pipe", "Strength", "Projection", "Life & class", "Nemi")
GROUPS = (ID, GROWTH, DENTS, SIZE, STRENGTH, PROJECTION, LIFE, NEMI)
BURST, SAFE = "Mod B31G Burst Pressure (psi)", "Mod B31G Safe Pressure (psi)"
ORIGINAL_NAMES = {BURST: "B31G Burst Pressure (psi)", SAFE: "B31G Safe Pressure (psi)"}

ANALYSIS_COLUMNS: tuple[tuple[str, str], ...] = (
    *((n, ID) for n in (
        "Feature ID", "Odometer (ft)", "Joint Number", "Joint Length (ft)", "U/S Weld Distance (ft)",
        "D/S Weld Distance (ft)", "Internal / External", "Orientation (deg)", "Orientation (o'clock)",
        "Feature Category", "Feature Type", "Cluster ID", "Vendor Comment", "Latitude", "Longitude", "Elevation")),
    *((n, GROWTH) for n in (
        "Corrosion Growth Rate (MPY)", "Growth Rate Source", "Metal Loss Depth (%)", "Prev. Anomaly Depth (%)",
        "Change in Anomaly Depth (%)", "Prev. Measurement Date", "Metal Loss Depth (in)",
        "Metal Loss Depth + Tolerance (%)", "Depth Tolerance (%)")),
    *((n, DENTS) for n in ("Dent Depth (in)", "Dent Depth (%)", "Is Dent on Top of Pipe")),
    *((n, SIZE) for n in (
        "Length (in)", "Width (in)", "Length Tolerance (in)", "Wall Thickness (in)", "Pipe Diameter (in)",
        "SMYS (psi)", "Seam Type", "Pipe Install Date", "Evaluation Pressure (psi)")),
    *((n, STRENGTH) for n in (BURST, SAFE, "ERF", "FPR")),
    *((n, PROJECTION) for n in (
        "Projected Reinspection ERF", "Projected Reinspection FPR",
        "Projected Reinspection Mod. B31G Burst Pressure (psi)",
        "Projected Reinspection Mod. B31G Safe Pressure (psi)", "Projected Depth", "Reinspection Date",
        "Projected Depth Condition Date", "Projected Burst Condition Date", "Projected Safe Condition Date")),
    *((n, LIFE) for n in (
        "Remaining Life (Years)", "Predicted Failure Mode", "IMP Classification", "Repair Category",
        "Governing Rule", "Due Date", "Is on Long Seam", "Is at Girth Weld", "Could Affect HCA")),
    *((n, NEMI) for n in ("Match ID", "Match Confidence", "Remaining Life, Factored (Years)", "Requires Review")),
)
_FEATURE_WORDS = {"metal_loss": "Metal loss", "dent": "Dent", "gouge": "Gouge"}
_ORIGIN_WORDS = {"corrosion": "Corrosion", "manufacturing": "Manufacturing"}
_SURFACE_WORDS = {"INT": "Internal", "EXT": "External"}
_METHOD_WORDS = {"raw": "Pit to pit", "shrinking_fixed": "Fixed growth rate", "half_life": "Half life",
                 "full_life": "Full life", "tolerance_adjusted": "Tolerance adjusted",
                 "segment_percentile": "Segment percentile", "new_feature": "New feature",
                 "default_rate": "Default rate"}
_TEXT_COLUMNS = ("Feature ID", "Joint Number", "Cluster ID", "Vendor Comment", "Match ID", "Seam Type",
                 "Governing Rule", "Match Confidence")


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _yn(values) -> pd.Series:
    """True/False as "Y"/"N"; unknown stays blank."""
    return pd.Series([None if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                      else ("Y" if bool(v) else "N") for v in values], dtype=object)


def _blank_if(values, mask) -> pd.Series:
    return pd.Series(values, dtype=object).where(~np.asarray(mask, dtype=bool), None)


def _numbers(series) -> np.ndarray:
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(float)


def _pipe_text(pipe: pd.DataFrame, pos: np.ndarray, column: str, tol_ft: float) -> pd.Series:
    """``column`` of the pipe table at each position (``reference.lookup``'s row rule); blank when the table
    has no such column, the position is unknown or is farther than ``tol_ft`` from every row."""
    if column not in pipe.columns or pipe.empty:
        return pd.Series([None] * len(pos), dtype=object)
    pipe = pipe.sort_values("start_ft", kind="stable")
    row, near = _nearest_row(pipe.start_ft.to_numpy(float), pipe.end_ft.to_numpy(float), pos)
    values = pipe[column].to_numpy(object)[row]
    ok = near <= tol_ft
    return pd.Series([v if o and not pd.isna(v) else None for v, o in zip(values, ok)], dtype=object)


def strength_columns(method: str) -> dict[str, str]:
    """Renames of the two strength columns for the strength method (none for modified B31G)."""
    return ORIGINAL_NAMES if method == "original_b31g" else {}


def build_analysis_table(inputs: ReportInputs) -> pd.DataFrame:
    """One row per classification row in log-distance order, columns ``ANALYSIS_COLUMNS`` (the two strength
    columns are named for the B31G method in the settings)."""
    cls = inputs.classification.reset_index(drop=True)
    keys = cls.match_id
    mt = _lookup(inputs.match_table, keys, [
        "curr_log_dist_ft", "curr_us_gw_id", "curr_dist_from_us_gw_ft", "curr_surface", "curr_ml_origin",
        "curr_vendor_cluster_id", "confidence"])
    g = _lookup(inputs.growth, keys, ["governing_mpy", "governing_method", "d_prev_pct", "d_curr_in"])
    rl = _lookup(inputs.remaining_life, keys, [
        "smys_psi", "erf", "fpr", "burst_psig", "safe_psig", "erf_next", "fpr_next", "burst_next_psig",
        "safe_next_psig", "depth_next_pct", "next_assessment_date", "tau_80_date", "tau_burst_date",
        "tau_safe_date", "rl_yr", "failure_mode", "rl_factored_yr"])
    imp = _lookup(imp_class_rows(cls, inputs.match_table, inputs.remaining_life, inputs.repairs, inputs.settings),
                  keys, ["imp_class"])
    settings = inputs.settings
    pos = _numbers(mt.curr_log_dist_ft)

    std = inputs.curr_std.drop_duplicates("feature_id").set_index("feature_id")
    first = pd.Series([str(f).split(";")[0] if isinstance(f, str) else f for f in cls.curr_feature_id], dtype=object)
    nearest = pd.Series([_nearest_member(f, p, std.log_dist_ft) for f, p in zip(cls.curr_feature_id, pos)],
                        dtype=object)
    joint_len = _numbers(first.map(std.joint_length_ft))
    us = _numbers(mt.curr_dist_from_us_gw_ft)
    metal = cls.feature_type.eq("metal_loss").to_numpy(bool)
    dent = cls.feature_type.eq("dent").to_numpy(bool)

    d_curr = _numbers(cls.depth_pct_reported)
    d_prev = np.where(metal, _numbers(g.d_prev_pct), np.nan)
    tol_applies = settings.strength.depth_basis == "reported_plus_tolerance"
    tol_ft = settings.alignment.anchor_tol_ft
    prev_date = inputs.prev_std.run_date.dropna()
    prev_date = "" if prev_date.empty else pd.Timestamp(prev_date.iloc[0]).date().isoformat()
    install = _numbers(_pipe_text(inputs.pipe, pos, "install_year", tol_ft))
    methods = g.governing_method.astype(object).map(
        lambda m: None if pd.isna(m) else
        f"{_ordinal(settings.growth.joint_percentile)} percentile per joint" if m == "joint_percentile"
        else _METHOD_WORDS.get(m, m)).where(metal, None)

    out = pd.DataFrame({
        "Feature ID": cls.curr_feature_id.astype(object), "Odometer (ft)": pos,
        "Joint Number": mt.curr_us_gw_id.astype(object), "Joint Length (ft)": joint_len,
        "U/S Weld Distance (ft)": us, "D/S Weld Distance (ft)": joint_len - us,
        "Internal / External": mt.curr_surface.map(lambda s: _SURFACE_WORDS.get(s, "Unknown")).astype(object),
        "Orientation (deg)": _numbers(cls.clock_min) * 0.5,
        "Orientation (o'clock)": pd.Series([_clock_text(c) or None for c in _numbers(cls.clock_min)], dtype=object),
        "Feature Category": cls.feature_type.map(lambda t: _FEATURE_WORDS.get(t, t)).astype(object),
        "Feature Type": mt.curr_ml_origin.map(lambda o: _ORIGIN_WORDS.get(o)).astype(object),
        "Cluster ID": mt.curr_vendor_cluster_id.astype(object), "Vendor Comment": first.map(std.comments),
        "Latitude": _numbers(nearest.map(std.lat)), "Longitude": _numbers(nearest.map(std.lon)),
        "Elevation": _numbers(nearest.map(std.elev_ft)),
        "Corrosion Growth Rate (MPY)": np.where(metal, _numbers(g.governing_mpy), np.nan),
        "Growth Rate Source": methods,
        "Metal Loss Depth (%)": np.where(metal, d_curr, np.nan), "Prev. Anomaly Depth (%)": d_prev,
        "Change in Anomaly Depth (%)": d_curr - d_prev,
        "Prev. Measurement Date": pd.Series([prev_date if not np.isnan(v) else None for v in d_prev], dtype=object),
        "Metal Loss Depth (in)": np.where(metal, _numbers(g.d_curr_in), np.nan),
        "Metal Loss Depth + Tolerance (%)": np.where(metal, _numbers(cls.depth_pct_assessed), np.nan),
        "Depth Tolerance (%)": np.where(metal, settings.tool.depth_tolerance_pct_wt if tol_applies else 0.0,
                                        np.nan),
        "Dent Depth (in)": np.where(dent, _numbers(cls.dent_depth_in), np.nan),
        "Dent Depth (%)": np.where(dent, _numbers(cls.dent_depth_pct_od), np.nan),
        "Is Dent on Top of Pipe": _blank_if(_yn(cls.top_of_pipe), ~dent),
        "Length (in)": _numbers(cls.length_in), "Width (in)": _numbers(cls.width_in),
        "Length Tolerance (in)": np.where(metal, settings.tool.length_tolerance_in, np.nan),
        "Wall Thickness (in)": _numbers(cls.wt_in), "Pipe Diameter (in)": _numbers(cls.od_in),
        "SMYS (psi)": _numbers(rl.smys_psi), "Seam Type": _pipe_text(inputs.pipe, pos, "seam_type", tol_ft),
        "Pipe Install Date": install, "Evaluation Pressure (psi)": _numbers(cls.mop_local),
        BURST: _numbers(rl.burst_psig), SAFE: _numbers(rl.safe_psig), "ERF": _numbers(rl.erf),
        "FPR": _numbers(rl.fpr),
        "Projected Reinspection ERF": _numbers(rl.erf_next), "Projected Reinspection FPR": _numbers(rl.fpr_next),
        "Projected Reinspection Mod. B31G Burst Pressure (psi)": _numbers(rl.burst_next_psig),
        "Projected Reinspection Mod. B31G Safe Pressure (psi)": _numbers(rl.safe_next_psig),
        "Projected Depth": _numbers(rl.depth_next_pct),
        "Reinspection Date": rl.next_assessment_date.astype(object),
        "Projected Depth Condition Date": rl.tau_80_date.astype(object),
        "Projected Burst Condition Date": rl.tau_burst_date.astype(object),
        "Projected Safe Condition Date": rl.tau_safe_date.astype(object),
        "Remaining Life (Years)": _numbers(rl.rl_yr),
        "Predicted Failure Mode": rl.failure_mode.map(
            lambda m: None if pd.isna(m) else str(m).capitalize()).astype(object),
        "IMP Classification": imp.imp_class.astype(object),
        "Repair Category": cls.governing_category.map(lambda c: CATEGORY_WORDS.get(c, c)).astype(object),
        "Governing Rule": cls.governing_rule.astype(object), "Due Date": cls.due_date.astype(object),
        "Is on Long Seam": _yn(cls.seam_interaction), "Is at Girth Weld": _yn(cls.near_girth_weld),
        "Could Affect HCA": _yn(cls.could_affect_hca),
        "Match ID": cls.match_id.astype(object), "Match Confidence": mt.confidence.astype(object),
        "Remaining Life, Factored (Years)": _numbers(rl.rl_factored_yr),
        "Requires Review": _yn(cls.requires_review),
    })
    out = out.sort_values("Odometer (ft)", kind="stable", na_position="last").reset_index(drop=True)
    out = out[[name for name, _ in ANALYSIS_COLUMNS]]
    return out.rename(columns=strength_columns(settings.strength.method))


def load_analysis_table(path: Path) -> pd.DataFrame:
    """Read ``analysis_table.csv`` back: text columns stay text (ids never become floats), empty is NaN.
    FileNotFoundError when it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in _TEXT_COLUMNS if c in header})
