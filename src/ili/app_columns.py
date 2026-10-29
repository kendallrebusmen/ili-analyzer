"""Plain-word column labels for the app's tables (presentation only).

``column_config(df)`` gives ``st.dataframe`` a label, and a number format for numeric columns, for each column
it knows; the data and the column order are untouched, and an unknown column keeps its raw name.

DRAFT - for review by qualified personnel.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# column -> (label, number format; None = not numeric)
LABELS: dict[str, tuple[str, str | None]] = {
    # Identity and location
    "match_id": ("Match", None), "match_type": ("Match type", None), "prev_feature_id": ("Feature before", None),
    "curr_feature_id": ("Feature", None), "feature_type": ("Feature type", None), "segment_id": ("Segment", None),
    "section_id": ("Section", None), "decision_id": ("Decision", None), "flag_id": ("Flag ID", None),
    "curr_log_dist_ft": ("Distance (ft)", "%.1f"), "curr_us_gw_id": ("Upstream girth weld", None),
    "curr_dist_from_us_gw_ft": ("Distance from upstream weld (ft)", "%.2f"),
    "clock_min": ("Clock (min)", "%.0f"), "lat": ("Latitude", "%.6f"), "lon": ("Longitude", "%.6f"),
    # Anomaly size
    "prev_depth_pct": ("Depth before (% WT)", "%.0f"), "curr_depth_pct": ("Depth now (% WT)", "%.0f"),
    "d_curr_pct": ("Depth now (% WT)", "%.0f"),
    "depth_pct_assessed": ("Assessed depth (% WT)", "%.0f"),
    "prev_length_in": ("Length before (in)", "%.2f"), "curr_length_in": ("Length now (in)", "%.2f"),
    "dent_depth_pct_od": ("Dent depth (% OD)", "%.2f"),
    # Growth
    "raw_mpy": ("Raw growth (mpy)", "%.2f"), "governing_mpy": ("Growth (mpy)", "%.2f"),
    "governing_method": ("Growth method", None),
    # Strength and remaining life
    "p_fail": ("Failure pressure (psig)", "%.0f"), "p_safe": ("Safe pressure (psig)", "%.0f"),
    "mop_local": ("MOP (psig)", "%.0f"), "erf": ("ERF", "%.3f"), "fpr": ("FPR", "%.3f"),
    "burst_next_psig": ("Burst pressure at next assessment (psig)", "%.0f"),
    "safe_next_psig": ("Safe pressure at next assessment (psig)", "%.0f"),
    "erf_next": ("ERF at next assessment", "%.3f"), "fpr_next": ("FPR at next assessment", "%.3f"),
    "exceeds_80pct_next": ("Over 80% WT at next assessment", None),
    "rl_yr": ("Remaining life (yr)", "%.1f"), "rl_governing": ("Remaining life limited by", None),
    "failure_mode": ("Failure mode", None), "imp_class": ("Impact class", None),
    "previously_repaired": ("Previously repaired", None),
    # Classification
    "governing_category": ("Repair category", None), "governing_rule": ("Governing rule", None),
    "triggered_rules": ("Rules triggered", None), "due_date": ("Due", None),
    "due_date_basis": ("Due date basis", None), "worst_case_category": ("Worst-case category", None),
    "requires_review": ("Needs review", None), "review_reasons": ("Review reasons", None),
    "action_note": ("Action note", None),
    # Review: match rows, decisions, flags
    "review_status": ("Review status", None), "score": ("Match score", "%.2f"),
    "confidence": ("Confidence", None), "missing_hint": ("Hint", None),
    "decision": ("Decision", None), "applied_status": ("Status", None), "status": ("Status", None),
    "reason": ("Reason", None), "reviewer": ("Reviewer", None), "decided_utc": ("Decided (UTC)", None),
    "note": ("Note", None), "prev_feature_ids": ("Features before", None),
    "curr_feature_ids": ("Features now", None), "flag": ("Flag", None), "value": ("Value", None),
    # Weld alignment: changed sections
    "prev_start_ft": ("Start before (ft)", "%.1f"), "prev_end_ft": ("End before (ft)", "%.1f"),
    "curr_start_ft": ("Start now (ft)", "%.1f"), "curr_end_ft": ("End now (ft)", "%.1f"),
    "match_types": ("Match types", None), "n_prev_joints": ("Joints before", "%d"),
    "n_curr_joints": ("Joints now", "%d"), "overlaps_repair": ("Overlaps a repair", None),
    # Match side-by-side fields (row labels)
    "log_dist_ft": ("Distance (ft)", None), "wm_row": ("Weld map row", None),
    "us_gw_id": ("Upstream girth weld", None), "dist_from_us_gw_ft": ("Distance from upstream weld (ft)", None),
    "wt_nom_in": ("Nominal wall (in)", None), "depth_pct": ("Depth (% WT)", None),
    "dent_depth_in": ("Dent depth (in)", None), "length_in": ("Length (in)", None),
    "width_in": ("Width (in)", None), "surface": ("Surface", None), "seam_interaction": ("Seam interaction", None),
    "dent_metal_loss": ("Dent with metal loss", None), "vendor_cluster_id": ("Vendor cluster", None),
    "ml_origin": ("Metal loss origin", None), "feature_id": ("Feature", None),
}


def field_label(name: str) -> str:
    """The plain-word label of a column or field name; unknown names read as words."""
    return LABELS.get(name, (name.replace("_", " "), None))[0]


def column_config(df: pd.DataFrame) -> dict[str, st.column_config.Column]:
    """One entry per column of ``df`` that has a label: numbers get a number format, true/false columns a
    checkbox, the rest a text column. Other columns get no entry (Streamlit shows the raw name)."""
    config: dict[str, st.column_config.Column] = {}
    for name in df.columns:
        if name not in LABELS:
            continue
        label, fmt = LABELS[name]
        dtype = df[name].dtype
        if pd.api.types.is_bool_dtype(dtype):
            config[name] = st.column_config.CheckboxColumn(label)
        elif fmt is not None and pd.api.types.is_numeric_dtype(dtype):
            config[name] = st.column_config.NumberColumn(label, format=fmt)
        else:
            config[name] = st.column_config.TextColumn(label)
    return config
