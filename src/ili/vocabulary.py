"""The published variable vocabulary for repair-criteria conditions (spec section 6.8).

Conditions in ``config/repair_criteria.yaml`` may use ONLY the names in ``VARIABLES``; the
criteria loader compiles every condition against this set, so a typo or an invented name is a
load-time error. Any variable may be None (unknown); the condition evaluator then returns an
unknown result rather than a silent pass or fail.

This module must not import ``ili.config`` (config imports it).

DRAFT - for review by qualified personnel.
"""

from __future__ import annotations

VARIABLES: dict[str, str] = {
    # Basics
    "feature_type": "feature class as a string: 'metal_loss', 'dent' or 'gouge'",
    "could_affect_hca": "True if the feature is in or could affect an HCA segment (boolean)",
    "od_in": "nominal outside diameter (in)",
    "wt_in": "nominal wall thickness (in)",
    "clock_min": "clock position of the feature (minutes on the 12-hour face, 0-719)",
    "top_of_pipe": "True if the feature is above the 4 and 8 o'clock positions, computed from clock_min (boolean)",
    # Metal loss and gouges
    "depth_pct_reported": "depth as reported by the vendor (percent of nominal wall)",
    "depth_pct_assessed": "tolerance-adjusted depth used for assessment (percent of nominal wall)",
    "length_in": "axial length (in)",
    "width_in": "circumferential width (in)",
    # Pressures
    "p_fail": "predicted burst pressure P_F at the anomaly (psi)",
    "p_safe": "safe operating pressure from the remaining-strength calculation (psi)",
    "mop_local": "maximum operating pressure at the anomaly location (psi)",
    # Dents
    "dent_depth_in": "dent depth (in)",
    "dent_depth_pct_od": "dent depth as percent of nominal outside diameter",
    "dent_metal_loss": "dent has any indication of metal loss, cracking or a stress riser: vendor flag or a reported metal-loss depth on the dent (boolean)",
    # Location
    "seam_interaction": "feature interacts with a longitudinal seam weld (boolean)",
    "near_girth_weld": "feature is within girth_weld_proximity_in of a girth weld (boolean)",
    "near_foreign_crossing": "feature is within crossing_proximity_ft of a foreign pipeline crossing (boolean)",
    "general_corrosion": "metal loss is general corrosion rather than pitting (boolean)",
    "widespread_circumferential": "metal loss is widespread circumferential corrosion (boolean)",
    # Flags
    "dig_required_by_remaining_life": "remaining-life analysis requires an excavation within the horizon (boolean)",
    "manual_immediate": "designated evaluator flagged the anomaly as requiring immediate action (manual entry, boolean)",
    "field_crack_confirmed": "potential crack indication confirmed as a crack at excavation (field entry, boolean)",
}
