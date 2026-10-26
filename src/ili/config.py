"""Load and validate the YAML configuration (spec section 7).

The models mirror ``config/settings.yaml`` and ``config/repair_criteria.yaml``
exactly. Unknown keys are rejected so a typo cannot silently fall back to a
default engineering value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ili.rules import compile_condition
from ili.schema import FeatureType
from ili.vocabulary import VARIABLES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_CRITERIA = PROJECT_ROOT / "config" / "repair_criteria.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCfg(_Strict):
    depth_tolerance_pct_wt: float
    depth_certainty: float
    length_tolerance_in: float
    length_certainty: float
    reporting_threshold_pct_wt: float


class ConfidenceBands(_Strict):
    high: float
    medium: float


class AlignmentCfg(_Strict):
    joint_length_tol_ft: float
    joint_length_tol_pct: float
    gap_penalty: float
    missed_weld_penalty: float
    gap_penalty_repair: float
    anchor_tol_ft: float
    band_joints: int
    confidence_bands: ConfidenceBands


class MatchWeights(_Strict):
    axial: float
    circ: float
    depth_change: float


class InteractionRule(_Strict):
    axial_spacing_in: float
    circ_spacing_in: float


class MatchingCfg(_Strict):
    axial_tol_in: float
    clock_tol_min: float
    near_weld_ft: float
    max_plausible_growth_mpy: float
    weights: MatchWeights
    confidence_bands: ConfidenceBands
    interaction_rule: InteractionRule


GrowthMethod = Literal["raw", "tolerance_adjusted", "segment_percentile", "new_feature", "full_life", "default_rate"]


class GoverningRule(_Strict):
    # At least one method per class: an empty list would leave every row of the class without a rate.
    matched: list[GrowthMethod] = Field(min_length=1)
    new: list[GrowthMethod] = Field(min_length=1)
    no_history: list[GrowthMethod] = Field(min_length=1)


class GrowthCfg(_Strict):
    k_sigma: float
    min_rate_mpy: float
    segment_definition: Literal["hca_segment", "fixed_length", "reference_segments"]
    fixed_segment_length_mi: float
    segment_percentile: float
    default_rate_mpy: float | None
    length_growth: Literal["none", "proportional", "fixed_rate"]
    length_rate_ipy: float
    governing: GoverningRule
    diagnostic_bin_mpy: float


class StrengthCfg(_Strict):
    method: Literal["modified_b31g", "original_b31g"]
    depth_basis: Literal["reported", "reported_plus_tolerance"]
    length_basis: Literal["reported", "reported_plus_tolerance"]
    design_factor: float


class NextAssessmentYears(_Strict):
    hca: float
    non_hca: float


class RemainingLifeCfg(_Strict):
    horizon_yr: float = Field(gt=0)
    time_step_yr: float = Field(gt=0)  # crossing-search grid step (numerical setting)
    rl_factor: float
    default_next_assessment_yr: NextAssessmentYears
    hca_max_interval_months: float


class ClassificationCfg(_Strict):
    criteria_file: str
    girth_weld_proximity_in: float | None
    crossing_proximity_ft: float | None
    seam_clock_min: float | None
    seam_clock_tol_min: float | None
    general_corrosion_min_dim_in: float | None
    widespread_circ_min_width_frac: float | None
    due_days: dict[Literal["60_day", "180_day"], int]
    discovery_max_days: int


class CpOverlayCfg(_Strict):
    cp_join_distance_ft: float
    criterion_instant_off_mv: float


class Settings(_Strict):
    units: Literal["us_customary"]
    tool: ToolCfg
    alignment: AlignmentCfg
    matching: MatchingCfg
    growth: GrowthCfg
    strength: StrengthCfg
    remaining_life: RemainingLifeCfg
    classification: ClassificationCfg
    cp_overlay: CpOverlayCfg


class Rule(_Strict):
    id: str
    category: Literal["immediate", "60_day", "180_day", "evaluate", "imp_scheduled"]
    applies_to: Literal["hca", "non_hca", "all"]
    feature_types: list[FeatureType]
    condition: str
    description: str
    citation: str
    requires_review: bool


REQUIRED_HCA_IDS = (
    "IMM-1", "IMM-2", "IMM-3", "IMM-4", "IMM-5", "D60-1", "D60-2",
    "D180-1", "D180-2", "D180-3", "D180-4", "D180-5", "D180-6", "D180-7", "D180-8", "D180-9",
)


class RepairCriteria(_Strict):
    version: str
    rules: list[Rule]

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, rules: list[Rule]) -> list[Rule]:
        seen: set[str] = set()
        dupes = sorted({r.id for r in rules if r.id in seen or seen.add(r.id)})
        if dupes:
            raise ValueError(f"duplicate rule ids: {', '.join(dupes)}")
        return rules

    @model_validator(mode="after")
    def _conditions_and_required_ids(self) -> "RepairCriteria":
        names = set(VARIABLES)
        for r in self.rules:
            try:
                compile_condition(r.condition, names)
            except ValueError as exc:
                raise ValueError(f"rule {r.id}: invalid condition: {exc}") from exc
        present = {r.id for r in self.rules}
        missing = [i for i in REQUIRED_HCA_IDS if i not in present]
        if missing:
            raise ValueError(f"missing required 195.452(h)(4) rule ids: {', '.join(missing)}")
        return self


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(path: Path = DEFAULT_SETTINGS) -> Settings:
    return Settings.model_validate(_read_yaml(Path(path)))


def load_repair_criteria(path: Path = DEFAULT_CRITERIA) -> RepairCriteria:
    return RepairCriteria.model_validate(_read_yaml(Path(path)))
