"""Load and validate the YAML configuration (spec section 7).

The models mirror ``config/settings.yaml`` and ``config/repair_criteria.yaml``
exactly. Unknown keys are rejected so a typo cannot silently fall back to a
default engineering value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

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


GrowthMethod = Literal["raw", "tolerance_adjusted", "segment_percentile", "new_feature", "full_life", "default_rate",
                    "half_life", "joint_percentile", "shrinking_fixed"]


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
    governing_profile: Literal["conservative", "pit_to_pit"]
    shrinking_rate_mpy: float | None = Field(ge=0)  # mpy; the constraint applies only when set
    joint_percentile: float = Field(ge=0, le=100)
    joint_percentile_min_matches: int = Field(strict=True, ge=1)
    exclude_recoated_from_stats: bool
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


# 195.452(h)(2): an operator must obtain sufficient information to discover a condition no later than
# 180 days after an assessment. A configured discovery limit beyond it would allow a non-compliant date.
REGULATORY_MAX_DISCOVERY_DAYS = 180
DUE_DAY_CATEGORIES = ("60_day", "180_day")
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ClassificationCfg(_Strict):
    criteria_file: str
    girth_weld_proximity_in: float | None
    crossing_proximity_ft: float | None
    seam_clock_min: float | None
    seam_clock_tol_min: float | None
    general_corrosion_min_dim_in: float | None
    widespread_circ_min_width_frac: float | None
    due_days: dict[Literal["60_day", "180_day"], _PositiveInt]
    discovery_max_days: Annotated[int, Field(strict=True, gt=0, le=REGULATORY_MAX_DISCOVERY_DAYS)]

    @field_validator("due_days")
    @classmethod
    def _both_timelines(cls, due_days: dict) -> dict:
        missing = [c for c in DUE_DAY_CATEGORIES if c not in due_days]
        if missing:
            raise ValueError(f"due_days needs a positive whole number of days for {', '.join(missing)}")
        return due_days


class CpOverlayCfg(_Strict):
    cp_join_distance_ft: float
    criterion_instant_off_mv: float


class ReportCfg(_Strict):
    # presentation only: nothing here changes a number in the analysis
    largest_changes_n: _PositiveInt  # rows in the report's "largest changes" table
    dig_list_sheet: str = Field(min_length=1)  # name of the dig list worksheet
    ytc_flag_years: float = Field(gt=0)  # years: the "YTC < N yrs" row of the IMP class summary
    depth_bins_pct_wt: list[float]  # % of wall; lower edges of the depth bins, the last bin is open-ended
    dent_bins_pct_od: list[float]  # % of OD; lower edges of the dent bins, the last bin is open-ended
    aoi_group_ft: float = Field(ge=0)  # ft: anomalies within this distance form one anomalies-of-interest group
    aoi_count: _PositiveInt  # groups listed in anomalies of interest

    @field_validator("depth_bins_pct_wt", "dent_bins_pct_od")
    @classmethod
    def _bins_increase_from_zero(cls, v: list[float]) -> list[float]:
        if not v or v[0] != 0 or any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("bin edges must start at 0 and be strictly increasing")
        return v

    @field_validator("dig_list_sheet")
    @classmethod
    def _excel_sheet_name(cls, v: str) -> str:
        # Excel's own rules, so a bad name fails here and not deep inside openpyxl
        if len(v) > 31 or any(c in v for c in "[]:*?/\\"):
            raise ValueError("a worksheet name is 1-31 characters, none of [ ] : * ? / \\")
        return v


class ReviewCfg(_Strict):
    # presentation only: nothing here changes a number in the analysis
    nearby_window_ft: float = Field(gt=0)  # the review page's nearby-features window (ft)


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
    report: ReportCfg
    review: ReviewCfg


class Rule(_Strict):
    id: str
    category: Literal["immediate", "60_day", "180_day", "evaluate", "imp_scheduled"]
    applies_to: Literal["hca", "non_hca", "all"]
    feature_types: list[FeatureType]
    condition: str
    description: str
    citation: str
    requires_review: bool


_ML, _DENT, _GOUGE = FeatureType.metal_loss, FeatureType.dent, FeatureType.gouge
_ALL_ANOMALIES = frozenset({_ML, _DENT, _GOUGE})
# The 16 HCA rules of spec 6.8 (49 CFR 195.452(h)(4)) with the category the regulation assigns and the
# feature types each must cover at least (as shipped). Their thresholds live in the criteria file; this is
# the structural floor an edit of the file may not go below: a required rule must apply to HCA features
# (applies_to hca or all), keep its regulatory category, and keep every one of these feature types.
REQUIRED_HCA_RULES: dict[str, tuple[str, frozenset[FeatureType]]] = {
    "IMM-1": ("immediate", frozenset({_ML, _GOUGE})),  # (i)(A); a gouge is metal loss too
    "IMM-2": ("immediate", frozenset({_ML})),  # (i)(B)
    "IMM-3": ("immediate", frozenset({_DENT})),  # (i)(C)
    "IMM-4": ("immediate", frozenset({_DENT})),  # (i)(D)
    "IMM-5": ("immediate", _ALL_ANOMALIES),  # (i)(E)
    "D60-1": ("60_day", frozenset({_DENT})),  # (ii)(A)
    "D60-2": ("60_day", frozenset({_DENT})),  # (ii)(B)
    "D180-1": ("180_day", frozenset({_DENT})),  # (iii)(A)
    "D180-2": ("180_day", frozenset({_DENT})),  # (iii)(B)
    "D180-3": ("180_day", frozenset({_DENT})),  # (iii)(C)
    "D180-4": ("180_day", frozenset({_ML})),  # (iii)(D)
    "D180-5": ("180_day", frozenset({_ML})),  # (iii)(E)
    "D180-6": ("180_day", frozenset({_ML})),  # (iii)(F)
    "D180-7": ("180_day", _ALL_ANOMALIES),  # (iii)(G)
    "D180-8": ("180_day", frozenset({_ML})),  # (iii)(H)
    "D180-9": ("180_day", frozenset({_GOUGE})),  # (iii)(I)
}
REQUIRED_HCA_IDS = tuple(REQUIRED_HCA_RULES)
HCA_APPLIES_TO = ("hca", "all")


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
        by_id = {r.id: r for r in self.rules}
        missing = [i for i in REQUIRED_HCA_IDS if i not in by_id]
        if missing:
            raise ValueError(f"missing required 195.452(h)(4) rule ids: {', '.join(missing)}")
        for rid, (category, types) in REQUIRED_HCA_RULES.items():
            r = by_id[rid]
            if r.applies_to not in HCA_APPLIES_TO:
                raise ValueError(f"rule {rid}: a required 195.452(h)(4) rule must apply to HCA features "
                                 f"(applies_to {' or '.join(HCA_APPLIES_TO)}), got {r.applies_to!r}")
            if r.category != category:
                raise ValueError(f"rule {rid}: a required 195.452(h)(4) rule keeps its regulatory category "
                                 f"{category!r}, got {r.category!r}")
            lost = sorted(t.value for t in types - set(r.feature_types))
            if lost:
                raise ValueError(f"rule {rid}: a required 195.452(h)(4) rule must cover feature types "
                                 f"{sorted(t.value for t in types)}; missing {lost}")
        return self


def _read_yaml(path: Path) -> dict:
    """The parsed YAML file (empty file: ``{}``). Malformed YAML is a ``ValueError`` naming the file, so
    every CLI reports it as a usage error (exit 2) rather than a traceback."""
    with open(path, encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: malformed YAML: {exc}") from None


def load_settings(path: Path = DEFAULT_SETTINGS) -> Settings:
    return Settings.model_validate(_read_yaml(Path(path)))


def load_repair_criteria(path: Path = DEFAULT_CRITERIA) -> RepairCriteria:
    return RepairCriteria.model_validate(_read_yaml(Path(path)))
