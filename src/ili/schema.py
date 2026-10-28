"""Standard ILI feature schema (spec section 5.1).

Every run is converted to this schema at ingestion, whatever the vendor layout.
Units: feet for distances, inches for wall/depth/length/width, percent of
nominal wall for metal-loss depth, clock minutes 0-719 from 12:00.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class FeatureType(StrEnum):
    girth_weld = "girth_weld"
    metal_loss = "metal_loss"
    dent = "dent"
    gouge = "gouge"
    valve = "valve"
    tee = "tee"
    fitting = "fitting"
    sleeve = "sleeve"
    casing = "casing"
    marker = "marker"
    other = "other"


class Surface(StrEnum):
    EXT = "EXT"
    INT = "INT"
    MID = "MID"
    UNK = "UNK"


class MlOrigin(StrEnum):
    """Metal-loss origin: from the vendor feature-type code, `unknown` when the map is silent."""

    corrosion = "corrosion"
    manufacturing = "manufacturing"
    unknown = "unknown"


class StandardFeature(BaseModel):
    run_id: str
    run_date: date
    feature_id: str
    feature_type: FeatureType
    log_dist_ft: float
    us_gw_id: str | None = None
    dist_from_us_gw_ft: float | None = None
    joint_length_ft: float | None = None
    wt_nom_in: float | None = Field(default=None, gt=0)
    depth_pct: float | None = Field(default=None, ge=0, le=100)
    dent_depth_in: float | None = Field(default=None, ge=0)
    length_in: float | None = Field(default=None, ge=0)
    width_in: float | None = Field(default=None, ge=0)
    clock_min: int | None = Field(default=None, ge=0, le=719)
    surface: Surface | None = None
    ml_origin: MlOrigin | None = None
    seam_interaction: bool | None = None
    dent_metal_loss: bool | None = None
    vendor_cluster_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    elev_ft: float | None = None
    comments: str | None = None


STANDARD_COLUMNS: list[str] = list(StandardFeature.model_fields)

# Fixed features that split the line into alignment sections (spec 6.3).
ANCHOR_TYPES = {FeatureType.valve, FeatureType.tee, FeatureType.fitting}

# Anomaly feature types that are matched between runs (spec 6.4).
ANOMALY_TYPES = ("metal_loss", "dent", "gouge")
