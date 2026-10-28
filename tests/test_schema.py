from datetime import date

import pytest
from pydantic import ValidationError

from ili.schema import ANCHOR_TYPES, STANDARD_COLUMNS, FeatureType, MlOrigin, StandardFeature, Surface


def _row(**overrides):
    base = dict(
        run_id="2021_vendorA",
        run_date=date(2021, 6, 15),
        feature_id="A-000001",
        feature_type="metal_loss",
        log_dist_ft=1234.5,
    )
    base.update(overrides)
    return base


def test_valid_row():
    f = StandardFeature(**_row(depth_pct=23.0, clock_min=180, surface="EXT"))
    assert f.feature_type is FeatureType.metal_loss
    assert f.surface is Surface.EXT
    assert f.dent_depth_in is None


def test_depth_bounds():
    for bad in (100.1, -0.1):
        with pytest.raises(ValidationError):
            StandardFeature(**_row(depth_pct=bad))
    for ok in (0, 100):
        assert StandardFeature(**_row(depth_pct=ok)).depth_pct == ok


def test_clock_bounds():
    for bad in (720, -1):
        with pytest.raises(ValidationError):
            StandardFeature(**_row(clock_min=bad))
    assert StandardFeature(**_row(clock_min=719)).clock_min == 719


def test_unknown_feature_type_rejected():
    with pytest.raises(ValidationError):
        StandardFeature(**_row(feature_type="corrosion"))


def test_required_fields():
    row = _row()
    del row["log_dist_ft"]
    with pytest.raises(ValidationError):
        StandardFeature(**row)


def test_columns_match_model():
    assert STANDARD_COLUMNS == list(StandardFeature.model_fields)
    assert STANDARD_COLUMNS[:3] == ["run_id", "run_date", "feature_id"]
    assert STANDARD_COLUMNS.index("ml_origin") == STANDARD_COLUMNS.index("surface") + 1
    assert STANDARD_COLUMNS[-4:] == ["lat", "lon", "elev_ft", "comments"]


def test_anchor_types():
    assert ANCHOR_TYPES == {FeatureType.valve, FeatureType.tee, FeatureType.fitting}


def test_ml_origin_enum():
    assert [m.value for m in MlOrigin] == ["corrosion", "manufacturing", "unknown"]
    assert StandardFeature(**_row()).ml_origin is None
