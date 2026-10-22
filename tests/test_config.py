import pytest
import yaml
from pydantic import ValidationError

from ili.config import (
    DEFAULT_CRITERIA,
    DEFAULT_SETTINGS,
    load_repair_criteria,
    load_settings,
)
from ili.schema import FeatureType


def test_shipped_settings_load():
    s = load_settings()
    assert s.tool.depth_tolerance_pct_wt == 10
    assert s.growth.default_rate_mpy is None
    assert s.strength.method == "modified_b31g"
    assert s.matching.weights.depth_change == 0.5
    assert s.remaining_life.default_next_assessment_yr.hca == 5
    assert s.classification.girth_weld_proximity_in is None


def _write_settings(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_unknown_key_rejected(tmp_path):
    path = _write_settings(tmp_path, lambda r: r["tool"].update(depth_tol=10))
    with pytest.raises(ValidationError):
        load_settings(path)


def test_bad_enum_rejected(tmp_path):
    path = _write_settings(tmp_path, lambda r: r["strength"].update(method="rstreng"))
    with pytest.raises(ValidationError):
        load_settings(path)


def test_missing_section_rejected(tmp_path):
    path = _write_settings(tmp_path, lambda r: r.pop("growth"))
    with pytest.raises(ValidationError):
        load_settings(path)


def test_shipped_criteria_load():
    c = load_repair_criteria()
    assert [r.id for r in c.rules] == ["IMM-1", "IMM-2", "D180-4"]
    assert c.rules[0].feature_types == [FeatureType.metal_loss]
    assert "VERIFY" in c.version


def test_duplicate_rule_id_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CRITERIA.read_text(encoding="utf-8"))
    raw["rules"].append(dict(raw["rules"][0]))
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="IMM-1"):
        load_repair_criteria(path)


def test_bad_category_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CRITERIA.read_text(encoding="utf-8"))
    raw["rules"][0]["category"] = "30_day"
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_repair_criteria(path)
