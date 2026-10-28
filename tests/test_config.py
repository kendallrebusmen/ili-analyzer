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


def test_alignment_extra_keys():
    a = load_settings().alignment
    assert a.missed_weld_penalty == 1.0
    assert a.gap_penalty_repair == 0.5
    assert a.anchor_tol_ft == 100
    assert a.band_joints == 30
    assert (a.confidence_bands.high, a.confidence_bands.medium) == (1.0, 2.0)


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
    assert [r.id for r in c.rules][:3] == ["IMM-1", "IMM-2", "IMM-3"]
    assert c.rules[0].feature_types == [FeatureType.metal_loss, FeatureType.gouge]
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


def test_growth_governing_rule_loads():
    g = load_settings().growth
    assert g.governing.matched == ["segment_percentile", "tolerance_adjusted"]
    assert g.governing.new == ["new_feature", "segment_percentile"]
    assert g.governing.no_history == ["full_life", "default_rate"]
    assert g.diagnostic_bin_mpy == 1.0


def test_growth_governing_rejects_unknown_method(tmp_path):
    path = _write_settings(tmp_path, lambda r: r["growth"]["governing"].update(matched=["segment_percentile", "median"]))
    with pytest.raises(ValueError):
        load_settings(path)


def test_growth_governing_rejects_empty_list(tmp_path):
    path = _write_settings(tmp_path, lambda r: r["growth"]["governing"].update(new=[]))
    with pytest.raises(ValueError, match="new"):
        load_settings(path)


def test_strength_length_basis_loads(tmp_path):
    assert load_settings().strength.length_basis == "reported"
    path = _write_settings(tmp_path, lambda r: r["strength"].update(length_basis="reported_plus_tolerance"))
    assert load_settings(path).strength.length_basis == "reported_plus_tolerance"
    bad = _write_settings(tmp_path, lambda r: r["strength"].update(length_basis="bogus"))
    with pytest.raises(ValidationError):
        load_settings(bad)


@pytest.mark.parametrize("loader,name", [(load_settings, "settings.yaml"), (load_repair_criteria, "criteria.yaml")])
def test_malformed_yaml_is_a_value_error_naming_the_file(tmp_path, loader, name):
    path = tmp_path / name
    path.write_text("rules: [\n  - id: x", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed YAML") as exc:
        loader(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("due_days", [
    {"60_day": 60},  # 180_day missing
    {"180_day": 180},  # 60_day missing
    {"60_day": 0, "180_day": 180},
    {"60_day": 60, "180_day": -1},
    {"60_day": 60.5, "180_day": 180},
    {"60_day": "60", "180_day": 180},
    {"60_day": True, "180_day": 180},
    {"60_day": 60, "180_day": 180, "30_day": 30},
])
def test_due_days_need_both_timelines_as_positive_ints(tmp_path, due_days):
    path = _write_settings(tmp_path, lambda r: r["classification"].update(due_days=due_days))
    with pytest.raises(ValidationError, match="due_days"):
        load_settings(path)


@pytest.mark.parametrize("days, ok", [(0, False), (-5, False), (1, True), (180, True), (181, False),
                                      (90.0, False)])
def test_discovery_max_days_bounds(tmp_path, days, ok):
    """195.452(h)(2): discovery no later than 180 days after the assessment."""
    path = _write_settings(tmp_path, lambda r: r["classification"].update(discovery_max_days=days))
    if ok:
        assert load_settings(path).classification.discovery_max_days == days
    else:
        with pytest.raises(ValidationError, match="discovery_max_days"):
            load_settings(path)


@pytest.mark.parametrize("name", ["", "x" * 32, "a:b", "a/b", "a\\b", "a?b", "a*b", "[a]", "a]b"])
def test_bad_dig_list_sheet_rejected(tmp_path, name):
    path = _write_settings(tmp_path, lambda r: r["report"].update(dig_list_sheet=name))
    with pytest.raises(ValidationError):
        load_settings(path)


def test_dig_list_sheet_31_chars_ok(tmp_path):
    path = _write_settings(tmp_path, lambda r: r["report"].update(dig_list_sheet="x" * 31))
    assert load_settings(path).report.dig_list_sheet == "x" * 31


@pytest.mark.parametrize("window", [0, -1])
def test_review_window_must_be_positive(tmp_path, window):
    path = _write_settings(tmp_path, lambda r: r["review"].update(nearby_window_ft=window))
    with pytest.raises(ValueError, match="nearby_window_ft"):
        load_settings(path)


def test_shipped_review_window():
    assert load_settings().review.nearby_window_ft == 10


def test_growth_profile_values(tmp_path):
    g = load_settings().growth
    assert g.governing_profile == "conservative"
    assert g.shrinking_rate_mpy is None
    assert g.joint_percentile == 90
    assert g.joint_percentile_min_matches == 3
    assert g.exclude_recoated_from_stats is True
    path = _write_settings(tmp_path, lambda r: r["growth"].update(governing_profile="median"))
    with pytest.raises(ValueError):
        load_settings(path)
    path = _write_settings(tmp_path, lambda r: r["growth"].update(shrinking_rate_mpy=-1.0))
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.parametrize("value", [101, -1])
def test_joint_percentile_bounds(tmp_path, value):
    path = _write_settings(tmp_path, lambda r: r["growth"].update(joint_percentile=value))
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.parametrize("value", [0, 2.5])
def test_min_matches_positive_int(tmp_path, value):
    path = _write_settings(tmp_path, lambda r: r["growth"].update(joint_percentile_min_matches=value))
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.parametrize("key", ["depth_bins_pct_wt", "dent_bins_pct_od"])
@pytest.mark.parametrize("bins", [[0, 10, 10], [5, 10], [0, 20, 10], []])
def test_bins_must_increase(tmp_path, key, bins):
    path = _write_settings(tmp_path, lambda r: r["report"].update({key: bins}))
    with pytest.raises(ValueError, match=key):
        load_settings(path)
