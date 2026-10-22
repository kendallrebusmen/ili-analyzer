import pytest
import yaml
from pydantic import ValidationError

from ili.config import PROJECT_ROOT
from ili.ingest import load_vendor_map
from ili.schema import FeatureType
from ili.synth import _A_EVENT, _B_EVENT

MAPS = PROJECT_ROOT / "config" / "vendor_maps"


def test_synthetic_maps_load():
    a = load_vendor_map(MAPS / "synthetic_vendor_a.yaml")
    b = load_vendor_map(MAPS / "synthetic_vendor_b.yaml")
    assert a.file.format == "csv" and a.clock_format == "hhmm" and a.run_date is None
    assert b.file.format == "excel" and b.file.header_row == 3 and b.units.distance == "m"
    assert b.run_date is not None
    assert a.run_label == "vendorA" and b.run_label == "vendorB"


def _write(tmp_path, mutate):
    raw = yaml.safe_load((MAPS / "synthetic_vendor_a.yaml").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_unknown_standard_field_rejected(tmp_path):
    path = _write(tmp_path, lambda r: r["columns"].update(depth="Depth (%)"))
    with pytest.raises(ValidationError, match="depth"):
        load_vendor_map(path)


def test_run_fields_not_mappable(tmp_path):
    path = _write(tmp_path, lambda r: r["columns"].update(run_id="Run"))
    with pytest.raises(ValidationError, match="run_id"):
        load_vendor_map(path)


def test_vocab_normalized():
    a = load_vendor_map(MAPS / "synthetic_vendor_a.yaml")
    assert a.feature_types["metal loss"] is FeatureType.metal_loss
    assert "External".casefold() in a.surfaces


def test_duplicate_vocab_after_normalization_rejected(tmp_path):
    path = _write(tmp_path, lambda r: r["feature_types"].update({"metal loss ": "dent"}))
    with pytest.raises(ValidationError, match="metal loss"):
        load_vendor_map(path)


def test_every_synth_vocab_mapped():
    a = load_vendor_map(MAPS / "synthetic_vendor_a.yaml")
    b = load_vendor_map(MAPS / "synthetic_vendor_b.yaml")
    assert {v.casefold() for v in _A_EVENT.values()} <= set(a.feature_types)
    assert {v.casefold() for v in [*_B_EVENT.values(), "ML-CLUS"]} <= set(b.feature_types)
