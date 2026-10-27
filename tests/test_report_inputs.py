import json
import re
import shutil
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ili import classify, config, remaining_life
from ili.report_data import ReportInputs, input_hashes, load_report_inputs
from test_classify_cli import _classify_args
from test_rl_cli import _args as rl_args
from test_rl_cli import _prepare


@pytest.fixture(scope="module")
def built(synth, tmp_path_factory):
    """The 5-mile synth through M2-M7 once: (data dir, out dir, standardized dir, comparison dir)."""
    _, data = synth
    out = tmp_path_factory.mktemp("report_inputs") / "out"
    std, folder = _prepare(data, out)
    assert remaining_life.main(rl_args(std, folder, data, out)) == 0
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    return data, out, std, folder


def _load(built, comparison=None, std=None, crossings=True, digs=True, criteria=config.DEFAULT_CRITERIA,
          settings=config.DEFAULT_SETTINGS):
    data, out, std0, folder = built
    std = std or std0
    ref = data / "reference"
    return load_report_inputs(comparison or folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", out,
                              ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                              ref / "foreign_crossings.csv" if crossings else None,
                              ref / "digs.csv" if digs else None, settings, criteria)


def _copy_comparison(built, tmp_path):
    folder = tmp_path / "cmp"
    shutil.copytree(built[3], folder)
    return folder


def test_loads_every_input(built):
    inputs = _load(built)
    assert isinstance(inputs, ReportInputs)
    for f in fields(inputs):
        assert getattr(inputs, f.name) is not None, f.name
    assert set(inputs.summaries) == {"alignment", "match", "growth", "remaining_life", "classification"}
    assert len(inputs.classification) > 0 and len(inputs.growth) > 0
    assert inputs.ingest_manifest["runs"][0]["run_id"] == "2021_vendorA"
    assert inputs.settings.report.largest_changes_n == 20
    assert all(p.is_absolute() and p.is_file() for p in inputs.paths.values())
    assert {"digs", "crossings", "criteria", "settings", "classification_summary"} <= set(inputs.paths)
    hashes = input_hashes(inputs)
    assert len(hashes) == len(inputs.paths) and list(hashes) == sorted(hashes)
    assert all(len(h) == 64 for h in hashes.values())


def test_run_id_mismatch_is_error(built, tmp_path):
    folder = _copy_comparison(built, tmp_path)
    path = folder / "growth_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["curr_run_id"] = "2030_other"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="growth_summary.json.*different runs|different runs.*growth_summary.json"):
        _load(built, comparison=folder)


def test_std_run_id_mismatch_is_error(built, tmp_path):
    std = tmp_path / "std"
    shutil.copytree(built[2], std)
    path = std / "2026_vendorB.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("2026_vendorB", "2026_vendorZ"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"2026_vendorB\.csv.*2026_vendorZ"):
        _load(built, std=std)


def test_criteria_changed_since_classification_is_error(built, tmp_path):
    changed = tmp_path / "criteria.yaml"
    changed.write_text(config.DEFAULT_CRITERIA.read_text(encoding="utf-8") + "\n# edited after classify\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match=r"criteria\.yaml.*classification_summary\.json"):
        _load(built, criteria=changed)


def test_missing_required_file_is_error(built, tmp_path):
    folder = _copy_comparison(built, tmp_path)
    (folder / "growth_segments.csv").unlink()
    with pytest.raises(ValueError, match="growth_segments.csv"):
        _load(built, comparison=folder)


def test_digs_missing_column_is_error(built, tmp_path):
    bad = tmp_path / "digs.csv"
    bad.write_text("feature_id,field_depth_pct\n1,10\n", encoding="utf-8")
    data, out, std, folder = built
    ref = data / "reference"
    with pytest.raises(ValueError, match=r"digs\.csv.*field_length_in"):
        load_report_inputs(folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", out,
                           ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                           None, bad, config.DEFAULT_SETTINGS, config.DEFAULT_CRITERIA)


def test_optional_inputs_absent(built):
    inputs = _load(built, crossings=False, digs=False)
    assert inputs.crossings is None and inputs.digs is None
    assert "digs" not in inputs.paths and "crossings" not in inputs.paths


def test_report_settings_validated(tmp_path):
    raw = yaml.safe_load(config.DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    raw["report"]["largest_changes_n"] = 0
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="largest_changes_n"):
        config.load_settings(path)


def _settings_file(tmp_path, section, key, value):
    raw = yaml.safe_load(config.DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("section, key, value, summary", [
    ("tool", "depth_tolerance_pct_wt", 5, "remaining_life_summary.json"),  # recorded by M6 and M7
    ("remaining_life", "rl_factor", 0.4, "remaining_life_summary.json"),
    ("strength", "method", "original_b31g", "remaining_life_summary.json"),
    ("classification", "discovery_max_days", 170, "classification_summary.json"),
    ("classification", "due_days", {"60_day": 60, "180_day": 170}, "classification_summary.json"),
])
def test_settings_changed_since_classification_is_error(built, tmp_path, section, key, value, summary):
    changed = _settings_file(tmp_path, section, key, value)
    with pytest.raises(ValueError, match=rf"{section}\.{key}.*{re.escape(summary)}") as err:
        _load(built, settings=changed)
    message = str(err.value)
    assert "settings.yaml" in message
    recorded = json.loads((built[3] / summary).read_text(encoding="utf-8"))
    for v in (value, recorded[key]):  # both values named (a dict's key order may differ)
        assert all(str(x) in message for x in (v.values() if isinstance(v, dict) else [v]))


def test_settings_float_round_trip_is_not_a_change(built, tmp_path):
    inputs = _load(built, settings=_settings_file(tmp_path, "strength", "design_factor", 0.72 + 1e-15))
    assert inputs.settings.strength.design_factor != 0.72


def test_report_time_settings_are_not_checked(built, tmp_path):
    inputs = _load(built, settings=_settings_file(tmp_path, "report", "largest_changes_n", 5))
    assert inputs.settings.report.largest_changes_n == 5


def test_criteria_differing_only_in_line_endings_is_named(built, tmp_path):
    raw = config.DEFAULT_CRITERIA.read_bytes()
    other = raw.replace(b"\r\n", b"\n") if b"\r\n" in raw else raw.replace(b"\n", b"\r\n")
    changed = tmp_path / "criteria.yaml"
    changed.write_bytes(other)
    with pytest.raises(ValueError, match=r"differ only in line endings"):
        _load(built, criteria=changed)
    changed.write_bytes(raw + b"# edited\n")
    with pytest.raises(ValueError, match=r"changed since") as err:
        _load(built, criteria=changed)
    assert "line endings" not in str(err.value)


def _ingest_copy(built, tmp_path):
    ingest = tmp_path / "ingest"
    ingest.mkdir()
    for name in ("ingest_manifest.json", "validation_issues.csv"):
        shutil.copy(built[1] / name, ingest / name)
    return ingest


def _load_with_ingest(built, ingest):
    data, out, std, folder = built
    ref = data / "reference"
    return load_report_inputs(folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", ingest,
                              ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                              None, None, config.DEFAULT_SETTINGS, config.DEFAULT_CRITERIA)


def test_ingest_manifest_without_the_comparison_runs_is_error(built, tmp_path):
    ingest = _ingest_copy(built, tmp_path)
    path = ingest / "ingest_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["runs"] = [r for r in manifest["runs"] if r["run_id"] != "2026_vendorB"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"ingest_manifest\.json.*2026_vendorB"):
        _load_with_ingest(built, ingest)


def test_validation_issues_filtered_to_the_comparison_runs(built, tmp_path):
    ingest = _ingest_copy(built, tmp_path)
    path = ingest / "validation_issues.csv"
    issues = pd.read_csv(path, dtype=str)
    extra = pd.DataFrame([{"run_id": "2030_other", "source_row": "1", "feature_id": "X-1", "field": "depth_pct",
                           "value": "200", "reason": "out_of_range"},
                          {"run_id": "2026_vendorB", "source_row": "2", "feature_id": "B-1", "field": "depth_pct",
                           "value": "200", "reason": "out_of_range"}], columns=issues.columns)
    pd.concat([issues, extra], ignore_index=True).to_csv(path, index=False, lineterminator="\n")
    inputs = _load_with_ingest(built, ingest)
    assert set(inputs.validation_issues.run_id) == {"2026_vendorB"}
    assert len(inputs.validation_issues) == len(issues[issues.run_id == "2026_vendorB"]) + 1


def test_missing_configuration_file_names_its_role(built, tmp_path):
    with pytest.raises(ValueError, match="settings not found") as err:
        _load(built, settings=tmp_path / "nope.yaml")
    assert "configuration" in str(err.value) and "alongside" not in str(err.value)


def test_unreadable_csv_names_the_file(built, tmp_path):
    empty = tmp_path / "digs.csv"
    empty.write_bytes(b"")
    data, out, std, folder = built
    ref = data / "reference"
    with pytest.raises(ValueError, match=r"digs\.csv"):
        load_report_inputs(folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", out,
                           ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                           None, empty, config.DEFAULT_SETTINGS, config.DEFAULT_CRITERIA)
