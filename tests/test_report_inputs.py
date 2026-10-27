import json
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
