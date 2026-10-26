import hashlib
import json

import pandas as pd
from conftest import MAP_A, MAP_B

from ili import classify, config, remaining_life
from test_rl_cli import _args as rl_args
from test_rl_cli import _prepare


def _classify_args(std, folder, data, out, *extra):
    ref = data / "reference"
    return ["--curr", str(std / "2026_vendorB.csv"), "--matches", str(folder / "match_table.csv"),
            "--remaining-life", str(folder / "remaining_life.csv"), "--pipe", str(ref / "pipe_properties.csv"),
            "--segments", str(ref / "hca_segments.csv"), "--crossings", str(ref / "foreign_crossings.csv"),
            "--out", str(out), *extra]


def _through_rl(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    assert remaining_life.main(rl_args(std, folder, data, out)) == 0
    capsys.readouterr()
    return data, out, std, folder


def test_cli_writes_outputs(synth, tmp_path, capsys):
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    code = classify.main(_classify_args(std, folder, data, out))
    assert code == 0, capsys.readouterr().err
    table = pd.read_csv(folder / "classification.csv", dtype={"match_id": str, "curr_feature_id": str})
    rl = pd.read_csv(folder / "remaining_life.csv")
    assert len(table) >= len(rl) > 0
    assert list(table.columns) == classify.CLASSIFICATION_COLUMNS
    assert set(table.governing_category) <= set(classify.CATEGORY_PRIORITY)
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert "DRAFT" in summary["label"]
    digest = hashlib.sha256(config.DEFAULT_CRITERIA.read_bytes()).hexdigest()
    assert summary["criteria_sha256"] == digest
    assert summary["n_features"] == len(table)
    assert sum(summary["counts_by_category"].values()) == len(table)
    gs = json.loads((folder / "growth_summary.json").read_text(encoding="utf-8"))
    assert summary["curr_run_id"] == gs["curr_run_id"] and summary["prev_run_id"] == gs["prev_run_id"]
    printed = capsys.readouterr().out
    assert "DRAFT" in printed and "requires review" in printed and "overdue" in printed
    assert "!!!!" in printed and summary["discovery_warning"] in printed
    assert summary["discovery_date_source"] == "latest_permissible"


def test_cli_entered_discovery_date_no_banner(synth, tmp_path, capsys):
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    run_date = pd.read_csv(std / "2026_vendorB.csv", usecols=["run_date"]).run_date.iloc[0]
    assert classify.main(_classify_args(std, folder, data, out, "--discovery-date", str(run_date)[:10])) == 0
    printed = capsys.readouterr().out
    assert "!!!!" not in printed
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["discovery_date_source"] == "entered" and summary["discovery_warning"] is None


def test_cli_bad_discovery_date_exit_2(synth, tmp_path, capsys):
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    assert classify.main(_classify_args(std, folder, data, out, "--discovery-date", "not-a-date")) == 2
    assert "error:" in capsys.readouterr().err
    assert not (folder / "classification.csv").exists()
