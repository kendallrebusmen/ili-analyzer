import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
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
    # the immediate list: one line per immediate feature (match id, feature id, rule, action note)
    immediate = table[table.governing_category == "immediate"]
    assert len(immediate) > 0 and f"immediate ({len(immediate)}):" in printed
    for r in immediate.itertuples(index=False):
        assert f"  {r.match_id}  {r.curr_feature_id}  {r.governing_rule}  {r.action_note}" in printed
    # every review reason, and the could-be-more-severe counts by worst category
    reasons = summary["counts_by_review_reason"]
    assert ("requires review: " + str(summary["n_requires_review"]) + " ("
            + ", ".join(f"{k} {v}" for k, v in reasons.items()) + ")") in printed
    more = summary["could_be_more_severe_by_worst_category"]
    assert reasons.get("could_be_more_severe", 0) == sum(more.values())
    assert (f"could be more severe: {sum(more.values())} (worst case: immediate {more['immediate']}, "
            f"60_day {more['60_day']}, 180_day {more['180_day']}, imp_scheduled {more['imp_scheduled']})") in printed
    more_rows = table.review_reasons.fillna("").str.split(";").map(lambda x: "could_be_more_severe" in x)
    assert (table.worst_case_category[more_rows].map(classify.CATEGORY_PRIORITY.index)
            < table.governing_category[more_rows].map(classify.CATEGORY_PRIORITY.index)).all()
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


def _settings_file(tmp_path, **classification):
    raw = yaml.safe_load(config.DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    raw["classification"].update(classification)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_cli_uses_criteria_file_from_settings(synth, tmp_path, capsys):
    """Without --criteria the settings' classification.criteria_file is used (relative: from the project
    root); the summary records the path and the sha256 of the file actually read."""
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert Path(summary["criteria_path"]) == config.PROJECT_ROOT / "config" / "repair_criteria.yaml"

    other = tmp_path / "other_criteria.yaml"
    other.write_text(config.DEFAULT_CRITERIA.read_text(encoding="utf-8").replace(
        'version: "', 'version: "OTHER - ', 1), encoding="utf-8")
    settings = _settings_file(tmp_path, criteria_file=str(other))
    capsys.readouterr()
    assert classify.main(_classify_args(std, folder, data, out, "--settings", str(settings))) == 0, \
        capsys.readouterr().err
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["criteria_path"] == str(other)
    assert summary["criteria_sha256"] == hashlib.sha256(other.read_bytes()).hexdigest()
    assert summary["criteria_sha256"] != hashlib.sha256(config.DEFAULT_CRITERIA.read_bytes()).hexdigest()
    assert summary["criteria_version"].startswith("OTHER - ")

    # --criteria overrides the setting
    assert classify.main(_classify_args(std, folder, data, out, "--settings", str(settings),
                                        "--criteria", str(config.DEFAULT_CRITERIA))) == 0
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["criteria_path"] == str(config.DEFAULT_CRITERIA)
    assert summary["criteria_sha256"] == hashlib.sha256(config.DEFAULT_CRITERIA.read_bytes()).hexdigest()


def test_cli_summary_records_every_input_setting(synth, tmp_path, capsys):
    data, out, std, folder = _through_rl(synth, tmp_path, capsys)
    settings = _settings_file(tmp_path, girth_weld_proximity_in=12.0, crossing_proximity_ft=50.0,
                              seam_clock_min=90.0, seam_clock_tol_min=15.0, general_corrosion_min_dim_in=6.0,
                              widespread_circ_min_width_frac=0.5, due_days={"60_day": 45, "180_day": 150},
                              discovery_max_days=120)
    args = _classify_args(std, folder, data, out, "--settings", str(settings))
    i = args.index("--crossings")
    del args[i:i + 2]  # no crossings file
    assert classify.main(args) == 0, capsys.readouterr().err
    text = (folder / "classification_summary.json").read_text(encoding="utf-8")
    summary = json.loads(text)
    run_date = str(pd.read_csv(std / "2026_vendorB.csv", usecols=["run_date"]).run_date.iloc[0])[:10]
    assert summary["assessment_date"] == run_date
    assert {k: summary[k] for k in (
        "girth_weld_proximity_in", "crossing_proximity_ft", "seam_clock_min", "seam_clock_tol_min",
        "general_corrosion_min_dim_in", "widespread_circ_min_width_frac", "due_days", "discovery_max_days",
        "crossings_file_given", "depth_basis", "depth_tolerance_pct_wt")} == {
        "girth_weld_proximity_in": 12.0, "crossing_proximity_ft": 50.0, "seam_clock_min": 90.0,
        "seam_clock_tol_min": 15.0, "general_corrosion_min_dim_in": 6.0, "widespread_circ_min_width_frac": 0.5,
        "due_days": {"60_day": 45, "180_day": 150}, "discovery_max_days": 120, "crossings_file_given": False,
        "depth_basis": "reported_plus_tolerance", "depth_tolerance_pct_wt": 10.0}
    assert "NaN" not in text and "Infinity" not in text
    # the default settings: PLACEHOLDERs null, crossings given
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["crossings_file_given"] is True and summary["seam_clock_min"] is None
    assert summary["girth_weld_proximity_in"] is None and summary["discovery_max_days"] == 180
