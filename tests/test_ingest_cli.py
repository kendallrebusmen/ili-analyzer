import json
from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B, assert_matches_oracle

from ili.config import PROJECT_ROOT
from ili.ingest import ISSUE_COLUMNS, _parser, ingest_tally, main


def _paths(synth):
    _, out = synth
    return out / "tally_prev_vendor_a.csv", out / "tally_curr_vendor_b.xlsx"


def test_ingest_tally_run_ids(synth):
    a, b = _paths(synth)
    ra = ingest_tally(a, MAP_A, run_date=date(2021, 6, 15))
    rb = ingest_tally(b, MAP_B)
    assert (ra.run_id, rb.run_id) == ("2021_vendorA", "2026_vendorB")
    assert rb.run_date == date(2026, 6, 15)
    assert ra.issues.empty and rb.issues.empty
    assert_matches_oracle(rb.standard, synth[0].curr_std)


def test_cli_ingests_synthetic(synth, tmp_path, capsys):
    a, b = _paths(synth)
    code = main(["--run", str(a), str(MAP_A), "2021-06-15", "--run", str(b), str(MAP_B), "--out", str(tmp_path)])
    assert code == 0, capsys.readouterr().err
    std_a = pd.read_csv(tmp_path / "standardized" / "2021_vendorA.csv")
    std_b = pd.read_csv(tmp_path / "standardized" / "2026_vendorB.csv")
    assert len(std_a) == len(synth[0].prev_std) and len(std_b) == len(synth[0].curr_std)
    assert "source_row" in std_a.columns
    issues = pd.read_csv(tmp_path / "validation_issues.csv")
    assert issues.empty and list(issues.columns) == ISSUE_COLUMNS
    m = json.loads((tmp_path / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert "DRAFT" in m["label"]
    runs = {r["run_id"]: r for r in m["runs"]}
    assert runs["2021_vendorA"]["rows_clean"] == len(std_a) and runs["2021_vendorA"]["issues"] == 0
    assert len(runs["2026_vendorB"]["tally_sha256"]) == 64 and len(runs["2026_vendorB"]["vendor_map_sha256"]) == 64
    assert "2021_vendorA" in capsys.readouterr().out


def test_missing_run_date_errors(synth, tmp_path, capsys):
    a, _ = _paths(synth)
    assert main(["--run", str(a), str(MAP_A), "--out", str(tmp_path)]) == 2
    assert "run_date" in capsys.readouterr().err


def test_conflicting_run_date_errors(synth, tmp_path, capsys):
    _, b = _paths(synth)
    assert main(["--run", str(b), str(MAP_B), "2020-01-01", "--out", str(tmp_path)]) == 2
    assert "2026-06-15" in capsys.readouterr().err


def test_duplicate_run_ids_rejected(synth, tmp_path, capsys):
    a, _ = _paths(synth)
    args = ["--run", str(a), str(MAP_A), "2021-06-15"]
    assert main(args + args + ["--out", str(tmp_path)]) == 2
    assert "2021_vendorA" in capsys.readouterr().err


def test_bad_run_arg_count(synth, tmp_path, capsys):
    a, _ = _paths(synth)
    assert main(["--run", str(a), "--out", str(tmp_path)]) == 2
    assert "--run" in capsys.readouterr().err


def test_issues_written_with_reasons(synth, tmp_path):
    a, _ = _paths(synth)
    df = pd.read_csv(a, dtype=str, keep_default_na=False)
    row = df.index[df.Event == "Metal Loss"][0]
    df.loc[row, "Depth (%)"] = "130"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    out = tmp_path / "out"
    assert main(["--run", str(bad), str(MAP_A), "2021-06-15", "--out", str(out)]) == 0
    issues = pd.read_csv(out / "validation_issues.csv")
    assert issues[["run_id", "source_row", "reason"]].values.tolist() == [["2021_vendorA", row + 2, "depth_out_of_range"]]
    assert len(pd.read_csv(out / "standardized" / "2021_vendorA.csv")) == len(df) - 1


def test_separate_invocations_keep_each_runs_issues(synth, tmp_path):
    a, b = _paths(synth)
    df = pd.read_csv(a, dtype=str, keep_default_na=False)
    df.loc[df.index[df.Event == "Metal Loss"][0], "Depth (%)"] = "130"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    out = tmp_path / "out"
    assert main(["--run", str(bad), str(MAP_A), "2021-06-15", "--out", str(out)]) == 0
    assert main(["--run", str(b), str(MAP_B), "--out", str(out)]) == 0
    issues = pd.read_csv(out / "validation_issues.csv")
    assert issues.run_id.tolist() == ["2021_vendorA"]
    m = json.loads((out / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert sorted(r["run_id"] for r in m["runs"]) == ["2021_vendorA", "2026_vendorB"]
    # Re-ingesting A cleanly replaces A's issues, not B's entry.
    assert main(["--run", str(a), str(MAP_A), "2021-06-15", "--out", str(out)]) == 0
    assert pd.read_csv(out / "validation_issues.csv").empty
    m = json.loads((out / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert {r["run_id"]: r["issues"] for r in m["runs"]} == {"2021_vendorA": 0, "2026_vendorB": 0}


def test_load_standardized_roundtrip(synth, tmp_path):
    from ili.ingest import load_standardized

    a, b = _paths(synth)
    assert main(["--run", str(a), str(MAP_A), "2021-06-15", "--run", str(b), str(MAP_B), "--out", str(tmp_path)]) == 0
    std = load_standardized(tmp_path / "standardized" / "2026_vendorB.csv")
    gw = std.loc[std.feature_type == "girth_weld", "us_gw_id"]
    assert gw.iloc[0] == "1" and all(isinstance(v, str) for v in gw)
    assert str(std.clock_min.dtype) == "Int64"
    assert std.run_date.iloc[0] == date(2026, 6, 15)
    assert std.seam_interaction.dropna().map(type).eq(bool).all()
    assert_matches_oracle(std.drop(columns="source_row"), synth[0].curr_std)


def test_default_out_is_project_root():
    assert _parser().parse_args(["--run", "x", "y"]).out == PROJECT_ROOT / "outputs"


def test_ingest_tally_missing_date_raises(synth):
    a, _ = _paths(synth)
    with pytest.raises(ValueError, match="run_date"):
        ingest_tally(a, MAP_A)
