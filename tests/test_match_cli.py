import json

import pandas as pd
from conftest import MAP_A, MAP_B

from ili import align_welds, ingest, match


def _prepare(data, out):
    assert ingest.main(["--run", str(data / "tally_prev_vendor_a.csv"), str(MAP_A), "2021-06-15",
                        "--run", str(data / "tally_curr_vendor_b.xlsx"), str(MAP_B), "--out", str(out)]) == 0
    std = out / "standardized"
    assert align_welds.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                             "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
                             "--out", str(out)]) == 0
    return std, out / "2021_vendorA_vs_2026_vendorB"


def test_cli_writes_outputs_and_metrics(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    capsys.readouterr()
    code = match.main([
        "--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
        "--weld-map", str(folder / "weld_map.csv"),
        "--pipe", str(data / "reference" / "pipe_properties.csv"),
        "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
        "--truth", str(data / "ground_truth.csv"),
        "--out", str(out),
    ])
    assert code == 0, capsys.readouterr().err
    table = pd.read_csv(folder / "match_table.csv", dtype={"prev_feature_id": str, "curr_feature_id": str})
    assert len(table) > 0 and {"match_type", "confidence", "needs_review"} <= set(table.columns)
    summary = json.loads((folder / "match_summary.json").read_text(encoding="utf-8"))
    assert "DRAFT" in summary["label"] and "n_needs_review" in summary
    metrics = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["anomaly_matching"]["precision"] >= 0.95
    printed = capsys.readouterr().out
    assert "precision" in printed and "needs review" in printed


def test_cli_missing_weld_map_exit_2(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, _ = _prepare(data, out)
    code = match.main([
        "--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
        "--weld-map", str(tmp_path / "nope_weld_map.csv"),
        "--pipe", str(data / "reference" / "pipe_properties.csv"),
        "--out", str(out),
    ])
    assert code == 2
    assert "nope_weld_map.csv" in capsys.readouterr().err
