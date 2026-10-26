import json

import pandas as pd
from conftest import MAP_A, MAP_B

from ili import align_welds, growth, ingest, match, remaining_life


def _prepare(data, out):
    assert ingest.main(["--run", str(data / "tally_prev_vendor_a.csv"), str(MAP_A), "2021-06-15",
                        "--run", str(data / "tally_curr_vendor_b.xlsx"), str(MAP_B), "--out", str(out)]) == 0
    std = out / "standardized"
    assert align_welds.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                             "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
                             "--out", str(out)]) == 0
    folder = out / "2021_vendorA_vs_2026_vendorB"
    ref = data / "reference"
    assert match.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                       "--weld-map", str(folder / "weld_map.csv"), "--pipe", str(ref / "pipe_properties.csv"),
                       "--repairs", str(ref / "repairs_between_runs.csv"), "--out", str(out)]) == 0
    assert growth.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                        "--matches", str(folder / "match_table.csv"), "--pipe", str(ref / "pipe_properties.csv"),
                        "--segments", str(ref / "hca_segments.csv"), "--out", str(out)]) == 0
    return std, folder


def _args(std, folder, data, out, mop="mop_profile.csv"):
    ref = data / "reference"
    return ["--curr", str(std / "2026_vendorB.csv"), "--growth", str(folder / "growth.csv"),
            "--pipe", str(ref / "pipe_properties.csv"), "--mop", str(ref / mop),
            "--segments", str(ref / "hca_segments.csv"),
            "--next-inspection", str(ref / "next_inspection.csv"), "--out", str(out)]


def test_cli_writes_outputs(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    capsys.readouterr()
    code = remaining_life.main(_args(std, folder, data, out))
    assert code == 0, capsys.readouterr().err
    table = pd.read_csv(folder / "remaining_life.csv", dtype={"match_id": str, "curr_feature_id": str,
                                                              "segment_id": str})
    growth_table = growth.load_growth_table(folder / "growth.csv")
    assert len(table) == len(growth_table) > 0
    assert {"rl_yr", "rl_governing", "repair_by_date", "dig_required_by_remaining_life"} <= set(table.columns)
    intervals = pd.read_csv(folder / "reinspection_intervals.csv", dtype={"segment_id": str})
    segments = pd.read_csv(data / "reference" / "hca_segments.csv", dtype={"segment_id": str})
    assert list(intervals.segment_id) == list(segments.segment_id)
    summary = json.loads((folder / "remaining_life_summary.json").read_text(encoding="utf-8"))
    assert "DRAFT" in summary["label"]
    gs = json.loads((folder / "growth_summary.json").read_text(encoding="utf-8"))
    assert summary["prev_run_id"] == gs["prev_run_id"] and summary["curr_run_id"] == gs["curr_run_id"]
    assert summary["n_features"] == len(table)
    printed = capsys.readouterr().out
    assert "dig" in printed and "needs review" in printed and "min RL" in printed


def test_cli_missing_mop_exit_2(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    capsys.readouterr()
    assert remaining_life.main(_args(std, folder, data, out, mop="nope.csv")) == 2
    assert "error:" in capsys.readouterr().err
    assert not (folder / "remaining_life.csv").exists()
