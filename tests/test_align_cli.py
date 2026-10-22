import json

import pandas as pd
from conftest import MAP_A, MAP_B

from ili import align_welds, ingest


def test_cli_writes_outputs_and_metrics(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    assert ingest.main(["--run", str(data / "tally_prev_vendor_a.csv"), str(MAP_A), "2021-06-15",
                        "--run", str(data / "tally_curr_vendor_b.xlsx"), str(MAP_B), "--out", str(out)]) == 0
    code = align_welds.main([
        "--prev", str(out / "standardized" / "2021_vendorA.csv"),
        "--curr", str(out / "standardized" / "2026_vendorB.csv"),
        "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
        "--truth", str(data / "ground_truth_welds.csv"),
        "--out", str(out),
    ])
    assert code == 0, capsys.readouterr().err
    folder = out / "2021_vendorA_vs_2026_vendorB"
    wm = pd.read_csv(folder / "weld_map.csv", dtype={"prev_gw_id": str, "curr_gw_id": str})
    assert len(wm) > 600 and wm.curr_gw_id.dropna().iloc[0] == "1"  # 5-mile fixture
    assert (folder / "changed_sections.csv").is_file()
    summary = json.loads((folder / "alignment_summary.json").read_text(encoding="utf-8"))
    assert "DRAFT" in summary["label"] and summary["alignment_rate"] >= 0.99
    metrics = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["weld_alignment"]["joint_accuracy"] >= 0.99
    printed = capsys.readouterr().out
    assert "alignment rate" in printed and "joint accuracy" in printed


def test_cli_missing_file_exit_2(tmp_path, capsys):
    code = align_welds.main(["--prev", str(tmp_path / "nope.csv"), "--curr", str(tmp_path / "nope2.csv"),
                             "--out", str(tmp_path)])
    assert code == 2
    assert "nope.csv" in capsys.readouterr().err
