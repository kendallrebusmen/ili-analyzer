import json

import pandas as pd
import yaml
from conftest import MAP_A, MAP_B

from ili import align_welds, growth, ingest, match
from ili.config import DEFAULT_SETTINGS


def _prepare(data, out):
    assert ingest.main(["--run", str(data / "tally_prev_vendor_a.csv"), str(MAP_A), "2021-06-15",
                        "--run", str(data / "tally_curr_vendor_b.xlsx"), str(MAP_B), "--out", str(out)]) == 0
    std = out / "standardized"
    assert align_welds.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                             "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
                             "--out", str(out)]) == 0
    folder = out / "2021_vendorA_vs_2026_vendorB"
    assert match.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                       "--weld-map", str(folder / "weld_map.csv"),
                       "--pipe", str(data / "reference" / "pipe_properties.csv"),
                       "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
                       "--out", str(out)]) == 0
    return std, folder


def _args(std, folder, data, out, *extra):
    return ["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
            "--matches", str(folder / "match_table.csv"),
            "--pipe", str(data / "reference" / "pipe_properties.csv"), "--out", str(out), *extra]


def test_cli_writes_outputs_and_metrics(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    capsys.readouterr()
    code = growth.main(_args(std, folder, data, out, "--segments", str(data / "reference" / "hca_segments.csv"),
                             "--truth", str(data / "ground_truth.csv")))
    assert code == 0, capsys.readouterr().err
    table = pd.read_csv(folder / "growth.csv", dtype={"prev_feature_id": str, "curr_feature_id": str})
    assert len(table) > 0 and {"governing_mpy", "governing_method", "needs_review"} <= set(table.columns)
    assert len(pd.read_csv(folder / "growth_segments.csv")) > 0
    summary = json.loads((folder / "growth_summary.json").read_text(encoding="utf-8"))
    assert "DRAFT" in summary["label"] and "n_needs_review" in summary
    metrics = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["growth"]["conservatism_rate"] >= 0.90
    printed = capsys.readouterr().out
    assert "conservatism" in printed and "needs review" in printed


def test_cli_segments_required_unless_fixed_length(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    capsys.readouterr()
    assert growth.main(_args(std, folder, data, out)) == 2
    assert "segments" in capsys.readouterr().err
    settings = yaml.safe_load(DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    settings["growth"]["segment_definition"] = "fixed_length"
    path = tmp_path / "fixed.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    assert growth.main(_args(std, folder, data, out, "--settings", str(path))) == 0
    assert (folder / "growth_segments.csv").exists()
