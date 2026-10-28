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


def test_growth_cli_repairs_optional(synth, tmp_path, capsys):
    _, data = synth
    segs = str(data / "reference" / "hca_segments.csv")
    for name, extra in (("without", []), ("with", ["--repairs", str(data / "reference" / "repairs_between_runs.csv")])):
        out = tmp_path / name
        std, folder = _prepare(data, out)
        assert growth.main(_args(std, folder, data, out, "--segments", segs, *extra)) == 0, capsys.readouterr().err
        summary = json.loads((folder / "growth_summary.json").read_text(encoding="utf-8"))
        assert "n_excluded_recoated" in summary
        table = growth.load_growth_table(folder / "growth.csv")
        assert table.in_recoat.dtype == bool or table.in_recoat.notna().all()


def test_growth_cli_rejects_stale_match_table(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    path = folder / "match_table.csv"
    pd.read_csv(path, dtype=str, keep_default_na=False).drop(columns=["curr_surface", "curr_ml_origin"]).to_csv(
        path, index=False)
    capsys.readouterr()
    assert growth.main(_args(std, folder, data, out, "--segments", str(data / "reference" / "hca_segments.csv"))) == 2
    err = capsys.readouterr().err
    assert "curr_surface" in err and "curr_ml_origin" in err and "ili.match" in err


def test_growth_cli_writes_summary_csv(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    args = _args(std, folder, data, out, "--segments", str(data / "reference" / "hca_segments.csv"),
                 "--repairs", str(data / "reference" / "repairs_between_runs.csv"))
    assert growth.main(args) == 0, capsys.readouterr().err
    table = pd.read_csv(folder / "growth_summary.csv")
    assert list(table.columns) == growth.SURFACE_SUMMARY_COLUMNS
    assert table[["surface", "scope"]].values.tolist() == [["EXT", "all"], ["EXT", "excluding_recoated"],
                                                           ["INT", "all"], ["INT", "excluding_recoated"]]
    assert (table.n_features[table.scope == "all"].to_numpy() >= table.n_features[table.scope != "all"].to_numpy()).all()
