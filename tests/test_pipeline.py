"""The pipeline runner: every stage in order, stopping at a failure, re-applying reviews (M9 done-when)."""

import sys
import threading
from dataclasses import replace

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili import pipeline
from ili.align_welds import DRAFT_LABEL
from ili.match import load_match_table
from ili.pipeline import STAGES, PipelineInputs, load_pipeline_inputs, run_pipeline
from ili.review import FLAGS_FILE, REVIEWS_FILE, append_flag, append_review

FOLDER = "2021_vendorA_vs_2026_vendorB"
OUTPUTS = ["weld_map.csv", "changed_sections.csv", "alignment_summary.json", "match_table.csv",
           "match_summary.json", "growth.csv", "growth_segments.csv", "growth_summary.json", "remaining_life.csv",
           "reinspection_intervals.csv", "remaining_life_summary.json", "classification.csv",
           "classification_summary.json", "report.html", "dig_list.xlsx", "category_changes.csv",
           "run_manifest.json", "pipeline_inputs.json"]


def _inputs(data, out, **over):
    ref = data / "reference"
    base = dict(prev_tally=data / "tally_prev_vendor_a.csv", prev_map=MAP_A, prev_run_date="2021-06-15",
                curr_tally=data / "tally_curr_vendor_b.xlsx", curr_map=MAP_B, pipe=ref / "pipe_properties.csv",
                mop=ref / "mop_profile.csv", segments=ref / "hca_segments.csv",
                next_inspection=ref / "next_inspection.csv", repairs=ref / "repairs_between_runs.csv",
                crossings=ref / "foreign_crossings.csv", digs=ref / "digs.csv", out=out)
    return PipelineInputs(**{**base, **over})


def test_full_pipeline_on_synth(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out", line_name="Line 7")
    results = run_pipeline(inputs)
    assert [r.stage for r in results] == list(STAGES)
    assert [r.exit_code for r in results] == [0] * len(STAGES), [r.stderr for r in results]
    folder = inputs.out / FOLDER
    assert all((folder / n).is_file() for n in OUTPUTS)
    assert DRAFT_LABEL in (folder / "report.html").read_text(encoding="utf-8")
    assert DRAFT_LABEL in results[-1].stdout
    assert load_pipeline_inputs(folder / "pipeline_inputs.json") == inputs


def test_stops_at_first_failure(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out", pipe=data / "reference" / "missing.csv")
    results = run_pipeline(inputs)
    assert [(r.stage, r.exit_code) for r in results] == [("ingest", 0), ("align_welds", 0), ("match", 2)]
    assert results[-1].stderr
    assert not (inputs.out / FOLDER / "growth.csv").exists()


def test_until_stops_after_stage(synth, tmp_path):
    _, data = synth
    results = run_pipeline(_inputs(data, tmp_path / "out"), until="align_welds")
    assert [r.stage for r in results] == ["ingest", "align_welds"]


def test_unexpected_exception_is_a_failed_stage(synth, tmp_path, monkeypatch):
    _, data = synth
    streams = sys.stdout, sys.stderr

    def boom(argv):
        print("partial")
        raise RuntimeError("kaboom")

    monkeypatch.setattr(pipeline.growth, "main", boom)
    results = run_pipeline(_inputs(data, tmp_path / "out"))
    assert [(r.stage, r.exit_code) for r in results][-1] == ("growth", 1)
    assert "RuntimeError: kaboom" in results[-1].stderr and "Traceback" in results[-1].stderr
    assert (sys.stdout, sys.stderr) == streams


def test_cli_prints_stages_and_returns_failing_code(synth, tmp_path, capsys):
    _, data = synth
    ref = data / "reference"
    argv = ["--prev-tally", str(data / "tally_prev_vendor_a.csv"), "--prev-map", str(MAP_A),
            "--prev-run-date", "2021-06-15", "--curr-tally", str(data / "tally_curr_vendor_b.xlsx"),
            "--curr-map", str(MAP_B), "--pipe", str(ref / "nope.csv"), "--mop", str(ref / "mop_profile.csv"),
            "--segments", str(ref / "hca_segments.csv"), "--next-inspection", str(ref / "next_inspection.csv"),
            "--out", str(tmp_path / "out")]
    assert pipeline.main(argv) == 2
    captured = capsys.readouterr()
    assert DRAFT_LABEL in captured.out and "== ingest: exit 0" in captured.out and "== match: exit 2" in captured.out
    assert captured.err


def test_reviews_reapplied_on_rerun(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out")
    assert all(r.exit_code == 0 for r in run_pipeline(inputs))
    folder = inputs.out / FOLDER
    base = load_match_table(folder / "match_table.csv")
    one = base[base.match_type == "matched_1to1"]
    assert len(one) >= 4
    a = one.iloc[0]
    # b and c: two 1:1 pairs of one type in the same weld-map row, so a manual re-pair across them is in range
    twins = one.iloc[1:].groupby(["prev_feature_type", "prev_wm_row"]).filter(lambda g: len(g) >= 3)
    assert len(twins) >= 3
    b, c, d = (twins.iloc[k] for k in range(3))
    append_review(folder / REVIEWS_FILE, "reject", [a.prev_feature_id], [a.curr_feature_id], "Pat", "not the same")
    append_review(folder / REVIEWS_FILE, "manual", [b.prev_feature_id], [c.curr_feature_id], "Pat")
    append_flag(folder / FLAGS_FILE, d.curr_feature_id, "manual_immediate", True, "Pat", "seen in the field")

    assert all(r.exit_code == 0 for r in run_pipeline(inputs))
    table = load_match_table(folder / "match_table.csv")
    prev_row = table[table.prev_feature_id == a.prev_feature_id].iloc[0]
    curr_row = table[table.curr_feature_id == a.curr_feature_id].iloc[0]
    assert prev_row.match_type == "missing" and prev_row.missing_hint == "rejected_by_review"
    assert curr_row.match_type == "new"
    paired = table[table.prev_feature_id == b.prev_feature_id].iloc[0]
    assert paired.curr_feature_id == c.curr_feature_id and paired.match_type == "matched_1to1"
    cls = pd.read_csv(folder / "classification.csv", dtype={"curr_feature_id": str})
    assert bool(cls[cls.curr_feature_id == d.curr_feature_id].iloc[0].manual_immediate)
    html = (folder / "report.html").read_text(encoding="utf-8")
    assert "R000001" in html and "R000002" in html and "F000001" in html and "seen in the field" in html

    second = {n: (folder / n).read_bytes() for n in ("match_table.csv", "classification.csv")}
    assert all(r.exit_code == 0 for r in run_pipeline(inputs))
    assert {n: (folder / n).read_bytes() for n in second} == second


def test_inputs_round_trip_and_errors(tmp_path):
    inputs = PipelineInputs(prev_tally=tmp_path / "a.csv", prev_map=tmp_path / "ma.yaml", curr_tally=tmp_path / "b.xlsx",
                            curr_map=tmp_path / "mb.yaml", pipe=tmp_path / "p.csv", mop=tmp_path / "m.csv",
                            segments=tmp_path / "s.csv", next_inspection=tmp_path / "n.csv", out=tmp_path / "o",
                            discovery_date="2026-10-01", curr_tool="MFL")
    path = pipeline.write_pipeline_inputs(inputs, tmp_path / "folder")
    assert b"\r" not in path.read_bytes() and DRAFT_LABEL in path.read_text(encoding="utf-8")
    assert load_pipeline_inputs(path) == inputs
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_pipeline_inputs(bad)
    with pytest.raises(ValueError):
        run_pipeline(replace(inputs), until="nope")


def test_busy_run_returns_at_once_and_streams_survive(synth, tmp_path, monkeypatch):
    _, data = synth
    streams = sys.stdout, sys.stderr
    started, release = threading.Event(), threading.Event()

    def slow(argv):
        started.set()
        release.wait(30)
        return 0

    monkeypatch.setattr(pipeline.ingest, "main", slow)
    first = []
    worker = threading.Thread(target=lambda: first.extend(run_pipeline(_inputs(data, tmp_path / "a"), until="ingest")))
    worker.start()
    try:
        assert started.wait(30)
        second = run_pipeline(_inputs(data, tmp_path / "b"))
        assert [(r.stage, r.exit_code, r.stdout) for r in second] == [("pipeline", 1, "")]
        assert "another pipeline run is in progress" in second[0].stderr
        assert not (tmp_path / "b").exists()
    finally:
        release.set()
        worker.join(30)
    assert (sys.stdout, sys.stderr) == streams
    assert pipeline._RUN_LOCK.acquire(blocking=False)
    pipeline._RUN_LOCK.release()


def test_busy_cli_exits_1(tmp_path, capsys):
    assert pipeline._RUN_LOCK.acquire(blocking=False)
    try:
        ref = tmp_path / "x"
        argv = []
        names = ("prev-tally", "prev-map", "curr-tally", "curr-map", "pipe", "mop", "segments", "next-inspection", "out")
        for n in names:
            argv += ["--" + n, str(ref)]
        assert pipeline.main(argv) == 1
    finally:
        pipeline._RUN_LOCK.release()
    assert "another pipeline run is in progress" in capsys.readouterr().err


@pytest.mark.parametrize("raised, code, err", [(None, 0, ""), (3, 3, ""), ("bad thing", 1, "bad thing")])
def test_system_exit_mapping(raised, code, err):
    def stage(argv):
        raise SystemExit(raised)

    pipeline._MODULES["growth"], saved = type("M", (), {"main": staticmethod(stage)}), pipeline._MODULES["growth"]
    try:
        result = pipeline._call("growth", [])
    finally:
        pipeline._MODULES["growth"] = saved
    assert result.exit_code == code and err in result.stderr


def test_pipeline_passes_repairs_to_growth(tmp_path):
    ref = tmp_path / "ref"
    with_rep = _inputs(tmp_path, tmp_path / "out", repairs=ref / "repairs_between_runs.csv")
    argv = pipeline._argv("growth", with_rep, "2021_vendorA", "2026_vendorB")
    assert argv[argv.index("--repairs") + 1] == str(ref / "repairs_between_runs.csv")
    assert "--repairs" not in pipeline._argv("growth", _inputs(tmp_path, tmp_path / "out", repairs=None),
                                             "2021_vendorA", "2026_vendorB")
