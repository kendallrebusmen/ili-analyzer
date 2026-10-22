from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili.align_welds import align_welds, build_joints
from ili.config import load_settings
from ili.ingest import ingest_tally
from ili.synth import SynthParams, generate, write_outputs
from ili.validation import score_alignment, write_metrics

SETTINGS = load_settings()


@pytest.fixture(scope="module")
def default_line(tmp_path_factory):
    """Default 20-mile dataset, through the real file path: synth -> files -> ingest."""
    out = tmp_path_factory.mktemp("default")
    write_outputs(generate(SynthParams()), out)
    prev = ingest_tally(out / "tally_prev_vendor_a.csv", MAP_A, date(2021, 6, 15)).standard
    curr = ingest_tally(out / "tally_curr_vendor_b.xlsx", MAP_B).standard
    repairs = pd.read_csv(out / "reference" / "repairs_between_runs.csv")
    truth = pd.read_csv(out / "ground_truth_welds.csv", dtype={"prev_gw_id": str, "curr_gw_id": str})
    return prev, curr, repairs, truth


def test_default_dataset_meets_acceptance(default_line):
    prev, curr, repairs, truth = default_line
    result = align_welds(prev, curr, SETTINGS, repairs)
    score = score_alignment(result.weld_map, prev, curr, truth)
    assert score["joint_accuracy"] >= 0.99, score["wrong_groups"]
    assert score["replaced_section_flagged"]
    assert score["missed_welds_handled"] and score["n_missed_weld_groups"] == 6
    assert result.summary["alignment_rate"] >= 0.99
    assert result.changed_sections.overlaps_repair.sum() == 1


@pytest.mark.parametrize("seed", range(1, 6))
def test_acceptance_across_seeds(seed):
    r = generate(SynthParams(seed=seed, line_miles=5))
    result = align_welds(r.prev_std, r.curr_std, SETTINGS, r.reference["repairs_between_runs"])
    score = score_alignment(result.weld_map, r.prev_std, r.curr_std, r.weld_truth)
    assert score["joint_accuracy"] >= 0.99, score["wrong_groups"]
    assert score["replaced_section_flagged"] and score["missed_welds_handled"]


def test_no_repairs_file_still_aligns():
    """Without repair records the replacement is still a changed section, but its edge can
    shift by a joint when an old joint's length coincides with a neighbour's (seed 2 does)."""
    r = generate(SynthParams(seed=2, line_miles=5))
    result = align_welds(r.prev_std, r.curr_std, SETTINGS, None)
    score = score_alignment(result.weld_map, r.prev_std, r.curr_std, r.weld_truth)
    assert score["joint_accuracy"] >= 0.99
    assert not result.changed_sections.overlaps_repair.any()
    rep = r.reference["repairs_between_runs"].iloc[0]
    cs = result.changed_sections
    gaps = cs[cs.match_types.str.contains("inserted") & cs.match_types.str.contains("removed")]
    assert ((gaps.curr_start_ft <= rep.start_ft + 1) & (gaps.curr_end_ft >= rep.end_ft - 1)).any()


def test_weld_map_covers_every_joint_once(default_line):
    prev, curr, repairs, _ = default_line
    wm = align_welds(prev, curr, SETTINGS, repairs).weld_map
    for run, std in (("prev", prev), ("curr", curr)):
        used = pd.concat([wm[f"{run}_gw_id"], wm[f"{run}_gw_id_2"]]).dropna()
        assert used.is_unique
        assert set(used) == set(build_joints(std).gw_id)


def test_weld_map_columns_and_confidence(default_line):
    prev, curr, repairs, _ = default_line
    result = align_welds(prev, curr, SETTINGS, repairs)
    wm = result.weld_map
    assert list(wm.columns) == [
        "section_id", "match_type", "prev_gw_id", "prev_gw_id_2", "curr_gw_id", "curr_gw_id_2",
        "prev_start_ft", "curr_start_ft", "prev_length_ft", "curr_length_ft", "curr_length_scaled_ft",
        "mismatch", "cost", "confidence", "in_repair", "is_anchor",
    ]
    assert set(wm.confidence) <= {"high", "medium", "low"}
    assert (wm.loc[wm.match_type.isin(["2:1", "1:2"]), "confidence"] != "high").all()
    gaps = wm[wm.match_type.isin(["inserted", "removed"])]
    assert (gaps.confidence == gaps.in_repair.map({True: "high", False: "low"})).all()
    assert wm.is_anchor.sum() == 10
    assert result.summary["counts_by_match_type"]["2:1"] == 3
    assert result.summary["counts_by_match_type"]["1:2"] == 3


def test_score_detects_wrong_map(default_line):
    prev, curr, repairs, truth = default_line
    wm = align_welds(prev, curr, SETTINGS, repairs).weld_map
    good = score_alignment(wm, prev, curr, truth)["joint_accuracy"]
    bad = wm.copy()
    rows = bad.index[bad.match_type == "1:1"][100:102]
    bad.loc[rows, "curr_gw_id"] = bad.loc[rows[::-1], "curr_gw_id"].to_numpy()  # swap two partners
    score = score_alignment(bad, prev, curr, truth)
    assert score["joint_accuracy"] < good
    assert len(score["wrong_groups"]) == 2


def test_write_metrics_merges(tmp_path):
    path = tmp_path / "validation_metrics.json"
    write_metrics("a", {"x": 1}, path)
    write_metrics("b", {"y": 2}, path)
    write_metrics("a", {"x": 3}, path)
    import json

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"x": 3}, "b": {"y": 2}}
