"""Seed sweeps and edge parameters that the single-seed tests cannot reach."""

import numpy as np
import pytest

from ili.synth import SynthParams, generate, main

# Seeds that exposed each defect in the M1 review (2-mile line).
CROSS_RUN_ADJACENT_SEEDS = [1, 20, 68, 85, 100]
WALL_CHANGE_SEEDS = [41, 56, 69, 89, 119]


@pytest.mark.parametrize("seed", CROSS_RUN_ADJACENT_SEEDS + list(range(2, 12)))
def test_missed_welds_not_adjacent_across_runs(seed):
    wt = generate(SynthParams(seed=seed, line_miles=2)).weld_truth
    prev = set(wt.loc[wt.prev_status == "missed", "true_weld_idx"])
    curr = set(wt.loc[wt.curr_status == "missed", "true_weld_idx"])
    assert all(abs(a - b) > 1 for a in prev for b in curr)


@pytest.mark.parametrize("seed", WALL_CHANGE_SEEDS)
def test_tally_wall_matches_true_wall(seed):
    r = generate(SynthParams(seed=seed, line_miles=2))
    gt = r.ground_truth
    for which, std in (("prev", r.prev_std), ("curr", r.curr_std)):
        m = gt[gt[f"{which}_status"] == "reported"].merge(
            std[["feature_id", "wt_nom_in"]], left_on=f"{which}_feature_id", right_on="feature_id",
            suffixes=("_true", "_tally"),
        )
        assert np.allclose(m.wt_nom_in_true, m.wt_nom_in_tally), which


@pytest.mark.parametrize("seed", range(1, 21))
def test_one_truth_class_per_vendor_feature(seed):
    gt = generate(SynthParams(seed=seed, line_miles=2)).ground_truth
    for col in ("prev_feature_id", "curr_feature_id"):
        classes = gt.dropna(subset=[col]).groupby(col).truth_class.nunique()
        assert (classes == 1).all(), classes[classes > 1].index.tolist()


def test_cluster_with_below_threshold_pit_is_matched_cluster():
    """Seed 42: B-002243 is a current-run cluster whose third pit was below threshold in prev."""
    gt = generate(SynthParams()).ground_truth
    members = gt[gt.curr_feature_id == "B-002243"]
    assert len(members) == 3
    assert set(members.truth_class) == {"matched_cluster"}
    assert (members.prev_status == "below_threshold").sum() == 1


@pytest.mark.parametrize(
    "field", ["n_missed_welds", "n_new", "n_severe", "n_internal", "n_dents", "n_cluster_sites", "n_ext_ml_prev"]
)
def test_zero_counts_allowed(field):
    r = generate(SynthParams(line_miles=2, **{field: 0}))
    pop = {"n_new": "new", "n_severe": "severe", "n_internal": "internal", "n_dents": "dent",
           "n_cluster_sites": "cluster_pit", "n_ext_ml_prev": "ext_ml"}
    if field in pop:
        assert (r.ground_truth.population == pop[field]).sum() == 0
    if field == "n_missed_welds":
        assert not (r.weld_truth.prev_status == "missed").any()


@pytest.mark.parametrize("field", ["n_new", "n_missed_welds", "n_ext_ml_prev", "n_valves"])
def test_negative_counts_rejected(field):
    with pytest.raises(ValueError, match=field):
        generate(SynthParams(line_miles=2, **{field: -1}))


def test_cli_reports_bad_count_cleanly(tmp_path, capsys):
    assert main(["--out", str(tmp_path), "--line-miles", "2", "--n-new", "-1"]) == 2
    assert "n_new" in capsys.readouterr().err
