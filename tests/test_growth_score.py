import json
import math
from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili import growth, reference
from ili.align_welds import align_welds
from ili.config import load_settings
from ili.ingest import ingest_tally
from ili.match import delta_t_years, match_anomalies
from ili.synth import SynthParams, generate, write_outputs
from ili.validation import MIN_GROWTH_CONSERVATISM, score_growth

SETTINGS = load_settings()
ID_DTYPES = {"prev_feature_id": str, "curr_feature_id": str}
NAN = float("nan")
# Test bound, not a tuning value: raw growth is unbiased by construction (same tool both runs), so a units
# or scaling bug in raw_mpy shows up as a bias well beyond this.
RAW_BIAS_BOUND_MPY = 1.0


def _growth(prev, curr, pipe, hca_segments, repairs):
    """align -> match -> growth for two standardized runs."""
    wm = align_welds(prev, curr, SETTINGS, repairs).weld_map
    table = match_anomalies(prev, curr, wm, SETTINGS, pipe, repairs).match_table
    welds = curr.log_dist_ft[curr.feature_type == "girth_weld"]
    seg = reference.segment_table(SETTINGS, hca_segments, float(welds.min()), float(welds.max()))
    return growth.compute_growth(table, delta_t_years(prev, curr), prev.run_date.iloc[0].year,
                                 curr.run_date.iloc[0].year, pipe, seg, SETTINGS).features


@pytest.fixture(scope="module")
def default_line(tmp_path_factory):
    """Default 20-mile dataset through the real file path: synth -> files -> ingest -> align -> match -> growth."""
    out = tmp_path_factory.mktemp("default_growth")
    write_outputs(generate(SynthParams()), out)
    prev = ingest_tally(out / "tally_prev_vendor_a.csv", MAP_A, date(2021, 6, 15)).standard
    curr = ingest_tally(out / "tally_curr_vendor_b.xlsx", MAP_B).standard
    ref = out / "reference"
    features = _growth(prev, curr, pd.read_csv(ref / "pipe_properties.csv"), pd.read_csv(ref / "hca_segments.csv"),
                       pd.read_csv(ref / "repairs_between_runs.csv"))
    truth = pd.read_csv(out / "ground_truth.csv", dtype=ID_DTYPES)
    return features, truth


def _assert_reports_errors(score: dict, features: pd.DataFrame):
    for part in ("raw", "governing"):
        m = score[part]
        assert m["n"] > 0
        assert math.isfinite(m["bias_mpy"]) and math.isfinite(m["rmse_mpy"])
        assert m["rmse_mpy"] >= abs(m["bias_mpy"])
    assert score["raw"]["n"] == int((features.history_class == "matched").sum())
    assert score["governing"]["n"] + score["n_no_governing"] == len(features)
    assert (score["n_truth_not_scored"], score["n_duplicated"]) == (0, 0), score
    json.dumps(score)  # write_metrics dumps it


def _assert_acceptance(score: dict, features: pd.DataFrame):
    _assert_reports_errors(score, features)
    assert abs(score["raw"]["bias_mpy"]) <= RAW_BIAS_BOUND_MPY, score["raw"]
    assert math.isfinite(score["governing"]["rmse_mpy"])
    assert score["conservatism_rate"] >= MIN_GROWTH_CONSERVATISM, score
    assert score["passes"] is True


def test_default_dataset_growth_acceptance(default_line):
    features, truth = default_line
    _assert_acceptance(score_growth(features, truth), features)


@pytest.mark.parametrize("seed", range(1, 6))
def test_growth_acceptance_across_seeds(seed):
    r = generate(SynthParams(seed=seed, line_miles=5))
    ref = r.reference
    features = _growth(r.prev_std, r.curr_std, ref["pipe_properties"], ref["hca_segments"],
                       ref["repairs_between_runs"])
    _assert_acceptance(score_growth(features, r.ground_truth), features)


def test_target_is_spec_value():
    assert MIN_GROWTH_CONSERVATISM == 0.90


def _features(rows):
    return pd.DataFrame(rows, columns=["match_type", "curr_feature_id", "history_class", "raw_mpy", "governing_mpy"])


def _truth(rows):
    return pd.DataFrame(rows, columns=["feature_type", "curr_feature_id", "curr_status", "truth_class", "growth_mpy"])


# A vendor cluster report B-2 holds two true pits; B-3 is a single pit in the same matched cluster row.
# The dent B-9, the pit missing from the current run and the unreported pit need no growth row.
TRUTH = _truth([
    ("metal_loss", "B-1", "reported", "matched_1to1", 4.0),
    ("metal_loss", "B-2", "reported_in_cluster", "matched_cluster", 2.0),
    ("metal_loss", "B-2", "reported_in_cluster", "matched_cluster", 6.0),
    ("metal_loss", "B-3", "reported", "matched_cluster", 1.0),
    ("metal_loss", "B-4", "reported", "new", 1.0),
    ("dent", "B-9", "reported", "matched_1to1", NAN),
    ("metal_loss", None, "below_threshold", "missing", 0.5),
    ("metal_loss", None, "below_threshold", "not_reported", 0.2),
])
HAND = _features([
    ("matched_1to1", "B-1", "matched", 3.0, 5.0),        # true 4: raw -1, gov +1, conservative
    ("matched_cluster", "B-2;B-3", "matched", 8.0, 5.5),  # true max(2, 6, 1) = 6: raw +2, gov -0.5, not
    ("new", "B-4", "new", NAN, 3.0),                      # true 1: gov +2, conservative
])


def test_score_growth_hand_built():
    s = score_growth(HAND, TRUTH)
    assert s["raw"] == {"bias_mpy": pytest.approx(0.5), "rmse_mpy": pytest.approx(math.sqrt(2.5)), "n": 2}
    assert s["governing"] == {"bias_mpy": pytest.approx(2.5 / 3), "rmse_mpy": pytest.approx(math.sqrt(1.75)),
                              "n": 3}
    assert s["conservatism_rate"] == pytest.approx(2 / 3)
    assert s["conservatism_by_class"] == {
        "matched": {"rate": pytest.approx(0.5), "n": 2},
        "new": {"rate": 1.0, "n": 1},
        "no_history": {"rate": None, "n": 0},
    }
    assert s["n_features"] == 3 and s["n_no_governing"] == 0
    assert (s["n_truth_not_scored"], s["n_duplicated"]) == (0, 0)
    assert s["passes"] is False
    assert s["non_conservative"] == [{"curr_feature_id": "B-2;B-3", "history_class": "matched",
                                      "governing_mpy": 5.5, "true_mpy": 6.0}]
    json.dumps(s)


def test_cluster_true_growth_is_max_of_its_pits():
    # Taking the first, mean or min pit instead of the max would make the cluster row conservative.
    s = score_growth(HAND.iloc[[1]], TRUTH)
    assert s["governing"]["bias_mpy"] == pytest.approx(-0.5)
    assert s["conservatism_rate"] == 0.0
    # The max over the ids holds whichever id carries the fastest pit.
    flipped = HAND.iloc[[1]].assign(curr_feature_id="B-3;B-2")
    assert score_growth(flipped, TRUTH)["governing"]["bias_mpy"] == pytest.approx(-0.5)


def test_no_governing_rate_counts_as_not_conservative():
    f = _features([
        ("matched_1to1", "B-1", "matched", 4.0, 4.0),  # equal to the truth is conservative
        ("new", "B-4", "no_history", NAN, NAN),
    ])
    s = score_growth(f, TRUTH)
    assert s["n_no_governing"] == 1
    assert s["governing"] == {"bias_mpy": 0.0, "rmse_mpy": 0.0, "n": 1}
    assert s["conservatism_rate"] == 0.5
    assert s["conservatism_by_class"]["no_history"] == {"rate": 0.0, "n": 1}
    assert s["conservatism_by_class"]["matched"] == {"rate": 1.0, "n": 1}


def test_passes_at_the_target():
    truth = _truth([("metal_loss", f"B-{k}", "reported", "matched_1to1", 4.0) for k in range(10)])
    rows = [("matched_1to1", f"B-{k}", "matched", 4.0, 4.0 if k < 9 else 3.0) for k in range(10)]
    s = score_growth(_features(rows), truth)
    assert s["conservatism_rate"] == pytest.approx(0.9) and s["passes"] is True
    rows[0] = ("matched_1to1", "B-0", "matched", 4.0, 3.9)
    assert score_growth(_features(rows), truth)["passes"] is False


# HAND with the cluster row's governing rate at its true 6 mpy: every row conservative, so it passes.
GOOD = HAND.assign(governing_mpy=[5.0, 6.0, 3.0])


def test_dropped_row_fails():
    assert score_growth(GOOD, TRUTH)["passes"] is True
    s = score_growth(GOOD.drop(index=1), TRUTH)  # the hard cluster row goes; the rest stay conservative
    assert s["conservatism_rate"] == 1.0
    assert (s["n_truth_not_scored"], s["truth_not_scored"], s["n_duplicated"]) == (2, ["B-2", "B-3"], 0)
    assert s["passes"] is False


def test_duplicated_row_fails():
    s = score_growth(pd.concat([GOOD, GOOD.iloc[[0]]], ignore_index=True), TRUTH)
    assert s["conservatism_rate"] == 1.0
    assert (s["n_truth_not_scored"], s["n_duplicated"], s["duplicated"]) == (0, 1, ["B-1"])
    assert s["passes"] is False
    twice = GOOD.assign(curr_feature_id=["B-1", "B-2;B-3;B-2", "B-4"])  # repeated inside one cell
    s = score_growth(twice, TRUTH)
    assert (s["n_truth_not_scored"], s["n_duplicated"], s["duplicated"]) == (0, 1, ["B-2"])
    assert s["passes"] is False


def test_score_growth_unknown_id_is_value_error():
    bad = HAND.copy()
    bad.loc[1, "curr_feature_id"] = "B-2;Z-999999"
    with pytest.raises(ValueError, match="Z-999999"):
        score_growth(bad, TRUTH)
    for col in ("growth_mpy", "curr_status", "truth_class", "feature_type"):
        with pytest.raises(ValueError, match=col):
            score_growth(HAND, TRUTH.drop(columns=col))
    with pytest.raises(ValueError, match="governing_mpy"):
        score_growth(HAND.drop(columns="governing_mpy"), TRUTH)


def test_unknown_history_class_is_value_error():
    # With every class unknown no row is raw-scored, yet conservatism alone would still pass.
    with pytest.raises(ValueError, match="bogus"):
        score_growth(GOOD.assign(history_class="bogus"), TRUTH)
    with pytest.raises(ValueError, match="history_class"):
        score_growth(GOOD.assign(history_class=["matched", "matched", None]), TRUTH)


def test_scaled_raw_rates_fail_the_bias_bound(default_line):
    # A mils/inches slip (x10) still passes conservatism; the raw bias bound in the acceptance tests catches it.
    features, truth = default_line
    s = score_growth(features.assign(raw_mpy=features.raw_mpy * 10, governing_mpy=features.governing_mpy * 10),
                     truth)
    assert s["passes"] is True
    assert abs(s["raw"]["bias_mpy"]) > RAW_BIAS_BOUND_MPY


def test_ids_compare_as_strings():
    truth = _truth([("metal_loss", "007", "reported", "new", 2.0)])
    f = _features([("new", "007", "new", NAN, 2.5)])
    assert score_growth(f, truth)["conservatism_rate"] == 1.0
    with pytest.raises(ValueError, match="'7'"):
        score_growth(_features([("new", "7", "new", NAN, 2.5)]), truth)
