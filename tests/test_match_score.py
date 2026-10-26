from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili.align_welds import align_welds
from ili.config import load_settings
from ili.ingest import ingest_tally
from ili.match import ANOMALY_TYPES, match_anomalies
from ili.synth import SynthParams, generate, write_outputs
from ili.validation import MIN_MATCH_PRECISION, MIN_MATCH_RECALL, score_matching

SETTINGS = load_settings()
ID_DTYPES = {"prev_feature_id": str, "curr_feature_id": str}


@pytest.fixture(scope="module")
def default_line(tmp_path_factory):
    """Default 20-mile dataset through the real file path: synth -> files -> ingest -> align -> match."""
    out = tmp_path_factory.mktemp("default_match")
    write_outputs(generate(SynthParams()), out)
    prev = ingest_tally(out / "tally_prev_vendor_a.csv", MAP_A, date(2021, 6, 15)).standard
    curr = ingest_tally(out / "tally_curr_vendor_b.xlsx", MAP_B).standard
    repairs = pd.read_csv(out / "reference" / "repairs_between_runs.csv")
    pipe = pd.read_csv(out / "reference" / "pipe_properties.csv")
    truth = pd.read_csv(out / "ground_truth.csv", dtype=ID_DTYPES)
    wm = align_welds(prev, curr, SETTINGS, repairs).weld_map
    table = match_anomalies(prev, curr, wm, SETTINGS, pipe, repairs).match_table
    return table, truth


@pytest.fixture(scope="module")
def small_run():
    r = generate(SynthParams(line_miles=5))
    repairs, pipe = r.reference["repairs_between_runs"], r.reference["pipe_properties"]
    wm = align_welds(r.prev_std, r.curr_std, SETTINGS, repairs).weld_map
    return match_anomalies(r.prev_std, r.curr_std, wm, SETTINGS, pipe, repairs).match_table, r.ground_truth


TABLE_CHECKS = ("n_unreported", "n_duplicated", "n_wrong_side", "n_duplicate_links")


def _assert_cluster_features_matched_as_clusters(score: dict, truth: pd.DataFrame) -> int:
    """Every truth matched_cluster anomaly feature id, in both runs (vendor cluster reports and the
    individually reported pits opposite them), sits in a matched_cluster row. Returns how many."""
    n = 0
    for run in ("prev", "curr"):
        ids = truth.loc[(truth.truth_class == "matched_cluster") & truth.feature_type.isin(ANOMALY_TYPES),
                        f"{run}_feature_id"].dropna().unique()
        assert score["confusion"][run].get("matched_cluster", {}) == {"matched_cluster": len(ids)}
        n += len(ids)
    return n


def test_default_dataset_meets_acceptance(default_line):
    table, truth = default_line
    score = score_matching(table, truth)
    assert score["precision"] >= 0.98, score["false_links"]
    assert score["recall"] >= 0.95, score["missed_links"]
    assert score["passes"] is True
    assert {k: score[k] for k in TABLE_CHECKS} == dict.fromkeys(TABLE_CHECKS, 0)
    assert _assert_cluster_features_matched_as_clusters(score, truth) > 0


@pytest.mark.parametrize("seed", range(1, 6))
def test_acceptance_across_seeds(seed):
    r = generate(SynthParams(seed=seed, line_miles=5))
    repairs, pipe = r.reference["repairs_between_runs"], r.reference["pipe_properties"]
    wm = align_welds(r.prev_std, r.curr_std, SETTINGS, repairs).weld_map
    table = match_anomalies(r.prev_std, r.curr_std, wm, SETTINGS, pipe, repairs).match_table
    score = score_matching(table, r.ground_truth)
    assert score["precision"] >= 0.98, score["false_links"]
    assert score["recall"] >= 0.95, score["missed_links"]
    assert score["passes"] is True
    assert {k: score[k] for k in TABLE_CHECKS} == dict.fromkeys(TABLE_CHECKS, 0)
    assert _assert_cluster_features_matched_as_clusters(score, r.ground_truth) > 0


def test_targets_are_spec_values():
    assert (MIN_MATCH_PRECISION, MIN_MATCH_RECALL) == (0.98, 0.95)


def test_score_detects_wrong_link(small_run):
    table, truth = small_run
    base = score_matching(table, truth)
    true_links = {(p, c) for p, c in zip(truth.prev_feature_id, truth.curr_feature_id)
                  if pd.notna(p) and pd.notna(c)}
    is_true = pd.Series([(p, c) in true_links for p, c in zip(table.prev_feature_id, table.curr_feature_id)],
                        index=table.index)
    ok = table[(table.match_type == "matched_1to1") & is_true]
    i, j = ok.index[:2]
    swapped = table.copy()
    swapped.loc[i, "curr_feature_id"], swapped.loc[j, "curr_feature_id"] = table.at[j, "curr_feature_id"], \
        table.at[i, "curr_feature_id"]
    s = score_matching(swapped, truth)

    n_pred, n_true, n_ok = base["n_pred_links"], base["n_true_links"], base["n_correct_links"]
    assert (s["n_pred_links"], s["n_true_links"], s["n_correct_links"]) == (n_pred, n_true, n_ok - 2)
    assert s["precision"] == pytest.approx((n_ok - 2) / n_pred)
    assert s["recall"] == pytest.approx((n_ok - 2) / n_true)
    assert base["precision"] - s["precision"] == pytest.approx(2 / n_pred)
    assert base["recall"] - s["recall"] == pytest.approx(2 / n_true)
    pi, pj, ci, cj = table.at[i, "prev_feature_id"], table.at[j, "prev_feature_id"], \
        table.at[i, "curr_feature_id"], table.at[j, "curr_feature_id"]
    false = {(d["prev_feature_id"], d["curr_feature_id"]) for d in s["false_links"]}
    missed = {(d["prev_feature_id"], d["curr_feature_id"]) for d in s["missed_links"]}
    assert {(pi, cj), (pj, ci)} <= false and {(pi, ci), (pj, cj)} <= missed


def test_score_unknown_id_is_value_error(small_run):
    table, truth = small_run
    bad = table.copy()
    k = bad.index[bad.curr_feature_id.notna()][0]
    bad.loc[k, "curr_feature_id"] = "Z-999999"
    with pytest.raises(ValueError, match="Z-999999"):
        score_matching(bad, truth)
    with pytest.raises(ValueError, match="truth_class"):
        score_matching(table, truth.drop(columns="truth_class"))


# ---- hand-built case -------------------------------------------------------------------------

def _truth(rows):
    """Rows: (feature_type, prev_feature_id, curr_feature_id, truth_class); an id present is reported."""
    df = pd.DataFrame(rows, columns=["feature_type", "prev_feature_id", "curr_feature_id", "truth_class"])
    for run in ("prev", "curr"):
        df[f"{run}_status"] = df[f"{run}_feature_id"].notna().map({True: "reported", False: "below_threshold"})
    return df


def _table(rows):
    """Rows: (match_type, prev_feature_id, curr_feature_id)."""
    return pd.DataFrame(rows, columns=["match_type", "prev_feature_id", "curr_feature_id"])


HAND_TRUTH = _truth([
    ("metal_loss", "A1", "B1", "matched_1to1"),
    ("metal_loss", "A2", "B2", "matched_1to1"),
    ("metal_loss", "A3", "B3", "matched_cluster"),   # vendor B reported the A3+A4 pits as one feature
    ("metal_loss", "A3", "B3", "matched_cluster"),   # a second pit inside both reports: duplicate link
    ("metal_loss", "A4", "B3", "matched_cluster"),
    ("metal_loss", None, "B3", "matched_cluster"),   # pit below threshold in prev: no link
    ("metal_loss", "A5", None, "missing"),
    ("metal_loss", None, "B6", "new"),
    ("metal_loss", None, None, "not_reported"),
    ("valve", "AV", "BV", "matched_1to1"),           # not an anomaly: no link, not in the table
])


def test_hand_built_score():
    table = _table([
        ("matched_1to1", "A1", "B1"),
        ("matched_1to1", "A2", "B6"),        # wrong: A2's true partner is B2
        ("matched_cluster", "A3;A4", "B3"),
        ("missing", "A5", None),
        ("new", None, "B2"),
    ])
    s = score_matching(table, HAND_TRUTH)
    assert (s["n_pred_links"], s["n_true_links"], s["n_correct_links"]) == (4, 4, 3)
    assert s["precision"] == pytest.approx(0.75) and s["recall"] == pytest.approx(0.75)
    assert s["passes"] is False
    assert s["false_links"] == [{"prev_feature_id": "A2", "curr_feature_id": "B6"}]
    assert s["missed_links"] == [{"prev_feature_id": "A2", "curr_feature_id": "B2"}]
    assert s["confusion"] == {
        "prev": {"matched_1to1": {"matched_1to1": 2}, "matched_cluster": {"matched_cluster": 2},
                 "missing": {"missing": 1}},
        "curr": {"matched_1to1": {"matched_1to1": 1, "new": 1}, "matched_cluster": {"matched_cluster": 1},
                 "new": {"matched_1to1": 1}},
    }


def test_hand_built_perfect_score_passes():
    table = _table([
        ("matched_1to1", "A1", "B1"), ("matched_1to1", "A2", "B2"),
        ("matched_cluster", "A3;A4", "B3"), ("missing", "A5", None), ("new", None, "B6"),
    ])
    s = score_matching(table, HAND_TRUTH)
    assert (s["precision"], s["recall"], s["passes"]) == (1.0, 1.0, True)
    assert s["false_links"] == [] and s["missed_links"] == []


def test_listed_links_are_capped():
    n = 30
    truth = _truth([("metal_loss", f"A{k}", f"B{k}", "matched_1to1") for k in range(n)])
    table = _table([("matched_1to1", f"A{k}", f"B{(k + 1) % n}") for k in range(n)])
    s = score_matching(table, truth)
    assert (s["n_correct_links"], len(s["false_links"]), len(s["missed_links"])) == (0, 20, 20)
    assert (s["precision"], s["recall"]) == (0.0, 0.0)


def test_conflicting_truth_class_is_value_error():
    truth = _truth([("metal_loss", "A1", "B1", "matched_1to1"), ("metal_loss", "A1", None, "missing")])
    with pytest.raises(ValueError, match="A1"):
        score_matching(_table([("matched_1to1", "A1", "B1")]), truth)


def test_unknown_prev_id_and_missing_columns_are_value_errors():
    with pytest.raises(ValueError, match="A9"):
        score_matching(_table([("missing", "A9", None)]), HAND_TRUTH)
    with pytest.raises(ValueError, match="curr_feature_id"):
        score_matching(_table([("missing", "A5", None)]), HAND_TRUTH.drop(columns="curr_feature_id"))
    with pytest.raises(ValueError, match="match_type"):
        score_matching(_table([("missing", "A5", None)]).drop(columns="match_type"), HAND_TRUTH)


def test_ids_compared_as_strings():
    """Numeric-looking ids are not coerced: "007" and "7" are different features."""
    truth = _truth([("metal_loss", "007", "08", "matched_1to1")])
    assert score_matching(_table([("matched_1to1", "007", "08")]), truth)["n_correct_links"] == 1
    with pytest.raises(ValueError, match="'7'"):
        score_matching(_table([("matched_1to1", "7", "08")]), truth)



# ---- the table must account for every reported anomaly exactly once ---------------------------

def test_small_run_table_checks_are_zero(small_run):
    s = score_matching(*small_run)
    assert {k: s[k] for k in TABLE_CHECKS} == dict.fromkeys(TABLE_CHECKS, 0) and s["passes"] is True


def test_dropping_unmatched_rows_fails(small_run):
    table, truth = small_run
    unmatched = table.match_type.isin(["missing", "new"])
    s = score_matching(table[~unmatched], truth)
    assert s["n_unreported"] == int(unmatched.sum()) > 0
    assert (s["precision"], s["recall"]) == pytest.approx((score_matching(table, truth)["precision"],
                                                           score_matching(table, truth)["recall"]))
    assert s["passes"] is False


def test_duplicated_rows_fail(small_run):
    table, truth = small_run
    one = table[table.match_type == "matched_1to1"]
    s = score_matching(pd.concat([table, one]), truth)
    assert s["n_duplicated"] == 2 * len(one) and s["n_duplicate_links"] == len(one)
    assert s["n_unreported"] == 0 and s["passes"] is False


def test_new_relabelled_missing_fails(small_run):
    table, truth = small_run
    bad = table.copy()
    new = bad.match_type == "new"
    bad.loc[new, "match_type"] = "missing"
    s = score_matching(bad, truth)
    assert s["n_wrong_side"] == int(new.sum()) > 0 and s["passes"] is False


def test_empty_table_against_empty_truth_fails():
    s = score_matching(_table([]), _truth([]))
    assert s["n_true_links"] == 0 and s["passes"] is False


def test_hand_built_table_checks():
    table = _table([
        ("matched_1to1", "A1", "B1"),
        ("matched_1to1", "A1", "B1"),          # duplicated row: A1, B1 twice and the link twice
        ("matched_1to1", "A2;A5", "B2"),       # two prev ids on a 1:1 row
        ("matched_cluster", "A3;A4", None),    # cluster with no curr side
        ("missing", None, "B6"),               # missing with only a curr id
        ("unmatched", None, "B3"),             # not a match type
    ])
    s = score_matching(table, HAND_TRUTH)
    assert (s["n_unreported"], s["n_duplicated"], s["n_wrong_side"], s["n_duplicate_links"]) == (0, 2, 4, 1)
    assert s["passes"] is False
    s = score_matching(_table([("matched_1to1", "A1", "B1"), ("matched_1to1", "A2", "B2")]), HAND_TRUTH)
    assert s["n_unreported"] == 3 + 2  # A3, A4, A5 and B3, B6 are reported but absent
