from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili.align_welds import align_welds
from ili.config import load_settings
from ili.ingest import ingest_tally
from ili.match import match_anomalies
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


def _assert_cluster_reports_matched_as_clusters(table: pd.DataFrame, truth: pd.DataFrame) -> int:
    """Every vendor cluster report (a feature id several true pits share, truth matched_cluster) sits
    in a matched_cluster row. Returns how many reports were checked."""
    n = 0
    for run in ("prev", "curr"):
        reports = set(truth.loc[(truth[f"{run}_status"] == "reported_in_cluster")
                                & (truth.truth_class == "matched_cluster"), f"{run}_feature_id"])
        rows = table[f"{run}_feature_id"].fillna("").str.split(";")
        kind = {fid: mt for ids, mt in zip(rows, table.match_type) for fid in ids if fid in reports}
        assert set(kind) == reports
        assert {f: k for f, k in kind.items() if k != "matched_cluster"} == {}
        n += len(reports)
    return n


def test_default_dataset_meets_acceptance(default_line):
    table, truth = default_line
    score = score_matching(table, truth)
    assert score["precision"] >= 0.98, score["false_links"]
    assert score["recall"] >= 0.95, score["missed_links"]
    assert score["passes"] is True
    assert _assert_cluster_reports_matched_as_clusters(table, truth) > 0


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
    assert _assert_cluster_reports_matched_as_clusters(table, r.ground_truth) > 0


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
    """Rows: (feature_type, prev_feature_id, curr_feature_id, truth_class)."""
    return pd.DataFrame(rows, columns=["feature_type", "prev_feature_id", "curr_feature_id", "truth_class"])


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
