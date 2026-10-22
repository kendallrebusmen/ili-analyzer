from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import MAP_A, MAP_B

from ili.ingest import ISSUE_COLUMNS, load_vendor_map, read_tally, standardize, validate
from ili.schema import StandardFeature

RUN = "2021_vendorA"


def _run_corrupted(synth, tmp_path, corrupt):
    """Corrupt a copy of the synthetic vendor-A tally, then read, standardize, validate."""
    _, out = synth
    df = pd.read_csv(out / "tally_prev_vendor_a.csv", dtype=str, keep_default_na=False)
    corrupt(df)
    path = tmp_path / "corrupt.csv"
    df.to_csv(path, index=False)
    vmap = load_vendor_map(MAP_A)
    raw, _ = read_tally(path, vmap)
    std, parse_issues = standardize(raw, vmap, RUN, date(2021, 6, 15))
    clean, issues = validate(std, RUN, parse_issues)
    return df, clean, issues


def _first(df, event):
    return df.index[df.Event == event][0]


def test_clean_input_has_no_issues(synth, tmp_path):
    df, clean, issues = _run_corrupted(synth, tmp_path, lambda d: None)
    assert issues.empty and list(issues.columns) == ISSUE_COLUMNS
    assert len(clean) == len(df)


def _assert_one_bad_row(df, clean, issues, row, reason):
    fid = df.loc[row, "Feature No"]
    hit = issues[issues.reason == reason]
    assert len(hit) >= 1, issues
    assert (hit.source_row == row + 2).any()
    assert fid not in set(clean.feature_id) or fid == ""
    assert len(clean) == len(df) - 1


def test_depth_out_of_range(synth, tmp_path):
    holder = {}

    def corrupt(d):
        holder["row"] = _first(d, "Metal Loss")
        d.loc[holder["row"], "Depth (%)"] = "120"

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    _assert_one_bad_row(df, clean, issues, holder["row"], "depth_out_of_range")
    assert issues.loc[issues.reason == "depth_out_of_range", "value"].iloc[0] == 120.0


def test_missing_feature_id(synth, tmp_path):
    def corrupt(d):
        d.loc[5, "Feature No"] = ""

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    hit = issues[issues.reason == "missing_required"]
    assert hit.field.tolist() == ["feature_id"] and hit.source_row.tolist() == [7]
    assert len(clean) == len(df) - 1


def test_duplicate_feature_id(synth, tmp_path):
    def corrupt(d):
        d.loc[10, "Feature No"] = d.loc[9, "Feature No"]

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    hit = issues[issues.reason == "duplicate_feature_id"]
    assert hit.source_row.tolist() == [12]  # second occurrence only
    assert len(clean) == len(df) - 1 and clean.feature_id.is_unique


def test_weld_out_of_order(synth, tmp_path):
    holder = {}

    def corrupt(d):
        welds = d.index[d.Event == "Girth Weld"]
        a, b = welds[20], welds[21]
        holder["rows"] = (a, b)
        d.loc[a, "Abs Dist (ft)"], d.loc[b, "Abs Dist (ft)"] = d.loc[b, "Abs Dist (ft)"], d.loc[a, "Abs Dist (ft)"]

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    hit = issues[issues.reason == "girth_weld_not_monotonic"]
    assert sorted(hit.source_row) == [r + 2 for r in holder["rows"]]
    assert len(clean) == len(df) - 2
    gw = clean[clean.feature_type == "girth_weld"].log_dist_ft.to_numpy()
    assert (np.diff(gw) > 0).all()


def test_degrees_out_of_range():
    vmap = load_vendor_map(MAP_B)
    raw = pd.DataFrame([{col: "" for col in vmap.columns.values()} | {
        "Feature ID": "B-1", "Feature": "ML-CORR", "Log Dist [m]": "5", "Orientation [deg]": "400",
    }])
    raw["source_row"] = 4
    std, parse_issues = standardize(raw, vmap, "r", date(2026, 1, 1))
    clean, issues = validate(std, "r", parse_issues)
    assert issues.reason.tolist() == ["clock_out_of_range"] and clean.empty


def test_unparseable_number_and_flag(synth, tmp_path):
    holder = {}

    def corrupt(d):
        ml = d.index[d.Event == "Metal Loss"]
        holder["rows"] = (ml[0], ml[1])
        d.loc[ml[0], "Depth (%)"] = "12,3"
        d.loc[ml[1], "Seam Interaction"] = "maybe"

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    by_reason = issues.set_index("reason")
    assert by_reason.loc["unparseable_number", "value"] == "12,3"
    assert by_reason.loc["unparseable_number", "field"] == "depth_pct"
    assert by_reason.loc["unparseable_flag", "value"] == "maybe"
    assert len(clean) == len(df) - 2


def test_unknown_event(synth, tmp_path):
    def corrupt(d):
        d.loc[3, "Event"] = "Bogus"

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    assert issues.reason.tolist() == ["unknown_feature_type"]  # not also missing_required
    assert issues.value.tolist() == ["Bogus"]
    assert len(clean) == len(df) - 1


def test_row_with_two_problems_excluded_once(synth, tmp_path):
    def corrupt(d):
        row = _first(d, "Metal Loss")
        d.loc[row, "Depth (%)"] = "150"
        d.loc[row, "O'clock"] = "25:00"

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    assert set(issues.reason) == {"depth_out_of_range", "unparseable_clock"}
    assert len(clean) == len(df) - 1


def test_clean_rows_validate_against_schema(synth, tmp_path):
    _, clean, _ = _run_corrupted(synth, tmp_path, lambda d: None)
    records = clean.drop(columns="source_row").astype(object).where(clean.notna(), None).to_dict("records")
    for rec in records:
        StandardFeature.model_validate(rec)


@pytest.mark.parametrize("bad", ["-0.5", "100.5"])
def test_depth_bounds_inclusive(synth, tmp_path, bad):
    def corrupt(d):
        ml = d.index[d.Event == "Metal Loss"]
        d.loc[ml[0], "Depth (%)"] = "0"
        d.loc[ml[1], "Depth (%)"] = "100"
        d.loc[ml[2], "Depth (%)"] = bad

    df, clean, issues = _run_corrupted(synth, tmp_path, corrupt)
    assert issues.reason.tolist() == ["depth_out_of_range"]
    assert len(clean) == len(df) - 1
