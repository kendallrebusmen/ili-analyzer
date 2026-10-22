from datetime import date

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B, assert_matches_oracle

from ili.ingest import load_vendor_map, parse_clock, read_tally, standardize
from ili.schema import STANDARD_COLUMNS


def _ingest(path, map_path, run_id, run_date):
    vmap = load_vendor_map(map_path)
    raw, _ = read_tally(path, vmap)
    return standardize(raw, vmap, run_id, run_date)


def test_vendor_a_matches_oracle(synth):
    result, out = synth
    std, issues = _ingest(out / "tally_prev_vendor_a.csv", MAP_A, "2021_vendorA", date(2021, 6, 15))
    assert issues.empty
    assert list(std.columns) == STANDARD_COLUMNS + ["source_row"]
    assert_matches_oracle(std, result.prev_std)


def test_vendor_b_matches_oracle(synth):
    result, out = synth
    std, issues = _ingest(out / "tally_curr_vendor_b.xlsx", MAP_B, "2026_vendorB", date(2026, 6, 15))
    assert issues.empty
    assert_matches_oracle(std, result.curr_std)


def test_source_rows(synth):
    _, out = synth
    vmap = load_vendor_map(MAP_B)
    raw, _ = read_tally(out / "tally_curr_vendor_b.xlsx", vmap)
    assert raw.source_row.iloc[0] == 4  # header on row 3
    vmap = load_vendor_map(MAP_A)
    raw, _ = read_tally(out / "tally_prev_vendor_a.csv", vmap)
    assert raw.source_row.iloc[0] == 2


def test_excel_ids_stay_integers(synth):
    _, out = synth
    raw, _ = read_tally(out / "tally_curr_vendor_b.xlsx", load_vendor_map(MAP_B))
    gw = raw["Ref GW"][raw["Ref GW"] != ""]
    assert gw.str.fullmatch(r"\d+").all()


def test_run_date_from_title_row(synth):
    _, out = synth
    assert read_tally(out / "tally_curr_vendor_b.xlsx", load_vendor_map(MAP_B))[1] == date(2026, 6, 15)
    assert read_tally(out / "tally_prev_vendor_a.csv", load_vendor_map(MAP_A))[1] is None


def test_parse_clock_formats():
    clock, bad = parse_clock(pd.Series(["12:30", "3:00", "", "13:00", "x"]), "hhmm")
    assert clock.tolist()[:2] == [30, 180] and pd.isna(clock[2])
    assert bad.tolist() == [False, False, False, True, True]
    clock, bad = parse_clock(pd.Series(["90", "360", "359.9", "abc"]), "degrees")
    assert clock.tolist()[:3] == [180, 0, 0] and bad.tolist() == [False, False, False, True]
    clock, bad = parse_clock(pd.Series(["3.5", "12.0", "0"]), "decimal_hours")
    assert clock.tolist() == [210, 0, 0] and not bad.any()


def test_parse_clock_keeps_out_of_range_for_validation():
    clock, bad = parse_clock(pd.Series(["400", "-10"]), "degrees")
    assert clock.tolist() == [800, -20] and not bad.any()


def _raw_b(**overrides):
    row = {
        "Log Dist [m]": "1", "Feature": "ML-CORR", "Feature ID": "B-000001", "Ref GW": "1",
        "Rel. Dist [m]": "0.5", "Joint Len [m]": "12", "WT [mm]": "6.35", "Peak Depth [%]": "20",
        "Dent Depth [mm]": "", "Length [mm]": "25.4", "Width [mm]": "50.8", "Orientation [deg]": "90",
        "Location": "EXT", "Seam": "0", "Dent ML": "", "Cluster": "", "Lat": "32", "Lon": "-84",
        "Elev [m]": "100", "Comment": "",
    }
    row.update(overrides)
    df = pd.DataFrame([row])
    df["source_row"] = 4
    return df


def test_unit_conversions():
    std, issues = standardize(_raw_b(), load_vendor_map(MAP_B), "r", date(2026, 1, 1))
    assert issues.empty
    row = std.iloc[0]
    assert row.log_dist_ft == pytest.approx(3.28084, abs=1e-5)
    assert row.wt_nom_in == pytest.approx(0.25)
    assert row.length_in == pytest.approx(1.0) and row.width_in == pytest.approx(2.0)
    assert row.elev_ft == pytest.approx(328.084, abs=1e-3)
    assert row.clock_min == 180 and row.seam_interaction is False and row.dent_metal_loss is None


def test_bom_csv_header(synth, tmp_path):
    _, out = synth
    text = (out / "tally_prev_vendor_a.csv").read_text(encoding="utf-8")
    bom = tmp_path / "bom.csv"
    bom.write_text(text, encoding="utf-8-sig")
    vmap = load_vendor_map(MAP_A)
    raw, _ = read_tally(bom, vmap)
    assert "Feature No" in raw.columns
    std, issues = standardize(raw, vmap, "2021_vendorA", date(2021, 6, 15))
    assert issues.empty and std.feature_id.iloc[0] == "A-000001"


def test_vocab_whitespace_case():
    std, issues = standardize(_raw_b(Feature=" ml-corr ", Location=" ext"), load_vendor_map(MAP_B), "r",
                              date(2026, 1, 1))
    assert issues.empty
    assert std.feature_type.iloc[0] == "metal_loss" and std.surface.iloc[0] == "EXT"


def test_missing_mapped_column_raises(synth, tmp_path):
    _, out = synth
    df = pd.read_csv(out / "tally_prev_vendor_a.csv", dtype=str).drop(columns=["O'clock"])
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="O'clock"):
        read_tally(path, load_vendor_map(MAP_A))
