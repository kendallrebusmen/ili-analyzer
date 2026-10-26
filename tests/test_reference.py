import numpy as np
import pandas as pd
import pytest

from ili.align_welds import align_welds
from ili.config import load_settings
from ili.match import load_match_table, lookup_od, match_anomalies, write_matches
from ili.reference import assign_segment, lookup, segment_table
from ili.synth import SynthParams, generate
from ili.units import FT_PER_MI

S = load_settings()


def _tbl():
    return pd.DataFrame({"start_ft": [0.0, 100.0], "end_ft": [50.0, 200.0], "v": [10.0, 20.0]})


def test_lookup_nearest_row_and_errors():
    t = _tbl()
    # inside row 0, start of row 1 (inclusive), end of row 0 is exclusive -> gap, 200 is past row 1's end
    got = lookup(t, [25.0, 100.0, 49.9, 54.0, 98.0, 52.0, 204.0, -3.0], "v", 5.0)
    assert got.tolist() == [10.0, 20.0, 10.0, 10.0, 20.0, 10.0, 20.0, 10.0]
    with pytest.raises(ValueError, match="206"):
        lookup(t, [206.0], "v", 5.0)
    with pytest.raises(ValueError):
        lookup(t, [75.0], "v", 5.0)  # gap midpoint, 25 ft from both rows
    with pytest.raises(ValueError):
        lookup(t.iloc[0:0], [1.0], "v", 5.0)
    with pytest.raises(ValueError, match="missing"):
        lookup(t, [1.0], "nope", 5.0)
    nan = t.assign(v=[10.0, np.nan])
    with pytest.raises(ValueError):
        lookup(nan, [150.0], "v", 5.0)
    assert lookup(nan, [25.0, 10.0], "v", 5.0).tolist() == [10.0, 10.0]  # NaN row unused: fine


def test_lookup_od_unchanged():
    pipe = pd.DataFrame({"start_ft": [0.0, 100.0], "end_ft": [50.0, 200.0], "od_in": [12.75, 16.0]})
    assert lookup_od([10.0, 100.0, 60.0], pipe, 15.0).tolist() == [12.75, 16.0, 12.75]
    with pytest.raises(ValueError) as e:
        lookup_od([300.0], pipe, 15.0)
    assert str(e.value) == "position 300.000 ft is outside the pipe properties table by more than 15.0 ft"
    with pytest.raises(ValueError) as e:
        lookup_od([1.0], pipe.iloc[0:0], 15.0)
    assert str(e.value) == "the pipe properties table has no rows"
    with pytest.raises(ValueError) as e:
        lookup_od([1.0], pipe.assign(od_in=[np.nan, 16.0]), 15.0)
    assert str(e.value) == "pipe properties row starting at 0.000 ft has no od_in"
    assert lookup_od([150.0], pipe.assign(od_in=[np.nan, 16.0]), 15.0).tolist() == [16.0]


def test_fixed_length_segments():
    s = S.model_copy(deep=True)
    s.growth.segment_definition = "fixed_length"
    s.growth.fixed_segment_length_mi = 1.0
    end = 2.5 * FT_PER_MI
    seg = segment_table(s, None, 0.0, end)
    assert seg.segment_id.tolist() == ["FIX-001", "FIX-002", "FIX-003"]
    assert seg.start_ft.tolist() == [0.0, 5280.0, 10560.0]
    assert seg.end_ft.tolist() == [5280.0, 10560.0, end]
    assert list(seg.columns) == ["start_ft", "end_ft", "segment_id"]


def test_segment_table_requires_segments_and_columns():
    with pytest.raises(ValueError):
        segment_table(S, None, 0.0, 100.0)
    with pytest.raises(ValueError, match="segment_id"):
        segment_table(S, pd.DataFrame({"start_ft": [0.0], "end_ft": [1.0]}), 0.0, 100.0)


def test_assign_segment_outside_is_none():
    seg = pd.DataFrame({"start_ft": [0.0, 100.0], "end_ft": [50.0, 200.0], "segment_id": ["A", "B"]})
    got = assign_segment(seg, [10.0, 50.0, 75.0, 100.0, 199.0, 250.0, -1.0])
    assert got.tolist() == ["A", None, None, "B", "B", None, None]
    assert got.dtype == object


def test_load_match_table_roundtrip(tmp_path):
    r = generate(SynthParams(line_miles=5))
    pipe, repairs = r.reference["pipe_properties"], r.reference["repairs_between_runs"]
    wm = align_welds(r.prev_std, r.curr_std, S, repairs).weld_map
    res = match_anomalies(r.prev_std, r.curr_std, wm, S, pipe, repairs)
    paths = write_matches(res, tmp_path)
    t = load_match_table(paths["match_table.csv"])
    src = res.match_table
    assert len(t) == len(src) and list(t.columns) == list(src.columns)
    for c in ("match_id", "curr_feature_id", "prev_feature_id", "curr_us_gw_id"):
        assert all(v is None or (isinstance(v, str) and v not in ("", "nan") and not v.endswith(".0")) for v in t[c])
    assert t.curr_feature_id.tolist() == src.curr_feature_id.tolist()
    assert t.needs_review.map(lambda v: isinstance(v, bool)).all()
    assert t.needs_review.tolist() == src.needs_review.tolist()
    assert t.absorbed.map(lambda v: isinstance(v, bool)).all()
    unmatched = t.match_type.isin(["missing", "new"])
    assert unmatched.any() and t.implausible[unmatched].map(lambda v: v is None).all()
    assert t.implausible[~unmatched].map(lambda v: isinstance(v, bool)).all()
    assert t.missing_hint[t.match_type != "missing"].map(lambda v: v is None).all()
    np.testing.assert_allclose(t.curr_depth_pct.to_numpy(float), src.curr_depth_pct.to_numpy(float), equal_nan=True)
    # integer columns: Int64 dtype and the same values, <NA> where the source is blank
    for c in ("n_prev", "n_curr", "prev_wm_row", "curr_wm_row", "prev_clock_min", "curr_clock_min"):
        assert str(t[c].dtype) == "Int64", c
        assert t[c].isna().tolist() == src[c].isna().tolist(), c
        assert t[c].dropna().astype(int).tolist() == src[c].dropna().astype(int).tolist(), c
    assert t.curr_clock_min.isna().any() and t.prev_wm_row.isna().any()
    # text columns: equal, None (not NaN or "") where blank
    for c in ("prev_feature_type", "curr_feature_type", "prev_surface", "curr_surface",
              "prev_vendor_cluster_id", "curr_vendor_cluster_id", "match_type", "confidence", "review_status"):
        assert t[c].tolist() == [None if pd.isna(v) else v for v in src[c]], c
        assert all(v is None or isinstance(v, str) for v in t[c]), c
    # seam / dent flags: True/False/None, same as the source
    for c in ("prev_seam_interaction", "curr_seam_interaction", "prev_dent_metal_loss", "curr_dent_metal_loss"):
        assert all(v is None or isinstance(v, bool) for v in t[c]), c
        assert t[c].tolist() == [None if pd.isna(v) else bool(v) for v in src[c]], c
    assert t.curr_seam_interaction.map(lambda v: v is None).any() and t.curr_seam_interaction.map(lambda v: v is not None).any()


def test_segment_table_rejects_duplicate_ids_and_overlap():
    # Unsorted input; segments that touch (end == next start) are fine.
    ok = pd.DataFrame({"start_ft": [100.0, 0.0], "end_ft": [200.0, 100.0], "segment_id": ["B", "A"]})
    assert segment_table(S, ok, 0.0, 200.0).segment_id.tolist() == ["B", "A"]
    dup = pd.DataFrame({"start_ft": [0.0, 1100.0], "end_ft": [1000.0, 2000.0], "segment_id": ["HCA-1", "HCA-1"]})
    with pytest.raises(ValueError, match="repeats segment_id 'HCA-1'"):
        segment_table(S, dup, 0.0, 2000.0)
    numeric = pd.DataFrame({"start_ft": [0.0, 100.0], "end_ft": [100.0, 200.0], "segment_id": [7, "7"]})
    with pytest.raises(ValueError, match="repeats segment_id '7'"):  # ids compare as strings
        segment_table(S, numeric, 0.0, 200.0)
    overlap = pd.DataFrame({"start_ft": [500.0, 0.0], "end_ft": [900.0, 600.0], "segment_id": ["B", "A"]})
    with pytest.raises(ValueError, match="'A' and 'B' overlap"):
        segment_table(S, overlap, 0.0, 900.0)
