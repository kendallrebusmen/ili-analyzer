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


def test_lookup_od_unchanged():
    pipe = pd.DataFrame({"start_ft": [0.0, 100.0], "end_ft": [50.0, 200.0], "od_in": [12.75, 16.0]})
    assert lookup_od([10.0, 100.0, 60.0], pipe, 15.0).tolist() == [12.75, 16.0, 12.75]
    with pytest.raises(ValueError, match="outside the pipe properties"):
        lookup_od([300.0], pipe, 15.0)


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
