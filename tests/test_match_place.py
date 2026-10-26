from datetime import date

import numpy as np
import pandas as pd
import pytest

from ili.align_welds import align_welds, load_weld_map, write_alignment
from ili.config import load_settings
from ili.match import PositionMap, clock_diff_min, delta_t_years, place
from ili.schema import STANDARD_COLUMNS


def _wm(*rows):
    cols = ["match_type", "prev_start_ft", "prev_length_ft", "curr_start_ft", "curr_length_ft"]
    return pd.DataFrame(rows, columns=cols)


def _std(rows, run_date=date(2021, 6, 15)):
    """Standardized frame from (feature_type, log_dist_ft) pairs."""
    df = pd.DataFrame({c: [None] * len(rows) for c in STANDARD_COLUMNS})
    df["feature_type"] = [t for t, _ in rows]
    df["log_dist_ft"] = [float(x) for _, x in rows]
    df["feature_id"] = [str(i) for i in range(len(rows))]
    df["run_date"] = run_date
    df["clock_min"] = pd.array([None] * len(rows), dtype="Int64")
    return df


def test_clock_diff_wraps():
    got = clock_diff_min(np.array([715, 0, 100], dtype=float), np.array([5, 360, np.nan]))
    assert got[0] == 10 and got[1] == 360 and np.isnan(got[2])


def test_delta_t_years():
    prev, curr = _std([("dent", 1)], date(2021, 6, 15)), _std([("dent", 1)], date(2026, 6, 15))
    assert delta_t_years(prev, curr) == pytest.approx(1826 / 365.25)
    for a, b in [(date(2021, 6, 15), date(2021, 6, 15)), (date(2026, 6, 15), date(2021, 6, 15)),
                 (None, date(2026, 6, 15))]:
        with pytest.raises(ValueError):
            delta_t_years(_std([("dent", 1)], a), _std([("dent", 1)], b))


def test_delta_t_rejects_mixed_dates():
    prev = _std([("dent", 1), ("dent", 2)])
    prev["run_date"] = [date(2021, 1, 1), date(2021, 6, 15)]
    with pytest.raises(ValueError):
        delta_t_years(prev, _std([("dent", 1)], date(2026, 6, 15)))


def test_position_map_1to1_scales_by_joint_ratio():
    pmap = PositionMap.from_weld_map(_wm(("1:1", 100.0, 40.0, 200.0, 40.4)))
    assert pmap.to_prev(np.array([220.2]))[0] == pytest.approx(120.0, abs=1e-9)
    assert pmap.to_curr(np.array([120.0]))[0] == pytest.approx(220.2, abs=1e-9)


def test_position_map_2to1_spans_merged_joint():
    pmap = PositionMap.from_weld_map(_wm(("2:1", 0.0, 80.0, 0.0, 79.2)))
    assert pmap.to_prev(np.array([39.6]))[0] == pytest.approx(40.0, abs=1e-9)


def test_position_map_extrapolates_with_overall_slope():
    pmap = PositionMap.from_weld_map(_wm(("1:1", 0.0, 40.0, 10.0, 40.0), ("1:1", 40.0, 40.0, 50.0, 40.0)))
    assert pmap.to_prev(np.array([-5.0, 100.0])).tolist() == pytest.approx([-15.0, 90.0])


def test_place_assigns_rows_and_flags_unmatched_rows():
    wm = _wm(("1:1", 100.0, 40.0, 100.0, 40.0),
             ("inserted", np.nan, np.nan, 140.0, 20.0),
             ("removed", 140.0, 30.0, np.nan, np.nan),
             ("1:1", 170.0, 40.0, 160.0, 40.0))
    pmap = PositionMap.from_weld_map(wm)
    curr = _std([("metal_loss", 50), ("metal_loss", 110), ("dent", 150), ("valve", 120), ("gouge", 199)])
    out = place(curr, wm, "curr", pmap)
    assert out.feature_type.tolist() == ["metal_loss", "metal_loss", "dent", "gouge"]  # valve dropped
    assert out.wm_row.tolist() == [-1, 0, 1, 3]
    assert out.in_matched_row.tolist() == [False, True, False, True]
    assert np.isnan(out.edge_dist_ft.iloc[0])
    assert out.edge_dist_ft.iloc[1] == pytest.approx(10.0)
    assert out.edge_dist_ft.iloc[3] == pytest.approx(1.0)
    assert (out.pos_curr_ft == out.log_dist_ft).all()
    assert out.pos_prev_ft.iloc[1] == pytest.approx(110.0)
    prev = _std([("metal_loss", 145), ("metal_loss", 99.0)])
    p = place(prev, wm, "prev", pmap)
    assert p.wm_row.tolist() == [2, -1]  # the removed row is a prev-side span
    assert (p.pos_prev_ft == p.log_dist_ft).all()
    assert p.span_start_ft.iloc[0] == 140.0 and p.span_end_ft.iloc[0] == 170.0 and np.isnan(p.span_start_ft.iloc[1])
    assert out.span_start_ft.tolist()[1:2] == [100.0] and out.span_end_ft.tolist()[1:2] == [140.0]


def test_load_weld_map_keeps_string_ids(synth, tmp_path):
    result, _ = synth
    aligned = align_welds(result.prev_std, result.curr_std, load_settings(), None)
    paths = write_alignment(aligned, tmp_path)
    wm = load_weld_map(paths["weld_map.csv"])
    assert wm.curr_gw_id.dropna().iloc[0] == "1"
    ids = pd.concat([wm[c] for c in ("prev_gw_id", "prev_gw_id_2", "curr_gw_id", "curr_gw_id_2")]).dropna()
    assert ids.map(type).eq(str).all() and not ids.str.endswith(".0").any()
    assert wm.in_repair.dtype == bool and wm.is_anchor.dtype == bool
