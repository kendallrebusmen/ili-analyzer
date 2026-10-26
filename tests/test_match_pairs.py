import numpy as np
import pandas as pd
import pytest

from ili.config import load_settings
from ili.match import (
    PositionMap, assign_one_to_one, candidates, depth_change_tol_pct, pair_metrics, place,
)
from tests.test_match_place import _std, _wm

DT = 5.0


def _settings(**matching):
    s = load_settings()
    return s.model_copy(update={"matching": s.matching.model_copy(update=matching)}) if matching else s


def _placed(rows, wm, run, **cols):
    """Placed frame from (feature_type, log_dist_ft) rows; cols are per-row lists set before placing."""
    df = _std(rows)
    for k, v in cols.items():
        df[k] = v
    return place(df, wm, run, PositionMap.from_weld_map(wm))


_WM1 = _wm(("1:1", 0.0, 100.0, 0.0, 100.0))
_WM2 = _wm(("1:1", 0.0, 100.0, 0.0, 100.0), ("1:1", 100.0, 100.0, 100.0, 100.0))


def _pair(prev_rows, curr_rows, wm=_WM1, prev_cols=None, curr_cols=None, **matching):
    prev = _placed(prev_rows, wm, "prev", **(prev_cols or {}))
    curr = _placed(curr_rows, wm, "curr", **(curr_cols or {}))
    return prev, curr, candidates(prev, curr, DT, _settings(**matching))


def test_score_zero_for_identical_position():
    prev, curr, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.0)])
    assert len(c) == 1 and c.score.iloc[0] == 0 and c.axial_diff_in.iloc[0] == 0


def test_clock_tolerance_wraps_at_noon():
    _, _, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.0)],
                    prev_cols={"clock_min": pd.array([710], dtype="Int64")},
                    curr_cols={"clock_min": pd.array([20], dtype="Int64")})
    assert len(c) == 1 and c.clock_diff_min.iloc[0] == 30
    _, _, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.0)],
                    prev_cols={"clock_min": pd.array([300], dtype="Int64")},
                    curr_cols={"clock_min": pd.array([20], dtype="Int64")})
    assert c.empty


def test_missing_clock_drops_circ_term():
    _, _, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.25)])
    assert c.score.iloc[0] == pytest.approx(0.5)  # 3 in / 6 in, circ term dropped


def test_surface_and_type_compatibility():
    def n(pt, ps, ct, cs):
        return len(_pair([(pt, 50.0)], [(ct, 50.0)], prev_cols={"surface": [ps]}, curr_cols={"surface": [cs]})[2])
    assert n("metal_loss", "EXT", "metal_loss", "INT") == 0
    assert n("metal_loss", "EXT", "metal_loss", "UNK") == 1
    assert n("metal_loss", "EXT", "metal_loss", "EXT") == 1
    assert n("metal_loss", "EXT", "dent", "EXT") == 0


def test_implausible_drop_and_growth():
    tol = depth_change_tol_pct(load_settings().tool)
    assert tol == pytest.approx(10 * 2 ** 0.5)

    def flags(d0, d1):
        m = pair_metrics(_placed([("metal_loss", 50.0)], _WM1, "prev", depth_pct=[d0], wt_nom_in=[0.25]),
                         _placed([("metal_loss", 50.0)], _WM1, "curr", depth_pct=[d1], wt_nom_in=[0.25]),
                         DT, load_settings())
        return bool(m.implausible.iloc[0])
    assert flags(30.0, 15.0) is True and flags(30.0, 16.0) is False       # drops of 15 and 14
    assert flags(10.0, 90.0) is True and flags(10.0, 80.0) is False       # growth 80 -> 32.9 mpy, 70 -> 27.9
    assert flags(None, 90.0) is False


def test_assignment_is_one_to_one_and_prefers_lower_total_cost():
    c = pd.DataFrame({"prev_i": [0, 0, 1], "curr_i": [0, 1, 1], "score": [0.1, 0.2, 0.3]})
    got = assign_one_to_one(c)
    assert sorted(zip(got.prev_i, got.curr_i)) == [(0, 0), (1, 1)]
    assert list(got.columns) == list(c.columns)
    assert assign_one_to_one(c.iloc[0:0]).empty


def test_assignment_drops_noncandidate_pairs_and_is_deterministic():
    # Hall violation: p0 and p1 both only fit c0, so one of them must go unmatched
    c = pd.DataFrame({"prev_i": [0, 1, 2, 2, 2], "curr_i": [0, 0, 0, 1, 2],
                      "score": [0.1, 0.2, 0.3, 0.1, 0.5]})
    got = assign_one_to_one(c)
    assert sorted(zip(got.prev_i, got.curr_i)) == [(0, 0), (2, 1)]
    assert got.equals(assign_one_to_one(c))


def test_cross_weld_candidate_only_near_weld():
    prev_rows, curr_rows = [("metal_loss", 99.98)], [("metal_loss", 100.02)]
    _, _, c = _pair(prev_rows, curr_rows, wm=_WM2)
    assert len(c) == 1 and (c.prev_i.iloc[0], c.curr_i.iloc[0]) == (0, 0)
    assert c.axial_diff_in.iloc[0] == pytest.approx(0.48)
    _, _, c = _pair(prev_rows, curr_rows, wm=_WM2, near_weld_ft=0.01)
    assert c.empty


def test_cross_weld_prev_feature_in_the_later_row():
    """The p_later=False branch: the previous-run feature is downstream of the weld (row 1) and the
    current-run feature upstream (row 0)."""
    prev, curr, c = _pair([("metal_loss", 100.02)], [("metal_loss", 99.98)], wm=_WM2)
    assert (prev.wm_row.iloc[0], curr.wm_row.iloc[0]) == (1, 0)
    assert len(c) == 1 and c.axial_diff_in.iloc[0] == pytest.approx(0.48)
    _, _, c = _pair([("metal_loss", 100.02)], [("metal_loss", 99.98)], wm=_WM2, near_weld_ft=0.01)
    assert c.empty


@pytest.mark.parametrize("prev_x, curr_x", [
    (99.98, 100.2),   # prev near the weld (0.02 ft), curr 0.2 ft past it
    (99.8, 100.02),   # prev 0.2 ft before the weld, curr near it
    (100.02, 99.8),   # prev in the later row and near, curr 0.2 ft back in the earlier row
    (100.2, 99.98),   # prev in the later row 0.2 ft past the weld, curr near it
])
def test_cross_weld_rejected_when_one_feature_is_far(prev_x, curr_x):
    """Within the axial tolerance (2.64 in) but only one of the two within near_weld_ft = 0.1 ft."""
    _, _, c = _pair([("metal_loss", prev_x)], [("metal_loss", curr_x)], wm=_WM2, near_weld_ft=0.1)
    assert c.empty
    _, _, c = _pair([("metal_loss", prev_x)], [("metal_loss", curr_x)], wm=_WM2, near_weld_ft=0.25)
    assert len(c) == 1


def test_cross_row_needs_matched_adjacent_rows():
    wm = _wm(("1:1", 0.0, 100.0, 0.0, 100.0), ("removed", 100.0, 100.0, np.nan, np.nan),
             ("1:1", 200.0, 100.0, 100.0, 100.0))
    prev = _placed([("metal_loss", 99.98)], wm, "prev")
    curr = _placed([("metal_loss", 100.02)], wm, "curr")
    assert candidates(prev, curr, DT, _settings()).empty


def test_vendor_cluster_rows_are_not_candidates():
    _, _, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.0)], curr_cols={"vendor_cluster_id": ["K1"]})
    assert c.empty
    _, _, c = _pair([("metal_loss", 50.0)], [("metal_loss", 50.0)], prev_cols={"vendor_cluster_id": ["K1"]})
    assert c.empty


def test_unmatched_row_features_are_not_candidates():
    wm = _wm(("inserted", np.nan, np.nan, 0.0, 100.0), ("removed", 0.0, 100.0, np.nan, np.nan),
             ("1:1", 500.0, 100.0, 500.0, 100.0))
    prev = _placed([("metal_loss", 50.0)], wm, "prev")
    curr = _placed([("metal_loss", 50.0)], wm, "curr")
    assert candidates(prev, curr, DT, _settings()).empty
