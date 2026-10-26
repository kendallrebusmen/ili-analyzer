import numpy as np
import pandas as pd
import pytest

from ili.config import load_settings
from ili.match import envelope, interaction_groups, match_clusters
from tests.test_match_pairs import _WM1, _placed

OD = 12.75
S = load_settings()
MIN_PER_IN = 720 / (np.pi * OD)  # clock minutes per inch of arc


def _frame(run, pos, clock=None, length=0.3, width=0.3, depth=30.0, vendor=None, surface=None, wm=_WM1):
    n = len(pos)
    cols = {
        "length_in": [length] * n if np.isscalar(length) else list(length),
        "width_in": [width] * n if np.isscalar(width) else list(width),
        "depth_pct": [depth] * n if np.isscalar(depth) else list(depth),
        "od_in": [OD] * n,
        "wt_nom_in": [0.25] * n,
        "clock_min": pd.array(clock if clock is not None else [100] * n, dtype="Int64"),
        "vendor_cluster_id": vendor if vendor is not None else [None] * n,
        "surface": surface if surface is not None else [None] * n,
    }
    return _placed([("metal_loss", float(x)) for x in pos], wm, run, **cols)


def test_pits_within_spacing_group():
    step = 0.6 / 12
    f = _frame("prev", [50 + i * step for i in range(3)], clock=[92, 100, 108])
    assert len(set(interaction_groups(f, S))) == 1
    f = _frame("prev", [50, 50 + step, 50 + 3 / 12 + 2 * step + 0.0], clock=[92, 100, 108])
    g = interaction_groups(f, S)
    assert len(set(g)) == 2 and g[0] == g[1] != g[2]


def test_surface_must_match_for_interaction():
    f = _frame("prev", [50, 50.01], surface=["EXT", "INT"])
    assert len(set(interaction_groups(f, S))) == 2


def test_envelope_box_and_max_depth():
    f = _frame("prev", [50.0, 50.5 / 12], length=[1.0, 1.0], depth=[20.0, 45.0])
    f["pos_prev_ft"] = [50.0, 50.0 + 2 / 12]
    f["pos_curr_ft"] = f.pos_prev_ft + 1
    e = envelope(f)
    assert e["length_in"] == pytest.approx(3.0)  # 0.5 in + 2 in + 0.5 in
    assert e["pos_prev_ft"] == pytest.approx(50.0 + 1 / 12) and e["pos_curr_ft"] == pytest.approx(51.0 + 1 / 12)
    assert e["depth_pct"] == 45.0
    g = _frame("prev", [50.0, 50.0], clock=[715, 5], width=[0.3, 0.3])
    e = envelope(g)
    assert e["clock_min"] == 0
    assert e["width_in"] == pytest.approx(10 * np.pi * OD / 720 + 0.3)


def _cluster_prev(**kw):
    return _frame("prev", [50.0], clock=[100], length=3.0, width=3.0, vendor=["C1"], **kw)


def test_vendor_cluster_matches_pit_group():
    prev = _cluster_prev()
    curr = _frame("curr", [50 - 0.6 / 12, 50.0, 50 + 0.6 / 12], clock=[96, 100, 104])
    out = match_clusters(prev, curr, 5.0, S)
    assert len(out) == 1
    p, c, m = out[0]
    assert p == [0] and c == [0, 1, 2] and m.score == pytest.approx(0.0)


def test_vendor_cluster_matches_single_surviving_pit():
    out = match_clusters(_cluster_prev(), _frame("curr", [50.0]), 5.0, S)
    assert len(out) == 1 and out[0][0] == [0] and out[0][1] == [0]


def test_single_vs_single_is_not_a_cluster():
    assert match_clusters(_frame("prev", [50.0]), _frame("curr", [50 + 3 / 12]), 5.0, S) == []


IN = 1 / 12  # ft per inch
CIRC_GAP_PER_MIN = lambda d_min, half_widths: d_min / MIN_PER_IN - half_widths  # noqa: E731


def _vendor(pos=50.0, clock=100, length=2.0, width=2.0, depth=30.0, surface=None):
    return _frame("prev", [pos], clock=[clock], length=length, width=width, depth=depth,
                  vendor=["C1"], surface=None if surface is None else [surface])


def test_axial_box_gap_accept_and_reject():
    prev = _vendor()  # box +-1 in; the pit is 0.3 in long, so gap = offset - 1.15
    inside = match_clusters(prev, _frame("curr", [50 + 7.05 * IN]), 5.0, S)
    assert len(inside) == 1 and inside[0][2].axial_diff_in == pytest.approx(5.9)
    assert match_clusters(prev, _frame("curr", [50 + 7.25 * IN]), 5.0, S) == []


def test_circ_box_gap_converted_to_minutes():
    prev = _vendor()  # half widths total 1.15 in
    out = match_clusters(prev, _frame("curr", [50.0], clock=[130]), 5.0, S)
    assert len(out) == 1
    expect = (30 / MIN_PER_IN - 1.15) * MIN_PER_IN
    assert out[0][2].clock_diff_min == pytest.approx(expect) and out[0][2].axial_diff_in == 0
    assert len(match_clusters(prev, _frame("curr", [50.0], clock=[160]), 5.0, S)) == 1   # 39.3 min
    assert match_clusters(prev, _frame("curr", [50.0], clock=[170]), 5.0, S) == []       # 49.3 min


def test_circ_gap_uses_current_od():
    prev = _vendor()
    curr = _frame("curr", [50.0], clock=[130])
    curr["od_in"] = 24.0
    out = match_clusters(prev, curr, 5.0, S)
    assert out[0][2].clock_diff_min == pytest.approx(30 - 1.15 * 720 / (np.pi * 24.0))


def test_circ_separation_breaks_interaction_and_wraps_noon():
    def n(clocks):
        return len(set(interaction_groups(_frame("prev", [50.0, 50.0], clock=clocks), S)))
    assert n([100, 120]) == 1 and n([100, 130]) == 2   # 20 min -> 0.77 in gap, 30 min -> 1.37 in
    assert n([715, 5]) == 1


def test_many_to_one_contention_lower_score_wins():
    prev = _frame("prev", [50.0, 50 + 4 * IN], clock=[100, 100], length=2.0, width=2.0, vendor=["A", "B"])
    out = match_clusters(prev, _frame("curr", [50.0]), 5.0, S)
    assert len(out) == 1 and out[0][0] == [0] and out[0][1] == [0]


def test_cluster_surface_compatibility():
    def n(ps, cs):
        return len(match_clusters(_vendor(surface=ps), _frame("curr", [50.0], surface=[cs]), 5.0, S))
    assert n("EXT", "INT") == 0 and n("EXT", "UNK") == 1 and n("EXT", "EXT") == 1


def test_implausible_on_envelope_max_depths():
    curr = _frame("curr", [50.0, 50.0 + 0.6 * IN], depth=[20.0, 30.0], clock=[100, 100])
    m = match_clusters(_vendor(depth=60.0), curr, 5.0, S)[0][2]
    assert m.depth_change_pct == pytest.approx(-30.0) and bool(m.implausible)
    assert m.score == pytest.approx(S.matching.weights.depth_change)
    m = match_clusters(_vendor(depth=35.0), curr, 5.0, S)[0][2]
    assert m.depth_change_pct == pytest.approx(-5.0) and not bool(m.implausible)


def test_envelope_includes_vendor_member_box():
    f = _frame("prev", [50.0, 50.0 + 2 * IN], length=[3.0, 0.3], width=[2.0, 0.3], vendor=["C1", None])
    f["clock_min"] = pd.array([100, 100], dtype="Int64")
    e = envelope(f)
    assert e["length_in"] == pytest.approx(3.65) and e["width_in"] == pytest.approx(2.0)
    assert e["clock_min"] == 100


def test_envelope_clockless_member_keeps_others_clock():
    f = _frame("prev", [50.0, 50.0], length=[2.0, 0.3], width=[2.0, 0.3], vendor=["C1", None])
    f["clock_min"] = pd.array([100, None], dtype="Int64")
    e = envelope(f)
    assert e["clock_min"] == 100 and e["width_in"] == pytest.approx(2.0)
    assert isinstance(e["clock_min"], float)


def test_envelope_no_clock_or_no_od_is_nan():
    f = _frame("prev", [50.0, 50.0], width=[2.0, 0.3])
    f["clock_min"] = pd.array([None, None], dtype="Int64")
    e = envelope(f)
    assert np.isnan(e["clock_min"]) and e["width_in"] == 2.0
    g = _frame("prev", [50.0, 50.0], width=[2.0, 0.3])
    g["od_in"] = np.nan
    e = envelope(g)
    assert np.isnan(e["clock_min"]) and e["width_in"] == 2.0


def test_nan_clock_and_od_interaction_and_matching():
    f = _frame("prev", [50.0, 50.0])
    f["clock_min"] = pd.array([100, None], dtype="Int64")
    assert len(set(interaction_groups(f, S))) == 1
    g = _frame("prev", [50.0, 50.0], clock=[100, 300])
    g["od_in"] = np.nan
    assert len(set(interaction_groups(g, S))) == 1
    curr = _frame("curr", [50.0])
    curr["clock_min"] = pd.array([None], dtype="Int64")
    out = match_clusters(_vendor(), curr, 5.0, S)
    assert len(out) == 1 and np.isnan(out[0][2].clock_diff_min)
    curr["od_in"] = np.nan
    assert len(match_clusters(_vendor(), curr, 5.0, S)) == 1


def test_dent_beside_pits_does_not_join_group():
    f = _frame("prev", [50.0, 50.0 + 0.5 * IN, 50.0 + 1.0 * IN])
    f["feature_type"] = ["metal_loss", "dent", "metal_loss"]
    g = interaction_groups(f, S)
    assert g[0] == g[2] and g[1] != g[0]
    curr = f.copy()
    out = match_clusters(_vendor(), curr, 5.0, S)
    assert len(out) == 1 and out[0][1] == [0, 2]


# ---- vendor cluster reports absorb leftover pit groups ---------------------------------------

def test_vendor_cluster_absorbs_pit_groups_the_interaction_rule_splits():
    """Two pits 2.1 in apart (edge gap > axial_spacing_in) are separate groups, but both lie inside
    the vendor report's +-2 in box: the report takes both, and the metrics use the merged envelope."""
    prev = _vendor(length=4.0, width=3.0, depth=38.0)
    curr = _frame("curr", [50 - 1.2 * IN, 50 + 1.2 * IN], clock=[100, 100], depth=[20.0, 40.0])
    assert len(set(interaction_groups(curr, S))) == 2
    out = match_clusters(prev, curr, 5.0, S)
    assert len(out) == 1
    p, c, m = out[0]
    assert p == [0] and c == [0, 1]
    assert m.depth_change_pct == pytest.approx(2.0)  # merged max depth 40, not the first group's 20
    assert m.axial_diff_in == 0 and m.score == pytest.approx(0.0)


def test_vendor_cluster_absorption_respects_axial_tolerance():
    """The vendor box is +-2 in and a 0.3 in pit adds 0.15 in: box gap = offset - 2.15 in."""
    prev = _vendor(length=4.0, width=3.0)
    near = match_clusters(prev, _frame("curr", [50.0, 50 + 8.05 * IN]), 5.0, S)   # gap 5.9 in
    assert len(near) == 1 and near[0][1] == [0, 1]
    far = match_clusters(prev, _frame("curr", [50.0, 50 + 8.25 * IN]), 5.0, S)    # gap 6.1 in
    assert len(far) == 1 and far[0][1] == [0]


def test_leftover_pit_joins_the_lowest_score_vendor_cluster():
    """Vendor reports at 50 ft and 50 ft + 10 in each match their own pit; the leftover pit at
    +6 in is within tolerance of both (gaps 4.85 and 2.85 in) and joins the nearer, second one."""
    prev = _frame("prev", [50.0, 50 + 10 * IN], clock=[100, 100], length=2.0, width=2.0, vendor=["A", "B"])
    curr = _frame("curr", [50.0, 50 + 6 * IN, 50 + 10 * IN], clock=[100, 100, 100])
    out = match_clusters(prev, curr, 5.0, S)
    assert [(p, c) for p, c, _ in out] == [([0], [0]), ([1], [1, 2])]
    assert out[1][2].axial_diff_in == 0


def test_absorption_needs_the_vendor_report_on_the_opposite_side():
    """A leftover curr pit joins only a match whose prev side holds a vendor report: with the report
    on the curr side (same side as the leftover) the pit stays unmatched."""
    curr = _frame("curr", [50.0, 50 + 0.6 * IN, 50 + 3 * IN], clock=[100, 100, 100])
    same = match_clusters(_frame("prev", [50.0, 50 + 0.6 * IN], clock=[100, 100]),
                          curr.assign(vendor_cluster_id=["V", None, None]), 5.0, S)
    assert [(p, c) for p, c, _ in same] == [([0, 1], [0, 1])]
    opposite = match_clusters(_frame("prev", [50.0, 50 + 0.6 * IN], clock=[100, 100], vendor=["V", None]),
                              curr, 5.0, S)
    assert [(p, c) for p, c, _ in opposite] == [([0, 1], [0, 1, 2])]
