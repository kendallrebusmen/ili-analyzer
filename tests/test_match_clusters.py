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
