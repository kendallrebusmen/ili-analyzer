import numpy as np
import pandas as pd
import pytest

from ili.align_welds import build_joints, match_anchors
from ili.config import load_settings
from ili.synth import SynthParams, generate

CFG = load_settings().alignment


@pytest.fixture(scope="module")
def runs():
    r = generate(SynthParams(line_miles=5))
    return r.prev_std, r.curr_std


def make_std(weld_logs, anchors=(), run_id="r"):
    """Minimal standard frame: girth welds at ``weld_logs`` and (type, log) anchors."""
    rows = [{"feature_id": f"W{i}", "feature_type": "girth_weld", "log_dist_ft": x, "us_gw_id": f"G{i}"}
            for i, x in enumerate(weld_logs)]
    rows += [{"feature_id": f"A{i}", "feature_type": t, "log_dist_ft": x, "us_gw_id": None}
             for i, (t, x) in enumerate(anchors)]
    df = pd.DataFrame(rows)
    df["run_id"] = run_id
    return df.sort_values("log_dist_ft", kind="stable").reset_index(drop=True)


def test_build_joints_counts(runs):
    prev, _ = runs
    j = build_joints(prev)
    n_welds = (prev.feature_type == "girth_weld").sum()
    assert len(j) == n_welds - 1
    assert (j.length_ft > 0).all()
    assert np.allclose(j.end_ft.to_numpy()[:-1], j.start_ft.to_numpy()[1:])
    assert j.anchor_type.notna().sum() == 10
    assert set(j.anchor_type.dropna()) == {"valve", "tee"}
    assert list(j.columns) == ["joint_idx", "gw_id", "start_ft", "end_ft", "length_ft", "anchor_type", "anchor_ft"]
    a = j[j.anchor_type.notna()]
    assert ((a.anchor_ft >= a.start_ft) & (a.anchor_ft < a.end_ft)).all()
    assert j[j.anchor_type.isna()].anchor_ft.isna().all()


def test_build_joints_ignores_non_weld_rows():
    std = make_std([0, 40, 80], anchors=[("valve", 50)])
    std = pd.concat([std, pd.DataFrame([{"feature_id": "M1", "feature_type": "metal_loss", "log_dist_ft": 10,
                                         "us_gw_id": "GHOST", "run_id": "r"}])], ignore_index=True)
    j = build_joints(std)
    assert j.gw_id.tolist() == ["G0", "G1"] and j.anchor_type.tolist() == [None, "valve"]


def test_anchor_pairs_synthetic(runs):
    pj, cj = (build_joints(s) for s in runs)
    pairs = match_anchors(pj, cj, CFG)
    pa = pj.index[pj.anchor_type.notna()].tolist()
    ca = cj.index[cj.anchor_type.notna()].tolist()
    assert pairs == list(zip(pa, ca))


def test_anchor_missing_in_one_run(runs):
    prev, curr = runs
    tee_row = curr.index[curr.feature_type == "tee"][1]
    pj, cj_full = build_joints(prev), build_joints(curr)
    cj = build_joints(curr.drop(index=tee_row))  # same welds, so same joint indices
    dropped_joint = int(np.searchsorted(cj_full.start_ft, curr.log_dist_ft[tee_row], side="right") - 1)
    pa = pj.index[pj.anchor_type.notna()].tolist()
    ca = cj_full.index[cj_full.anchor_type.notna()].tolist()
    expected = [(p, c) for p, c in zip(pa, ca) if c != dropped_joint]
    pairs = match_anchors(pj, cj, CFG)
    assert len(pairs) == 9
    assert pairs == expected


def test_anchor_type_must_match():
    prev = make_std([0, 40, 80, 120, 160, 200], anchors=[("valve", 50), ("tee", 130)])
    curr = make_std([0, 40, 80, 120, 160, 200], anchors=[("tee", 50), ("tee", 130)])
    pairs = match_anchors(build_joints(prev), build_joints(curr), CFG)
    assert pairs == [(3, 3)]


def test_anchor_too_far_not_paired():
    prev = make_std([0, 40, 80, 120, 4000, 4040], anchors=[("valve", 50), ("tee", 4010)])
    curr = make_std([0, 40, 2000, 2040, 4000, 4040], anchors=[("valve", 2010), ("tee", 4010)])
    pairs = match_anchors(build_joints(prev), build_joints(curr), CFG)
    assert pairs == [(4, 4)]
