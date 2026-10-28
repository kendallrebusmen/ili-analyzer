"""Alignment robustness cases found in the M3 final review: runs covering different extents,
missed welds beside anchors, local odometer jumps, head offsets beyond the band, and a
scorer that must reject duplicated or contradictory rows."""

import functools

import numpy as np
import pandas as pd
import pytest

from ili.align_welds import align_section, align_welds, build_joints, match_anchors
from ili.config import load_settings
from ili.synth import SynthParams, generate
from ili.validation import score_alignment

SETTINGS = load_settings()
CFG = SETTINGS.alignment


@functools.cache
def dataset(seed):
    return generate(SynthParams(line_miles=5, seed=seed))


@pytest.fixture(scope="module")
def line():
    return dataset(1)


def welds(std):
    return std[std.feature_type == "girth_weld"].sort_values("log_dist_ft", kind="stable")


def score(r, prev=None, curr=None, repairs="default"):
    prev = r.prev_std if prev is None else prev
    curr = r.curr_std if curr is None else curr
    repairs = r.reference["repairs_between_runs"] if isinstance(repairs, str) else repairs
    result = align_welds(prev, curr, SETTINGS, repairs)
    return result, score_alignment(result.weld_map, prev, curr, r.weld_truth)


def drop_weld(std, gw_id):
    return std[~((std.feature_type == "girth_weld") & (std.us_gw_id == gw_id))]


# --- different run extents -------------------------------------------------------------------

def trim(std, k, end):
    """Drop a run's first (end="start") or last k welds and the features beyond its new end."""
    w = welds(std)
    if end == "start":
        return std[std.log_dist_ft >= w.log_dist_ft.iloc[k]]
    return std[std.log_dist_ft <= w.log_dist_ft.iloc[-k - 1]]


@pytest.mark.parametrize("seed, run, end, k", [
    (1, "curr", "start", 5), (1, "curr", "start", 20), (1, "curr", "start", 31), (1, "curr", "start", 60),
    (1, "curr", "end", 5), (1, "curr", "end", 10),
    (2, "prev", "start", 40), (2, "prev", "start", 100), (2, "prev", "end", 10), (3, "prev", "end", 100),
])
def test_runs_covering_different_extents(seed, run, end, k):
    r = dataset(seed)
    base = len(match_anchors(build_joints(r.prev_std), build_joints(r.curr_std), CFG))
    runs = {"prev": r.prev_std, "curr": r.curr_std}
    runs[run] = trim(runs[run], k, end)
    result, sc = score(r, **runs)
    trimmed_away = (build_joints(r.prev_std if run == "prev" else r.curr_std).anchor_type.notna().sum()
                    - build_joints(runs[run]).anchor_type.notna().sum())
    assert result.summary["n_anchor_pairs"] >= base - trimmed_away
    assert sc["joint_accuracy"] >= 0.99, sc["wrong_groups"]
    assert sc["missed_welds_handled"]


# --- missed weld beside an anchor ------------------------------------------------------------

def test_missed_weld_after_anchor_gives_2to1(line):
    cj = build_joints(line.curr_std)
    anchor = int(cj.index[cj.anchor_type.notna()][0])
    curr = drop_weld(line.curr_std, cj.gw_id[anchor + 1])
    result, sc = score(line, curr=curr)
    assert sc["joint_accuracy"] >= 0.99, sc["wrong_groups"]
    assert sc["missed_welds_handled"]
    row = result.weld_map[result.weld_map.curr_gw_id == cj.gw_id[anchor]].iloc[0]
    assert row.match_type == "2:1"


def test_missed_weld_after_anchor_in_short_section():
    r = generate(SynthParams(line_miles=5, seed=1, n_tees=40, n_valves=20))
    cj = build_joints(r.curr_std)
    anchors = cj[cj.anchor_type.notna()].joint_idx.to_numpy()
    gaps = np.diff(anchors)
    k = int(np.argmin(np.where(gaps > 3, gaps, len(cj))))
    curr = drop_weld(r.curr_std, cj.gw_id[anchors[k] + 1])
    _, sc = score(r, curr=curr)
    assert sc["joint_accuracy"] >= 0.995, sc["wrong_groups"]
    assert sc["missed_welds_handled"]


# --- local odometer jump ---------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.3, 0.6])
def test_local_odometer_jump_stays_local(line, frac):
    curr = line.curr_std.copy()
    x = welds(curr).log_dist_ft.quantile(frac) + 10
    curr.loc[curr.log_dist_ft > x, "log_dist_ft"] += 150
    repairs = line.reference["repairs_between_runs"].copy()  # keyed to curr log distance: shift too
    repairs.loc[repairs.start_ft > x, ["start_ft", "end_ft"]] += 150
    result, sc = score(line, curr=curr, repairs=repairs)
    assert sc["joint_accuracy"] >= 0.99, sc["wrong_groups"]
    low = result.weld_map[result.weld_map.confidence == "low"]
    assert len(low) <= 4  # the jumped joint, not its whole section


# --- band ------------------------------------------------------------------------------------

def test_head_offset_beyond_band_is_still_optimal():
    rng = np.random.default_rng(1)
    common = rng.uniform(38, 42, 600)
    k = CFG.band_joints + 10
    prev = np.concatenate([rng.uniform(38, 42, k), common])
    steps = align_section(prev, common, np.zeros(len(prev), bool), np.zeros(600, bool), CFG)
    ok = sum(s.match_type == "1:1" and s.prev_idx[0] - k == s.curr_idx[0] for s in steps)
    assert ok == 600


# --- scorer strictness -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(line):
    result, sc = score(line)
    assert sc["joint_accuracy"] == 1.0
    return result.weld_map


def test_score_rejects_duplicated_rows(line, scored):
    wm = pd.concat([scored, scored.iloc[[10]]], ignore_index=True)
    sc = score_alignment(wm, line.prev_std, line.curr_std, line.weld_truth)
    assert sc["joint_accuracy"] < 1.0


def test_score_rejects_contradictory_extra_rows(line, scored):
    row = scored[scored.match_type == "1:1"].iloc[5]
    extra = pd.DataFrame([{**row.to_dict(), "match_type": "removed", "curr_gw_id": None},
                          {**row.to_dict(), "match_type": "inserted", "prev_gw_id": None}])
    wm = pd.concat([scored, extra], ignore_index=True)
    sc = score_alignment(wm, line.prev_std, line.curr_std, line.weld_truth)
    assert sc["joint_accuracy"] < 1.0


def test_score_rejects_missing_joint(line, scored):
    sc = score_alignment(scored.drop(index=scored.index[3]), line.prev_std, line.curr_std, line.weld_truth)
    assert sc["joint_accuracy"] < 1.0


def test_score_unknown_weld_id_is_value_error(line, scored):
    wm = scored.copy()
    wm.loc[wm.index[0], "curr_gw_id"] = "no-such-weld"
    with pytest.raises(ValueError, match="no-such-weld"):
        score_alignment(wm, line.prev_std, line.curr_std, line.weld_truth)


def test_score_truth_from_other_dataset_is_value_error(line, scored):
    other = generate(SynthParams(line_miles=2, seed=9)).weld_truth
    with pytest.raises(ValueError, match="truth"):
        score_alignment(scored, line.prev_std, line.curr_std, other)


def test_repairs_without_span_columns_is_value_error(line):
    with pytest.raises(ValueError, match="start_ft"):
        align_welds(line.prev_std, line.curr_std, SETTINGS, pd.DataFrame({"x": [1.0]}))


def test_alignment_ignores_recoat_span(line):
    rep = line.reference["repairs_between_runs"]
    recoat = rep[rep.repair_type == "recoat"]
    assert len(recoat) == 1
    with_recoat = align_welds(line.prev_std, line.curr_std, SETTINGS, recoat).weld_map
    without = align_welds(line.prev_std, line.curr_std, SETTINGS, None).weld_map
    pd.testing.assert_frame_equal(with_recoat, without)
