import numpy as np
import pytest

from ili.align_welds import Step, align_section
from ili.config import load_settings

CFG = load_settings().alignment


def run(prev, curr, prev_rep=None, curr_rep=None):
    prev, curr = np.asarray(prev, float), np.asarray(curr, float)
    prev_rep = np.zeros(len(prev), bool) if prev_rep is None else np.asarray(prev_rep, bool)
    curr_rep = np.zeros(len(curr), bool) if curr_rep is None else np.asarray(curr_rep, bool)
    return align_section(prev, curr, prev_rep, curr_rep, CFG)


def shape(steps):
    return [(s.match_type, s.prev_idx, s.curr_idx) for s in steps]


def assert_each_joint_once(steps, n, m):
    prev_used = [i for s in steps for i in s.prev_idx]
    curr_used = [j for s in steps for j in s.curr_idx]
    assert prev_used == list(range(n)) and curr_used == list(range(m))


def test_identical_sequences_all_1to1():
    steps = run([40, 39, 41, 38], [40, 39, 41, 38])
    assert shape(steps) == [("1:1", (i,), (i,)) for i in range(4)]
    assert all(s.mismatch == 0 for s in steps)
    assert isinstance(steps[0], Step)


def test_small_noise_still_1to1():
    steps = run([40, 39, 41, 38], [40.1, 38.9, 41.05, 38.2])
    assert [s.match_type for s in steps] == ["1:1"] * 4
    assert max(s.mismatch for s in steps) < 1


def test_curr_missed_weld_gives_2to1():
    steps = run([40, 39, 41, 38], [40, 80, 38])
    assert shape(steps) == [("1:1", (0,), (0,)), ("2:1", (1, 2), (1,)), ("1:1", (3,), (2,))]


def test_prev_missed_weld_gives_1to2():
    steps = run([40, 80, 38], [40, 39, 41, 38])
    assert shape(steps) == [("1:1", (0,), (0,)), ("1:2", (1,), (1, 2)), ("1:1", (2,), (3,))]


def test_inserted_and_removed_outside_repair():
    steps = run([40, 39, 41, 38], [40, 39, 25, 41, 38])
    assert ("inserted", (), (2,)) in shape(steps)
    assert_each_joint_once(steps, 4, 5)
    steps = run([40, 39, 25, 41, 38], [40, 39, 41, 38])
    assert ("removed", (2,), ()) in shape(steps)


def test_repair_zone_prefers_gaps():
    steps = run([40, 41, 39], [30, 31, 29, 30], [1, 1, 1], [1, 1, 1, 1])
    types = [s.match_type for s in steps]
    assert types.count("removed") == 3 and types.count("inserted") == 4
    assert all(s.cost == CFG.gap_penalty_repair for s in steps)


def test_offset_start():
    steps = run([41.7, 40, 39, 41, 38], [40, 39, 41, 38])
    assert shape(steps)[0] == ("removed", (0,), ())
    assert [s.match_type for s in steps[1:]] == ["1:1"] * 4


def test_offset_end():
    steps = run([40, 39, 41, 38], [40, 39, 41, 38, 37.3])
    assert shape(steps)[-1] == ("inserted", (), (4,))


def test_empty_side():
    assert shape(run([], [40, 39])) == [("inserted", (), (0,)), ("inserted", (), (1,))]
    assert shape(run([40], [])) == [("removed", (0,), ())]
    assert run([], []) == []


def test_double_missed_weld_no_crash():
    steps = run([40, 39, 41, 38, 42], [40, 118, 42])
    assert_each_joint_once(steps, 5, 3)


def test_zero_band_still_spans_the_start_to_end_offset():
    """With band 0 the search still covers the start- and end-aligned diagonals, so the
    9-joint offset between them is searched and the optimum found."""
    narrow = CFG.model_copy(update={"band_joints": 0})
    prev = np.array([40.0])
    curr = np.array([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 40])  # no pair sums to ~40
    steps = align_section(prev, curr, np.zeros(1, bool), np.zeros(10, bool), narrow)
    assert shape(steps)[-1] == ("1:1", (0,), (9,))
    assert [s.match_type for s in steps].count("inserted") == 9


def test_ties_are_deterministic():
    a = run([40, 40, 40, 40], [40, 40, 40])
    b = run([40, 40, 40, 40], [40, 40, 40])
    assert shape(a) == shape(b)


@pytest.mark.parametrize("seed", range(20))
def test_every_joint_used_once(seed):
    rng = np.random.default_rng(seed)
    prev = list(rng.uniform(38, 42, rng.integers(0, 60)))
    curr = list(prev)
    for _ in range(rng.integers(0, 4)):
        if len(curr) > 2:
            k = int(rng.integers(0, len(curr) - 1))
            op = rng.integers(0, 3)
            if op == 0:
                curr[k:k + 2] = [curr[k] + curr[k + 1]]  # missed weld
            elif op == 1:
                curr.insert(k, float(rng.uniform(10, 30)))  # inserted joint
            else:
                del curr[k]  # removed joint
    steps = run(prev, curr)
    assert_each_joint_once(steps, len(prev), len(curr))
    assert all(s.match_type in {"1:1", "2:1", "1:2", "inserted", "removed"} for s in steps)


def test_missed_weld_recovered_in_long_random_section():
    rng = np.random.default_rng(3)
    prev = rng.uniform(38, 42, 200)
    curr = list(prev + rng.normal(0, 0.07, 200))
    curr[100:102] = [curr[100] + curr[101]]
    steps = run(prev, curr)
    assert ("2:1", (100, 101), (100,)) in shape(steps)
    assert sum(s.match_type == "1:1" for s in steps) == 198


def test_free_ends_prefer_gaps_over_coincidental_matches():
    """prev ends partway through curr. With charged end gaps the DP would rather pair prev's
    last joints with later curr joints of similar length than leave curr's tail unmatched."""
    prev = [40.0, 38.0, 41.0, 39.5, 42.0]
    curr = [40.0, 38.0, 41.0, 81.5, 39.6, 42.1, 40.3]  # prev 3+4 missed in curr (2:1), then 3 extra joints
    z = np.zeros
    charged = align_section(np.array(prev), np.array(curr), z(5, bool), z(7, bool), CFG)
    free = align_section(np.array(prev), np.array(curr), z(5, bool), z(7, bool), CFG, free_end=True)
    assert ("2:1", (3, 4), (3,)) not in shape(charged)
    assert shape(free)[3] == ("2:1", (3, 4), (3,))
    assert [s.match_type for s in free[4:]] == ["inserted"] * 3 and all(s.cost == 0 for s in free[4:])
    assert_each_joint_once(free, 5, 7)


def test_free_start_skips_leading_extra_joints():
    curr = [40.0, 38.0, 41.0]
    prev = [39.0, 41.5, 40.0, 38.0, 41.0]
    steps = align_section(np.array(prev), np.array(curr), np.zeros(5, bool), np.zeros(3, bool), CFG,
                          free_start=True)
    assert shape(steps) == [("removed", (0,), ()), ("removed", (1,), ()),
                            ("1:1", (2,), (0,)), ("1:1", (3,), (1,)), ("1:1", (4,), (2,))]
    assert steps[0].cost == steps[1].cost == 0
