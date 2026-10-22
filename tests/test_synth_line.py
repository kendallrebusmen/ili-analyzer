import numpy as np
import pytest

from ili.synth import SynthParams, build_line


@pytest.fixture(scope="module")
def line():
    p = SynthParams()
    return build_line(p, np.random.default_rng(p.seed))


def test_line_length(line):
    assert line.welds.chainage_ft.iloc[-1] >= 20 * 5280
    assert line.length_ft == pytest.approx(line.welds.chainage_ft.iloc[-1])


def test_joint_lengths(line):
    pipe = line.joints[line.joints.fitting.isna()]
    share = pipe.length_ft.between(38, 42).mean()
    assert share >= 0.90
    assert (pipe.length_ft < 38).any(), "expected some short joints"


def test_fittings(line):
    counts = line.joints.fitting.value_counts()
    assert counts["valve"] == 4
    assert counts["tee"] == 6
    fit = line.joints[line.joints.fitting.notna()]
    assert fit.length_ft.between(3, 8).all()


def test_replacement(line):
    j = line.joints
    removed = j[j.in_prev & ~j.in_curr]
    inserted = j[~j.in_prev & j.in_curr]
    assert len(removed) == 3
    assert len(inserted) == 4
    assert removed.length_ft.sum() == pytest.approx(inserted.length_ft.sum(), abs=1e-6)
    lo, hi = line.replacement
    assert removed.start_ft.min() == pytest.approx(lo)
    assert (removed.start_ft + removed.length_ft).max() == pytest.approx(hi)
    assert removed.fitting.isna().all()


def test_replacement_welds(line):
    w = line.welds
    assert (~w.in_curr).sum() == 2, "two interior old welds removed"
    assert (~w.in_prev).sum() == 3, "three interior new welds added"
    lo, hi = line.replacement
    inside = w.chainage_ft.between(lo, hi, inclusive="neither")
    assert (inside == (~w.in_curr | ~w.in_prev)).all()


def test_joint_tiling(line):
    """In each run the existing joints tile the line exactly."""
    for run in ("in_prev", "in_curr"):
        j = line.joints[line.joints[run]].sort_values("start_ft")
        ends = (j.start_ft + j.length_ft).to_numpy()
        assert np.allclose(j.start_ft.to_numpy()[1:], ends[:-1])
        assert j.start_ft.iloc[0] == 0
        assert ends[-1] == pytest.approx(line.length_ft)


def test_us_weld_matches_joint_start(line):
    w = line.welds.set_index("true_weld_idx").chainage_ft
    assert np.allclose(w.loc[line.joints.us_weld_idx].to_numpy(), line.joints.start_ft.to_numpy())


def test_weld_chainage_monotonic(line):
    c = line.welds.chainage_ft.to_numpy()
    assert (np.diff(c) > 0).all()
    assert (line.welds.true_weld_idx.to_numpy() == np.arange(len(c))).all()


def test_wall_thickness(line):
    assert set(line.joints.wt_nom_in.unique()) == {0.250, 0.312}
    seg = line.pipe_segments
    assert seg.start_ft.iloc[0] == 0
    assert seg.end_ft.iloc[-1] == pytest.approx(line.length_ft)
    assert np.allclose(seg.start_ft.to_numpy()[1:], seg.end_ft.to_numpy()[:-1])


def test_too_short_raises():
    p = SynthParams(line_miles=0.1)
    with pytest.raises(ValueError, match="line_miles"):
        build_line(p, np.random.default_rng(1))


def test_seed_reproducible():
    p = SynthParams(line_miles=2)
    a = build_line(p, np.random.default_rng(5))
    b = build_line(p, np.random.default_rng(5))
    assert a.joints.equals(b.joints) and a.welds.equals(b.welds)
