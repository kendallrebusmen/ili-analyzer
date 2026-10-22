import numpy as np
import pytest

from ili.synth import SynthParams, build_features, build_line, delta_t_years, run_dates


@pytest.fixture(scope="module")
def setup():
    p = SynthParams()
    rng = np.random.default_rng(p.seed)
    line = build_line(p, rng)
    return p, line, build_features(p, line, rng)


def test_run_dates():
    prev, curr = run_dates(SynthParams())
    assert prev.isoformat() == "2021-06-15"
    assert (curr - prev).days == round(5 * 365.25)
    assert delta_t_years(SynthParams()) == pytest.approx(5.0, abs=0.01)


def test_population_counts(setup):
    _, line, f = setup
    counts = f.population.value_counts()
    assert counts["ext_ml"] == 500
    assert counts["new"] == 50
    assert counts["severe"] == 5
    assert counts["internal"] == 8
    assert counts["dent"] == 12
    assert counts["anchor"] == line.joints.fitting.notna().sum()
    pits = f[f.population == "cluster_pit"]
    assert pits.cluster_site.nunique() == 6
    assert pits.groupby("cluster_site").size().between(2, 4).all()


def test_true_ids_unique_and_sorted(setup):
    _, _, f = setup
    assert f.true_id.is_unique
    assert f.chainage_ft.is_monotonic_increasing


def test_growth_distribution(setup):
    _, _, f = setup
    g = f.loc[f.population == "ext_ml", "growth_mpy"]
    assert g.between(0, 10).all()
    assert g.median() < 2.0
    assert g.max() > 5


def test_depth_growth_consistent(setup):
    p, line, f = setup
    ml = f[f.feature_type == "metal_loss"]
    wt = line.joints.set_index("joint_key").wt_nom_in.loc[ml.joint_key].to_numpy()
    expected = ml.depth_prev_true_pct + ml.growth_mpy / 1000 * delta_t_years(p) / wt * 100
    assert np.allclose(ml.depth_curr_true_pct, expected.clip(upper=99.0))


def test_depths_in_range(setup):
    _, _, f = setup
    for col in ["depth_prev_true_pct", "depth_curr_true_pct", "depth_prev_meas_pct", "depth_curr_meas_pct"]:
        vals = f[col].dropna()
        assert vals.between(0, 100).all(), col
    assert (f.loc[f.feature_type == "metal_loss", "depth_prev_true_pct"] < 100).all()


def test_prev_reporting_draws(setup):
    p, _, f = setup
    thr = p.reporting_threshold_pct
    ext = f[f.population == "ext_ml"]
    assert (ext.depth_prev_meas_pct >= thr).all()
    new = f[f.population == "new"]
    assert (new.depth_prev_meas_pct < thr).all()
    assert (new.depth_curr_meas_pct >= thr).all()


def test_severe_exceed_80(setup):
    _, _, f = setup
    assert (f.loc[f.population == "severe", "depth_curr_true_pct"] > 80).all()


def test_dent_positions(setup):
    _, _, f = setup
    d = f[f.population == "dent"]
    clock = d.clock_min.astype(int)
    top = (clock >= 480) | (clock <= 240)
    assert top.sum() >= 3 and (~top).sum() >= 3
    assert d.dent_metal_loss.sum() == 4
    assert d[d.dent_metal_loss.astype(bool)].depth_prev_true_pct.notna().all()
    pct_od = d.dent_depth_in / 12.75 * 100
    assert pct_od.min() < 2 and pct_od.max() > 6
    assert (d.growth_mpy == 0).all()


def test_features_in_replaced_section(setup):
    _, line, f = setup
    replaced = set(line.joints.loc[line.joints.in_prev & ~line.joints.in_curr, "joint_key"])
    in_rep = f[f.joint_key.isin(replaced)]
    assert len(in_rep) >= 3
    assert (in_rep.population == "ext_ml").all()


def test_no_features_on_new_joints(setup):
    _, line, f = setup
    new_joints = set(line.joints.loc[~line.joints.in_prev, "joint_key"])
    assert not f.joint_key.isin(new_joints).any()


def test_features_inside_their_joint(setup):
    _, line, f = setup
    j = line.joints.set_index("joint_key").loc[f.joint_key]
    assert (f.chainage_ft.to_numpy() > j.start_ft.to_numpy()).all()
    assert (f.chainage_ft.to_numpy() < (j.start_ft + j.length_ft).to_numpy()).all()


def test_anchors_in_fitting_joints(setup):
    _, line, f = setup
    a = f[f.population == "anchor"]
    fit = line.joints.set_index("joint_key").fitting.loc[a.joint_key]
    assert (fit.to_numpy() == a.feature_type.to_numpy()).all()


def test_cluster_pits_within_interaction(setup):
    _, _, f = setup
    for _, site in f[f.population == "cluster_pit"].groupby("cluster_site"):
        gaps_in = np.diff(np.sort(site.chainage_ft.to_numpy())) * 12
        assert gaps_in.max() <= 1.0
        assert site.joint_key.nunique() == 1


def test_seam_interaction_flag(setup):
    p, _, f = setup
    ml = f[f.feature_type == "metal_loss"]
    d = np.abs(ml.clock_min.astype(int) - p.seam_clock_min)
    near = np.minimum(d, 720 - d) <= 10
    assert (ml.seam_interaction.astype(bool) == near).all()


def test_unsatisfiable_population_raises():
    p = SynthParams(line_miles=2, reporting_threshold_pct=99.9)
    rng = np.random.default_rng(1)
    line = build_line(p, rng)
    with pytest.raises(ValueError, match="n_ext_ml_prev"):
        build_features(p, line, rng)
