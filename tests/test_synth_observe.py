import numpy as np
import pandas as pd
import pytest

from ili.schema import STANDARD_COLUMNS
from ili.synth import SynthParams, build_features, build_line, observe_run


@pytest.fixture(scope="module")
def obs():
    p = SynthParams()
    rng = np.random.default_rng(p.seed)
    line = build_line(p, rng)
    f = build_features(p, line, rng)
    prev = observe_run("prev", p, line, f, rng)
    missed_prev = set(prev.weld_link.loc[prev.weld_link.weld_status == "missed", "true_weld_idx"])
    curr = observe_run("curr", p, line, f, rng, exclude_welds=missed_prev)
    return p, line, f, {"prev": prev, "curr": curr}


def _ml(std):
    return std[std.feature_type == "metal_loss"]


def test_standard_columns(obs):
    *_, runs = obs
    assert list(runs["prev"].std.columns) == STANDARD_COLUMNS
    assert runs["prev"].std.run_id.unique().tolist() == ["2021_vendorA"]
    assert runs["curr"].std.run_id.unique().tolist() == ["2026_vendorB"]


def test_feature_ids(obs):
    *_, runs = obs
    a, b = runs["prev"].std, runs["curr"].std
    assert a.feature_id.is_unique and a.feature_id.str.match(r"^A-\d{6}$").all()
    assert b.feature_id.is_unique and b.feature_id.str.match(r"^B-\d{6}$").all()
    assert a.log_dist_ft.is_monotonic_increasing


def test_weld_ids(obs):
    *_, runs = obs
    gw_a = runs["prev"].std.query("feature_type == 'girth_weld'").us_gw_id
    assert gw_a.str.match(r"^GW\d{5}$").all() and gw_a.iloc[0] == "GW00010"
    gw_b = runs["curr"].std.query("feature_type == 'girth_weld'").us_gw_id
    assert gw_b.tolist()[:3] == ["1", "2", "3"]


def test_threshold(obs):
    p, *_, runs = obs
    for r in runs.values():
        assert (_ml(r.std).depth_pct >= p.reporting_threshold_pct).all()


def test_500_reported_prev(obs):
    _, _, f, runs = obs
    link = runs["prev"].feat_link.merge(f[["true_id", "population"]], on="true_id")
    assert ((link.population == "ext_ml") & (link.status == "reported")).sum() == 500


def test_depth_noise_sigma(obs):
    _, _, f, runs = obs
    for which, col in [("prev", "depth_prev_true_pct"), ("curr", "depth_curr_true_pct")]:
        r = runs[which]
        link = r.feat_link[r.feat_link.status == "reported"]
        m = link.merge(f, on="true_id").merge(r.std[["feature_id", "depth_pct"]], on="feature_id")
        m = m[(m.feature_type == "metal_loss") & (m[col] >= 30) & (m[col] <= 70)]
        assert len(m) > 100
        assert (m.depth_pct - m[col]).std() == pytest.approx(7.8, abs=0.8)


def test_odometer_drift(obs):
    _, line, _, runs = obs
    for r in runs.values():
        link = r.weld_link[r.weld_link.weld_status == "reported"]
        chain = line.welds.set_index("true_weld_idx").chainage_ft.loc[link.true_weld_idx]
        gw = r.std[r.std.feature_type == "girth_weld"].set_index("us_gw_id").log_dist_ft
        slope = np.polyfit(chain.to_numpy(), gw.loc[link.gw_id].to_numpy(), 1)[0]
        assert 0.995 <= slope <= 1.005
        assert abs(slope - (1 + r.drift)) < 1e-4
    assert runs["prev"].drift == 0.0


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_curr_drift_is_meaningful(seed):
    """Drift is 'up to 0.5 %' but never negligible, so alignment is always exercised."""
    p = SynthParams(seed=seed, line_miles=2)
    rng = np.random.default_rng(seed)
    line = build_line(p, rng)
    f = build_features(p, line, rng)
    observe_run("prev", p, line, f, rng)
    curr = observe_run("curr", p, line, f, rng)
    assert 0.0025 <= abs(curr.drift) <= 0.005


def test_missed_welds(obs):
    p, line, _, runs = obs
    missed = {}
    for which, r in runs.items():
        wl = r.weld_link
        missed[which] = set(wl.loc[wl.weld_status == "missed", "true_weld_idx"])
        assert len(missed[which]) == p.n_missed_welds
        n_gw = (r.std.feature_type == "girth_weld").sum()
        assert n_gw == (wl.weld_status == "reported").sum()
    assert not missed["prev"] & missed["curr"]
    lo, hi = line.replacement
    chain = line.welds.set_index("true_weld_idx").chainage_ft
    for idx in missed["prev"] | missed["curr"]:
        assert not lo <= chain.loc[idx] <= hi


def test_not_present_welds(obs):
    _, line, _, runs = obs
    for which in ("prev", "curr"):
        wl = runs[which].weld_link.set_index("true_weld_idx")
        absent = line.welds.loc[~line.welds[f"in_{which}"], "true_weld_idx"]
        assert (wl.weld_status.loc[absent] == "not_present").all()


def test_feature_near_missed_weld(obs):
    _, line, f, runs = obs
    welds = line.welds.set_index("true_weld_idx").chainage_ft
    for r in runs.values():
        missed = r.weld_link.loc[r.weld_link.weld_status == "missed", "true_weld_idx"]
        checked = 0
        for idx in missed:
            nxt = welds.index[welds.index > idx][0]
            inside = f[(f.chainage_ft > welds.loc[idx]) & (f.chainage_ft < welds.loc[nxt])]
            m = inside.merge(r.feat_link[r.feat_link.status == "reported"], on="true_id")
            m = m.merge(r.std[["feature_id", "us_gw_id", "dist_from_us_gw_ft", "joint_length_ft"]], on="feature_id")
            for row in m.itertuples():
                true_dist = row.chainage_ft - welds.loc[idx]
                assert row.dist_from_us_gw_ft > true_dist + 3.0  # measured from the weld before the missed one
                assert row.joint_length_ft > 45  # two joints merged
                checked += 1
        assert checked >= 1


def test_ranges(obs):
    *_, runs = obs
    for r in runs.values():
        s = r.std
        feat = s[s.feature_type != "girth_weld"]
        assert s.depth_pct.dropna().between(0, 100).all()
        assert (feat.length_in.dropna() > 0).all() and (feat.width_in.dropna() > 0).all()
        assert (feat.dist_from_us_gw_ft >= 0).all()
        assert (feat.dist_from_us_gw_ft <= feat.joint_length_ft + 1e-9).all()
        assert s.clock_min.dropna().between(0, 719).all()


def test_log_dist_consistent_with_weld(obs):
    *_, runs = obs
    for r in runs.values():
        s = r.std
        gw = s[s.feature_type == "girth_weld"].set_index("us_gw_id").log_dist_ft
        feat = s[s.feature_type != "girth_weld"]
        assert np.allclose(feat.log_dist_ft, gw.loc[feat.us_gw_id].to_numpy() + feat.dist_from_us_gw_ft)


def test_welds_monotonic(obs):
    *_, runs = obs
    for r in runs.values():
        gw = r.std[r.std.feature_type == "girth_weld"].log_dist_ft.to_numpy()
        assert (np.diff(gw) > 0).all()


def test_clusters(obs):
    p, _, f, runs = obs
    pits = f[f.population == "cluster_pit"]
    for site, group in pits.groupby("cluster_site"):
        clustered_in = []
        for which, r in runs.items():
            link = r.feat_link.set_index("true_id").loc[group.true_id]
            if (link.status == "reported_in_cluster").any():
                assert link.feature_id.nunique() == 1
                row = r.std.set_index("feature_id").loc[link.feature_id.iloc[0]]
                assert row.comments == "CLUSTER" and pd.notna(row.vendor_cluster_id)
                clustered_in.append(which)
        assert clustered_in == (["prev"] if site % 2 == 0 else ["curr"])


def test_replaced_features_removed(obs):
    _, line, f, runs = obs
    replaced = set(line.joints.loc[line.joints.in_prev & ~line.joints.in_curr, "joint_key"])
    ids = f.loc[f.joint_key.isin(replaced), "true_id"]
    curr = runs["curr"].feat_link.set_index("true_id").loc[ids]
    assert (curr.status == "removed_by_repair").all() and curr.feature_id.isna().all()


def test_anchors_and_dents_always_reported(obs):
    _, _, f, runs = obs
    ids = f.loc[f.population.isin(["anchor", "dent"]), "true_id"]
    for r in runs.values():
        assert (r.feat_link.set_index("true_id").loc[ids].status == "reported").all()


def test_every_true_feature_linked(obs):
    _, _, f, runs = obs
    for r in runs.values():
        assert set(r.feat_link.true_id) == set(f.true_id)
        assert r.feat_link.status.isin(
            ["reported", "reported_in_cluster", "below_threshold", "removed_by_repair", "not_present"]
        ).all()
