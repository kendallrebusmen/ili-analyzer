"""M12 per-anomaly comparison and explanation codes (hand-built frames)."""
import math
from pathlib import Path

import pandas as pd

from ili.config import load_settings
from ili.parity import EXPLANATIONS, PARITY_COLUMNS, compare, ours_by_feature

SETTINGS = load_settings(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
NAN = float("nan")
OURS_COLS = ["feature_id", "match_id", "log_dist_ft", "depth_ours", "rate_ours", "method_ours", "history_ours",
             "erf_ours", "rl_ours", "rl_governing_ours", "category_ours"]
THEIRS_COLS = ["feature_id", "log_dist_ft", "joint_id", "depth_pct", "prev_depth_pct", "rate_mpy", "method",
               "rate_source", "erf", "rl_yr", "vendor_type"]


def _o(fid, depth=30.0, rate=4.0, method="raw", history="matched", erf=0.9, rl=12.0, rlg="crossing",
       cat="monitor", dist=10.0):
    return dict(feature_id=fid, match_id=f"m{fid}", log_dist_ft=dist, depth_ours=depth, rate_ours=rate,
                method_ours=method, history_ours=history, erf_ours=erf, rl_ours=rl, rl_governing_ours=rlg,
                category_ours=cat)


def _t(fid, depth=30.0, rate=4.0, method="raw", erf=0.9, rl=12.0, vtype="Corrosion", dist=10.0):
    return dict(feature_id=fid, log_dist_ft=dist, joint_id=1, depth_pct=depth, prev_depth_pct=20.0, rate_mpy=rate,
                method=method, rate_source="x", erf=erf, rl_yr=rl, vendor_type=vtype)


def _cmp(ours, theirs, **kw):
    o = pd.DataFrame(ours, columns=OURS_COLS)
    t = pd.DataFrame(theirs, columns=THEIRS_COLS).astype(object)  # the reader returns object dtype
    return compare(o, t, SETTINGS, **kw)


def test_agree():
    out = _cmp([_o("1", rate=4.04)], [_t("1")])
    assert list(out.columns) == list(PARITY_COLUMNS)
    assert out.explanation.tolist() == ["agree"]
    assert out.side.tolist() == ["both"]


def test_each_explanation():
    ours = [
        _o("2"),                                                  # agree
        _o("3", cat="removed"),                                   # removed_after_run (ours only)
        _o("4"),                                                  # not_in_their_set (ours only)
        _o("5", depth=35.0),                                      # depth_differs
        _o("6", history="new", rate=5.0),                   # match_differs
        _o("7", method="joint_percentile", history="new", rate=5.0),           # joint_pool_differs
        _o("8", method="half_life", history="new", rate=5.0),                  # age_basis
        _o("9", rl=500.0, rlg="beyond_horizon"),                  # rl_horizon
        _o("10", erf=0.8),                                        # unexplained
    ]
    theirs = [
        _t("1", vtype="Box"),                                     # vendor_cluster_box (theirs only)
        _t("12"),                                                 # not_in_our_run (theirs only)
        _t("2"),
        _t("5"),
        _t("6", method="raw"),
        _t("7", method="joint_percentile"),
        _t("8", method="half_life"),
        _t("9", rl=612.0),
        _t("10"),
    ]
    out = _cmp(ours, theirs, box_types=["Box"])
    got = dict(zip(out.feature_id, out.explanation))
    assert got == {"1": "vendor_cluster_box", "12": "not_in_our_run", "3": "removed_after_run",
                   "2": "agree", "5": "depth_differs", "6": "match_differs", "7": "joint_pool_differs",
                   "8": "age_basis", "9": "rl_horizon", "10": "unexplained", "4": "not_in_their_set"}
    assert sorted(got.values()) == sorted(EXPLANATIONS) and len(EXPLANATIONS) == 11
    assert list(EXPLANATIONS) == ["vendor_cluster_box", "not_in_our_run", "removed_after_run", "not_in_their_set",
                                  "agree", "depth_differs", "match_differs", "joint_pool_differs", "age_basis",
                                  "rl_horizon", "unexplained"]


def test_order_depth_before_match():
    out = _cmp([_o("1", depth=40.0, history="new")], [_t("1")])
    assert out.explanation.tolist() == ["depth_differs"]


def test_blank_quantity_is_none_not_false():
    ours = [_o("1", rate=NAN, method=None, history=None, rl=NAN, rlg=None), _o("2", erf=0.6)]
    theirs = [_t("1", rate=NAN, method=None, rl=NAN), _t("2", erf=NAN)]
    out = _cmp(ours, theirs).set_index("feature_id")
    assert out.loc["1", "agree_rate"] is None and out.loc["1", "agree_rl"] is None
    assert out.loc["1", "agree_depth"] is True and out.loc["1", "agree_erf"] is True
    assert out.loc["1", "explanation"] == "agree"
    assert out.loc["2", "agree_erf"] is None


def test_remaining_life_tolerance_and_inf():
    inf = math.inf
    out = _cmp([_o("1", rl=100.5), _o("2", rl=inf), _o("3", rl=inf), _o("4", rl=11.95)],
               [_t("1", rl=100.0), _t("2", rl=inf), _t("3", rl=50.0), _t("4", rl=12.0)]).set_index("feature_id")
    assert out.loc["1", "agree_rl"] is True  # 1 % of 100 = 1.0 >= 0.5
    assert out.loc["2", "agree_rl"] is True
    assert out.loc["3", "agree_rl"] is False
    assert out.loc["4", "agree_rl"] is True


def test_sorted_by_distance_then_id():
    out = _cmp([_o("b", dist=5.0), _o("a", dist=9.0)], [_t("b", dist=5.0), _t("a", dist=9.0), _t("c", dist=5.0)])
    assert out.feature_id.tolist() == ["b", "c", "a"]


def test_cluster_members_share_values():
    growth = pd.DataFrame({"match_id": ["m1", "m2"], "curr_feature_id": ["B-1;B-2", "C-1"],
                           "curr_log_dist_ft": [10.0, 20.0], "d_curr_pct": [30.0, 20.0],
                           "governing_mpy": [4.0, 3.0], "governing_method": ["raw", "raw"],
                           "history_class": ["matched", "new"]})
    rl = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "erf": [0.9, 0.8, 0.5], "rl_yr": [12.0, 9.0, NAN],
                       "rl_governing": ["crossing", "crossing", None]})
    cls = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "curr_feature_id": ["B-1;B-2", "C-1", "D-1"],
                        "governing_category": ["monitor", "monitor", "monitor"],
                        "depth_pct_reported": [30.0, 20.0, 7.0]})
    out = ours_by_feature(growth, rl, cls)
    assert sorted(out.feature_id) == ["B-1", "B-2", "C-1", "D-1"]
    b = out.set_index("feature_id")
    assert b.loc["B-1", "rate_ours"] == b.loc["B-2", "rate_ours"] == 4.0
    assert b.loc["B-2", "match_id"] == "m1"
    assert pd.isna(b.loc["D-1", "rate_ours"]) and b.loc["D-1", "depth_ours"] == 7.0 and b.loc["D-1", "erf_ours"] == 0.5


def test_settings_parity_defaults():
    p = SETTINGS.parity
    assert (p.depth_tol_pct, p.rate_tol_mpy, p.erf_tol, p.rl_tol_yr, p.rl_tol_frac) == (0.05, 0.05, 0.005, 0.1, 0.01)


def test_one_sided_blank_is_not_agree():
    out = _cmp([_o("1", erf=0.6), _o("2", rate=NAN, method=None, rl=NAN, rlg=None, history=None)],
               [_t("1", erf=NAN), _t("2")]).set_index("feature_id")
    assert out.loc["1", "agree_erf"] is None and out.loc["1", "explanation"] != "agree"
    assert out.loc["1", "explanation"] == "unexplained"
    assert out.loc["2", "agree_rate"] is None and out.loc["2", "explanation"] != "agree"


def test_ours_only_dent_keeps_position():
    growth = pd.DataFrame({"match_id": ["m1"], "curr_feature_id": ["A"], "curr_log_dist_ft": [10.0],
                           "d_curr_pct": [30.0], "governing_mpy": [4.0], "governing_method": ["raw"],
                           "history_class": ["matched"]})
    rl = pd.DataFrame({"match_id": ["m1", "m2"], "erf": [0.9, 0.5], "rl_yr": [12.0, NAN], "rl_governing": ["x", None]})
    cls = pd.DataFrame({"match_id": ["m1", "m2"], "curr_feature_id": ["A", "D"],
                        "governing_category": ["monitor", "monitor"], "depth_pct_reported": [30.0, 7.0]})
    mt = pd.DataFrame({"match_id": ["m1", "m2"], "curr_log_dist_ft": [10.0, 77.5]})
    out = ours_by_feature(growth, rl, cls, mt).set_index("feature_id")
    assert out.loc["D", "log_dist_ft"] == 77.5 and out.loc["A", "log_dist_ft"] == 10.0
    assert _cmp_from(out.reset_index()).loc[0, "log_dist_ft"] == 77.5


def _cmp_from(ours):
    d = ours[ours.feature_id == "D"]
    return compare(d, pd.DataFrame(columns=THEIRS_COLS), SETTINGS)
