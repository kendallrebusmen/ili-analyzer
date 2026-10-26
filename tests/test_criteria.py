"""Threshold tests for the 49 CFR 195.452(h)(4) rule set (spec 6.8, 8)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ili.config import DEFAULT_CRITERIA, load_repair_criteria, load_settings
from ili.rules import compile_condition
from ili.vocabulary import VARIABLES

HCA_IDS = ["IMM-1", "IMM-2", "IMM-3", "IMM-4", "IMM-5", "D60-1", "D60-2"] + [f"D180-{i}" for i in range(1, 10)]

CRITERIA = load_repair_criteria()
CONDITIONS = {r.id: compile_condition(r.condition, set(VARIABLES)) for r in CRITERIA.rules}


def base(**over):
    """A full variable dict: a benign 16-in pipe, nothing triggered."""
    v = dict(
        feature_type="metal_loss", could_affect_hca=True, top_of_pipe=False, dent_metal_loss=False,
        seam_interaction=False, near_girth_weld=False, near_foreign_crossing=False,
        general_corrosion=False, widespread_circumferential=False, dig_required_by_remaining_life=False,
        manual_immediate=False, field_crack_confirmed=False, od_in=16.0, wt_in=0.25, clock_min=360.0,
        depth_pct_reported=10.0, depth_pct_assessed=10.0, length_in=2.0, width_in=2.0, p_fail=1500.0,
        p_safe=1200.0, mop_local=1000.0, dent_depth_in=0.0, dent_depth_pct_od=0.0,
    )
    assert set(v) == set(VARIABLES)
    v.update(over)
    return v


def ev(rule_id, **over):
    return CONDITIONS[rule_id].evaluate(base(**over))


def write_criteria(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CRITERIA.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_criteria_loads_all_hca_rules():
    hca = [r.id for r in CRITERIA.rules if r.applies_to == "hca"]
    assert hca == HCA_IDS


def test_missing_rule_id_is_error(tmp_path):
    def drop(raw):
        raw["rules"] = [r for r in raw["rules"] if r["id"] != "D180-9"]

    with pytest.raises(ValidationError, match="D180-9"):
        load_repair_criteria(write_criteria(tmp_path, drop))


def test_unknown_variable_in_condition_is_error(tmp_path):
    def bad(raw):
        raw["rules"][0]["condition"] = "depth_pct_made_up > 80"

    with pytest.raises(ValidationError, match="IMM-1"):
        load_repair_criteria(write_criteria(tmp_path, bad))


def test_vocabulary_has_every_published_name():
    expected = {
        "feature_type", "could_affect_hca", "top_of_pipe", "dent_metal_loss", "seam_interaction",
        "near_girth_weld", "near_foreign_crossing", "general_corrosion", "widespread_circumferential",
        "dig_required_by_remaining_life", "manual_immediate", "field_crack_confirmed", "od_in", "wt_in",
        "clock_min", "depth_pct_reported", "depth_pct_assessed", "length_in", "width_in", "p_fail",
        "p_safe", "mop_local", "dent_depth_in", "dent_depth_pct_od",
    }
    assert set(VARIABLES) == expected
    assert all(isinstance(m, str) and m for m in VARIABLES.values())


# (rule, driving variable, fixed context, threshold)
STRICT = [
    ("IMM-1", "depth_pct_assessed", {}, 80.0),
    ("IMM-4", "dent_depth_pct_od", {"feature_type": "dent", "top_of_pipe": True}, 6.0),
    ("D60-1", "dent_depth_pct_od", {"feature_type": "dent", "top_of_pipe": True, "od_in": 12.75}, 3.0),
    ("D60-1", "dent_depth_pct_od", {"feature_type": "dent", "top_of_pipe": True, "od_in": 24.0}, 3.0),
    ("D60-1", "dent_depth_in", {"feature_type": "dent", "top_of_pipe": True, "od_in": 10.75}, 0.250),
    ("D180-1", "dent_depth_pct_od", {"feature_type": "dent", "near_girth_weld": True, "od_in": 16.0}, 2.0),
    ("D180-1", "dent_depth_pct_od", {"feature_type": "dent", "seam_interaction": True, "od_in": 12.75}, 2.0),
    ("D180-1", "dent_depth_in", {"feature_type": "dent", "near_girth_weld": True, "od_in": 10.75}, 0.250),
    ("D180-2", "dent_depth_pct_od", {"feature_type": "dent", "top_of_pipe": True, "od_in": 16.0}, 2.0),
    ("D180-2", "dent_depth_in", {"feature_type": "dent", "top_of_pipe": True, "od_in": 8.625}, 0.250),
    ("D180-3", "dent_depth_pct_od", {"feature_type": "dent", "top_of_pipe": False}, 6.0),
    ("D180-5", "depth_pct_assessed", {"general_corrosion": True}, 50.0),
    ("D180-6", "depth_pct_assessed", {"near_foreign_crossing": True}, 50.0),
    ("D180-6", "depth_pct_assessed", {"widespread_circumferential": True}, 50.0),
    ("D180-6", "depth_pct_assessed", {"near_girth_weld": True}, 50.0),
    ("D180-9", "depth_pct_assessed", {"feature_type": "gouge"}, 12.5),
    ("IMM-2", "p_fail", {"mop_local": 1000.0}, 1000.0),
    ("D180-4", "p_safe", {"mop_local": 1000.0}, 1000.0),
]


@pytest.mark.parametrize("rule_id,var,ctx,thr", STRICT)
def test_rule_thresholds(rule_id, var, ctx, thr):
    eps = 0.0001 if var == "dent_depth_in" else 0.01
    below, at, above = (thr - eps, thr, thr + eps)
    if var in ("p_fail", "p_safe"):  # fires BELOW the threshold (strictly less than MOP)
        assert ev(rule_id, **ctx, **{var: below}) is True
        assert ev(rule_id, **ctx, **{var: at}) is False
        assert ev(rule_id, **ctx, **{var: above}) is False
    else:
        assert ev(rule_id, **ctx, **{var: below}) is False
        assert ev(rule_id, **ctx, **{var: at}) is False
        assert ev(rule_id, **ctx, **{var: above}) is True


def test_nps12_boundary_selects_percent_or_inches():
    dent = dict(feature_type="dent", top_of_pipe=True)
    # At OD 12.75 (NPS 12) the percent test governs; the 0.250 in alternative does not.
    assert ev("D60-1", **dent, od_in=12.75, dent_depth_pct_od=1.0, dent_depth_in=0.9) is False
    # Just below NPS 12 the inch test governs; the percent figure is ignored.
    assert ev("D60-1", **dent, od_in=12.74, dent_depth_pct_od=9.0, dent_depth_in=0.2) is False
    assert ev("D60-1", **dent, od_in=12.74, dent_depth_pct_od=0.1, dent_depth_in=0.26) is True


@pytest.mark.parametrize("rule_id,flag,ctx", [
    ("IMM-3", "dent_metal_loss", {"feature_type": "dent", "top_of_pipe": True}),
    ("D60-2", "dent_metal_loss", {"feature_type": "dent", "top_of_pipe": False}),
    ("IMM-5", "manual_immediate", {}),
    ("D180-7", "field_crack_confirmed", {}),
    ("D180-8", "seam_interaction", {"feature_type": "metal_loss"}),
])
def test_flag_driven_rules(rule_id, flag, ctx):
    assert ev(rule_id, **ctx, **{flag: True}) is True
    assert ev(rule_id, **ctx, **{flag: False}) is False
    assert ev(rule_id, **ctx, **{flag: None}) is None


def test_position_gates():
    assert ev("IMM-3", dent_metal_loss=True, top_of_pipe=False) is False
    assert ev("IMM-4", dent_depth_pct_od=7.0, top_of_pipe=False) is False
    assert ev("D180-2", dent_depth_pct_od=3.0, top_of_pipe=False) is False
    assert ev("D60-2", dent_metal_loss=True, top_of_pipe=True) is False
    assert ev("D180-3", dent_depth_pct_od=7.0, top_of_pipe=True) is False
    # the clock -> top_of_pipe mapping (240/241/479/480) is Task 3's; here only the variable matters
    for rid in ("IMM-3", "IMM-4", "D60-1", "D60-2", "D180-2", "D180-3"):
        rule = next(r for r in CRITERIA.rules if r.id == rid)
        assert "top_of_pipe" in rule.condition
        assert "clock_min" not in rule.condition


def test_feature_type_gates():
    assert ev("D180-8", seam_interaction=True, feature_type="dent") is False
    assert ev("D180-9", feature_type="metal_loss", depth_pct_assessed=50.0) is False


def test_d180_5_d180_6_gates():
    assert ev("D180-5", general_corrosion=False, depth_pct_assessed=90.0) is False
    assert ev("D180-6", depth_pct_assessed=90.0) is False  # none of the three locations


UNKNOWN = [
    ("IMM-1", {"depth_pct_assessed": None}),
    ("IMM-2", {"p_fail": None}),
    ("IMM-2", {"mop_local": None}),
    ("IMM-3", {"dent_metal_loss": None, "top_of_pipe": True}),
    ("IMM-4", {"dent_depth_pct_od": None, "top_of_pipe": True}),
    ("IMM-4", {"top_of_pipe": None, "dent_depth_pct_od": 7.0}),
    ("D60-1", {"od_in": None, "top_of_pipe": True, "dent_depth_pct_od": 4.0, "dent_depth_in": 0.5}),
    ("D60-2", {"top_of_pipe": None, "dent_metal_loss": True}),
    ("D180-1", {"near_girth_weld": None, "dent_depth_pct_od": 3.0}),
    ("D180-2", {"dent_depth_pct_od": None, "top_of_pipe": True, "od_in": 16.0}),
    ("D180-3", {"dent_depth_pct_od": None, "top_of_pipe": False}),
    ("D180-4", {"p_safe": None}),
    ("D180-5", {"depth_pct_assessed": None, "general_corrosion": True}),
    ("D180-6", {"depth_pct_assessed": None, "near_girth_weld": True}),
    ("D180-6", {"near_girth_weld": None, "depth_pct_assessed": 60.0}),
    ("D180-9", {"depth_pct_assessed": None, "feature_type": "gouge"}),
]


@pytest.mark.parametrize("rule_id,over", UNKNOWN)
def test_unknown_input_gives_none(rule_id, over):
    assert ev(rule_id, **over) is None


def test_unknown_does_not_mask_a_definite_false():
    assert ev("D180-5", general_corrosion=False, depth_pct_assessed=None) is False
    assert ev("IMM-4", top_of_pipe=False, dent_depth_pct_od=None) is False


def test_feature_types_per_rule():
    ft = {r.id: [str(f) for f in r.feature_types] for r in CRITERIA.rules if r.applies_to == "hca"}
    for rid in ("IMM-1", "IMM-2", "D180-4", "D180-5", "D180-6", "D180-8"):
        assert ft[rid] == ["metal_loss"]
    for rid in ("IMM-3", "IMM-4", "D60-1", "D60-2", "D180-1", "D180-2", "D180-3"):
        assert ft[rid] == ["dent"]
    assert ft["D180-9"] == ["gouge"]
    assert ft["IMM-5"] == ft["D180-7"] == ["metal_loss", "dent", "gouge"]


def test_requires_review_follows_spec():
    review = {r.id for r in CRITERIA.rules if r.applies_to == "hca" and r.requires_review}
    assert review == {"D180-1", "D180-5", "D180-6", "D180-7", "D180-8"}


def test_categories():
    cat = {r.id: r.category for r in CRITERIA.rules}
    for i in range(1, 6):
        assert cat[f"IMM-{i}"] == "immediate"
    assert cat["D60-1"] == cat["D60-2"] == "60_day"
    for i in range(1, 10):
        assert cat[f"D180-{i}"] == "180_day"


def test_non_hca_rules_mirror_hca_and_require_review():
    hca = {r.id: r for r in CRITERIA.rules if r.applies_to == "hca"}
    nh = {r.id: r for r in CRITERIA.rules if r.applies_to == "non_hca"}
    assert set(nh) == {f"NH-{i}" for i in hca}
    for rid, h in hca.items():
        n = nh[f"NH-{rid}"]
        assert (n.condition, n.feature_types, n.category) == (h.condition, h.feature_types, h.category)
        assert n.requires_review is True
        assert n.description.startswith("PLACEHOLDER pending IMP non-HCA criteria: ")
        assert "PLACEHOLDER" in n.citation and "195.416(g)" in n.citation


def test_classification_settings():
    c = load_settings().classification
    for name in ("seam_clock_min", "seam_clock_tol_min", "general_corrosion_min_dim_in", "widespread_circ_min_width_frac"):
        assert getattr(c, name) is None
    assert c.due_days == {"60_day": 60, "180_day": 180}
    assert c.discovery_max_days == 180
