import math

import numpy as np
import pytest

from ili.config import load_settings
from ili.strength import (
    STRENGTH_METHODS,
    assessed_depth_in,
    assessed_length_in,
    burst_pressure,
    failure_stress,
    folias_modified,
    folias_original,
    rstreng_effective_area,
    safe_pressure,
    z_param,
)

D, T, SMYS, DEPTH, LEN = 12.75, 0.250, 52_000.0, 0.125, 4.0


def _pf(method, d=DEPTH, L=LEN, t=T):
    z, m, flow, sf = failure_stress(method, D, t, d, L, SMYS)
    return z, m, flow, sf, burst_pressure(sf, t, D)


def test_reference_case_modified():
    z, m, flow, sf, pf = _pf("modified_b31g")
    assert z == pytest.approx(5.0196, abs=1e-3)
    assert m == pytest.approx(2.0161, abs=1e-3)
    assert flow == pytest.approx(62_000.0)
    assert sf == pytest.approx(45_173, abs=5)
    assert pf == pytest.approx(1_771.5, abs=1)


def test_reference_case_original():
    z, m, flow, sf, pf = _pf("original_b31g")
    assert z == pytest.approx(5.0196, abs=1e-3)
    assert m == pytest.approx(2.2396, abs=1e-3)
    assert flow == pytest.approx(57_200.0)
    assert sf == pytest.approx(44_803, abs=5)
    assert pf == pytest.approx(1_757.2, abs=1)


def _L_for_z(z, t=T):
    return math.sqrt(z * D * t)


def test_z20_boundary_original():
    flow = 1.1 * SMYS
    r = DEPTH / T
    at20 = _L_for_z(20.0)
    z, m, _, sf = failure_stress("original_b31g", D, T, DEPTH, at20, SMYS)
    assert z == pytest.approx(20.0)
    assert m == pytest.approx(math.sqrt(17.0))
    assert sf == pytest.approx(flow * (1 - 2 / 3 * r) / (1 - 2 / 3 * r / math.sqrt(17)), rel=1e-9)
    above = _L_for_z(20.0 + 1e-9)
    z2, m2, _, sf2 = failure_stress("original_b31g", D, T, DEPTH, above, SMYS)
    assert z2 > 20.0
    assert np.isnan(m2)
    assert sf2 == pytest.approx(flow * (1 - r), rel=1e-9)
    assert sf2 < sf  # the jump is the spec's formula, not smoothed


def test_z50_boundary_modified():
    assert folias_modified(50.0) == pytest.approx(math.sqrt(1 + 31.375 - 8.4375))
    assert folias_modified(50.0) == pytest.approx(4.8926, abs=1e-4)
    assert folias_modified(50.0 + 1e-9) == pytest.approx(0.032 * 50 + 3.3)
    assert folias_modified(50.0 + 1e-9) == pytest.approx(4.9, abs=1e-6)
    zs = np.linspace(0, 100, 1001)
    m = folias_modified(zs)
    assert np.all(np.diff(m) >= 0)


def test_folias_original_branch():
    assert folias_original(20.0) == pytest.approx(math.sqrt(17.0))
    assert np.isnan(folias_original(20.0 + 1e-9))
    assert folias_original(0.0) == 1.0


@pytest.mark.parametrize("method,flow", [("modified_b31g", 62_000.0), ("original_b31g", 57_200.0)])
def test_zero_depth_gives_flow_stress(method, flow):
    for L in (4.0, 100.0):  # both Folias branches
        _, _, f, sf = failure_stress(method, D, T, 0.0, L, SMYS)
        assert f == pytest.approx(flow)
        assert sf == pytest.approx(flow)


def test_very_short_length():
    z, m, flow, sf = failure_stress("modified_b31g", D, T, DEPTH, 1e-9, SMYS)
    assert m == pytest.approx(1.0, abs=1e-9)
    assert sf == pytest.approx(flow, rel=1e-6)


@pytest.mark.parametrize("method", list(STRENGTH_METHODS))
def test_dt_clipped(method):
    over = _pf(method, d=1.2 * T)
    at = _pf(method, d=T)
    assert np.isfinite(over[4])
    assert over[4] == pytest.approx(at[4])
    neg = _pf(method, d=-0.05)
    assert neg[4] == pytest.approx(_pf(method, d=0.0)[4])


def test_nan_in_nan_out():
    z, m, flow, sf = failure_stress("modified_b31g", D, T, np.array([np.nan, DEPTH]), LEN, SMYS)
    assert np.isnan(sf[0]) and np.isfinite(sf[1])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        failure_stress("rstreng", D, T, DEPTH, LEN, SMYS)


def test_assessed_depth_and_length_basis():
    s = load_settings()
    s_rep = s.model_copy(update={"strength": s.strength.model_copy(update={"depth_basis": "reported", "length_basis": "reported"})})
    s_tol = s.model_copy(update={"strength": s.strength.model_copy(update={"depth_basis": "reported_plus_tolerance", "length_basis": "reported_plus_tolerance"})})
    assert assessed_depth_in(50.0, 0.25, s_rep) == pytest.approx(0.125)
    assert assessed_depth_in(50.0, 0.25, s_tol) == pytest.approx(0.150)
    assert assessed_depth_in(95.0, 0.25, s_tol) == pytest.approx(0.25)
    assert assessed_length_in(4.0, s_rep) == pytest.approx(4.0)
    assert assessed_length_in(4.0, s_tol) == pytest.approx(4.75)
    assert np.isnan(assessed_depth_in(np.nan, 0.25, s_tol))


def test_z_param():
    assert z_param(4.0, 12.75, 0.25) == pytest.approx(16 / (12.75 * 0.25))


def test_safe_pressure():
    assert safe_pressure(1000.0, load_settings()) == pytest.approx(720.0)


def test_rstreng_stub_raises():
    assert "rstreng" not in STRENGTH_METHODS
    with pytest.raises(NotImplementedError, match="RSTRENG is out of scope for v1"):
        rstreng_effective_area(None, D, T, SMYS)
