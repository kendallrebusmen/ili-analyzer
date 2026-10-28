import pandas as pd
import pytest

from ili.config import PROJECT_ROOT
from ili.synth import SynthParams, generate, write_outputs

MAPS = PROJECT_ROOT / "config" / "vendor_maps"
MAP_A = MAPS / "synthetic_vendor_a.yaml"
MAP_B = MAPS / "synthetic_vendor_b.yaml"

# Largest difference the vendor files' rounding can introduce, per standard field.
FLOAT_TOL = {
    "log_dist_ft": 2e-3,
    "dist_from_us_gw_ft": 2e-3,
    "joint_length_ft": 2e-3,
    "wt_nom_in": 1e-3,
    "depth_pct": 0.051,
    "dent_depth_in": 1e-3,
    "length_in": 6e-3,
    "width_in": 6e-3,
    "lat": 1e-6,
    "lon": 1e-6,
    "elev_ft": 0.051,
}
EXACT = [
    "run_id", "run_date", "feature_id", "feature_type", "us_gw_id", "clock_min", "surface", "ml_origin",
    "seam_interaction", "dent_metal_loss", "vendor_cluster_id", "comments",
]


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """A 5-mile synthetic dataset written to disk: (SynthResult, output dir)."""
    out = tmp_path_factory.mktemp("synth")
    result = generate(SynthParams(line_miles=5))
    write_outputs(result, out)
    return result, out


def _plain(s: pd.Series) -> list:
    return [None if v is None or (not isinstance(v, str) and pd.isna(v)) else v for v in s.astype(object)]


def assert_matches_oracle(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    got = got.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    assert len(got) == len(expected)
    for col in EXACT:
        g = [str(v) if col in ("feature_type", "surface", "ml_origin") and v is not None else v for v in _plain(got[col])]
        e = [None if v is None else (bool(v) if col in ("seam_interaction", "dent_metal_loss") else v)
             for v in _plain(expected[col])]
        if col == "clock_min":
            e = [None if v is None else int(v) for v in e]
            g = [None if v is None else int(v) for v in g]
        bad = [i for i, (a, b) in enumerate(zip(g, e)) if a != b]
        assert not bad, f"{col}: first mismatch at row {bad[0]}: got {g[bad[0]]!r}, expected {e[bad[0]]!r}"
    for col, tol in FLOAT_TOL.items():
        g = got[col].astype(float).to_numpy()
        e = expected[col].astype(float).to_numpy()
        same_nan = pd.isna(g) == pd.isna(e)
        assert same_nan.all(), f"{col}: NaN pattern differs at row {(~same_nan).argmax()}"
        diff = abs(g - e)[~pd.isna(e)]
        assert (diff <= tol).all(), f"{col}: max diff {diff.max()} > {tol}"
