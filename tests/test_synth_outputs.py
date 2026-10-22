import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ili.synth import SynthParams, generate, write_outputs
from ili.units import FT_PER_M, IN_PER_MM, degrees_to_clock_min, hhmm_to_clock_min

EXPECTED_FILES = {
    "tally_prev_vendor_a.csv",
    "tally_curr_vendor_b.xlsx",
    "ground_truth.csv",
    "ground_truth_welds.csv",
    "synth_manifest.json",
    "reference/pipe_properties.csv",
    "reference/mop_profile.csv",
    "reference/hca_segments.csv",
    "reference/repairs_between_runs.csv",
    "reference/foreign_crossings.csv",
    "reference/next_inspection.csv",
    "reference/digs.csv",
}


@pytest.fixture(scope="module")
def result():
    return generate(SynthParams(line_miles=5))


@pytest.fixture(scope="module")
def written(result, tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    paths = write_outputs(result, out)
    return out, paths


def test_writes_all_files(written):
    out, paths = written
    assert set(paths) == EXPECTED_FILES
    for rel in EXPECTED_FILES:
        assert (out / rel).is_file(), rel


def test_vendor_a_format(written, result):
    out, _ = written
    a = pd.read_csv(out / "tally_prev_vendor_a.csv", dtype=str, keep_default_na=False)
    assert list(a.columns)[:3] == ["Feature No", "Event", "Abs Dist (ft)"]
    clocks = a.loc[a["O'clock"] != "", "O'clock"]
    assert len(clocks) > 400
    assert clocks.str.match(r"^(1[0-2]|[1-9]):[0-5]\d$").all()
    assert set(a.Event) == {"Girth Weld", "Metal Loss", "Dent", "Valve", "Tee"}
    assert set(a["Int/Ext"]) <= {"External", "Internal", ""}
    assert set(a["Seam Interaction"]) <= {"Y", "N", ""}


def test_vendor_a_values_match_standard(written, result):
    out, _ = written
    a = pd.read_csv(out / "tally_prev_vendor_a.csv", dtype={"US GW No": str, "O'clock": str})
    std = result.prev_std
    assert (a["Feature No"] == std.feature_id).all()
    assert np.allclose(a["Abs Dist (ft)"], std.log_dist_ft, atol=6e-4)
    has_clock = std.clock_min.notna().to_numpy()
    parsed = a.loc[has_clock, "O'clock"].map(hhmm_to_clock_min)
    assert (parsed.to_numpy() == std.clock_min[has_clock].astype(int).to_numpy()).all()


def test_vendor_b_format(written):
    out, _ = written
    title = pd.read_excel(out / "tally_curr_vendor_b.xlsx", sheet_name="Pipe Tally", header=None, nrows=2)
    assert "Vendor B" in str(title.iloc[0, 0]) and "Run date" in str(title.iloc[1, 0])
    b = pd.read_excel(out / "tally_curr_vendor_b.xlsx", sheet_name="Pipe Tally", header=2)
    assert list(b.columns)[:3] == ["Log Dist [m]", "Feature", "Feature ID"]
    assert b["Orientation [deg]"].dropna().between(0, 360, inclusive="left").all()
    assert set(b.Feature) == {"GWLD", "ML-CORR", "ML-CLUS", "DENT", "VALVE", "TEE"}


def test_vendor_roundtrip_values(written, result):
    out, _ = written
    b = pd.read_excel(out / "tally_curr_vendor_b.xlsx", sheet_name="Pipe Tally", header=2)
    std = result.curr_std
    assert (b["Feature ID"] == std.feature_id).all()
    assert np.allclose(b["Log Dist [m]"] * FT_PER_M, std.log_dist_ft, atol=2e-3)
    assert np.allclose(b["WT [mm]"] * IN_PER_MM, std.wt_nom_in, atol=1e-3, equal_nan=True)
    has_clock = std.clock_min.notna().to_numpy()
    clocks = degrees_to_clock_min(b.loc[has_clock, "Orientation [deg]"].to_numpy())
    assert (clocks == std.clock_min[has_clock].astype(int).to_numpy()).all()


def test_ids_resolve(written, result):
    out, _ = written
    gt = pd.read_csv(out / "ground_truth.csv", dtype=str)
    a_ids = set(result.vendor_a["Feature No"])
    b_ids = set(result.vendor_b["Feature ID"])
    assert set(gt.prev_feature_id.dropna()) <= a_ids
    assert set(gt.curr_feature_id.dropna()) <= b_ids


def test_truth_classes(result):
    gt = result.ground_truth
    classes = set(gt.truth_class)
    assert {"matched_1to1", "matched_cluster", "new", "missing"} <= classes
    assert classes <= {"matched_1to1", "matched_cluster", "new", "missing", "not_reported"}
    assert (gt.truth_class == "new").sum() >= 50
    missing = gt[gt.truth_class == "missing"]
    assert (missing.curr_status == "removed_by_repair").sum() >= 3


def test_ground_truth_has_no_measurements(result):
    assert not any(c.endswith("_meas_pct") for c in result.ground_truth.columns)


def test_weld_truth(result):
    wt = result.weld_truth
    assert list(wt.columns) == [
        "true_weld_idx", "chainage_ft", "prev_gw_id", "curr_gw_id", "prev_status", "curr_status", "in_replacement",
    ]
    assert (wt.prev_status == "missed").sum() == 3
    assert wt.in_replacement.sum() == 5  # 2 removed + 3 inserted interior welds


def _contiguous(df, end):
    df = df.sort_values("start_ft")
    return (
        df.start_ft.iloc[0] == 0
        and np.allclose(df.start_ft.to_numpy()[1:], df.end_ft.to_numpy()[:-1])
        and df.end_ft.iloc[-1] == pytest.approx(end)
    )


def test_reference_coverage(result):
    ref = result.reference
    end = result.curr_std.log_dist_ft.max()
    for name in ("hca_segments", "mop_profile", "pipe_properties"):
        assert _contiguous(ref[name], end), name
    rep = ref["repairs_between_runs"]
    assert len(rep) == 1 and rep.repair_type.iloc[0] == "pipe_replacement"
    new_welds = result.weld_truth.query("in_replacement and prev_status == 'not_present'").curr_gw_id
    gw = result.curr_std.query("feature_type == 'girth_weld'").set_index("us_gw_id").log_dist_ft
    assert gw.loc[new_welds].between(rep.start_ft.iloc[0], rep.end_ft.iloc[0]).all()


def test_reference_contents(result):
    ref = result.reference
    pp = ref["pipe_properties"]
    assert (pp.od_in == 12.75).all() and (pp.smys_psi == 52000).all()
    assert set(pp.wt_nom_in) == {0.250, 0.312}
    assert set(ref["mop_profile"].mop_psig) == {1440, 1200}
    hca = ref["hca_segments"]
    assert hca.segment_id.tolist() == ["SEG-01", "SEG-02", "SEG-03", "SEG-04"]
    assert "SEG-03" not in set(ref["next_inspection"].segment_id)
    assert len(ref["foreign_crossings"]) == 3
    digs = ref["digs"]
    assert len(digs) == 15
    assert set(digs.feature_id) <= set(result.curr_std.feature_id)


def test_manifest(written, result):
    out, _ = written
    m = json.loads((out / "synth_manifest.json").read_text(encoding="utf-8"))
    assert "DRAFT" in m["label"]
    assert m["params"]["seed"] == 42
    assert m["run_dates"] == {"prev": "2021-06-15", "curr": "2026-06-15"}
    assert m["curr_drift"] == pytest.approx(result.curr_drift)
    assert all(k.endswith(".csv") for k in m["sha256"])
    assert len(m["sha256"]) == 10


def test_deterministic():
    a = generate(SynthParams(seed=7, line_miles=2))
    b = generate(SynthParams(seed=7, line_miles=2))
    c = generate(SynthParams(seed=8, line_miles=2))
    for name in ("prev_std", "curr_std", "vendor_a", "vendor_b", "ground_truth", "weld_truth"):
        assert getattr(a, name).equals(getattr(b, name)), name
    for name in a.reference:
        assert a.reference[name].equals(b.reference[name]), name
    assert not a.prev_std.equals(c.prev_std)


def test_unsatisfiable_raises():
    with pytest.raises(ValueError, match="n_ext_ml_prev"):
        generate(SynthParams(line_miles=2, reporting_threshold_pct=99.9))


def test_cli(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "ili.synth", "--out", str(tmp_path), "--seed", "3"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    for rel in EXPECTED_FILES:
        assert (tmp_path / rel).is_file(), rel
    assert "matched_1to1" in proc.stdout


def test_cli_bad_params_exit_code(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "ili.synth", "--out", str(tmp_path), "--line-miles", "0.1"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2
    assert "line_miles" in proc.stderr
