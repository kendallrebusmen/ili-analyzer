"""Synthetic ILI data generator (spec section 6.1).

Builds a *true* pipeline and anomaly population, then observes it twice as two
ILI runs with odometer drift, local slip, missed welds, sizing noise, a
reporting threshold, and cluster-vs-pit reporting differences. Writes the runs
in two deliberately different vendor formats, plus synthetic reference files
and complete ground truth, so every later module can be scored.

All parameters are CLI arguments (``python -m ili.synth --help``). They describe
the synthetic *truth* and are deliberately independent of config/settings.yaml.

DRAFT – for review by qualified personnel. Synthetic data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

FT_PER_MILE = 5280.0

# Line geometry (D: see docs/DECISIONS.md)
JOINT_LEN_FT = (38.0, 42.0)
SHORT_JOINT_LEN_FT = (10.0, 30.0)
SHORT_JOINT_SHARE = 0.03
FITTING_JOINT_LEN_FT = (3.0, 8.0)
WT_STD_IN = 0.250
WT_HEAVY_IN = 0.312
HEAVY_WALL_SPAN = (0.45, 0.55)  # fraction of line length
REPLACED_JOINTS = 3
REPLACEMENT_NEW_JOINTS = 4
REPLACEMENT_CLEARANCE_JOINTS = 5  # from any fitting joint or line end


@dataclass
class SynthParams:
    seed: int = 42
    line_miles: float = 20.0
    years_between: float = 5.0
    prev_run_date: date = date(2021, 6, 15)
    odo_drift_max: float = 0.005
    slip_sigma_ft: float = 0.05
    n_missed_welds: int = 3
    depth_tol_pct: float = 10.0
    length_tol_in: float = 0.75
    clock_tol_min: float = 15.0
    tol_certainty: float = 0.80
    axial_sigma_in: float = 1.0
    reporting_threshold_pct: float = 10.0
    n_ext_ml_prev: int = 500
    n_new: int = 50
    n_internal: int = 8
    n_dents: int = 12
    n_valves: int = 4
    n_tees: int = 6
    n_cluster_sites: int = 6
    n_severe: int = 5
    growth_mean_mpy: float = 1.5
    max_growth_mpy: float = 10.0
    od_in: float = 12.75
    smys_psi: float = 52000.0
    seam_clock_min: int = 60


@dataclass
class LineTruth:
    welds: pd.DataFrame  # true_weld_idx, chainage_ft, in_prev, in_curr
    joints: pd.DataFrame  # joint_key, us_weld_idx, start_ft, length_ft, wt_nom_in, fitting, in_prev, in_curr
    pipe_segments: pd.DataFrame  # start_ft, end_ft, wt_nom_in (true chainage)
    replacement: tuple[float, float]  # true chainage of the replaced span
    length_ft: float


def _pipe_joint_lengths(target_ft: float, rng: np.random.Generator) -> list[float]:
    n = int(target_ft / SHORT_JOINT_LEN_FT[0]) + 2  # enough draws even if all short
    short = rng.random(n) < SHORT_JOINT_SHARE
    lengths = np.where(short, rng.uniform(*SHORT_JOINT_LEN_FT, n), rng.uniform(*JOINT_LEN_FT, n))
    stop = int(np.searchsorted(np.cumsum(lengths), target_ft)) + 1
    return lengths[:stop].tolist()


def _insert_fittings(lengths: list[float], p: SynthParams, rng) -> list[str | None]:
    """Insert one short joint per valve/tee, spread evenly; returns per-joint fitting labels."""
    kinds = [("valve", (i + 0.5) / p.n_valves) for i in range(p.n_valves)]
    kinds += [("tee", (i + 0.5) / p.n_tees) for i in range(p.n_tees)]
    kinds.sort(key=lambda k: k[1])
    fitting: list[str | None] = [None] * len(lengths)
    n_pipe = len(lengths)
    for offset, (kind, frac) in enumerate(kinds):
        idx = int(round(frac * n_pipe)) + offset
        lengths.insert(idx, float(rng.uniform(*FITTING_JOINT_LEN_FT)))
        fitting.insert(idx, kind)
    return fitting


def _pick_replacement(fitting: list[str | None], rng) -> int:
    n = len(fitting)
    fit_idx = np.array([i for i, f in enumerate(fitting) if f is not None])
    c = REPLACEMENT_CLEARANCE_JOINTS
    eligible = []
    for r in range(c, n - c - REPLACED_JOINTS + 1):
        span = np.arange(r, r + REPLACED_JOINTS)
        if fit_idx.size == 0 or np.min(np.abs(fit_idx[:, None] - span[None, :])) > c:
            eligible.append(r)
    if not eligible:
        raise ValueError(
            "line_miles too short for the requested fittings and a "
            f"{REPLACED_JOINTS}-joint replacement section"
        )
    return int(rng.choice(eligible))


def build_line(p: SynthParams, rng: np.random.Generator) -> LineTruth:
    target = p.line_miles * FT_PER_MILE
    lengths = _pipe_joint_lengths(target, rng)
    fitting = _insert_fittings(lengths, p, rng)
    r = _pick_replacement(fitting, rng)

    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    length_ft = float(np.sum(lengths))
    heavy_lo, heavy_hi = (f * length_ft for f in HEAVY_WALL_SPAN)
    # Wall changes happen at welds: first joint starting inside the heavy span.
    heavy_start = float(starts[np.searchsorted(starts, heavy_lo)])
    heavy_end = float(starts[np.searchsorted(starts, heavy_hi)])

    old = pd.DataFrame(
        {
            "start_ft": starts,
            "length_ft": lengths,
            "fitting": pd.Series(fitting, dtype=object),
            "in_prev": True,
            "in_curr": True,
        }
    )
    old.loc[r : r + REPLACED_JOINTS - 1, "in_curr"] = False
    rep_lo = float(starts[r])
    rep_hi = float(starts[r + REPLACED_JOINTS])  # downstream tie-in weld (clearance guarantees it exists)

    weights = rng.uniform(0.7, 1.3, REPLACEMENT_NEW_JOINTS)
    new_len = (rep_hi - rep_lo) * weights / weights.sum()
    new_start = rep_lo + np.concatenate([[0.0], np.cumsum(new_len)[:-1]])
    new = pd.DataFrame(
        {
            "start_ft": new_start,
            "length_ft": new_len,
            "fitting": pd.Series([None] * REPLACEMENT_NEW_JOINTS, dtype=object),
            "in_prev": False,
            "in_curr": True,
        }
    )

    joints = pd.concat([old, new], ignore_index=True)
    joints = joints.sort_values(["start_ft", "in_curr"], kind="stable").reset_index(drop=True)
    joints["wt_nom_in"] = np.where(
        (joints.start_ft >= heavy_start) & (joints.start_ft < heavy_end), WT_HEAVY_IN, WT_STD_IN
    )

    # Welds: one at each distinct joint start, plus the line end.
    # Rounded keys merge coincident starts (old/new joints at a tie-in); raw values are stored.
    keys = np.round(joints.start_ft.to_numpy(), 9)
    weld_rows: dict[float, tuple[float, bool, bool]] = {}
    for key, raw, in_prev, in_curr in zip(keys, joints.start_ft, joints.in_prev, joints.in_curr):
        raw0, prev, curr = weld_rows.get(key, (raw, False, False))
        weld_rows[key] = (raw0, prev or in_prev, curr or in_curr)
    weld_rows[np.round(length_ft, 9)] = (length_ft, True, True)
    chain = np.array(sorted(weld_rows))
    welds = pd.DataFrame(
        {
            "true_weld_idx": np.arange(len(chain)),
            "chainage_ft": [weld_rows[c][0] for c in chain],
            "in_prev": [weld_rows[c][1] for c in chain],
            "in_curr": [weld_rows[c][2] for c in chain],
        }
    )
    joints["us_weld_idx"] = np.searchsorted(chain, keys)
    joints.insert(0, "joint_key", np.arange(len(joints)))
    joints = joints[
        ["joint_key", "us_weld_idx", "start_ft", "length_ft", "wt_nom_in", "fitting", "in_prev", "in_curr"]
    ]

    pipe_segments = pd.DataFrame(
        {
            "start_ft": [0.0, heavy_start, heavy_end],
            "end_ft": [heavy_start, heavy_end, length_ft],
            "wt_nom_in": [WT_STD_IN, WT_HEAVY_IN, WT_STD_IN],
        }
    )
    return LineTruth(welds, joints, pipe_segments, (rep_lo, rep_hi), length_ft)
