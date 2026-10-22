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

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import ili
from ili.schema import STANDARD_COLUMNS
from ili.units import (
    CLOCK_MINUTES,
    FT_PER_M,
    IN_PER_MM,
    clock_min_to_degrees,
    clock_min_to_hhmm,
    sigma_from_tolerance,
    wrap_clock_min,
)

FT_PER_MILE = 5280.0
DAYS_PER_YEAR = 365.25

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

# Feature population (D: see docs/DECISIONS.md)
EXT_ML_PREV_DEPTH_PCT = (5.0, 55.0)
NEW_PREV_DEPTH_PCT = (2.0, 12.0)
SEVERE_PREV_DEPTH_PCT = (70.0, 75.0)
SEVERE_GROWTH_MPY = (8.0, 10.0)
INTERNAL_PREV_DEPTH_PCT = (10.0, 40.0)
PIT_PREV_DEPTH_PCT = (15.0, 40.0)
ML_LENGTH_IN = (0.5, 4.0)
ML_WIDTH_IN = (0.5, 3.0)
PIT_SIZE_IN = (0.3, 0.8)
PIT_SPACING_IN = (0.3, 0.9)
PIT_CLOCK_SPREAD_MIN = 10
PITS_PER_SITE = (2, 4)
DENT_DEPTH_PCT_OD = (1.0, 7.0)
DENT_SIZE_IN = (3.0, 12.0)
DENT_ML_DEPTH_PCT = (10.0, 30.0)
SEAM_PROXIMITY_MIN = 10
MAX_TRUE_DEPTH_PCT = 99.0
WELD_CLEARANCE_FT = 0.25  # features sit at least this far from a weld
N_IN_REPLACED_SECTION = 3
REJECTION_GUARD = 100  # max draws = guard x requested count


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

    def __post_init__(self):
        for fld in fields(self):
            value = getattr(self, fld.name)
            if fld.name.startswith("n_") and value < 0:
                raise ValueError(f"{fld.name} must be >= 0, got {value}")
        for name in ("line_miles", "years_between", "od_in"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")


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


# ---------------------------------------------------------------------------
# True feature population
# ---------------------------------------------------------------------------


def run_dates(p: SynthParams) -> tuple[date, date]:
    return p.prev_run_date, p.prev_run_date + timedelta(days=round(p.years_between * DAYS_PER_YEAR))


def delta_t_years(p: SynthParams) -> float:
    prev, curr = run_dates(p)
    return (curr - prev).days / DAYS_PER_YEAR


def _growth_rates(p: SynthParams, rng, n: int) -> np.ndarray:
    """Right-skewed true growth: exponential, capped (many near zero)."""
    return np.minimum(rng.exponential(p.growth_mean_mpy, n), p.max_growth_mpy)


def _rejection_sample(
    n: int, name: str, draw: Callable[[int], dict], accept: Callable[[dict], np.ndarray]
) -> dict:
    """Draw candidate batches until n are accepted; raise rather than spin forever."""
    if n == 0:
        return {k: np.asarray(v)[:0] for k, v in draw(0).items()}
    kept: dict[str, list] = {}
    got = drawn = 0
    while got < n:
        if drawn >= REJECTION_GUARD * max(n, 1):
            raise ValueError(f"could not generate {name}={n} features within {drawn} draws; check parameters")
        batch = draw(max(2 * (n - got), 8))
        drawn += len(next(iter(batch.values())))
        ok = accept(batch)
        for k, v in batch.items():
            kept.setdefault(k, []).append(np.asarray(v)[ok])
        got += int(ok.sum())
    return {k: np.concatenate(v)[:n] for k, v in kept.items()}


class _Placer:
    """Picks joints and in-joint positions for features."""

    def __init__(self, line: LineTruth, rng):
        j = line.joints
        self.rng = rng
        self.joints = j.set_index("joint_key")
        self.normal = j.loc[j.fitting.isna() & j.in_prev & j.in_curr, "joint_key"].to_numpy()
        self.replaced = j.loc[j.in_prev & ~j.in_curr, "joint_key"].to_numpy()

    def place(self, n: int, pool: np.ndarray | None = None, span_ft: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Random joints and positions; a span of ``span_ft`` after each position stays in the joint."""
        pool = self.normal if pool is None else pool
        keys = self.rng.choice(pool, n)
        start = self.joints.start_ft.loc[keys].to_numpy()
        length = self.joints.length_ft.loc[keys].to_numpy()
        pos = start + self.rng.uniform(WELD_CLEARANCE_FT, length - WELD_CLEARANCE_FT - span_ft)
        return keys, pos

    def wt(self, keys) -> np.ndarray:
        return self.joints.wt_nom_in.loc[keys].to_numpy()


def _ml_block(p, rng, placer, population, keys, chain, prev_true, growth, surface,
              length, width, clock, cluster_site=None, draws=None) -> pd.DataFrame:
    n = len(keys)
    dt = delta_t_years(p)
    sigma = sigma_from_tolerance(p.depth_tol_pct, p.tol_certainty)
    curr_true = np.minimum(prev_true + growth / 1000 * dt / placer.wt(keys) * 100, MAX_TRUE_DEPTH_PCT)
    if draws is None:
        draws = {"prev_meas": prev_true + rng.normal(0, sigma, n),
                 "curr_meas": curr_true + rng.normal(0, sigma, n)}
    d = np.abs(clock - p.seam_clock_min)
    return pd.DataFrame({
        "feature_type": "metal_loss",
        "surface": surface,
        "chainage_ft": chain,
        "joint_key": keys,
        "clock_min": clock.astype(int),
        "depth_prev_true_pct": prev_true,
        "depth_curr_true_pct": curr_true,
        "depth_prev_meas_pct": np.clip(draws["prev_meas"], 0, 100),
        "depth_curr_meas_pct": np.clip(draws["curr_meas"], 0, 100),
        "growth_mpy": growth,
        "length_in": length,
        "width_in": width,
        "dent_depth_in": np.nan,
        "dent_metal_loss": False,
        "seam_interaction": np.minimum(d, CLOCK_MINUTES - d) <= SEAM_PROXIMITY_MIN,
        "cluster_site": cluster_site if cluster_site is not None else pd.NA,
        "population": population,
    })


def _reported_ml(p, rng, placer, population, n, depth_range, accept, pool=None, growth=None, surface="EXT"):
    """Metal loss whose pre-drawn run measurements satisfy ``accept``."""
    dt = delta_t_years(p)
    sigma = sigma_from_tolerance(p.depth_tol_pct, p.tol_certainty)

    def draw(m):
        keys, chain = placer.place(m, pool)
        g = growth(m) if growth else _growth_rates(p, rng, m)
        prev = rng.uniform(*depth_range, m)
        curr = np.minimum(prev + g / 1000 * dt / placer.wt(keys) * 100, MAX_TRUE_DEPTH_PCT)
        return {"keys": keys, "chain": chain, "prev": prev, "growth": g,
                "prev_meas": prev + rng.normal(0, sigma, m), "curr_meas": curr + rng.normal(0, sigma, m)}

    s = _rejection_sample(n, f"n_{population}" if population != "ext_ml" else "n_ext_ml_prev", draw, accept)
    return _ml_block(
        p, rng, placer, population, s["keys"], s["chain"], s["prev"], s["growth"], surface,
        rng.uniform(*ML_LENGTH_IN, n), rng.uniform(*ML_WIDTH_IN, n), rng.integers(0, CLOCK_MINUTES, n),
        draws={"prev_meas": s["prev_meas"], "curr_meas": s["curr_meas"]},
    )


def build_features(p: SynthParams, line: LineTruth, rng: np.random.Generator) -> pd.DataFrame:
    placer = _Placer(line, rng)
    thr = p.reporting_threshold_pct
    reported_prev = lambda b: b["prev_meas"] >= thr  # noqa: E731
    blocks = []

    # External metal loss reported in the previous run; a few sit in the replaced joints.
    n_rep = min(N_IN_REPLACED_SECTION, p.n_ext_ml_prev)
    blocks.append(_reported_ml(p, rng, placer, "ext_ml", n_rep, EXT_ML_PREV_DEPTH_PCT, reported_prev,
                               pool=placer.replaced))
    blocks.append(_reported_ml(p, rng, placer, "ext_ml", p.n_ext_ml_prev - n_rep, EXT_ML_PREV_DEPTH_PCT,
                               reported_prev))
    # New: below threshold in the previous run, reported in the current run.
    blocks.append(_reported_ml(p, rng, placer, "new", p.n_new, NEW_PREV_DEPTH_PCT,
                               lambda b: (b["prev_meas"] < thr) & (b["curr_meas"] >= thr)))
    # Severe: pass 80 % of wall by the current run.
    blocks.append(_reported_ml(p, rng, placer, "severe", p.n_severe, SEVERE_PREV_DEPTH_PCT, reported_prev,
                               growth=lambda m: rng.uniform(*SEVERE_GROWTH_MPY, m)))
    blocks.append(_reported_ml(p, rng, placer, "internal", p.n_internal, INTERNAL_PREV_DEPTH_PCT,
                               reported_prev, surface="INT"))

    # Cluster sites: 2-4 pits closer together than the interaction spacing.
    for site in range(p.n_cluster_sites):
        k = int(rng.integers(PITS_PER_SITE[0], PITS_PER_SITE[1] + 1))
        offsets_ft = np.concatenate([[0.0], np.cumsum(rng.uniform(*PIT_SPACING_IN, k - 1))]) / 12
        keys, base = placer.place(1, span_ft=offsets_ft[-1])
        chain = base[0] + offsets_ft
        clock = np.mod(rng.integers(0, CLOCK_MINUTES) + rng.integers(-PIT_CLOCK_SPREAD_MIN,
                                                                      PIT_CLOCK_SPREAD_MIN + 1, k), CLOCK_MINUTES)
        blocks.append(_ml_block(
            p, rng, placer, "cluster_pit", np.repeat(keys, k), chain, rng.uniform(*PIT_PREV_DEPTH_PCT, k),
            _growth_rates(p, rng, k), "EXT", rng.uniform(*PIT_SIZE_IN, k), rng.uniform(*PIT_SIZE_IN, k), clock,
            cluster_site=site,
        ))

    blocks.append(_dents(p, rng, placer))
    blocks.append(_anchors(line))

    f = pd.concat(blocks, ignore_index=True).sort_values("chainage_ft", kind="stable").reset_index(drop=True)
    f.insert(0, "true_id", [f"T{i:05d}" for i in range(1, len(f) + 1)])
    f.insert(5, "wt_nom_in", placer.wt(f.joint_key))  # true nominal wall of the feature's joint
    f["clock_min"] = f.clock_min.astype("Int64")
    f["cluster_site"] = f.cluster_site.astype("Int64")
    return f


def _dents(p: SynthParams, rng, placer: _Placer) -> pd.DataFrame:
    n = p.n_dents
    n_top = (n + 1) // 2
    top = np.arange(n) < n_top
    clock = np.where(
        top,
        np.mod(rng.integers(480, 480 + 481, n), CLOCK_MINUTES),  # 8:00 through 4:00 over the top
        rng.integers(241, 480, n),  # strictly below 4 and 8 o'clock
    )
    with_ml = np.zeros(n, dtype=bool)
    with_ml[:2] = True  # two top
    with_ml[n_top:n_top + 2] = True  # two bottom
    pct_od = rng.permutation(np.linspace(*DENT_DEPTH_PCT_OD, n))
    keys, chain = placer.place(n)
    ml_prev = np.where(with_ml, rng.uniform(*DENT_ML_DEPTH_PCT, n), np.nan)
    sigma = sigma_from_tolerance(p.depth_tol_pct, p.tol_certainty)
    d = np.abs(clock - p.seam_clock_min)
    return pd.DataFrame({
        "feature_type": "dent",
        "surface": "EXT",
        "chainage_ft": chain,
        "joint_key": keys,
        "clock_min": clock,
        "depth_prev_true_pct": ml_prev,
        "depth_curr_true_pct": ml_prev,  # dents do not grow
        "depth_prev_meas_pct": np.clip(ml_prev + rng.normal(0, sigma, n), 0, 100),
        "depth_curr_meas_pct": np.clip(ml_prev + rng.normal(0, sigma, n), 0, 100),
        "growth_mpy": 0.0,
        "length_in": rng.uniform(*DENT_SIZE_IN, n),
        "width_in": rng.uniform(*DENT_SIZE_IN, n),
        "dent_depth_in": pct_od / 100 * p.od_in,
        "dent_metal_loss": with_ml,
        "seam_interaction": np.minimum(d, CLOCK_MINUTES - d) <= SEAM_PROXIMITY_MIN,
        "cluster_site": pd.NA,
        "population": "dent",
    })


def _anchors(line: LineTruth) -> pd.DataFrame:
    fit = line.joints[line.joints.fitting.notna()]
    return pd.DataFrame({
        "feature_type": fit.fitting.to_numpy(),
        "surface": None,
        "chainage_ft": (fit.start_ft + fit.length_ft / 2).to_numpy(),
        "joint_key": fit.joint_key.to_numpy(),
        "clock_min": pd.NA,
        "depth_prev_true_pct": np.nan,
        "depth_curr_true_pct": np.nan,
        "depth_prev_meas_pct": np.nan,
        "depth_curr_meas_pct": np.nan,
        "growth_mpy": np.nan,
        "length_in": np.nan,
        "width_in": np.nan,
        "dent_depth_in": np.nan,
        "dent_metal_loss": False,
        "seam_interaction": False,
        "cluster_site": pd.NA,
        "population": "anchor",
    })


# ---------------------------------------------------------------------------
# Observing a run
# ---------------------------------------------------------------------------

VENDOR = {"prev": "A", "curr": "B"}
DENT_DEPTH_SIGMA_IN = 0.02
MIN_REPORTED_SIZE_IN = 0.1
SLIP_CLIP_SIGMAS = 3.0
LINE_ORIGIN = (32.0, -84.0)  # lat, lon of chainage 0
LINE_BEARING_DEG = 40.0
FT_PER_DEG_LAT = 364_000.0


@dataclass
class ObservedRun:
    std: pd.DataFrame  # standard schema, girth welds included
    feat_link: pd.DataFrame  # true_id, feature_id, status
    weld_link: pd.DataFrame  # true_weld_idx, gw_id, weld_status
    drift: float  # odometer scale error of this run


def _gps(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth synthetic route: a meandering line on a fixed bearing, rolling terrain."""
    th = np.radians(LINE_BEARING_DEG)
    meander = 300.0 * np.sin(2 * np.pi * chain / 40_000)
    north = chain * np.cos(th) - meander * np.sin(th)
    east = chain * np.sin(th) + meander * np.cos(th)
    lat = LINE_ORIGIN[0] + north / FT_PER_DEG_LAT
    lon = LINE_ORIGIN[1] + east / (FT_PER_DEG_LAT * np.cos(np.radians(LINE_ORIGIN[0])))
    elev = 400.0 + 60.0 * np.sin(2 * np.pi * chain / 30_000)
    return lat, lon, elev


def _pick_missed_welds(which, p, line, features, rng, exclude) -> set[int]:
    """Welds this run fails to report. Preferably each sits upstream of a joint holding
    metal loss. Never at a fitting, the replaced span, a wall change, the line ends,
    next to another missed weld in this run, or at/next to one missed by the other run
    (``exclude``) - those would make joint correspondences no match type can express."""
    if p.n_missed_welds == 0:
        return set()
    w = line.welds
    present = w[w[f"in_{which}"]]
    lo, hi = line.replacement
    fit = line.joints[line.joints.fitting.notna()]
    banned = set(fit.us_weld_idx) | set(fit.us_weld_idx + 1)
    banned |= {int(e) + d for e in exclude for d in (-1, 0, 1)}
    banned |= {int(w.true_weld_idx.iloc[0]), int(w.true_weld_idx.iloc[-1])}
    banned |= set(w.loc[w.chainage_ft.between(lo, hi), "true_weld_idx"])
    wall_change = line.pipe_segments.start_ft.to_numpy()[1:]
    banned |= set(w.loc[np.isin(w.chainage_ft.to_numpy(), wall_change), "true_weld_idx"])
    joints = line.joints[line.joints[f"in_{which}"]]
    ml_joints = features.loc[features.population == "ext_ml", "joint_key"]
    useful = set(joints.loc[joints.joint_key.isin(ml_joints), "us_weld_idx"])
    eligible = [int(i) for i in present.true_weld_idx if i not in banned]
    preferred = [i for i in eligible if i in useful]
    chosen: list[int] = []

    def take(pool):
        for idx in rng.permutation(pool):
            if all(abs(int(idx) - c) > 1 for c in chosen):
                chosen.append(int(idx))
            if len(chosen) == p.n_missed_welds:
                return True
        return False

    if take(preferred) or take([i for i in eligible if i not in useful and i not in chosen]):
        return set(chosen)
    raise ValueError(f"n_missed_welds={p.n_missed_welds} cannot be placed on this line; reduce it")


def _cluster_envelope(rows: pd.DataFrame, od_in: float) -> dict:
    """One vendor 'cluster' report: axial/circumferential bounding box, max depth."""
    half = rows.length_in.to_numpy() / 24  # ft
    start = (rows.log_dist_ft - half).min()
    end = (rows.log_dist_ft + half).max()
    c0 = int(rows.clock_min.iloc[0])
    rel_min = (rows.clock_min.to_numpy().astype(int) - c0 + 360) % CLOCK_MINUTES - 360
    circ_in = rel_min / CLOCK_MINUTES * np.pi * od_in
    half_w = rows.width_in.to_numpy() / 2
    c_lo, c_hi = (circ_in - half_w).min(), (circ_in + half_w).max()
    mid_min = (c_lo + c_hi) / 2 / (np.pi * od_in) * CLOCK_MINUTES
    return {
        "log_center": (start + end) / 2,
        "length_in": (end - start) * 12,
        "width_in": c_hi - c_lo,
        "clock_min": wrap_clock_min(c0 + mid_min),
        "depth_pct": rows.depth_pct.max(),
        "seam_interaction": bool(rows.seam_interaction.any()),
    }


def observe_run(
    which: str,
    p: SynthParams,
    line: LineTruth,
    features: pd.DataFrame,
    rng: np.random.Generator,
    exclude_welds: set[int] | frozenset = frozenset(),
) -> ObservedRun:
    vendor = VENDOR[which]
    run_date = dict(zip(("prev", "curr"), run_dates(p)))[which]
    run_id = f"{run_date.year}_vendor{vendor}"
    thr = p.reporting_threshold_pct

    # Odometer: scale drift plus cumulative local slip at each weld.
    w = line.welds[line.welds[f"in_{which}"]]
    chain_w = w.chainage_ft.to_numpy()
    # Current-run drift is at least half the maximum, either sign, so it always matters.
    drift = 0.0 if which == "prev" else float(rng.choice([-1, 1]) * rng.uniform(p.odo_drift_max / 2, p.odo_drift_max))
    clip = SLIP_CLIP_SIGMAS * p.slip_sigma_ft
    steps = np.clip(rng.normal(0, p.slip_sigma_ft, len(w)), -clip, clip)
    steps[0] = 0.0
    log_w = chain_w * (1 + drift) + np.cumsum(steps)
    if not (np.diff(log_w) > 0).all():
        raise ValueError("slip_sigma_ft too large: odometer distance would run backwards")

    missed = _pick_missed_welds(which, p, line, features, rng, exclude_welds)
    keep = ~w.true_weld_idx.isin(missed).to_numpy()
    rep_idx = w.true_weld_idx.to_numpy()[keep]
    rep_chain, rep_log = chain_w[keep], log_w[keep]
    gw_ids = [f"GW{(i + 1) * 10:05d}" if vendor == "A" else str(i + 1) for i in range(len(rep_idx))]
    jlen = np.append(np.diff(rep_log), np.nan)
    run_joints = line.joints[line.joints[f"in_{which}"]]
    wt = run_joints.set_index("us_weld_idx").wt_nom_in.reindex(rep_idx).to_numpy()

    # Per-feature observation noise (drawn for every true feature, in a fixed order).
    f = features
    n = len(f)
    sig_len = sigma_from_tolerance(p.length_tol_in, p.tol_certainty)
    sig_clk = sigma_from_tolerance(p.clock_tol_min, p.tol_certainty)
    axial_noise_ft = rng.normal(0, p.axial_sigma_in / 12, n)
    length_obs = np.maximum(f.length_in.to_numpy() + rng.normal(0, sig_len, n), MIN_REPORTED_SIZE_IN)
    width_obs = np.maximum(f.width_in.to_numpy() + rng.normal(0, sig_len, n), MIN_REPORTED_SIZE_IN)
    clock_noise = rng.normal(0, sig_clk, n)
    dent_obs = np.maximum(f.dent_depth_in.to_numpy() + rng.normal(0, DENT_DEPTH_SIGMA_IN, n), 0)
    gps_noise = rng.normal(0, 1e-5, (n, 2))

    k = np.searchsorted(rep_chain, f.chainage_ft.to_numpy(), side="right") - 1
    log_true = np.interp(f.chainage_ft.to_numpy(), chain_w, log_w)
    dist = np.clip(log_true - rep_log[k] + axial_noise_ft, 0, jlen[k])
    lat, lon, elev = _gps(f.chainage_ft.to_numpy())
    clock = f.clock_min.to_numpy(dtype=float, na_value=np.nan)
    clock_obs = np.where(np.isnan(clock), np.nan, np.mod(np.rint(clock + clock_noise), CLOCK_MINUTES))
    meas = f[f"depth_{which}_meas_pct"].to_numpy()
    is_ml = (f.feature_type == "metal_loss").to_numpy()
    is_dent = (f.feature_type == "dent").to_numpy()
    obs = pd.DataFrame({
        "_true_id": f.true_id,
        "feature_type": f.feature_type,
        "log_dist_ft": rep_log[k] + dist,
        "us_gw_id": np.asarray(gw_ids, dtype=object)[k],
        "dist_from_us_gw_ft": dist,
        "joint_length_ft": jlen[k],
        "wt_nom_in": wt[k],
        "depth_pct": np.where(is_ml | (is_dent & f.dent_metal_loss.to_numpy(dtype=bool)), meas, np.nan),
        "dent_depth_in": np.where(is_dent, dent_obs, np.nan),
        "length_in": length_obs,
        "width_in": width_obs,
        "clock_min": clock_obs,
        "surface": f.surface,
        "seam_interaction": np.where(f.population == "anchor", None, f.seam_interaction.astype(object)),
        "dent_metal_loss": np.where(is_dent, f.dent_metal_loss.astype(object), None),
        "vendor_cluster_id": None,
        "lat": lat + gps_noise[:, 0],
        "lon": lon + gps_noise[:, 1],
        "elev_ft": elev,
        "comments": None,
    })

    # Which true features this run reports, and how.
    in_run = f.joint_key.isin(run_joints.joint_key).to_numpy()
    status = np.where(in_run, "reported", "removed_by_repair").astype(object)
    below = in_run & is_ml & (meas < thr)
    status[below] = "below_threshold"

    rows = []
    pit_sites = f.cluster_site.to_numpy(dtype=float, na_value=np.nan)
    cluster_parity = 0 if which == "prev" else 1
    n_clusters = 0
    for site in sorted(set(pit_sites[~np.isnan(pit_sites)].astype(int))):
        if site % 2 != cluster_parity:
            continue
        members = np.nonzero((pit_sites == site) & in_run)[0]
        if len(members) == 0:
            continue
        pits = obs.iloc[members]
        if pits.depth_pct.max() < thr:
            status[members] = "below_threshold"
            continue
        env = _cluster_envelope(pits, p.od_in)
        kk = int(np.searchsorted(rep_log, env["log_center"], side="right") - 1)
        d = float(np.clip(env["log_center"] - rep_log[kk], 0, jlen[kk]))
        n_clusters += 1
        first = pits.iloc[0]
        rows.append(pd.DataFrame([{
            "feature_type": "metal_loss",
            "log_dist_ft": rep_log[kk] + d,
            "us_gw_id": gw_ids[kk],
            "dist_from_us_gw_ft": d,
            "joint_length_ft": jlen[kk],
            "wt_nom_in": wt[kk],
            "depth_pct": env["depth_pct"],
            "dent_depth_in": np.nan,
            "length_in": env["length_in"],
            "width_in": env["width_in"],
            "clock_min": env["clock_min"],
            "surface": "EXT",
            "seam_interaction": env["seam_interaction"],
            "dent_metal_loss": None,
            "vendor_cluster_id": f"{vendor}C{n_clusters:03d}",
            "lat": first.lat,
            "lon": first.lon,
            "elev_ft": first.elev_ft,
            "comments": "CLUSTER",
            "_tids": tuple(f.true_id.iloc[members]),
        }]))
        status[members] = "reported_in_cluster"

    # Individually reported features (after clustering has claimed its pits).
    single = status == "reported"
    rows.append(obs[single].drop(columns="_true_id").assign(_tids=obs.loc[single, "_true_id"].map(lambda t: (t,))))

    # Girth welds.
    wlat, wlon, welev = _gps(rep_chain)
    rows.append(pd.DataFrame({
        "feature_type": "girth_weld",
        "log_dist_ft": rep_log,
        "us_gw_id": gw_ids,
        "dist_from_us_gw_ft": 0.0,
        "joint_length_ft": jlen,
        "wt_nom_in": wt,
        "lat": wlat,
        "lon": wlon,
        "elev_ft": welev,
        "_tids": [()] * len(rep_idx),
    }))

    std = pd.concat([r for r in rows if len(r)], ignore_index=True)
    std["_order"] = (std.feature_type != "girth_weld").astype(int)
    std = std.sort_values(["log_dist_ft", "_order"], kind="stable").reset_index(drop=True)
    std["feature_id"] = [f"{vendor}-{i:06d}" for i in range(1, len(std) + 1)]
    std["run_id"] = run_id
    std["run_date"] = run_date
    std["clock_min"] = std.clock_min.astype("Int64")

    tid_to_fid = {t: fid for fid, tids in zip(std.feature_id, std._tids) for t in tids}
    feat_link = pd.DataFrame({
        "true_id": f.true_id.to_numpy(),
        "feature_id": [tid_to_fid.get(t) for t in f.true_id],
        "status": status,
    })
    weld_status = np.where(line.welds[f"in_{which}"], "reported", "not_present").astype(object)
    weld_status[line.welds.true_weld_idx.isin(missed).to_numpy()] = "missed"
    gw_of = dict(zip(rep_idx.tolist(), gw_ids))
    weld_link = pd.DataFrame({
        "true_weld_idx": line.welds.true_weld_idx.to_numpy(),
        "gw_id": [gw_of.get(int(i)) for i in line.welds.true_weld_idx],
        "weld_status": weld_status,
    })
    return ObservedRun(std[STANDARD_COLUMNS], feat_link, weld_link, drift)


# ---------------------------------------------------------------------------
# Vendor formats
# ---------------------------------------------------------------------------

DRAFT_LABEL = "DRAFT – for review by qualified personnel. SYNTHETIC DATA – not from any real pipeline."

_A_EVENT = {"girth_weld": "Girth Weld", "metal_loss": "Metal Loss", "dent": "Dent", "valve": "Valve", "tee": "Tee"}
_A_SURFACE = {"EXT": "External", "INT": "Internal", "MID": "Mid-wall", "UNK": "Unknown"}
_B_EVENT = {"girth_weld": "GWLD", "metal_loss": "ML-CORR", "dent": "DENT", "valve": "VALVE", "tee": "TEE"}


def _yn(s: pd.Series) -> list[str]:
    return ["" if v is None or pd.isna(v) else ("Y" if v else "N") for v in s]


def _flag01(s: pd.Series) -> pd.Series:
    return pd.Series([pd.NA if v is None or pd.isna(v) else int(bool(v)) for v in s], dtype="Int64")


def to_vendor_a(std: pd.DataFrame) -> pd.DataFrame:
    """Vendor A: CSV, feet and inches, hh:mm clock, spelled-out vocabulary, Y/N flags."""
    return pd.DataFrame({
        "Feature No": std.feature_id.to_numpy(),
        "Event": std.feature_type.map(_A_EVENT).to_numpy(),
        "Abs Dist (ft)": std.log_dist_ft.round(3).to_numpy(),
        "US GW No": std.us_gw_id.to_numpy(),
        "Dist to US GW (ft)": std.dist_from_us_gw_ft.round(3).to_numpy(),
        "Jt Len (ft)": std.joint_length_ft.round(3).to_numpy(),
        "WT (in)": std.wt_nom_in.round(3).to_numpy(),
        "Depth (%)": std.depth_pct.round(1).to_numpy(),
        "Dent Depth (in)": std.dent_depth_in.round(3).to_numpy(),
        "Length (in)": std.length_in.round(2).to_numpy(),
        "Width (in)": std.width_in.round(2).to_numpy(),
        "O'clock": ["" if pd.isna(m) else clock_min_to_hhmm(int(m)) for m in std.clock_min],
        "Int/Ext": std.surface.map(_A_SURFACE).fillna("").to_numpy(),
        "Seam Interaction": _yn(std.seam_interaction),
        "Dent w/ ML": _yn(std.dent_metal_loss),
        "Cluster ID": std.vendor_cluster_id.to_numpy(),
        "Latitude": std.lat.round(6).to_numpy(),
        "Longitude": std.lon.round(6).to_numpy(),
        "Elevation (ft)": std.elev_ft.round(1).to_numpy(),
        "Comments": std.comments.to_numpy(),
    })


def to_vendor_b(std: pd.DataFrame) -> pd.DataFrame:
    """Vendor B: metres and millimetres, degrees, coded vocabulary, 1/0 flags, own column order."""
    feature = std.feature_type.map(_B_EVENT)
    feature = feature.where(std.vendor_cluster_id.isna(), "ML-CLUS")
    m_per_ft, mm_per_in = 1 / FT_PER_M, 1 / IN_PER_MM
    return pd.DataFrame({
        "Log Dist [m]": (std.log_dist_ft * m_per_ft).round(3).to_numpy(),
        "Feature": feature.to_numpy(),
        "Feature ID": std.feature_id.to_numpy(),
        "Ref GW": pd.to_numeric(std.us_gw_id).astype("Int64").to_numpy(),
        "Rel. Dist [m]": (std.dist_from_us_gw_ft * m_per_ft).round(3).to_numpy(),
        "Joint Len [m]": (std.joint_length_ft * m_per_ft).round(3).to_numpy(),
        "WT [mm]": (std.wt_nom_in * mm_per_in).round(2).to_numpy(),
        "Peak Depth [%]": std.depth_pct.round(1).to_numpy(),
        "Dent Depth [mm]": (std.dent_depth_in * mm_per_in).round(2).to_numpy(),
        "Length [mm]": (std.length_in * mm_per_in).round(1).to_numpy(),
        "Width [mm]": (std.width_in * mm_per_in).round(1).to_numpy(),
        "Orientation [deg]": clock_min_to_degrees(std.clock_min.to_numpy(dtype=float, na_value=np.nan)),
        "Location": std.surface.to_numpy(),
        "Seam": _flag01(std.seam_interaction).to_numpy(),
        "Dent ML": _flag01(std.dent_metal_loss).to_numpy(),
        "Cluster": std.vendor_cluster_id.to_numpy(),
        "Lat": std.lat.round(6).to_numpy(),
        "Lon": std.lon.round(6).to_numpy(),
        "Elev [m]": (std.elev_ft * m_per_ft).round(2).to_numpy(),
        "Comment": std.comments.to_numpy(),
    })


# ---------------------------------------------------------------------------
# Reference files and ground truth
# ---------------------------------------------------------------------------

OD_IN = 12.75
GRADE = "X52"
SEAM_TYPE = "ERW"
INSTALL_YEAR = 1975
COATING = "coal tar"
REPLACEMENT_COATING = "FBE"
MOP_PSIG = (1440, 1200)  # upstream half, downstream half
N_HCA_SEGMENTS = 4
OMITTED_NEXT_INSPECTION = "SEG-03"
NEXT_ASSESSMENT_YEARS = {True: 5, False: 7}  # could_affect_hca -> years after current run
N_CROSSINGS = 3
N_DIGS = 15
DIG_DEPTH_SIGMA_PCT = 2.0
DIG_LENGTH_SIGMA_IN = 0.25


def _chain_to_curr_log(curr: ObservedRun, line: LineTruth):
    link = curr.weld_link[curr.weld_link.weld_status == "reported"]
    chain = line.welds.set_index("true_weld_idx").chainage_ft.loc[link.true_weld_idx].to_numpy()
    gw = curr.std[curr.std.feature_type == "girth_weld"].set_index("us_gw_id").log_dist_ft
    log = gw.loc[link.gw_id].to_numpy()
    return lambda c: np.interp(c, chain, log)


def _segments(bounds_chain, to_log) -> tuple[np.ndarray, np.ndarray]:
    edges = np.round(to_log(np.asarray(bounds_chain, dtype=float)), 3)
    return edges[:-1], edges[1:]


def build_reference(p: SynthParams, line: LineTruth, features: pd.DataFrame, curr: ObservedRun,
                    rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Synthetic reference files, keyed to current-run log distance (spec 5.2)."""
    to_log = _chain_to_curr_log(curr, line)
    prev_date, curr_date = run_dates(p)
    L = line.length_ft
    lo, hi = line.replacement
    repair_date = prev_date + (curr_date - prev_date) / 2

    seg = line.pipe_segments
    bounds = sorted(set(seg.start_ft) | set(seg.end_ft) | {lo, hi})
    mids = (np.array(bounds[:-1]) + np.array(bounds[1:])) / 2
    replaced = (mids > lo) & (mids < hi)
    starts, ends = _segments(bounds, to_log)
    pipe_properties = pd.DataFrame({
        "start_ft": starts, "end_ft": ends, "od_in": OD_IN,
        "wt_nom_in": seg.wt_nom_in.to_numpy()[np.searchsorted(seg.end_ft.to_numpy(), mids)],
        "grade": GRADE, "smys_psi": p.smys_psi, "seam_type": SEAM_TYPE,
        "install_year": np.where(replaced, repair_date.year, INSTALL_YEAR),
        "coating_type": np.where(replaced, REPLACEMENT_COATING, COATING),
    })

    starts, ends = _segments([0, L / 2, L], to_log)
    mop_profile = pd.DataFrame({"start_ft": starts, "end_ft": ends, "mop_psig": list(MOP_PSIG)})

    starts, ends = _segments(np.linspace(0, L, N_HCA_SEGMENTS + 1), to_log)
    seg_ids = [f"SEG-{i:02d}" for i in range(1, N_HCA_SEGMENTS + 1)]
    could_affect = [i % 2 == 0 for i in range(N_HCA_SEGMENTS)]
    hca_segments = pd.DataFrame({"start_ft": starts, "end_ft": ends, "segment_id": seg_ids,
                                 "could_affect_hca": could_affect})

    rep_lo, rep_hi = to_log(np.array([lo, hi]))
    repairs = pd.DataFrame({"start_ft": [rep_lo], "end_ft": [rep_hi], "repair_type": ["pipe_replacement"],
                            "repair_date": [repair_date.isoformat()],
                            "notes": [f"Synthetic {REPLACED_JOINTS}-joint replacement"]})

    # Crossings sit beside existing metal loss so crossing rules have something to find.
    ext = features[features.population == "ext_ml"]
    if len(ext) >= N_CROSSINGS:
        near = rng.choice(ext.chainage_ft.to_numpy(), N_CROSSINGS, replace=False)
    else:
        near = np.linspace(0, L, N_CROSSINGS + 2)[1:-1]
    crossings = pd.DataFrame({"log_dist_ft": np.sort(to_log(near + rng.uniform(-3, 3, N_CROSSINGS))),
                              "description": "Foreign pipeline crossing (synthetic)"})

    next_inspection = pd.DataFrame([
        {"segment_id": s,
         "planned_next_assessment_date": curr_date.replace(year=curr_date.year + NEXT_ASSESSMENT_YEARS[c]).isoformat()}
        for s, c in zip(seg_ids, could_affect) if s != OMITTED_NEXT_INSPECTION
    ])

    # Digs on the deepest individually reported current-run metal loss.
    link = curr.feat_link[curr.feat_link.status == "reported"].merge(features, on="true_id")
    link = link[link.feature_type == "metal_loss"]
    m = link.merge(curr.std[["feature_id", "depth_pct"]], on="feature_id").nlargest(N_DIGS, "depth_pct")
    digs = pd.DataFrame({
        "feature_id": m.feature_id.to_numpy(),
        "field_depth_pct": np.clip(m.depth_curr_true_pct + rng.normal(0, DIG_DEPTH_SIGMA_PCT, len(m)), 0, 100).round(1),
        "field_length_in": np.maximum(m.length_in + rng.normal(0, DIG_LENGTH_SIGMA_IN, len(m)), 0.1).round(2),
        "dig_date": [(curr_date + timedelta(days=int(d))).isoformat() for d in rng.integers(90, 300, len(m))],
    })
    return {
        "pipe_properties": pipe_properties,
        "mop_profile": mop_profile,
        "hca_segments": hca_segments,
        "repairs_between_runs": repairs,
        "foreign_crossings": crossings,
        "next_inspection": next_inspection,
        "digs": digs,
    }


_REPORTED = {"reported", "reported_in_cluster"}


def build_ground_truth(features: pd.DataFrame, prev: ObservedRun, curr: ObservedRun) -> pd.DataFrame:
    gt = features.drop(columns=["depth_prev_meas_pct", "depth_curr_meas_pct"])
    for which, run in (("prev", prev), ("curr", curr)):
        link = run.feat_link.set_index("true_id").loc[gt.true_id]
        gt[f"{which}_feature_id"] = link.feature_id.to_numpy()
        gt[f"{which}_status"] = link.status.to_numpy()
    in_prev = gt.prev_status.isin(_REPORTED)
    in_curr = gt.curr_status.isin(_REPORTED)
    clustered = (gt.prev_status == "reported_in_cluster") | (gt.curr_status == "reported_in_cluster")
    gt["truth_class"] = np.select(
        [in_prev & in_curr & clustered, in_prev & in_curr, ~in_prev & in_curr, in_prev & ~in_curr],
        ["matched_cluster", "matched_1to1", "new", "missing"],
        default="not_reported",
    )
    # One class per vendor cluster report: if any pit in it is matched, the whole report is
    # matched_cluster (pits below threshold in the other run keep that in their *_status).
    for which in ("prev", "curr"):
        in_cluster = gt[f"{which}_status"] == "reported_in_cluster"
        matched_ids = gt.loc[in_cluster & (gt.truth_class == "matched_cluster"), f"{which}_feature_id"]
        gt.loc[in_cluster & gt[f"{which}_feature_id"].isin(matched_ids), "truth_class"] = "matched_cluster"
    return gt


def build_weld_truth(line: LineTruth, prev: ObservedRun, curr: ObservedRun) -> pd.DataFrame:
    lo, hi = line.replacement
    w = line.welds
    return pd.DataFrame({
        "true_weld_idx": w.true_weld_idx.to_numpy(),
        "chainage_ft": w.chainage_ft.to_numpy(),
        "prev_gw_id": prev.weld_link.gw_id.to_numpy(),
        "curr_gw_id": curr.weld_link.gw_id.to_numpy(),
        "prev_status": prev.weld_link.weld_status.to_numpy(),
        "curr_status": curr.weld_link.weld_status.to_numpy(),
        "in_replacement": ((w.chainage_ft > lo) & (w.chainage_ft < hi)).to_numpy(),
    })


# ---------------------------------------------------------------------------
# Orchestration, output, CLI
# ---------------------------------------------------------------------------


@dataclass
class SynthResult:
    prev_std: pd.DataFrame
    curr_std: pd.DataFrame
    vendor_a: pd.DataFrame
    vendor_b: pd.DataFrame
    ground_truth: pd.DataFrame
    weld_truth: pd.DataFrame
    reference: dict[str, pd.DataFrame]
    params: SynthParams
    curr_drift: float


def generate(p: SynthParams) -> SynthResult:
    rng = np.random.default_rng(p.seed)
    line = build_line(p, rng)
    features = build_features(p, line, rng)
    prev = observe_run("prev", p, line, features, rng)
    missed_prev = set(prev.weld_link.loc[prev.weld_link.weld_status == "missed", "true_weld_idx"])
    curr = observe_run("curr", p, line, features, rng, exclude_welds=missed_prev)
    return SynthResult(
        prev_std=prev.std,
        curr_std=curr.std,
        vendor_a=to_vendor_a(prev.std),
        vendor_b=to_vendor_b(curr.std),
        ground_truth=build_ground_truth(features, prev, curr),
        weld_truth=build_weld_truth(line, prev, curr),
        reference=build_reference(p, line, features, curr, rng),
        params=p,
        curr_drift=curr.drift,
    )


def write_outputs(r: SynthResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def csv(rel: str, df: pd.DataFrame) -> None:
        df.to_csv(out_dir / rel, index=False, lineterminator="\n")
        paths[rel] = out_dir / rel

    csv("tally_prev_vendor_a.csv", r.vendor_a)
    rel = "tally_curr_vendor_b.xlsx"
    prev_date, curr_date = run_dates(r.params)
    with pd.ExcelWriter(out_dir / rel, engine="openpyxl") as xw:
        r.vendor_b.to_excel(xw, sheet_name="Pipe Tally", startrow=2, index=False)
        ws = xw.sheets["Pipe Tally"]
        ws["A1"] = "Synthetic Vendor B – MFL Run (SYNTHETIC DATA)"
        ws["A2"] = f"Run date: {curr_date.isoformat()}"
        ws.freeze_panes = "A4"
    paths[rel] = out_dir / rel
    csv("ground_truth.csv", r.ground_truth)
    csv("ground_truth_welds.csv", r.weld_truth)
    for name, df in r.reference.items():
        csv(f"reference/{name}.csv", df)

    params = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(r.params).items()}
    manifest = {
        "label": DRAFT_LABEL,
        "generator": f"ili.synth {ili.__version__}",
        "params": params,
        "run_dates": {"prev": prev_date.isoformat(), "curr": curr_date.isoformat()},
        "curr_drift": r.curr_drift,
        "sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items() if k.endswith(".csv")},
        "note": "xlsx bytes embed a timestamp, so only CSVs are hashed; compare xlsx by content.",
    }
    rel = "synth_manifest.json"
    (out_dir / rel).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths[rel] = out_dir / rel
    return paths


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.synth", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"), help="output directory")
    for fld in fields(SynthParams):
        default = getattr(SynthParams, fld.name)
        kind = date.fromisoformat if isinstance(default, date) else type(default)
        ap.add_argument(f"--{fld.name.replace('_', '-')}", dest=fld.name, type=kind, default=default,
                        help=f"default: {default}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = vars(_parser().parse_args(argv))
    out = args.pop("out")
    try:
        result = generate(SynthParams(**args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_outputs(result, out)
    print(DRAFT_LABEL)
    print(f"wrote {out}")
    print(f"current-run odometer drift: {result.curr_drift:+.4%}")
    print("ground truth by class:")
    print(result.ground_truth.truth_class.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
