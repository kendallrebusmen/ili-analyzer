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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ili.units import CLOCK_MINUTES, sigma_from_tolerance

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
