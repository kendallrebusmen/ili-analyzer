"""Anomaly matching between two ILI runs (spec section 6.4).

Placement and helpers: each anomaly is placed in the weld map by its log distance, positions
are carried between runs through the matched joints, and clock and time differences are
computed on the circular clock and in decimal years.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ili.align_welds import DRAFT_LABEL, comparison_dir, load_weld_map
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT, Settings, ToolCfg, load_settings
from ili.ingest import load_standardized
from ili.reference import lookup
from ili.schema import ANOMALY_TYPES
from ili.units import CLOCK_MINUTES, wrap_clock_min
from ili.validation import score_matching, write_metrics

_MATCHED = ("1:1", "2:1", "1:2")
WELD_MAP_COLUMNS = ("match_type", "prev_start_ft", "curr_start_ft", "prev_length_ft", "curr_length_ft", "in_repair")
DAYS_PER_YEAR = 365.25
NONCANDIDATE_COST = 1e9  # cost of a non-candidate cell in the assignment; pairs chosen at it are dropped
FT_PER_IN = 1 / 12
MILS_PER_IN = 1000.0


def clock_diff_min(a, b) -> np.ndarray:
    """Circular clock difference in minutes (0 to CLOCK_MINUTES / 2); NaN where either is missing."""
    x, y = (pd.Series(v, dtype="Float64").to_numpy(dtype=float, na_value=np.nan) for v in (a, b))
    d = np.abs(x - y) % CLOCK_MINUTES
    return np.minimum(d, CLOCK_MINUTES - d)


def _run_date(std: pd.DataFrame):
    dates = std.run_date.dropna().unique()
    if len(dates) != 1 or std.run_date.isna().any():
        raise ValueError(f"each run needs exactly one run_date, found {list(std.run_date.unique())}")
    return dates[0]


def delta_t_years(prev_std: pd.DataFrame, curr_std: pd.DataFrame) -> float:
    days = (_run_date(curr_std) - _run_date(prev_std)).days
    if days <= 0:
        raise ValueError("the current run must be later than the previous run")
    return days / DAYS_PER_YEAR


class PositionMap:
    """Piecewise-linear map between the runs' log distances through matched joints."""

    def __init__(self, prev_knots: np.ndarray, curr_knots: np.ndarray):
        self._prev, self._curr = prev_knots, curr_knots

    @classmethod
    def from_weld_map(cls, wm: pd.DataFrame) -> PositionMap:
        m = wm[wm.match_type.isin(_MATCHED)]
        if m.empty:
            raise ValueError("the weld map has no matched joints")
        ps, cs = m.prev_start_ft.to_numpy(float), m.curr_start_ft.to_numpy(float)
        pe, ce = ps + m.prev_length_ft.to_numpy(float), cs + m.curr_length_ft.to_numpy(float)
        prev, curr = np.concatenate([ps, pe]), np.concatenate([cs, ce])
        order = np.lexsort((prev, curr))
        return cls(prev[order], curr[order])

    @staticmethod
    def _interp(x, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        slope = (dst[-1] - dst[0]) / (src[-1] - src[0])
        out = np.interp(x, src, dst)
        out = np.where(x < src[0], dst[0] + (x - src[0]) * slope, out)
        return np.where(x > src[-1], dst[-1] + (x - src[-1]) * slope, out)

    def to_prev(self, x_curr) -> np.ndarray:
        return self._interp(x_curr, self._curr, self._prev)

    def to_curr(self, x_prev) -> np.ndarray:
        order = np.argsort(self._prev, kind="stable")
        return self._interp(x_prev, self._prev[order], self._curr[order])


def place(std: pd.DataFrame, wm: pd.DataFrame, run: str, pmap: PositionMap) -> pd.DataFrame:
    """The run's anomalies with the weld-map row holding each one and both runs' positions.

    A row's own-run span is [start, start + length) in that run's columns; rows with no side in
    the run hold no features of it. ``edge_dist_ft`` is the distance to the nearer span end;
    ``span_start_ft`` / ``span_end_ft`` are the ends of the holding row's own-run span."""
    if run not in ("prev", "curr"):
        raise ValueError(f"run must be 'prev' or 'curr', got {run!r}")
    out = std[std.feature_type.isin(ANOMALY_TYPES)].reset_index(drop=True)
    x = out.log_dist_ft.to_numpy(dtype=float)
    start = wm[f"{run}_start_ft"].to_numpy(dtype=float)
    end = start + wm[f"{run}_length_ft"].to_numpy(dtype=float)
    rows = np.flatnonzero(~np.isnan(start) & ~np.isnan(end))
    rows = rows[np.argsort(start[rows], kind="stable")]
    k = np.searchsorted(start[rows], x, side="right") - 1
    cand = rows[np.maximum(k, 0)]
    hit = (k >= 0) & (x < end[cand])
    wm_row = np.where(hit, cand, -1)
    out["wm_row"] = wm_row
    out["in_matched_row"] = hit & wm.match_type.isin(_MATCHED).to_numpy()[cand]
    mapped = pmap.to_curr(x) if run == "prev" else pmap.to_prev(x)
    out["pos_prev_ft"], out["pos_curr_ft"] = (x, mapped) if run == "prev" else (mapped, x)
    out["span_start_ft"] = np.where(hit, start[cand], np.nan)
    out["span_end_ft"] = np.where(hit, end[cand], np.nan)
    out["edge_dist_ft"] = np.where(hit, np.minimum(x - start[cand], end[cand] - x), np.nan)
    return out


def depth_change_tol_pct(tool: ToolCfg) -> float:
    """Tolerance of a two-run depth difference: sqrt(2) times the single-run tolerance (% of wall)."""
    return float(np.sqrt(2.0) * tool.depth_tolerance_pct_wt)


def _num(series) -> np.ndarray:
    return pd.Series(series, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)


def _metrics(axial, clock, d_prev, d_curr, wt_prev, wt_curr, dt_yr: float, settings: Settings) -> pd.DataFrame:
    """Score and plausibility from axial (in) and clock (min) differences and the two depths.

    score = sqrt((w_axial a^2 + w_circ c^2) / (w_axial + w_circ)) + w_depth_change * implausible,
    with a, c the axial and clock differences in units of their tolerances; a missing clock
    drops the circ term and its weight. Implausible: a depth drop beyond the two-run tolerance,
    or growth beyond it that exceeds the plausible rate (current-run nominal wall, else previous)."""
    mt = settings.matching
    w = mt.weights
    a, c = axial / mt.axial_tol_in, clock / mt.clock_tol_min
    has_clock = ~np.isnan(c)
    num = w.axial * a**2 + np.where(has_clock, w.circ * np.nan_to_num(c) ** 2, 0.0)
    den = w.axial + np.where(has_clock, w.circ, 0.0)
    delta = _num(d_curr) - _num(d_prev)
    tol = depth_change_tol_pct(settings.tool)
    wt = _num(wt_curr)
    wt = np.where(np.isnan(wt), _num(wt_prev), wt)
    with np.errstate(invalid="ignore"):  # NaN compares False: a missing depth is never implausible
        growth_mpy = (delta - tol) / 100.0 * wt * MILS_PER_IN / dt_yr
        implausible = (delta < -tol) | (growth_mpy > mt.max_plausible_growth_mpy)
    return pd.DataFrame({
        "axial_diff_in": axial,
        "clock_diff_min": clock,
        "depth_change_pct": delta,
        "implausible": implausible,
        "score": np.sqrt(num / den) + w.depth_change * implausible,
    })


def pair_metrics(prev: pd.DataFrame, curr: pd.DataFrame, dt_yr: float, settings: Settings) -> pd.DataFrame:
    """Per-pair differences, plausibility flag and score for aligned (equal-length) placed frames
    (rule in ``_metrics``; axial and clock differences are between the features themselves)."""
    axial = np.abs(_num(prev.pos_prev_ft) - _num(curr.pos_prev_ft)) / FT_PER_IN
    clock = clock_diff_min(prev.clock_min, curr.clock_min)
    return _metrics(axial, clock, prev.depth_pct, curr.depth_pct, prev.wt_nom_in, curr.wt_nom_in, dt_yr, settings)


def _near_shared_weld(p: pd.DataFrame, q: pd.DataFrame, pi, qi, limit_ft: float) -> np.ndarray:
    """Pairs in adjacent rows: both features within ``limit_ft`` of the weld between the rows."""
    p_later = q.wm_row.to_numpy()[qi] > p.wm_row.to_numpy()[pi]
    px, qx = p.log_dist_ft.to_numpy(float)[pi], q.log_dist_ft.to_numpy(float)[qi]
    dp = np.where(p_later, p.span_end_ft.to_numpy()[pi] - px, px - p.span_start_ft.to_numpy()[pi])
    dq = np.where(p_later, qx - q.span_start_ft.to_numpy()[qi], q.span_end_ft.to_numpy()[qi] - qx)
    return (dp <= limit_ft) & (dq <= limit_ft)


def _window(sorted_pos: np.ndarray, x: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """All (k, j) with sorted_pos[j] within ``reach`` of x[k]: positional pairs, no dense matrix."""
    lo = np.searchsorted(sorted_pos, x - reach, side="left")
    hi = np.searchsorted(sorted_pos, x + reach, side="right")
    n = hi - lo
    return np.repeat(np.arange(len(x)), n), np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)


def _near_reports(f: pd.DataFrame, reports: pd.DataFrame, f_is_curr: bool, settings: Settings) -> np.ndarray:
    """Per row of ``f``: a metal-loss feature in a matched row whose box interacts with the box of a
    vendor cluster report of the other run (``reports``), on the previous-run axis."""
    rule = settings.matching.interaction_rule
    out = np.zeros(len(f), bool)
    ml = (f.feature_type == "metal_loss").to_numpy() & f.in_matched_row.to_numpy(bool)
    is_rep = ((reports.feature_type == "metal_loss").to_numpy() & reports.in_matched_row.to_numpy(bool)
              & reports.vendor_cluster_id.notna().to_numpy())
    fi, ri = np.flatnonzero(ml), np.flatnonzero(is_rep)
    if not len(fi) or not len(ri):
        return out
    length_f, length_r = (np.nan_to_num(_num(d.length_in)) for d in (f, reports))
    width_f, width_r = (np.nan_to_num(_num(d.width_in)) for d in (f, reports))
    pos_f = _num(f.pos_prev_ft)[fi]
    order = np.argsort(pos_f, kind="stable")
    reach = (rule.axial_spacing_in + (length_f[fi].max() + length_r[ri].max()) / 2) * FT_PER_IN * (1 + 1e-9)
    k, j = _window(pos_f[order], _num(reports.pos_prev_ft)[ri], reach)
    r, i = ri[k], fi[order[j]]
    axial_gap = (np.abs(_num(reports.pos_prev_ft)[r] - _num(f.pos_prev_ft)[i]) / FT_PER_IN
                 - (length_r[r] + length_f[i]) / 2)
    od_f, od_r = _num(f.od_in)[i], _num(reports.od_in)[r]
    od_c, od_p = (od_f, od_r) if f_is_curr else (od_r, od_f)
    od = np.where(np.isnan(od_c), od_p, od_c)  # the current-run OD, else the previous
    circ_gap = (_arc_in(clock_diff_min(_num(reports.clock_min)[r], _num(f.clock_min)[i]), od)
                - (width_r[r] + width_f[i]) / 2)
    with np.errstate(invalid="ignore"):  # a missing clock or OD leaves the circumferential test passing
        ok = ((_surface(reports)[r] == _surface(f)[i]) & (axial_gap <= rule.axial_spacing_in)
              & ~(circ_gap > rule.circ_spacing_in))
    out[i[ok]] = True
    return out


def near_vendor_reports(prev: pd.DataFrame, curr: pd.DataFrame, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (prev, curr) of features to withhold from the 1:1 pass.

    A metal-loss feature in a matched row is withheld when its box interacts with the box of a vendor
    cluster report of the other run: the interaction rule (same surface, UNK only with UNK; axial
    edge gap within ``axial_spacing_in``; circumferential edge gap, through the current-run OD,
    within ``circ_spacing_in``; a missing clock or OD passing), compared on the previous-run axis.
    The report claims such pits, so they go to the cluster pass with it rather than being taken
    one-to-one by a nearby individual feature."""
    return _near_reports(prev, curr, False, settings), _near_reports(curr, prev, True, settings)


def candidates(prev: pd.DataFrame, curr: pd.DataFrame, dt_yr: float, settings: Settings,
               withheld: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Candidate pairs (positional ``prev_i``, ``curr_i``) with their metrics.

    Both features sit in matched rows, within the axial tolerance on the previous-run axis, with
    equal types, compatible surfaces (equal or either UNK), clocks within tolerance (or either
    missing), neither a vendor cluster report nor ``withheld`` (prev mask, curr mask), and either
    the same row or adjacent rows with both features near the shared weld. Curr features are
    sorted by position and each prev feature searches a window, so no dense matrix is built."""
    mt = settings.matching
    skip_p, skip_c = withheld if withheld is not None else (np.zeros(len(prev), bool), np.zeros(len(curr), bool))
    pi_ok = np.flatnonzero(prev.in_matched_row.to_numpy(bool) & prev.vendor_cluster_id.isna().to_numpy() & ~skip_p)
    ci_ok = np.flatnonzero(curr.in_matched_row.to_numpy(bool) & curr.vendor_cluster_id.isna().to_numpy() & ~skip_c)
    pos = curr.pos_prev_ft.to_numpy(float)[ci_ok]
    order = np.argsort(pos, kind="stable")
    ci_sorted, pos_sorted = ci_ok[order], pos[order]
    reach = mt.axial_tol_in * FT_PER_IN * (1 + 1e-9)  # slack only widens the window; the exact test follows
    k, j = _window(pos_sorted, prev.pos_prev_ft.to_numpy(float)[pi_ok], reach)
    pi, ci = pi_ok[k], ci_sorted[j]

    def surface(df):
        return df.surface.fillna("UNK").astype(str).to_numpy()

    keep = prev.feature_type.to_numpy()[pi] == curr.feature_type.to_numpy()[ci]
    sp, sc = surface(prev)[pi], surface(curr)[ci]
    keep &= (sp == sc) | (sp == "UNK") | (sc == "UNK")
    gap = np.abs(prev.wm_row.to_numpy()[pi] - curr.wm_row.to_numpy()[ci])
    keep &= gap <= 1
    cross = np.flatnonzero(keep & (gap == 1))
    if len(cross):
        keep[cross[~_near_shared_weld(prev, curr, pi[cross], ci[cross], mt.near_weld_ft)]] = False
    pi, ci = pi[keep], ci[keep]
    m = pair_metrics(prev.iloc[pi].reset_index(drop=True), curr.iloc[ci].reset_index(drop=True), dt_yr, settings)
    ok = (m.axial_diff_in <= mt.axial_tol_in) & ~(m.clock_diff_min > mt.clock_tol_min)  # missing clock passes
    out = pd.concat([pd.DataFrame({"prev_i": pi, "curr_i": ci}), m], axis=1)[ok.to_numpy()]
    return out.sort_values(["prev_i", "curr_i"], kind="stable").reset_index(drop=True)


def assign_one_to_one(cands: pd.DataFrame) -> pd.DataFrame:
    """Chosen rows of ``cands``: minimum total score, each feature used once.

    The bipartite candidate graph is split into connected components; each is solved with
    ``linear_sum_assignment`` on a small dense matrix whose non-candidate cells cost
    NONCANDIDATE_COST (pairs assigned at that cost are dropped). Inputs are put in sorted order,
    so ties resolve the same way every run."""
    if cands.empty:
        return cands.copy()
    c = cands.sort_values(["prev_i", "curr_i"], kind="stable")
    _, pinv = np.unique(c.prev_i.to_numpy(), return_inverse=True)
    _, cinv = np.unique(c.curr_i.to_numpy(), return_inverse=True)
    n = pinv.max() + 1
    graph = coo_matrix((np.ones(len(c)), (pinv, n + cinv)), shape=(n + cinv.max() + 1,) * 2)
    _, label = connected_components(graph, directed=False)
    score = c.score.to_numpy(float)
    chosen = []
    comp = label[pinv]
    order = np.argsort(comp, kind="stable")  # rows grouped by component once, in sorted order within each
    for rows in np.split(order, np.flatnonzero(np.diff(comp[order])) + 1):
        r_ids, ri = np.unique(pinv[rows], return_inverse=True)
        c_ids, cj = np.unique(cinv[rows], return_inverse=True)
        cost = np.full((len(r_ids), len(c_ids)), NONCANDIDATE_COST)
        cost[ri, cj] = score[rows]
        r, k = linear_sum_assignment(cost)
        real = cost[r, k] < NONCANDIDATE_COST
        at = {(a, b): rows[i] for i, (a, b) in enumerate(zip(ri, cj))}
        chosen.extend(at[(a, b)] for a, b in zip(r[real], k[real]))
    return c.iloc[sorted(chosen)]


def _arc_in(minutes, od_in) -> np.ndarray:
    """Circumferential arc (in) of a clock-minute span on a pipe of outside diameter ``od_in``."""
    return minutes / CLOCK_MINUTES * np.pi * od_in


def _surface(df: pd.DataFrame) -> np.ndarray:
    return df.surface.fillna("UNK").astype(str).to_numpy()


def interaction_groups(f: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Group label per row: connected components of "interacts" within one run.

    Only metal_loss features interact (spec 10.6 clusters metal loss); every other row is its own
    singleton group. Two features interact when they share a surface (exactly; UNK only with UNK), their axial edge
    gap |dpos| * 12 - (L1 + L2) / 2 is within ``axial_spacing_in`` and their circumferential edge
    gap arc(dclock) - (W1 + W2) / 2 is within ``circ_spacing_in``. A missing clock or OD leaves
    the circumferential test passing; missing sizes count as 0. Neighbours come from a window
    over sorted positions, so no dense matrix is built."""
    rule = settings.matching.interaction_rule
    n = len(f)
    if n == 0:
        return np.zeros(0, dtype=int)
    pos = _num(f.log_dist_ft)
    length = np.nan_to_num(_num(f.length_in))
    width = np.nan_to_num(_num(f.width_in))
    clock, od = _num(f.clock_min), _num(f.od_in)
    surf = _surface(f)
    order = np.argsort(pos, kind="stable")
    ps = pos[order]
    reach = (rule.axial_spacing_in + length.max()) * FT_PER_IN * (1 + 1e-9)  # (L1 + L2) / 2 <= max L
    lo = np.arange(n)
    hi = np.searchsorted(ps, ps + reach, side="right")
    cnt = hi - lo - 1  # later neighbours only
    i = np.repeat(order, cnt)
    j = order[np.repeat(lo + 1, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    axial_gap = np.abs(pos[i] - pos[j]) / FT_PER_IN - (length[i] + length[j]) / 2
    circ_gap = _arc_in(clock_diff_min(clock[i], clock[j]), od[i]) - (width[i] + width[j]) / 2
    ml = (f.feature_type == "metal_loss").to_numpy()
    with np.errstate(invalid="ignore"):
        ok = ml[i] & ml[j] & (surf[i] == surf[j]) & (axial_gap <= rule.axial_spacing_in) & ~(circ_gap > rule.circ_spacing_in)
    graph = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def envelope(f: pd.DataFrame) -> dict:
    """Bounding box of the member rows in the previous-run frame, as one feature.

    Axial: member pos_prev_ft +- length / 24 ft, so the box spans the outer edges (pos_curr_ft the
    same way from the members' current-run positions). Circumferential: member clocks are unwrapped
    relative to the first member into (-CLOCK_MINUTES / 2, CLOCK_MINUTES / 2], converted to arc inches
    (pi * OD), boxed, and the
    centre converted back to minutes and wrapped (as ``synth._cluster_envelope``); a missing clock
    uses only the members that have one; with none (or no OD) ``clock_min`` is NaN and the width the widest member. Depth is the maximum; the surface
    is the first non-UNK member surface; ``wt_nom_in`` the first non-missing value."""
    half = np.nan_to_num(_num(f.length_in)) / 24  # ft

    def axial(col):
        x = _num(f[col])
        lo, hi = (x - half).min(), (x + half).max()
        return (lo + hi) / 2, (hi - lo) * 12

    centre_prev, length = axial("pos_prev_ft")
    centre_curr, _ = axial("pos_curr_ft")
    width = np.nan_to_num(_num(f.width_in))
    clock, od = _num(f.clock_min), _num(f.od_in)
    od = od[~np.isnan(od)][0] if (~np.isnan(od)).any() else np.nan
    has = ~np.isnan(clock)
    if not has.any() or np.isnan(od):
        clock_min, width_in = np.nan, width.max()
    else:  # box of the members that report a clock; clockless members add axial extent only
        c, w = clock[has], width[has]
        half_turn = CLOCK_MINUTES / 2
        rel = (c - c[0] + half_turn) % CLOCK_MINUTES - half_turn
        arc = _arc_in(rel, od)
        lo, hi = (arc - w / 2).min(), (arc + w / 2).max()
        clock_min = float(wrap_clock_min(c[0] + (lo + hi) / 2 / (np.pi * od) * CLOCK_MINUTES))
        width_in = hi - lo
    surf = [s for s in _surface(f) if s != "UNK"]
    wt = _num(f.wt_nom_in)
    wt = wt[~np.isnan(wt)]
    depth = _num(f.depth_pct)
    return {
        "pos_prev_ft": centre_prev, "pos_curr_ft": centre_curr, "length_in": length, "width_in": width_in,
        "clock_min": clock_min, "depth_pct": np.nan if np.isnan(depth).all() else np.nanmax(depth),
        "surface": surf[0] if surf else "UNK", "wt_nom_in": wt[0] if len(wt) else np.nan,
        "od_in": od,
    }


def _group_table(f: pd.DataFrame, settings: Settings):
    """Envelope table (one row per group, ordered by label), member positions and multi flags."""
    label = interaction_groups(f, settings)
    order = np.argsort(f.log_dist_ft.to_numpy(float), kind="stable")  # members in log order
    members = [order[label[order] == g].tolist() for g in range(label.max() + 1)] if len(f) else []
    env = pd.DataFrame([envelope(f.iloc[m]) for m in members])
    vendor = f.vendor_cluster_id.notna().to_numpy()
    env["vendor"] = [bool(vendor[m].any()) for m in members]
    env["multi"] = [len(m) >= 2 or v for m, v in zip(members, env["vendor"])]
    return env, members


def _box_gaps(p: pd.DataFrame, q: pd.DataFrame, od: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial gap (in) and circumferential gap (clock min) between aligned envelope rows; 0 on overlap,
    NaN circ gap when either clock is missing."""
    axial = np.abs(_num(p.pos_prev_ft) - _num(q.pos_prev_ft)) / FT_PER_IN - (_num(p.length_in) + _num(q.length_in)) / 2
    arc = _arc_in(clock_diff_min(p.clock_min, q.clock_min), od)
    circ = arc - (_num(p.width_in) + _num(q.width_in)) / 2
    return np.maximum(axial, 0.0), np.maximum(circ, 0.0) / (np.pi * od) * CLOCK_MINUTES


def _envelope_pairs(p: pd.DataFrame, q: pd.DataFrame, dt_yr: float, settings: Settings):
    """For aligned prev/curr envelope rows: whether each pair passes the box rule (compatible
    surfaces, axial box gap <= ``axial_tol_in``, circumferential gap in minutes, with the current
    OD else the previous, <= ``clock_tol_min``) and the ``_metrics`` of the gaps and max depths."""
    mt = settings.matching
    sp, sq = p.surface.to_numpy(str), q.surface.to_numpy(str)
    od = np.where(np.isnan(_num(q.od_in)), _num(p.od_in), _num(q.od_in))
    axial, circ = _box_gaps(p, q, od)
    with np.errstate(invalid="ignore"):
        ok = ((sp == sq) | (sp == "UNK") | (sq == "UNK")) & (axial <= mt.axial_tol_in) & ~(circ > mt.clock_tol_min)
    return ok, _metrics(axial, circ, p.depth_pct, q.depth_pct, p.wt_nom_in, q.wt_nom_in, dt_yr, settings)


def _absorb(pairs: list, pe: pd.DataFrame, ce: pd.DataFrame, dt_yr: float, settings: Settings) -> list:
    """Extend matched group pairs with the groups left unassigned.

    ``pairs`` holds (prev group, curr group) labels. A leftover group of one run joins the match
    whose opposite group holds a vendor cluster report and passes the box rule against it, the
    lowest-score match winning (ties: the earlier match). Returns [prev labels, curr labels,
    absorbed metrics] per match: one ``_metrics`` row per absorbed group, scored by its box gap to
    the host's original (pre-absorption) opposite group."""
    out = [[[a], [b], []] for a, b in pairs]
    for side, own, opp in ((0, pe, ce), (1, ce, pe)):
        used = {pair[side] for pair in pairs}
        left = [g for g in range(len(own)) if g not in used]
        hosts = [k for k, pair in enumerate(pairs) if opp.vendor.iat[pair[1 - side]]]
        if not left or not hosts:
            continue
        li, ki = np.repeat(left, len(hosts)), np.tile(hosts, len(left))
        oi = np.array([pairs[k][1 - side] for k in ki])
        a, b = own.iloc[li].reset_index(drop=True), opp.iloc[oi].reset_index(drop=True)
        ok, m = _envelope_pairs(a, b, dt_yr, settings) if side == 0 else _envelope_pairs(b, a, dt_yr, settings)
        score = np.where(ok, m.score.to_numpy(float), np.inf).reshape(len(left), len(hosts))
        for row, g in enumerate(left):
            best = int(np.argmin(score[row]))  # first minimum: the earlier match on ties
            if np.isfinite(score[row, best]):
                out[hosts[best]][side].append(g)
                out[hosts[best]][2].append(m.iloc[row * len(hosts) + best])
    return out


def match_clusters(prev: pd.DataFrame, curr: pd.DataFrame, dt_yr: float, settings: Settings):
    """Many-to-one matches between interaction groups of the features left after the 1:1 pass.

    Only metal_loss rows take part (others are ignored). Inputs are placed frames already filtered
    to unmatched features in matched rows, with ``od_in``. Groups are formed per run; a group is
    multi with >= 2 members or a vendor cluster report. A group pair is a candidate when one side
    is multi and it passes the box rule of ``_envelope_pairs`` (surfaces equal or either UNK; box
    gaps within ``axial_tol_in`` and ``clock_tol_min``); candidates are scored by ``_metrics`` on
    the gaps and the envelopes' max depths, then assigned one-to-one. A vendor report asserts that
    one envelope holds its pits, which tool noise can scatter beyond the interaction rule, so each
    group left unassigned then joins the match whose opposite group holds a vendor report and
    passes the box rule against it (lowest score wins; ties go to the earlier match). A match that
    absorbed groups keeps the worst evidence: depth_change_pct and implausible come from the merged
    envelopes' max depths (one member pit's depth says nothing against a whole envelope), and the
    score is the largest over the group pair and each absorbed group (its box gap to the host's
    original opposite group) of the ``_metrics`` score with that merged implausible flag; the axial
    and clock gaps shown are the largest of those components. Returns (prev rows, curr rows,
    metrics) per match, rows being positional member indices in log order; the metrics carry
    ``absorbed`` (the match absorbed at least one group)."""
    pk, ck = (np.flatnonzero((f.feature_type == "metal_loss").to_numpy()) for f in (prev, curr))
    if not len(pk) or not len(ck):
        return []
    mt = settings.matching
    pe, pm = _group_table(prev.iloc[pk], settings)
    ce, cm = _group_table(curr.iloc[ck], settings)
    pm, cm = [pk[m].tolist() for m in pm], [ck[m].tolist() for m in cm]
    ps = np.argsort(ce.pos_prev_ft.to_numpy(float), kind="stable")
    cpos = ce.pos_prev_ft.to_numpy(float)[ps]
    reach = (mt.axial_tol_in + (pe.length_in.max() + ce.length_in.max()) / 2) * FT_PER_IN * (1 + 1e-9)
    x = pe.pos_prev_ft.to_numpy(float)
    lo, hi = np.searchsorted(cpos, x - reach, side="left"), np.searchsorted(cpos, x + reach, side="right")
    n = hi - lo
    gi = np.repeat(np.arange(len(pe)), n)
    gj = ps[np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
    p, q = pe.iloc[gi].reset_index(drop=True), ce.iloc[gj].reset_index(drop=True)
    ok, m = _envelope_pairs(p, q, dt_yr, settings)
    keep = (p.multi.to_numpy() | q.multi.to_numpy()) & ok
    cands = pd.concat([pd.DataFrame({"prev_i": gi, "curr_i": gj}), m], axis=1)[keep]
    chosen = assign_one_to_one(cands.reset_index(drop=True))
    cols = ["axial_diff_in", "clock_diff_min", "depth_change_pct", "implausible", "score"]
    pairs = [(int(a), int(b)) for a, b in zip(chosen.prev_i, chosen.curr_i)]
    merged = _absorb(pairs, pe, ce, dt_yr, settings)
    found = []
    for (_, row), (gp, gc, extra) in zip(chosen.iterrows(), merged):
        if not extra:
            found.append((pm[gp[0]], cm[gc[0]], pd.Series({**row[cols].to_dict(), "absorbed": False})))
            continue
        rows_p, rows_c = (_log_order(f, [i for g in gs for i in members[g]])
                          for f, gs, members in ((prev, gp, pm), (curr, gc, cm)))
        env_p = pd.DataFrame([envelope(prev.iloc[rows_p])])
        env_c = pd.DataFrame([envelope(curr.iloc[rows_c])])
        final = _envelope_pairs(env_p, env_c, dt_yr, settings)[1].iloc[0]
        parts = pd.DataFrame([row[cols], *(e[cols] for e in extra)])
        gap_score = _num(parts.score) - mt.weights.depth_change * parts.implausible.astype(bool).to_numpy()
        clock = _num(parts.clock_diff_min)
        found.append((rows_p, rows_c, pd.Series({
            "axial_diff_in": float(_num(parts.axial_diff_in).max()),
            "clock_diff_min": float(np.nan if np.isnan(clock).all() else np.nanmax(clock)),
            "depth_change_pct": final.depth_change_pct, "implausible": bool(final.implausible),
            "score": float(gap_score.max() + mt.weights.depth_change * bool(final.implausible)), "absorbed": True,
        })))
    return found


def _log_order(f: pd.DataFrame, rows: list) -> list:
    """Positional rows sorted by log distance (ties by row), as ``_group_table`` orders members."""
    pos = f.log_dist_ft.to_numpy(float)
    return sorted(rows, key=lambda i: (pos[i], i))


# --- Whole-run matching and the match table ---------------------------------------------------

SIDE_FIELDS = (
    "feature_type", "log_dist_ft", "wm_row", "us_gw_id", "dist_from_us_gw_ft", "wt_nom_in", "depth_pct",
    "dent_depth_in", "length_in", "width_in", "clock_min", "surface", "seam_interaction", "dent_metal_loss",
    "vendor_cluster_id",
)
_FLOAT_FIELDS = ("log_dist_ft", "dist_from_us_gw_ft", "wt_nom_in", "depth_pct", "dent_depth_in", "length_in", "width_in")
_INT_FIELDS = ("wm_row", "clock_min")
_OBJECT_FIELDS = ("feature_type", "us_gw_id", "surface", "seam_interaction", "dent_metal_loss", "vendor_cluster_id")
METRIC_COLS = ["axial_diff_in", "clock_diff_min", "depth_change_pct", "implausible", "score"]
MATCH_TYPES = ("matched_1to1", "matched_cluster", "missing", "new")
CONFIDENCES = ("high", "medium", "low")
HINTS = ("in_repaired_section", "near_threshold", "unexplained")
REVIEW_PENDING = "pending"
_SIDE_EXTRA = ("pos_prev_ft", "feature_id", "n")
TABLE_COLUMNS = (
    ["match_id", "match_type", "score", "confidence", "review_status", "needs_review", "missing_hint",
     "n_prev", "n_curr", "absorbed", "prev_feature_id", "curr_feature_id"]
    + [f"{s}_{c}" for s in ("prev", "curr") for c in SIDE_FIELDS]
    + ["curr_pos_in_prev_ft"] + METRIC_COLS[:-1]
)


class MatchResult:
    """The match table (one row per match or unmatched feature) and its summary dict."""

    def __init__(self, match_table: pd.DataFrame, summary: dict):
        self.match_table, self.summary = match_table, summary


def _require_columns(df: pd.DataFrame, cols: tuple, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s) {missing}")


def lookup_od(pos_ft, pipe: pd.DataFrame, tol_ft: float) -> np.ndarray:
    """Outside diameter (in) at each current-run position; see ``reference.lookup`` for the rule."""
    return lookup(pipe, pos_ft, "od_in", tol_ft, what="pipe properties table")


def _with_od(placed: pd.DataFrame, pipe: pd.DataFrame, tol_ft: float) -> pd.DataFrame:
    """``od_in`` for features in matched rows, the only ones that use it; NaN elsewhere, so a run
    covering more line than the other (or than the pipe table) does not need OD there."""
    od = np.full(len(placed), np.nan)
    use = placed.in_matched_row.to_numpy(bool)
    od[use] = lookup_od(placed.pos_curr_ft.to_numpy(float)[use], pipe, tol_ft)
    return placed.assign(od_in=od)


def _feature_side(f: pd.DataFrame) -> pd.DataFrame:
    """Side frame of single features, one row each (their own values)."""
    return f[[*SIDE_FIELDS, "pos_prev_ft", "feature_id"]].reset_index(drop=True).assign(n=1)


def _any_flag(values: pd.Series):
    v = values.astype(object)
    return None if v.isna().all() else bool(v.fillna(False).astype(bool).any())


def _cluster_side_row(members: pd.DataFrame, run: str) -> dict:
    """One side of a cluster match: the members' envelope, ids joined (members are in log order)."""
    env = envelope(members)
    first = members.iloc[0]
    vendor = members.vendor_cluster_id.dropna()
    clock = np.nan if np.isnan(env["clock_min"]) else round(env["clock_min"]) % CLOCK_MINUTES
    return {
        "feature_type": "metal_loss", "log_dist_ft": env[f"pos_{run}_ft"], "wm_row": first.wm_row,
        "us_gw_id": first.us_gw_id, "dist_from_us_gw_ft": first.dist_from_us_gw_ft, "wt_nom_in": env["wt_nom_in"],
        "depth_pct": env["depth_pct"], "dent_depth_in": np.nan, "length_in": env["length_in"],
        "width_in": env["width_in"], "clock_min": clock, "surface": env["surface"],
        "seam_interaction": _any_flag(members.seam_interaction), "dent_metal_loss": _any_flag(members.dent_metal_loss),
        "vendor_cluster_id": vendor.iloc[0] if len(vendor) else None,
        "pos_prev_ft": env["pos_prev_ft"], "feature_id": ";".join(members.feature_id.astype(str)), "n": len(members),
    }


def _blank_side(n: int) -> pd.DataFrame:
    return pd.DataFrame(index=range(n), columns=[*SIDE_FIELDS, *_SIDE_EXTRA], dtype=object).assign(n=0)


def _part(match_type: str, prev: pd.DataFrame, curr: pd.DataFrame, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of one match type from aligned side frames (one row per match) and their metrics.

    Every part has the same columns, also when empty, so the table can always be assembled."""
    n = len(prev)
    out = pd.DataFrame({"match_type": [match_type] * n, "missing_hint": [None] * n,
                        "n_prev": prev.n.to_numpy(), "n_curr": curr.n.to_numpy(),
                        "prev_feature_id": prev.feature_id.to_numpy(), "curr_feature_id": curr.feature_id.to_numpy()})
    for s, side in (("prev", prev), ("curr", curr)):
        for c in SIDE_FIELDS:
            out[f"{s}_{c}"] = side[c].to_numpy()
    out["prev_pos_prev_ft"] = prev.pos_prev_ft.to_numpy()
    out["curr_pos_in_prev_ft"] = curr.pos_prev_ft.to_numpy()
    m = metrics.reset_index(drop=True) if metrics is not None else pd.DataFrame(index=range(n), columns=METRIC_COLS)
    for c in METRIC_COLS:
        out[c] = m[c].to_numpy()
    out["absorbed"] = m["absorbed"].to_numpy(bool) if "absorbed" in m.columns else np.zeros(n, bool)
    return out


def _cluster_part(prev: pd.DataFrame, curr: pd.DataFrame, found: list) -> pd.DataFrame:
    rows = [(_cluster_side_row(prev.iloc[p], "prev"), _cluster_side_row(curr.iloc[c], "curr")) for p, c, _ in found]
    sides = [pd.DataFrame([r[k] for r in rows]) for k in (0, 1)]
    return _part("matched_cluster", *sides, pd.DataFrame([m for _, _, m in found]))


def _repair_flags(prev: pd.DataFrame, wm: pd.DataFrame, repairs) -> np.ndarray:
    """Prev features whose row was removed in a repair, or whose current-run position is in a repairs span."""
    in_repair = wm.in_repair.fillna(False).astype(bool).to_numpy()
    removed = (wm.match_type == "removed").to_numpy() & in_repair
    row = prev.wm_row.to_numpy()
    flag = (row >= 0) & removed[np.maximum(row, 0)]
    if repairs is not None and len(repairs):
        x = prev.pos_curr_ft.to_numpy(float)[:, None]
        flag |= ((x >= repairs.start_ft.to_numpy(float)) & (x <= repairs.end_ft.to_numpy(float))).any(axis=1)
    return flag


def missing_hints(prev: pd.DataFrame, wm: pd.DataFrame, repairs, settings: Settings) -> np.ndarray:
    """Hint per prev feature: repaired section, else near the reporting threshold, else unexplained."""
    tool = settings.tool
    with np.errstate(invalid="ignore"):  # a missing depth is never near the threshold
        near = _num(prev.depth_pct) < tool.reporting_threshold_pct_wt + tool.depth_tolerance_pct_wt
    return np.select([_repair_flags(prev, wm, repairs), near], [HINTS[0], HINTS[1]], HINTS[2]).astype(object)


def _confidence(score: np.ndarray, settings: Settings) -> np.ndarray:
    bands = settings.matching.confidence_bands
    return np.select([score <= bands.high, score <= bands.medium], [CONFIDENCES[0], CONFIDENCES[1]], CONFIDENCES[2])


def _none_for_na(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None)


def _finish(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Sort by prev-frame position, number the rows, fill confidence/review fields and fix dtypes."""
    pp, cp = _num(table.pop("prev_pos_prev_ft")), _num(table.curr_pos_in_prev_ft)
    key = pd.DataFrame({"pos": np.where(np.isnan(pp), cp, pp), "p": table.prev_feature_id.fillna(""),
                        "c": table.curr_feature_id.fillna("")})
    table = table.iloc[key.sort_values(["pos", "p", "c"], kind="stable").index].reset_index(drop=True)
    table["match_id"] = pd.Series([f"M{i + 1:05d}" for i in range(len(table))], dtype="str")
    table["match_type"] = table.match_type.astype("str")
    matched = table.match_type.isin(("matched_1to1", "matched_cluster")).to_numpy()
    table["score"] = _num(table.score)
    absorbed = table.absorbed.astype(bool).to_numpy()
    conf = _confidence(np.nan_to_num(table.score), settings)
    conf = np.where(absorbed & (conf == CONFIDENCES[0]), CONFIDENCES[1], conf)  # absorbed: at best medium
    table["absorbed"] = absorbed
    table["confidence"] = pd.Series(np.where(matched, conf, None), dtype=object)
    table["review_status"] = pd.Series([REVIEW_PENDING] * len(table), dtype="str")
    table["needs_review"] = ((table.confidence == CONFIDENCES[2]) | (table.missing_hint == HINTS[2])
                             | table.absorbed).astype(bool)
    for s in ("prev", "curr"):
        for c in _FLOAT_FIELDS:
            table[f"{s}_{c}"] = _num(table[f"{s}_{c}"])
        for c in _INT_FIELDS:
            table[f"{s}_{c}"] = pd.array(_num(table[f"{s}_{c}"]).round(), dtype="Int64")
        for c in _OBJECT_FIELDS:
            table[f"{s}_{c}"] = _none_for_na(table[f"{s}_{c}"])
    for c in ("axial_diff_in", "clock_diff_min", "depth_change_pct", "curr_pos_in_prev_ft"):
        table[c] = _num(table[c])
    for c in ("prev_feature_id", "curr_feature_id", "implausible", "missing_hint"):
        table[c] = _none_for_na(table[c])
    table["n_prev"], table["n_curr"] = table.n_prev.astype("int64"), table.n_curr.astype("int64")
    return table[TABLE_COLUMNS]


def _run_id(std: pd.DataFrame):
    ids = std.run_id.dropna()
    return str(ids.iloc[0]) if len(ids) else None


def _summary(prev_std, curr_std, dt: float, table: pd.DataFrame, n_prev: int, n_curr: int) -> dict:
    def counts(col, keys):
        got = table[col].value_counts()
        return {k: int(got.get(k, 0)) for k in keys}
    return {
        "prev_run_id": _run_id(prev_std), "curr_run_id": _run_id(curr_std),
        "delta_t_years": dt, "n_prev_anomalies": n_prev, "n_curr_anomalies": n_curr,
        "counts_by_match_type": counts("match_type", MATCH_TYPES),
        "counts_by_confidence": counts("confidence", CONFIDENCES),
        "counts_by_hint": counts("missing_hint", HINTS),
        "n_absorbed": int(table.absorbed.sum()),
        "n_needs_review": int(table.needs_review.sum()),
    }


def _unmatched_part(f: pd.DataFrame, keep: np.ndarray, run: str, hints=None) -> pd.DataFrame:
    side, blank = _feature_side(f.iloc[keep]), _blank_side(len(keep))
    part = _part("missing", side, blank) if run == "prev" else _part("new", blank, side)
    if hints is not None:
        part["missing_hint"] = hints[keep]
    return part


def _unused(n: int, used) -> np.ndarray:
    mask = np.ones(n, bool)
    mask[used] = False
    return mask


def match_anomalies(prev_std, curr_std, weld_map, settings: Settings, pipe, repairs=None) -> MatchResult:
    """Match the anomalies of two runs: 1:1 pairs, then many-to-one clusters, then missing and new.

    Every anomaly of each run lands in exactly one row of the match table."""
    _require_columns(weld_map, WELD_MAP_COLUMNS, "weld map")
    if repairs is not None:
        _require_columns(repairs, ("start_ft", "end_ft"), "repairs")
    tol = settings.alignment.anchor_tol_ft
    pmap = PositionMap.from_weld_map(weld_map)
    prev = _with_od(place(prev_std, weld_map, "prev", pmap), pipe, tol)
    curr = _with_od(place(curr_std, weld_map, "curr", pmap), pipe, tol)
    dt = delta_t_years(prev_std, curr_std)
    chosen = assign_one_to_one(candidates(prev, curr, dt, settings, near_vendor_reports(prev, curr, settings)))
    used_p, used_c = chosen.prev_i.to_numpy(int), chosen.curr_i.to_numpy(int)
    parts = [_part("matched_1to1", _feature_side(prev.iloc[used_p]), _feature_side(curr.iloc[used_c]), chosen[METRIC_COLS])]
    left_p, left_c = _unused(len(prev), used_p), _unused(len(curr), used_c)
    pk, ck = (np.flatnonzero(m & f.in_matched_row.to_numpy(bool)) for m, f in ((left_p, prev), (left_c, curr)))
    found = match_clusters(prev.iloc[pk].reset_index(drop=True), curr.iloc[ck].reset_index(drop=True), dt, settings)
    found = [(pk[p], ck[c], m) for p, c, m in found]
    if found:
        parts.append(_cluster_part(prev, curr, found))
        left_p[np.concatenate([p for p, _, _ in found])] = False
        left_c[np.concatenate([c for _, c, _ in found])] = False
    hints = missing_hints(prev, weld_map, repairs, settings)
    parts.append(_unmatched_part(prev, np.flatnonzero(left_p), "prev", hints))
    parts.append(_unmatched_part(curr, np.flatnonzero(left_c), "curr"))
    table = _finish(pd.concat(parts, ignore_index=True), settings)
    return MatchResult(table, _summary(prev_std, curr_std, dt, table, len(prev), len(curr)))



# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def write_matches(result: MatchResult, out_dir: Path) -> dict[str, Path]:
    s = result.summary
    folder = comparison_dir(out_dir, s["prev_run_id"], s["curr_run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"match_table.csv": folder / "match_table.csv", "match_summary.json": folder / "match_summary.json"}
    result.match_table.to_csv(paths["match_table.csv"], index=False, lineterminator="\n")
    paths["match_summary.json"].write_text(json.dumps({"label": DRAFT_LABEL, **s}, indent=2), encoding="utf-8")
    return paths


_TABLE_TEXT = (
    "match_id", "match_type", "confidence", "review_status", "missing_hint", "prev_feature_id", "curr_feature_id",
    *(f"{s}_{c}" for s in ("prev", "curr") for c in ("feature_type", "us_gw_id", "surface", "vendor_cluster_id")),
)
_TABLE_FLAGS = (
    "needs_review", "absorbed", "implausible",
    *(f"{s}_{c}" for s in ("prev", "curr") for c in ("seam_interaction", "dent_metal_loss")),
)
_TABLE_INTS = ("n_prev", "n_curr", "prev_wm_row", "curr_wm_row", "prev_clock_min", "curr_clock_min")


def load_match_table(path: Path) -> pd.DataFrame:
    """Read a ``match_table.csv`` back with its types: ids and labels stay text (empty is None),
    flags are True/False/None, counts and clock/row numbers are Int64, the rest float."""
    header = pd.read_csv(path, nrows=0).columns
    text = [c for c in _TABLE_TEXT if c in header]
    flags = [c for c in _TABLE_FLAGS if c in header]
    ints = [c for c in _TABLE_INTS if c in header]
    df = pd.read_csv(path, dtype={c: str for c in text + flags}, keep_default_na=False,
                     na_values={c: [""] for c in header if c not in text + flags})
    for c in text:
        df[c] = df[c].astype(object).where(df[c] != "", None)
    for c in flags:
        df[c] = df[c].map({"True": True, "False": False}).astype(object).where(df[c] != "", None)
    for c in ints:
        df[c] = df[c].astype("Int64")
    return df


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.match",
                                 description="Match anomalies between two ILI runs. " + DRAFT_LABEL)
    ap.add_argument("--prev", type=Path, required=True, help="standardized CSV of the previous run")
    ap.add_argument("--curr", type=Path, required=True, help="standardized CSV of the current run")
    ap.add_argument("--weld-map", type=Path, required=True, help="weld_map.csv from ili.align_welds")
    ap.add_argument("--pipe", type=Path, required=True, help="pipe_properties.csv (current-run log distance)")
    ap.add_argument("--repairs", type=Path, help="repairs_between_runs.csv (current-run log distance)")
    ap.add_argument("--truth", type=Path, help="ground_truth.csv (synthetic data only) to score against")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs", help="output directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prev, curr = load_standardized(args.prev), load_standardized(args.curr)
        weld_map = load_weld_map(args.weld_map)
        pipe = pd.read_csv(args.pipe)
        repairs = pd.read_csv(args.repairs) if args.repairs else None
        result = match_anomalies(prev, curr, weld_map, load_settings(args.settings), pipe, repairs)
        paths = write_matches(result, args.out)
        score = None
        if args.truth:
            truth = pd.read_csv(args.truth, dtype={"prev_feature_id": str, "curr_feature_id": str})
            score = score_matching(result.match_table, truth)
            write_metrics("anomaly_matching", score, Path(args.out) / "validation_metrics.json")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    print(DRAFT_LABEL)
    print(f"{s['prev_run_id']} vs {s['curr_run_id']}: {s['n_prev_anomalies']} / {s['n_curr_anomalies']} anomalies, "
          f"delta t {s['delta_t_years']:.2f} years")
    print("matches: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_match_type"].items()))
    print("confidence: " + ", ".join(f"{k} {v}" for k, v in s["counts_by_confidence"].items()))
    print(f"needs review: {s['n_needs_review']} (absorbed clusters: {s['n_absorbed']})")
    if score is not None:
        print(f"precision {score['precision']:.4f}, recall {score['recall']:.4f} vs ground truth "
              f"({score['n_correct_links']}/{score['n_true_links']} links, "
              f"{len(score['false_links'])} false, passes: {score['passes']})")
        print(f"table check: {score['n_unreported']} unreported, {score['n_duplicated']} duplicated, "
              f"{score['n_wrong_side']} wrong side, {score['n_duplicate_links']} duplicate links")
    print(f"wrote {paths['match_table.csv'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
