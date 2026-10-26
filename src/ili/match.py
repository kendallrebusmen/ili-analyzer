"""Anomaly matching between two ILI runs (spec section 6.4).

Placement and helpers: each anomaly is placed in the weld map by its log distance, positions
are carried between runs through the matched joints, and clock and time differences are
computed on the circular clock and in decimal years.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ili.config import Settings, ToolCfg
from ili.units import CLOCK_MINUTES, wrap_clock_min

ANOMALY_TYPES = ("metal_loss", "dent", "gouge")
_MATCHED = ("1:1", "2:1", "1:2")
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


def candidates(prev: pd.DataFrame, curr: pd.DataFrame, dt_yr: float, settings: Settings) -> pd.DataFrame:
    """Candidate pairs (positional ``prev_i``, ``curr_i``) with their metrics.

    Both features sit in matched rows, within the axial tolerance on the previous-run axis, with
    equal types, compatible surfaces (equal or either UNK), clocks within tolerance (or either
    missing), neither a vendor cluster report, and either the same row or adjacent rows with both
    features near the shared weld. Curr features are sorted by position and each prev feature
    searches a window, so no dense matrix is built."""
    mt = settings.matching
    pi_ok = np.flatnonzero(prev.in_matched_row.to_numpy(bool) & prev.vendor_cluster_id.isna().to_numpy())
    ci_ok = np.flatnonzero(curr.in_matched_row.to_numpy(bool) & curr.vendor_cluster_id.isna().to_numpy())
    pos = curr.pos_prev_ft.to_numpy(float)[ci_ok]
    order = np.argsort(pos, kind="stable")
    ci_sorted, pos_sorted = ci_ok[order], pos[order]
    reach = mt.axial_tol_in * FT_PER_IN * (1 + 1e-9)  # slack only widens the window; the exact test follows
    x = prev.pos_prev_ft.to_numpy(float)[pi_ok]
    lo = np.searchsorted(pos_sorted, x - reach, side="left")
    hi = np.searchsorted(pos_sorted, x + reach, side="right")
    n = hi - lo
    pi = np.repeat(pi_ok, n)
    ci = ci_sorted[np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]

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
    relative to the first member into (-360, 360], converted to arc inches (pi * OD), boxed, and the
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
        rel = (c - c[0] + 360) % CLOCK_MINUTES - 360
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
    env["multi"] = [len(m) >= 2 or bool(vendor[m].any()) for m in members]
    return env, members


def _box_gaps(p: pd.DataFrame, q: pd.DataFrame, od: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial gap (in) and circumferential gap (clock min) between aligned envelope rows; 0 on overlap,
    NaN circ gap when either clock is missing."""
    axial = np.abs(_num(p.pos_prev_ft) - _num(q.pos_prev_ft)) / FT_PER_IN - (_num(p.length_in) + _num(q.length_in)) / 2
    arc = _arc_in(clock_diff_min(p.clock_min, q.clock_min), od)
    circ = arc - (_num(p.width_in) + _num(q.width_in)) / 2
    return np.maximum(axial, 0.0), np.maximum(circ, 0.0) / (np.pi * od) * CLOCK_MINUTES


def match_clusters(prev: pd.DataFrame, curr: pd.DataFrame, dt_yr: float, settings: Settings):
    """Many-to-one matches between interaction groups of the features left after the 1:1 pass.

    Only metal_loss rows take part (others are ignored). Inputs are placed frames already filtered to unmatched features in matched rows, with ``od_in``.
    Groups are formed per run; a group is multi with >= 2 members or a vendor cluster report. A
    group pair is a candidate when one side is multi, the surfaces are equal or either UNK, and the
    box gaps (axial in inches; circumferential converted to minutes with the current OD, else the
    previous) are within ``axial_tol_in`` and ``clock_tol_min``. Scored by ``_metrics`` on the gaps
    and the envelopes' max depths, then assigned one-to-one. Returns (prev rows, curr rows, metrics)
    per match, rows being positional member indices in log order."""
    mt = settings.matching
    pk, ck = (np.flatnonzero((f.feature_type == "metal_loss").to_numpy()) for f in (prev, curr))
    if not len(pk) or not len(ck):
        return []
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
    sp, sq = p.surface.to_numpy(str), q.surface.to_numpy(str)
    od = np.where(np.isnan(_num(q.od_in)), _num(p.od_in), _num(q.od_in))
    axial, circ = _box_gaps(p, q, od)
    with np.errstate(invalid="ignore"):
        keep = (p.multi.to_numpy() | q.multi.to_numpy()) & ((sp == sq) | (sp == "UNK") | (sq == "UNK"))
        keep &= (axial <= mt.axial_tol_in) & ~(circ > mt.clock_tol_min)
    m = _metrics(axial, circ, p.depth_pct, q.depth_pct, p.wt_nom_in, q.wt_nom_in, dt_yr, settings)
    cands = pd.concat([pd.DataFrame({"prev_i": gi, "curr_i": gj}), m], axis=1)[keep]
    chosen = assign_one_to_one(cands.reset_index(drop=True))
    cols = ["axial_diff_in", "clock_diff_min", "depth_change_pct", "implausible", "score"]
    return [(pm[int(row.prev_i)], cm[int(row.curr_i)], row[cols]) for _, row in chosen.iterrows()]
