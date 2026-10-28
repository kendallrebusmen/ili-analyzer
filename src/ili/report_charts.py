"""Plotly figures for the HTML report (spec 6.10 item 7). Every figure is drawn from the report input tables
as they are; nothing is recomputed and nothing is fetched (no map tiles, fonts or CDN assets)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ili.classify import CATEGORY_PRIORITY
from ili.report_data import BacktestResult, ReportInputs
from ili.report_tables import category_masks
from ili.schema import FeatureType, Surface

# One fill per category, most intense first. Task 6 reuses these as xlsx row fills under black text, so
# each keeps a contrast ratio of at least 4.5 against black.
CATEGORY_COLOURS = dict(zip(CATEGORY_PRIORITY, ("#FF6B5E", "#FF9E4A", "#FFC857", "#FFE9A8", "#D5DADF"), strict=True))
RUN_COLOURS = ("#1F77B4", "#E8833A")  # previous run, current run
GROWTH_COLOURS = {"governing_mpy": "#1F77B4", "raw_mpy": "#9AA5B1"}
INK = "#2B2F36"
MUTED = "#6B7480"
MARKER_SIZE = 8
MARKER_LINE = {"width": 1, "color": "#FFFFFF"}
DEPTH_MARKER_MIN_PX = 4  # marker diameter at 0 % wall, and for a feature without a depth
DEPTH_MARKER_MAX_PX = 16  # marker diameter at DEPTH_MARKER_FULL_PCT
DEPTH_MARKER_FULL_PCT = 100.0  # depth (% wall) drawn at the largest marker; deeper features are clipped to it
MINUTES_PER_HOUR = 60.0
TEMPLATE = "simple_white"

DEPTH_AXIS = "Depth (% wall)"
GROWTH_AXIS = "Growth rate (mpy)"
CURR_DIST_AXIS = "Current-run log distance (ft)"


def _figure(title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title=title, xaxis_title=x_title, yaxis_title=y_title,
                      font={"color": INK}, legend={"title": None})
    return fig


def _metal_loss(std):
    return std[std.feature_type == "metal_loss"]


def depth_vs_distance(inputs: ReportInputs) -> go.Figure:
    fig = _figure("Metal-loss depth against distance", "Log distance (ft; each run's odometer)", DEPTH_AXIS)
    run_ids = (inputs.summaries["match"]["prev_run_id"], inputs.summaries["match"]["curr_run_id"])
    for std, run_id, colour in zip((inputs.prev_std, inputs.curr_std), run_ids, RUN_COLOURS, strict=True):
        ml = _metal_loss(std)
        fig.add_trace(go.Scatter(x=ml.log_dist_ft, y=ml.depth_pct, mode="markers", name=str(run_id),
                                   marker={"size": MARKER_SIZE, "color": colour, "line": MARKER_LINE, "opacity": 0.7}))
    return fig


def growth_vs_distance(inputs: ReportInputs) -> go.Figure:
    fig = _figure("Growth rate against distance", CURR_DIST_AXIS, GROWTH_AXIS)
    g = inputs.growth
    for column, colour in GROWTH_COLOURS.items():
        fig.add_trace(go.Scatter(x=g.curr_log_dist_ft, y=g[column], mode="markers", name=column,
                                   marker={"size": MARKER_SIZE, "color": colour, "line": MARKER_LINE,
                                           "opacity": 0.7}))
    return fig


def growth_histograms(inputs: ReportInputs) -> go.Figure:
    """Raw growth rates, one panel per segment so the bins line up for comparison."""
    bin_mpy = inputs.settings.growth.diagnostic_bin_mpy
    g = inputs.growth[inputs.growth.raw_mpy.notna()]
    seg_ids = sorted(g.segment_id.dropna().unique())
    start = bin_mpy * float(np.floor(min(g.raw_mpy.min(), 0.0) / bin_mpy)) if len(g) else 0.0  # keep negatives
    fig = make_subplots(rows=max(len(seg_ids), 1), cols=1, shared_xaxes=True, subplot_titles=[str(s) for s in seg_ids],
                        vertical_spacing=0.04)
    for row, seg in enumerate(seg_ids, start=1):
        fig.add_trace(go.Histogram(x=g.raw_mpy[g.segment_id == seg], name=str(seg), showlegend=False,
                                   xbins={"size": bin_mpy, "start": start}, marker={"color": GROWTH_COLOURS["governing_mpy"]}),
                      row=row, col=1)
        fig.update_yaxes(title_text="Features", row=row, col=1)
    fig.update_xaxes(title_text="Raw growth rate (mpy)", row=len(seg_ids) or 1, col=1)
    fig.update_layout(template=TEMPLATE, title="Raw growth rate by segment", font={"color": INK},
                      height=max(320, 220 * len(seg_ids)))
    return fig


def rl_distribution(inputs: ReportInputs) -> go.Figure:
    rl = inputs.remaining_life.rl_yr.astype(float)
    fig = _figure("Remaining life distribution", "Remaining life (years)", "Features")
    fig.add_trace(go.Histogram(x=rl[np.isfinite(rl)], name="rl_yr", showlegend=False,
                               marker={"color": GROWTH_COLOURS["governing_mpy"]}))
    return fig


def feature_map(inputs: ReportInputs, dig_list) -> go.Figure | None:
    """Dig-list features by category over the girth-weld path, in plain lon/lat at equal scale."""
    if not (dig_list.lat.notna() & dig_list.lon.notna()).any():
        return None
    fig = _figure("Dig-list features", "Longitude (deg)", "Latitude (deg)")
    welds = inputs.curr_std[inputs.curr_std.feature_type == "girth_weld"].sort_values("log_dist_ft")
    welds = welds[welds.lat.notna() & welds.lon.notna()]
    fig.add_trace(go.Scatter(x=welds.lon, y=welds.lat, mode="lines", name="Girth-weld path",
                             line={"width": 2, "color": MUTED}, hoverinfo="skip"))
    for category in CATEGORY_PRIORITY:
        rows = dig_list[(dig_list.governing_category == category) & dig_list.lat.notna() & dig_list.lon.notna()]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows.lon, y=rows.lat, mode="markers", name=category, text=rows.curr_feature_id,
            marker={"size": MARKER_SIZE + 2, "color": CATEGORY_COLOURS[category], "line": {"width": 1, "color": INK}}))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def unity_chart(backtest: BacktestResult | None, tolerance: float) -> go.Figure | None:
    """Field depth against ILI depth with the 1:1 line and the tool tolerance band."""
    if backtest is None:
        return None
    t = backtest.table
    fig = _figure("ILI depth against field depth", "Field depth (% wall)", "ILI depth (% wall)")
    depths = np.concatenate([t.field_depth_pct.to_numpy(float), t.ili_depth_pct.to_numpy(float)])
    depths = depths[np.isfinite(depths)]
    span = [0.0, depths.max() * 1.05 if len(depths) else 100.0]
    for name, offset, dash in (("1:1", 0.0, "solid"), (f"+{tolerance:g} % wall", tolerance, "dash"),
                               (f"-{tolerance:g} % wall", -tolerance, "dash")):
        fig.add_trace(go.Scatter(x=span, y=[v + offset for v in span], mode="lines", name=name,
                                 line={"width": 2, "dash": dash, "color": MUTED}, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t.field_depth_pct, y=t.ili_depth_pct, mode="markers", name="Digs", text=t.feature_id,
                             marker={"size": MARKER_SIZE, "color": GROWTH_COLOURS["governing_mpy"],
                                     "line": MARKER_LINE}))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


ORIENTATION_TITLES = {"ext_corrosion": "External corrosion", "int_corrosion": "Internal corrosion",
                      "manufacturing": "Manufacturing"}


def _depth_marker_sizes(depth_pct) -> np.ndarray:
    """Marker diameters in pixels, linear in depth from DEPTH_MARKER_MIN_PX (0 % wall) to DEPTH_MARKER_MAX_PX
    (DEPTH_MARKER_FULL_PCT, clipped); NaN depth gets the minimum."""
    fraction = np.clip(np.nan_to_num(np.asarray(depth_pct, dtype=float), nan=0.0) / DEPTH_MARKER_FULL_PCT, 0.0, 1.0)
    return DEPTH_MARKER_MIN_PX + fraction * (DEPTH_MARKER_MAX_PX - DEPTH_MARKER_MIN_PX)


def orientation_charts(inputs: ReportInputs) -> dict[str, go.Figure]:
    """Clock position (hours, 0-12, 12 at the top) against log distance, one figure per category (external
    corrosion, internal corrosion, manufacturing) with one trace per run and marker size by depth. A row without
    a clock position plots as a gap (NaN)."""
    run_ids = (inputs.summaries["match"]["prev_run_id"], inputs.summaries["match"]["curr_run_id"])
    figures = {}
    for key, label in ORIENTATION_TITLES.items():
        fig = _figure(f"{label}: clock position against distance", "Log distance (ft; each run's odometer)",
                      "Clock position (hours)")
        for std, run_id, colour in zip((inputs.prev_std, inputs.curr_std), run_ids, RUN_COLOURS, strict=True):
            ml = _metal_loss(std)
            rows = ml[category_masks(ml)[key]]
            fig.add_trace(go.Scatter(x=rows.log_dist_ft, y=rows.clock_min.to_numpy(float) / MINUTES_PER_HOUR,
                                     mode="markers", name=str(run_id),
                                     marker={"size": _depth_marker_sizes(rows.depth_pct), "color": colour,
                                             "line": MARKER_LINE, "opacity": 0.7}))
        fig.update_yaxes(range=[12, 0], dtick=1)
        figures[key] = fig
    return figures


def elevation_chart(inputs: ReportInputs) -> go.Figure | None:
    """The current run's girth-weld elevation (``elev_ft``) against log distance, with internal metal loss marked
    at its own ``elev_ft``, sized by depth. None when no girth weld has an elevation."""
    std = inputs.curr_std
    welds = std[(std.feature_type == FeatureType.girth_weld.value) & std.elev_ft.notna()].sort_values("log_dist_ft")
    if welds.empty:
        return None
    fig = _figure("Elevation profile with internal metal loss", "Log distance (ft; current run's odometer)",
                  "Elevation (ft)")
    fig.add_trace(go.Scatter(x=welds.log_dist_ft, y=welds.elev_ft, mode="lines", name="Girth-weld elevation",
                             line={"width": 2, "color": MUTED}, hoverinfo="skip"))
    ml = _metal_loss(std)
    internal = ml[(ml.surface.astype(str) == Surface.INT.value) & ml.elev_ft.notna()]
    fig.add_trace(go.Scatter(x=internal.log_dist_ft, y=internal.elev_ft, mode="markers", name="Internal metal loss",
                             marker={"size": _depth_marker_sizes(internal.depth_pct), "color": RUN_COLOURS[1],
                                     "line": MARKER_LINE, "opacity": 0.7}))
    return fig


def figure_div(fig: go.Figure) -> str:
    """The figure as an HTML fragment; the page supplies plotly.js once."""
    return fig.to_html(full_html=False, include_plotlyjs=False)
