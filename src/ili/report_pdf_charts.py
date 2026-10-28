"""Static charts for the PDF report (``report_pdf``), drawn with matplotlib from the same report input tables
and in the same colours as the interactive HTML charts (``report_charts``). Nothing is recomputed and nothing is
fetched: each chart is rendered off-screen (Agg) to PNG bytes at ``DPI``.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # off-screen; no display or GUI toolkit needed

import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
import numpy as np  # noqa: E402

from ili.report_charts import (CLOCK_HOURS, CLOCK_LABELS, GROWTH_COLOURS, INK, MINUTES_PER_HOUR, MUTED,  # noqa: E402
                               ORIENTATION_TITLES, RUN_COLOURS, _metal_loss)
from ili.report_data import BacktestResult, ReportInputs  # noqa: E402
from ili.report_tables import category_masks  # noqa: E402
from ili.schema import FeatureType, Surface  # noqa: E402

DPI = 200  # print resolution of the embedded PNGs
WIDTH_IN = 6.9  # chart width (in): the portrait text frame
HEIGHT_IN = 2.9  # one chart's height (in)
FONT = "DejaVu Sans"  # ships with matplotlib, so the charts match the PDF text on any machine
GRID = "#E3E7EC"
MARKER_AREA = 14  # scatter marker area (pt^2)
DEPTH_AREA_MIN, DEPTH_AREA_MAX = 4.0, 40.0  # marker area (pt^2) at 0 % and at 100 % of wall
CHART_DEPTH_FULL_PCT = 100.0  # depth drawn at the largest marker; deeper features are clipped to it

plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7.5,
})


def _axes(title: str, x_label: str, y_label: str, height_in: float = HEIGHT_IN):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height_in))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _png(fig, layout: bool = True) -> bytes:
    """The figure as PNG bytes at DPI (laid out tight unless the caller already did), then closed."""
    buffer = io.BytesIO()
    if layout:
        fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=DPI)
    plt.close(fig)
    return buffer.getvalue()


def _run_ids(inputs: ReportInputs) -> tuple[str, str]:
    return str(inputs.summaries["match"]["prev_run_id"]), str(inputs.summaries["match"]["curr_run_id"])


def _depth_areas(depth_pct) -> np.ndarray:
    """Marker areas, linear in depth from DEPTH_AREA_MIN (0 % wall) to DEPTH_AREA_MAX (CHART_DEPTH_FULL_PCT,
    clipped); NaN depth gets the minimum."""
    fraction = np.clip(np.nan_to_num(np.asarray(depth_pct, dtype=float), nan=0.0) / CHART_DEPTH_FULL_PCT, 0.0, 1.0)
    return DEPTH_AREA_MIN + fraction * (DEPTH_AREA_MAX - DEPTH_AREA_MIN)


def depth_vs_distance(inputs: ReportInputs) -> bytes:
    fig, ax = _axes("Metal-loss depth against distance", "Log distance (ft; each run's odometer)", "Depth (% wall)")
    for std, run_id, colour in zip((inputs.prev_std, inputs.curr_std), _run_ids(inputs), RUN_COLOURS, strict=True):
        ml = _metal_loss(std)
        ax.scatter(ml.log_dist_ft, ml.depth_pct, s=MARKER_AREA, color=colour, alpha=0.6, linewidths=0, label=run_id)
    ax.legend(loc="upper right")
    return _png(fig)


def growth_vs_distance(inputs: ReportInputs) -> bytes:
    fig, ax = _axes("Growth rate against distance", "Current-run log distance (ft)", "Growth rate (mpy)")
    g = inputs.growth
    for column, colour in GROWTH_COLOURS.items():
        ax.scatter(g.curr_log_dist_ft, g[column], s=MARKER_AREA, color=colour, alpha=0.6, linewidths=0, label=column)
    ax.legend(loc="upper right")
    return _png(fig)


def rl_distribution(inputs: ReportInputs) -> bytes:
    fig, ax = _axes("Remaining life distribution", "Remaining life (years)", "Features")
    rl = inputs.remaining_life.rl_yr.to_numpy(float)
    rl = rl[np.isfinite(rl)]
    if len(rl):
        ax.hist(rl, bins="auto", color=GROWTH_COLOURS["governing_mpy"], edgecolor="white", linewidth=0.5)
    return _png(fig)


def orientation(inputs: ReportInputs) -> bytes:
    """The three clock-position charts (external corrosion, internal corrosion, manufacturing) stacked in one
    figure, clock 12:00 at the top, marker area by depth."""
    fig, axes = plt.subplots(len(ORIENTATION_TITLES), 1, figsize=(WIDTH_IN, 2.3 * len(ORIENTATION_TITLES)),
                             sharex=True)
    for ax, (key, label) in zip(axes, ORIENTATION_TITLES.items(), strict=True):
        for std, run_id, colour in zip((inputs.prev_std, inputs.curr_std), _run_ids(inputs), RUN_COLOURS,
                                       strict=True):
            ml = _metal_loss(std)
            rows = ml[category_masks(ml)[key]]
            ax.scatter(rows.log_dist_ft, rows.clock_min.to_numpy(float) / MINUTES_PER_HOUR,
                       s=_depth_areas(rows.depth_pct), color=colour, alpha=0.55, linewidths=0, label=run_id)
        ax.set_title(f"{label}: clock position against distance")
        ax.set_ylim(12, 0)
        ax.set_yticks(CLOCK_HOURS[::3], CLOCK_LABELS[::3])
        ax.set_ylabel("Clock")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Log distance (ft; each run's odometer)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, markerscale=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # room for the figure legend above the first panel
    return _png(fig, layout=False)


def elevation(inputs: ReportInputs) -> bytes | None:
    """The current run's girth-weld elevation with internal metal loss marked; None without elevations."""
    std = inputs.curr_std
    welds = std[(std.feature_type == FeatureType.girth_weld.value) & std.elev_ft.notna()].sort_values("log_dist_ft")
    if welds.empty:
        return None
    fig, ax = _axes("Elevation profile with internal metal loss", "Log distance (ft; current run's odometer)",
                    "Elevation (ft)")
    ax.plot(welds.log_dist_ft, welds.elev_ft, color=MUTED, linewidth=1.2, label="Girth-weld elevation")
    ml = _metal_loss(std)
    internal = ml[(ml.surface.astype(str) == Surface.INT.value) & ml.elev_ft.notna()]
    ax.scatter(internal.log_dist_ft, internal.elev_ft, s=_depth_areas(internal.depth_pct), color=RUN_COLOURS[1],
               alpha=0.8, linewidths=0, label="Internal metal loss", zorder=3)
    ax.legend(loc="upper right")
    return _png(fig)


def unity(backtest: BacktestResult | None, tolerance: float) -> bytes | None:
    """ILI depth against field depth with the 1:1 line and the tool tolerance band; None without digs."""
    if backtest is None:
        return None
    t = backtest.table
    fig, ax = _axes("ILI depth against field depth", "Field depth (% wall)", "ILI depth (% wall)", height_in=3.6)
    depths = np.concatenate([t.field_depth_pct.to_numpy(float), t.ili_depth_pct.to_numpy(float)])
    depths = depths[np.isfinite(depths)]
    top = float(depths.max()) * 1.05 if len(depths) else 100.0
    span = np.array([0.0, top])
    ax.plot(span, span, color=MUTED, linewidth=1.2, label="1:1")
    ax.fill_between(span, span - tolerance, span + tolerance, color=GRID, alpha=0.7,
                    label=f"±{tolerance:g} % wall")
    ax.scatter(t.field_depth_pct, t.ili_depth_pct, s=MARKER_AREA * 1.5, color=GROWTH_COLOURS["governing_mpy"],
               zorder=3, label="Digs")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    return _png(fig)
