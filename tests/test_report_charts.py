import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ili import report_charts
from ili.classify import CATEGORY_PRIORITY
from ili.report_data import BacktestResult, dig_list
from test_report_inputs import _load, built  # noqa: F401  (built is the module fixture)


def _axis_titles(fig):
    """The x and y titles; for a multi-panel figure the bottom panel's x title and any panel's y title."""
    xs = [a.title.text for a in fig.select_xaxes() if a.title.text]
    ys = [a.title.text for a in fig.select_yaxes() if a.title.text]
    return (xs[-1] if xs else None), (ys[0] if ys else None)


def _luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_each_chart_has_data_and_axis_titles(built):  # noqa: F811
    inputs = _load(built)
    digs = dig_list(inputs)
    figs = {
        "depth": report_charts.depth_vs_distance(inputs),
        "growth": report_charts.growth_vs_distance(inputs),
        "hist": report_charts.growth_histograms(inputs),
        "rl": report_charts.rl_distribution(inputs),
        "map": report_charts.feature_map(inputs, digs),
    }
    for name, fig in figs.items():
        assert isinstance(fig, go.Figure), name
        assert fig.layout.title.text, name
        assert any(len(t.x) for t in fig.data), name
        assert all(_axis_titles(fig)), name
    depth = figs["depth"]
    assert [t.name for t in depth.data] == ["2021_vendorA", "2026_vendorB"]
    assert "each run's odometer" in depth.layout.xaxis.title.text
    assert {t.name for t in figs["growth"].data} == {"governing_mpy", "raw_mpy"}
    assert figs["map"].layout.yaxis.scaleanchor == "x"
    assert not set(t.name for t in figs["map"].data if t.name in CATEGORY_PRIORITY) - set(digs.governing_category)


def test_map_none_without_gps(built):  # noqa: F811
    inputs = _load(built)
    digs = dig_list(inputs)
    assert not digs.empty
    assert report_charts.feature_map(inputs, digs.assign(lat=np.nan)) is None
    assert report_charts.feature_map(inputs, digs.assign(lon=np.nan)) is None


def _backtest():
    table = pd.DataFrame({"feature_id": ["a", "b"], "field_depth_pct": [20.0, 40.0], "ili_depth_pct": [25.0, 38.0],
                          "diff_pct": [5.0, -2.0], "within_tolerance": [True, True]})
    return BacktestResult(table, 2, 100.0, [], [])


def test_unity_none_without_digs():
    assert report_charts.unity_chart(None, 10.0) is None


def test_unity_lines():
    fig = report_charts.unity_chart(_backtest(), 10.0)
    assert _axis_titles(fig) == ("Field depth (% wall)", "ILI depth (% wall)")
    lines = {t.name: t for t in fig.data if t.mode == "lines"}
    assert len(lines) == 3
    for name, t in lines.items():
        offset = {"1:1": 0.0, "+10 % wall": 10.0, "-10 % wall": -10.0}[name]
        assert np.allclose(np.array(t.y) - np.array(t.x), offset)
    points = next(t for t in fig.data if t.mode == "markers")
    assert list(points.x) == [20.0, 40.0] and list(points.y) == [25.0, 38.0]


def test_category_colours_legible_under_black_text():
    assert list(report_charts.CATEGORY_COLOURS) == list(CATEGORY_PRIORITY)
    for colour in report_charts.CATEGORY_COLOURS.values():
        assert len(colour) == 7 and colour[0] == "#"
        assert (_luminance(colour) + 0.05) / 0.05 >= 4.5, colour
    assert len(set(report_charts.CATEGORY_COLOURS.values())) == 5


def test_figure_div_has_no_script_src(built):  # noqa: F811
    inputs = _load(built)
    html = report_charts.figure_div(report_charts.rl_distribution(inputs))
    assert 'src="http' not in html and "https://" not in html.replace("http://www.w3.org", "")
    assert "plotly.js v" not in html and len(html) < 1_000_000
    assert "<html" not in html


def test_histogram_bins_keep_negative_rates(built):  # noqa: F811
    inputs = _load(built)
    growth = inputs.growth.copy()
    growth.loc[growth.index[:5], "raw_mpy"] = [-3.7, -0.2, -12.4, -1.0, -5.5]
    fig = report_charts.growth_histograms(dataclasses.replace(inputs, growth=growth))
    bin_mpy = inputs.settings.growth.diagnostic_bin_mpy
    for trace in fig.data:
        low = np.min(trace.x)
        assert trace.xbins.start <= low
        assert np.isclose(trace.xbins.start / bin_mpy, round(trace.xbins.start / bin_mpy))
    assert min(t.xbins.start for t in fig.data) <= -12.4


def test_no_webgl_traces(built):  # noqa: F811
    inputs = _load(built)
    figs = [report_charts.depth_vs_distance(inputs), report_charts.growth_vs_distance(inputs),
            report_charts.growth_histograms(inputs), report_charts.rl_distribution(inputs),
            report_charts.feature_map(inputs, dig_list(inputs)), report_charts.unity_chart(_backtest(), 10.0)]
    assert all(t.type != "scattergl" for fig in figs for t in fig.data)


def test_empty_run_tables(built):  # noqa: F811
    inputs = _load(built)
    empty = dataclasses.replace(inputs, prev_std=inputs.prev_std.iloc[0:0], curr_std=inputs.curr_std.iloc[0:0])
    fig = report_charts.depth_vs_distance(empty)
    assert [t.name for t in fig.data] == [inputs.summaries["match"]["prev_run_id"],
                                          inputs.summaries["match"]["curr_run_id"]]


def test_unity_without_finite_pairs():
    table = pd.DataFrame({"feature_id": ["a"], "field_depth_pct": [np.nan], "ili_depth_pct": [np.nan],
                          "diff_pct": [np.nan], "within_tolerance": [False]})
    fig = report_charts.unity_chart(BacktestResult(table, 0, 0.0, [], []), 10.0)
    line = next(t for t in fig.data if t.name == "1:1")
    assert list(line.x) == [0.0, 100.0]


def test_orientation_charts_one_per_category(built):  # noqa: F811
    inputs = _load(built)
    figs = report_charts.orientation_charts(inputs)
    assert list(figs) == ["ext_corrosion", "int_corrosion", "manufacturing"]
    for fig in figs.values():
        assert [t.name for t in fig.data] == ["2021_vendorA", "2026_vendorB"]
        assert tuple(fig.layout.yaxis.range) == (12, 0)
        assert all(_axis_titles(fig))
    ml = inputs.curr_std[inputs.curr_std.feature_type == "metal_loss"]
    plotted = sum(len(f.data[1].x) for f in figs.values())
    assert plotted == len(ml)  # the categories partition metal loss
    assert len(figs["int_corrosion"].data[1].x) > 0 and len(figs["manufacturing"].data[1].x) > 0


def _tiny(**cols):
    base = {"feature_type": "metal_loss", "surface": "EXT", "ml_origin": "corrosion", "log_dist_ft": 100.0,
            "clock_min": 360.0, "depth_pct": 50.0, "elev_ft": 10.0}
    return pd.DataFrame([{**base, **{k: v[i] for k, v in cols.items()}} for i in range(len(next(iter(cols.values()))))])


def test_clock_hours(built):  # noqa: F811
    inputs = _load(built)
    std = _tiny(clock_min=[360.0, np.nan])
    fig = report_charts.orientation_charts(dataclasses.replace(inputs, prev_std=std, curr_std=std))["ext_corrosion"]
    assert fig.data[0].y[0] == 6.0 and np.isnan(fig.data[0].y[1])


def test_marker_size_scaled_by_depth(built):  # noqa: F811
    inputs = _load(built)
    std = _tiny(depth_pct=[0.0, 50.0, 100.0, 150.0, np.nan])
    fig = report_charts.orientation_charts(dataclasses.replace(inputs, prev_std=std, curr_std=std))["ext_corrosion"]
    lo, hi = report_charts.DEPTH_MARKER_MIN_PX, report_charts.DEPTH_MARKER_MAX_PX
    assert list(fig.data[0].marker.size) == [lo, (lo + hi) / 2, hi, hi, lo]


def test_elevation_none_without_elev(built):  # noqa: F811
    inputs = _load(built)
    assert isinstance(report_charts.elevation_chart(inputs), go.Figure)
    bare = dataclasses.replace(inputs, curr_std=inputs.curr_std.assign(elev_ft=np.nan))
    assert report_charts.elevation_chart(bare) is None


def test_elevation_marks_internal_metal_loss(built):  # noqa: F811
    inputs = _load(built)
    fig = report_charts.elevation_chart(inputs)
    line, marks = fig.data
    assert line.mode == "lines" and list(line.x) == sorted(line.x)
    ml = inputs.curr_std[(inputs.curr_std.feature_type == "metal_loss") & (inputs.curr_std.surface == "INT")
                         & inputs.curr_std.elev_ft.notna()]
    assert len(marks.x) == len(ml) > 0
