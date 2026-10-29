"""The Nemi look: theme config, sidebar navigation, page header, status pills, and the app's layout."""

import json
import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ili import app_state, app_theme, report_data
from ili.review import REVIEWS_FILE, append_review
from ili.align_welds import DRAFT_LABEL
from test_app_review import _paired
from test_app_setup import APP, PAGES, SCRIPT_CONFIG, _app, isolated_outputs, ran, script_config  # noqa: F401 (fixtures)

HOSTILE = "<script>alert(1)</script>"


def test_palette_matches_config():
    config = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))
    theme = config["theme"]
    light, dark = theme["light"], theme["dark"]
    assert "base" not in theme  # one palette per theme; the reviewer or the computer picks which
    assert light["primaryColor"] == app_theme.PRIMARY == "#0F766E"
    assert dark["primaryColor"] == app_theme.DARK_PRIMARY == "#0F766E"
    assert light["backgroundColor"] == app_theme.PAGE_BACKGROUND == "#F6F7F8"
    assert light["secondaryBackgroundColor"] == app_theme.SURFACE == "#FFFFFF"
    assert light["textColor"] == app_theme.INK == "#1F2A33"
    assert light["linkColor"] == app_theme.LINK and light["borderColor"] == app_theme.LINE
    assert light["redColor"] == app_theme.DANGER == "#B42318"
    assert light["sidebar"]["backgroundColor"] == app_theme.SIDEBAR == "#EEF2F3"
    assert dark["backgroundColor"] == app_theme.DARK_PAGE_BACKGROUND == "#111518"
    assert dark["secondaryBackgroundColor"] == app_theme.DARK_SURFACE == "#1A2025"
    assert dark["textColor"] == app_theme.DARK_INK
    assert dark["linkColor"] == app_theme.DARK_LINK and dark["borderColor"] == app_theme.DARK_LINE
    assert dark["redColor"] == app_theme.DARK_DANGER
    assert dark["sidebar"]["backgroundColor"] == app_theme.DARK_SIDEBAR == "#151A1E"
    assert "http" not in theme["font"]  # system fonts only: the app makes no network calls
    assert config["server"]["address"] == "localhost" and config["browser"]["gatherUsageStats"] is False
    assert config["client"]["toolbarMode"] == "viewer"


def _luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


_T = app_theme
_PAIRS = []
for _ink, _bgs in [
    ((_T.INK, _T.DARK_INK), ((_T.PAGE_BACKGROUND, _T.DARK_PAGE_BACKGROUND), (_T.SURFACE, _T.DARK_SURFACE),
                             (_T.SIDEBAR, _T.DARK_SIDEBAR))),
    ((_T.MUTED, _T.DARK_MUTED), ((_T.PAGE_BACKGROUND, _T.DARK_PAGE_BACKGROUND), (_T.SURFACE, _T.DARK_SURFACE),
                                 (_T.SIDEBAR, _T.DARK_SIDEBAR))),
    # accent text: PRIMARY in light, LINK in dark (dark PRIMARY is for button fills, see the white-on pair below)
    ((_T.PRIMARY, _T.DARK_LINK), ((_T.PAGE_BACKGROUND, _T.DARK_PAGE_BACKGROUND), (_T.SURFACE, _T.DARK_SURFACE),
                                  (_T.SIDEBAR, _T.DARK_SIDEBAR))),
    ((_T.LINK, _T.DARK_LINK), ((_T.SURFACE, _T.DARK_SURFACE),)),
    (("#FFFFFF", "#FFFFFF"), ((_T.PRIMARY, _T.DARK_PRIMARY),)),  # Streamlit's primary buttons carry white text
    ((_T.WARNING_INK, _T.DARK_WARNING_INK), ((_T.WARNING_BG, _T.DARK_WARNING_BG),)),
    ((_T.DANGER, _T.DARK_DANGER), ((_T.DANGER_SOFT, _T.DARK_DANGER_SOFT), (_T.SURFACE, _T.DARK_SURFACE))),
    ((_T.SUCCESS, _T.DARK_SUCCESS), ((_T.SURFACE, _T.DARK_SURFACE),)),
]:
    for _bg in _bgs:
        _PAIRS += [(_ink[0], _bg[0]), (_ink[1], _bg[1])]


@pytest.mark.parametrize("ink, background", _PAIRS)
def test_palette_meets_aa(ink, background):
    assert _contrast(ink, background) >= 4.5  # WCAG AA for body text


def test_css_follows_the_active_theme():
    css = app_theme.APP_CSS
    for light, dark in [(app_theme.INK, app_theme.DARK_INK), (app_theme.LINE, app_theme.DARK_LINE),
                        (app_theme.SURFACE, app_theme.DARK_SURFACE), (app_theme.DANGER, app_theme.DARK_DANGER),
                        (app_theme.PRIMARY, app_theme.DARK_PRIMARY)]:
        assert f"light-dark({light}, {dark})" in css
    assert "--nemi-accent" in css
    rules = css.split("}", 1)[1]  # past the variable block, no hard-coded colour that vanishes in the other theme
    assert "background: #FFFFFF" not in rules and f"color: {app_theme.INK}" not in rules


def test_css_has_forced_colors_rules():
    css = app_theme.APP_CSS
    assert "@media (forced-colors: active)" in css and "outline: 2px solid Highlight" in css
    assert "forced-color-adjust: none" in css


def test_no_tab_css():
    assert "data-selected" not in app_theme.APP_CSS and "ili-breadcrumb" not in app_theme.APP_CSS


def test_css_defines_the_nemi_classes():
    for cls in ["nemi-wordmark", "nemi-header", "nemi-subtitle", "nemi-badge", "nemi-kpi", "nemi-kpi--alert",
                "nemi-attn", "nemi-attn-dot", "nemi-attn-dot--warn", "nemi-attn-dot--info", "nemi-attn-dot--ok",
                "nemi-facts", "nemi-step", "nemi-truncate"]:
        assert f".{cls}" in app_theme.APP_CSS


def test_style_chart():
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]), layout=go.Layout(font={"color": "#2B2F36"},
                                                                     template="simple_white"))
    out = app_theme.style_chart(fig)
    assert out is fig and out.layout.font.color is None
    assert out.layout.paper_bgcolor == out.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert out.layout.colorway[0] == app_theme.PRIMARY
    assert out.layout.xaxis.gridcolor == out.layout.yaxis.gridcolor == app_theme.CHART_GRID
    assert "#1F2A33" not in out.layout.colorway  # the dark ink would vanish on the dark page
    assert list(out.data[0].y) == [3, 4]


def test_page_header_html_has_title_subtitle_and_one_badge():
    html = app_theme.page_header_html(f"Match {HOSTILE}", f"Line {HOSTILE}")
    assert 'class="nemi-header"' in html and "<h1>" in html
    assert html.count("nemi-badge") == 1 and DRAFT_LABEL in html
    assert "<script>" not in html and html.count("&lt;script&gt;") == 3  # title, tooltip, subtitle
    assert "nemi-subtitle" not in app_theme.page_header_html("Run setup", None)


def test_long_names_truncate():
    html = app_theme.page_header_html("Home", "x" * 120)
    assert 'title="' + "x" * 120 + '"' in html and "nemi-truncate" in html and "nemi-subtitle" in html


def test_sidebar_brand_html():
    html = app_theme.sidebar_brand_html()
    assert 'class="nemi-wordmark"' in html and "<svg" in html and "<span>Nemi</span>" in html
    assert "<span>ILI run comparison</span>" in html


def test_breadcrumb_helpers_are_gone():
    for name in ("shell_bar_html", "breadcrumb_html", "_initials"):
        assert not hasattr(app_theme, name)


@pytest.mark.parametrize("tone", app_theme.PILL_TONES)
def test_pill_escapes_text(tone):
    html = app_theme.pill_html(HOSTILE, tone)
    assert f"ili-pill-{tone}" in html and "&lt;script&gt;" in html and "<script>" not in html


def test_unknown_pill_tone_is_an_error():
    with pytest.raises(ValueError, match="tone"):
        app_theme.pill_html("x", "loud")


@pytest.mark.parametrize("confidence, tone", [("low", "danger"), ("medium", "warning"), ("high", "success"),
                                              (None, "info")])
def test_confidence_pill_tone(confidence, tone):
    assert f"ili-pill-{tone}" in app_theme.confidence_pill_html(confidence)


def test_css_is_local_and_self_contained():
    css = app_theme.APP_CSS
    assert "http" not in css and "@import" not in css and "url(" not in css


def test_shell_sidebar():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.radio(key="page").value == "Home"
    assert at.sidebar.text_input(key="reviewer").value == ""
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "nemi-wordmark" in sidebar
    markdown = " ".join(m.value for m in at.markdown)
    assert "ili-shell" not in markdown and "ili-breadcrumb" not in markdown


@pytest.mark.parametrize("page", list(PAGES))
def test_page_header_on_every_page(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="page").set_value(page).run()
    assert not at.exception
    assert any("nemi-header" in m.value and DRAFT_LABEL in m.value for m in at.markdown)


def _match_review_label(at):
    return [o for o in at.sidebar.radio(key="page").options if o.startswith("Match review")][0]


def _pending_lines(at):
    return [m.value for m in at.sidebar.markdown if "nemi-pending" in m.value]


def _decide_first(work):
    """Record a decision and apply it to the table, standing in for the match stage's rerun."""
    row = _paired(work)
    append_review(work / REVIEWS_FILE, "accept", [row.prev_feature_id], [row.curr_feature_id], "A. Reviewer")
    table = pd.read_csv(work / "match_table.csv", dtype=str, keep_default_na=False)
    table.loc[table.match_id == row.match_id, "review_status"] = "accepted"
    table.to_csv(work / "match_table.csv", index=False)


def test_menu_labels_are_static_and_the_count_sits_under_the_menu(ran, tmp_path):
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    at = _app(comparison_dir=str(work)).run()
    n = len(report_data.pending_reviews(app_state.read_comparison(work).match_table))
    labels = list(at.sidebar.radio(key="page").options)
    assert n > 1 and _match_review_label(at) == "Match review"
    assert _pending_lines(at) == [f'<div class="nemi-pending">{n} matches waiting for review</div>']
    _decide_first(work)
    at.run()
    assert list(at.sidebar.radio(key="page").options) == labels  # identical labels for n and n - 1 pending
    assert _pending_lines(at) == [app_theme.sidebar_pending_html(n - 1)]
    assert at.sidebar.radio(key="page").value == "Home"


def test_pending_line_text_and_none_at_zero():
    assert app_theme.sidebar_pending_html(0) == ""
    assert "1 match waiting for review" in app_theme.sidebar_pending_html(1)
    assert "3 matches waiting for review" in app_theme.sidebar_pending_html(3)
    at = AppTest.from_file(APP, default_timeout=60).run()  # no comparison at all: 0 pending, no line, no error
    assert not at.exception and _pending_lines(at) == []


def test_review_page_survives_a_rerun_that_changes_the_pending_count(ran, tmp_path):
    """The browser keeps sending the label it last held; it must still match after the count changes."""
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    at = _app(comparison_dir=str(work)).run()
    at.radio(key="page").set_value("Match review queue").run()
    held = WidgetStates()
    for w in at._tree.get_widget_states().widgets:  # what the browser holds now
        if w.WhichOneof("value") != "trigger_value":
            held.widgets.append(w)
    _decide_first(work)
    at._run(held)  # a rerun with the count changed
    at._run(held)  # the reviewer's next interaction replays the held value
    assert at.session_state["page"] == "Match review queue" and not at.exception


def test_nav_labels_are_short_and_static():
    assert app_theme.nav_label_text("Match review queue") == "Match review"
    assert app_theme.nav_label_text("Growth and remaining life explorer") == "Growth & life"
    assert app_theme.nav_label_text("Dig list and report export") == "Dig list & export"
    assert app_theme.nav_label_text("Home") == "Home"



def test_app_menu_has_no_links_off_this_computer():
    source = Path(APP).read_text(encoding="utf-8")
    assert '"Get help": None' in source and '"Report a bug": None' in source


def test_object_header_escapes_title_and_facts():
    html = app_theme.object_header_html(f"Match {HOSTILE}", [app_theme.pill_html("Pending", "warning")],
                                        [("Hint", HOSTILE), ("Score", 1.11)])
    assert 'class="ili-object"' in html and "ili-pill-warning" in html and "<strong>1.11</strong>" in html
    assert "<script>" not in html and html.count("&lt;script&gt;") == 2


def test_comparison_context(tmp_path):
    assert app_state.comparison_context(None) is None
    assert app_state.comparison_context(tmp_path) is None  # no pipeline_inputs.json
    (tmp_path / "pipeline_inputs.json").write_text(json.dumps(
        {"line_name": "Line 930", "prev_run_date": "2021-05-01", "curr_run_date": "2026-05-02"}), encoding="utf-8")
    assert app_state.comparison_context(tmp_path) == "Line 930 · 2021-05-01 → 2026-05-02"
    (tmp_path / "pipeline_inputs.json").write_text(json.dumps(
        {"prev_run_date": "2021-05-01", "curr_run_date": "2026-05-02"}), encoding="utf-8")
    assert app_state.comparison_context(tmp_path) == "2021-05-01 → 2026-05-02"
