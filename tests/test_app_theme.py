"""The Nemi look: theme config, shell bar, breadcrumb, status pills, and the app's layout."""

import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ili import app_theme
from ili.align_welds import DRAFT_LABEL
from test_app_setup import APP, SCRIPT_CONFIG, isolated_outputs, script_config  # noqa: F401 (fixtures)

HOSTILE = "<script>alert(1)</script>"


def test_palette_matches_config():
    config = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))
    theme = config["theme"]
    light, dark = theme["light"], theme["dark"]
    assert "base" not in theme  # one palette per theme; the reviewer or the computer picks which
    assert light["primaryColor"] == app_theme.PRIMARY == "#0F766E"
    assert dark["primaryColor"] == app_theme.DARK_PRIMARY == "#14B8A6"
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
    ((_T.PRIMARY, _T.DARK_PRIMARY), ((_T.PAGE_BACKGROUND, _T.DARK_PAGE_BACKGROUND), (_T.SURFACE, _T.DARK_SURFACE),
                                     (_T.SIDEBAR, _T.DARK_SIDEBAR))),
    ((_T.LINK, _T.DARK_LINK), ((_T.SURFACE, _T.DARK_SURFACE),)),
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
    assert out.layout.xaxis.gridcolor == app_theme.LINE
    assert list(out.data[0].y) == [3, 4]


def test_shell_bar_carries_the_draft_label_and_escapes_the_reviewer():
    html = app_theme.shell_bar_html(f"Ann {HOSTILE}")  # has initials, so the name reaches the avatar label
    assert DRAFT_LABEL in html and ">A</span>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("reviewer, initials", [("K. Rebusmen", "KR"), ("ada", "A"), ("  ", None), ("", None)])
def test_shell_bar_initials(reviewer, initials):
    html = app_theme.shell_bar_html(reviewer)
    if initials:
        assert f">{initials}</span>" in html
    else:
        assert "ili-avatar" not in html


@pytest.mark.xfail(reason="rewritten in Task 3")
def test_breadcrumb_escapes_folder_and_page():
    html = app_theme.breadcrumb_html(HOSTILE, "Match review queue")
    assert "Run comparisons" in html and "Match review queue" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "No comparison loaded" in app_theme.breadcrumb_html(None, "Run setup")


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


@pytest.mark.xfail(reason="rewritten in Task 3")
def test_app_renders_the_shell_with_tabs_and_no_sidebar():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert 'class="ili-shell"' in markdown and DRAFT_LABEL in markdown and 'class="ili-breadcrumb"' in markdown
    assert at.radio(key="page").value == "Run setup"  # tab navigation in the page body
    assert at.text_input(key="reviewer").value == ""
    assert not at.sidebar.radio and not at.sidebar.text_input


def test_app_menu_has_no_links_off_this_computer():
    source = Path(APP).read_text(encoding="utf-8")
    assert '"Get help": None' in source and '"Report a bug": None' in source


def test_object_header_escapes_title_and_facts():
    html = app_theme.object_header_html(f"Match {HOSTILE}", [app_theme.pill_html("Pending", "warning")],
                                        [("Hint", HOSTILE), ("Score", 1.11)])
    assert 'class="ili-object"' in html and "ili-pill-warning" in html and "<strong>1.11</strong>" in html
    assert "<script>" not in html and html.count("&lt;script&gt;") == 2
