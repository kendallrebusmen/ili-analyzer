"""The Enterprise Suite look: theme config, shell bar, breadcrumb, status pills, and the app's layout."""

import tomllib

import pytest
from streamlit.testing.v1 import AppTest

from ili import app_theme
from ili.align_welds import DRAFT_LABEL
from test_app_setup import APP, SCRIPT_CONFIG, isolated_outputs, script_config  # noqa: F401 (fixtures)

HOSTILE = "<script>alert(1)</script>"


def test_config_sets_the_enterprise_theme():
    theme = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))["theme"]
    assert theme["base"] == "light"
    assert theme["primaryColor"] == app_theme.PRIMARY == "#0A6ED1"
    assert theme["backgroundColor"] == app_theme.PAGE_BACKGROUND == "#F5F6F7"
    assert theme["secondaryBackgroundColor"] == "#FFFFFF"
    assert theme["textColor"] == app_theme.INK == "#223548"
    assert "http" not in theme["font"]  # system fonts only: the app makes no network calls


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
    css = app_theme.ENTERPRISE_CSS
    assert "http" not in css and "@import" not in css and "url(" not in css


def test_app_renders_the_shell_with_tabs_and_no_sidebar():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert 'class="ili-shell"' in markdown and DRAFT_LABEL in markdown and 'class="ili-breadcrumb"' in markdown
    assert at.radio(key="page").value == "Run setup"  # tab navigation in the page body
    assert at.text_input(key="reviewer").value == ""
    assert not at.sidebar.radio and not at.sidebar.text_input


def test_object_header_escapes_title_and_facts():
    html = app_theme.object_header_html(f"Match {HOSTILE}", [app_theme.pill_html("Pending", "warning")],
                                        [("Hint", HOSTILE), ("Score", 1.11)])
    assert 'class="ili-object"' in html and "ili-pill-warning" in html and "<strong>1.11</strong>" in html
    assert "<script>" not in html and html.count("&lt;script&gt;") == 2
