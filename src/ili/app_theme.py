"""The review app's Enterprise Suite look: a navy shell bar, breadcrumb, tab navigation, white cards and
status pills over a light grey page. DRAFT – for review by qualified personnel.

Colours, radius and fonts that Streamlit themes natively live in ``app/.streamlit/config.toml``; this module
holds the rest as one CSS block plus small HTML builders. Everything is local: system fonts only, no
``url()`` or ``@import``, so the app still makes no network calls. Every piece of text that reaches the HTML
(reviewer name, folder, page, pill text) is escaped.
"""

from html import escape

import streamlit as st

from ili.align_welds import DRAFT_LABEL

# Palette (presentation only; the theme section of the app's config.toml uses the same values).
PRIMARY = "#0A6ED1"  # buttons, the active tab
LINK = "#0A5FB4"
SHELL = "#1D2D3E"  # shell bar
PAGE_BACKGROUND = "#F5F6F7"
INK = "#223548"  # body text
MUTED = "#556B82"  # labels, breadcrumb, inactive tabs
LINE = "#D9DEE3"
DANGER = "#AA0808"  # Reject, low confidence
DRAFT_BACKGROUND = "#E9730C"  # the DRAFT pill: black text on it is 6.6:1

PILL_TONES = ("danger", "warning", "success", "info")
CONFIDENCE_TONES = {"low": "danger", "medium": "warning", "high": "success"}

ENTERPRISE_CSS = f"""
[data-testid="stDecoration"], [data-testid="stHeader"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1320px; padding-top: 0; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: 0; }}

.ili-shell {{ background: {SHELL}; color: #FFFFFF; padding: 0 20px; min-height: 48px;
  display: flex; flex-wrap: wrap; align-items: center; gap: 16px; border-radius: 0 0 6px 6px; }}
.ili-shell-product {{ font-weight: 700; font-size: 15px; }}
.ili-shell-app {{ color: #B8C4D0; }}
.ili-spacer {{ flex: 1 1 auto; }}
.ili-draft {{ background: {DRAFT_BACKGROUND}; color: #000000; font-size: 12px; font-weight: 700;
  padding: 4px 10px; border-radius: 4px; }}
.ili-avatar {{ width: 32px; height: 32px; border-radius: 50%; background: {PRIMARY}; color: #FFFFFF;
  display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; }}
.ili-breadcrumb {{ color: {MUTED}; font-size: 13px; padding: 12px 0 4px; }}
.ili-breadcrumb strong {{ color: {INK}; font-weight: 600; }}

.ili-pill {{ display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 10px; border-radius: 12px;
  margin-right: 6px; border: 1px solid transparent; }}
.ili-pill-danger {{ color: {DANGER}; background: #FFEAF4; border-color: #F4B8C6; }}
.ili-pill-warning {{ color: #6A4A00; background: #FFF3B8; border-color: #E8D27A; }}
.ili-pill-success {{ color: #256F3A; background: #E8F5E9; border-color: #A9D5B4; }}
.ili-pill-info {{ color: {LINK}; background: #E5F0FA; border-color: #A7C8EA; }}

.ili-object {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 8px; padding: 14px 18px;
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin-bottom: 12px; }}
.ili-object-title {{ font-size: 20px; font-weight: 700; color: {INK}; margin-right: 4px; }}
.ili-object-facts {{ margin-left: auto; display: flex; flex-wrap: wrap; gap: 4px 24px; font-size: 13px; }}
.ili-object-facts span {{ display: flex; flex-direction: column; color: {MUTED}; }}
.ili-object-facts strong {{ color: {INK}; font-size: 15px; }}

/* Page navigation: the "page" radio drawn as tabs under the header */
.st-key-page [role="radiogroup"] {{ gap: 24px; border-bottom: 1px solid {LINE}; flex-wrap: wrap; }}
.st-key-page [data-testid="stRadioOption"] {{ padding: 10px 0 8px; margin: 0; color: {MUTED};
  border-bottom: 3px solid transparent; }}
.st-key-page [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
.st-key-page [data-testid="stRadioOption"] p {{ color: inherit; font-weight: 600; }}
.st-key-page [data-testid="stRadioOption"][data-selected="true"] {{ color: {LINK};
  border-bottom-color: {PRIMARY}; }}

/* Cards: bordered containers keyed "<prefix>card_<name>" are the white cards of the design */
[class*="st-key-"][class*="card_"] {{ background: #FFFFFF; border-radius: 8px; }}

/* Reject is the destructive action: red outline, never filled */
.st-key-review_reject button, .st-key-fix_reject button {{ background: #FFFFFF; color: {DANGER};
  border: 1px solid {DANGER}; }}
.st-key-review_reject button:hover, .st-key-fix_reject button:hover {{ background: #FFEAF4; color: {DANGER};
  border-color: {DANGER}; }}

/* The decision buttons stay in reach at the bottom of the window, like the design's action bar */
.st-key-review_actions, .st-key-fix_actions {{ position: sticky; bottom: 0; z-index: 5; background: #FFFFFF;
  border-top: 1px solid {LINE}; padding: 10px 0; }}
"""


def apply_theme() -> None:
    """Inject the CSS once per run (a constant: nothing user-supplied reaches it)."""
    st.markdown(f"<style>{ENTERPRISE_CSS}</style>", unsafe_allow_html=True)


def _initials(reviewer: str) -> str:
    words = [w for w in reviewer.replace(".", " ").split() if w[:1].isalpha()]
    return "".join(w[0].upper() for w in words[:2])


def shell_bar_html(reviewer: str) -> str:
    """The navy bar across the top: product, app, the DRAFT label and the reviewer's initials."""
    initials = _initials(reviewer or "")
    avatar = (f'<span class="ili-avatar" role="img" aria-label="Reviewer: {escape(reviewer.strip())}">'
              f"{escape(initials)}</span>" if initials else "")
    return ('<div class="ili-shell"><span class="ili-shell-product">Integrity Management</span>'
            '<span class="ili-shell-app">ILI Analyzer</span><span class="ili-spacer"></span>'
            f'<span class="ili-draft">{escape(DRAFT_LABEL)}</span>{avatar}</div>')


def breadcrumb_html(folder, page: str) -> str:
    """Run comparisons / <comparison folder> / <page>; the folder's full path is its tooltip."""
    if folder is None:
        where = "No comparison loaded"
    else:
        where = f'<span title="{escape(str(folder))}">{escape(getattr(folder, "name", str(folder)))}</span>'
    return (f'<nav class="ili-breadcrumb" aria-label="Breadcrumb">Run comparisons / {where} / '
            f"<strong>{escape(page)}</strong></nav>")


def pill_html(text: str, tone: str) -> str:
    if tone not in PILL_TONES:
        raise ValueError(f"unknown pill tone {tone!r}; expected one of {PILL_TONES}")
    return f'<span class="ili-pill ili-pill-{tone}">{escape(str(text))}</span>'


def object_header_html(title: str, pills: list[str], facts: list[tuple[str, object]]) -> str:
    """A record header: title, status pills (already-built ``pill_html``), and label/value facts on the right."""
    cells = "".join(f"<span>{escape(label)}<strong>{escape(str(value))}</strong></span>" for label, value in facts)
    return (f'<div class="ili-object"><span class="ili-object-title">{escape(title)}</span>{"".join(pills)}'
            f'<span class="ili-object-facts">{cells}</span></div>')


def confidence_pill_html(confidence) -> str:
    """Low / medium / high confidence, or "No match" for a one-sided (missing or new) row."""
    if confidence in CONFIDENCE_TONES:
        return pill_html(f"{str(confidence).capitalize()} confidence", CONFIDENCE_TONES[confidence])
    return pill_html("No match", "info")
