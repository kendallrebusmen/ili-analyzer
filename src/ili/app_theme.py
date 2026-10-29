"""The review app's Nemi look: clean engineering (light by default, deep-teal accent, light grey sidebar),
sidebar navigation, page header with the DRAFT badge, cards and status pills, in a light and a dark theme.
DRAFT – for review by qualified personnel.

Colours, radius and fonts that Streamlit themes natively live in ``app/.streamlit/config.toml`` (one palette
per theme, sidebar background included; the app follows the computer's setting until the reviewer picks one in
the menu); this module holds the rest as one CSS block plus small HTML builders. Everything is local: system
fonts only, no ``url()`` or ``@import``, so the app still makes no network calls. Every piece of text that
reaches the HTML (reviewer name, folder, page, pill text) is escaped.
"""

from html import escape

import streamlit as st

from ili.align_welds import DRAFT_LABEL
from ili.brand import PRODUCT_NAME, PRODUCT_TAGLINE, wordmark_svg

# Palette (presentation only; the [theme.light] and [theme.dark] sections of the app's config.toml use the
# same values). Light first; the DARK_ twin follows each colour. Every text/background pair is >= 4.5:1.
PRIMARY = "#0F766E"  # the deep-teal accent: buttons; light-theme selected menu row and wordmark arc
LINK = "#0F766E"
PAGE_BACKGROUND = "#F6F7F8"
SURFACE = "#FFFFFF"  # cards, the action bar
SIDEBAR = "#EEF2F3"
INK = "#1F2A33"  # body text
MUTED = "#56636E"  # labels, captions
LINE = "#D8DEE2"
DANGER = "#B42318"  # real repair conditions only
DANGER_SOFT = "#FFF6F5"
WARNING_INK = "#93370D"  # attention / DRAFT: text ...
WARNING_BG = "#FEF0C7"  # ... on this background
SUCCESS = "#067647"

DARK_PRIMARY = "#0F766E"  # button fill (white text, AA); dark accent TEXT is DARK_LINK
DARK_LINK = "#5EEAD4"
DARK_PAGE_BACKGROUND = "#111518"
DARK_SURFACE = "#1A2025"
DARK_SIDEBAR = "#151A1E"
DARK_INK = "#E6EAED"
DARK_MUTED = "#A3AEB7"
DARK_LINE = "#33404A"
DARK_DANGER = "#FF8A80"
DARK_DANGER_SOFT = "#3A1F1D"
DARK_WARNING_INK = "#FEC84B"
DARK_WARNING_BG = "#3D2E12"
DARK_SUCCESS = "#47CD89"

CHART_GRID = "rgba(138,150,160,0.35)"  # a mid grey that reads on the light and the dark page
CHART_COLORWAY = [PRIMARY, "#6B7C93", "#B54708", "#8A96A0", "#14B8A6"]

PILL_TONES = ("danger", "warning", "success", "info")
CONFIDENCE_TONES = {"low": "danger", "medium": "warning", "high": "success"}

# Streamlit sets color-scheme on the app root to match the active theme, so each light-dark() pair below
# resolves to the theme on screen, including after the reviewer switches theme in the menu.
APP_CSS = f"""
[data-testid="stApp"] {{
  --nemi-primary: light-dark({PRIMARY}, {DARK_PRIMARY}); --nemi-accent: light-dark({PRIMARY}, {DARK_LINK});
  --nemi-link: light-dark({LINK}, {DARK_LINK});
  --nemi-surface: light-dark({SURFACE}, {DARK_SURFACE}); --nemi-sidebar: light-dark({SIDEBAR}, {DARK_SIDEBAR});
  --nemi-ink: light-dark({INK}, {DARK_INK}); --nemi-muted: light-dark({MUTED}, {DARK_MUTED});
  --nemi-line: light-dark({LINE}, {DARK_LINE});
  --nemi-danger: light-dark({DANGER}, {DARK_DANGER});
  --nemi-danger-soft: light-dark({DANGER_SOFT}, {DARK_DANGER_SOFT});
  --nemi-warning-ink: light-dark({WARNING_INK}, {DARK_WARNING_INK});
  --nemi-warning-bg: light-dark({WARNING_BG}, {DARK_WARNING_BG});
  --nemi-success: light-dark({SUCCESS}, {DARK_SUCCESS});
}}
[data-testid="stDecoration"] {{ display: none; }}
/* Streamlit's header keeps only its menu (the System / Light / Dark choice) in the top-right corner */
[data-testid="stHeader"] {{ background: transparent; height: 48px; pointer-events: none; }}
[data-testid="stHeader"] [data-testid="stToolbar"] {{ pointer-events: auto; }}
[data-testid="stMainMenu"] button {{ color: var(--nemi-muted); opacity: 1; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1320px; padding-top: 1.5rem; }}
h1, h2, h3 {{ color: var(--nemi-ink); letter-spacing: 0; }}

/* The sidebar's background comes from config.toml ([theme.light.sidebar] / [theme.dark.sidebar]) */
.nemi-wordmark {{ display: grid; grid-template-columns: auto 1fr; column-gap: 10px; align-items: center;
  padding: 4px 4px 14px; color: var(--nemi-ink); }}
.nemi-wordmark svg {{ grid-row: 1 / 3; }}
.nemi-wordmark span:first-of-type {{ font-size: 21px; font-weight: 700; line-height: 1.1; }}
.nemi-wordmark span:last-of-type {{ font-size: 11.5px; color: var(--nemi-muted); }}
.nemi-caption {{ font-size: 11.5px; color: var(--nemi-muted); text-transform: uppercase; letter-spacing: 0.6px;
  margin-top: 14px; padding: 0 4px; }}
.nemi-folder {{ font-size: 13.5px; font-weight: 600; color: var(--nemi-ink); padding: 0 4px 8px; }}
/* Bottom block of the sidebar (comparison name, reviewer): a hairline and some space above it */
.nemi-sidebar-foot {{ border-top: 1px solid var(--nemi-line); margin-top: 20px; padding-top: 4px; }}

/* Page header: title block left, the DRAFT badge right */
.nemi-header {{ display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between;
  gap: 8px 16px; padding-bottom: 14px; margin-bottom: 20px; border-bottom: 1px solid var(--nemi-line); }}
.nemi-header h1 {{ margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; line-height: 1.25;
  color: var(--nemi-ink); }}
.nemi-subtitle {{ color: var(--nemi-muted); font-size: 14px; margin-top: 2px; }}
.nemi-badge {{ display: inline-block; background: var(--nemi-warning-bg); color: var(--nemi-warning-ink);
  font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 999px; white-space: nowrap; }}

/* KPI tiles; the alert variant is for a repair count above 0 */
.nemi-kpi {{ background: var(--nemi-surface); border: 1px solid var(--nemi-line); border-radius: 8px;
  padding: 14px 18px; color: var(--nemi-muted); font-size: 13px; }}
.nemi-kpi strong {{ display: block; color: var(--nemi-ink); font-size: 30px; font-weight: 700; line-height: 1.2;
  margin: 2px 0; }}
.nemi-kpi-label, .nemi-kpi-note {{ display: block; }}
.nemi-kpi-note {{ font-size: 12.5px; }}
.nemi-kpi--alert {{ background: var(--nemi-danger-soft); border-color: var(--nemi-danger);
  color: var(--nemi-danger); }}
.nemi-kpi--alert strong {{ color: var(--nemi-danger); }}

/* Home section and card titles */
.nemi-section-title {{ font-size: 13px; font-weight: 600; color: var(--nemi-muted); text-transform: uppercase;
  letter-spacing: 0.6px; margin: 4px 0 8px; }}
.nemi-card-title {{ font-size: 16px; font-weight: 600; color: var(--nemi-ink); margin: 0 0 4px; }}
.nemi-muted {{ color: var(--nemi-muted); font-size: 13px; margin-bottom: 8px; }}
.nemi-dl {{ display: flex; flex-direction: column; justify-content: center; gap: 2px; font-size: 14.5px;
  color: var(--nemi-ink); }}
.nemi-dl span {{ color: var(--nemi-muted); font-size: 13px; }}

/* Needs-attention list: dot, bold title, muted detail */
.nemi-attn {{ display: flex; align-items: center; gap: 12px; padding: 6px 0; color: var(--nemi-ink);
  font-size: 14.5px; }}
.nemi-attn-detail {{ display: block; color: var(--nemi-muted); font-size: 13px; font-weight: 400; }}
[class*="st-key-home_attn_row_"] {{ border-top: 1px solid var(--nemi-line); padding-top: 6px; }}
/* Breathing room between the repair-condition tiles and the two Home cards below them */
.st-key-home_card_attention, .st-key-home_card_facts {{ margin-top: 24px; }}
/* Consecutive keyed cards on the other pages: the same vertical rhythm */
[class*="st-key-"][class*="card_"] + [class*="st-key-"][class*="card_"] {{ margin-top: 20px; }}
[data-testid="stMain"] [data-testid="stExpander"] {{ margin-top: 20px; }}
.nemi-attn-dot {{ flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--nemi-muted); }}
.nemi-attn-dot--warn {{ background: var(--nemi-warning-ink); }}
.nemi-attn-dot--info {{ background: var(--nemi-accent); }}
.nemi-attn-dot--ok {{ background: var(--nemi-success); }}

.nemi-facts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; font-size: 13px;
  color: var(--nemi-muted); }}
.nemi-facts strong {{ display: block; color: var(--nemi-ink); font-size: 15px; font-weight: 600; }}
.nemi-step {{ display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px;
  border-radius: 50%; background: var(--nemi-accent); color: var(--nemi-surface); font-size: 13px;
  font-weight: 700; margin-right: 8px; }}
.nemi-step-head {{ display: flex; align-items: center; margin-bottom: 4px; font-size: 16px; }}
.nemi-step-head .nemi-caption {{ margin-left: 8px; }}
.nemi-truncate {{ overflow-wrap: anywhere; }}

.ili-pill {{ display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 10px; border-radius: 12px;
  margin-right: 6px; border: 1px solid transparent; }}
.ili-pill-danger {{ color: var(--nemi-danger); background: var(--nemi-danger-soft);
  border-color: var(--nemi-danger); }}
.ili-pill-warning {{ color: var(--nemi-warning-ink); background: var(--nemi-warning-bg);
  border-color: var(--nemi-warning-ink); }}
.ili-pill-success {{ color: var(--nemi-success); background: var(--nemi-surface);
  border-color: var(--nemi-success); }}
.ili-pill-info {{ color: var(--nemi-link); background: var(--nemi-surface); border-color: var(--nemi-link); }}

.ili-object {{ background: var(--nemi-surface); border: 1px solid var(--nemi-line); border-radius: 8px;
  padding: 14px 18px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin-bottom: 12px; }}
.ili-object-title {{ font-size: 20px; font-weight: 700; color: var(--nemi-ink); margin-right: 4px; }}
.ili-object-facts {{ margin-left: auto; display: flex; flex-wrap: wrap; gap: 4px 24px; font-size: 13px; }}
.ili-object-facts span {{ display: flex; flex-direction: column; color: var(--nemi-muted); }}
.ili-object-facts strong {{ color: var(--nemi-ink); font-size: 15px; }}

/* Sidebar page menu: the "page" radio drawn as menu rows. Presentation only: it relies on Streamlit's stable
   stSidebar / stRadio test ids and degrades to a plain radio if Streamlit renames them. */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 4px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ width: 100%; padding: 9px 12px; margin: 0;
  border-radius: 6px; border: 1px solid transparent; color: var(--nemi-ink); cursor: pointer; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{ color: inherit; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: var(--nemi-surface); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background: var(--nemi-surface);
  border-color: var(--nemi-line); color: var(--nemi-accent); font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{ font-weight: 600; }}

.nemi-pending {{ background: var(--nemi-warning-bg); color: var(--nemi-warning-ink); font-size: 12.5px;
  font-weight: 600; padding: 6px 12px; border-radius: 8px; margin: 6px 0 10px; }}

/* Windows high-contrast mode: keep the selected menu row and the status dots visible */
@media (forced-colors: active) {{
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ outline: 2px solid Highlight; }}
  .nemi-attn-dot {{ forced-color-adjust: none; border: 2px solid CanvasText; }}
}}

/* Cards: bordered containers keyed "<prefix>card_<name>" are the white cards of the design */
[class*="st-key-"][class*="card_"] {{ background: var(--nemi-surface); border: 1px solid var(--nemi-line);
  border-radius: 8px; }}

/* Reject is the destructive action: red outline, never filled */
.st-key-review_reject button, .st-key-fix_reject button {{ background: var(--nemi-surface);
  color: var(--nemi-danger); border: 1px solid var(--nemi-danger); }}
.st-key-review_reject button:hover, .st-key-fix_reject button:hover {{ background: var(--nemi-danger-soft);
  color: var(--nemi-danger); border-color: var(--nemi-danger); }}

/* The decision buttons stay in reach at the bottom of the window */
.st-key-review_actions, .st-key-fix_actions {{ position: sticky; bottom: 0; z-index: 5;
  background: var(--nemi-surface); border-top: 1px solid var(--nemi-line); padding: 10px 0; }}
"""


CHART_HEIGHT = 360


def style_chart(fig):
    """Restyle a Plotly figure in place to the Nemi look (transparent, theme-following, one height and compact
    margins so side-by-side charts line up) and return it. The title lives in the card title, so the figure's
    own is cleared."""
    fig.update_layout(template=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=CHART_COLORWAY, height=CHART_HEIGHT, title_text=None,
                      margin={"l": 8, "r": 8, "t": 8, "b": 8})
    fig.layout.font.color = None
    fig.update_xaxes(gridcolor=CHART_GRID, automargin=True)
    fig.update_yaxes(gridcolor=CHART_GRID, automargin=True)
    return fig


def apply_theme() -> None:
    """Inject the CSS once per run (a constant: nothing user-supplied reaches it)."""
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def sidebar_brand_html() -> str:
    """The wordmark and the product tagline at the top of the sidebar."""
    return (f'<div class="nemi-wordmark">{wordmark_svg()}<span>{escape(PRODUCT_NAME)}</span>'
            f"<span>{escape(PRODUCT_TAGLINE)}</span></div>")


def sidebar_foot_html() -> str:
    """Opens the sidebar's bottom block (a hairline with space above); the comparison name follows it."""
    return '<div class="nemi-sidebar-foot"></div>'


def section_title(text: str) -> str:
    """The small uppercase muted section heading, as HTML (escaped)."""
    return f'<div class="nemi-section-title">{escape(text)}</div>'


def card_title(text: str) -> str:
    """A card's 16px/600 title, as HTML (escaped)."""
    return f'<div class="nemi-card-title">{escape(text)}</div>'


def muted_html(text: str) -> str:
    """A muted 13px detail line under a card title, as HTML (escaped)."""
    return f'<div class="nemi-muted">{escape(text)}</div>'


def sidebar_folder_html(folder) -> str:
    """The "Comparison" caption and the comparison folder's name (full path as the tooltip)."""
    if folder is None:
        name = "No comparison loaded"
        tip = name
    else:
        name, tip = getattr(folder, "name", str(folder)), str(folder)
    return (f'<div class="nemi-caption">Comparison</div>'
            f'<div class="nemi-folder nemi-truncate" title="{escape(tip)}">{escape(name)}</div>')


def page_header_html(title: str, subtitle: str | None) -> str:
    """Title (and subtitle) left, the DRAFT badge right; every piece of text is escaped."""
    sub = (f'<span class="nemi-subtitle nemi-truncate" title="{escape(subtitle)}">{escape(subtitle)}</span>'
           if subtitle else "")
    return (f'<header class="nemi-header"><div><h1>{escape(title)}</h1>{sub}</div>'
            f'<span class="nemi-badge">{escape(DRAFT_LABEL)}</span></header>')


def page_header(title: str, subtitle: str | None) -> None:
    st.markdown(page_header_html(title, subtitle), unsafe_allow_html=True)


_NAV_LABELS = {"Match review queue": "Match review", "Growth and remaining life explorer": "Growth & life", "Dig list and report export": "Dig list & export"}


def nav_label_text(page: str) -> str:
    """The sidebar menu label of a page. Static on purpose: Streamlit round-trips the formatted label, so a label
    that changed between runs (a live count) would no longer match the browser's held value and reset the page."""
    return _NAV_LABELS.get(page, page)


_NAV_ICONS = {"Home": "home", "Run setup": "play_circle", "Weld alignment": "straighten",
              "Match review queue": "compare_arrows", "Growth and remaining life explorer": "trending_up",
              "Dig list and report export": "assignment"}


def nav_menu_label(page: str) -> str:
    """The radio's row label: a Material icon (bundled with Streamlit, local) then the plain text. Static, like
    ``nav_label_text``."""
    icon = _NAV_ICONS.get(page)
    return f":material/{icon}: {nav_label_text(page)}" if icon else nav_label_text(page)


def sidebar_pending_html(pending: int) -> str:
    """The amber "N matches waiting for review" line under the menu; empty when nothing is pending."""
    if pending <= 0:
        return ""
    noun = "match" if pending == 1 else "matches"
    return f'<div class="nemi-pending">{int(pending)} {noun} waiting for review</div>'


def step_heading(n: int, title: str, optional: bool) -> str:
    """A numbered step heading (round badge, title, "optional" tag) as HTML; the text is escaped."""
    tag = ' <span class="nemi-caption">optional</span>' if optional else ""
    return f'<div class="nemi-step-head"><span class="nemi-step">{int(n)}</span><strong>{escape(title)}</strong>{tag}</div>'


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
