"""Nemi: ILI run-comparison review app. DRAFT output only.

    streamlit run app/streamlit_app.py --server.address localhost --browser.gatherUsageStats false

Local only: ``app/.streamlit/config.toml`` (script-level config, read whatever the launch directory) sets
localhost, no telemetry and headless, and the app refuses to run when the options in force say otherwise
(a user config, an environment variable or a flag can override the file).
"""

import streamlit as st

from ili import app_state, app_theme, report_data
from ili.align_welds import DRAFT_LABEL
from ili.brand import PRODUCT_TAGLINE, product_label
from ili.app_pages_home import home_page
from ili.app_pages_results import explorer_page, export_page
from ili.app_pages_review import review_queue_page
from ili.app_pages_setup import run_setup_page, weld_alignment_page

TITLE = "Nemi"
LOCAL_ADDRESSES = ("localhost", "127.0.0.1")
PAGES = {
    "Home": home_page,
    "Run setup": run_setup_page,
    "Weld alignment": weld_alignment_page,
    "Match review queue": review_queue_page,
    "Growth and remaining life explorer": explorer_page,
    "Dig list and report export": export_page,
}


def not_local_reason() -> str | None:
    """Why the options in force are not local-only, or None when they are."""
    address = st.get_option("server.address")
    if address not in LOCAL_ADDRESSES:
        return f"server.address is {address!r}, not localhost"
    if st.get_option("browser.gatherUsageStats"):
        return "browser.gatherUsageStats is on"
    return None


# The menu keeps the System / Light / Dark theme choice; its links to Streamlit's website go, so nothing points off
# this computer.
st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded",
                   menu_items={"Get help": None, "Report a bug": None, "About": f"{product_label()} — {PRODUCT_TAGLINE}. {DRAFT_LABEL}"})
app_theme.apply_theme()
if reason := not_local_reason():
    app_theme.page_header(TITLE, None)
    st.error(f"The review app runs on this computer only, with no usage statistics, and refuses to start: "
             f"{reason}. Start it with: streamlit run app/streamlit_app.py --server.address localhost "
             f"--browser.gatherUsageStats false")
    st.caption(DRAFT_LABEL)
    st.stop()
folder = app_state.comparison_from_session()


def pending_count() -> int:
    """Matches awaiting a decision in the comparison shown; 0 without data, never raises."""
    try:
        data = None if folder is None else app_state.load_comparison(folder)
        return 0 if data is None or data.match_table.empty else len(report_data.pending_reviews(data.match_table))
    except Exception:  # the navigation must not fail because a page's data cannot be read
        return 0


def nav_label(page: str) -> str:
    return app_theme.nav_label_text(page)


with st.sidebar:
    st.markdown(app_theme.sidebar_brand_html(), unsafe_allow_html=True)
    choice = st.radio("Pages", list(PAGES), key="page", format_func=nav_label, label_visibility="collapsed")
    if pending := pending_count():
        st.markdown(app_theme.sidebar_pending_html(pending), unsafe_allow_html=True)
    st.markdown(app_theme.sidebar_folder_html(folder), unsafe_allow_html=True)
    st.text_input("Reviewer", key=app_state.REVIEWER_KEY, placeholder="Your name (required to decide)")
PAGES[choice]()
