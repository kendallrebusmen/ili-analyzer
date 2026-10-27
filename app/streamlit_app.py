"""ILI run-comparison review app. DRAFT output only.

    streamlit run app/streamlit_app.py --server.address localhost --browser.gatherUsageStats false

Local only: ``app/.streamlit/config.toml`` (script-level config, read whatever the launch directory) sets
localhost, no telemetry and headless, and the app refuses to run when the options in force say otherwise
(a user config, an environment variable or a flag can override the file).
"""

import streamlit as st

from ili import app_state, app_theme
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_results import explorer_page, export_page
from ili.app_pages_review import review_queue_page
from ili.app_pages_setup import run_setup_page, weld_alignment_page

TITLE = "ILI Analyzer — DRAFT"
LOCAL_ADDRESSES = ("localhost", "127.0.0.1")
PAGES = {
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
st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="collapsed",
                   menu_items={"Get help": None, "Report a bug": None, "About": f"{TITLE}. {DRAFT_LABEL}"})
app_theme.apply_theme()
st.markdown(app_theme.shell_bar_html(st.session_state.get(app_state.REVIEWER_KEY, "")), unsafe_allow_html=True)
if reason := not_local_reason():
    st.error(f"The review app runs on this computer only, with no usage statistics, and refuses to start: "
             f"{reason}. Start it with: streamlit run app/streamlit_app.py --server.address localhost "
             f"--browser.gatherUsageStats false")
    st.caption(DRAFT_LABEL)
    st.stop()
folder = app_state.comparison_from_session()
crumb, reviewer = st.columns([3, 1], vertical_alignment="center")
crumb.markdown(app_theme.breadcrumb_html(folder, st.session_state.get("page", next(iter(PAGES)))),
               unsafe_allow_html=True)
reviewer.text_input("Reviewer name (required before any decision)", key=app_state.REVIEWER_KEY,
                    label_visibility="collapsed", placeholder="Reviewer name (required to decide)")
PAGES[st.radio("Page", list(PAGES), key="page", horizontal=True, label_visibility="collapsed")]()
