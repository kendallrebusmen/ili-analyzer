"""ILI run-comparison review app: ``streamlit run app/streamlit_app.py``. DRAFT output only."""

import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_results import explorer_page, export_page
from ili.app_pages_review import review_queue_page
from ili.app_pages_setup import run_setup_page, weld_alignment_page

TITLE = "ILI Analyzer — DRAFT"
PAGES = {
    "Run setup": run_setup_page,
    "Weld alignment": weld_alignment_page,
    "Match review queue": review_queue_page,
    "Growth and remaining life explorer": explorer_page,
    "Dig list and report export": export_page,
}

st.set_page_config(page_title=TITLE)
with st.sidebar:
    st.warning(DRAFT_LABEL)
    st.text_input("Reviewer name (required before any decision)", key=app_state.REVIEWER_KEY)
    folder = app_state.comparison_from_session()
    st.caption(f"Comparison: {folder}" if folder else "Comparison: none loaded")
PAGES[st.sidebar.radio("Page", list(PAGES), key="page")]()
