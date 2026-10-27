"""Match review queue page of the review app. DRAFT output only."""

import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_setup import NO_DATA


def review_queue_page() -> None:
    st.title("Match review queue")
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.load_comparison(path) if path else None
    if data is None or data.match_table.empty:
        st.info(NO_DATA)
        return
    st.dataframe(data.match_table, hide_index=True)
