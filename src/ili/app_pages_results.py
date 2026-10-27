"""Growth and remaining life explorer, and dig list and report export pages. DRAFT output only."""

import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_setup import NO_DATA


def _table_page(title: str, attribute: str) -> None:
    st.title(title)
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.load_comparison(path) if path else None
    table = getattr(data, attribute) if data is not None else None
    if table is None or table.empty:
        st.info(NO_DATA)
        return
    st.dataframe(table, hide_index=True)


def explorer_page() -> None:
    _table_page("Growth and remaining life explorer", "remaining_life")


def export_page() -> None:
    _table_page("Dig list and report export", "classification")
