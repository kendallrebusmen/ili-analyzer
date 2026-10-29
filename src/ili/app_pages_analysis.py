"""Analysis table page: the classified anomalies in the consultant's layout, with filters, a column-group
selector and the Excel download. Reads ``analysis_table.csv`` as the report stage wrote it. DRAFT output only.
"""

import math

import pandas as pd
import streamlit as st

from ili import app_state, app_theme
from ili.align_welds import DRAFT_LABEL
from ili.analysis_table import ANALYSIS_COLUMNS, ANALYSIS_TABLE_FILE, ANALYSIS_XLSX_FILE, GROUPS, load_analysis_table
from ili.app_pages_setup import NO_DATA

NOT_WRITTEN = "Run the pipeline (the report stage writes the analysis table)."
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DEFAULT_GROUPS = ["Identity & location", "Growth", "Strength", "Life & class"]
ALWAYS = ("Feature ID", "Odometer (ft)")
ODOMETER, CATEGORY, SURFACE, SOURCE = "Odometer (ft)", "Repair Category", "Internal / External", "Growth Rate Source"
GROUP_OF = {name: group for name, group in ANALYSIS_COLUMNS}


def number_format(name: str) -> str:
    """The display format of a numeric column, by what it measures."""
    n = name.lower()
    if n in ("latitude", "longitude"):
        return "%.6f"
    if n in ("wall thickness (in)", "metal loss depth (in)", "dent depth (in)"):
        return "%.3f"
    if "erf" in n or "fpr" in n:
        return "%.3f"
    if "(years)" in n:
        return "%.1f"
    if "(psi)" in n:
        return "%.0f"
    if "(mpy)" in n:
        return "%.2f"
    if "(%)" in n or n == "projected depth" or "install date" in n:
        return "%.0f"
    if n in ("odometer (ft)", "elevation", "orientation (deg)"):
        return "%.1f"
    return "%.2f"


def column_groups(columns) -> dict[str, str]:
    """Each shown column's group; the strength columns are renamed for original B31G, so groups are read by
    position when the table has the standard number of columns."""
    columns = list(columns)
    if len(columns) == len(ANALYSIS_COLUMNS):
        return {c: g for c, (_, g) in zip(columns, ANALYSIS_COLUMNS)}
    return {c: GROUP_OF[c] for c in columns if c in GROUP_OF}


def filter_rows(table: pd.DataFrame, categories, surfaces, methods, odometer_range) -> pd.DataFrame:
    """Rows in the chosen repair categories, surfaces and growth sources (an empty choice means all) and in
    the odometer range (rows without an odometer stay only when the range is the whole line)."""
    out = table
    for column, chosen in ((CATEGORY, categories), (SURFACE, surfaces), (SOURCE, methods)):
        if chosen:
            out = out[out[column].isin(chosen)]
    low, high = odometer_range
    odo = pd.to_numeric(table[ODOMETER], errors="coerce")
    whole = low <= odo.min() and high >= odo.max()
    keep = odo.between(low, high) | (odo.isna() & whole)
    return out[keep.reindex(out.index, fill_value=False)].reset_index(drop=True)


def _options(table: pd.DataFrame, column: str) -> list[str]:
    return sorted(table[column].dropna().astype(str).unique()) if column in table.columns else []


def analysis_page() -> None:
    path = app_state.comparison_from_session()
    app_theme.page_header("Analysis table", app_state.comparison_context(path))
    data = app_state.page_data(path)
    if data is None or data.classification.empty:
        st.info(NO_DATA)
        return
    try:
        table = load_analysis_table(path / ANALYSIS_TABLE_FILE)
    except FileNotFoundError:
        st.info(NOT_WRITTEN)
        return
    except (ValueError, OSError) as exc:
        st.error(f"{path / ANALYSIS_TABLE_FILE} cannot be read: {exc}")
        return
    groups = column_groups(table.columns)
    odo = pd.to_numeric(table[ODOMETER], errors="coerce").dropna()
    low, high = (math.floor(odo.min()), math.ceil(odo.max())) if len(odo) else (0, 0)
    with st.container(border=True, key="card_analysis_filters"):
        st.markdown(app_theme.card_title("Filters"), unsafe_allow_html=True)
        left, mid, right = st.columns(3)
        categories = left.multiselect("Repair Category", _options(table, CATEGORY), key="analysis_categories")
        surfaces = mid.multiselect("Internal / External", _options(table, SURFACE), key="analysis_surfaces")
        methods = right.multiselect("Growth Rate Source", _options(table, SOURCE), key="analysis_methods")
        if high > low:
            chosen = st.slider("Odometer (ft)", float(low), float(high), (float(low), float(high)), step=1.0,
                               key="analysis_range")
        else:
            chosen = (float(low), float(high))
        shown_groups = st.multiselect("Columns", list(GROUPS), default=DEFAULT_GROUPS, key="analysis_groups",
                                      help="Feature ID and Odometer (ft) are always shown.")
    rows = filter_rows(table, categories, surfaces, methods, chosen)
    columns = [c for c in table.columns if c in ALWAYS or groups.get(c) in shown_groups]
    shown = rows[columns]
    config = {}
    for name in columns:
        pinned = {"pinned": True} if name in ALWAYS else {}
        if pd.api.types.is_numeric_dtype(shown[name].dtype) and not pd.api.types.is_bool_dtype(shown[name].dtype):
            config[name] = st.column_config.NumberColumn(name, format=number_format(name), **pinned)
        else:
            config[name] = st.column_config.TextColumn(name, **pinned)
    with st.container(border=True, key="card_analysis_table"):
        st.markdown(app_theme.card_title("Classified anomalies"), unsafe_allow_html=True)
        st.markdown(app_theme.muted_html(f"{len(rows)} row{'' if len(rows) == 1 else 's'}"), unsafe_allow_html=True)
        st.dataframe(shown, hide_index=True, column_config=config)
    workbook = path / ANALYSIS_XLSX_FILE
    if workbook.is_file():
        with st.container(border=True, key="card_analysis_download"):
            st.download_button(f"Download {ANALYSIS_XLSX_FILE}", workbook.read_bytes(), file_name=ANALYSIS_XLSX_FILE,
                               mime=XLSX_MIME, key="analysis_download", icon=":material/download:")
            st.caption(f"All {len(table)} classified anomalies, the consultant's layout. {DRAFT_LABEL}")
