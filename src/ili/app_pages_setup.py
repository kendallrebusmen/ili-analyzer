"""Run setup and Weld alignment pages of the review app. DRAFT output only."""

from dataclasses import fields
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_state import comparison_of
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT
from ili.pipeline import _PATH_FIELDS, _REQUIRED, STAGES, PipelineInputs, StageResult, run_pipeline
from ili.report_charts import INK, MUTED, TEMPLATE

DATA_DIR = PROJECT_ROOT / "data"
NO_DATA = "No data yet: run the pipeline on the Run setup page."
ALL_OK = "Every stage finished with exit 0."
SHADE = "rgba(255, 158, 74, 0.25)"
MATCH_COLOURS = {"1:1": "#1F77B4", "1:2": "#E8833A", "2:1": "#2CA02C", "inserted": "#D62728",
                 "removed": "#7F7F7F"}


def field_key(name: str) -> str:
    return f"setup_{name}"


def data_files(root: Path = DATA_DIR) -> list[str]:
    """Files under ``data/`` a field can be picked from."""
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep")


def build_inputs(values: dict[str, str]) -> PipelineInputs:
    """``PipelineInputs`` from the typed values; ValueError names every missing required field."""
    missing = [n for n in _REQUIRED if not values.get(n, "").strip()]
    if missing:
        raise ValueError("required: " + ", ".join(missing))
    kwargs = {}
    for f in fields(PipelineInputs):
        text = values.get(f.name, "").strip()
        if text:
            kwargs[f.name] = Path(text) if f.name in _PATH_FIELDS else text
    return PipelineInputs(**kwargs)


def show_results(results: list[StageResult]) -> None:
    """Each stage's output in an expander (open when it failed); the failing stage's last error line in
    ``st.error`` with its full stderr below it, or a success line when every stage finished."""
    for r in results:
        with st.expander(f"{r.stage}: exit {r.exit_code}", expanded=r.exit_code != 0):
            st.code(r.stdout or "(no output)")
    failed = [r for r in results if r.exit_code != 0]
    if failed:
        f = failed[0]
        lines = [line for line in f.stderr.splitlines() if line.strip()]
        st.error(f"{f.stage} failed (exit {f.exit_code})" + (f": {lines[-1]}" if lines else ""))
        if f.stderr.strip():
            st.code(f.stderr)
    elif len(results) == len(STAGES):
        st.success(ALL_OK)


def run_and_refresh(inputs: PipelineInputs, results_key: str) -> None:
    """Run the pipeline, keep its results for the next script run and rerun the script, so every page
    element (the breadcrumb's comparison folder included) shows the new outputs."""
    with st.spinner("Running the pipeline"):
        results = run_pipeline(inputs)
    st.session_state[results_key] = results
    st.rerun()


def alignment_chart(weld_map: pd.DataFrame, changed: pd.DataFrame) -> go.Figure:
    """Joint length mismatch along the current-run log distance, by match type, changed sections shaded."""
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title="Joint length mismatch along the current run",
                      xaxis_title="Current-run log distance (ft)", yaxis_title="Length mismatch (ft)",
                      font={"color": INK}, legend_title_text="Match type")
    for _, sec in changed.iterrows():
        fig.add_vrect(x0=sec["curr_start_ft"], x1=sec["curr_end_ft"], fillcolor=SHADE, line_width=0,
                      layer="below")
    shown = weld_map.dropna(subset=["curr_start_ft", "mismatch"])
    for match_type, part in shown.groupby("match_type"):
        fig.add_trace(go.Scatter(
            x=part["curr_start_ft"], y=part["mismatch"], mode="markers", name=str(match_type),
            marker={"size": 6, "color": MATCH_COLOURS.get(str(match_type), MUTED)}))
    return fig


def _setup_defaults() -> dict[str, str]:
    setup = app_state.inputs_from_session(app_state.comparison_from_session())
    out = {f.name: "" if setup is None or getattr(setup, f.name) is None else str(getattr(setup, f.name))
           for f in fields(PipelineInputs)}
    if not out["settings"]:
        out["settings"] = str(DEFAULT_SETTINGS)
    return out


def _pick(name: str) -> None:
    chosen = st.session_state.get(field_key(name) + "_pick")
    if chosen:
        st.session_state[field_key(name)] = chosen


def run_setup_page() -> None:
    st.title("Run setup")
    st.caption(DRAFT_LABEL)
    defaults = _setup_defaults()
    files = [""] + data_files()
    for f in fields(PipelineInputs):
        key = field_key(f.name)
        if key not in st.session_state:
            st.session_state[key] = defaults[f.name]
        left, right = st.columns([3, 2])
        left.text_input(f.name.replace("_", " ") + (" *" if f.name in _REQUIRED else ""), key=key)
        if f.name in _PATH_FIELDS and f.name != "out":
            right.selectbox("Pick from data/", files, key=key + "_pick", on_change=_pick, args=(f.name,))
    if st.button("Run", type="primary"):
        try:
            inputs = build_inputs({f.name: st.session_state[field_key(f.name)] for f in fields(PipelineInputs)})
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("Running the pipeline"):
            results = run_pipeline(inputs)
        st.session_state[app_state.INPUTS_KEY] = inputs
        folder = comparison_of(inputs)
        if folder is not None:
            st.session_state[app_state.COMPARISON_KEY] = str(folder)
        st.session_state["setup_results"] = results
        st.rerun()  # the breadcrumb was drawn before the run: draw it again with the new folder
    if (results := st.session_state.pop("setup_results", None)) is not None:
        show_results(results)


def weld_alignment_page() -> None:
    st.title("Weld alignment")
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.page_data(path)
    if data is None or not data.alignment_summary:
        st.info(NO_DATA)
        return
    s = data.alignment_summary
    cols = st.columns(4)
    cols[0].metric("Alignment rate", f"{s['alignment_rate']:.1%}")
    cols[1].metric("Anchor pairs", s["n_anchor_pairs"])
    cols[2].metric("Sections", s["n_sections"])
    cols[3].metric("Changed sections", s["n_changed_sections"])
    st.subheader("Changed sections")
    st.dataframe(data.changed_sections, hide_index=True)
    if not data.weld_map.empty:
        st.plotly_chart(alignment_chart(data.weld_map, data.changed_sections))
