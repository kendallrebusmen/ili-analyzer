"""Run setup and Weld alignment pages of the review app. DRAFT output only.

Run setup is four steps: the two inspections; the pipeline data (pipe properties, MOP, HCA segments, next inspection
dates), each chosen as a file or typed into a table; optional data (repairs, crossings, digs); and the report
details. The output folder, settings and criteria files sit under "Advanced". Typed tables are saved as the spec 5.2
CSVs under ``<output folder>/entered/`` before the run (``app_setup_tables``)."""

from dataclasses import fields
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ili import app_setup_tables as tables
from ili import app_state, app_theme
from ili.align_welds import DRAFT_LABEL
from ili.app_state import comparison_of
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT
from ili.pipeline import _PATH_FIELDS, _REQUIRED, STAGES, PipelineInputs, StageResult, run_pipeline
from ili.reference import REPAIR_TYPES
from ili.report_charts import INK, MUTED, TEMPLATE

DATA_DIR = PROJECT_ROOT / "data"
VENDOR_MAP_DIR = PROJECT_ROOT / "config" / "vendor_maps"
USE_FILE, TYPE_IN = "Choose a file", "Type it in"
PICK_PROMPT = "pick a file…"
REFERENCE_REQUIRED = ("pipe", "mop", "segments", "next_inspection")  # step 2: file or typed
REFERENCE_OPTIONAL = ("repairs", "crossings")  # step 3: file or typed
REPORT_FIELDS = ("line_name", "discovery_date", "prepared_for", "prepared_by", "prev_tool", "curr_tool")  # step 4
ADVANCED_FIELDS = ("out", "settings", "criteria")
LABELS = {  # every PipelineInputs field, as the page names it
    "prev_tally": "Tally", "prev_map": "Vendor map", "curr_tally": "Tally", "curr_map": "Vendor map",
    "prev_run_date": "Run date", "curr_run_date": "Run date", "pipe": "Pipe properties", "mop": "MOP profile",
    "segments": "HCA segments", "next_inspection": "Next inspection dates", "repairs": "Repairs between runs",
    "crossings": "Foreign crossings", "digs": "Dig results", "line_name": "Line name",
    "discovery_date": "Discovery date", "prepared_for": "Prepared for (client)", "prepared_by": "Prepared by",
    "prev_tool": "Previous run tool type", "curr_tool": "Current run tool type", "out": "Output folder",
    "settings": "Settings file", "criteria": "Repair criteria file",
}
NEEDED_NAMES = {"prev_tally": "previous run tally", "prev_map": "previous run vendor map",
                "curr_tally": "current run tally", "curr_map": "current run vendor map"}  # where "Tally" is ambiguous
PLACEHOLDERS = {
    "line_name": "e.g. Line 930, 16 in", "discovery_date": "YYYY-MM-DD; blank uses the latest permissible date",
    "prepared_for": "client named on the PDF cover", "prepared_by": "author named on the PDF cover",
    "prev_tool": "e.g. MFL axial", "curr_tool": "e.g. MFL axial + caliper",
    "criteria": "blank uses classification.criteria_file of the settings",
}
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
        raise ValueError("Still needed: " + ", ".join(needed_name(n) for n in missing))
    kwargs = {}
    for f in fields(PipelineInputs):
        text = values.get(f.name, "").strip()
        if text:
            kwargs[f.name] = Path(text) if f.name in _PATH_FIELDS else text
    return PipelineInputs(**kwargs)


def needed_name(name: str) -> str:
    """A required field's name in a "still needed" list (the run is named where the label alone is ambiguous)."""
    return NEEDED_NAMES.get(name, LABELS[name])


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
    if not out["out"]:
        out["out"] = str(app_state.OUTPUTS_DIR)
    return out


def _pick(name: str) -> None:
    chosen = st.session_state.get(field_key(name) + "_pick")
    if chosen:
        st.session_state[field_key(name)] = chosen


def vendor_maps(root: Path = VENDOR_MAP_DIR) -> list[str]:
    """The vendor maps a run can use (``config/vendor_maps/*.yaml``)."""
    return sorted(str(p) for p in root.glob("*.yaml")) if root.is_dir() else []


def _label(name: str) -> str:
    return LABELS[name] + (" *" if name in _REQUIRED else "")


def _heading(name: str) -> str:
    """A field's bold heading as markdown; the required mark is escaped so it is not read as emphasis."""
    return f"**{LABELS[name]}**" + (r" \*" if name in _REQUIRED else "")


def _shown_path(path: str) -> str:
    """A picker option as shown: relative to the project folder when inside it, a prompt for the empty option."""
    if not path:
        return PICK_PROMPT
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return path


def _file_field(name: str, choices: list[str], label: bool = True) -> None:
    """A path field: type or paste a path, or pick one of ``choices``."""
    left, right = st.columns([3, 2])
    left.text_input(_label(name), key=field_key(name), label_visibility="visible" if label else "collapsed",
                    placeholder="path to the file")
    right.selectbox("or pick a file", [""] + choices, key=field_key(name) + "_pick", on_change=_pick, args=(name,),
                    format_func=_shown_path, label_visibility="visible" if label else "collapsed")


def _column_config(spec) -> dict:
    """Editor columns by kind; a required column's header is marked *."""
    config = {}
    for column, kind in spec.columns.items():
        label = tables.COLUMN_LABELS[column] + (" *" if column in spec.required else "")
        if kind == tables.YEAR:
            config[column] = st.column_config.NumberColumn(label, step=1, format="%d")
        elif kind == tables.NUMBER:
            config[column] = st.column_config.NumberColumn(label)
        elif kind == tables.FLAG:
            config[column] = st.column_config.CheckboxColumn(label, default=False)
        elif kind == tables.DATE:
            config[column] = st.column_config.DateColumn(label, format="YYYY-MM-DD")
        elif kind == tables.REPAIR_TYPE:
            config[column] = st.column_config.SelectboxColumn(label, options=list(REPAIR_TYPES))
        else:
            config[column] = st.column_config.TextColumn(label)
    return config


def _reference_field(name: str, files: list[str]):
    """A reference table: chosen as a file, or typed into an editor. Returns the typed rows (a DataFrame), or None
    when a file is used."""
    spec = tables.TABLES[name]
    mode_key = field_key(name) + "_mode"
    if mode_key not in st.session_state:  # a field with a file already set opens on it; an empty one on typing
        st.session_state[mode_key] = USE_FILE if st.session_state[field_key(name)].strip() else TYPE_IN
    st.markdown(_heading(name))
    if st.radio(spec.label, (USE_FILE, TYPE_IN), key=mode_key, horizontal=True,
                label_visibility="collapsed") == USE_FILE:
        _file_field(name, files, label=False)
        return None
    seed_key = field_key(name) + "_seed"
    if seed_key not in st.session_state:  # starts from the chosen file's rows, so typing edits it
        st.session_state[seed_key] = tables.seed_table(name, st.session_state[field_key(name)])
    edited = st.data_editor(st.session_state[seed_key], key=field_key(name) + "_table", num_rows="dynamic",
                            hide_index=True, column_config=_column_config(spec), width="stretch")
    st.caption(f"{spec.help} Columns marked * are needed. Saved as {tables.ENTERED_DIR}/{spec.file_name} in the "
               "output folder when you run.")
    return edited


def _save_typed(typed: dict, values: dict[str, str]) -> dict[str, str]:
    """``values`` with each typed table saved under the output folder and its path filled in. An optional table
    with no rows is left out (as if no file were given); a required one with no rows is a ValueError."""
    out_dir = Path(values.get("out", "").strip() or app_state.OUTPUTS_DIR)
    values = dict(values)
    for name, df in typed.items():
        if df is None:
            continue
        if name not in _REQUIRED and tables.clean_table(name, df).empty:
            values[name] = ""
            continue
        values[name] = str(tables.write_table(name, df, out_dir))
    return values


def _still_needed(typed: dict) -> list[str]:
    """Required fields with neither a path nor a typed table."""
    return [needed_name(n) for n in _REQUIRED
            if typed.get(n) is None and not st.session_state[field_key(n)].strip()]


def run_setup_page() -> None:
    app_theme.page_header("Run setup", None)
    st.markdown("Four steps. Items marked * are needed for a run; everything else is optional.")
    defaults = _setup_defaults()
    for f in fields(PipelineInputs):
        if field_key(f.name) not in st.session_state:
            st.session_state[field_key(f.name)] = defaults[f.name]
    files, maps = data_files(), vendor_maps()
    typed = {}

    with st.container(border=True):
        st.subheader("1  Inspections")
        st.caption("The two in-line inspection tallies to compare, and the vendor map that reads each one.")
        left, right = st.columns(2)
        for column, run in ((left, "prev"), (right, "curr")):
            with column:
                st.markdown(f"**{'Previous' if run == 'prev' else 'Current'} run**")
                _file_field(f"{run}_tally", files)
                _file_field(f"{run}_map", maps)
                st.text_input(_label(f"{run}_run_date"), key=field_key(f"{run}_run_date"),
                              placeholder="YYYY-MM-DD; leave blank when the tally carries it")

    with st.container(border=True):
        st.subheader("2  Pipeline data")
        st.caption("Choose a file, or type the values in. Distances are current-run log distance in feet.")
        for name in REFERENCE_REQUIRED:
            typed[name] = _reference_field(name, files)

    with st.expander("3  Optional data: repairs, foreign crossings and dig results"):
        for name in REFERENCE_OPTIONAL:
            typed[name] = _reference_field(name, files)
        st.markdown(_heading("digs"))
        st.caption("Field measurements from digs, for the back-test (digs.csv).")
        _file_field("digs", files, label=False)

    with st.expander("4  Report details: names and dates on the report and the PDF cover"):
        left, right = st.columns(2)
        for i, name in enumerate(REPORT_FIELDS):
            (left if i % 2 == 0 else right).text_input(_label(name), key=field_key(name),
                                                       placeholder=PLACEHOLDERS.get(name, ""))

    with st.expander("Advanced: output folder, settings and repair criteria"):
        for name in ADVANCED_FIELDS:
            st.text_input(_label(name), key=field_key(name), placeholder=PLACEHOLDERS.get(name, ""))

    needed = _still_needed(typed)
    if needed:
        st.caption("Still needed: " + ", ".join(needed) + ".")
    if st.button("Run", type="primary"):
        try:
            values = _save_typed(typed, {f.name: st.session_state[field_key(f.name)] for f in fields(PipelineInputs)})
            inputs = build_inputs(values)
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
    path = app_state.comparison_from_session()
    app_theme.page_header("Weld alignment", app_state.comparison_context(path))
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
