"""Growth and remaining life explorer, and dig list, flag editor and report export pages. DRAFT output only.

The flag editor appends to ``feature_flags.csv`` (``review.append_flag``); a flag takes effect when the
pipeline is rerun. Each flag shows the status the last classification run recorded (``flag_status.csv``);
a flag saved since then is marked "saved after the last run — rerun to apply" (as in the report).
"""

from html import escape
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from ili import app_state, app_theme
from ili.align_welds import DRAFT_LABEL
from ili.app_columns import CATEGORY_WORDS, column_config, display_frame
from ili.app_pages_review import NEEDS_REVIEWER, SAVED_AFTER_RUN, rerun_button, write_error
from ili.app_pages_setup import NO_DATA
from ili.classify import criteria_path
from ili.config import PROJECT_ROOT, load_settings
from ili.pipeline import PipelineInputs
from ili.report_charts import growth_vs_distance, rl_distribution
from ili.report_data import dig_list, load_report_inputs
from ili.review import FLAG_NAMES, FLAGS_FILE, append_flag, effective_flags, latest_flag_ids

FLAG_SAVED = "Saved — rerun the pipeline to apply"
EXPLORER_COLUMNS = ["match_id", "match_type", "curr_feature_id", "curr_log_dist_ft", "segment_id", "raw_mpy",
                    "governing_mpy", "governing_method", "rl_yr", "rl_governing", "governing_category",
                    "governing_rule", "due_date"]
EXPORTS = (("report.pdf", "application/pdf", "the PDF report, laid out for print and for the client"),
           ("report.html", "text/html", "the HTML report"),
           ("dig_list.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "the Excel dig list"),
           ("run_manifest.json", "application/json", "the run manifest"))
FLAG_HELP = {"manual_immediate": "Treat as immediate (manual flag)",
             "field_crack_confirmed": "Crack confirmed in the field",
             "removed_after_run": "Pipe removed from the line after the inspection"}


def explorer_rows(data, segments, categories, methods) -> pd.DataFrame:
    """Growth, remaining life and classification joined by ``match_id``, limited to the chosen segments,
    governing categories and governing growth methods; an empty selection means all."""
    g = data.growth
    out = g[[c for c in EXPLORER_COLUMNS if c in g.columns]]
    for table, cols in ((data.remaining_life, ["rl_yr", "rl_governing"]),
                        (data.classification, ["governing_category", "governing_rule", "due_date"])):
        out = out.merge(table[["match_id", *[c for c in cols if c in table.columns]]].drop_duplicates("match_id"),
                        on="match_id", how="left")
    for column, chosen in (("segment_id", segments), ("governing_category", categories),
                           ("governing_method", methods)):
        if chosen:
            out = out[out[column].isin(chosen)]
    return out.reset_index(drop=True)


def _options(table: pd.DataFrame, column: str) -> list[str]:
    return sorted(table[column].dropna().astype(str).unique()) if column in table.columns else []


def explorer_page() -> None:
    path = app_state.comparison_from_session()
    app_theme.page_header("Growth and remaining life explorer", app_state.comparison_context(path))
    data = app_state.page_data(path)
    if data is None or data.growth.empty or data.remaining_life.empty or data.classification.empty:
        st.info(NO_DATA)
        return
    everything = explorer_rows(data, [], [], [])
    with st.container(border=True, key="card_explorer_filters"):
        st.markdown(app_theme.card_title("Filters"), unsafe_allow_html=True)
        left, mid, right = st.columns(3)
        segments = left.multiselect("Segment", _options(everything, "segment_id"), key="explorer_segments")
        categories = mid.multiselect("Repair category", _options(everything, "governing_category"),
                                     key="explorer_categories", format_func=lambda c: CATEGORY_WORDS.get(c, c))
        methods = right.multiselect("Growth method", _options(everything, "governing_method"),
                                    key="explorer_methods")
    rows = explorer_rows(data, segments, categories, methods)
    shown_rows = display_frame(rows)  # readable category words; the stored values drive the filters
    with st.container(border=True, key="card_explorer_table"):
        st.markdown(app_theme.card_title("Growth and remaining life by feature"), unsafe_allow_html=True)
        st.markdown(app_theme.muted_html(f"{len(rows)} row{'' if len(rows) == 1 else 's'}"), unsafe_allow_html=True)
        st.dataframe(shown_rows, hide_index=True, column_config=column_config(shown_rows, pin=True))
    shown = SimpleNamespace(growth=data.growth[data.growth.match_id.isin(rows.match_id)],
                            remaining_life=data.remaining_life[data.remaining_life.match_id.isin(rows.match_id)])
    chart_left, chart_right = st.columns(2)
    with chart_left, st.container(border=True, key="card_explorer_growth"):
        st.markdown(app_theme.card_title("Growth rate along the line"), unsafe_allow_html=True)
        st.plotly_chart(app_theme.style_chart(growth_vs_distance(shown)), theme="streamlit")
    with chart_right, st.container(border=True, key="card_explorer_life"):
        st.markdown(app_theme.card_title("Remaining life distribution"), unsafe_allow_html=True)
        st.plotly_chart(app_theme.style_chart(rl_distribution(shown)), theme="streamlit")


def _members(classification: pd.DataFrame) -> list[str]:
    """Every current-run feature id the classification covers (a cluster's members are separate ids)."""
    ids = {m for cell in classification.curr_feature_id.dropna().astype(str) for m in cell.split(";") if m}
    return sorted(ids)


def flag_table(data) -> pd.DataFrame:
    """The flag in force per feature and flag, with the status recorded by the last run (blank log rows
    that no run has seen are marked ``SAVED_AFTER_RUN``)."""
    flags = data.flags
    if flags.empty:
        return pd.DataFrame(columns=["curr_feature_id", "flag", "value", "status", "reviewer", "decided_utc",
                                     "note"])
    winners = flags[flags.flag_id.isin(latest_flag_ids(flags))]
    recorded = {} if data.flag_status.empty else dict(zip(data.flag_status.flag_id, data.flag_status.applied_status))
    reasons = {} if data.flag_status.empty else dict(zip(data.flag_status.flag_id, data.flag_status.reason))
    status = []
    for fid in winners.flag_id:
        if fid not in recorded:
            status.append(SAVED_AFTER_RUN)
        else:
            status.append(recorded[fid] + (f": {reasons[fid]}" if reasons.get(fid) else ""))
    out = winners.assign(status=status)
    return out[["curr_feature_id", "flag", "value", "status", "reviewer", "decided_utc", "note"]].reset_index(drop=True)


def _save_flags(path: Path, feature: str, wanted: dict[str, bool], current: dict[str, bool], note: str) -> None:
    reviewer = st.session_state[app_state.REVIEWER_KEY]
    changed = [f for f in FLAG_NAMES if wanted[f] != current.get(f, False)]
    if not changed:
        st.info("No flag changed; nothing was saved.")
        return
    try:
        for name in changed:
            append_flag(path / FLAGS_FILE, feature, name, wanted[name], reviewer, note or None)
    except ValueError as exc:
        st.error(str(exc))
        return
    except OSError as exc:  # e.g. PermissionError: the CSV is open in Excel
        st.error(write_error(path / FLAGS_FILE, exc))
        return
    st.session_state["flag_saved"] = FLAG_SAVED
    st.rerun()


def _flag_editor(path: Path, data) -> None:
    if saved := st.session_state.pop("flag_saved", None):
        st.success(saved)
    reviewer = st.session_state.get(app_state.REVIEWER_KEY, "").strip()
    ids = _members(data.classification)
    with st.container(border=True, key="card_flag_edit"):
        st.markdown(app_theme.card_title("Flag a feature"), unsafe_allow_html=True)
        st.markdown(app_theme.muted_html("Flags are per current-run feature (a cluster's members are separate "
                                         "ids). A flag takes effect when the pipeline is rerun."),
                    unsafe_allow_html=True)
        feature = st.selectbox("Current-run feature", ids, key="flag_feature")
        current = {k[1]: v for k, v in effective_flags(data.flags).items() if k[0] == feature}
        with st.form("flag_form"):
            wanted = {name: st.checkbox(FLAG_HELP[name], value=current.get(name, False),
                                        key=f"flag_{name}_{feature}") for name in FLAG_NAMES}
            note = st.text_input("Note", key="flag_note")
            submitted = st.form_submit_button("Save flags", disabled=not reviewer)
        if not reviewer:
            st.caption(NEEDS_REVIEWER)
        if submitted:
            _save_flags(path, feature, wanted, current, note)
    flags = flag_table(data)
    with st.container(border=True, key="card_flags_in_force"):
        st.markdown(app_theme.card_title("Flags in force"), unsafe_allow_html=True)
        st.markdown(app_theme.muted_html("With the status the last run recorded."), unsafe_allow_html=True)
        st.dataframe(flags, hide_index=True, column_config=column_config(flags))


def _absolute(p):
    """A recorded input path, resolved against the project root when relative (not the cwd)."""
    return None if p is None else (Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p)


def _file_stamps(path: Path, setup) -> tuple:
    """Modification times of the comparison folder's files and of the setup's input files: the cache key."""
    inputs = [_absolute(getattr(setup, f)) for f in ("pipe", "mop", "segments", "crossings", "digs", "settings",
                                                     "criteria")]
    files = sorted(p for p in Path(path).iterdir() if p.is_file()) + [p for p in inputs if p is not None]
    return tuple((str(p), p.stat().st_mtime_ns if p.is_file() else None) for p in files)


def _compute_view(path: str, prev_id: str, curr_id: str, setup, stamps: tuple):
    out = _absolute(setup.out)
    try:
        settings_path = _absolute(setup.settings)
        settings = load_settings(settings_path)
        inputs = load_report_inputs(
            Path(path), out / "standardized" / f"{prev_id}.csv", out / "standardized" / f"{curr_id}.csv", out,
            _absolute(setup.pipe), _absolute(setup.mop), _absolute(setup.segments), _absolute(setup.crossings),
            _absolute(setup.digs), settings_path, criteria_path(_absolute(setup.criteria), settings),
            repairs=_absolute(setup.repairs))
        return dig_list(inputs), inputs.log_warnings, None
    except (ValueError, OSError, KeyError) as exc:
        return None, [], str(exc)


try:  # cached on the files' modification times, like app_state; the setup object is not hashed
    _compute_view = st.cache_data(show_spinner=False, hash_funcs={PipelineInputs: repr})(_compute_view)
except Exception:  # pragma: no cover
    pass


def _report_view(path: Path, data, setup):
    """The dig list and the report's log warnings; (None, [], error) when the report refuses the inputs."""
    summary = data.match_summary
    return _compute_view(str(path), summary["prev_run_id"], summary["curr_run_id"], setup,
                         _file_stamps(path, setup))


def _downloads(path: Path) -> None:
    present = [e for e in EXPORTS if (path / e[0]).is_file()]
    with st.container(border=True, key="card_downloads"):
        st.markdown(app_theme.card_title("Report files"), unsafe_allow_html=True)
        if not present:
            st.caption("No report files yet: rerun the pipeline to write them.")
        for i, (name, mime, what) in enumerate(present):
            with st.container(key=f"download_row_{i}"):
                text, action = st.columns([4, 1.6], vertical_alignment="center")
                text.markdown(f'<div class="nemi-dl">{escape(name)}<span>{escape(what[:1].upper() + what[1:])}'
                              f'</span></div>', unsafe_allow_html=True)
                text.caption(DRAFT_LABEL)
                action.download_button(f"Download {name}", (path / name).read_bytes(), file_name=name, mime=mime,
                                       key=f"download_{name}", icon=":material/download:", width="stretch")


def dig_list_page() -> None:
    """The dig list and the manual flags that feed it (with the rerun that applies them)."""
    path = app_state.comparison_from_session()
    app_theme.page_header("Dig list", app_state.comparison_context(path))
    data = app_state.page_data(path)
    if data is None or data.classification.empty:
        st.info(NO_DATA)
        return
    setup = app_state.inputs_from_session(path)
    dig_tab, flags_tab = st.tabs(["Dig list", "Manual flags"])
    with dig_tab:
        if setup is None:
            st.warning("The dig list needs the run setup, which is not available: run the pipeline on the Run "
                       "setup page.")
        else:
            digs, warnings, error = _report_view(path, data, setup)
            for text in warnings:
                st.warning(text)
            if error:
                st.error(error)
            else:
                shown_digs = display_frame(digs)
                with st.container(border=True, key="card_dig_list"):
                    st.markdown(app_theme.card_title("Dig list"), unsafe_allow_html=True)
                    st.caption(f"{len(digs)} features on the dig list as of the last run. {DRAFT_LABEL}")
                    st.dataframe(shown_digs, hide_index=True, column_config=column_config(shown_digs, pin=True))
    with flags_tab:
        _flag_editor(path, data)
        rerun_button(path, key="export_rerun")


def reports_page() -> None:
    """The report files of the comparison, each with its download button."""
    path = app_state.comparison_from_session()
    app_theme.page_header("Reports", app_state.comparison_context(path))
    data = app_state.page_data(path)
    if data is None or data.classification.empty:
        st.info(NO_DATA)
        return
    _downloads(path)
