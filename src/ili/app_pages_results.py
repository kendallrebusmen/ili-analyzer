"""Growth and remaining life explorer, and dig list, flag editor and report export pages. DRAFT output only.

The flag editor appends to ``feature_flags.csv`` (``review.append_flag``); a flag takes effect when the
pipeline is rerun. Each flag shows the status the last classification run recorded (``flag_status.csv``);
a flag saved since then is marked "saved after the last run — rerun to apply" (as in the report).
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_review import NEEDS_REVIEWER, rerun_button
from ili.app_pages_setup import NO_DATA
from ili.classify import criteria_path
from ili.config import load_settings
from ili.report_charts import growth_vs_distance, rl_distribution
from ili.report_data import dig_list, load_report_inputs
from ili.review import FLAG_NAMES, FLAGS_FILE, append_flag, effective_flags, latest_flag_ids

SAVED_AFTER_RUN = "saved after the last run — rerun to apply"
FLAG_SAVED = "Saved — rerun the pipeline to apply"
EXPLORER_COLUMNS = ["match_id", "match_type", "curr_feature_id", "curr_log_dist_ft", "segment_id", "raw_mpy",
                    "governing_mpy", "governing_method", "rl_yr", "rl_governing", "governing_category",
                    "governing_rule", "due_date"]
EXPORTS = (("report.html", "text/html", "the HTML report"),
           ("dig_list.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "the Excel dig list"),
           ("run_manifest.json", "application/json", "the run manifest"))
FLAG_HELP = {"manual_immediate": "Treat as immediate (manual flag)",
             "field_crack_confirmed": "Crack confirmed in the field"}


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
    st.title("Growth and remaining life explorer")
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.load_comparison(path) if path else None
    if data is None or data.growth.empty or data.remaining_life.empty or data.classification.empty:
        st.info(NO_DATA)
        return
    everything = explorer_rows(data, [], [], [])
    left, mid, right = st.columns(3)
    segments = left.multiselect("Segment", _options(everything, "segment_id"), key="explorer_segments")
    categories = mid.multiselect("Governing category", _options(everything, "governing_category"),
                                 key="explorer_categories")
    methods = right.multiselect("Governing growth method", _options(everything, "governing_method"),
                                key="explorer_methods")
    rows = explorer_rows(data, segments, categories, methods)
    st.metric("Rows shown", len(rows))
    st.dataframe(rows, hide_index=True)
    shown = SimpleNamespace(growth=data.growth[data.growth.match_id.isin(rows.match_id)],
                            remaining_life=data.remaining_life[data.remaining_life.match_id.isin(rows.match_id)])
    st.plotly_chart(growth_vs_distance(shown))
    st.plotly_chart(rl_distribution(shown))


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
    st.session_state["flag_saved"] = FLAG_SAVED
    st.rerun()


def _flag_editor(path: Path, data) -> None:
    st.subheader("Manual flags")
    st.caption("Flags are per current-run feature (a cluster's members are separate ids). A flag takes effect "
               "when the pipeline is rerun.")
    if saved := st.session_state.pop("flag_saved", None):
        st.success(saved)
    reviewer = st.session_state.get(app_state.REVIEWER_KEY, "").strip()
    ids = _members(data.classification)
    feature = st.selectbox("Current-run feature", ids, key="flag_feature")
    current = {k[1]: v for k, v in effective_flags(data.flags).items() if k[0] == feature}
    with st.form("flag_form"):
        wanted = {name: st.checkbox(FLAG_HELP[name], value=current.get(name, False), key=f"flag_{name}_{feature}")
                  for name in FLAG_NAMES}
        note = st.text_input("Note", key="flag_note")
        submitted = st.form_submit_button("Save flags", disabled=not reviewer)
    if not reviewer:
        st.caption(NEEDS_REVIEWER)
    if submitted:
        _save_flags(path, feature, wanted, current, note)
    st.caption("Flags in force, with the status the last run recorded.")
    st.dataframe(flag_table(data), hide_index=True)


def _report_view(path: Path, data, setup):
    """The dig list and the report's log warnings, from the files the report reads; (None, error) on refusal."""
    summary = data.match_summary
    out = Path(setup.out)
    try:
        settings = load_settings(setup.settings)
        inputs = load_report_inputs(
            path, out / "standardized" / f"{summary['prev_run_id']}.csv",
            out / "standardized" / f"{summary['curr_run_id']}.csv", out, setup.pipe, setup.mop, setup.segments,
            setup.crossings, setup.digs, setup.settings, criteria_path(setup.criteria, settings))
        return dig_list(inputs), inputs.log_warnings, None
    except (ValueError, OSError, KeyError) as exc:
        return None, [], str(exc)


def _downloads(path: Path) -> None:
    st.subheader("Downloads")
    present = [e for e in EXPORTS if (path / e[0]).is_file()]
    if not present:
        st.caption("No report files yet: rerun the pipeline to write them.")
    for name, mime, what in present:
        st.download_button(f"Download {name}", (path / name).read_bytes(), file_name=name, mime=mime,
                           key=f"download_{name}")
        st.caption(f"{name} is {what}. {DRAFT_LABEL}")


def export_page() -> None:
    st.title("Dig list and report export")
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.load_comparison(path) if path else None
    if data is None or data.classification.empty:
        st.info(NO_DATA)
        return
    setup = app_state.inputs_from_session(path)
    st.subheader("Dig list")
    if setup is None:
        st.warning("The dig list needs the run setup, which is not available: run the pipeline on the Run setup "
                   "page.")
    else:
        digs, warnings, error = _report_view(path, data, setup)
        for text in warnings:
            st.warning(text)
        if error:
            st.error(error)
        else:
            st.caption(f"{len(digs)} features on the dig list as of the last run. {DRAFT_LABEL}")
            st.dataframe(digs, hide_index=True)
    _flag_editor(path, data)
    st.subheader("Rerun")
    rerun_button(path, key="export_rerun")
    _downloads(path)
