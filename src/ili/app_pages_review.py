"""Match review queue page of the review app. DRAFT output only.

The page appends decisions to ``reviews.csv``; it never edits a match row. A decision takes effect when
the pipeline is rerun, which re-validates it (``ili.match``).

A decision is bound to what was on screen. Match ids renumber on every pipeline run, so the queue is
keyed by the row's feature-id pair, and each button's callback carries the comparison folder, the feature
ids shown and the ``match_table.csv`` modification time of the render. If the table changed in between (a
rerun by this or another session), the click saves nothing and says so.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ili import app_state, app_theme
from ili.align_welds import DRAFT_LABEL
from ili.app_columns import column_config, field_label
from ili.app_pages_setup import NO_DATA, run_and_refresh, show_results
from ili.config import load_settings
from ili.match import CONFIDENCES, SIDE_FIELDS
from ili.review import REVIEWS_FILE, append_review, effective_reviews

SELECTED_OUTLINE = "#F79009"  # amber: reads on the light and the dark page
NEARBY_COLUMNS = ["run", "feature_id", "pos_prev_ft", "clock_min", "depth_pct", "match_id", "match_type"]
RUN_COLOURS = {"prev": "#1F77B4", "curr": "#E8833A"}
SAVED = "Saved — rerun the pipeline to apply"
SAVED_AFTER_RUN = "saved after the last run — rerun to apply"
NEEDS_REVIEWER = "Enter your name in the Reviewer box in the sidebar to enable decisions."
ONE_SIDED = "One-sided rows can be accepted (confirms missing/new) but not rejected."
MANUAL_RULES = ("Rules checked on rerun: all members in the same or adjacent weld-map row (an accept of a "
                "row the matcher itself makes is exempt); a 1:1 pair has one feature type; a side with several "
                "members is metal_loss only. A decision that breaks one is listed under Decisions not applied.")
CLUSTER_NOTE = "Members of a cluster share the cluster's position and values in the plot."
NO_SETUP = "Rerun is disabled: there is no run setup in this session. Run the pipeline on the Run setup page first."
NO_QUEUE = "No matches are pending review."
NO_DECISIONS = "No decisions are in force yet."
NEED_BOTH = "Pick at least one previous-run and one current-run anomaly."
STALE = ("The comparison was rerun after this page was drawn, so nothing was saved. The page now shows the "
         "new results: check the row and decide again.")
CORRECTION_HELP = ("Every decision in force: the latest log row for its features, with the status the last run "
                   "recorded. A new decision on any of its features supersedes it on the next rerun.")
DECISION_COLUMNS = ["decision_id", "decision", "prev_feature_ids", "curr_feature_ids", "reviewer", "decided_utc",
                    "note", "status"]
ROW_COLUMNS = ["match_id", "match_type", "review_status", "prev_feature_id", "curr_feature_id", "score",
               "confidence", "missing_hint"]
_SAVED_KEY, _ERROR_KEY = "review_saved", "review_error"


def _ids(value) -> list[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    return [i for i in str(value).split(";") if i]


def row_key(prev, curr) -> str:
    """A row's queue key: its feature-id pair (stable across reruns, unlike ``match_id``)."""
    return f"{';'.join(_ids(prev))}|{';'.join(_ids(curr))}"


def write_error(path: Path, exc: OSError) -> str:
    """The message for a log that could not be written: on Windows a CSV open in Excel is locked."""
    return (f"{Path(path).name} could not be written ({exc}). If it is open in Excel or another program, "
            f"close it and save again.")


def table_stamp(folder: Path) -> int | None:
    """The modification time of the folder's ``match_table.csv``: what a decision on screen was based on."""
    p = Path(folder) / "match_table.csv"
    return p.stat().st_mtime_ns if p.is_file() else None


def row_position_series(table: pd.DataFrame) -> pd.Series:
    """Each row's previous-run-frame position: the previous side's, else the current side's mapped one."""
    prev = pd.to_numeric(table.prev_log_dist_ft, errors="coerce").astype(float)
    return prev.fillna(pd.to_numeric(table.curr_pos_in_prev_ft, errors="coerce").astype(float))


def review_queue(match_table: pd.DataFrame) -> pd.DataFrame:
    """Rows that need review and are still pending: lowest confidence first, then position."""
    if match_table.empty:
        return match_table.iloc[0:0]
    pending = match_table.needs_review.fillna(False).astype(bool) & (match_table.review_status == "pending")
    t = match_table[pending].copy()
    t["_rank"] = t.confidence.map({c: i for i, c in enumerate(reversed(CONFIDENCES))}).fillna(len(CONFIDENCES))
    t["_pos"] = row_position_series(t)
    return (t.sort_values(["_rank", "_pos", "match_id"], kind="stable", na_position="last")
            .drop(columns=["_rank", "_pos"]).reset_index(drop=True))


def _side_features(table: pd.DataFrame, run: str, pos_col: str) -> pd.DataFrame:
    """One row per feature id on a side (a cluster's members share the cluster's position and values)."""
    base = pd.DataFrame({
        "run": run, "pos_prev_ft": pd.to_numeric(table[pos_col], errors="coerce").astype(float),
        "clock_min": pd.to_numeric(table[f"{run}_clock_min"], errors="coerce").astype(float),
        "depth_pct": pd.to_numeric(table[f"{run}_depth_pct"], errors="coerce").astype(float),
        "match_id": table.match_id, "match_type": table.match_type,
        "feature_id": [_ids(v) for v in table[f"{run}_feature_id"]]})
    return base.explode("feature_id").dropna(subset=["feature_id"])


def nearby_features(match_table: pd.DataFrame, row, window_ft: float) -> pd.DataFrame:
    """Every anomaly of both runs within ``window_ft`` (edges included) of the row's previous-run-frame
    position, one row per run and feature id."""
    if row is None:
        return pd.DataFrame(columns=NEARBY_COLUMNS)
    ref = row_position_series(pd.DataFrame([row])).iloc[0]
    if match_table.empty or pd.isna(ref):
        return pd.DataFrame(columns=NEARBY_COLUMNS)
    both = pd.concat([_side_features(match_table, "prev", "prev_log_dist_ft"),
                      _side_features(match_table, "curr", "curr_pos_in_prev_ft")], ignore_index=True)
    both = both[(both.pos_prev_ft - ref).abs() <= window_ft + 1e-9].drop_duplicates(["run", "feature_id"])
    return both[NEARBY_COLUMNS].sort_values(["pos_prev_ft", "run", "feature_id"]).reset_index(drop=True)


def nearby_chart(nearby: pd.DataFrame, selected_ids: dict[str, list[str]]) -> go.Figure:
    """Position (x) against clock (y); depth sets the marker size, one colour per run, the selected
    row's features outlined."""
    fig = go.Figure()
    fig.update_layout(title="Nearby features in both runs",
                      xaxis_title="Position in the previous-run frame (ft)", yaxis_title="Clock (minutes)",
                      legend_title_text="Run")
    for run, name in (("prev", "Previous run"), ("curr", "Current run")):
        part = nearby[nearby.run == run]
        chosen = part.feature_id.isin(selected_ids.get(run, [])).to_numpy(bool)
        depth = pd.to_numeric(part.depth_pct, errors="coerce").astype(float).fillna(0).to_numpy()
        fig.add_trace(go.Scatter(
            x=part.pos_prev_ft, y=part.clock_min, mode="markers", name=name, text=part.feature_id,
            marker={"size": (6 + depth * 0.6).clip(max=40), "color": RUN_COLOURS[run],
                    "opacity": 0.6, "line": {"width": [3 if c else 0 for c in chosen], "color": SELECTED_OUTLINE}}))
    return fig


def _label(row) -> str:
    ids = f"{row.prev_feature_id or '(none)'} / {row.curr_feature_id or '(none)'}"
    return f"{ids}: {row.match_type}, {row.confidence}, {row.missing_hint or 'matched'} ({row.match_id})"


SIDE_DECIMALS = 3  # presentation only: the match table keeps full precision


def _cell(value) -> str:
    """A side-by-side cell: blank when missing, floats to at most ``SIDE_DECIMALS`` places, else as text."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return ""
    if isinstance(value, float):
        return f"{value:.{SIDE_DECIMALS}f}".rstrip("0").rstrip(".")
    return str(value)


def _side_frame(row) -> pd.DataFrame:
    return pd.DataFrame({
        "field": [field_label(f) for f in (*SIDE_FIELDS, "feature_id")],
        "previous run": [_cell(row[f"prev_{f}"]) for f in SIDE_FIELDS] + [_cell(row["prev_feature_id"])],
        "current run": [_cell(row[f"curr_{f}"]) for f in SIDE_FIELDS] + [_cell(row["curr_feature_id"])]})


def not_applied(data) -> pd.DataFrame:
    """The decision log's entries that did not take (status other than applied), with their ids."""
    status = data.review_status
    if status.empty:
        return status
    shown = status[status.applied_status != "applied"]
    cols = [c for c in ("decision_id", "prev_feature_ids", "curr_feature_ids", "decided_utc")
            if c in data.reviews.columns]
    return shown.merge(data.reviews[cols], on="decision_id", how="left")


def decisions_in_force(data) -> pd.DataFrame:
    """The decisions in force in the current log (``effective_reviews``: the latest row for each feature)
    with the status the last run recorded (``applied``, ``not_applied: reason``), or ``SAVED_AFTER_RUN``
    for a row no run has seen. A not-applied decision is still in force: it superseded earlier ones."""
    if data.reviews.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    rows = effective_reviews(data.reviews).decisions.reset_index(drop=True)
    recorded = data.review_status
    outcome = {} if recorded.empty else {
        r.decision_id: r.applied_status + (f": {r.reason}" if isinstance(r.reason, str) and r.reason else "")
        for r in recorded.itertuples()}
    return rows.assign(status=[outcome.get(d, SAVED_AFTER_RUN) for d in rows.decision_id])[DECISION_COLUMNS]


def save_decision(path: str, stamp: int | None, decision: str, prev_ids: list[str], curr_ids: list[str],
                  note: str | None = None) -> None:
    """Button callback: append the decision with the folder and ids bound when the page was drawn. Saves
    nothing when ``match_table.csv`` changed since (``STALE``). The outcome is shown on the next draw."""
    if table_stamp(Path(path)) != stamp:
        st.session_state[_ERROR_KEY] = STALE
        return
    reviewer = st.session_state.get(app_state.REVIEWER_KEY, "").strip()
    if not reviewer:
        st.session_state[_ERROR_KEY] = NEEDS_REVIEWER
        return
    log = Path(path) / REVIEWS_FILE
    try:
        append_review(log, decision, prev_ids, curr_ids, reviewer, note or None)
    except ValueError as exc:
        st.session_state[_ERROR_KEY] = str(exc)
    except OSError as exc:
        st.session_state[_ERROR_KEY] = write_error(log, exc)
    else:
        st.session_state[_SAVED_KEY] = SAVED


def save_manual(path: str, stamp: int | None, prefix: str) -> None:
    """Form callback: the manual re-pair picked in the form with key prefix ``prefix``."""
    prev = list(st.session_state.get(f"{prefix}manual_prev") or [])
    curr = list(st.session_state.get(f"{prefix}manual_curr") or [])
    if not (prev and curr):
        st.session_state[_ERROR_KEY] = NEED_BOTH
        return
    save_decision(path, stamp, "manual", prev, curr, st.session_state.get(f"{prefix}manual_note"))


def _decision_panel(path: Path, data, stamp, reviewer: str, prev_ids: list[str], curr_ids: list[str], ref_row,
                    prefix: str, manual_prefix: str, lead=None) -> None:
    """Two columns: the record card (what ``lead`` draws, then the Accept / Reject action bar) and the
    nearby-features chart card; below them the Manual re-pair form. Each action is bound (callback arguments)
    to the folder, ``prev_ids`` / ``curr_ids`` and ``stamp``."""
    nearby = nearby_features(data.match_table, ref_row, load_settings().review.nearby_window_ft)
    paired = bool(prev_ids and curr_ids)
    left_col, right_col = st.columns([3, 2])
    with left_col, st.container(border=True, key=f"{prefix}card_side"):
        if lead is not None:
            lead()
        if not reviewer:
            st.caption(NEEDS_REVIEWER)
        elif not paired:
            st.caption(ONE_SIDED)
        with st.container(key=f"{prefix}actions"):  # the design's action bar: kept in view by app_theme's CSS
            left, right = st.columns(2)
            left.button("Accept", type="primary", disabled=not (reviewer and (prev_ids or curr_ids)),
                        key=f"{prefix}accept", on_click=save_decision,
                        args=(str(path), stamp, "accept", prev_ids, curr_ids))
            right.button("Reject", disabled=not (reviewer and paired), key=f"{prefix}reject",
                         on_click=save_decision, args=(str(path), stamp, "reject", prev_ids, curr_ids))
    with right_col, st.container(border=True, key=f"{prefix}card_chart"):
        st.markdown(app_theme.card_title("Nearby features"), unsafe_allow_html=True)
        st.plotly_chart(app_theme.style_chart(nearby_chart(nearby, {"prev": prev_ids, "curr": curr_ids})),
                        theme="streamlit", key=f"{prefix}chart")
        st.caption(CLUSTER_NOTE)
    with st.form(f"{prefix}manual"):
        st.subheader("Manual re-pair")
        st.caption(MANUAL_RULES)
        st.multiselect("Previous-run anomalies", sorted(nearby[nearby.run == "prev"].feature_id),
                       key=f"{manual_prefix}manual_prev")
        st.multiselect("Current-run anomalies", sorted(nearby[nearby.run == "curr"].feature_id),
                       key=f"{manual_prefix}manual_curr")
        st.text_input("Note", key=f"{manual_prefix}manual_note")
        st.form_submit_button("Save manual re-pair", disabled=not reviewer, on_click=save_manual,
                              args=(str(path), stamp, manual_prefix))


def _queue(path: Path, data, stamp, reviewer: str) -> None:
    queue = review_queue(data.match_table)
    if queue.empty:
        st.info(NO_QUEUE)
        return
    keys = [row_key(p, c) for p, c in zip(queue.prev_feature_id, queue.curr_feature_id)]
    labels = dict(zip(keys, (_label(r) for r in queue.itertuples())))
    picked = st.selectbox("Queue (lowest confidence first)", keys, key="review_pick", format_func=labels.get)
    position = keys.index(picked)
    row = queue.iloc[position]
    score = row["score"]

    def lead() -> None:
        st.markdown(app_theme.object_header_html(
            f"Match {row['match_id']}",
            [app_theme.confidence_pill_html(row["confidence"]), app_theme.pill_html("Pending", "warning")],
            [("Type", row["match_type"]), ("Score", "—" if pd.isna(score) else f"{score:.2f}"),
             ("Hint", row["missing_hint"] or "matched"),
             ("Queue", f"{position + 1} of {len(queue)} pending")]), unsafe_allow_html=True)
        st.dataframe(_side_frame(row), hide_index=True)

    _decision_panel(path, data, stamp, reviewer, _ids(row["prev_feature_id"]), _ids(row["curr_feature_id"]), row,
                    "review_", "", lead)


def _rows_of(table: pd.DataFrame, prev_ids: list[str], curr_ids: list[str]) -> pd.DataFrame:
    """The match rows that hold any of the ids (as the last run left them)."""
    hit = (table.prev_feature_id.map(lambda c: bool(set(_ids(c)) & set(prev_ids)))
           | table.curr_feature_id.map(lambda c: bool(set(_ids(c)) & set(curr_ids))))
    return table[hit]


def _corrections(path: Path, data, stamp, reviewer: str) -> None:
    with st.expander("Correct an earlier decision"):
        _correction_body(path, data, stamp, reviewer)


def _correction_body(path: Path, data, stamp, reviewer: str) -> None:
    st.caption(CORRECTION_HELP)
    decided = decisions_in_force(data)
    if decided.empty:
        st.caption(NO_DECISIONS)
        return
    by_id = {r.decision_id: r for r in decided.itertuples()}
    picked = st.selectbox(
        "Decision in force", list(by_id), key="fix_pick",
        format_func=lambda d: (f"{d} {by_id[d].decision}: {by_id[d].prev_feature_ids or '(none)'} / "
                               f"{by_id[d].curr_feature_ids or '(none)'} ({by_id[d].status})"))
    d = by_id[picked]
    prev_ids, curr_ids = _ids(d.prev_feature_ids), _ids(d.curr_feature_ids)
    rows = _rows_of(data.match_table, prev_ids, curr_ids)
    shown = rows[ROW_COLUMNS]

    def lead() -> None:
        st.write(f"Current decision: {d.decision} by {d.reviewer} at {d.decided_utc}. Status: {d.status}.")
        st.dataframe(shown, hide_index=True, column_config=column_config(shown))

    _decision_panel(path, data, stamp, reviewer, prev_ids, curr_ids, rows.iloc[0] if len(rows) else None,
                    "fix_", "fix_", lead)


def rerun_button(path, key: str = "review_rerun") -> None:
    """The Rerun button. A finished run redraws the page (``st.rerun``) so everything on it shows the new
    outputs, then lists the stage results; disabled when no run setup is known for the folder."""
    setup = app_state.inputs_from_session(path)
    if st.button("Rerun the pipeline", disabled=setup is None, key=key):
        run_and_refresh(setup, f"{key}_results")
    if (results := st.session_state.pop(f"{key}_results", None)) is not None:
        show_results(results)
    if setup is None:
        st.caption(NO_SETUP)


def review_queue_page() -> None:
    path = app_state.comparison_from_session()
    app_theme.page_header("Match review queue", app_state.comparison_context(path))
    data = app_state.page_data(path)
    if data is None or data.match_table.empty:
        st.info(NO_DATA)
        return
    if saved := st.session_state.pop(_SAVED_KEY, None):
        st.success(saved)
    if error := st.session_state.pop(_ERROR_KEY, None):
        st.error(error)
    stamp = table_stamp(path)
    reviewer = st.session_state.get(app_state.REVIEWER_KEY, "").strip()
    unapplied = not_applied(data)
    if not unapplied.empty:
        with st.container(border=True, key="card_unapplied"):
            st.markdown(app_theme.card_title("Decisions not applied"), unsafe_allow_html=True)
            st.markdown(app_theme.muted_html("These log entries did not take effect on the last run; the reason "
                                             "says why."), unsafe_allow_html=True)
            st.dataframe(unapplied, hide_index=True, column_config=column_config(unapplied))
    _queue(path, data, stamp, reviewer)
    _corrections(path, data, stamp, reviewer)
    rerun_button(path)
