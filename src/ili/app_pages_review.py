"""Match review queue page of the review app. DRAFT output only.

The page appends decisions to ``reviews.csv``; it never edits a match row. A decision takes effect when
the pipeline is rerun, which re-validates it (``ili.match``).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.app_pages_setup import NO_DATA
from ili.config import load_settings
from ili.match import CONFIDENCES, SIDE_FIELDS
from ili.pipeline import STAGES, run_pipeline
from ili.report_charts import INK, TEMPLATE
from ili.review import REVIEWS_FILE, append_review

NEARBY_COLUMNS = ["run", "feature_id", "pos_prev_ft", "clock_min", "depth_pct", "match_id", "match_type"]
RUN_COLOURS = {"prev": "#1F77B4", "curr": "#E8833A"}
SAVED = "Saved — rerun the pipeline to apply"
NEEDS_REVIEWER = "Enter the reviewer name in the sidebar to enable decisions."
NEEDS_BOTH_SIDES = "Accept and Reject need a feature on each side; use Manual re-pair to pair this one."
NO_SETUP = "Rerun is disabled: there is no run setup in this session. Run the pipeline on the Run setup page first."
NO_QUEUE = "No matches are pending review."


def _ids(value) -> list[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    return [i for i in str(value).split(";") if i]


def row_position_series(table: pd.DataFrame) -> pd.Series:
    """Each row's previous-run-frame position: the previous side's, else the current side's mapped one."""
    prev = pd.to_numeric(table.prev_log_dist_ft, errors="coerce").astype(float)
    return prev.fillna(pd.to_numeric(table.curr_pos_in_prev_ft, errors="coerce").astype(float))


def review_queue(match_table: pd.DataFrame) -> pd.DataFrame:
    """Rows that need review and are still pending: lowest confidence first, then position."""
    if match_table.empty:
        return match_table.iloc[0:0]
    t = match_table[match_table.needs_review.fillna(False).astype(bool) & (match_table.review_status == "pending")].copy()
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
    fig.update_layout(template=TEMPLATE, title="Nearby features in both runs", font={"color": INK},
                      xaxis_title="Position in the previous-run frame (ft)", yaxis_title="Clock (minutes)",
                      legend_title_text="Run")
    for run, name in (("prev", "Previous run"), ("curr", "Current run")):
        part = nearby[nearby.run == run]
        chosen = part.feature_id.isin(selected_ids.get(run, []))
        fig.add_trace(go.Scatter(
            x=part.pos_prev_ft, y=part.clock_min, mode="markers", name=name, text=part.feature_id,
            marker={"size": (6 + part.depth_pct.fillna(0) * 0.6).clip(upper=40), "color": RUN_COLOURS[run],
                    "opacity": 0.6, "line": {"width": chosen.map({True: 3, False: 0}), "color": INK}}))
    return fig


def _label(row) -> str:
    return f"{row.match_id}: {row.match_type}, {row.confidence}, {row.missing_hint or 'matched'}"


def _side_frame(row) -> pd.DataFrame:
    return pd.DataFrame({
        "field": [*SIDE_FIELDS, "feature_id"],
        "previous run": [row[f"prev_{f}"] for f in SIDE_FIELDS] + [row["prev_feature_id"]],
        "current run": [row[f"curr_{f}"] for f in SIDE_FIELDS] + [row["curr_feature_id"]]}).astype(str)


def not_applied(data) -> pd.DataFrame:
    """The decision log's entries that did not take (status other than applied), with their ids."""
    status = data.review_status
    if status.empty:
        return status
    shown = status[status.applied_status != "applied"]
    cols = [c for c in ("decision_id", "prev_feature_ids", "curr_feature_ids", "decided_utc") if c in data.reviews.columns]
    return shown.merge(data.reviews[cols], on="decision_id", how="left")


def _save(path, decision, prev_ids, curr_ids, note) -> None:
    try:
        append_review(path / REVIEWS_FILE, decision, prev_ids, curr_ids, st.session_state[app_state.REVIEWER_KEY],
                      note or None)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["review_saved"] = SAVED
    st.rerun()


def _rerun(path) -> None:
    setup = app_state.inputs_from_session(path)
    if st.button("Rerun the pipeline", disabled=setup is None, key="review_rerun"):
        with st.spinner("Running the pipeline"):
            results = run_pipeline(setup)
        failed = [r for r in results if r.exit_code != 0]
        if failed:
            st.error(f"{failed[0].stage} failed (exit {failed[0].exit_code})\n\n{failed[0].stderr}")
        elif len(results) == len(STAGES):
            st.success("Every stage finished with exit 0.")
    if setup is None:
        st.caption(NO_SETUP)


def review_queue_page() -> None:
    st.title("Match review queue")
    st.caption(DRAFT_LABEL)
    path = app_state.comparison_from_session()
    data = app_state.load_comparison(path) if path else None
    if data is None or data.match_table.empty:
        st.info(NO_DATA)
        return
    if saved := st.session_state.pop("review_saved", None):
        st.success(saved)
    reviewer = st.session_state.get(app_state.REVIEWER_KEY, "").strip()
    unapplied = not_applied(data)
    if not unapplied.empty:
        st.subheader("Decisions not applied")
        st.caption("These log entries did not take effect on the last run; the reason says why.")
        st.dataframe(unapplied, hide_index=True)
    queue = review_queue(data.match_table)
    if queue.empty:
        st.info(NO_QUEUE)
        _rerun(path)
        return
    st.metric("Pending review", len(queue))
    by_id = {r.match_id: r for r in queue.itertuples()}
    picked = st.selectbox("Queue (lowest confidence first)", list(by_id), key="review_pick",
                          format_func=lambda m: _label(by_id[m]))
    row = queue[queue.match_id == picked].iloc[0]
    st.write(f"Score {row['score']}, confidence {row['confidence']}, hint {row['missing_hint'] or 'none'}")
    st.dataframe(_side_frame(row), hide_index=True)
    nearby = nearby_features(data.match_table, row, load_settings().review.nearby_window_ft)
    prev_ids, curr_ids = _ids(row["prev_feature_id"]), _ids(row["curr_feature_id"])
    st.plotly_chart(nearby_chart(nearby, {"prev": prev_ids, "curr": curr_ids}))
    paired = bool(prev_ids and curr_ids)
    if not reviewer:
        st.caption(NEEDS_REVIEWER)
    elif not paired:
        st.caption(NEEDS_BOTH_SIDES)
    left, right = st.columns(2)
    if left.button("Accept", disabled=not (reviewer and paired), key="review_accept"):
        _save(path, "accept", prev_ids, curr_ids, None)
    if right.button("Reject", disabled=not (reviewer and paired), key="review_reject"):
        _save(path, "reject", prev_ids, curr_ids, None)
    with st.form("review_manual"):
        st.subheader("Manual re-pair")
        pick_prev = st.multiselect("Previous-run anomalies", sorted(nearby[nearby.run == "prev"].feature_id),
                                   key="manual_prev")
        pick_curr = st.multiselect("Current-run anomalies", sorted(nearby[nearby.run == "curr"].feature_id),
                                   key="manual_curr")
        note = st.text_input("Note", key="manual_note")
        submitted = st.form_submit_button("Save manual re-pair", disabled=not reviewer)
    if submitted:
        if not (pick_prev and pick_curr):
            st.error("Pick at least one previous-run and one current-run anomaly.")
        else:
            _save(path, "manual", pick_prev, pick_curr, note)
    _rerun(path)
