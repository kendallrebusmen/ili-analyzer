"""The match review queue page: queue order, nearby features, and the decisions it writes."""

import shutil

from ili import pipeline

import pandas as pd
import pytest

from ili.align_welds import DRAFT_LABEL
from ili import app_pages_review
from ili.app_pages_review import NO_SETUP, SAVED, nearby_features, review_queue
from ili.review import REVIEWS_FILE, load_reviews
from test_app_setup import _all_text, _app, isolated_outputs, ran  # noqa: F401  (fixtures)

PAGE = "Match review queue"


def _table(rows):
    cols = ["match_id", "match_type", "confidence", "review_status", "needs_review", "prev_log_dist_ft",
            "curr_pos_in_prev_ft", "prev_feature_id", "curr_feature_id", "prev_clock_min", "curr_clock_min",
            "prev_depth_pct", "curr_depth_pct"]
    return pd.DataFrame(rows, columns=cols)


def test_review_queue_order():
    t = _table([
        ["M1", "1:1", "medium", "pending", True, 50.0, 50.0, "P1", "C1", 100, 100, 10.0, 11.0],
        ["M2", "1:1", "low", "pending", True, 80.0, 80.0, "P2", "C2", 100, 100, 10.0, 11.0],
        ["M3", "1:1", "low", "pending", True, 20.0, 20.0, "P3", "C3", 100, 100, 10.0, 11.0],
        ["M4", "1:1", "low", "accepted", True, 10.0, 10.0, "P4", "C4", 100, 100, 10.0, 11.0],
        ["M5", "1:1", "high", "pending", False, 5.0, 5.0, "P5", "C5", 100, 100, 10.0, 11.0],
        ["M6", "removed", "low", "pending", True, None, None, None, None, None, None, None, None]])
    t.loc[5, ["prev_log_dist_ft", "curr_pos_in_prev_ft"]] = [None, None]
    assert list(review_queue(t).match_id) == ["M3", "M2", "M6", "M1"]


def test_nearby_features_window():
    t = _table([
        ["M1", "1:1", "low", "pending", True, 100.0, 100.0, "P1", "C1", 100, 110, 10.0, 11.0],
        ["M2", "1:1", "low", "pending", True, 90.0, 90.0, "P2", "C2", 100, 110, 10.0, 11.0],
        ["M3", "1:1", "low", "pending", True, 110.0, 110.0, "P3", "C3", 100, 110, 10.0, 11.0],
        ["M4", "1:1", "low", "pending", True, 89.9, 110.1, "P4", "C4", 100, 110, 10.0, 11.0],
        ["M5", "inserted", "low", "pending", True, None, 105.0, None, "C5", None, 110, None, 11.0]])
    got = nearby_features(t, t.iloc[0], 10.0)
    assert list(got.columns) == ["run", "feature_id", "pos_prev_ft", "clock_min", "depth_pct", "match_id",
                                 "match_type"]
    assert set(zip(got.run, got.feature_id)) == {
        ("prev", "P1"), ("curr", "C1"), ("prev", "P2"), ("curr", "C2"), ("prev", "P3"), ("curr", "C3"),
        ("curr", "C5")}


def _page(folder, tmp_path, **state):
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    at = _app(comparison_dir=str(work), **state).run()
    at.sidebar.radio(key="page").set_value(PAGE).run()
    assert not at.exception
    return at, work


def _paired(folder):
    from ili.app_state import read_comparison
    queue = review_queue(read_comparison(folder).match_table)
    both = queue[queue.prev_feature_id.notna() & queue.curr_feature_id.notna()]
    assert not both.empty, "the synthetic comparison has no pending paired row"
    return both.iloc[0]


def test_accept_writes_log(ran, tmp_path):
    _, folder = ran
    row = _paired(folder)
    at, work = _page(folder, tmp_path)
    assert DRAFT_LABEL in _all_text(at)
    at.selectbox(key="review_pick").set_value(row.match_id).run()
    at.sidebar.text_input(key="reviewer").set_value("A. Reviewer").run()
    [b for b in at.button if b.label == "Accept"][0].click().run()
    assert not at.exception
    assert [e.value for e in at.success] == [SAVED]
    log = load_reviews(work / REVIEWS_FILE)
    assert len(log) == 1
    got = log.iloc[0]
    assert (got.decision, got.prev_feature_ids, got.curr_feature_ids, got.reviewer) == (
        "accept", row.prev_feature_id, row.curr_feature_id, "A. Reviewer")


def test_manual_cluster_writes_log(ran, tmp_path, monkeypatch):
    _, folder = ran
    wide = app_pages_review.load_settings()
    wide = wide.model_copy(update={"review": wide.review.model_copy(update={"nearby_window_ft": 500.0})})
    monkeypatch.setattr(app_pages_review, "load_settings", lambda: wide)
    row = _paired(folder)
    at, work = _page(folder, tmp_path)
    at.selectbox(key="review_pick").set_value(row.match_id).run()
    at.sidebar.text_input(key="reviewer").set_value("A. Reviewer").run()
    prev_opts, curr_opts = at.multiselect(key="manual_prev").options, at.multiselect(key="manual_curr").options
    assert len(prev_opts) >= 2 and len(curr_opts) >= 1
    want_prev, want_curr = [prev_opts[1], prev_opts[0]], [curr_opts[0]]
    at.multiselect(key="manual_prev").set_value(want_prev)
    at.multiselect(key="manual_curr").set_value(want_curr)
    at.text_input(key="manual_note").set_value("two pits, one report")
    [b for b in at.button if b.label == "Save manual re-pair"][0].click().run()
    assert not at.exception
    got = load_reviews(work / REVIEWS_FILE).iloc[0]
    assert got.decision == "manual"
    assert got.prev_feature_ids == ";".join(sorted(want_prev)) and got.curr_feature_ids == want_curr[0]
    assert got.note == "two pits, one report"


def test_buttons_disabled_without_reviewer(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path)
    labels = {b.label: b for b in at.button}
    assert labels["Accept"].disabled and labels["Reject"].disabled
    assert labels["Save manual re-pair"].disabled
    assert "reviewer" in _all_text(at)
    # the setup the pipeline recorded in the folder is found, so Rerun is enabled
    assert labels["Rerun the pipeline"].disabled is False and NO_SETUP not in _all_text(at)
    assert not (work / REVIEWS_FILE).exists()


def test_rerun_disabled_without_setup(ran, tmp_path):
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    (work / "pipeline_inputs.json").unlink()
    at = _app(comparison_dir=str(work)).run()
    at.sidebar.radio(key="page").set_value(PAGE).run()
    assert [b for b in at.button if b.label == "Rerun the pipeline"][0].disabled
    assert NO_SETUP in _all_text(at)


def test_not_applied_decisions_are_shown(ran, tmp_path):
    from ili.review import append_review
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    append_review(work / REVIEWS_FILE, "accept", ["NOPE-1"], ["NOPE-2"], "A. Reviewer")
    status = pd.DataFrame({"decision_id": ["R000001"], "decision": ["accept"], "applied_status": ["not_applied"],
                           "reason": ["unknown_feature"]})
    status.to_csv(work / "review_status.csv", index=False)
    at = _app(comparison_dir=str(work)).run()
    at.sidebar.radio(key="page").set_value(PAGE).run()
    assert not at.exception
    shown = " ".join(str(df.value.to_dict("list")) for df in at.dataframe)
    assert "unknown_feature" in shown and "NOPE-1" in shown


def _one_sided(folder):
    from ili.app_state import read_comparison
    queue = review_queue(read_comparison(folder).match_table)
    one = queue[queue.prev_feature_id.isna() | queue.curr_feature_id.isna()]
    assert not one.empty, "the synthetic comparison has no pending one-sided row"
    return one.iloc[0]


def test_one_sided_accept_writes_log_and_reject_is_disabled(ran, tmp_path):
    _, folder = ran
    row = _one_sided(folder)
    at, work = _page(folder, tmp_path, reviewer="A. Reviewer")
    at.selectbox(key="review_pick").set_value(row.match_id).run()
    labels = {b.label: b for b in at.button}
    assert labels["Reject"].disabled and not labels["Accept"].disabled
    assert "can be accepted" in _all_text(at)
    labels["Accept"].click().run()
    assert not at.exception
    got = load_reviews(work / REVIEWS_FILE).iloc[0]
    prev, curr = row.prev_feature_id, row.curr_feature_id
    assert got.decision == "accept"
    assert (got.prev_feature_ids, got.curr_feature_ids) == (prev if isinstance(prev, str) else "",
                                                            curr if isinstance(curr, str) else "")


def test_reject_writes_log(ran, tmp_path):
    _, folder = ran
    row = _paired(folder)
    at, work = _page(folder, tmp_path, reviewer="A. Reviewer")
    at.selectbox(key="review_pick").set_value(row.match_id).run()
    [b for b in at.button if b.label == "Reject"][0].click().run()
    assert not at.exception and [e.value for e in at.success] == [SAVED]
    got = load_reviews(work / REVIEWS_FILE).iloc[0]
    assert (got.decision, got.prev_feature_ids, got.curr_feature_ids) == (
        "reject", row.prev_feature_id, row.curr_feature_id)


def test_manual_needs_both_sides(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path, reviewer="A. Reviewer")
    opts = at.multiselect(key="manual_prev").options
    at.multiselect(key="manual_prev").set_value(opts[:1])
    [b for b in at.button if b.label == "Save manual re-pair"][0].click().run()
    assert [e.value for e in at.error] == ["Pick at least one previous-run and one current-run anomaly."]
    assert not (work / REVIEWS_FILE).exists()
    assert "same or adjacent weld-map row" in _all_text(at)


def test_rerun_runs_the_pipeline(ran):
    inputs, folder = ran
    at = _app(comparison_dir=str(folder), pipeline_inputs=inputs).run()
    at.sidebar.radio(key="page").set_value(PAGE).run()
    [b for b in at.button if b.label == "Rerun the pipeline"][0].click().run()
    assert not at.exception and not at.error
    assert [e.label for e in at.expander] == [f"{s}: exit 0" for s in pipeline.STAGES]
    assert at.success


def test_rerun_busy_shows_the_busy_result(ran):
    inputs, folder = ran
    at = _app(comparison_dir=str(folder), pipeline_inputs=inputs).run()
    at.sidebar.radio(key="page").set_value(PAGE).run()
    assert pipeline._RUN_LOCK.acquire(blocking=False)
    try:
        [b for b in at.button if b.label == "Rerun the pipeline"][0].click().run()
    finally:
        pipeline._RUN_LOCK.release()
    assert not at.exception
    assert [e.label for e in at.expander] == ["pipeline: exit 1"]
    assert at.error and pipeline.BUSY_MESSAGE in at.error[0].value
