"""Manual flags in classification and the review/flag logs in the report (spec 6.8 IMM-5, D180-7; 6.10.8)."""

import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from conftest import MAP_A, MAP_B
from test_classify_cli import _classify_args
from test_classify_vars import CROSSINGS, CURR_STD, PIPE, RL_EMPTY, SEGS, mrow, table
from test_rl_cli import _args as rl_args

from ili import align_welds, classify, growth, ingest, match, remaining_life
from ili.category_change import category_changes
from ili.classify import feature_variables
from ili.config import DEFAULT_CRITERIA, DEFAULT_SETTINGS, load_repair_criteria, load_settings
from ili.report import ReportHeader, render_report
from ili.report_data import load_report_inputs, pending_reviews
from ili.review import (FLAGS_FILE, REVIEWS_FILE, append_flag, append_review, effective_flags, effective_reviews,
                        load_flags)

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def _t(k):
    return T0 + timedelta(minutes=k)


def _flags(folder, *rows):
    """Effective flags of a log made from (feature, flag, value) rows, one minute apart."""
    folder.mkdir(parents=True, exist_ok=True)
    for k, (fid, flag, value) in enumerate(rows):
        append_flag(folder / FLAGS_FILE, fid, flag, value, "Pat", now=_t(k))
    return effective_flags(load_flags(folder / FLAGS_FILE))


def _categories(rows, flags, settings):
    v = feature_variables(table(rows), RL_EMPTY, CURR_STD, PIPE, SEGS, CROSSINGS, settings, flags=flags)
    res = classify.classify(v, load_repair_criteria(), settings, T0.date(), discovery_date=T0.date())
    return v, dict(zip(res.table.match_id, res.table.governing_category)), res


@pytest.fixture
def settings():
    return load_settings()


def test_flags_drive_imm5_and_d180_7(settings, tmp_path):
    rows = [mrow("M1", pos=1020.0, depth_pct=10.0)]  # HCA segment, shallow: nothing triggers by itself
    assert _categories(rows, None, settings)[1]["M1"] == "evaluate"
    crack = _flags(tmp_path / "a", ("B-M1", "field_crack_confirmed", True))
    v, cat, res = _categories(rows, crack, settings)
    assert cat["M1"] == "180_day" and res.table.governing_rule.iloc[0] == "D180-7"
    assert v.field_crack_confirmed.iloc[0] is True and v.manual_immediate.iloc[0] is False
    both = _flags(tmp_path / "b", ("B-M1", "field_crack_confirmed", True), ("B-M1", "manual_immediate", True))
    _, cat, res = _categories(rows, both, settings)
    assert cat["M1"] == "immediate" and res.table.governing_rule.iloc[0] == "IMM-5"
    # a later row clears the manual flag: back to 180-day
    cleared = _flags(tmp_path / "c", ("B-M1", "field_crack_confirmed", True), ("B-M1", "manual_immediate", True),
                     ("B-M1", "manual_immediate", False))
    assert _categories(rows, cleared, settings)[1]["M1"] == "180_day"
    # a flag on another feature changes nothing
    other = _flags(tmp_path / "d", ("B-OTHER", "manual_immediate", True))
    v, cat, _ = _categories(rows, other, settings)
    assert cat["M1"] == "evaluate" and v.manual_immediate.iloc[0] is False


def test_cluster_flag_any_member(settings, tmp_path):
    cluster = mrow("M1", mtype="matched_cluster", pos=1020.0, depth_pct=10.0)
    cluster["curr_feature_id"] = "B-1;B-2;B-3"
    plain_row = mrow("M2", pos=1100.0, depth_pct=10.0)
    on = _flags(tmp_path / "a", ("B-2", "manual_immediate", True))
    v, cat, _ = _categories([cluster, plain_row], on, settings)
    assert cat == {"M1": "immediate", "M2": "evaluate"}
    assert list(v.manual_immediate) == [True, False]
    off = _flags(tmp_path / "b", ("B-2", "manual_immediate", True), ("B-2", "manual_immediate", False))
    v, _, _ = _categories([cluster, plain_row], off, settings)
    assert list(v.manual_immediate) == [False, False] and list(v.field_crack_confirmed) == [False, False]


def test_flag_status_lists_every_row(tmp_path):
    path = tmp_path / FLAGS_FILE
    append_flag(path, "B-1", "manual_immediate", True, "Pat", now=_t(0))
    append_flag(path, "B-1", "manual_immediate", False, "Pat", now=_t(1))
    append_flag(path, "GONE", "manual_immediate", True, "Pat", now=_t(2))
    append_flag(path, "W-1", "field_crack_confirmed", True, "Pat", now=_t(3))
    status = classify.flag_status(load_flags(path), classified={"B-1"}, known={"B-1", "W-1"})
    assert list(status.columns) == ["flag_id", "curr_feature_id", "flag", "applied_status", "reason"]
    assert list(zip(status.flag_id, status.applied_status, status.reason)) == [
        ("F000001", "superseded", None), ("F000002", "applied", None),
        ("F000003", "not_applied", "unknown_feature"), ("F000004", "not_applied", "not_anomaly")]
    assert classify.flag_status(load_flags(tmp_path / "none.csv"), set(), set()).empty


# ---- pipeline ---------------------------------------------------------------------------------------

def _pipeline(data, out, *, reviews=(), flags=()):
    """ingest, align, the log files into the comparison dir, match, growth and remaining life."""
    assert ingest.main(["--run", str(data / "tally_prev_vendor_a.csv"), str(MAP_A), "2021-06-15",
                        "--run", str(data / "tally_curr_vendor_b.xlsx"), str(MAP_B), "--out", str(out)]) == 0
    std = out / "standardized"
    ref = data / "reference"
    assert align_welds.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                             "--repairs", str(ref / "repairs_between_runs.csv"), "--out", str(out)]) == 0
    folder = out / "2021_vendorA_vs_2026_vendorB"
    for k, (decision, prev, curr, note) in enumerate(reviews):
        append_review(folder / REVIEWS_FILE, decision, prev, curr, "Pat <Lee>", note, now=_t(k))
    for k, (fid, flag, value, note) in enumerate(flags):
        append_flag(folder / FLAGS_FILE, fid, flag, value, "Pat <Lee>", note, now=_t(k))
    assert match.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                       "--weld-map", str(folder / "weld_map.csv"), "--pipe", str(ref / "pipe_properties.csv"),
                       "--repairs", str(ref / "repairs_between_runs.csv"), "--out", str(out)]) == 0
    assert growth.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                        "--matches", str(folder / "match_table.csv"), "--pipe", str(ref / "pipe_properties.csv"),
                        "--segments", str(ref / "hca_segments.csv"), "--out", str(out)]) == 0
    assert remaining_life.main(rl_args(std, folder, data, out)) == 0
    return std, folder


@pytest.fixture(scope="module")
def plain(synth, tmp_path_factory):
    _, data = synth
    out = tmp_path_factory.mktemp("flags_plain") / "out"
    std, folder = _pipeline(data, out)
    return data, out, std, folder


def test_unknown_flag_feature_listed(plain, tmp_path, capsys):
    data, out0, std, folder0 = plain
    out = tmp_path / "out"
    shutil.copytree(out0, out)
    folder = out / folder0.name
    mt = pd.read_csv(folder / "match_table.csv", dtype=str)
    known = mt[mt.match_type.isin(["matched_1to1", "new"])].curr_feature_id.iloc[0]
    append_flag(folder / FLAGS_FILE, "NO-SUCH-FEATURE", "manual_immediate", True, "Pat", now=_t(0))
    append_flag(folder / FLAGS_FILE, known, "manual_immediate", True, "Pat", now=_t(1))
    capsys.readouterr()
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    printed = capsys.readouterr().out
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["flags_not_applied"] == [{"flag_id": "F000001", "curr_feature_id": "NO-SUCH-FEATURE",
                                             "flag": "manual_immediate", "reason": "unknown_feature"}]
    assert summary["flags_file"] == str(folder / FLAGS_FILE)
    assert summary["flags_sha256"] == hashlib.sha256((folder / FLAGS_FILE).read_bytes()).hexdigest()
    assert "NO-SUCH-FEATURE" in printed and "not applied" in printed
    cls = classify.load_classification_table(folder / "classification.csv")
    hit = cls[cls.curr_feature_id.map(lambda c: known in c.split(";"))]
    assert len(hit) == 1 and hit.manual_immediate.iloc[0] is True and hit.governing_category.iloc[0] == "immediate"
    assert cls.manual_immediate.sum() == 1
    # without a log: no flags, recorded as null; an explicit --flags file that is missing is exit 2
    (folder / FLAGS_FILE).unlink()
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    summary = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert summary["flags_file"] is None and summary["flags_sha256"] is None and summary["flags_not_applied"] == []
    capsys.readouterr()
    assert classify.main(_classify_args(std, folder, data, out, "--flags", str(tmp_path / "missing.csv"))) == 2
    assert "missing.csv" in capsys.readouterr().err


def _row(html, key):
    hit = re.search(rf"<tr[^>]*>(?:(?!</tr>).)*>{key}<(?:(?!</tr>).)*</tr>", html, re.S)
    assert hit, key
    return hit.group(0)


def test_report_shows_review_and_flag_logs(plain, synth, tmp_path, capsys):
    _, data = synth
    base = pd.read_csv(plain[3] / "match_table.csv", dtype=str)
    pending_before = len(pending_reviews(base.assign(needs_review=base.needs_review == "True")))
    one = base[(base.match_type == "matched_1to1") & (base.needs_review == "True")]
    miss = base[(base.match_type == "missing") & (base.needs_review == "True")].iloc[0]
    new = base[base.match_type == "new"].iloc[0]
    reviews = [("accept", [one.prev_feature_id.iloc[0]], [one.curr_feature_id.iloc[0]], 'needs <b>care</b> & "q"'),
               ("reject", [one.prev_feature_id.iloc[1]], [one.curr_feature_id.iloc[1]], None),
               ("manual", [miss.prev_feature_id], [new.curr_feature_id], None),
               ("accept", ["NO-PREV"], ["NO-CURR"], None)]
    flags = [(one.curr_feature_id.iloc[0], "manual_immediate", True, "field <i>note</i>"),
             ("NO-SUCH-FEATURE", "field_crack_confirmed", True, None)]
    out = tmp_path / "out"
    std, folder = _pipeline(data, out, reviews=reviews, flags=flags)
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    capsys.readouterr()
    ref = data / "reference"
    inputs = load_report_inputs(folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", out,
                                ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                                ref / "foreign_crossings.csv", ref / "digs.csv", DEFAULT_SETTINGS, DEFAULT_CRITERIA)
    assert len(inputs.reviews) == 4 and len(inputs.flags) == 2 and len(inputs.review_status) == 4
    assert inputs.paths["reviews"] == (folder / REVIEWS_FILE).resolve()
    assert inputs.paths["flags"] == (folder / FLAGS_FILE).resolve()
    assert inputs.paths["review_status"] == (folder / "review_status.csv").resolve()
    changes = category_changes(inputs.match_table, inputs.remaining_life, inputs.classification, inputs.curr_std,
                               inputs.pipe, inputs.mop, inputs.segments, inputs.crossings, inputs.settings,
                               inputs.criteria, inputs.prev_std.run_date.dropna().iloc[0])
    html = render_report(inputs, changes, ReportHeader(None, None, None), datetime(2026, 10, 27, tzinfo=timezone.utc))
    status = dict(zip(inputs.review_status.decision_id, inputs.review_status.applied_status))
    assert status["R000004"] == "not_applied" and status["R000001"] == "applied"
    for decision_id in inputs.reviews.decision_id:
        assert status[decision_id] in _row(html, decision_id), decision_id
    assert "unknown_feature" in _row(html, "R000004")
    assert "applied" in _row(html, "F000001") and "not_applied" in _row(html, "F000002")
    assert "unknown_feature" in _row(html, "F000002")
    # reviewer and note text is user text: escaped, never markup
    assert "needs &lt;b&gt;care&lt;/b&gt;" in html and "<b>care</b>" not in html
    assert "field &lt;i&gt;note&lt;/i&gt;" in html and "<i>note</i>" not in html
    assert "Pat &lt;Lee&gt;" in html and "Pat <Lee>" not in html
    # the pending count drops by the reviewed rows (the accepted and the rejected one)
    pending_after = len(pending_reviews(inputs.match_table))
    assert pending_after <= pending_before - 2 and f"{pending_after} matches pending review —" in html


# ---- logs saved after the last run -----------------------------------------------------------------

def _report(data, std, folder, out):
    ref = data / "reference"
    inputs = load_report_inputs(folder, std / "2021_vendorA.csv", std / "2026_vendorB.csv", out,
                                ref / "pipe_properties.csv", ref / "mop_profile.csv", ref / "hca_segments.csv",
                                ref / "foreign_crossings.csv", ref / "digs.csv", DEFAULT_SETTINGS, DEFAULT_CRITERIA)
    changes = category_changes(inputs.match_table, inputs.remaining_life, inputs.classification, inputs.curr_std,
                               inputs.pipe, inputs.mop, inputs.segments, inputs.crossings, inputs.settings,
                               inputs.criteria, inputs.prev_std.run_date.dropna().iloc[0])
    html = render_report(inputs, changes, ReportHeader(None, None, None), datetime(2026, 10, 27, tzinfo=timezone.utc))
    return inputs, html


@pytest.fixture
def classified(plain, tmp_path, capsys):
    """A copy of the plain pipeline (no logs) with classification run on it."""
    data, out0, std, folder0 = plain
    out = tmp_path / "out"
    shutil.copytree(out0, out)
    folder = out / folder0.name
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    capsys.readouterr()
    return data, out, std, folder


def test_flag_status_file_is_written(classified):
    _, _, _, folder = classified
    text = (folder / "flag_status.csv").read_text(encoding="utf-8")
    assert text == "flag_id,curr_feature_id,flag,applied_status,reason\n"
    assert json.loads((folder / "match_summary.json").read_text(encoding="utf-8"))["reviews_sha256"] is None


def test_unchanged_logs_give_no_warning(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _pipeline(data, out, reviews=[("accept", ["NO-PREV"], ["NO-CURR"], None)],
                            flags=[("NO-SUCH", "manual_immediate", True, None)])
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    capsys.readouterr()
    summary = json.loads((folder / "match_summary.json").read_text(encoding="utf-8"))
    assert summary["reviews_sha256"] == hashlib.sha256((folder / REVIEWS_FILE).read_bytes()).hexdigest()
    inputs, html = _report(data, std, folder, out)
    assert inputs.log_warnings == [] and "saved after the last run" not in html
    assert list(inputs.flag_status.applied_status) == ["not_applied"]


def test_flag_saved_after_classify_is_unrecorded(classified):
    data, out, std, folder = classified
    base, html0 = _report(data, std, folder, out)
    pending0 = len(pending_reviews(base.match_table))
    assert base.log_warnings == []
    append_flag(folder / FLAGS_FILE, "B-LATE", "manual_immediate", True, "Pat", "late <b>", now=_t(0))
    inputs, html = _report(data, std, folder, out)
    assert len(inputs.log_warnings) == 1 and FLAGS_FILE in inputs.log_warnings[0]
    assert "saved after the last run — rerun to apply" in _row(html, "F000001")
    assert f"{pending0} matches pending review, plus 1 log row saved after the last run —" in html
    assert 'class="warning">feature_flags.csv has changed' in html


def test_review_saved_after_match_is_unrecorded(classified):
    data, out, std, folder = classified
    pending0 = len(pending_reviews(_report(data, std, folder, out)[0].match_table))
    append_review(folder / REVIEWS_FILE, "accept", ["NO-PREV"], ["NO-CURR"], "Pat", None, now=_t(0))
    inputs, html = _report(data, std, folder, out)
    assert len(inputs.log_warnings) == 1 and REVIEWS_FILE in inputs.log_warnings[0]
    assert "saved after the last run — rerun to apply" in _row(html, "R000001")
    assert f"{pending0} matches pending review, plus 1 log row saved after the last run —" in html


def test_pending_wording_is_the_same_in_report_and_cli(classified, capsys):
    """M1: pending matches and log rows saved after the last run are counted apart, with the same words in
    the report and the report CLI; either one keeps the not-final warning."""
    from test_report_cli import _argv

    from ili import report
    data, out, std, folder = classified
    pending0 = len(pending_reviews(_report(data, std, folder, out)[0].match_table))
    assert pending0 > 1
    append_review(folder / REVIEWS_FILE, "accept", ["NO-PREV"], ["NO-CURR"], "Pat", None, now=_t(0))
    append_flag(folder / FLAGS_FILE, "B-LATE", "manual_immediate", True, "Pat", None, now=_t(1))
    inputs, html = _report(data, std, folder, out)
    text = f"{pending0} matches pending review, plus 2 log rows saved after the last run"
    assert f"{text} — this report cannot be final." in html
    capsys.readouterr()
    assert report.main(_argv((data, out, std, folder))) == 0
    assert f"pending: {text} (the report cannot be final)" in capsys.readouterr().out
    assert report.pending_text(1, 0) == "1 match pending review"
    assert report.pending_text(0, 1) == "0 matches pending review, plus 1 log row saved after the last run"
    # no pending match, but a log row no run has applied: still not final
    import dataclasses
    done = dataclasses.replace(inputs, match_table=inputs.match_table.assign(review_status="accepted"))
    changes = category_changes(inputs.match_table, inputs.remaining_life, inputs.classification, inputs.curr_std,
                               inputs.pipe, inputs.mop, inputs.segments, inputs.crossings, inputs.settings,
                               inputs.criteria, inputs.prev_std.run_date.dropna().iloc[0])
    html = render_report(done, changes, ReportHeader(None, None, None), datetime(2026, 10, 27, tzinfo=timezone.utc))
    assert ("0 matches pending review, plus 2 log rows saved after the last run — this report cannot be final."
            in html)


def test_recorded_hashes_are_of_the_parsed_bytes(plain, tmp_path, monkeypatch, capsys):
    """A log appended to after it was parsed must not be recorded as applied: the summaries hash the
    bytes that were parsed (read once), so the report then sees the file as changed."""
    data, out0, std, folder0 = plain
    out = tmp_path / "out"
    shutil.copytree(out0, out)
    folder = out / folder0.name
    append_review(folder / REVIEWS_FILE, "accept", ["NO-PREV"], ["NO-CURR"], "Pat", None, now=_t(0))
    append_flag(folder / FLAGS_FILE, "NO-SUCH", "manual_immediate", True, "Pat", None, now=_t(0))
    parsed = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in (REVIEWS_FILE, FLAGS_FILE)}

    def late_review(log):  # another session appends right after the log was parsed
        append_review(folder / REVIEWS_FILE, "accept", ["LATE-P"], ["LATE-C"], "Pat", None, now=_t(9))
        return effective_reviews(log)

    def late_flag(log):
        append_flag(folder / FLAGS_FILE, "LATE", "manual_immediate", True, "Pat", None, now=_t(9))
        return effective_flags(log)
    monkeypatch.setattr(match, "effective_reviews", late_review)
    monkeypatch.setattr(classify, "effective_flags", late_flag)
    ref = data / "reference"
    assert match.main(["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
                       "--weld-map", str(folder / "weld_map.csv"), "--pipe", str(ref / "pipe_properties.csv"),
                       "--repairs", str(ref / "repairs_between_runs.csv"), "--out", str(out)]) == 0
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    capsys.readouterr()
    msum = json.loads((folder / "match_summary.json").read_text(encoding="utf-8"))
    csum = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))
    assert msum["reviews_sha256"] == parsed[REVIEWS_FILE]
    assert csum["flags_sha256"] == parsed[FLAGS_FILE]
