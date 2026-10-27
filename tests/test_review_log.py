"""Decision and flag logs (spec 6.4, 6.11): append-only, validated, atomic."""

import os
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ili import review
from ili.review import (FLAG_COLUMNS, REVIEW_COLUMNS, append_flag, append_review, effective_flags,
                        effective_reviews, load_flags, load_reviews)

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def _t(minutes: int) -> datetime:
    return T0.replace(minute=minutes)


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "reviews.csv"
    r1 = append_review(path, "manual", ["A2", "A1"], ["B1"], "Pat", note="pair, with comma", now=_t(1))
    r2 = append_review(path, "reject", ["A9"], ["B9", "B8"], " Lee ", now=_t(2))
    assert (r1["decision_id"], r2["decision_id"]) == ("R000001", "R000002")
    df = load_reviews(path)
    assert list(df.columns) == REVIEW_COLUMNS
    assert df.prev_feature_ids.tolist() == ["A1;A2", "A9"]
    assert df.curr_feature_ids.tolist() == ["B1", "B8;B9"]
    assert df.reviewer.tolist() == ["Pat", "Lee"]
    assert df.decided_utc.tolist() == ["2026-10-27T12:01:00Z", "2026-10-27T12:02:00Z"]
    assert df.note.tolist() == ["pair, with comma", ""]
    assert b"\r" not in path.read_bytes()

    fpath = tmp_path / "feature_flags.csv"
    f1 = append_flag(fpath, "B1", "manual_immediate", True, "Pat", now=_t(3))
    f2 = append_flag(fpath, "B1", "manual_immediate", False, "Pat", now=_t(4))
    assert (f1["flag_id"], f2["flag_id"]) == ("F000001", "F000002")
    fl = load_flags(fpath)
    assert list(fl.columns) == FLAG_COLUMNS
    assert fl.value.tolist() == ["true", "false"]
    assert b"\r" not in fpath.read_bytes()


def test_one_sided_accept_is_valid(tmp_path):
    path = tmp_path / "reviews.csv"
    assert append_review(path, "accept", ["A1"], [], "Pat", now=T0)["curr_feature_ids"] == ""
    assert append_review(path, "accept", [], ["B1"], "Pat", now=T0)["prev_feature_ids"] == ""
    assert list(load_reviews(path).decision) == ["accept", "accept"]


def test_missing_log_is_empty(tmp_path):
    assert load_reviews(tmp_path / "none.csv").empty
    assert list(load_reviews(tmp_path / "none.csv").columns) == REVIEW_COLUMNS
    assert list(load_flags(tmp_path / "none.csv").columns) == FLAG_COLUMNS
    assert effective_reviews(load_reviews(tmp_path / "none.csv")).superseded == []
    assert effective_flags(load_flags(tmp_path / "none.csv")) == {}


def test_invalid_rows_are_errors(tmp_path):
    path = tmp_path / "reviews.csv"
    header = ",".join(REVIEW_COLUMNS)
    good = "R000001,accept,A1,B1,Pat,2026-10-27T12:00:00Z,"
    bad_rows = {
        "reviewer": "R000002,accept,A1,B1, ,2026-10-27T12:00:00Z,",
        "decision": "R000002,approve,A1,B1,Pat,2026-10-27T12:00:00Z,",
        "reject": "R000002,reject,A1,,Pat,2026-10-27T12:00:00Z,",
        "decided_utc": "R000002,accept,A1,B1,Pat,2026-10-27T12:00:00,",
    }
    for key, row in bad_rows.items():
        path.write_text(f"{header}\n{good}\n{row}\n", encoding="utf-8", newline="")
        with pytest.raises(ValueError, match=r"reviews\.csv.*line 3") as exc:
            load_reviews(path)
        assert key in str(exc.value)

    fpath = tmp_path / "feature_flags.csv"
    fpath.write_text(",".join(FLAG_COLUMNS) + "\nF000001,B1,bogus,true,Pat,2026-10-27T12:00:00Z,\n",
                     encoding="utf-8", newline="")
    with pytest.raises(ValueError, match=r"feature_flags\.csv.*line 2"):
        load_flags(fpath)

    # the append entry points refuse the same input and write nothing
    with pytest.raises(ValueError, match="reviewer"):
        append_review(tmp_path / "x.csv", "accept", ["A1"], ["B1"], "  ", now=T0)
    with pytest.raises(ValueError, match="reject"):
        append_review(tmp_path / "x.csv", "reject", ["A1"], [], "Pat", now=T0)
    with pytest.raises(ValueError, match="manual"):
        append_review(tmp_path / "x.csv", "manual", [], ["B1"], "Pat", now=T0)
    with pytest.raises(ValueError, match="at least one feature id"):
        append_review(tmp_path / "x.csv", "accept", [], [], "Pat", now=T0)
    with pytest.raises(ValueError, match="timezone"):
        append_review(tmp_path / "x.csv", "accept", ["A1"], ["B1"], "Pat", now=datetime(2026, 10, 27))
    with pytest.raises(ValueError, match="flag"):
        append_flag(tmp_path / "y.csv", "B1", "nope", True, "Pat", now=T0)
    assert not (tmp_path / "x.csv").exists() and not (tmp_path / "y.csv").exists()


def test_append_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "reviews.csv"
    append_review(path, "accept", ["A1"], ["B1"], "Pat", now=_t(1))
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        append_review(path, "accept", ["A2"], ["B2"], "Pat", now=_t(2))
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["reviews.csv"]


def test_latest_decision_wins_and_supersedes(tmp_path):
    path = tmp_path / "reviews.csv"
    append_review(path, "accept", ["A1"], ["B1"], "Pat", now=_t(1))
    append_review(path, "manual", ["A1"], ["B2"], "Pat", now=_t(2))
    append_review(path, "reject", ["A3"], ["B9"], "Pat", now=_t(3))
    append_review(path, "manual", ["A3", "A4"], ["B3"], "Pat", now=_t(4))
    append_review(path, "accept", ["A7"], ["B7"], "Pat", now=_t(0))  # earlier time, later in file
    eff = effective_reviews(load_reviews(path))
    assert eff.superseded == ["R000001", "R000003"]
    assert eff.decisions.decision_id.tolist() == ["R000005", "R000002", "R000004"]
    assert eff.decisions.curr_feature_ids.tolist() == ["B7", "B2", "B3"]
    assert eff.decisions.prev_feature_ids.tolist() == ["A7", "A1", "A3;A4"]


def test_effective_flags_latest_value(tmp_path):
    path = tmp_path / "feature_flags.csv"
    append_flag(path, "B1", "manual_immediate", True, "Pat", now=_t(1))
    append_flag(path, "B1", "field_crack_confirmed", True, "Pat", now=_t(2))
    append_flag(path, "B1", "manual_immediate", False, "Lee", now=_t(3))
    append_flag(path, "B2", "manual_immediate", True, "Pat", now=_t(0))
    assert effective_flags(load_flags(path)) == {
        ("B1", "manual_immediate"): False, ("B1", "field_crack_confirmed"): True,
        ("B2", "manual_immediate"): True}
    assert review.REVIEWS_FILE == "reviews.csv" and review.FLAGS_FILE == "feature_flags.csv"


def test_curr_only_overlap_supersedes(tmp_path):
    path = tmp_path / "reviews.csv"
    append_review(path, "accept", ["A1"], ["B1"], "Pat", now=_t(1))
    append_review(path, "manual", ["A2"], ["B1"], "Pat", now=_t(2))
    eff = effective_reviews(load_reviews(path))
    assert eff.superseded == ["R000001"]
    assert eff.decisions.decision_id.tolist() == ["R000002"]


def test_concurrent_appends_lose_nothing(tmp_path):
    path = tmp_path / "reviews.csv"
    n = 8
    threads = [threading.Thread(target=append_review, args=(path, "accept", [f"A{i}"], [f"B{i}"], "Pat"),
                                kwargs={"now": _t(i)}) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    df = load_reviews(path)
    assert df.decision_id.tolist() == [f"R{i:06d}" for i in range(1, n + 1)]
    assert sorted(df.prev_feature_ids) == sorted(f"A{i}" for i in range(n))
    assert [p.name for p in tmp_path.iterdir()] == ["reviews.csv"]


def test_times_are_stored_as_utc(tmp_path):
    path = tmp_path / "reviews.csv"
    plus5 = datetime(2026, 10, 27, 17, 0, tzinfo=timezone(timedelta(hours=5)))
    assert append_review(path, "accept", ["A1"], ["B1"], "Pat", now=plus5)["decided_utc"] == "2026-10-27T12:00:00Z"
    text = path.read_text(encoding="utf-8").replace("2026-10-27T12:00:00Z", "2026-10-27T17:00:00+05:00")
    path.write_text(text, encoding="utf-8", newline="")
    assert load_reviews(path).decided_utc.tolist() == ["2026-10-27T12:00:00Z"]


def test_malformed_ids_and_field_counts_are_errors(tmp_path):
    path = tmp_path / "reviews.csv"
    header = ",".join(REVIEW_COLUMNS)
    ok = "accept,A1,B1,Pat,2026-10-27T12:00:00Z,"
    cases = {
        "bad id": f"X1,{ok}",
        "repeat": f"R000001,{ok}",
        "extra": f"R000002,{ok},oops",
        "short": "R000002,accept,A1,B1",
    }
    for key, row in cases.items():
        path.write_text(f"{header}\nR000001,{ok}\n{row}\n", encoding="utf-8", newline="")
        with pytest.raises(ValueError, match=r"reviews\.csv: line 3.*by hand"):
            load_reviews(path)


def test_next_id_follows_the_highest_id(tmp_path):
    path = tmp_path / "reviews.csv"
    for i in range(3):
        append_review(path, "accept", [f"A{i}"], [f"B{i}"], "Pat", now=_t(i))
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:2] + lines[3:]), encoding="utf-8", newline="")  # R000002 removed by hand
    assert append_review(path, "accept", ["A9"], ["B9"], "Pat", now=_t(9))["decision_id"] == "R000004"
    assert load_reviews(path).decision_id.tolist() == ["R000001", "R000003", "R000004"]


def test_failed_write_keeps_the_original_error_and_leaves_no_temp(tmp_path, monkeypatch):
    path = tmp_path / "reviews.csv"
    real = review.tempfile.NamedTemporaryFile

    class FailingWrite:
        def __init__(self, *args, **kwargs):
            self._f = real(*args, **kwargs)
            self.name = self._f.name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()

        def write(self, data):
            raise OSError("disk full")

    monkeypatch.setattr(review.tempfile, "NamedTemporaryFile", FailingWrite)
    with pytest.raises(OSError, match="disk full"):
        append_review(path, "accept", ["A1"], ["B1"], "Pat", now=T0)
    assert list(tmp_path.iterdir()) == []


def test_excel_bom_is_read_and_header_error_has_the_recovery_hint(tmp_path):
    # an Excel "CSV UTF-8" save starts the file with a byte-order mark and ends lines with CRLF
    path = tmp_path / "reviews.csv"
    text = ",".join(REVIEW_COLUMNS) + "\r\nR000001,accept,A1,B1,Pat,2026-10-27T12:00:00Z,\r\n"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    got = load_reviews(path)
    assert list(got.decision_id) == ["R000001"] and got.prev_feature_ids.iloc[0] == "A1"
    append_review(path, "reject", ["A1"], ["B1"], "Pat", now=_t(5))
    assert list(load_reviews(path).decision_id) == ["R000001", "R000002"]
    path.write_text("decision,decision_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"reviews\.csv: line 1: .*fix or remove the row by hand"):
        load_reviews(path)


def test_hashed_load_hashes_the_bytes_it_parsed(tmp_path, monkeypatch):
    import hashlib
    path = tmp_path / "reviews.csv"
    append_review(path, "accept", ["A1"], ["B1"], "Pat", now=_t(1))
    parsed = path.read_bytes()
    reads = []
    real = type(path).read_bytes

    def counting(self):
        reads.append(self)
        return real(self)
    monkeypatch.setattr(type(path), "read_bytes", counting)
    log, digest = review.load_reviews_hashed(path)
    assert len(reads) == 1 and digest == hashlib.sha256(parsed).hexdigest() and len(log) == 1
    assert review.load_reviews_hashed(tmp_path / "none.csv")[1] is None
    flags, fdigest = review.load_flags_hashed(tmp_path / "none.csv")
    assert flags.empty and fdigest is None
