"""Review logs (spec 6.4, 6.11): the append-only decision log and manual-flag log.

Both logs are CSV files beside the run outputs. A row is never edited or deleted: a correction is a new
row, and the latest row by ``decided_utc`` wins. Rows are keyed by feature ids, never by ``match_id``
(match ids renumber on every rerun); a set of ids is sorted and ``;``-joined, like the match table's
cluster ids. A log is written atomically (temporary file, then ``os.replace``) as UTF-8 with LF line
endings, and a malformed row is a ``ValueError`` naming the file and line.

``effective_reviews`` and ``effective_flags`` reduce a log to what currently applies.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REVIEWS_FILE = "reviews.csv"
FLAGS_FILE = "feature_flags.csv"
DECISIONS = ("accept", "reject", "manual")
FLAG_NAMES = ("manual_immediate", "field_crack_confirmed")
REVIEW_COLUMNS = ["decision_id", "decision", "prev_feature_ids", "curr_feature_ids", "reviewer", "decided_utc",
                  "note"]
FLAG_COLUMNS = ["flag_id", "curr_feature_id", "flag", "value", "reviewer", "decided_utc", "note"]

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class EffectiveReviews:
    decisions: pd.DataFrame  # the decisions in force, in the order they were made (REVIEW_COLUMNS)
    superseded: list[str]  # decision_ids replaced by a later decision on a shared feature, sorted


def _ids(text: str) -> list[str]:
    return sorted({i.strip() for i in text.split(";") if i.strip()})


def _parse_time(text: str) -> datetime:
    """An ISO 8601 time that carries a timezone, as UTC."""
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"decided_utc {text!r} is not an ISO 8601 time") from None
    if when.tzinfo is None:
        raise ValueError(f"decided_utc {text!r} has no timezone")
    return when.astimezone(timezone.utc)


def _check_review(row: dict[str, str]) -> dict[str, str]:
    """One decision row, validated and normalized (ids sorted and joined); ValueError otherwise."""
    if row["decision"] not in DECISIONS:
        raise ValueError(f"decision {row['decision']!r} is not one of {', '.join(DECISIONS)}")
    prev, curr = _ids(row["prev_feature_ids"]), _ids(row["curr_feature_ids"])
    if not prev or not curr:
        raise ValueError(f"a {row['decision']} decision needs at least one feature id on each side")
    if not row["reviewer"].strip():
        raise ValueError("reviewer is blank")
    _parse_time(row["decided_utc"])
    return {**row, "prev_feature_ids": ";".join(prev), "curr_feature_ids": ";".join(curr),
            "reviewer": row["reviewer"].strip(), "decided_utc": row["decided_utc"].strip()}


def _check_flag(row: dict[str, str]) -> dict[str, str]:
    if not row["curr_feature_id"].strip():
        raise ValueError("curr_feature_id is blank")
    if row["flag"] not in FLAG_NAMES:
        raise ValueError(f"flag {row['flag']!r} is not one of {', '.join(FLAG_NAMES)}")
    value = row["value"].strip().casefold()
    if value not in ("true", "false"):
        raise ValueError(f"value {row['value']!r} must be true or false")
    if not row["reviewer"].strip():
        raise ValueError("reviewer is blank")
    _parse_time(row["decided_utc"])
    return {**row, "curr_feature_id": row["curr_feature_id"].strip(), "value": value,
            "reviewer": row["reviewer"].strip(), "decided_utc": row["decided_utc"].strip()}


def _read(path: Path, columns: list[str], check) -> list[dict[str, str]]:
    """Every row of a log, validated; a missing file is no rows."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: line 1: columns are {reader.fieldnames}, expected {columns}")
        for raw in reader:
            row = {c: (raw[c] or "") for c in columns}
            try:
                rows.append(check(row))
            except ValueError as e:
                raise ValueError(f"{path.name}: line {reader.line_num}: {e}") from None
    return rows


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Replace the log with ``rows`` through a temporary file beside it; the old file survives a failure."""
    path = Path(path)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(buf.getvalue().encode("utf-8"))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stamp(now: datetime | None) -> str:
    when = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("now has no timezone")
    return when.astimezone(timezone.utc).strftime(_TIME_FORMAT)


def _clean_note(note: str | None) -> str:
    """One line, so a note never moves the line numbers in an error message."""
    return " ".join((note or "").split())


def load_reviews(path: Path) -> pd.DataFrame:
    """The decision log (REVIEW_COLUMNS, all text); a missing file is an empty frame."""
    return pd.DataFrame(_read(path, REVIEW_COLUMNS, _check_review), columns=REVIEW_COLUMNS, dtype=object)


def load_flags(path: Path) -> pd.DataFrame:
    """The manual-flag log (FLAG_COLUMNS, all text; ``value`` is ``true``/``false``)."""
    return pd.DataFrame(_read(path, FLAG_COLUMNS, _check_flag), columns=FLAG_COLUMNS, dtype=object)


def append_review(path: Path, decision: str, prev_ids: list[str], curr_ids: list[str], reviewer: str,
                  note: str | None = None, now: datetime | None = None) -> dict[str, str]:
    """Add one decision to the log and return the row written."""
    rows = _read(path, REVIEW_COLUMNS, _check_review)
    row = _check_review({
        "decision_id": f"R{len(rows) + 1:06d}", "decision": decision,
        "prev_feature_ids": ";".join(prev_ids), "curr_feature_ids": ";".join(curr_ids),
        "reviewer": reviewer, "decided_utc": _stamp(now), "note": _clean_note(note)})
    _write(path, REVIEW_COLUMNS, rows + [row])
    return row


def append_flag(path: Path, curr_feature_id: str, flag: str, value: bool, reviewer: str,
                note: str | None = None, now: datetime | None = None) -> dict[str, str]:
    """Add one manual flag value to the log and return the row written."""
    rows = _read(path, FLAG_COLUMNS, _check_flag)
    row = _check_flag({
        "flag_id": f"F{len(rows) + 1:06d}", "curr_feature_id": curr_feature_id, "flag": flag,
        "value": "true" if value else "false", "reviewer": reviewer, "decided_utc": _stamp(now),
        "note": _clean_note(note)})
    _write(path, FLAG_COLUMNS, rows + [row])
    return row


def _in_time_order(df: pd.DataFrame, id_col: str) -> list[dict]:
    """Rows by ``decided_utc`` (as a time), ties by id."""
    rows = df.to_dict("records")
    return sorted(rows, key=lambda r: (_parse_time(r["decided_utc"]), r[id_col]))


def effective_reviews(reviews: pd.DataFrame) -> EffectiveReviews:
    """The decisions in force. Rows are taken in ``decided_utc`` order (ties by ``decision_id``); a row
    that shares any feature id on the same side with an effective earlier row supersedes it."""
    effective: list[tuple[dict, set, set]] = []
    superseded: list[str] = []
    for row in _in_time_order(reviews, "decision_id"):
        prev, curr = set(_ids(row["prev_feature_ids"])), set(_ids(row["curr_feature_ids"]))
        keep = []
        for old in effective:
            if old[1] & prev or old[2] & curr:
                superseded.append(old[0]["decision_id"])
            else:
                keep.append(old)
        effective = keep + [(row, prev, curr)]
    frame = pd.DataFrame([e[0] for e in effective], columns=REVIEW_COLUMNS, dtype=object)
    return EffectiveReviews(frame, sorted(superseded))


def effective_flags(flags: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """The latest value per (feature, flag) by ``decided_utc``, ties by ``flag_id``."""
    out: dict[tuple[str, str], bool] = {}
    for row in _in_time_order(flags, "flag_id"):
        out[(row["curr_feature_id"], row["flag"])] = row["value"] == "true"
    return out
