"""Reference tables typed in on the Run setup page instead of chosen as files (spec 5.2 layouts).

Each table the app lets a person type in has a ``TableSpec``: its file name, its columns with their kind, and which
columns every row needs. ``seed_table`` gives the starting rows for the editor (the chosen file's rows when there is
one, else none); ``clean_table`` drops blank rows and refuses a row missing a required value, naming the table and the
row; ``write_table`` saves the rows as the CSV the pipeline reads, in ``<output folder>/entered/``, so a run on typed
data is as traceable as one on files (the run manifest hashes the saved file).

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ili.reference import REPAIR_TYPES

ENTERED_DIR = "entered"  # under the output folder: where typed tables are saved

NUMBER, TEXT, FLAG, DATE, YEAR, REPAIR_TYPE = "number", "text", "flag", "date", "year", "repair_type"


@dataclass(frozen=True)
class TableSpec:
    label: str  # as the page names it
    file_name: str  # the spec 5.2 file name it is saved as
    columns: dict[str, str]  # column -> kind (NUMBER, TEXT, FLAG, DATE, YEAR, REPAIR_TYPE)
    required: tuple[str, ...]  # columns every row needs
    help: str  # one line under the table


TABLES: dict[str, TableSpec] = {
    "pipe": TableSpec(
        "Pipe properties", "pipe_properties.csv",
        {"start_ft": NUMBER, "end_ft": NUMBER, "od_in": NUMBER, "wt_nom_in": NUMBER, "grade": TEXT,
         "smys_psi": NUMBER, "seam_type": TEXT, "install_year": YEAR, "coating_type": TEXT},
        ("start_ft", "end_ft", "od_in", "wt_nom_in", "smys_psi", "install_year"),
        "One row per stretch of pipe, by current-run log distance (ft). OD and wall in inches, SMYS in psi."),
    "mop": TableSpec(
        "MOP profile", "mop_profile.csv", {"start_ft": NUMBER, "end_ft": NUMBER, "mop_psig": NUMBER},
        ("start_ft", "end_ft", "mop_psig"), "Maximum operating pressure (psig) by current-run log distance (ft)."),
    "segments": TableSpec(
        "HCA segments", "hca_segments.csv",
        {"start_ft": NUMBER, "end_ft": NUMBER, "segment_id": TEXT, "could_affect_hca": FLAG},
        ("start_ft", "end_ft", "segment_id"),
        "Segments by current-run log distance (ft); tick the ones that could affect a high-consequence area."),
    "next_inspection": TableSpec(
        "Next inspection dates", "next_inspection.csv",
        {"segment_id": TEXT, "planned_next_assessment_date": DATE}, ("segment_id",),
        "The planned next assessment of each segment. Leave a date blank to use the default interval."),
    "repairs": TableSpec(
        "Repairs between runs", "repairs_between_runs.csv",
        {"start_ft": NUMBER, "end_ft": NUMBER, "repair_type": REPAIR_TYPE, "repair_date": DATE, "notes": TEXT},
        ("start_ft", "end_ft", "repair_type"),
        "Repairs made between the two inspections, by current-run log distance (ft)."),
    "crossings": TableSpec(
        "Foreign crossings", "foreign_crossings.csv", {"log_dist_ft": NUMBER, "description": TEXT},
        ("log_dist_ft",), "Foreign pipeline crossings by current-run log distance (ft)."),
}

COLUMN_LABELS = {  # editor headings; the saved CSV keeps the spec 5.2 column names
    "start_ft": "Start (ft)", "end_ft": "End (ft)", "od_in": "OD (in)", "wt_nom_in": "Wall (in)", "grade": "Grade",
    "smys_psi": "SMYS (psi)", "seam_type": "Seam", "install_year": "Install year", "coating_type": "Coating",
    "mop_psig": "MOP (psig)", "segment_id": "Segment", "could_affect_hca": "Could affect HCA",
    "planned_next_assessment_date": "Next assessment", "repair_type": "Repair type", "repair_date": "Repair date",
    "notes": "Notes", "log_dist_ft": "Log distance (ft)", "description": "Description",
}


def _typed(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """``df`` with exactly the spec's columns, in order, each converted to its kind (blanks stay blank)."""
    out = pd.DataFrame(index=df.index)
    for column, kind in spec.columns.items():
        values = df[column] if column in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        if kind in (NUMBER, YEAR):
            out[column] = pd.to_numeric(values, errors="coerce")
        elif kind == FLAG:
            out[column] = values.map(_flag).astype(bool)
        elif kind == DATE:
            out[column] = values.map(_date)
        else:
            out[column] = values.map(lambda v: None if _blank(v) else str(v).strip()).astype(object)
    return out


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        not isinstance(value, str) and pd.isna(value))


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return False if _blank(value) else bool(value)


def _date(value) -> date | None:
    if _blank(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def blank_table(name: str) -> pd.DataFrame:
    """An empty table with the spec's columns and kinds (the editor's starting point without a file)."""
    spec = TABLES[name]
    return _typed(pd.DataFrame(columns=list(spec.columns)), spec)


def seed_table(name: str, path_text: str) -> pd.DataFrame:
    """The editor's starting rows: the file at ``path_text`` when it can be read (so typing edits it), else an
    empty table."""
    path = Path(path_text.strip()) if path_text and path_text.strip() else None
    if path is None or not path.is_file():
        return blank_table(name)
    try:
        return _typed(pd.read_csv(path, dtype=str, keep_default_na=False), TABLES[name])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
        return blank_table(name)


def clean_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The typed rows ready to save: fully blank rows dropped; a ValueError naming the table, the row number (as
    shown, from 1) and the columns for a row missing a required value or holding a repair type outside spec 5.2."""
    spec = TABLES[name]
    typed = _typed(df.reset_index(drop=True), spec)
    content = [c for c, kind in spec.columns.items() if kind != FLAG]
    keep = ~typed[content].map(_blank).all(axis=1)
    problems = []
    for i in typed.index[keep]:
        missing = [c for c in spec.required if _blank(typed.at[i, c])]
        if missing:
            problems.append(f"row {i + 1} needs {', '.join(missing)}")
        if "repair_type" in spec.columns and not _blank(typed.at[i, "repair_type"]) \
                and typed.at[i, "repair_type"] not in REPAIR_TYPES:
            problems.append(f"row {i + 1} repair_type must be one of {', '.join(REPAIR_TYPES)}")
    if problems:
        raise ValueError(f"{spec.label}: " + "; ".join(problems))
    return typed[keep].reset_index(drop=True)


def write_table(name: str, df: pd.DataFrame, out_dir: Path) -> Path:
    """Save the cleaned rows as ``<out_dir>/entered/<file name>`` (LF, UTF-8) and return the path; a ValueError
    when there are no rows (an empty reference table cannot be run)."""
    spec = TABLES[name]
    rows = clean_table(name, df)
    if rows.empty:
        raise ValueError(f"{spec.label}: type at least one row, or choose a file")
    path = Path(out_dir) / ENTERED_DIR / spec.file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    for column, kind in spec.columns.items():
        if kind == YEAR:
            rows[column] = rows[column].astype("Int64")
        elif kind == DATE:
            rows[column] = rows[column].map(lambda d: "" if d is None else d.isoformat())
    rows.to_csv(path, index=False, lineterminator="\n")
    return path
