"""Load vendor ILI tallies into the standard schema (spec sections 5.1 and 6.2).

Each vendor layout is described by a YAML vendor map in ``config/vendor_maps/``:
column names, units, clock format, and the vendor's vocabulary. Ingestion
converts units, parses clock positions, maps vocabulary, and validates. Rows
that fail go to ``validation_issues.csv`` with a reason; none are dropped
silently.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import openpyxl
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from ili.schema import STANDARD_COLUMNS, FeatureType, Surface
from ili.units import CLOCK_MINUTES, FT_PER_M, IN_PER_MM, hhmm_to_clock_min

MAPPABLE_FIELDS = [c for c in STANDARD_COLUMNS if c not in ("run_id", "run_date")]
ISSUE_COLUMNS = ["run_id", "source_row", "feature_id", "field", "value", "reason"]

# Standard field -> which vendor-map unit governs it.
UNIT_GROUP = {
    "log_dist_ft": "distance",
    "dist_from_us_gw_ft": "distance",
    "joint_length_ft": "distance",
    "wt_nom_in": "wall",
    "dent_depth_in": "dent_depth",
    "length_in": "size",
    "width_in": "size",
    "elev_ft": "elevation",
}
TO_US = {"ft": 1.0, "in": 1.0, "m": FT_PER_M, "mm": IN_PER_MM}
PLAIN_NUMBERS = ["depth_pct", "lat", "lon"]
TEXT_FIELDS = ["feature_id", "us_gw_id", "vendor_cluster_id", "comments"]
FLAG_FIELDS = ["seam_interaction", "dent_metal_loss"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileSpec(_Strict):
    format: Literal["csv", "excel"]
    sheet: str | None = None
    header_row: int = 1  # 1-based


class RunDateSpec(_Strict):
    cell: str  # e.g. "A2"
    pattern: str  # regex with one group holding an ISO date


class Units(_Strict):
    distance: Literal["ft", "m"]
    wall: Literal["in", "mm"]
    size: Literal["in", "mm"]
    dent_depth: Literal["in", "mm"]
    elevation: Literal["ft", "m"]


def _norm(token: str) -> str:
    return str(token).strip().casefold()


def _normalized_vocab(vocab: dict) -> dict:
    out: dict = {}
    for key, value in vocab.items():
        k = _norm(key)
        if k in out:
            raise ValueError(f"vocabulary key {k!r} appears twice after normalization")
        out[k] = value
    return out


class VendorMap(_Strict):
    vendor: str
    run_label: str
    file: FileSpec
    run_date: RunDateSpec | None = None
    columns: dict[str, str]
    units: Units
    clock_format: Literal["hhmm", "degrees", "decimal_hours"]
    feature_types: dict[str, FeatureType]
    surfaces: dict[str, Surface]
    true_values: list[str]
    false_values: list[str]

    @field_validator("columns")
    @classmethod
    def _known_fields(cls, columns: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(columns) - set(MAPPABLE_FIELDS))
        if unknown:
            raise ValueError(f"not mappable standard fields: {', '.join(unknown)}")
        return columns

    @field_validator("feature_types", "surfaces", mode="before")
    @classmethod
    def _normalize_keys(cls, vocab: dict) -> dict:
        return _normalized_vocab(vocab)

    @field_validator("true_values", "false_values")
    @classmethod
    def _normalize_flags(cls, values: list[str]) -> list[str]:
        return [_norm(v) for v in values]


def load_vendor_map(path: Path) -> VendorMap:
    with open(path, encoding="utf-8") as fh:
        return VendorMap.model_validate(yaml.safe_load(fh) or {})


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _cell(path: Path, vmap: VendorMap, ref: str) -> str | None:
    col_letters, row = re.fullmatch(r"([A-Za-z]+)(\d+)", ref).groups()
    if vmap.file.format == "excel":
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb[vmap.file.sheet] if vmap.file.sheet else wb.worksheets[0]
            value = ws[ref].value
        finally:
            wb.close()
        return None if value is None else str(value)
    col = openpyxl.utils.column_index_from_string(col_letters) - 1
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for i, line in enumerate(csv.reader(fh), start=1):
            if i == int(row):
                return line[col] if col < len(line) else None
    return None


def read_tally(path: Path, vmap: VendorMap) -> tuple[pd.DataFrame, date | None]:
    """Raw tally as strings (blank = ""), with ``source_row`` = 1-based file row, and the run
    date if the vendor map says where to find it."""
    path = Path(path)
    skip = vmap.file.header_row - 1
    if vmap.file.format == "csv":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", skiprows=skip)
    else:
        raw = pd.read_excel(path, sheet_name=vmap.file.sheet or 0, header=skip, dtype=str,
                            keep_default_na=False, engine="openpyxl")
    raw = raw.fillna("").astype(object).apply(lambda s: s.str.strip())
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [v for v in vmap.columns.values() if v not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: columns required by vendor map {vmap.vendor!r} are missing: "
                         + ", ".join(repr(m) for m in missing))
    raw["source_row"] = np.arange(len(raw)) + vmap.file.header_row + 1

    run_date = None
    if vmap.run_date is not None:
        text = _cell(path, vmap, vmap.run_date.cell) or ""
        match = re.search(vmap.run_date.pattern, text)
        if not match:
            raise ValueError(f"{path.name}: run date not found in cell {vmap.run_date.cell} ({text!r})")
        run_date = date.fromisoformat(match.group(1))
    return raw, run_date


# ---------------------------------------------------------------------------
# Standardizing
# ---------------------------------------------------------------------------


def parse_clock(values: pd.Series, fmt: str) -> tuple[pd.Series, pd.Series]:
    """Clock positions to minutes from 12:00. Returns (Int64 minutes, unparseable mask).

    Exactly one full turn (360 deg, 12.0 h) maps to 0. Other out-of-range numbers are
    kept as-is so validation can report them rather than wrapping them silently.
    """
    values = values.astype(object).fillna("").astype(str).str.strip()
    blank = values == ""
    minutes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    bad = pd.Series(False, index=values.index)
    if fmt == "hhmm":
        for idx, text in values[~blank].items():
            try:
                minutes[idx] = hhmm_to_clock_min(text)
            except ValueError:
                bad[idx] = True
        return minutes, bad
    per_unit = {"degrees": 2.0, "decimal_hours": 60.0}[fmt]
    num = pd.to_numeric(values.where(~blank), errors="coerce")
    bad = ~blank & num.isna()
    m = np.rint(num * per_unit)
    m = m.where(m != CLOCK_MINUTES, 0)
    ok = m.notna()
    minutes[ok] = m[ok].astype(int)
    return minutes, bad


def _issues(run_id, rows: pd.DataFrame, mask, field: str, values, reason: str) -> pd.DataFrame:
    mask = np.asarray(mask, dtype=bool)
    return pd.DataFrame({
        "run_id": run_id,
        "source_row": rows.source_row.to_numpy()[mask],
        "feature_id": rows.feature_id.to_numpy()[mask],
        "field": field,
        "value": np.asarray(values, dtype=object)[mask],
        "reason": reason,
    }, columns=ISSUE_COLUMNS)


def _vocab(values: pd.Series, mapping: dict) -> tuple[list, np.ndarray]:
    keys = [str(v).strip().casefold() for v in values]
    mapped = [None if k == "" else (mapping[k].value if k in mapping else None) for k in keys]
    unknown = np.array([k != "" and k not in mapping for k in keys])
    return mapped, unknown


def standardize(raw: pd.DataFrame, vmap: VendorMap, run_id: str, run_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename, convert units, parse clocks and flags, and map vocabulary.

    Returns the standard frame (``STANDARD_COLUMNS`` + ``source_row``, one row per input
    row) and the issues found while parsing. Rows with parse issues are kept here, with the
    bad field blank; ``ingest_tally`` excludes them.
    """
    src = {std: raw[col] if col else pd.Series("", index=raw.index)
           for std, col in ((f, vmap.columns.get(f)) for f in MAPPABLE_FIELDS)}
    out = pd.DataFrame(index=raw.index)
    out["run_id"] = run_id
    out["run_date"] = run_date
    issues = []
    ident = pd.DataFrame({"source_row": raw.source_row,
                          "feature_id": src["feature_id"].replace("", None)})

    for f in TEXT_FIELDS:
        out[f] = src[f].astype(object).where(src[f] != "", None)

    out["feature_type"], unknown = _vocab(src["feature_type"], vmap.feature_types)
    issues.append(_issues(run_id, ident, unknown, "feature_type", src["feature_type"], "unknown_feature_type"))
    out["surface"], unknown = _vocab(src["surface"], vmap.surfaces)
    issues.append(_issues(run_id, ident, unknown, "surface", src["surface"], "unknown_surface"))

    for f in [*UNIT_GROUP, *PLAIN_NUMBERS]:
        text = src[f]
        num = pd.to_numeric(text.where(text != ""), errors="coerce")
        issues.append(_issues(run_id, ident, (text != "") & num.isna(), f, text, "unparseable_number"))
        if f in UNIT_GROUP:
            num = num * TO_US[getattr(vmap.units, UNIT_GROUP[f])]
        out[f] = num.astype(float)

    out["clock_min"], bad = parse_clock(src["clock_min"], vmap.clock_format)
    issues.append(_issues(run_id, ident, bad, "clock_min", src["clock_min"], "unparseable_clock"))

    for f in FLAG_FIELDS:
        keys = src[f].astype(str).str.strip().str.casefold()
        is_true, is_false = keys.isin(vmap.true_values), keys.isin(vmap.false_values)
        out[f] = [True if t else (False if fl else None) for t, fl in zip(is_true, is_false)]
        out[f] = out[f].astype(object)
        issues.append(_issues(run_id, ident, (keys != "") & ~is_true & ~is_false, f, src[f], "unparseable_flag"))

    out["source_row"] = raw.source_row.to_numpy()
    issues_df = pd.concat([i for i in issues if len(i)], ignore_index=True) if any(len(i) for i in issues) \
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    return out[STANDARD_COLUMNS + ["source_row"]].reset_index(drop=True), issues_df
