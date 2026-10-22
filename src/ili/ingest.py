"""Load vendor ILI tallies into the standard schema (spec sections 5.1 and 6.2).

Each vendor layout is described by a YAML vendor map in ``config/vendor_maps/``:
column names, units, clock format, and the vendor's vocabulary. Ingestion
converts units, parses clock positions, maps vocabulary, and validates. Rows
that fail go to ``validation_issues.csv`` with a reason; none are dropped
silently.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from ili.schema import STANDARD_COLUMNS, FeatureType, Surface

MAPPABLE_FIELDS = [c for c in STANDARD_COLUMNS if c not in ("run_id", "run_date")]


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
