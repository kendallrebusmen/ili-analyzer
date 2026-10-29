"""M12 parity with an outside analysis: read a consultant's Excel workbook into one table per anomaly.

The consultant's layout is described by a YAML map in ``config/consultant_maps/`` (sheet, header row,
column headers, how their growth-rate source labels name this program's growth methods). The reader
returns the anomalies in the program's vocabulary; it refuses (``ValueError``) a missing sheet or column
and duplicate feature ids. Feature ids are text: an integral Excel number (``143``, ``144.0``) becomes
``"143"``, ``"144"``. Later tasks compare these rows with the program's own results.

DRAFT – for review by qualified personnel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import field_validator

from ili.config import _Strict
from ili.growth import METHODS

CONSULTANT_COLUMNS = ("feature_id", "log_dist_ft", "joint_id", "depth_pct", "prev_depth_pct", "rate_mpy",
                      "method", "rate_source", "erf", "rl_yr", "vendor_type")
_ANOMALY_FIELDS = tuple(c for c in CONSULTANT_COLUMNS if c != "method")
_IMP_FIELDS = ("imp_class", "n_classified", "n_previously_repaired", "n_unrepaired")


class AnomalySheet(_Strict):
    sheet: str
    header_row: int = 1  # 1-based
    category_column: str
    anomaly_category: str
    columns: dict[str, str]
    rate_sources: dict[str, str]
    box_types: list[str] = []

    @field_validator("columns")
    @classmethod
    def _fields(cls, columns: dict[str, str]) -> dict[str, str]:
        missing = sorted(set(_ANOMALY_FIELDS) - set(columns))
        unknown = sorted(set(columns) - set(_ANOMALY_FIELDS))
        if missing or unknown:
            raise ValueError(f"columns: missing {missing}, unknown {unknown}")
        return columns

    @field_validator("rate_sources")
    @classmethod
    def _methods(cls, sources: dict[str, str]) -> dict[str, str]:
        bad = sorted({m for m in sources.values() if m not in METHODS})
        if bad:
            raise ValueError(f"rate_sources: not growth methods {bad}; valid: {', '.join(METHODS)}")
        return sources


class ImpSheet(_Strict):
    sheet: str
    header_row: int = 1
    columns: dict[str, str]
    classes: dict[str, str]

    @field_validator("columns")
    @classmethod
    def _fields(cls, columns: dict[str, str]) -> dict[str, str]:
        missing = sorted(set(_IMP_FIELDS) - set(columns))
        unknown = sorted(set(columns) - set(_IMP_FIELDS))
        if missing or unknown:
            raise ValueError(f"columns: missing {missing}, unknown {unknown}")
        return columns


class ConsultantMap(_Strict):
    consultant: str
    anomalies: AnomalySheet
    imp_summary: ImpSheet | None = None


def load_consultant_map(path: Path) -> ConsultantMap:
    with open(path, encoding="utf-8") as fh:
        return ConsultantMap.model_validate(yaml.safe_load(fh) or {})


@dataclass(frozen=True)
class ConsultantData:
    anomalies: pd.DataFrame
    imp_summary: pd.DataFrame | None
    unknown_sources: dict[str, int]


def _read_sheet(xl: pd.ExcelFile, path: Path, sheet: str, header_row: int, columns: list[str]) -> pd.DataFrame:
    if sheet not in xl.sheet_names:
        raise ValueError(f"{path.name}: no sheet {sheet!r} (sheets: {', '.join(map(str, xl.sheet_names))})")
    df = pd.read_excel(xl, sheet_name=sheet, header=header_row - 1, keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip() for c in df.columns]
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ValueError(f"{path.name}: sheet {sheet!r} has no column {', '.join(map(repr, absent))}")
    return df


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def _text_id(value) -> str | None:
    if _blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def read_consultant(path: Path, cmap: ConsultantMap) -> ConsultantData:
    path = Path(path)
    a = cmap.anomalies
    with pd.ExcelFile(path) as xl:
        raw = _read_sheet(xl, path, a.sheet, a.header_row, [a.category_column, *a.columns.values()])
        raw = raw[raw[a.category_column].astype(str).str.strip() == a.anomaly_category]
        out = pd.DataFrame({f: raw[a.columns[f]].to_numpy() for f in _ANOMALY_FIELDS}, dtype=object)
        out["feature_id"] = out["feature_id"].map(_text_id)
        dups = sorted(out.loc[out.feature_id.duplicated(keep=False), "feature_id"].dropna().unique())
        if dups:
            raise ValueError(f"{path.name}: duplicate feature ids: {', '.join(dups)}")

        labels = out["rate_source"].map(lambda v: None if _blank(v) else str(v).strip())
        out["rate_source"] = labels
        out["method"] = pd.Series([None if s is None else a.rate_sources.get(s) for s in labels], index=out.index, dtype=object)
        unknown: dict[str, int] = {}
        for s in labels[labels.notna() & out["method"].isna()]:
            unknown[s] = unknown.get(s, 0) + 1
        anomalies = out[list(CONSULTANT_COLUMNS)].reset_index(drop=True)

        imp = None
        if cmap.imp_summary is not None:
            m = cmap.imp_summary
            sheet = _read_sheet(xl, path, m.sheet, m.header_row, list(m.columns.values()))
            sheet = pd.DataFrame({f: sheet[m.columns[f]].to_numpy() for f in _IMP_FIELDS}, dtype=object)
            sheet["imp_class"] = sheet["imp_class"].map(lambda v: str(v).strip())
            order = {label: i for i, label in enumerate(m.classes)}
            sheet = sheet[sheet.imp_class.isin(order)].copy()
            sheet = sheet.sort_values("imp_class", key=lambda s: s.map(order), kind="stable")
            sheet["imp_class"] = sheet.imp_class.map(m.classes)
            imp = sheet.reset_index(drop=True)
    return ConsultantData(anomalies=anomalies, imp_summary=imp, unknown_sources=unknown)
