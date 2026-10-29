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


EXPLANATIONS = ("vendor_cluster_box", "not_in_our_run", "removed_after_run", "not_in_their_set", "agree",
                "depth_differs", "match_differs", "joint_pool_differs", "age_basis", "rl_horizon", "unexplained")
PARITY_COLUMNS = ("feature_id", "side", "match_id", "log_dist_ft", "depth_ours", "depth_theirs", "rate_ours",
                  "rate_theirs", "method_ours", "method_theirs", "history_ours", "erf_ours", "erf_theirs", "rl_ours",
                  "rl_theirs", "rl_governing_ours", "category_ours", "vendor_type_theirs", "agree_depth",
                  "agree_rate", "agree_erf", "agree_rl", "explanation")
OURS_COLUMNS = ("feature_id", "match_id", "log_dist_ft", "depth_ours", "rate_ours", "method_ours", "history_ours",
                "erf_ours", "rl_ours", "rl_governing_ours", "category_ours")
_THEIRS_MATCHED_METHODS = ("raw", "shrinking_fixed")  # their methods that use the previous run depth


def _lookup(source: pd.DataFrame, keys: pd.Series, columns: list[str]) -> pd.DataFrame:
    """``source[columns]`` by ``match_id``, aligned to ``keys`` (NaN where a match id has no row)."""
    by_id = source.drop_duplicates("match_id").set_index("match_id")[columns]
    return by_id.reindex(keys.to_numpy()).set_axis(keys.index)


def ours_by_feature(growth: pd.DataFrame, remaining_life: pd.DataFrame, classification: pd.DataFrame,
                    match_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """This program results, one row per current-run member feature id (columns ``OURS_COLUMNS``).

    Classification rows with no growth row (dents) are kept with a blank rate and depth from
    ``depth_pct_reported``; their position (classification has none) comes from ``match_table.curr_log_dist_ft``
    by ``match_id`` when the match table is given. A cluster id (``"B-1;B-2"``) is split: every member carries the cluster values."""
    g = growth.drop_duplicates("match_id")
    base = pd.DataFrame({"match_id": g.match_id.to_numpy(), "feature_ids": g.curr_feature_id.to_numpy(),
                         "log_dist_ft": g.curr_log_dist_ft.to_numpy(), "depth_ours": g.d_curr_pct.to_numpy(),
                         "rate_ours": g.governing_mpy.to_numpy(), "method_ours": g.governing_method.to_numpy(),
                         "history_ours": g.history_class.to_numpy()})
    cls = classification.drop_duplicates("match_id")
    extra = cls[~cls.match_id.isin(base.match_id)]
    extra = pd.DataFrame({"match_id": extra.match_id.to_numpy(), "feature_ids": extra.curr_feature_id.to_numpy(),
                          "depth_ours": extra.depth_pct_reported.to_numpy()})
    base = pd.concat([base, extra], ignore_index=True)
    keys = base.match_id
    if match_table is not None:
        pos = _lookup(match_table, keys, ["curr_log_dist_ft"]).curr_log_dist_ft
        base["log_dist_ft"] = base["log_dist_ft"].where(base["log_dist_ft"].notna(), pos) if "log_dist_ft" in base else pos
    base = base.join(_lookup(remaining_life, keys, ["erf", "rl_yr", "rl_governing"]).rename(
        columns={"erf": "erf_ours", "rl_yr": "rl_ours", "rl_governing": "rl_governing_ours"}))
    base = base.join(_lookup(cls, keys, ["governing_category"]).rename(columns={"governing_category": "category_ours"}))
    base["feature_id"] = base.feature_ids.map(lambda ids: [m.strip() for m in str(ids).split(";") if m.strip()])
    base = base.explode("feature_id").drop(columns="feature_ids")
    base = base[base.feature_id.notna()].reset_index(drop=True)
    for col in OURS_COLUMNS:
        if col not in base:
            base[col] = float("nan")
    return base[list(OURS_COLUMNS)]


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _agree(a: float, b: float, tol: float) -> bool | None:
    """``|a - b| <= tol``; None if either is blank; inf equals inf only."""
    if math.isnan(a) or math.isnan(b):
        return None
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol


def _text(value) -> str | None:
    return None if _blank(value) else str(value)


def compare(ours: pd.DataFrame, theirs: pd.DataFrame, settings, *, box_types=()) -> pd.DataFrame:
    """Join ``ours`` (``ours_by_feature``) with ``theirs`` (``ConsultantData.anomalies``) on the feature id and
    give every row an explanation code (``EXPLANATIONS``, first that applies). ``box_types`` are the vendor
    anomaly types that are vendor cluster boxes (``ConsultantMap.anomalies.box_types``). Columns ``PARITY_COLUMNS``,
    sorted by log distance then feature id."""
    cfg = settings.parity
    box = set(box_types)
    o = ours.drop_duplicates("feature_id").set_index("feature_id")
    t = theirs.drop_duplicates("feature_id").set_index("feature_id")
    rows = []
    for fid in list(dict.fromkeys([*t.index, *o.index])):
        mine = o.loc[fid] if fid in o.index else None
        theirrow = t.loc[fid] if fid in t.index else None
        side = "both" if mine is not None and theirrow is not None else "theirs_only" if mine is None else "ours_only"

        def pick(row, col):
            return None if row is None else row[col]

        r = {"feature_id": fid, "side": side, "match_id": _text(pick(mine, "match_id")),
             "depth_ours": _num(pick(mine, "depth_ours")), "depth_theirs": _num(pick(theirrow, "depth_pct")),
             "rate_ours": _num(pick(mine, "rate_ours")), "rate_theirs": _num(pick(theirrow, "rate_mpy")),
             "method_ours": _text(pick(mine, "method_ours")), "method_theirs": _text(pick(theirrow, "method")),
             "history_ours": _text(pick(mine, "history_ours")),
             "erf_ours": _num(pick(mine, "erf_ours")), "erf_theirs": _num(pick(theirrow, "erf")),
             "rl_ours": _num(pick(mine, "rl_ours")), "rl_theirs": _num(pick(theirrow, "rl_yr")),
             "rl_governing_ours": _text(pick(mine, "rl_governing_ours")),
             "category_ours": _text(pick(mine, "category_ours")),
             "vendor_type_theirs": _text(pick(theirrow, "vendor_type"))}
        dist = _num(pick(theirrow, "log_dist_ft"))
        r["log_dist_ft"] = dist if not math.isnan(dist) else _num(pick(mine, "log_dist_ft"))
        r["agree_depth"] = _agree(r["depth_ours"], r["depth_theirs"], cfg.depth_tol_pct)
        r["agree_rate"] = _agree(r["rate_ours"], r["rate_theirs"], cfg.rate_tol_mpy)
        r["agree_erf"] = _agree(r["erf_ours"], r["erf_theirs"], cfg.erf_tol)
        rl_theirs = r["rl_theirs"]
        rl_tol = cfg.rl_tol_yr if math.isnan(rl_theirs) or math.isinf(rl_theirs) \
            else max(cfg.rl_tol_yr, cfg.rl_tol_frac * abs(rl_theirs))
        r["agree_rl"] = _agree(r["rl_ours"], rl_theirs, rl_tol)
        r["explanation"] = _explain(r, box)
        rows.append(r)
    out = pd.DataFrame(rows, columns=list(PARITY_COLUMNS)).astype(object)
    out = out.assign(_d=out.log_dist_ft.astype(float)).sort_values(["_d", "feature_id"], na_position="last", kind="stable")
    return out.drop(columns="_d").reset_index(drop=True)


def _explain(r: dict, box: set) -> str:
    side = r["side"]
    if side == "theirs_only":
        return "vendor_cluster_box" if r["vendor_type_theirs"] in box else "not_in_our_run"
    if side == "ours_only":
        return "removed_after_run" if r["category_ours"] == "removed" else "not_in_their_set"
    # a None flag counts as agreement only when BOTH sides are blank for that quantity
    pairs = (("depth_ours", "depth_theirs", "agree_depth"), ("rate_ours", "rate_theirs", "agree_rate"),
             ("erf_ours", "erf_theirs", "agree_erf"), ("rl_ours", "rl_theirs", "agree_rl"))
    if all(r[f] is True or (r[f] is None and math.isnan(r[a]) and math.isnan(r[b])) for a, b, f in pairs):
        return "agree"
    if r["agree_depth"] is False:
        return "depth_differs"
    if (r["history_ours"] == "matched") != (r["method_theirs"] in _THEIRS_MATCHED_METHODS):
        return "match_differs"
    both = (r["method_ours"], r["method_theirs"])
    if r["agree_rate"] is False:
        if both == ("joint_percentile", "joint_percentile"):
            return "joint_pool_differs"
        if both == ("half_life", "half_life"):
            return "age_basis"
    elif r["agree_rate"] is True and r["agree_rl"] is False and r["rl_governing_ours"] == "beyond_horizon":
        return "rl_horizon"
    return "unexplained"
