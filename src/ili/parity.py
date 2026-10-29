"""M12 parity with an outside analysis: read a consultant's Excel workbook into one table per anomaly.

The consultant's layout is described by a YAML map in ``config/consultant_maps/`` (sheet, header row,
column headers, how their growth-rate source labels name this program's growth methods). The reader
returns the anomalies in the program's vocabulary; it refuses (``ValueError``) a missing sheet or column
and duplicate feature ids. Feature ids are text: an integral Excel number (``143``, ``144.0``) becomes
``"143"``, ``"144"``. Later tasks compare these rows with the program's own results.

DRAFT – for review by qualified personnel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from pydantic import Field, field_validator

from ili import __version__
from ili.align_welds import DRAFT_LABEL
from ili.classify import load_classification_table
from ili.config import DEFAULT_SETTINGS, PROJECT_ROOT, Settings, _Strict, load_settings
from ili.growth import METHODS, load_growth_table
from ili.match import load_match_table
from ili.remaining_life import load_remaining_life_table

CONSULTANT_COLUMNS = ("feature_id", "log_dist_ft", "joint_id", "depth_pct", "prev_depth_pct", "rate_mpy",
                      "method", "rate_source", "erf", "rl_yr", "vendor_type")
_ANOMALY_FIELDS = tuple(c for c in CONSULTANT_COLUMNS if c != "method")
_IMP_FIELDS = ("imp_class", "n_classified", "n_previously_repaired", "n_unrepaired")


class AnomalySheet(_Strict):
    sheet: str
    header_row: int = Field(default=1, ge=1)  # 1-based
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
    header_row: int = Field(default=1, ge=1)
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
        blank = out.feature_id.isna().to_numpy()
        if blank.any():
            rows = [int(i) + a.header_row + 1 for i in raw.index[blank]]
            raise ValueError(f"{path.name}: sheet {a.sheet!r} has {len(rows)} anomaly row(s) with a blank feature id "
                             f"(sheet rows {', '.join(map(str, rows))})")
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
    # ERF does not depend on the growth rate: an ERF (or one-sided depth) disagreement is never a growth/matching code
    if r["agree_erf"] is False or math.isnan(r["depth_ours"]) != math.isnan(r["depth_theirs"])             or math.isnan(r["erf_ours"]) != math.isnan(r["erf_theirs"]):
        return "unexplained"
    unknown_theirs = r["method_theirs"] is None and not math.isnan(r["rate_theirs"])
    if not unknown_theirs and (r["history_ours"] == "matched") != (r["method_theirs"] in _THEIRS_MATCHED_METHODS):
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


# ---- IMP class comparison, summary and the command line --------------------------------------------------

DEFAULT_MAP = PROJECT_ROOT / "config" / "consultant_maps" / "integrity_solutions.yaml"
IMP_CLASS_ORDER = ("Immediate", "60-day", "180-day", "YTC < N yrs", "None")
IMP_COMPARISON_COLUMNS = ("imp_class", "n_classified_ours", "n_classified_theirs", "n_previously_repaired_ours",
                          "n_previously_repaired_theirs", "n_unrepaired_ours", "n_unrepaired_theirs")


def _count(value):
    """An int, or None for a blank or non-numeric cell."""
    n = _num(value)
    return None if math.isnan(n) else int(n)


def imp_class_comparison(ours_imp: pd.DataFrame, theirs: pd.DataFrame, ytc_years: float = 5) -> pd.DataFrame:
    """The IMP class counts of both sides, one row per class in ``IMP_CLASS_ORDER`` (``N`` is ``ytc_years``).

    ``ours_imp`` is ``imp_class.csv`` (one row per classified feature, read with ``keep_default_na=False`` for
    ``imp_class``); the counts are recounted from it. Previously repaired is blank when the file has no repair
    information at all. ``theirs`` is ``ConsultantData.imp_summary``."""
    cls = ours_imp.imp_class.astype(str)
    present = sorted({c for c in cls if c.startswith("YTC")})  # the label the run itself used
    ytc = present[0] if present else f"YTC < {ytc_years:g} yrs"
    classes = [ytc if c == "YTC < N yrs" else c for c in IMP_CLASS_ORDER]
    repaired = ours_imp["previously_repaired"] if "previously_repaired" in ours_imp else pd.Series(dtype=object)
    flag = repaired.map(lambda v: str(v).strip().lower() == "true")
    known = bool(repaired.map(lambda v: not _blank(v)).any())
    by_theirs = {} if theirs is None else {r.imp_class: r for r in theirs.itertuples(index=False)}
    rows = []
    for c in classes:
        n = int((cls == c).sum())
        rep = int(((cls == c) & flag).sum()) if known else None
        t = by_theirs.get(c)
        rows.append({"imp_class": c, "n_classified_ours": n,
                     "n_classified_theirs": _count(t.n_classified) if t else None,
                     "n_previously_repaired_ours": rep,
                     "n_previously_repaired_theirs": _count(t.n_previously_repaired) if t else None,
                     "n_unrepaired_ours": None if rep is None else n - rep,
                     "n_unrepaired_theirs": _count(t.n_unrepaired) if t else None})
    return pd.DataFrame(rows, columns=list(IMP_COMPARISON_COLUMNS), dtype=object)


def parity_summary(table: pd.DataFrame, consultant: ConsultantData, settings: Settings, *, workbook_sha256: str,
                   map_sha256: str, comparison: str, imp_class_compared: bool = False,
                   imp_class_note: str = "", method_ours: dict | None = None) -> dict:
    """The tally of a ``compare`` table with the inputs' hashes (JSON-ready)."""
    by_expl = {code: int((table.explanation == code).sum()) for code in EXPLANATIONS}
    shared = table[table.side == "both"]
    agree = {}
    for q in ("depth", "rate", "erf", "rl"):
        col = shared[f"agree_{q}"]
        agree[q] = {"true": int(col.map(lambda v: v is True).sum()),
                    "false": int(col.map(lambda v: v is False).sum()),
                    "none": int(col.map(lambda v: v is None or (isinstance(v, float) and math.isnan(v))).sum())}
    return {"label": DRAFT_LABEL, "generator": f"ili.parity {__version__}",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "workbook_sha256": workbook_sha256, "map_sha256": map_sha256, "comparison": comparison,
            "tolerances": settings.parity.model_dump(),
            "counts_by_side": {k: int((table.side == k).sum()) for k in ("both", "ours_only", "theirs_only")},
            "counts_by_explanation": by_expl, "n_agree": by_expl["agree"], "n_unexplained": by_expl["unexplained"],
            "agree_counts": agree, "unknown_sources": dict(consultant.unknown_sources),
            "method_ours": method_ours or {}, "imp_class_compared": imp_class_compared, "imp_class_note": imp_class_note}


_GROWTH_KEYS = ("governing_profile", "age_basis", "days_per_year", "rate_decimals", "shrinking_applies_to",
                "joint_percentile_pool")


def _method_ours(folder: Path) -> dict:
    """How this program's side was run, from the comparison folder's summaries (null where a summary lacks the key)."""
    def load(name):
        f = folder / name
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    g, rl = load("growth_summary.json"), load("remaining_life_summary.json")
    return {**{k: g.get(k) for k in _GROWTH_KEYS}, "depth_basis": rl.get("depth_basis")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.parity",
                                 description="Compare this program's per-anomaly results with a consultant's "
                                             "workbook. " + DRAFT_LABEL)
    ap.add_argument("--comparison", type=Path, required=True, help="the <prev>_vs_<curr> output folder")
    ap.add_argument("--consultant", type=Path, required=True, help="the consultant's Excel workbook")
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP, help="consultant map YAML")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="settings YAML")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    folder = args.comparison
    try:
        settings = load_settings(args.settings)
        cmap = load_consultant_map(args.map)
        theirs = read_consultant(args.consultant, cmap)
        growth = load_growth_table(folder / "growth.csv")
        rl = load_remaining_life_table(folder / "remaining_life.csv")
        cls = load_classification_table(folder / "classification.csv")
        mt = load_match_table(folder / "match_table.csv")
        ours = ours_by_feature(growth, rl, cls, mt)
        table = compare(ours, theirs.anomalies, settings, box_types=cmap.anomalies.box_types)
        imp_table, note = None, ""
        imp_path = folder / "imp_class.csv"
        if theirs.imp_summary is None:
            note = "the consultant map has no IMP summary sheet"
        elif not imp_path.is_file():
            note = "imp_class.csv not found (run ili.report first)"
        else:
            ours_imp = pd.read_csv(imp_path, dtype=str, keep_default_na=False)
            imp_table = imp_class_comparison(ours_imp, theirs.imp_summary, settings.report.ytc_flag_years)
        summary = parity_summary(table, theirs, settings, workbook_sha256=_sha256(args.consultant),
                                 map_sha256=_sha256(args.map), comparison=folder.name,
                                 imp_class_compared=imp_table is not None, imp_class_note=note,
                                 method_ours=_method_ours(folder))
        out = folder / "parity"
        out.mkdir(exist_ok=True)
        table.to_csv(out / "parity_by_anomaly.csv", index=False, lineterminator="\n")
        if imp_table is not None:
            imp_table.to_csv(out / "parity_imp_class.csv", index=False, lineterminator="\n")
        if imp_table is None:
            (out / "parity_imp_class.csv").unlink(missing_ok=True)  # a stale file from an earlier run
        (out / "parity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(DRAFT_LABEL)
    print("by explanation: " + ", ".join(f"{k} {v}" for k, v in summary["counts_by_explanation"].items()))
    print(f"unexplained: {summary['n_unexplained']}")
    if theirs.unknown_sources:
        print(f"unknown growth sources: {theirs.unknown_sources}")
    if note:
        print(f"IMP class not compared: {note}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
