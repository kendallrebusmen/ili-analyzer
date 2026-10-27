"""Report inputs (spec 6.10): load every file the HTML report and the Excel dig list draw on, and check
that they describe one comparison made with one set of criteria.

``load_report_inputs`` reads the M2-M7 outputs with their own loaders, so every table arrives with the
types its producer wrote. It refuses (``ValueError`` naming the files) a missing required file, outputs
from different runs, a classification made with other criteria than the ones now on disk, and a dig
file without the columns the report needs. ``input_hashes`` fingerprints every file read, for the
report's methods appendix and the run manifest.

The table builders below (dig list, pending reviews, match statistics, largest growth, back-test, and the
configuration and criteria listings) take their values from those tables unchanged; none is recomputed.
Missing values stay real (None / NaN) so the HTML and Excel writers can render them as empty cells.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ili.classify import CATEGORY_PRIORITY, load_classification_table
from ili.config import RepairCriteria, Settings, load_repair_criteria, load_settings
from ili.growth import load_growth_table
from ili.ingest import load_standardized
from ili.match import load_match_table
from ili.remaining_life import load_remaining_life_table

SUMMARY_FILES = {
    "alignment": "alignment_summary.json",
    "match": "match_summary.json",
    "growth": "growth_summary.json",
    "remaining_life": "remaining_life_summary.json",
    "classification": "classification_summary.json",
}
DIG_COLUMNS = ("feature_id", "field_depth_pct", "field_length_in", "dig_date")  # spec 5.2 digs.csv
DIG_LIST_CATEGORIES = CATEGORY_PRIORITY[:-1]  # everything that goes on the dig list; "evaluate" does not
DIG_LIST_COLUMNS = (
    "governing_category", "governing_rule", "match_id", "prev_feature_id", "curr_feature_id", "feature_type",
    "curr_log_dist_ft", "curr_us_gw_id", "curr_dist_from_us_gw_ft", "clock_min", "lat", "lon",
    "prev_depth_pct", "curr_depth_pct", "depth_pct_assessed", "prev_length_in", "curr_length_in",
    "dent_depth_pct_od",
    "governing_mpy", "governing_method",
    "p_fail", "p_safe", "mop_local",
    "rl_yr", "rl_governing",
    "triggered_rules", "due_date", "due_date_basis", "worst_case_category", "requires_review",
    "review_reasons", "action_note",
)
_ID_COLUMNS = ("match_id", "curr_feature_id", "feature_id", "segment_id")


@dataclass(frozen=True)
class ReportInputs:
    prev_std: pd.DataFrame
    curr_std: pd.DataFrame
    match_table: pd.DataFrame
    growth: pd.DataFrame
    growth_segments: pd.DataFrame
    remaining_life: pd.DataFrame
    intervals: pd.DataFrame
    classification: pd.DataFrame
    changed_sections: pd.DataFrame
    validation_issues: pd.DataFrame
    pipe: pd.DataFrame
    mop: pd.DataFrame
    segments: pd.DataFrame
    crossings: pd.DataFrame | None
    digs: pd.DataFrame | None
    summaries: dict[str, dict]
    ingest_manifest: dict
    settings: Settings
    criteria: RepairCriteria
    criteria_path: Path
    settings_path: Path
    paths: dict[str, Path]  # every file read, by role, for the manifest


def _require(path: Path, what: str, *, needed_by: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: {what} not found (the report needs it alongside {needed_by})")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    """A plain reference or output CSV; id-like columns stay text (``"1"``, never ``1.0``)."""
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in _ID_COLUMNS if c in header})


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_report_inputs(comparison: Path, prev_std: Path, curr_std: Path, ingest_dir: Path, pipe: Path, mop: Path,
                       segments: Path, crossings: Path | None, digs: Path | None, settings_path: Path,
                       criteria_path: Path) -> ReportInputs:
    """Load and cross-check everything the report reads. ``comparison`` is the
    ``<prev>_vs_<curr>`` folder; ``ingest_dir`` holds ``ingest_manifest.json`` and
    ``validation_issues.csv``; ``crossings`` and ``digs`` are optional (None)."""
    comparison, ingest_dir = Path(comparison), Path(ingest_dir)
    files = {
        "prev_std": Path(prev_std), "curr_std": Path(curr_std),
        "match_table": comparison / "match_table.csv", "growth": comparison / "growth.csv",
        "growth_segments": comparison / "growth_segments.csv",
        "remaining_life": comparison / "remaining_life.csv",
        "intervals": comparison / "reinspection_intervals.csv",
        "classification": comparison / "classification.csv",
        "changed_sections": comparison / "changed_sections.csv",
        **{f"{k}_summary": comparison / name for k, name in SUMMARY_FILES.items()},
        "ingest_manifest": ingest_dir / "ingest_manifest.json",
        "validation_issues": ingest_dir / "validation_issues.csv",
        "pipe": Path(pipe), "mop": Path(mop), "segments": Path(segments),
        "settings": Path(settings_path), "criteria": Path(criteria_path),
    }
    if crossings is not None:
        files["crossings"] = Path(crossings)
    if digs is not None:
        files["digs"] = Path(digs)
    for role, path in files.items():
        _require(path, role.replace("_", " "), needed_by=comparison)
    paths = {role: path.resolve() for role, path in files.items()}

    summaries = {k: _read_json(files[f"{k}_summary"]) for k in SUMMARY_FILES}
    prev_curr = {k: (s.get("prev_run_id"), s.get("curr_run_id")) for k, s in summaries.items()}
    if len(set(prev_curr.values())) != 1:
        detail = "; ".join(f"{SUMMARY_FILES[k]}: {p} vs {c}" for k, (p, c) in prev_curr.items())
        raise ValueError(f"{comparison}: the summaries describe different runs ({detail})")
    prev_id, curr_id = next(iter(prev_curr.values()))

    prev_df, curr_df = load_standardized(files["prev_std"]), load_standardized(files["curr_std"])
    for path, df, expected in ((files["prev_std"], prev_df, prev_id), (files["curr_std"], curr_df, curr_id)):
        found = sorted(set(df.run_id.dropna()))
        if found != [expected]:
            raise ValueError(f"{path}: run_id {found} does not match {expected!r} in the summaries under "
                             f"{comparison}")

    criteria_sha = summaries["classification"].get("criteria_sha256")
    if criteria_sha != _sha256(files["criteria"]):
        raise ValueError(f"{files['criteria']}: changed since the classification was made "
                         f"({files['classification_summary']} records criteria_sha256 {criteria_sha}); "
                         "re-run classify, or restore the criteria file")

    dig_table = None
    if digs is not None:
        dig_table = _read_csv(files["digs"])
        missing = [c for c in DIG_COLUMNS if c not in dig_table.columns]
        if missing:
            raise ValueError(f"{files['digs']}: missing column(s) {missing}")

    return ReportInputs(
        prev_std=prev_df, curr_std=curr_df,
        match_table=load_match_table(files["match_table"]),
        growth=load_growth_table(files["growth"]),
        growth_segments=_read_csv(files["growth_segments"]),
        remaining_life=load_remaining_life_table(files["remaining_life"]),
        intervals=_read_csv(files["intervals"]),
        classification=load_classification_table(files["classification"]),
        changed_sections=_read_csv(files["changed_sections"]),
        validation_issues=_read_csv(files["validation_issues"]),
        pipe=_read_csv(files["pipe"]), mop=_read_csv(files["mop"]), segments=_read_csv(files["segments"]),
        crossings=_read_csv(files["crossings"]) if crossings is not None else None,
        digs=dig_table, summaries=summaries, ingest_manifest=_read_json(files["ingest_manifest"]),
        settings=load_settings(files["settings"]), criteria=load_repair_criteria(files["criteria"]),
        criteria_path=files["criteria"], settings_path=files["settings"], paths=paths)


def input_hashes(inputs: ReportInputs) -> dict[str, str]:
    """``{str(path): sha256}`` for every file read, sorted by path so the order is stable."""
    return {str(p): _sha256(p) for p in sorted(set(inputs.paths.values()), key=str)}


# --- tables (spec 6.10) -------------------------------------------------------------------------------


def _lookup(source: pd.DataFrame, keys: pd.Series, columns: list[str]) -> pd.DataFrame:
    """``source[columns]`` by ``match_id``, aligned to ``keys`` (NaN where a match id has no row)."""
    by_id = source.drop_duplicates("match_id").set_index("match_id")[columns]
    return by_id.reindex(keys.to_numpy()).set_axis(keys.index)


def dig_list(inputs: ReportInputs) -> pd.DataFrame:
    """The classification rows in ``DIG_LIST_CATEGORIES`` with the columns ``DIG_LIST_COLUMNS``, most severe
    category first, then shortest remaining life (none last), earliest due date (none last), match id."""
    cls = inputs.classification
    out = cls[cls.governing_category.isin(DIG_LIST_CATEGORIES)].reset_index(drop=True)
    keys = out.match_id
    out = out.join(_lookup(inputs.match_table, keys, [
        "prev_feature_id", "curr_log_dist_ft", "curr_us_gw_id", "curr_dist_from_us_gw_ft", "prev_depth_pct",
        "curr_depth_pct", "prev_length_in", "curr_length_in"]))
    out = out.join(_lookup(inputs.growth, keys, ["governing_mpy", "governing_method"]))
    out = out.join(_lookup(inputs.remaining_life, keys, ["rl_yr", "rl_governing"]))
    std = inputs.curr_std.drop_duplicates("feature_id").set_index("feature_id")
    for col in ("lat", "lon"):
        out[col] = out.curr_feature_id.map(std[col])
    order = (out.assign(_rank=out.governing_category.map(DIG_LIST_CATEGORIES.index),
                        _rl=out.rl_yr.astype(float), _due=pd.to_datetime(out.due_date))
             .sort_values(["_rank", "_rl", "_due", "match_id"], na_position="last").index)
    return out.loc[order, list(DIG_LIST_COLUMNS)].reset_index(drop=True)


def pending_reviews(match_table: pd.DataFrame) -> pd.DataFrame:
    """Matches a person still has to review (spec 6.10.3): ``needs_review`` and status ``pending``."""
    mask = match_table.needs_review.fillna(False).astype(bool) & (match_table.review_status == "pending")
    return match_table[mask].reset_index(drop=True)


def match_stats(match_table: pd.DataFrame) -> pd.DataFrame:
    """Counts by ``match_type`` (rows) and ``confidence`` (columns), plus an ``all`` column."""
    totals = match_table.match_type.value_counts()
    counts = pd.crosstab(match_table.match_type, match_table.confidence).reindex(totals.index, fill_value=0)
    counts["all"] = totals  # includes rows with no confidence (unmatched features)
    return counts


def largest_growth(inputs: ReportInputs, n: int) -> pd.DataFrame:
    """The ``n`` growth rows with the largest ``governing_mpy`` (ties by match id; no rate sorts last)."""
    g = inputs.growth.sort_values(["governing_mpy", "match_id"], ascending=[False, True], na_position="last")
    return g.head(n).reset_index(drop=True)


@dataclass(frozen=True)
class BacktestResult:
    table: pd.DataFrame  # one row per dig found in the current run
    n_within: int
    pct_within: float  # of the rows in ``table``; NaN when there are none
    unmatched_ids: list[str]  # dig feature ids not in the current run


def backtest(digs: pd.DataFrame | None, curr_std: pd.DataFrame, settings: Settings) -> BacktestResult | None:
    """Reported depth against field depth for each dig (spec 8). None without digs."""
    if digs is None or digs.empty:
        return None
    found = digs.feature_id.isin(curr_std.feature_id)
    unmatched = [str(i) for i in digs.feature_id[~found]]
    std = curr_std.drop_duplicates("feature_id").set_index("feature_id")
    dug = digs[found].reset_index(drop=True)
    table = pd.DataFrame({"feature_id": dug.feature_id, "ili_depth_pct": dug.feature_id.map(std.depth_pct),
                          "field_depth_pct": dug.field_depth_pct})
    table["diff_pct"] = table.ili_depth_pct - table.field_depth_pct
    table["within_tolerance"] = table.diff_pct.abs().round(9) <= settings.tool.depth_tolerance_pct_wt
    n_within = int(table.within_tolerance.sum())
    pct = 100.0 * n_within / len(table) if len(table) else float("nan")
    return BacktestResult(table, n_within, pct, unmatched)


_YAML_KEY = re.compile(r"^(\s*)([^\s#:-][^:]*?)\s*:(?:\s|$)")


def _placeholder_paths(text: str) -> set[tuple[str, ...]]:
    """Key paths whose line in the YAML text carries ``PLACEHOLDER`` (found by tracking the indentation path)."""
    stack: list[tuple[int, str]] = []
    found: set[tuple[str, ...]] = set()
    for line in text.splitlines():
        m = _YAML_KEY.match(line)
        if m is None:
            continue
        indent, key = len(m.group(1)), m.group(2).strip("\"'")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if "PLACEHOLDER" in line:
            found.add(tuple(k for _, k in stack))
    return found


def _leaves(node, path: tuple[str, ...] = ()):
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from _leaves(v, (*path, str(k)))
    else:
        yield path, node  # lists and None are leaves


def config_rows(settings: Settings, settings_path: Path) -> pd.DataFrame:
    """Every setting as ``key`` (dotted), ``value`` and ``placeholder``. A ``PLACEHOLDER`` comment on a key's
    line marks that key and every leaf beneath it."""
    marked = _placeholder_paths(Path(settings_path).read_text(encoding="utf-8"))
    rows = [(".".join(path), value, any(path[:i] in marked for i in range(1, len(path) + 1)))
            for path, value in _leaves(settings.model_dump())]
    out = pd.DataFrame(rows, columns=["key", "value", "placeholder"])
    out["value"] = out.value.astype(object)
    return out


def criteria_rows(criteria: RepairCriteria) -> pd.DataFrame:
    """The repair rules in file order, with ``feature_types`` joined into one comma-separated string."""
    return pd.DataFrame([
        {"id": r.id, "category": r.category, "applies_to": r.applies_to,
         "feature_types": ", ".join(getattr(t, "value", str(t)) for t in r.feature_types),
         "condition": r.condition, "requires_review": r.requires_review, "description": r.description}
        for r in criteria.rules
    ], columns=["id", "category", "applies_to", "feature_types", "condition", "requires_review", "description"])
