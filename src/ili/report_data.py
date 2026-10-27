"""Report inputs (spec 6.10): load every file the HTML report and the Excel dig list draw on, and check
that they describe one comparison made with one set of criteria.

``load_report_inputs`` reads the M2-M7 outputs with their own loaders, so every table arrives with the
types its producer wrote. It refuses (``ValueError`` naming the files) a missing required file, outputs
from different runs, an ingest folder that did not ingest both runs, a classification made with other
criteria than the ones now on disk, settings that differ from the values the M6 and M7 summaries record
(``RECORDED_SETTINGS``), and a dig file without the columns the report needs. Validation issues are kept
for the two compared runs only. ``input_hashes`` fingerprints every file read, for the
report's methods appendix and the run manifest.

The table builders below (dig list, pending reviews, match statistics, largest growth, back-test, and the
configuration and criteria listings) take their values from those tables unchanged; none is recomputed.
Missing values stay real (None / NaN) so the HTML and Excel writers can render them as empty cells.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ili.classify import CATEGORY_PRIORITY, FLAG_STATUS_COLUMNS, FLAG_STATUS_FILE, load_classification_table
from ili.config import RepairCriteria, Settings, load_repair_criteria, load_settings
from ili.growth import load_growth_table
from ili.ingest import load_standardized
from ili.match import load_match_table
from ili.remaining_life import load_remaining_life_table
from ili.review import FLAG_COLUMNS, FLAGS_FILE, REVIEW_COLUMNS, REVIEWS_FILE, load_flags, load_reviews

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
_ID_COLUMNS = ("match_id", "curr_feature_id", "feature_id", "segment_id", "decision_id")
REVIEW_STATUS_COLUMNS = ["decision_id", "decision", "applied_status", "reason"]  # match's review_status.csv
CLUSTER_SEP = ";"  # a matched cluster's member feature ids are joined with this (M4)
# Settings the summaries record, as (summary, summary key, dotted settings key). The report refuses settings
# that differ from them; alignment, matching and growth settings are not recorded and are used as they are.
RECORDED_SETTINGS = (
    *(("remaining_life", key, f"{section}.{key}") for section, key in (
        ("strength", "method"), ("strength", "design_factor"), ("strength", "depth_basis"),
        ("strength", "length_basis"), ("tool", "depth_tolerance_pct_wt"), ("tool", "depth_certainty"),
        ("tool", "length_tolerance_in"), ("remaining_life", "time_step_yr"), ("remaining_life", "horizon_yr"),
        ("remaining_life", "rl_factor"), ("remaining_life", "hca_max_interval_months"),
        ("remaining_life", "default_next_assessment_yr"))),
    *(("classification", key, f"classification.{key}") for key in (
        "girth_weld_proximity_in", "crossing_proximity_ft", "seam_clock_min", "seam_clock_tol_min",
        "general_corrosion_min_dim_in", "widespread_circ_min_width_frac", "due_days", "discovery_max_days")),
    ("classification", "depth_basis", "strength.depth_basis"),
    ("classification", "depth_tolerance_pct_wt", "tool.depth_tolerance_pct_wt"),
)
SETTING_REL_TOL = 1e-9  # recorded numbers round-trip through JSON; a real edit is far larger


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
    reviews: pd.DataFrame  # the match decision log (REVIEW_COLUMNS); empty when there is none
    review_status: pd.DataFrame  # match's review_status.csv: what became of each decision; empty when absent
    flags: pd.DataFrame  # the manual flag log (FLAG_COLUMNS); empty when there is none
    flag_status: pd.DataFrame  # classify's flag_status.csv: what became of each flag row; empty when absent
    log_warnings: list[str]  # logs that changed since the run that used them (a warning, not an error)
    summaries: dict[str, dict]
    ingest_manifest: dict
    settings: Settings
    criteria: RepairCriteria
    criteria_path: Path
    settings_path: Path
    paths: dict[str, Path]  # every file read, by role, for the manifest


def _require(path: Path, what: str, *, context: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: {what} not found ({context})")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    """A plain reference or output CSV; id-like columns stay text (``"1"``, never ``1.0``). An empty or
    malformed file is a ValueError naming it."""
    try:
        header = pd.read_csv(path, nrows=0).columns
        return pd.read_csv(path, dtype={c: str for c in _ID_COLUMNS if c in header})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path}: unreadable CSV: {exc}") from None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _differs_only_in_line_endings(path: Path, recorded_sha256: str | None) -> bool:
    """True when ``recorded_sha256`` is the hash of ``path`` with its line endings made all LF or all CRLF
    (a diagnostic for a CRLF checkout; the recorded hashes themselves are of the raw bytes)."""
    lf = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return recorded_sha256 in {hashlib.sha256(b).hexdigest() for b in (lf, lf.replace(b"\n", b"\r\n"))}


def _same_setting(a, b) -> bool:
    """Equal settings values: numbers within ``SETTING_REL_TOL``, dicts key by key, anything else ``==``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_setting(a[k], b[k]) for k in a)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
        return math.isclose(a, b, rel_tol=SETTING_REL_TOL, abs_tol=1e-12)
    return a == b


def _check_recorded_settings(settings: Settings, settings_path: Path, summaries: dict[str, dict],
                             summary_paths: dict[str, Path]) -> None:
    """Refuse settings that differ from a value a summary records (``RECORDED_SETTINGS``); a key the summary
    does not hold is not checked."""
    current = settings.model_dump(mode="json")
    for summary, key, dotted in RECORDED_SETTINGS:
        if key not in summaries[summary]:
            continue
        value = current
        for part in dotted.split("."):
            value = value[part]
        recorded = summaries[summary][key]
        if not _same_setting(value, recorded):
            raise ValueError(f"{settings_path}: {dotted} is {value!r} but {summary_paths[summary]} records "
                             f"{key} {recorded!r}: the outputs were made with other settings; re-run the "
                             "analysis with these settings, or restore the settings file")


def _nearest_member(feature_id, position: float, positions: pd.Series):
    """The current-run feature whose coordinates a dig-list row takes: the id itself, or for a cluster
    (``CLUSTER_SEP``-joined ids) the member whose ``log_dist_ft`` is nearest ``position`` (the first listed
    on a tie, or when no distance is known)."""
    if not isinstance(feature_id, str) or CLUSTER_SEP not in feature_id:
        return feature_id
    members = feature_id.split(CLUSTER_SEP)
    distance = (positions.reindex(members).astype(float) - float(position)).abs().to_numpy()
    return members[int(np.nanargmin(distance))] if np.isfinite(distance).any() else members[0]


def _log_warnings(comparison: Path, summaries: dict[str, dict]) -> list[str]:
    """One message per log (reviews.csv, feature_flags.csv) whose current sha256 differs from the one the
    run that used it recorded (a log that appeared or disappeared since counts). A summary that records
    no log (made before M9) is not checked."""
    out = []
    for stage, sha_key, name in (("match", "reviews_sha256", REVIEWS_FILE),
                                 ("classification", "flags_sha256", FLAGS_FILE)):
        summary = summaries[stage]
        if sha_key not in summary:
            continue
        path = comparison / name
        now = _sha256(path) if path.is_file() else None
        if now != summary[sha_key]:
            out.append(f"{name} has changed since the {stage} step used it (recorded "
                       f"{summary[sha_key] or 'no file'}, now {now or 'no file'}); decisions saved since are not "
                       f"in this report's results. Rerun {stage} and the steps after it.")
    return out


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
    for role, name in (("reviews", REVIEWS_FILE), ("review_status", "review_status.csv"), ("flags", FLAGS_FILE),
                       ("flag_status", FLAG_STATUS_FILE)):
        if (comparison / name).is_file():  # optional: M9 outputs, absent before any review
            files[role] = comparison / name
    if crossings is not None:
        files["crossings"] = Path(crossings)
    if digs is not None:
        files["digs"] = Path(digs)
    contexts = {
        "ingest_manifest": f"an ingest output, looked for in {ingest_dir}",
        "validation_issues": f"an ingest output, looked for in {ingest_dir}",
        "prev_std": "a standardized run given to the report", "curr_std": "a standardized run given to the report",
        **{r: "a reference file given to the report" for r in ("pipe", "mop", "segments", "crossings", "digs")},
        **{r: "a configuration file given to the report" for r in ("settings", "criteria")},
    }
    for role, path in files.items():
        _require(path, role.replace("_", " "),
                 context=contexts.get(role, f"the report needs it alongside {comparison}"))
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

    manifest = _read_json(files["ingest_manifest"])
    ingested = sorted({str(r.get("run_id")) for r in manifest.get("runs") or [] if isinstance(r, dict)})
    absent = [i for i in (prev_id, curr_id) if i not in ingested]
    if absent:
        raise ValueError(f"{files['ingest_manifest']}: no ingest record of run(s) {absent} compared in "
                         f"{comparison} (it lists {ingested}); give the ingest folder of these runs")
    issues = _read_csv(files["validation_issues"])
    if "run_id" not in issues.columns:
        raise ValueError(f"{files['validation_issues']}: missing column run_id")
    issues = issues[issues.run_id.isin([prev_id, curr_id])].reset_index(drop=True)

    criteria_sha = summaries["classification"].get("criteria_sha256")
    if criteria_sha != _sha256(files["criteria"]):
        endings = (" The files differ only in line endings (CRLF against LF): restore the file as it was when "
                   "classified (config YAML is checked out LF, see .gitattributes), or re-run classify."
                   if _differs_only_in_line_endings(files["criteria"], criteria_sha) else "")
        raise ValueError(f"{files['criteria']}: changed since the classification was made "
                         f"({files['classification_summary']} records criteria_sha256 {criteria_sha}); "
                         f"re-run classify, or restore the criteria file.{endings}")
    settings = load_settings(files["settings"])
    _check_recorded_settings(settings, files["settings"], summaries,
                             {k: files[f"{k}_summary"] for k in SUMMARY_FILES})

    dig_table = None
    if digs is not None:
        dig_table = _read_csv(files["digs"])
        missing = [c for c in DIG_COLUMNS if c not in dig_table.columns]
        if missing:
            raise ValueError(f"{files['digs']}: missing column(s) {missing}")

    reviews = load_reviews(files["reviews"]) if "reviews" in files else pd.DataFrame(columns=REVIEW_COLUMNS)
    flags = load_flags(files["flags"]) if "flags" in files else pd.DataFrame(columns=FLAG_COLUMNS)
    review_status = (_read_csv(files["review_status"]) if "review_status" in files
                     else pd.DataFrame(columns=REVIEW_STATUS_COLUMNS))

    flag_status = (_read_csv(files["flag_status"]) if "flag_status" in files
                   else pd.DataFrame(columns=FLAG_STATUS_COLUMNS))

    return ReportInputs(
        prev_std=prev_df, curr_std=curr_df,
        match_table=load_match_table(files["match_table"]),
        growth=load_growth_table(files["growth"]),
        growth_segments=_read_csv(files["growth_segments"]),
        remaining_life=load_remaining_life_table(files["remaining_life"]),
        intervals=_read_csv(files["intervals"]),
        classification=load_classification_table(files["classification"]),
        changed_sections=_read_csv(files["changed_sections"]),
        validation_issues=issues,
        pipe=_read_csv(files["pipe"]), mop=_read_csv(files["mop"]), segments=_read_csv(files["segments"]),
        crossings=_read_csv(files["crossings"]) if crossings is not None else None,
        digs=dig_table, reviews=reviews, review_status=review_status, flags=flags, flag_status=flag_status,
        log_warnings=_log_warnings(comparison, summaries), summaries=summaries, ingest_manifest=manifest,
        settings=settings, criteria=load_repair_criteria(files["criteria"]),
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
    located = pd.Series([_nearest_member(f, p, std.log_dist_ft)
                         for f, p in zip(out.curr_feature_id, out.curr_log_dist_ft)], index=out.index, dtype=object)
    for col in ("lat", "lon"):
        out[col] = located.map(std[col])
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
    table: pd.DataFrame  # one row per dig found in the current run with a finite ILI and field depth
    n_within: int
    pct_within: float  # of the rows in ``table``; NaN when there are none
    unmatched_ids: list[str]  # dig feature ids not in the current run
    no_depth_ids: list[str]  # dig feature ids in the current run without an ILI or a field depth (not compared)


def backtest(digs: pd.DataFrame | None, curr_std: pd.DataFrame, settings: Settings) -> BacktestResult | None:
    """Reported depth against field depth for each dig (spec 8). None without digs. A dig without a finite ILI
    or field depth goes to ``no_depth_ids`` and counts neither within nor outside the tolerance."""
    if digs is None or digs.empty:
        return None
    found = digs.feature_id.isin(curr_std.feature_id)
    unmatched = [str(i) for i in digs.feature_id[~found]]
    std = curr_std.drop_duplicates("feature_id").set_index("feature_id")
    dug = digs[found].reset_index(drop=True)
    table = pd.DataFrame({"feature_id": dug.feature_id,
                          "ili_depth_pct": pd.to_numeric(dug.feature_id.map(std.depth_pct), errors="coerce"),
                          "field_depth_pct": pd.to_numeric(dug.field_depth_pct, errors="coerce")})
    finite = np.isfinite(table.ili_depth_pct.to_numpy(float)) & np.isfinite(table.field_depth_pct.to_numpy(float))
    no_depth = [str(i) for i in table.feature_id[~finite]]
    table = table[finite].reset_index(drop=True)
    table["diff_pct"] = table.ili_depth_pct - table.field_depth_pct
    table["within_tolerance"] = table.diff_pct.abs().round(9) <= settings.tool.depth_tolerance_pct_wt
    n_within = int(table.within_tolerance.sum())
    pct = 100.0 * n_within / len(table) if len(table) else float("nan")
    return BacktestResult(table, n_within, pct, unmatched, no_depth)


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
