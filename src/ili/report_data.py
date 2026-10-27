"""Report inputs (spec 6.10): load every file the HTML report and the Excel dig list draw on, and check
that they describe one comparison made with one set of criteria.

``load_report_inputs`` reads the M2-M7 outputs with their own loaders, so every table arrives with the
types its producer wrote. It refuses (``ValueError`` naming the files) a missing required file, outputs
from different runs, a classification made with other criteria than the ones now on disk, and a dig
file without the columns the report needs. ``input_hashes`` fingerprints every file read, for the
report's methods appendix and the run manifest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ili.classify import load_classification_table
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
