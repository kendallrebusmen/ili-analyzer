"""Session state and cached reads for the review app. DRAFT output only.

The app never writes the comparison folder's ``pipeline_inputs.json``; only the pipeline does.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ili.align_welds import load_weld_map
from ili.classify import FLAG_STATUS_FILE, load_classification_table
from ili.config import PROJECT_ROOT
from ili.growth import load_growth_table
from ili.match import load_match_table
from ili.pipeline import INPUTS_FILE, PipelineInputs, load_pipeline_inputs
from ili.remaining_life import load_remaining_life_table
from ili.review import FLAGS_FILE, REVIEWS_FILE, load_flags, load_reviews

COMPARISON_KEY = "comparison_dir"
INPUTS_KEY = "pipeline_inputs"
REVIEWER_KEY = "reviewer"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REVIEW_STATUS_FILE = "review_status.csv"
_FILES = ("weld_map.csv", "changed_sections.csv", "alignment_summary.json", "match_table.csv",
          "match_summary.json", "growth.csv", "remaining_life.csv", "classification.csv",
          "classification_summary.json", REVIEW_STATUS_FILE, FLAG_STATUS_FILE, REVIEWS_FILE, FLAGS_FILE)


@dataclass(frozen=True)
class ComparisonData:
    """The outputs of one comparison folder; a stage that has not run leaves an empty frame or dict."""
    weld_map: pd.DataFrame
    changed_sections: pd.DataFrame
    alignment_summary: dict
    match_table: pd.DataFrame
    match_summary: dict
    growth: pd.DataFrame
    remaining_life: pd.DataFrame
    classification: pd.DataFrame
    classification_summary: dict
    review_status: pd.DataFrame
    reviews: pd.DataFrame
    flags: pd.DataFrame


def latest_comparison(outputs: Path | None = None) -> Path | None:
    """The most recently modified ``*_vs_*`` folder that holds a ``pipeline_inputs.json``."""
    found = [p for p in Path(OUTPUTS_DIR if outputs is None else outputs).glob("*_vs_*") if (p / INPUTS_FILE).is_file()]
    return max(found, key=lambda p: (p / INPUTS_FILE).stat().st_mtime, default=None)


def comparison_from_session(session_state=None, outputs: Path | None = None) -> Path | None:
    """The comparison folder in use: the session's, else the latest on disk, else None."""
    if session_state is None:
        import streamlit as st
        session_state = st.session_state
    chosen = session_state.get(COMPARISON_KEY)
    return Path(chosen) if chosen else latest_comparison(OUTPUTS_DIR if outputs is None else outputs)


def inputs_from_session(path: Path | None, session_state=None) -> PipelineInputs | None:
    """The session's setup, else the one the pipeline recorded in the comparison folder, else None."""
    if session_state is None:
        import streamlit as st
        session_state = st.session_state
    setup = session_state.get(INPUTS_KEY)
    if setup is None and path is not None and (Path(path) / INPUTS_FILE).is_file():
        try:
            setup = load_pipeline_inputs(Path(path) / INPUTS_FILE)
        except ValueError:
            setup = None
    return setup


def _mtimes(path: Path) -> tuple:
    return tuple((p.stat().st_mtime_ns if (p := path / n).is_file() else None) for n in _FILES)


def _table(path: Path, loader) -> pd.DataFrame:
    return loader(path) if path.is_file() else pd.DataFrame()


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.is_file() else pd.DataFrame()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_comparison(path: Path) -> ComparisonData:
    p = Path(path)
    cs = p / "changed_sections.csv"
    return ComparisonData(
        weld_map=_table(p / "weld_map.csv", load_weld_map),
        changed_sections=pd.read_csv(cs) if cs.is_file() else pd.DataFrame(),
        alignment_summary=_json(p / "alignment_summary.json"),
        match_table=_table(p / "match_table.csv", load_match_table),
        match_summary=_json(p / "match_summary.json"),
        growth=_table(p / "growth.csv", load_growth_table),
        remaining_life=_table(p / "remaining_life.csv", load_remaining_life_table),
        classification=_table(p / "classification.csv", load_classification_table),
        classification_summary=_json(p / "classification_summary.json"),
        review_status=_csv(p / REVIEW_STATUS_FILE),
        reviews=load_reviews(p / REVIEWS_FILE),
        flags=load_flags(p / FLAGS_FILE))


def _cached(path: str, mtimes: tuple) -> ComparisonData:
    return read_comparison(Path(path))


try:
    import streamlit as _st
    _cached = _st.cache_data(show_spinner=False)(_cached)
except ImportError:  # pragma: no cover - streamlit is a requirement
    pass


def load_comparison(path: Path) -> ComparisonData:
    """Read the comparison folder's outputs; cached, keyed by the files' mtimes so a rerun refreshes it."""
    return _cached(str(Path(path)), _mtimes(Path(path)))
