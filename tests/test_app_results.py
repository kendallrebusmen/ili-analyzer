"""The growth and remaining life explorer, and the dig list, flag editor and export page."""

import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from ili.align_welds import DRAFT_LABEL
from ili.app_pages_results import SAVED_AFTER_RUN, explorer_rows
from ili.review import FLAGS_FILE, load_flags
from test_app_setup import _all_text, _app, isolated_outputs, ran, script_config  # noqa: F401  (fixtures)

EXPLORER = "Growth and remaining life explorer"
EXPORT = "Dig list and report export"


def _data():
    growth = pd.DataFrame({
        "match_id": ["M1", "M2", "M3", "M4"], "match_type": ["1:1"] * 4, "curr_feature_id": ["C1", "C2", "C3", "C4"],
        "curr_log_dist_ft": [10.0, 20.0, 30.0, 40.0], "segment_id": ["S1", "S1", "S2", "S2"],
        "raw_mpy": [1.0, 2.0, 3.0, 4.0], "governing_mpy": [1.5, 2.5, 3.5, 4.5],
        "governing_method": ["a", "b", "a", "b"]})
    rl = pd.DataFrame({"match_id": ["M1", "M2", "M3", "M4"], "rl_yr": [5.0, 6.0, 7.0, 8.0],
                       "rl_governing": ["tau_80", "tau_80", "tau_burst", "tau_burst"]})
    cls = pd.DataFrame({"match_id": ["M1", "M2", "M3", "M4"],
                        "governing_category": ["immediate", "evaluate", "evaluate", "180_day"],
                        "governing_rule": ["r1", None, None, "r2"], "due_date": [None] * 4})
    return SimpleNamespace(growth=growth, remaining_life=rl, classification=cls)


def test_explorer_filters():
    d = _data()
    ids = lambda *a: list(explorer_rows(d, *a).match_id)  # noqa: E731
    assert ids([], [], []) == ["M1", "M2", "M3", "M4"]
    assert ids(["S1"], [], []) == ["M1", "M2"]
    assert ids([], ["evaluate"], []) == ["M2", "M3"]
    assert ids([], [], ["a"]) == ["M1", "M3"]
    assert ids(["S2"], ["evaluate", "180_day"], ["a"]) == ["M3"]
    assert ids(["S1"], ["180_day"], []) == []
    rows = explorer_rows(d, [], [], [])
    assert {"segment_id", "governing_mpy", "governing_method", "rl_yr", "governing_category"} <= set(rows.columns)


def _page(folder, tmp_path, page, **state):
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    at = _app(comparison_dir=str(work), **state).run()
    at.radio(key="page").set_value(page).run()
    assert not at.exception
    return at, work


def test_explorer_page_filters_and_charts(ran, tmp_path):
    _, folder = ran
    at, _ = _page(folder, tmp_path, EXPLORER)
    assert DRAFT_LABEL in _all_text(at)
    segs = at.multiselect(key="explorer_segments")
    assert segs.options
    segs.set_value(segs.options[:1]).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 2
    assert set(at.dataframe[0].value.segment_id) == set(segs.options[:1])


def _feature(folder):
    from ili.app_state import read_comparison
    return str(read_comparison(folder).classification.curr_feature_id.iloc[0]).split(";")[0]


def test_flag_editor_writes_log(ran, tmp_path):
    _, folder = ran
    feature = _feature(folder)
    at, work = _page(folder, tmp_path, EXPORT, reviewer="A. Reviewer")
    at.selectbox(key="flag_feature").set_value(feature).run()
    at.checkbox(key=f"flag_manual_immediate_{feature}").set_value(True)
    at.text_input(key="flag_note").set_value("seen in the field")
    [b for b in at.button if b.label == "Save flags"][0].click().run()
    assert not at.exception
    log = load_flags(work / FLAGS_FILE)
    assert len(log) == 1  # only the changed flag is written
    got = log.iloc[0]
    assert (got.curr_feature_id, got.flag, got.value, got.reviewer, got.note) == (
        feature, "manual_immediate", "true", "A. Reviewer", "seen in the field")
    # saved after the last run: shown as such, and the editor now shows the saved value
    assert SAVED_AFTER_RUN in " ".join(str(df.value.to_dict("list")) for df in at.dataframe)


def test_flag_editor_needs_reviewer(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path, EXPORT)
    assert [b for b in at.button if b.label == "Save flags"][0].disabled
    assert not (work / FLAGS_FILE).exists()


def test_flag_unchanged_writes_nothing(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path, EXPORT, reviewer="A. Reviewer")
    [b for b in at.button if b.label == "Save flags"][0].click().run()
    assert not at.exception and not (work / FLAGS_FILE).exists()


def test_export_buttons_present_after_report(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path, EXPORT)
    assert DRAFT_LABEL in _all_text(at)
    assert len(at.get("download_button")) == 4
    assert any(b.label == "Rerun the pipeline" for b in at.button)
    assert not at.info and not at.error
    assert len(at.dataframe) >= 2 and len(at.dataframe[0].value) > 0  # the dig list, then the flags table
    for name in ("report.pdf", "report.html", "dig_list.xlsx", "run_manifest.json"):
        assert name in _all_text(at)
    (work / "dig_list.xlsx").unlink()
    at.run()
    assert len(at.get("download_button")) == 3


def test_flag_save_on_a_locked_log_says_close_the_file(ran, tmp_path, monkeypatch):
    """M3: a PermissionError (the CSV open in Excel) is a message, not a traceback."""
    from ili import app_pages_results

    def locked(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "feature_flags.csv")
    monkeypatch.setattr(app_pages_results, "append_flag", locked)
    _, folder = ran
    feature = _feature(folder)
    at, work = _page(folder, tmp_path, EXPORT, reviewer="A. Reviewer")
    at.selectbox(key="flag_feature").set_value(feature).run()
    at.checkbox(key=f"flag_manual_immediate_{feature}").set_value(True)
    [b for b in at.button if b.label == "Save flags"][0].click().run()
    assert not at.exception
    assert [e.value for e in at.error if "feature_flags.csv could not be written" in e.value and "close it" in e.value]
