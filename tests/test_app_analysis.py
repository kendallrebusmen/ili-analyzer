"""The Analysis table page: filters, the column-group selector, and the Excel download."""

import shutil

import pandas as pd

from ili.align_welds import DRAFT_LABEL
from ili.analysis_table import ANALYSIS_TABLE_FILE, ANALYSIS_XLSX_FILE, load_analysis_table
from test_app_setup import _all_text, _app, isolated_outputs, ran, script_config  # noqa: F401  (fixtures)

PAGE = "Analysis table"


def _page(folder, tmp_path, **state):
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    at = _app(comparison_dir=str(work), **state).run()
    at.radio(key="page").set_value(PAGE).run()
    assert not at.exception
    return at, work


def _shown(at) -> pd.DataFrame:
    return at.dataframe[0].value


def test_page_follows_the_explorer_and_has_a_plain_label():
    from ili import app_theme
    at = _app().run()
    options = list(at.radio(key="page").options)  # the sidebar shows the plain labels
    assert options.index("Analysis Table") == options.index("Growth Rates") + 1
    assert app_theme.nav_label_text(PAGE) == "Analysis Table"


def test_page_without_a_comparison_shows_the_neutral_text():
    at = _app().run()
    at.radio(key="page").set_value(PAGE).run()
    assert not at.exception
    assert any("nemi-badge" in m.value and DRAFT_LABEL in m.value for m in at.markdown)
    assert [i.value for i in at.info] == ["No data yet: run the pipeline on the Run setup page."]


def test_page_without_the_table_file_says_to_run_the_pipeline(ran, tmp_path):
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    (work / ANALYSIS_TABLE_FILE).unlink()
    at = _app(comparison_dir=str(work)).run()
    at.radio(key="page").set_value(PAGE).run()
    assert not at.exception
    assert any("the report stage writes the analysis table" in i.value for i in at.info)
    assert not at.dataframe


def test_page_shows_the_table_with_the_default_groups(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path)
    assert DRAFT_LABEL in _all_text(at)
    full = load_analysis_table(work / ANALYSIS_TABLE_FILE)
    shown = _shown(at)
    assert len(shown) == len(full) > 0
    assert list(shown.columns)[:2] == ["Feature ID", "Odometer (ft)"]
    assert "Corrosion Growth Rate (MPY)" in shown.columns  # Growth is on by default
    assert "Dent Depth (in)" not in shown.columns  # Dents is not
    assert f"{len(full)} rows" in _all_text(at)
    assert at.multiselect(key="analysis_groups").value == ["Identity & location", "Growth", "Strength", "Life & class"]


def test_group_selector_changes_the_columns(ran, tmp_path):
    _, folder = ran
    at, _ = _page(folder, tmp_path)
    at.multiselect(key="analysis_groups").set_value(["Dents"]).run()
    assert not at.exception
    cols = list(_shown(at).columns)
    assert cols == ["Feature ID", "Odometer (ft)", "Dent Depth (in)", "Dent Depth (%)", "Is Dent on Top of Pipe"]


def test_filters_reduce_the_rows(ran, tmp_path):
    _, folder = ran
    at, work = _page(folder, tmp_path)
    full = load_analysis_table(work / ANALYSIS_TABLE_FILE)
    n = len(_shown(at))
    surfaces = at.multiselect(key="analysis_surfaces")
    surfaces.set_value([surfaces.options[0]]).run()
    assert 0 < len(_shown(at)) < n
    assert set(_shown(at)["Feature ID"]) <= set(full["Feature ID"])
    at.multiselect(key="analysis_surfaces").set_value([]).run()
    assert len(_shown(at)) == n
    categories = at.multiselect(key="analysis_categories")
    categories.set_value([categories.options[0]]).run()
    assert 0 < len(_shown(at)) <= n
    at.multiselect(key="analysis_categories").set_value([]).run()
    low, high = at.slider(key="analysis_range").value
    mid = (low + high) / 2
    at.slider(key="analysis_range").set_value((low, mid)).run()
    assert 0 < len(_shown(at)) < n
    assert float(_shown(at)["Odometer (ft)"].max()) <= mid + 1e-6
    methods = at.multiselect(key="analysis_methods")
    assert methods.options  # the growth rate sources present in the data


def test_download_button_serves_the_workbook(ran, tmp_path):
    _, folder = ran
    at, _ = _page(folder, tmp_path)
    [button] = at.get("download_button")
    assert button.proto.label.endswith(ANALYSIS_XLSX_FILE)
    assert DRAFT_LABEL in _all_text(at)
