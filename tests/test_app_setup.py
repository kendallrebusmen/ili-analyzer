"""The review app shell, Run setup and Weld alignment pages, and the local-only Streamlit config."""

import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ili import app_state
from ili.align_welds import DRAFT_LABEL
from ili.pipeline import STAGES, run_pipeline
from test_pipeline import FOLDER, _inputs

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Run setup", "Weld alignment", "Match review queue", "Growth and remaining life explorer",
         "Dig list and report export"]
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def ran(synth, tmp_path_factory):
    _, data = synth
    inputs = _inputs(data, tmp_path_factory.mktemp("app_out"))
    results = run_pipeline(inputs)
    assert [r.exit_code for r in results] == [0] * len(STAGES)
    return inputs, inputs.out / FOLDER


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _all_text(at):
    return " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.title])


def test_app_starts_with_draft_label():
    at = _app().run()
    assert not at.exception
    assert any(DRAFT_LABEL in w.value for w in at.sidebar.warning)
    assert at.text_input(key="reviewer").value == ""


def test_run_setup_runs_pipeline(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out")
    at = _app().run()
    for name, value in vars(inputs).items():
        if value is not None:
            at.text_input(key=f"setup_{name}").set_value(str(value))
    at.button[0].click().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert labels == [f"{s}: exit 0" for s in STAGES]
    assert not at.error
    assert at.session_state[app_state.COMPARISON_KEY] == str(inputs.out / FOLDER)
    assert (inputs.out / FOLDER / "report.html").is_file()


def test_run_setup_shows_failing_stderr(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out", pipe=data / "reference" / "missing.csv")
    at = _app().run()
    for name, value in vars(inputs).items():
        if value is not None:
            at.text_input(key=f"setup_{name}").set_value(str(value))
    at.button[0].click().run()
    assert at.error and "match failed" in at.error[0].value and "missing.csv" in at.error[0].value


def test_run_setup_requires_inputs():
    at = _app().run()
    at.button[0].click().run()
    assert at.error and "prev_tally" in at.error[0].value


def test_weld_alignment_page_shows_changed_sections(ran):
    _, folder = ran
    at = _app(comparison_dir=str(folder)).run()
    at.sidebar.radio(key="page").set_value("Weld alignment").run()
    assert not at.exception
    assert [m.label for m in at.metric][-1] == "Changed sections"
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 7
    assert DRAFT_LABEL in _all_text(at)


def test_pages_without_data_show_neutral_text_and_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(app_state, "OUTPUTS_DIR", tmp_path)
    at = _app().run()
    for page in PAGES[1:]:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception
        assert [i.value for i in at.info] == ["No data yet: run the pipeline on the Run setup page."]
        assert DRAFT_LABEL in _all_text(at)


def test_pages_with_data_show_tables_and_draft(ran):
    _, folder = ran
    at = _app(comparison_dir=str(folder)).run()
    for page in PAGES[2:]:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception and len(at.dataframe) == 1 and not at.info
        assert DRAFT_LABEL in _all_text(at)


def test_alignment_chart_traces_and_shading(ran):
    from ili.align_welds import load_weld_map
    from ili.app_pages_setup import alignment_chart
    import pandas as pd
    _, folder = ran
    fig = alignment_chart(load_weld_map(folder / "weld_map.csv"), pd.read_csv(folder / "changed_sections.csv"))
    assert {t.name for t in fig.data} >= {"1:1"}
    assert len(fig.layout.shapes) == 7


def test_comparison_loads_setup_from_disk_after_restart(ran, tmp_path):
    inputs, folder = ran
    assert app_state.latest_comparison(inputs.out) == folder
    assert app_state.comparison_from_session({}, inputs.out) == folder
    assert app_state.inputs_from_session(folder, {}) == inputs
    assert app_state.comparison_from_session({"comparison_dir": str(tmp_path)}, inputs.out) == tmp_path
    assert app_state.latest_comparison(tmp_path) is None


def test_load_comparison_reads_outputs_and_refreshes_on_change(ran, tmp_path):
    import shutil
    _, folder = ran
    copy = tmp_path / FOLDER
    shutil.copytree(folder, copy)
    data = app_state.load_comparison(copy)
    assert len(data.changed_sections) == 7 and data.alignment_summary["n_sections"] == 11
    assert not data.match_table.empty and not data.classification.empty and data.reviews.empty
    (copy / "alignment_summary.json").write_text('{"n_sections": 1}', encoding="utf-8")
    assert app_state.load_comparison(copy).alignment_summary == {"n_sections": 1}
    assert app_state.read_comparison(tmp_path / "nothing").weld_map.empty


def test_streamlit_config_is_local():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["server"]["address"] == "localhost"
    assert cfg["server"]["headless"] is True
