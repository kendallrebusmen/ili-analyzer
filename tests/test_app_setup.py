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
SCRIPT_CONFIG = ROOT / "app" / ".streamlit" / "config.toml"


def _set_options(options: dict, where: str) -> dict:
    from streamlit import config
    old = {k: config.get_option(k) for k in options}
    for k, v in options.items():
        config._set_option(k, v, where)
    return old


def _flatten(table, prefix=""):
    """{"theme": {"light": {"base": "light"}}} -> {"theme.light.base": "light"}: nested sections like [theme.light]."""
    flat = {}
    for key, value in table.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.fixture(autouse=True)
def script_config():
    """The options of app/.streamlit/config.toml, as ``streamlit run app/streamlit_app.py`` applies them
    (AppTest reads no config file); the app refuses to run without them."""
    cfg = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))
    old = _set_options(_flatten(cfg), str(SCRIPT_CONFIG))
    yield
    _set_options(old, "test restore")


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """No test sees the real outputs/ folder: an earlier comparison there would pre-fill the Run setup form."""
    empty = tmp_path / "no_outputs"
    empty.mkdir()
    monkeypatch.setattr(app_state, "OUTPUTS_DIR", empty)


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


def _files_app():
    """The app with every pipeline-data table set to "Choose a file" (an empty field otherwise opens on typing)."""
    from ili.app_pages_setup import REFERENCE_OPTIONAL, REFERENCE_REQUIRED, USE_FILE
    return _app(**{f"setup_{n}_mode": USE_FILE for n in (*REFERENCE_REQUIRED, *REFERENCE_OPTIONAL)})


def _all_text(at):
    return " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.title])


def test_app_starts_with_draft_label():
    at = _app().run()
    assert not at.exception
    assert any("nemi-badge" in m.value and DRAFT_LABEL in m.value for m in at.markdown)
    assert at.text_input(key="reviewer").value == ""


def test_run_setup_runs_pipeline(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out")
    at = _files_app().run()
    for name, value in vars(inputs).items():
        if value is not None:
            at.text_input(key=f"setup_{name}").set_value(str(value))
    at.button[0].click().run()
    assert not at.exception
    labels = [e.label for e in at.expander if ": exit " in e.label]  # the stage results
    assert labels == [f"{s}: exit 0" for s in STAGES]
    assert not at.error
    assert at.session_state[app_state.COMPARISON_KEY] == str(inputs.out / FOLDER)
    assert (inputs.out / FOLDER / "report.html").is_file()
    # the sidebar names the new folder, with its full path as the tooltip
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert FOLDER in side and str(inputs.out / FOLDER) in side


def test_run_setup_shows_failing_stderr(synth, tmp_path):
    _, data = synth
    inputs = _inputs(data, tmp_path / "out", pipe=data / "reference" / "missing.csv")
    at = _files_app().run()
    for name, value in vars(inputs).items():
        if value is not None:
            at.text_input(key=f"setup_{name}").set_value(str(value))
    at.button[0].click().run()
    assert at.error and "match failed" in at.error[0].value and "missing.csv" in at.error[0].value
    assert any("missing.csv" in c.value for c in at.code[-1:])  # the full stderr, under the message


def test_run_setup_requires_inputs():
    at = _files_app().run()
    assert any("Still needed: previous run tally" in c.value for c in at.caption)
    at.button[0].click().run()
    assert at.error and "Still needed: previous run tally" in at.error[0].value
    assert "Pipe properties" in at.error[0].value


def test_every_input_has_one_place_on_the_page():
    from dataclasses import fields as dc_fields
    from ili import app_pages_setup as page
    from ili.pipeline import PipelineInputs
    placed = [*(f"{r}_{x}" for r in ("prev", "curr") for x in ("tally", "map", "run_date")),
              *page.REFERENCE_REQUIRED, *page.REFERENCE_OPTIONAL, "digs", *page.REPORT_FIELDS, *page.ADVANCED_FIELDS]
    assert sorted(placed) == sorted(f.name for f in dc_fields(PipelineInputs))


def test_empty_fields_open_on_typing_and_set_ones_on_the_file(tmp_path):
    from ili.app_pages_setup import TYPE_IN, USE_FILE
    at = _app().run()
    assert not at.exception
    assert at.session_state["setup_pipe_mode"] == TYPE_IN
    assert at.session_state["setup_out"]  # the output folder has a default
    at = _app(setup_pipe=str(tmp_path / "pipe.csv")).run()
    assert at.session_state["setup_pipe_mode"] == USE_FILE


def test_typed_tables_are_saved_and_run(synth, tmp_path):
    """Pipeline data typed in (seeded here from the synthetic files) is saved under <out>/entered/ and runs."""
    import pandas as pd
    from ili.app_pages_setup import REFERENCE_REQUIRED, TYPE_IN, USE_FILE
    from ili.app_setup_tables import TABLES, seed_table
    _, data = synth
    inputs = _inputs(data, tmp_path / "out")
    state = {f"setup_{n}_mode": TYPE_IN for n in REFERENCE_REQUIRED}
    state |= {f"setup_{n}_seed": seed_table(n, str(getattr(inputs, n))) for n in REFERENCE_REQUIRED}
    state |= {f"setup_{n}_mode": USE_FILE for n in ("repairs", "crossings")}
    at = _app(**state).run()
    for name in ("prev_tally", "prev_map", "curr_tally", "curr_map", "prev_run_date", "out", "repairs"):
        value = getattr(inputs, name)
        if value is not None:
            at.text_input(key=f"setup_{name}").set_value(str(value))
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert [e.label for e in at.expander if ": exit " in e.label] == [f"{s}: exit 0" for s in STAGES]
    ran = at.session_state[app_state.INPUTS_KEY]
    for name in REFERENCE_REQUIRED:
        saved = inputs.out / "entered" / TABLES[name].file_name
        assert getattr(ran, name) == saved and saved.is_file()
    original = pd.read_csv(inputs.pipe)
    assert pd.read_csv(ran.pipe).install_year.tolist() == original.install_year.tolist()


def test_typed_required_table_without_rows_is_an_error(tmp_path):
    at = _app(setup_out=str(tmp_path / "out")).run()  # empty fields: the pipeline data opens on typing
    for name in ("prev_tally", "prev_map", "curr_tally", "curr_map"):
        at.text_input(key=f"setup_{name}").set_value(str(tmp_path / f"{name}.csv"))
    at.button[0].click().run()
    assert at.error and "Pipe properties: type at least one row" in at.error[0].value


def test_weld_alignment_page_shows_changed_sections(ran):
    _, folder = ran
    at = _app(comparison_dir=str(folder)).run()
    at.radio(key="page").set_value("Weld alignment").run()
    assert not at.exception
    assert [m.label for m in at.metric][-1] == "Changed sections"
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 7
    assert DRAFT_LABEL in _all_text(at)


def test_pages_without_data_show_neutral_text_and_draft():
    at = _app().run()
    for page in PAGES[1:]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception
        assert [i.value for i in at.info] == ["No data yet: run the pipeline on the Run setup page."]
        assert DRAFT_LABEL in _all_text(at)


def test_pages_with_data_show_tables_and_draft(ran):
    _, folder = ran
    at = _app(comparison_dir=str(folder)).run()
    for page in PAGES[2:]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception and not at.info and not at.error
        assert len(at.dataframe) >= (2 if page == PAGES[4] else 1)
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
    # script-level config: streamlit reads it beside the script whatever the launch directory
    cfg = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["server"]["address"] == "localhost"
    assert cfg["server"]["headless"] is True
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "streamlit run app/streamlit_app.py --server.address localhost --browser.gatherUsageStats false" in readme


@pytest.mark.parametrize("option, value", [("server.address", "0.0.0.0"), ("server.address", None),
                                           ("browser.gatherUsageStats", True)])
def test_app_refuses_to_run_when_not_local(option, value):
    old = _set_options({option: value}, "test")
    try:
        at = _app().run()
    finally:
        _set_options(old, "test restore")
    assert not at.exception
    assert len(at.error) == 1 and "refuses to start" in at.error[0].value and option in at.error[0].value
    assert any("nemi-badge" in m.value and DRAFT_LABEL in m.value for m in at.markdown)
    assert DRAFT_LABEL in _all_text(at)
    assert not at.text_input and not at.radio  # no reviewer field, no page


def test_unreadable_log_is_an_error_on_every_page_not_a_traceback(ran, tmp_path):
    import shutil
    from ili.review import REVIEWS_FILE, REVIEW_COLUMNS
    _, folder = ran
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    (work / REVIEWS_FILE).write_text(",".join(REVIEW_COLUMNS) + "\nR000001,approve,A,B,Pat,2026-10-27T12:00:00Z,\n",
                                     encoding="utf-8")
    at = _app(comparison_dir=str(work)).run()
    for page in PAGES[1:]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception, page
        assert len(at.error) == 1 and "reviews.csv: line 2" in at.error[0].value, page
        assert DRAFT_LABEL in _all_text(at)


def test_rerun_setup_is_the_shown_folders(ran, tmp_path):
    """A setup run in this session for one folder is never used to rerun another folder (M7)."""
    import dataclasses
    import shutil
    from ili.pipeline import write_pipeline_inputs
    inputs, folder = ran
    other = tmp_path / FOLDER
    shutil.copytree(folder, other)
    recorded = dataclasses.replace(inputs, line_name="the other folder's setup")
    write_pipeline_inputs(recorded, other)
    session = {app_state.INPUTS_KEY: inputs}
    assert app_state.inputs_from_session(folder, session) is inputs
    assert app_state.inputs_from_session(other, session) == recorded
    (other / "pipeline_inputs.json").unlink()
    assert app_state.inputs_from_session(other, session) is None
    assert app_state.inputs_from_session(None, session) is inputs
