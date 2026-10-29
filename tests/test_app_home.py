"""The Home overview: its pure view model and the page."""

import json
import shutil

import pytest

from ili import app_state, report_data
from ili.app_pages_home import Attention, HomeView, home_view
from test_app_setup import _all_text, _app, isolated_outputs, ran, script_config  # noqa: F401  (fixtures)


def _copy(folder, tmp_path):
    work = tmp_path / "folder"
    shutil.copytree(folder, work)
    return work


def test_home_view_unloaded():
    view = home_view(None, None)
    assert isinstance(view, HomeView) and view.loaded is False


def test_home_view_counts_match_summary(ran):
    _, folder = ran
    data = app_state.read_comparison(folder)
    view = home_view(data, folder)
    by_cat = json.loads((folder / "classification_summary.json").read_text(encoding="utf-8"))["counts_by_category"]
    rl = json.loads((folder / "remaining_life_summary.json").read_text(encoding="utf-8"))
    assert view.loaded
    assert {k: view.counts[k] for k in ("immediate", "60_day", "180_day", "imp_scheduled")} == \
        {k: by_cat[k] for k in ("immediate", "60_day", "180_day", "imp_scheduled")}
    assert view.min_rl_yr == rl["min_rl_yr"] and view.min_rl_factored_yr == rl["min_rl_factored_yr"]
    labels = [label for label, _ in view.facts]
    assert "Growth method" in labels and "Depth basis" in labels


def test_home_view_pending_item(ran):
    _, folder = ran
    data = app_state.read_comparison(folder)
    n = len(report_data.pending_reviews(data.match_table))
    assert n > 0
    first = home_view(data, folder).attention[0]
    assert isinstance(first, Attention)
    assert first.level == "warn" and first.target == "Match review queue" and str(n) in first.title


def test_home_view_old_folder(ran, tmp_path):
    _, folder = ran
    work = _copy(folder, tmp_path)
    (work / "remaining_life_summary.json").unlink()
    (work / "classification_summary.json").unlink()
    view = home_view(app_state.read_comparison(work), work)
    assert view.counts["immediate"] is None and view.min_rl_yr is None
    assert not any(label == "Depth basis" for label, _ in view.facts)


def test_home_page_renders_without_comparison():
    at = _app().run()
    assert not at.exception
    assert at.radio(key="page").value == "Home"
    assert [h.value for h in at.subheader] == ["No comparison loaded yet"]
    at.button[0].click().run()
    assert at.radio(key="page").value == "Run setup"


def test_home_page_renders_loaded(ran, tmp_path):
    _, folder = ran
    work = _copy(folder, tmp_path)
    at = _app(comparison_dir=str(work)).run()
    assert not at.exception
    assert at.radio(key="page").value == "Home"
    assert any("nemi-kpi" in m.value for m in at.markdown)
    assert any("nemi-facts" in m.value for m in at.markdown)
    go = [b for b in at.button if (b.key or "").startswith("attn_")]
    assert go
    go[0].click().run()
    assert not at.exception and at.radio(key="page").value in {"Match review queue", "Run setup",
                                                              "Dig list"}


@pytest.mark.parametrize("n, title", [(1, "1 match waiting for review"), (2, "2 matches waiting for review")])
def test_attention_titles_use_the_right_plural(ran, monkeypatch, n, title):
    from ili import app_pages_home
    _, folder = ran
    data = app_state.read_comparison(folder)
    monkeypatch.setattr(app_pages_home, "pending_reviews", lambda table: list(range(n)))
    assert app_pages_home._attention(data)[0].title == title


def test_plural_helper_in_counted_titles():
    from ili.app_pages_home import _plural
    assert _plural(1, "flag", "flags") == "1 flag" and _plural(2, "anomaly", "anomalies") == "2 anomalies"
