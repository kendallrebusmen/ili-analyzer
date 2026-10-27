import dataclasses
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from ili.align_welds import DRAFT_LABEL
from ili.category_change import category_changes
from ili.classify import IMMEDIATE_ACTION_NOTE
from ili.report import SECTION_IDS, ReportHeader, _css_string, cell, render_report
from ili.report_data import backtest, pending_reviews
from test_report_inputs import _load, built  # noqa: F401  (built is the module fixture)

GENERATED = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
HOSTILE = "<script>alert(1)</script>"
HOSTILE_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"


@pytest.fixture(scope="module")
def report(built):  # noqa: F811
    """(inputs, changes, html) for the 5-mile synth, rendered once."""
    inputs = _load(built)
    changes = category_changes(inputs.match_table, inputs.remaining_life, inputs.classification, inputs.curr_std,
                               inputs.pipe, inputs.mop, inputs.segments, inputs.crossings, inputs.settings,
                               inputs.criteria, inputs.prev_std.run_date.dropna().iloc[0])
    html = render_report(inputs, changes, ReportHeader(None, None, None), GENERATED)
    return inputs, changes, html


class _Sections(HTMLParser):
    """Each ``<section>``'s id in order, its first child element (tag, class, text) and its visible text;
    plus all visible text outside ``<script>`` and ``<style>``."""

    def __init__(self):
        super().__init__()
        self.ids, self.first_child, self.text = [], {}, {}
        self.all_text = []
        self._current = self._child = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "section":
            self._current = attrs.get("id")
            self.ids.append(self._current)
            self.text[self._current] = []
            self._child = None
        elif self._current is not None and self._current not in self.first_child:
            self.first_child[self._current] = [tag, attrs.get("class", ""), ""]
            self._child = self._current

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        if tag == "section":
            self._current = None
        self._child = None

    def handle_data(self, data):
        if self._skip:
            return
        self.all_text.append(data)
        if self._current is not None:
            self.text[self._current].append(data)
            if self._child is not None:
                self.first_child[self._child][2] += data


def _parse(html):
    p = _Sections()
    p.feed(html)
    p.close()
    return p


def _section(html, section_id):
    """The raw html of one section."""
    start = html.index(f'<section id="{section_id}"')
    return html[start:html.index("</section>", start)]


def _section_text(html, section_id):
    return " ".join(" ".join(_parse(html).text[section_id]).split())


def test_every_section_present_and_labelled(report):
    _, _, html = report
    p = _parse(html)
    assert p.ids == list(SECTION_IDS)
    assert SECTION_IDS == ("header", "summary", "data-quality", "growth", "dig-list", "largest-changes", "charts",
                           "methods")
    for sid in SECTION_IDS:
        tag, cls, text = p.first_child[sid]
        assert "draft" in cls.split(), sid
        assert text.strip() == DRAFT_LABEL, sid
    assert re.search(r'class="draft-banner"[^>]*>\s*' + re.escape(DRAFT_LABEL), html)
    # print: the label sits in the page's top margin box on every page; the screen banner is hidden
    assert re.search(r'@page\s*\{[^@}]*@top-left\s*\{\s*content:\s*"' + re.escape(DRAFT_LABEL) + '"', html)
    assert re.search(r"@media print\s*\{[^}]*\.draft-banner\s*\{\s*display:\s*none;", html)
    assert "position: fixed" not in _style(html)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def test_css_string_refuses_unsafe_label():
    assert _css_string(DRAFT_LABEL) == DRAFT_LABEL
    for bad in ('say "x"', "a\\b", "a</style>", "a & b"):
        with pytest.raises(ValueError, match="margin box"):
            _css_string(bad)


def test_header(report):
    inputs, _, html = report
    text = _section_text(html, "header")
    for value in ("2021_vendorA", "2026_vendorB", "2021-06-15", "2026-06-15", "synthetic_vendor_a",
                  "synthetic_vendor_b"):
        assert value in text, value
    assert text.count("not stated") == 3  # line name and both tool types
    named = render_report(inputs, report[1], ReportHeader("Line 7", "MFL-A", "MFL-B"), GENERATED)
    text = _section_text(named, "header")
    assert "not stated" not in text and all(v in text for v in ("Line 7", "MFL-A", "MFL-B"))


def test_self_contained(report):
    _, _, html = report
    urls = re.findall(r'\b(?:src|href)\s*=\s*["\']?([^"\'\s>]*)', _outside_scripts(html), flags=re.I)
    assert not [u for u in urls if u.lower().startswith(("http", "//", "file:"))]
    assert html.count("plotly.js v") == 1
    assert not re.search(r"<link[^>]*stylesheet", html, flags=re.I)
    assert not re.search(r"<script[^>]*\bsrc\s*=", html, flags=re.I)
    assert html.count('class="plotly-graph-div"') == 6  # histograms + depth, growth, RL, map, unity


def _outside_scripts(html):
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)


def test_hostile_text_is_escaped(report):
    inputs, changes, _ = report
    cls = inputs.classification.copy()
    row = cls.index[cls.governing_category == "immediate"][0]
    cls.loc[row, "curr_feature_id"] = HOSTILE
    issue = {"run_id": "2026_vendorB", "source_row": 7, "feature_id": "B-000007", "field": "depth_pct",
             "value": HOSTILE, "reason": "out_of_range"}
    issues = pd.DataFrame([issue], columns=inputs.validation_issues.columns)
    hostile = dataclasses.replace(inputs, classification=cls, validation_issues=issues)
    html = render_report(hostile, changes, ReportHeader(HOSTILE, None, None), GENERATED)
    assert HOSTILE not in html
    for sid in ("header", "summary", "data-quality", "dig-list"):
        assert HOSTILE_ESCAPED in _section(html, sid), sid
        assert HOSTILE in _section_text(html, sid), sid  # shown as text, not parsed as a tag


def test_pending_warning_and_never_final(report):
    inputs, changes, html = report
    n = len(pending_reviews(inputs.match_table))
    assert n > 0
    warning = f"{n} match reviews pending — this report cannot be final."
    assert warning in _section_text(html, "data-quality")
    accepted = dataclasses.replace(inputs, match_table=inputs.match_table.assign(review_status="accepted"))
    clean = render_report(accepted, changes, ReportHeader(None, None, None), GENERATED)
    assert "match reviews pending" not in clean and "cannot be final" not in clean
    for page in (html, clean):
        p = _parse(page)
        assert DRAFT_LABEL in p.first_child["summary"][2]
        visible = " ".join(p.all_text)
        assert re.search(r'class="draft-banner"[^>]*>\s*' + re.escape(DRAFT_LABEL), page)
        assert not re.search(r"\bFINAL\b", visible)
        assert not re.search(r"status\W*(final|approved)", visible, flags=re.I)


def test_no_immediate_message(report):
    inputs, changes, html = report
    summary = _section_text(html, "summary")
    assert "No immediate conditions." not in summary
    assert IMMEDIATE_ACTION_NOTE in summary
    cls = inputs.classification.copy()
    cls.loc[cls.governing_category == "immediate", "governing_category"] = "180_day"
    none = render_report(dataclasses.replace(inputs, classification=cls), changes, ReportHeader(None, None, None),
                         GENERATED)
    assert "No immediate conditions." in _section_text(none, "summary")


def test_blank_cells_not_nan(report):
    _, _, html = report
    for bad in (">nan<", ">None<", ">NaT<", ">&lt;NA&gt;<", ">NaN<"):
        assert bad not in html, bad
    assert "<td></td>" in _section(html, "dig-list")  # blanks exist, and render empty


def test_dig_list_rows_carry_category_class(report):
    _, _, html = report
    dig = _section(html, "dig-list")
    assert re.search(r'<tr class="cat-immediate">', dig)
    assert re.search(r'<tr class="cat-imp_scheduled">', dig)
    assert 'class="cat-evaluate"' not in dig


def test_missing_optional_notes(report):
    inputs, changes, _ = report
    paths = {k: v for k, v in inputs.paths.items() if k not in ("digs", "crossings")}
    bare = dataclasses.replace(inputs, digs=None, crossings=None, paths=paths,
                               curr_std=inputs.curr_std.assign(lat=np.nan, lon=np.nan))
    html = render_report(bare, changes, ReportHeader(None, None, None), GENERATED)
    charts = _section_text(html, "charts")
    assert "digs.csv not provided" in charts
    assert "No map: no dig-list feature has a latitude and longitude." in charts
    assert "Foreign crossings file not provided" in _section_text(html, "data-quality")
    assert html.count('class="plotly-graph-div"') == 4


def test_placeholder_rows_highlighted(report):
    _, _, html = report
    methods = _section(html, "methods")
    assert re.search(r'<tr class="placeholder"><td>matching\.interaction_rule\.axial_spacing_in</td>', methods)
    assert re.search(r"<tr><td>units</td>", methods)  # not a placeholder: no class


def test_unity_sentence_and_unmatched_ids(report):
    inputs, changes, html = report
    tolerance = cell(inputs.settings.tool.depth_tolerance_pct_wt)
    extra = pd.DataFrame([{"feature_id": "B-999999", "field_depth_pct": 50.0, "field_length_in": 1.0,
                           "dig_date": "2027-01-01"}])
    digs = pd.concat([inputs.digs, extra], ignore_index=True)
    with_unknown = dataclasses.replace(inputs, digs=digs)
    bt = backtest(digs, inputs.curr_std, inputs.settings)
    assert bt.unmatched_ids == ["B-999999"] and len(bt.table) == len(inputs.digs)
    charts = _section_text(render_report(with_unknown, changes, ReportHeader(None, None, None), GENERATED), "charts")
    assert (f"{bt.n_within} of {len(bt.table)} digs ({cell(bt.pct_within)} %) have the reported current-run depth "
            f"within ±{tolerance} % wall of the field depth.") in charts
    assert "Dig feature ids not in the current run: B-999999." in charts
    assert "Every dig feature id is in the current run." in _section_text(html, "charts")


def test_empty_digs_file_is_not_missing(report):
    inputs, changes, _ = report
    empty = dataclasses.replace(inputs, digs=inputs.digs.iloc[0:0])
    charts = _section_text(render_report(empty, changes, ReportHeader(None, None, None), GENERATED), "charts")
    assert "No unity chart: digs.csv has no rows." in charts
    assert "not provided" not in charts
