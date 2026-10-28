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
from ili.report import (DIG_LIST_PRINT_COLUMNS, MOVED_PRINT_COLUMNS, SECTION_IDS, ReportHeader, _css_string, _vendor,
                        cell, render_report)
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
    assert html.count('class="plotly-graph-div"') == 10  # histograms + depth, growth, RL, 3 orientation, elevation, map, unity


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
    warning = f"{n} matches pending review — this report cannot be final."
    assert warning in _section_text(html, "data-quality")
    accepted = dataclasses.replace(inputs, match_table=inputs.match_table.assign(review_status="accepted"))
    clean = render_report(accepted, changes, ReportHeader(None, None, None), GENERATED)
    assert "pending review —" not in clean and "cannot be final" not in clean
    assert "Pending: 0 matches pending review." in clean
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
    html = html.replace("<tr><td>None</td>", "<tr>")  # the IMP class named "None" (spec 6.8) is a label, not a blank
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
    assert html.count('class="plotly-graph-div"') == 8


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


def _headers(section_html):
    """``{column name: th class}`` of the section's first table."""
    head = section_html[section_html.index("<thead>"):section_html.index("</thead>")]
    return {name: cls for cls, name in re.findall(r'<th(?: class="([^"]*)")?>([^<]*)</th>', head)}


def test_print_shows_a_compact_column_subset(report):
    _, _, html = report
    style = _style(html)
    print_css = style[style.index("@media print"):]
    assert re.search(r"\.noprint\s*\{\s*display:\s*none;", print_css)
    assert re.search(r"\.print-only\s*\{\s*display:\s*block;", print_css)
    assert re.search(r"(?<!\S)\.print-only\s*\{\s*display:\s*none;", style[:style.index("@media print")])
    dig = _section(html, "dig-list")
    headers = _headers(dig)
    shown = [c for c, cls in headers.items() if "noprint" not in cls.split()]
    assert shown == list(DIG_LIST_PRINT_COLUMNS)
    assert len(shown) == 12 and "noprint" in headers["lat"] and "noprint" in headers["review_reasons"]
    assert re.search(r'<td class="noprint">', dig)  # body cells hidden with their header
    assert "dig_list.xlsx" in _section_text(html, "dig-list")
    moved = _section(html, "largest-changes")
    moved = moved[moved.index("Moved to a more severe category"):]
    shown = [c for c, cls in _headers(moved).items() if "noprint" not in cls.split()]
    assert shown == list(MOVED_PRINT_COLUMNS) and "category_changes.csv" in moved


def test_moved_table_shows_current_rules_not_evaluable(report):
    _, changes, html = report
    assert list(changes.table.columns).index("curr_rules_not_evaluable") == \
        list(changes.table.columns).index("curr_governing_rule") + 1
    moved = _section(html, "largest-changes")
    assert "curr_rules_not_evaluable" in _headers(moved[moved.index("Moved to a more severe category"):])


def test_methods_caption_says_which_settings_are_verified(report):
    _, _, html = report
    text = _section_text(html, "methods")
    assert ("Strength, remaining-life and classification values, and the tool tolerances they use, were verified "
            "against the M6 and M7 summaries; alignment, matching and growth values (and the other tool values) are "
            "the settings at report time, as no summary records them.") in text


def test_methods_lists_new_formulas(report):
    _, _, html = report
    text = _section_text(html, "methods")
    for method in ("half_life", "joint_percentile", "shrinking_fixed"):
        assert method in text
    assert "conservative" in text and "pit_to_pit" in text
    assert "d_curr / ((current run year − install_year) / 2)" in text
    assert "joint_percentile_min_matches" in text
    assert "Δd ≥ 0" in text and "Δd < 0" in text
    assert "ERF = MOP_local / P_safe" in text and "FPR = P_F / MOP_local" in text
    assert "leak when τ_80 ≤ τ_burst, otherwise burst" in text
    assert "blank when τ_80 or τ_burst is NaN, or both are inf" in text
    assert "blank where P_safe or MOP is not a finite positive number" in text
    assert "P_design = 2 · SMYS · t · design_factor / D" in text
    assert "MOP · D / (2 · t · SMYS)" in text


def test_methods_states_the_next_assessment_projection_rules(report):
    _, _, html = report
    text = _section_text(html, "methods")
    assert "exceeds_80pct_next when depth(τ_next) ≥ 0.80·t" in text
    assert ("burst_next_psig, safe_next_psig, erf_next and fpr_next are blank where depth(τ_next) ≥ t (projected "
            "through the wall)") in text


def test_missing_pct_negative_raw_is_not_available(report):
    inputs, changes, html = report
    assert re.search(r"Negative raw growth rates: [0-9.]+ % of matched features", _section_text(html, "growth"))
    growth = {k: v for k, v in inputs.summaries["growth"].items() if k != "pct_negative_raw"}
    bare = dataclasses.replace(inputs, summaries={**inputs.summaries, "growth": growth})
    text = _section_text(render_report(bare, changes, ReportHeader(None, None, None), GENERATED), "growth")
    assert "Negative raw growth rates: not available (included in the segment percentiles)." in text
    assert "of matched features" not in text


def test_backtest_digs_without_depth_listed(report):
    inputs, changes, _ = report
    first = inputs.digs.feature_id.iloc[0]
    digs = inputs.digs.assign(field_depth_pct=inputs.digs.field_depth_pct.where(inputs.digs.feature_id != first))
    bt = backtest(digs, inputs.curr_std, inputs.settings)
    assert bt.no_depth_ids == [first] and len(bt.table) == len(inputs.digs) - 1
    html = render_report(dataclasses.replace(inputs, digs=digs), changes, ReportHeader(None, None, None), GENERATED)
    charts = _section_text(html, "charts")
    assert f"{bt.n_within} of {len(bt.table)} digs" in charts
    assert f"Digs without both a reported and a field depth (not compared): {first}." in charts
    none = digs.assign(field_depth_pct=np.nan)
    charts = _section_text(render_report(dataclasses.replace(inputs, digs=none), changes,
                                         ReportHeader(None, None, None), GENERATED), "charts")
    assert "No dig in the current run has both a reported and a field depth." in charts
    assert "No dig feature id is in the current run." not in charts


def test_vendor_map_relative_to_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = {"runs": [{"run_id": "R1", "vendor_map": "config/vendor_maps/synthetic_vendor_a.yaml"}]}
    assert _vendor(manifest, "R1") == "synthetic_vendor_a"
    assert _vendor({"runs": [{"run_id": "R1", "vendor_map": "config/vendor_maps/nope.yaml"}]}, "R1") == "not stated"


def test_report_html_has_imp_class_and_profile(report):
    inputs, _, html = report
    summary = _section_text(html, "summary")
    assert "IMP class summary" in summary and "YTC &lt; 5 yrs".replace("&lt;", "<") in summary
    assert "approved" not in summary.lower() and "final" not in summary.lower().replace("not final", "")
    growth = _section_text(html, "growth")
    assert "conservative" in growth and "Growth summary by surface" in growth
    assert "excluding_recoated" in growth and "rows inside recoat spans, whatever the setting" in growth


def _growth_text(inputs, changes, settings=None, **summary):
    """The growth section rendered with ``summary`` keys replacing growth_summary.json's (and other settings)."""
    bare = dataclasses.replace(inputs, summaries={**inputs.summaries, "growth": {**inputs.summaries["growth"],
                                                                                 **summary}},
                               settings=settings or inputs.settings)
    return _section_text(render_report(bare, changes, ReportHeader(None, None, None), GENERATED), "growth")


def test_recoat_note_sits_with_the_segment_statistics(report):
    inputs, changes, html = report
    text = _growth_text(inputs, changes, repairs_file_given=True, n_excluded_recoated=3)
    note = ("3 recoated feature(s) left out of the segment and joint percentiles and these statistics "
            "(exclude_recoated_from_stats: true).")
    assert note in text
    assert text.index("Growth summary by surface") < text.index(note) < text.index("Per-segment statistics")
    assert "% of matched features in the statistics (recoated left out when the setting is on)" in text
    off = inputs.settings.model_copy(update={"growth": inputs.settings.growth.model_copy(
        update={"exclude_recoated_from_stats": False})})
    text = _growth_text(inputs, changes, settings=off, repairs_file_given=True, n_excluded_recoated=0)
    assert ("No recoated feature was left out of the segment and joint percentiles and these statistics, because "
            "exclude_recoated_from_stats is false.") in text
    assert "No repairs file was given to the growth step" in _growth_text(inputs, changes, repairs_file_given=False)


def test_methods_say_what_the_joint_and_shrinking_rates_leave_out(report):
    _, _, html = report
    text = _section_text(html, "methods")
    assert ("recoated features (when exclude_recoated_from_stats is true) and non-finite raw rates are left out"
            in text)
    assert "empty when shrinking_rate_mpy is null (under pit_to_pit the row is then no_governing)" in text


def test_growth_section_shows_depth_and_dent_bins_with_notes(report):
    inputs, _, html = report
    growth = _section_text(html, "growth")
    assert "Depth bins" in growth and "Dent bins" in growth
    for title in ("all metal loss", "metal loss at a girth weld", "metal loss on the long seam"):
        assert f"Depth bins, {title}" in growth
    assert ">=80" in growth and "0-2" in growth
    assert "classification.girth_weld_proximity_in is not set" in growth  # the default leaves the setting null


def test_report_header_has_pipe_summary(report):
    inputs, _, html = report
    text = _section_text(html, "header")
    assert "Pipe summary" in text and "pipe_properties.csv" in text and "mop_profile.csv" in text
    assert f"design factor F = {inputs.settings.strength.design_factor:g}" in text
    assert "Length check" in text and "Design pressure (psig)" in text and "Stress at MOP (% SMYS)" in text
    for run_id in (inputs.prev_std.run_id.iloc[0], inputs.curr_std.run_id.iloc[0]):
        assert run_id in text


def test_report_html_has_orientation_and_elevation(report):
    inputs, changes, html = report
    charts = _section_text(html, "charts")
    for label in ("External corrosion: clock position", "Internal corrosion: clock position",
                  "Manufacturing: clock position"):
        assert label in html
    assert "Elevation profile with internal metal loss" in html
    assert "elevation is not available" not in charts
    bare = dataclasses.replace(inputs, curr_std=inputs.curr_std.assign(elev_ft=np.nan))
    text = _section_text(render_report(bare, changes, ReportHeader(None, None, None), GENERATED), "charts")
    assert "elevation is not available" in text
