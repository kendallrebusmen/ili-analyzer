import io
import re
from datetime import datetime, timezone

import pytest
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics

from ili.category_change import category_changes
from ili.report import ReportHeader, Table, report_context
from ili.report_pdf import (BOLD, CELL_PAD, FONT, REPAIR_CATEGORIES, TABLE_FONT_SIZE, _column_widths, _dig_rows,
                            _display, _rounded, build_pdf)
from test_report_inputs import _load, built  # noqa: F401  (built is the module fixture)

GENERATED = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
HOSTILE_LINE = "<b>Line & 7</b>"
CLIENT, AUTHOR = "Example Pipeline Operator LLC", "Integrity Engineering Group"
SECTIONS = ("Contents", "1 Executive summary", "2 Pipeline and inspections", "3 Repair conditions",
            "4 Corrosion growth and remaining life", "5 Anomaly distribution", "6 Data quality and review status",
            "Appendix A Methods and assumptions", "Appendix B Input files",
            "Appendix C Anomalies scheduled by remaining life")


@pytest.fixture(scope="module")
def pdf(built):  # noqa: F811
    """(context, page texts) for the 5-mile synth, built once with a hostile line name and cover names."""
    inputs = _load(built)
    changes = category_changes(inputs.match_table, inputs.remaining_life, inputs.classification, inputs.curr_std,
                               inputs.pipe, inputs.mop, inputs.segments, inputs.crossings, inputs.settings,
                               inputs.criteria, inputs.prev_std.run_date.dropna().iloc[0])
    header = ReportHeader(HOSTILE_LINE, "MFL", "MFL + caliper", prepared_for=CLIENT, prepared_by=AUTHOR)
    context = report_context(inputs, changes, header, GENERATED)
    data = build_pdf(context, inputs, GENERATED)
    assert data.startswith(b"%PDF")
    pages = [re.sub(r"\s+", " ", p.extract_text()) for p in PdfReader(io.BytesIO(data)).pages]
    return context, pages


def test_every_page_carries_the_draft_label_and_a_number(pdf):
    _, pages = pdf
    assert len(pages) >= 10
    for n, text in enumerate(pages, start=1):
        assert "DRAFT" in text and "for review by qualified personnel" in text, f"page {n}"
        if n > 1:  # the cover carries no number
            assert f"Page {n} of {len(pages)}" in text, f"page {n}"


def test_cover_carries_the_nemi_brand(pdf):
    _, pages = pdf
    assert "NEMI · INTEGRITY ASSESSMENT REPORT" in pages[0] and "Nemi 0.1.0" in pages[0]
    assert all("DRAFT" in p for p in pages)


def test_sections_in_order_and_in_the_contents(pdf):
    _, pages = pdf
    text = " ".join(pages)
    positions = [text.find(s, text.find("Contents") + 1 if s != "Contents" else 0) for s in SECTIONS]
    assert all(p >= 0 for p in positions), dict(zip(SECTIONS, positions))
    contents = pages[1]
    assert all(s in contents for s in SECTIONS[1:]), contents  # the contents fit on one page


def test_cover_names_the_line_client_and_author_as_text(pdf):
    _, pages = pdf
    cover = pages[0]
    assert HOSTILE_LINE in cover  # escaped: shown as text, not taken as markup
    assert CLIENT in cover and AUTHOR in cover
    assert "In-Line Inspection Run Comparison" in cover


def test_never_marked_approved(pdf):
    _, pages = pdf
    assert not any("approved" in p.lower() for p in pages)


def test_repair_conditions_hold_only_repair_categories(pdf):
    context, _ = pdf
    dig = context["dig_list"]
    repair = _dig_rows(dig, REPAIR_CATEGORIES)
    scheduled = _dig_rows(dig, ("imp_scheduled",))
    assert {c.removeprefix("cat-") for c in repair.row_classes} <= set(REPAIR_CATEGORIES)
    assert {c.removeprefix("cat-") for c in scheduled.row_classes} <= {"imp_scheduled"}
    assert len(repair.rows) + len(scheduled.rows) == len(dig.rows)


def test_display_rounds_and_words_values():
    assert _display("curr_log_dist_ft", "22590.715") == "22,590.7"
    assert _display("p_fail", "1393.277") == "1,393"
    assert _display("governing_category", "60_day") == "60-day"
    assert _display("requires_review", "True") == "Yes"
    assert _display("feature_type", "metal_loss") == "metal loss"
    assert _display("triggered_rules", "IMM-1;D180-4") == "IMM-1; D180-4"
    assert _display("p_fail", "") == ""
    assert _display("curr_feature_id", "B-000009") == "B-000009"  # no format for ids


def test_rounded_rounds_every_number_in_text():
    assert _rounded("1468.235 to 1832.358", 0) == "1,468 to 1,832"
    assert _rounded("not available", 1) == "not available"


def test_column_widths_fill_the_frame_without_breaking_words():
    headings = ["Category", "Feature", "Description"]
    rows = [["Immediate", "B-000009", "a long free-text description " * 4]]
    widths = _column_widths(headings, rows, 400.0)
    assert sum(widths) == pytest.approx(400.0)
    for j, h in enumerate(headings):
        longest_word = max(pdfmetrics.stringWidth(w, BOLD, TABLE_FONT_SIZE) for w in [h, *rows[0][j].split()])
        assert widths[j] >= longest_word + 2 * CELL_PAD


def test_empty_table_prints_none(pdf):
    from ili.report_pdf import _styles, _table
    out = _table(_styles(), Table(["a"], []), 200.0)
    assert out[0].getPlainText() == "None."


def test_fonts_come_from_matplotlib(pdf):
    assert {FONT, BOLD} <= set(pdfmetrics.getRegisteredFontNames())
