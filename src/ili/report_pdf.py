"""The report as a print-ready PDF deliverable (spec 3: optional PDF export; ReportLab and matplotlib install as
pure wheels on Windows, with no system libraries).

``build_pdf`` renders the same section tables the HTML report renders (``report.report_context``), so the two
always show the same values; charts are drawn by ``report_pdf_charts`` from the report input tables. Nothing is
recomputed and nothing is fetched: the fonts (DejaVu Sans) ship with matplotlib.

The document is always DRAFT: the label is in the footer of every page, a watermark crosses every page, the cover
states it, and no wording marks the report final or approved. Every piece of text from the data (vendor ids, file
paths, the line and client names) is escaped before it reaches ReportLab's paragraph markup.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import io
import math
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from ili import __version__, report_pdf_charts
from ili.align_welds import DRAFT_LABEL
from ili.report_charts import CATEGORY_COLOURS
from ili.report_data import ReportInputs, backtest

# --- look -------------------------------------------------------------------------------------------------------

NAVY = colors.HexColor("#0B2545")
ACCENT = colors.HexColor("#0A6ED1")
INK = colors.HexColor("#1F2A37")
MUTED = colors.HexColor("#5B6675")
RULE = colors.HexColor("#C9D1DA")
ZEBRA = colors.HexColor("#F4F6F8")
HEAD_FILL = colors.HexColor("#E8EDF3")
DRAFT_COLOUR = colors.HexColor("#B45309")
WARN_FILL = colors.HexColor("#FFF4E5")
WARN_RULE = colors.HexColor("#F0B35A")
DANGER_FILL = colors.HexColor("#FDECEC")
DANGER_RULE = colors.HexColor("#D64545")
WATERMARK_ALPHA = 0.05

PAGE = letter
MARGIN = 0.75 * inch
HEADER_Y = 0.45 * inch  # header rule, from the top edge
FOOTER_Y = 0.45 * inch  # footer baseline, from the bottom edge
TABLE_FONT_SIZE = 7.2
FONT, BOLD, ITALIC, BOLD_ITALIC = "DejaVu", "DejaVu-Bold", "DejaVu-Oblique", "DejaVu-BoldOblique"
FONT_FILES = {FONT: "DejaVuSans.ttf", BOLD: "DejaVuSans-Bold.ttf", ITALIC: "DejaVuSans-Oblique.ttf",
              BOLD_ITALIC: "DejaVuSans-BoldOblique.ttf"}
TITLE = "In-Line Inspection Run Comparison"
NOT_STATED = "not stated"
MIN_COLUMN_IN = 0.42  # narrowest table column (in)
MAX_COLUMN_CHARS = 60  # longer cell text no longer widens its column (it wraps)
CELL_PAD = 3  # table cell left/right padding (pt)
ROW_TINT = 0.35  # share of the category colour kept in a printed dig-list row fill

HEADINGS = {  # printed headings for columns that appear in several tables
    "imp_class": "IMP class", "n_classified": "Classified", "n_previously_repaired": "Previously repaired",
    "n_unrepaired": "Unrepaired", "n_rl_not_evaluated": "RL not evaluated", "category": "Category",
    "count": "Count", "segment_id": "Segment", "could_affect_hca": "Could affect HCA",
    "n_features": "Features", "n_matched": "Matched", "min_rl_factored_yr": "Min RL × factor (yr)",
    "regulatory_max_yr": "Regulatory max (yr)", "recommended_interval_yr": "Recommended interval (yr)",
    "governing_match_id": "Governing match", "needs_review": "Review", "run_id": "Run", "bin": "Bin",
    "ext_corrosion": "External corrosion", "int_corrosion": "Internal corrosion", "manufacturing": "Manufacturing",
    "total": "Total", "n_location_unknown": "Location unknown", "surface": "Surface", "scope": "Scope",
    "wt_nom_in": "Wall (in)", "length_ft": "Length (ft)", "ili_length_ft": "ILI length (ft)",
    "reference_length_ft": "Reference length (ft)", "diff_ft": "Difference (ft)", "diff_pct": "Difference (%)",
}
CATEGORY_COLUMNS = ("governing_category", "category")
CATEGORY_LABELS = {"immediate": "Immediate", "60_day": "60-day", "180_day": "180-day",
                   "imp_scheduled": "IMP scheduled", "evaluate": "Evaluate"}
COLUMN_DECIMALS = {  # printed decimals; the CSV and Excel outputs keep full precision
    "curr_log_dist_ft": 1, "start_ft": 1, "end_ft": 1, "depth_pct_assessed": 1, "max_depth_pct": 1,
    "governing_mpy": 1, "p_fail": 0, "mop_local": 0, "erf": 2, "fpr": 2, "erf_next": 2, "rl_yr": 1,
    "min_rl_yr": 1, "min_rl_factored_yr": 1, "regulatory_max_yr": 1, "recommended_interval_yr": 1,
    "length_ft": 1, "ili_length_ft": 1, "reference_length_ft": 1, "diff_ft": 1, "diff_pct": 2,
}
YES_NO = {"True": "Yes", "False": "No"}  # flags as printed
WORD_COLUMNS = ("feature_type", "match_type", "history_class")  # vocabulary values printed as words
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
LIST_SEP = ";"  # joins cluster member ids and triggered rules in the tables
REPAIR_CATEGORIES = ("immediate", "60_day", "180_day")  # the dig-list rows printed in section 3

# Dig-list columns that fit a landscape page, in order, with their printed headings.
DIG_LIST_COLUMNS = {
    "governing_category": "Category", "governing_rule": "Rule", "curr_feature_id": "Feature",
    "curr_log_dist_ft": "Log dist. (ft)", "clock_min": "Clock (min)",
    "depth_pct_assessed": "Depth assessed (% WT)", "governing_mpy": "Growth (mpy)", "p_fail": "P_F (psig)",
    "mop_local": "MOP (psig)", "erf": "ERF", "fpr": "FPR", "erf_next": "ERF next", "rl_yr": "RL (yr)",
    "failure_mode": "Mode", "due_date": "Due date", "requires_review": "Review",
}
IMMEDIATE_COLUMNS = {"curr_feature_id": "Feature", "feature_type": "Type", "curr_log_dist_ft": "Log dist. (ft)",
                     "governing_rule": "Rule", "triggered_rules": "Triggered rules"}
AOI_COLUMNS = {"rank": "#", "start_ft": "Start (ft)", "end_ft": "End (ft)", "surfaces": "Surface",
               "max_depth_pct": "Max depth (% WT)", "n_anomalies": "n", "min_rl_yr": "Min RL (yr)", "hca": "HCA",
               "description": "Description"}


def _register_fonts() -> None:
    """DejaVu Sans from matplotlib's own data folder (installed with it), so no system font is needed."""
    if FONT in pdfmetrics.getRegisteredFontNames():
        return
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in FONT_FILES.items():
        path = folder / file
        if not path.is_file():
            raise ValueError(f"PDF font {file} not found in matplotlib's data folder {folder}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=ITALIC, boldItalic=BOLD_ITALIC)


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName=FONT, fontSize=9, leading=12.5, textColor=INK, spaceAfter=6)
    return {
        "body": base,
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=base, fontName=BOLD, fontSize=16, leading=20, textColor=NAVY,
                             spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base, fontName=BOLD, fontSize=11, leading=14, textColor=NAVY,
                             spaceBefore=12, spaceAfter=5, keepWithNext=1),
        "h2_cover": ParagraphStyle("h2_cover", parent=base, fontName=BOLD, fontSize=11, leading=14,
                                   textColor=NAVY, spaceBefore=12, spaceAfter=5, keepWithNext=1),
        "cell": ParagraphStyle("cell", parent=base, fontSize=TABLE_FONT_SIZE, leading=TABLE_FONT_SIZE * 1.25,
                               spaceAfter=0),
        "head": ParagraphStyle("head", parent=base, fontName=BOLD, fontSize=TABLE_FONT_SIZE,
                               leading=TABLE_FONT_SIZE * 1.25, textColor=NAVY, spaceAfter=0),
        "kpi_value": ParagraphStyle("kpi_value", parent=base, fontName=BOLD, fontSize=20, leading=24,
                                    textColor=NAVY, alignment=TA_CENTER, spaceAfter=0),
        "kpi_label": ParagraphStyle("kpi_label", parent=base, fontSize=7.5, leading=9.5, textColor=MUTED,
                                    alignment=TA_CENTER, spaceAfter=0),
        "toc1": ParagraphStyle("toc1", parent=base, fontName=BOLD, fontSize=10, leading=13, spaceBefore=6,
                               spaceAfter=1, leftIndent=0),
        "toc2": ParagraphStyle("toc2", parent=base, fontSize=8, leading=10.5, spaceAfter=0, leftIndent=14,
                               textColor=MUTED),
        "cover_label": ParagraphStyle("cover_label", parent=base, fontSize=8, leading=11, textColor=MUTED,
                                      alignment=TA_LEFT, spaceAfter=0),
        "cover_value": ParagraphStyle("cover_value", parent=base, fontName=BOLD, fontSize=10.5, leading=13.5,
                                      spaceAfter=0),
    }


def _text(value) -> str:
    """Data text made safe for ReportLab paragraph markup."""
    return escape("" if value is None else str(value))


def _humanize(column: str) -> str:
    text = column.replace("_", " ")
    return text[:1].upper() + text[1:]


def _rounded(text: str, decimals: int) -> str:
    """Every number in ``text`` (e.g. "1468.235 to 1832.358") to ``decimals`` places with thousands separators."""
    return NUMBER.sub(lambda m: f"{float(m.group()):,.{decimals}f}", text)


# --- document ---------------------------------------------------------------------------------------------------


class _NumberedCanvas(rl_canvas.Canvas):
    """Holds every page until the end so each footer can say "Page n of N" (the cover has no number)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages = []

    def showPage(self):  # noqa: N802  (ReportLab API)
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont(FONT, 7.5)
                self.setFillColor(MUTED)
                width = self._pagesize[0]
                self.drawRightString(width - MARGIN, FOOTER_Y, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


class _Doc(BaseDocTemplate):
    """Registers h1/h2 headings with the table of contents (needs ``multiBuild``)."""

    def afterFlowable(self, flowable):  # noqa: N802  (ReportLab API)
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))


def _watermark(canv, width: float, height: float) -> None:
    canv.saveState()
    canv.setFillColor(NAVY)
    canv.setFillAlpha(WATERMARK_ALPHA)
    canv.setFont(BOLD, 120)
    canv.translate(width / 2, height / 2)
    canv.rotate(35)
    canv.drawCentredString(0, -40, "DRAFT")
    canv.restoreState()


def _page_decorator(running_title: str, running_right: str):
    def draw(canv, doc):
        width, height = canv._pagesize
        _watermark(canv, width, height)
        canv.saveState()
        canv.setStrokeColor(NAVY)
        canv.setLineWidth(1.2)
        canv.line(MARGIN, height - HEADER_Y, width - MARGIN, height - HEADER_Y)
        canv.setFont(BOLD, 8)
        canv.setFillColor(NAVY)
        canv.drawString(MARGIN, height - HEADER_Y + 5, running_title)
        canv.setFont(FONT, 8)
        canv.setFillColor(MUTED)
        canv.drawRightString(width - MARGIN, height - HEADER_Y + 5, running_right)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN, FOOTER_Y + 10, width - MARGIN, FOOTER_Y + 10)
        canv.setFont(BOLD, 7.5)
        canv.setFillColor(DRAFT_COLOUR)
        canv.drawString(MARGIN, FOOTER_Y, DRAFT_LABEL)
        canv.restoreState()
    return draw


def _cover_decorator(head: dict):
    def draw(canv, doc):
        width, height = canv._pagesize
        _watermark(canv, width, height)
        band = 3.6 * inch
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, height - band, width, band, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, height - band - 6, width, 6, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont(FONT, 10)
        canv.drawString(MARGIN, height - 1.05 * inch, "INTEGRITY ASSESSMENT REPORT")
        canv.setFont(BOLD, 26)
        canv.drawString(MARGIN, height - 1.65 * inch, TITLE)
        canv.setFont(FONT, 13)
        canv.drawString(MARGIN, height - 2.1 * inch, f"{head['line_name']}")
        canv.setFont(FONT, 10.5)
        canv.drawString(MARGIN, height - 2.45 * inch, f"{head['prev_run_id']}  versus  {head['curr_run_id']}")
        canv.setFillColor(DRAFT_COLOUR)
        canv.roundRect(MARGIN, height - 3.15 * inch, 1.15 * inch, 0.32 * inch, 4, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont(BOLD, 11)
        canv.drawCentredString(MARGIN + 0.575 * inch, height - 3.15 * inch + 0.1 * inch, "DRAFT")
        canv.setFont(BOLD, 7.5)
        canv.setFillColor(DRAFT_COLOUR)
        canv.drawString(MARGIN, FOOTER_Y, DRAFT_LABEL)
        canv.restoreState()
    return draw


# --- building blocks --------------------------------------------------------------------------------------------


def _text_width(text: str, font: str) -> float:
    return pdfmetrics.stringWidth(text, font, TABLE_FONT_SIZE)


def _column_widths(headings: list[str], rows: list[list[str]], total: float) -> list[float]:
    """Widths in points measured from the text: each column wants its longest cell on one line (capped at
    MAX_COLUMN_CHARS), and is never narrower than its longest single word, so words are not broken. Spare width
    goes to every column in proportion; a shortfall is taken from the columns that can wrap."""
    pad = 2 * CELL_PAD + 1
    want, need = [], []
    for j, heading in enumerate(headings):
        texts = [r[j] for r in rows]
        words = heading.split() + [w for t in texts for w in t.split()]
        need.append(max([_text_width(w, BOLD) for w in words] + [MIN_COLUMN_IN * inch - pad]) + pad)
        longest = max([_text_width(t[:MAX_COLUMN_CHARS], FONT) for t in texts] + [0.0])
        want.append(max(longest + pad, need[-1]))
    want_a, need_a = np.array(want), np.array(need)
    if want_a.sum() <= total:
        return list(want_a + (total - want_a.sum()) * want_a / want_a.sum())
    if need_a.sum() >= total:
        return list(need_a * total / need_a.sum())
    slack = want_a - need_a
    return list(need_a + slack * (total - need_a.sum()) / slack.sum())


def _tint(hex_colour: str, amount: float = ROW_TINT):
    """``hex_colour`` mixed towards white (``amount`` of the colour kept), for quiet printed row fills."""
    c = colors.HexColor(hex_colour)
    return colors.Color(1 - amount * (1 - c.red), 1 - amount * (1 - c.green), 1 - amount * (1 - c.blue))


def _display(column: str, value: str) -> str:
    """A cell's printed text: categories in words, numbers to the column's decimals with thousands separators,
    flags as Yes / No, vocabulary words without underscores, and ``;``-joined lists (cluster ids, triggered rules)
    with a space after each ``;`` so they can wrap."""
    if LIST_SEP in value:
        return value.replace(LIST_SEP, LIST_SEP + " ")
    if column in CATEGORY_COLUMNS:
        return CATEGORY_LABELS.get(value, value)
    if value in YES_NO:
        return YES_NO[value]
    if column in WORD_COLUMNS:
        return value.replace("_", " ")
    decimals = COLUMN_DECIMALS.get(column)
    if decimals is None or value == "":
        return value
    try:
        return f"{float(value):,.{decimals}f}"
    except ValueError:
        return value


def _table(styles, table, width: float, columns: dict[str, str] | None = None, row_fills: list | None = None,
           max_rows: int | None = None) -> list:
    """A report ``Table`` (strings) as a ReportLab table: optional column subset and headings, zebra rows or
    ``row_fills``, headings repeated on each page. Returns flowables (a note when rows were cut)."""
    names = list(table.columns) if columns is None else [c for c in columns if c in table.columns]
    idx = [list(table.columns).index(c) for c in names]
    rows = [[_display(name, r[j]) for name, j in zip(names, idx, strict=True)] for r in table.rows]
    fills = None if row_fills is None else list(row_fills)
    cut = 0
    if max_rows is not None and len(rows) > max_rows:
        cut = len(rows) - max_rows
        rows, fills = rows[:max_rows], None if fills is None else fills[:max_rows]
    headings = [columns[c] if columns else HEADINGS.get(c, _humanize(c)) for c in names]
    if not rows:
        return [Paragraph("None.", styles["small"])]
    data = [[Paragraph(_text(h), styles["head"]) for h in headings]]
    data += [[Paragraph(_text(v), styles["cell"]) for v in r] for r in rows]
    t = Table(data, colWidths=_column_widths(headings, rows, width), repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_FILL), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD), ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD)]
    for i in range(1, len(data)):
        fill = None if fills is None else fills[i - 1]
        if fill is not None:
            style.append(("BACKGROUND", (0, i), (-1, i), fill))
        elif i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    out = [t]
    if cut:
        out.append(Paragraph(f"{cut} more row(s) not printed; the full table is in the CSV and Excel outputs.",
                             styles["small"]))
    return out


def _key_values(styles, pairs: list[tuple[str, str]], width: float) -> Table:
    data = [[Paragraph(_text(k), styles["cover_label"]), Paragraph(_text(v), styles["cell"])] for k, v in pairs]
    t = Table(data, colWidths=[width * 0.34, width * 0.66], hAlign="LEFT")
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _callout(styles, text: str, fill, rule, width: float) -> Table:
    t = Table([[Paragraph(text, styles["body"])]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("LINEBEFORE", (0, 0), (0, -1), 3, rule),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


def _kpis(styles, tiles: list[tuple[str, str]], width: float) -> Table:
    per_row = 3
    cells = [[Paragraph(_text(v), styles["kpi_value"]), Paragraph(_text(k), styles["kpi_label"])] for k, v in tiles]
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    rows[-1] += [""] * (per_row - len(rows[-1]))
    t = Table(rows, colWidths=[width / per_row] * per_row, hAlign="LEFT")
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.6, RULE),
                           ("BACKGROUND", (0, 0), (-1, -1), ZEBRA), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _chart(png: bytes, width: float) -> Image:
    image = Image(io.BytesIO(png))
    scale = width / image.imageWidth
    image.drawWidth, image.drawHeight = width, image.imageHeight * scale
    return image


def _fmt_number(value, decimals: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:,.{decimals}f}"


# --- sections ---------------------------------------------------------------------------------------------------


def _cover(styles, ctx: dict, generated_utc: datetime, width: float) -> list:
    head = ctx["head"]
    runs = {r[0]: r for r in head["runs"].rows}  # role -> [role, run id, date, vendor, tool]
    pairs = [
        ("Pipeline", head["line_name"]),
        ("Previous inspection", " · ".join(runs["previous"][1:])),
        ("Current inspection", " · ".join(runs["current"][1:])),
        ("Prepared for", head.get("prepared_for") or NOT_STATED),
        ("Prepared by", head.get("prepared_by") or NOT_STATED),
        ("Report date", generated_utc.strftime("%d %B %Y")),
        ("Status", DRAFT_LABEL),
    ]
    disclaimer = (
        "This report is a draft decision aid produced by an automated comparison of two in-line inspections. "
        "Every match, growth rate, remaining life and repair category in it must be reviewed by a qualified "
        "integrity engineer before it is used for repair, pressure or regulatory decisions under 49 CFR 195. "
        "Engineering parameters marked PLACEHOLDER in Appendix A have not yet been confirmed.")
    return [
        Spacer(1, 3.75 * inch - MARGIN),
        _key_values(styles, pairs, width),
        Spacer(1, 0.3 * inch),
        Paragraph("At a glance", styles["h2_cover"]),
        _kpis(styles, _headline_kpis(ctx), width),
        Spacer(1, 0.3 * inch),
        _callout(styles, _text(disclaimer), WARN_FILL, WARN_RULE, width),
        Spacer(1, 0.15 * inch),
        Paragraph(_text(f"Generated {head['generated_utc']} by {head['generator']}."), styles["small"]),
    ]


def _headline_kpis(ctx: dict) -> list[tuple[str, str]]:
    counts = {r[0]: r[1] for r in ctx["summary"]["category_counts"].rows}
    s = ctx["summary"]
    min_rl = ctx["_min_rl_yr"]
    return [
        ("Features classified", str(s["n_features"])),
        ("Immediate conditions", counts.get("immediate", "0")),
        ("60-day conditions", counts.get("60_day", "0")),
        ("180-day conditions", counts.get("180_day", "0")),
        ("Digs driven by remaining life", str(s["n_rl_dig"])),
        ("Shortest remaining life (yr)", _fmt_number(min_rl)),
    ]


def _executive_summary(styles, ctx: dict, inputs: ReportInputs, width: float) -> list:
    head, s, q = ctx["head"], ctx["summary"], ctx["quality"]
    story = [Paragraph("1  Executive summary", styles["h1"])]
    story.append(Paragraph(_text(
        f"This report compares the {head['prev_run_id']} and {head['curr_run_id']} in-line inspections of "
        f"{head['line_name']}. {s['n_features']} current-run anomalies were classified against the repair criteria "
        f"of 49 CFR 195.452(h)(4); growth governs under the {ctx['growth']['profile'] or 'recorded'} profile."),
        styles["body"]))
    story.append(_kpis(styles, _headline_kpis(ctx), width))
    story.append(Spacer(1, 8))
    if q["not_final"]:
        story.append(_callout(styles, f"<b>Not final:</b> {_text(q['pending_text'])}. The report cannot be final "
                              "while reviews are pending.", WARN_FILL, WARN_RULE, width))
    if s["discovery_warning"]:
        story.append(_callout(styles, f"<b>Warning:</b> {_text(s['discovery_warning'])}", WARN_FILL, WARN_RULE,
                              width))
    for warning in ctx["log_warnings"]:
        story.append(_callout(styles, _text(warning), WARN_FILL, WARN_RULE, width))

    story.append(Paragraph("Immediate conditions", styles["h2"]))
    if s["immediate"].rows:
        story.append(_callout(styles, f"<b>{_text(s['immediate_note'])}</b>", DANGER_FILL, DANGER_RULE, width))
        story.append(Spacer(1, 4))
        story += _table(styles, s["immediate"], width, IMMEDIATE_COLUMNS)
    else:
        story.append(Paragraph("No immediate repair conditions were identified.", styles["body"]))

    story.append(Paragraph("IMP class summary", styles["h2"]))
    story += _table(styles, s["imp_class"], width)
    story.append(Paragraph(_text(f"YTC < {s['ytc_flag_years']} yrs: features with no repair category whose "
                                 "remaining life is under that many years. A summary view only: the repair "
                                 "categories drive due dates."), styles["small"]))

    story.append(Paragraph("Anomalies of interest", styles["h2"]))
    story += _table(styles, s["aoi"], width, AOI_COLUMNS)
    story.append(Paragraph("Groups of nearby metal loss with the shortest remaining life (deepest first on a tie).",
                           styles["small"]))

    story.append(Paragraph("Recommended reinspection interval by segment", styles["h2"]))
    story += _table(styles, s["intervals"], width)
    return story


def _pipeline(styles, ctx: dict, width: float) -> list:
    head = ctx["head"]
    pipe = head["pipe"]
    story = [PageBreak(), Paragraph("2  Pipeline and inspections", styles["h1"]),
             Paragraph("Inspections compared", styles["h2"])]
    story += _table(styles, head["runs"], width)
    story.append(Paragraph("Pipe summary", styles["h2"]))
    story.append(_key_values(styles, [
        ("Design pressure (psig)", _rounded(pipe["design_pressure"], 0)),
        ("Stress at MOP (% SMYS)", _rounded(pipe["stress"], 1)),
        ("Design factor", pipe["design_factor"]),
        ("Install year(s)", pipe["install_years"]),
        ("Coating(s)", pipe["coatings"]),
        ("Could affect an HCA", f"{_rounded(pipe['hca_length_ft'], 0)} ft ({_rounded(pipe['hca_pct'], 1)} % of the "
                                "pipe extent)"),
    ], width))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Length by wall thickness and grade", styles["h2"]))
    story += _table(styles, pipe["by_wall_grade"], width * 0.6)
    story.append(Paragraph("Length check", styles["h2"]))
    story += _table(styles, pipe["length_check"], width)
    story.append(Paragraph("Values from pipe_properties.csv and mop_profile.csv.", styles["small"]))
    return story


def _dig_rows(table, categories) -> object:
    """The dig-list rows whose category is in ``categories``, as a table of the same kind."""
    keep = [c.removeprefix("cat-") in categories for c in (table.row_classes or [])]
    rows = [r for r, k in zip(table.rows, keep, strict=True) if k]
    classes = [c for c, k in zip(table.row_classes, keep, strict=True) if k]
    return type(table)(list(table.columns), rows, classes)


def _dig_table(styles, table, width: float) -> list:
    fills = [_tint(CATEGORY_COLOURS[c.removeprefix("cat-")]) for c in (table.row_classes or [])]
    return _table(styles, table, width, DIG_LIST_COLUMNS, row_fills=fills)


def _dig_list(styles, ctx: dict, landscape_width: float) -> list:
    counts = ctx["summary"]["category_counts"]
    story = [NextPageTemplate("landscape"), PageBreak(), Paragraph("3  Repair conditions", styles["h1"])]
    story.append(Paragraph(_text(
        "Anomalies in the immediate, 60-day and 180-day repair categories of 195.452(h)(4), most severe first, then "
        "shortest remaining life. Rows are tinted by category. Due dates run from the discovery date "
        "(195.452(h)(2)). Anomalies scheduled by remaining life are listed in Appendix C; the Excel dig list "
        "holds every column."), styles["body"]))
    story += _table(styles, counts, landscape_width * 0.3)
    story.append(Spacer(1, 8))
    story += _dig_table(styles, _dig_rows(ctx["dig_list"], REPAIR_CATEGORIES), landscape_width)
    story.append(NextPageTemplate("portrait"))
    return story


def _scheduled(styles, ctx: dict, landscape_width: float) -> list:
    rows = _dig_rows(ctx["dig_list"], ("imp_scheduled",))
    story = [NextPageTemplate("landscape"), PageBreak(),
             Paragraph("Appendix C  Anomalies scheduled by remaining life", styles["h1"]),
             Paragraph(_text("Anomalies with no repair category whose remaining life, times the remaining-life "
                             "factor, falls before the next planned assessment (IMP scheduled)."), styles["body"])]
    story += _dig_table(styles, rows, landscape_width)
    return story


def _growth(styles, ctx: dict, inputs: ReportInputs, width: float) -> list:
    g = ctx["growth"]
    story = [PageBreak(), Paragraph("4  Corrosion growth and remaining life", styles["h1"])]
    story.append(Paragraph(_text(f"Governing growth profile: {g['profile'] or 'not recorded'}. "
                                 f"{g['recoat_note']}"), styles["body"]))
    story.append(Paragraph("Growth summary by surface", styles["h2"]))
    story += _table(styles, g["by_surface"], width)
    story.append(Paragraph("Rows inside recoat spans are left out of the excluding_recoated scope whatever the "
                           "setting; manufacturing features are not corrosion and are left out.", styles["small"]))
    story.append(Paragraph("Governing method", styles["h2"]))
    story += _table(styles, g["methods"], width * 0.45)
    story.append(Paragraph("Per-segment growth statistics", styles["h2"]))
    story += _table(styles, g["segments"], width)
    story.append(Paragraph(_text(f"Negative raw growth: {g['pct_negative_raw']}."), styles["small"]))
    story.append(CondPageBreak(3.2 * inch))
    story.append(_chart(report_pdf_charts.growth_vs_distance(inputs), width))
    story.append(_chart(report_pdf_charts.depth_vs_distance(inputs), width))
    story.append(_chart(report_pdf_charts.rl_distribution(inputs), width))
    return story


def _distribution(styles, ctx: dict, inputs: ReportInputs, width: float) -> list:
    d, j = ctx["distribution"], ctx["joints"]
    story = [PageBreak(), Paragraph("5  Anomaly distribution", styles["h1"]),
             Paragraph(_text("Metal loss in each run by depth (% of nominal wall), split into external corrosion, "
                             "internal corrosion and manufacturing."), styles["body"])]
    for title, table, notes in d["depth"]:
        block = [Paragraph(f"Depth bins: {_text(title)}", styles["h2"])] + _table(styles, table, width)
        block += [Paragraph(_text(n), styles["small"]) for n in notes]
        story.append(KeepTogether(block))
    block = [Paragraph("Dent bins (% of OD)", styles["h2"])] + _table(styles, d["dent"], width)
    block += [Paragraph(_text(n), styles["small"]) for n in d["dent_notes"]]
    story.append(KeepTogether(block))
    story.append(Paragraph(_text(f"Joints with the most metal loss (top {j['n']})"), styles["h2"]))
    story += _table(styles, j["by_count"], width)
    story.append(Paragraph(_text(f"Joints with the largest increase (top {j['n']})"), styles["h2"]))
    story += _table(styles, j["by_change"], width) if j["by_change"].rows else [
        Paragraph("No joint increased in metal loss.", styles["body"])]
    story.append(PageBreak())
    story.append(Paragraph("Orientation and elevation", styles["h2"]))
    story.append(_chart(report_pdf_charts.orientation(inputs), width))
    elevation = report_pdf_charts.elevation(inputs)
    if elevation is None:
        story.append(Paragraph("Elevation is not available: no girth weld carries an elevation.", styles["body"]))
    else:
        story.append(_chart(elevation, width))
    return story


def _data_quality(styles, ctx: dict, inputs: ReportInputs, width: float) -> list:
    q, charts = ctx["quality"], ctx["charts"]
    story = [PageBreak(), Paragraph("6  Data quality and review status", styles["h1"])]
    story.append(_key_values(styles, [
        ("Validation issues", str(q["n_issues"])),
        ("Girth-weld alignment rate", q["alignment_rate"]),
        ("Alignment rate outside repairs", q["alignment_rate_outside_repairs"]),
        ("Reviews", q["pending_text"]),
        ("Features flagged for review", "" if q["n_requires_review"] is None else str(q["n_requires_review"])),
    ], width))
    story.append(Paragraph("Validation issues by reason", styles["h2"]))
    story += _table(styles, q["issue_counts"], width)
    story.append(Paragraph("Match statistics by type and confidence", styles["h2"]))
    story += _table(styles, q["match_stats"], width)
    story.append(Paragraph("Review reasons", styles["h2"]))
    story += _table(styles, q["review_reasons"], width * 0.6)
    bt = charts["backtest"]
    story.append(Paragraph("Back-test against field measurements", styles["h2"]))
    if bt is None:
        story.append(Paragraph("No dig results were given, so the back-test was not run.", styles["body"]))
    else:
        story.append(Paragraph(_text(f"{bt['n_within']} of {bt['n']} digs ({bt['pct_within']} %) within ±"
                                     f"{charts['tolerance']} % of wall."), styles["body"]))
        png = report_pdf_charts.unity(backtest(inputs.digs, inputs.curr_std, inputs.settings),
                                      inputs.settings.tool.depth_tolerance_pct_wt)
        if png is not None:
            story.append(_chart(png, width * 0.7))
    return story


def _appendices(styles, ctx: dict, width: float) -> list:
    m = ctx["methods"]
    placeholder_fill = colors.HexColor("#FFF7D6")
    fills = [placeholder_fill if c == "placeholder" else None for c in (m["config"].row_classes or [])]
    story = [PageBreak(), Paragraph("Appendix A  Methods and assumptions", styles["h1"])]
    story.append(Paragraph(_text(
        "Growth rates, remaining strength (ASME B31G / Modified B31G), remaining life and repair categories are "
        "computed as set out in the methods section of report.html, which states every formula. The settings "
        "used are listed below; rows marked PLACEHOLDER need integrity-engineer sign-off before the results are "
        "relied on."), styles["body"]))
    story.append(Paragraph("Tool tolerances", styles["h2"]))
    story += _table(styles, m["tool"], width * 0.6)
    story.append(Paragraph("Repair criteria", styles["h2"]))
    story.append(Paragraph(_text(f"{m['criteria_version']} (sha256 {str(m['criteria_sha256'])[:16]}…)"),
                           styles["small"]))
    story += _table(styles, m["criteria"], width, {"id": "Rule", "category": "Category", "applies_to": "Applies to",
                                                    "feature_types": "Feature types", "description": "Description"})
    story.append(Paragraph("Configuration", styles["h2"]))
    story += _table(styles, m["config"], width, row_fills=fills)
    story.append(PageBreak())
    story.append(Paragraph("Appendix B  Input files", styles["h1"]))
    story.append(Paragraph("SHA-256 fingerprints of every file this report was built from.", styles["body"]))
    hashes = m["hashes"]
    rows = [[Path(p).name, p, h[:16] + "…"] for p, h in hashes.rows]
    short = type(hashes)(["file", "path", "sha256 (first 16)"], rows)
    story += _table(styles, short, width)
    return story


# --- entry point ------------------------------------------------------------------------------------------------


def build_pdf(context: dict, inputs: ReportInputs, generated_utc: datetime) -> bytes:
    """The report as PDF bytes, rendered from ``report.report_context`` and the report inputs (for the charts)."""
    _register_fonts()
    styles = _styles()
    head = context["head"]
    rl = inputs.remaining_life.rl_yr.to_numpy(float)
    ctx = {**context, "_min_rl_yr": float(np.nanmin(rl)) if np.isfinite(rl).any() else None}

    buffer = io.BytesIO()
    width, height = PAGE
    doc = _Doc(buffer, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
               bottomMargin=MARGIN, title=f"{TITLE}: {head['line_name']}", author=head.get("prepared_by") or "",
               subject=f"{head['prev_run_id']} vs {head['curr_run_id']} ({DRAFT_LABEL})",
               creator=f"ili.report {__version__}")
    frame_w = width - 2 * MARGIN
    land_w, land_h = landscape(PAGE)
    running_title = f"{TITLE}  ·  {head['line_name']}"
    running_right = f"{head['prev_run_id']} vs {head['curr_run_id']}"
    page = _page_decorator(running_title, running_right)
    doc.addPageTemplates([
        PageTemplate("cover", [Frame(MARGIN, MARGIN, frame_w, height - 2 * MARGIN, id="cover")],
                     onPage=_cover_decorator(head), pagesize=PAGE),
        PageTemplate("portrait", [Frame(MARGIN, MARGIN, frame_w, height - 2 * MARGIN, id="portrait")],
                     onPage=page, pagesize=PAGE),
        PageTemplate("landscape", [Frame(MARGIN, MARGIN, land_w - 2 * MARGIN, land_h - 2 * MARGIN, id="land")],
                     onPage=page, pagesize=landscape(PAGE)),
    ])
    toc = TableOfContents()
    toc.levelStyles = [styles["toc1"], styles["toc2"]]
    toc.tableStyle = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)])

    story = _cover(styles, ctx, generated_utc, frame_w)
    story += [NextPageTemplate("portrait"), PageBreak(), Paragraph("Contents", styles["h1"].clone("contents")),
              toc, PageBreak()]
    story += _executive_summary(styles, ctx, inputs, frame_w)
    story += _pipeline(styles, ctx, frame_w)
    story += _dig_list(styles, ctx, land_w - 2 * MARGIN)
    story += _growth(styles, ctx, inputs, frame_w)
    story += _distribution(styles, ctx, inputs, frame_w)
    story += _data_quality(styles, ctx, inputs, frame_w)
    story += _appendices(styles, ctx, frame_w)
    story += _scheduled(styles, ctx, land_w - 2 * MARGIN)
    doc.multiBuild(story, canvasmaker=_NumberedCanvas)
    return buffer.getvalue()
