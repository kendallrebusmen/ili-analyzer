"""Report (spec 6.10): the self-contained HTML report.

``render_report`` draws every section from the report inputs (``report_data``), the category changes
(``category_change``) and the charts (``report_charts``) as they are; nothing is recomputed. The page is
fully local: plotly.js is inlined once and every chart is a div without its own copy, and nothing is
fetched. jinja2 escapes every value; only the plotly.js bundle and the chart divs are marked safe, so a
vendor id, a file path or the line name can never become markup. None / NaN render as an empty cell.

The report is always DRAFT: every section opens with the label, a banner carries it on screen and a page
margin box repeats it on each printed page,
and no wording marks it final or approved.

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd
import plotly.offline
import yaml
from markupsafe import Markup

from ili import __version__
from ili.align_welds import DRAFT_LABEL
from ili.category_change import CHANGE_COLUMNS, CategoryChangeResult
from ili.classify import CATEGORY_PRIORITY, IMMEDIATE_ACTION_NOTE
from ili.match import CONFIDENCES
from ili.report_charts import (CATEGORY_COLOURS, depth_vs_distance, feature_map, figure_div, growth_histograms,
                               growth_vs_distance, rl_distribution, unity_chart)
from ili.report_data import (ReportInputs, backtest, config_rows, criteria_rows, dig_list, input_hashes,
                             largest_growth, match_stats, pending_reviews)

TEMPLATES = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"
SECTION_IDS = ("header", "summary", "data-quality", "growth", "dig-list", "largest-changes", "charts", "methods")
NOT_STATED = "not stated"
DECIMALS = 3  # presentation only: the CSV outputs keep full precision
COORD_DECIMALS = 6
COORD_COLUMNS = ("lat", "lon")
CSS_UNSAFE = "\"\\'<>&\n"  # changed by CSS string syntax or by HTML autoescaping inside <style>

IMMEDIATE_COLUMNS = ["match_id", "prev_feature_id", "curr_feature_id", "feature_type", "curr_log_dist_ft",
                     "governing_rule", "triggered_rules", "action_note"]
LARGEST_GROWTH_COLUMNS = ["match_id", "match_type", "prev_feature_id", "curr_feature_id", "curr_log_dist_ft",
                          "segment_id", "history_class", "d_prev_pct", "d_curr_pct", "raw_mpy", "governing_mpy",
                          "governing_method", "needs_review"]
MOVED_COLUMNS = [c for c in CHANGE_COLUMNS if c != "moved_more_severe"]
REVIEW_LOG_COLUMNS = ["match_id", "match_type", "prev_feature_id", "curr_feature_id", "confidence",
                      "missing_hint", "review_status"]


@dataclass(frozen=True)
class ReportHeader:
    """Header fields the tallies do not carry (CLI options); None renders as "not stated"."""
    line_name: str | None
    prev_tool: str | None
    curr_tool: str | None


@dataclass(frozen=True)
class Table:
    """A table ready for the template: every cell already a string ("" for a blank)."""
    columns: list[str]
    rows: list[list[str]]
    row_classes: list[str] | None = None


def _number(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def cell(value, column: str | None = None) -> str:
    """One value as display text: None, NaN, NaT and NA are ""; floats to ``DECIMALS`` places
    (``COORD_DECIMALS`` for lat/lon) without trailing zeros; dates as ISO; lists comma-joined."""
    if isinstance(value, (list, tuple)):
        return ", ".join(cell(v) for v in value)
    if value is None or (not isinstance(value, (str, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _number(float(value), COORD_DECIMALS if column in COORD_COLUMNS else DECIMALS)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def table(df: pd.DataFrame, columns: list[str] | None = None, row_classes: list[str] | None = None) -> Table:
    """``df[columns]`` as a ``Table``; ``row_classes`` gives each row's class ("" for none)."""
    columns = list(df.columns) if columns is None else list(columns)
    rows = [[cell(v, c) for c, v in zip(columns, r, strict=True)] for r in df[columns].itertuples(index=False)]
    return Table(columns, rows, row_classes)


def _css_string(text: str) -> str:
    """``text`` for a double-quoted CSS string in an autoescaped ``<style>``: refused if it holds a character
    that CSS or the HTML escaping would change."""
    unsafe = sorted(set(text) & set(CSS_UNSAFE))
    if unsafe:
        raise ValueError(f"label {text!r} contains {unsafe}, which cannot go in the print margin box")
    return text


def _counts(mapping: dict, key: str) -> Table:
    return Table([key, "count"], [[cell(k), cell(v)] for k, v in mapping.items()])


def _vendor(manifest: dict, run_id: str) -> str:
    """The ``vendor`` key of the run's vendor map named in the ingest manifest; "not stated" if unreadable."""
    try:
        run = next(r for r in manifest["runs"] if r["run_id"] == run_id)
        vmap = yaml.safe_load(Path(run["vendor_map"]).read_text(encoding="utf-8"))
        vendor = vmap["vendor"]
    except (KeyError, TypeError, StopIteration, OSError, yaml.YAMLError):
        return NOT_STATED
    return cell(vendor) or NOT_STATED


def _run_dates(std: pd.DataFrame) -> str:
    return ", ".join(sorted({cell(d) for d in std.run_date.dropna()})) or NOT_STATED


def _rate(value) -> str:
    return "not available" if value is None else f"{100.0 * float(value):.2f} %"


def _header(inputs: ReportInputs, header: ReportHeader, generated_utc: datetime) -> dict:
    match = inputs.summaries["match"]
    prev_id, curr_id = match["prev_run_id"], match["curr_run_id"]
    runs = Table(["run", "run id", "run date", "vendor", "tool type"], [
        [role, run_id, _run_dates(std), _vendor(inputs.ingest_manifest, run_id), tool or NOT_STATED]
        for role, run_id, std, tool in (("previous", prev_id, inputs.prev_std, header.prev_tool),
                                        ("current", curr_id, inputs.curr_std, header.curr_tool))])
    return {"line_name": header.line_name or NOT_STATED, "prev_run_id": prev_id, "curr_run_id": curr_id,
            "runs": runs, "generated_utc": generated_utc.isoformat(timespec="seconds"),
            "generator": f"ili.report {__version__}"}


def _summary(inputs: ReportInputs, dig: pd.DataFrame) -> dict:
    cls = inputs.classification
    counts = cls.governing_category.value_counts().reindex(list(CATEGORY_PRIORITY), fill_value=0)
    return {
        "immediate": table(dig[dig.governing_category == "immediate"], IMMEDIATE_COLUMNS),
        "immediate_note": IMMEDIATE_ACTION_NOTE,
        "category_counts": Table(["category", "count"], [[c, cell(n)] for c, n in counts.items()],
                                 [f"cat-{c}" for c in counts.index]),
        "n_features": len(cls),
        "n_rl_dig": int(inputs.remaining_life.dig_required_by_remaining_life.eq(True).sum()),
        "n_rl_feature": len(inputs.remaining_life),
        "n_imp_scheduled": int(counts["imp_scheduled"]),
        "intervals": table(inputs.intervals),
        "discovery_warning": inputs.summaries["classification"].get("discovery_warning"),
    }


def _match_stats(match_table: pd.DataFrame) -> pd.DataFrame:
    """``match_stats`` with the confidence columns in band order (high, medium, low), then ``all``."""
    stats = match_stats(match_table)
    confidences = [c for c in CONFIDENCES if c in stats.columns]
    others = [c for c in stats.columns if c not in confidences and c != "all"]
    return stats[[*confidences, *others, "all"]].reset_index()


def _data_quality(inputs: ReportInputs) -> dict:
    issues = inputs.validation_issues
    by_reason = issues.groupby(["run_id", "reason"], dropna=False).size().rename("count").reset_index()
    align = inputs.summaries["alignment"]
    cls_summary = inputs.summaries["classification"]
    return {
        "n_issues": len(issues),
        "issue_counts": table(by_reason),
        "issues": table(issues),
        "alignment_rate": _rate(align.get("alignment_rate")),
        "alignment_rate_outside_repairs": _rate(align.get("alignment_rate_outside_repairs")),
        "match_stats": table(_match_stats(inputs.match_table)),
        "n_pending": len(pending_reviews(inputs.match_table)),
        "n_requires_review": cls_summary.get("n_requires_review"),
        "review_reasons": _counts(cls_summary.get("counts_by_review_reason") or {}, "reason"),
        "crossings_given": inputs.crossings is not None,
        "classified_with_crossings": cls_summary.get("crossings_file_given"),
    }


def _growth(inputs: ReportInputs) -> dict:
    summary = inputs.summaries["growth"]
    return {
        "segments": table(inputs.growth_segments),
        "pct_negative_raw": cell(summary.get("pct_negative_raw")),
        "methods": _counts(summary.get("counts_by_governing_method") or {}, "governing_method"),
        "histograms": Markup(figure_div(growth_histograms(inputs))),
    }


def _charts(inputs: ReportInputs, dig: pd.DataFrame) -> dict:
    tolerance = inputs.settings.tool.depth_tolerance_pct_wt
    bt = backtest(inputs.digs, inputs.curr_std, inputs.settings)
    fmap, unity = feature_map(inputs, dig), unity_chart(bt, tolerance)
    return {
        "depth": Markup(figure_div(depth_vs_distance(inputs))),
        "growth": Markup(figure_div(growth_vs_distance(inputs))),
        "rl": Markup(figure_div(rl_distribution(inputs))),
        "map": None if fmap is None else Markup(figure_div(fmap)),
        "unity": None if unity is None else Markup(figure_div(unity)),
        "backtest": None if bt is None else {
            "n_within": bt.n_within, "n": len(bt.table), "pct_within": cell(bt.pct_within),
            "unmatched": ", ".join(bt.unmatched_ids)},
        "tolerance": cell(tolerance),
        "digs_given": inputs.digs is not None,
    }


def _methods(inputs: ReportInputs) -> dict:
    mt = inputs.match_table
    reviewed = mt[mt.needs_review.fillna(False).astype(bool)]
    tool = inputs.settings.tool.model_dump()
    config = config_rows(inputs.settings, inputs.settings_path)
    return {
        "config": table(config.assign(placeholder=config.placeholder.map({True: "PLACEHOLDER", False: ""})),
                        row_classes=["placeholder" if p else "" for p in config.placeholder]),
        "settings_path": str(inputs.settings_path),
        "tool": Table(["setting", "value"], [[f"tool.{k}", cell(v)] for k, v in tool.items()]),
        "criteria_version": inputs.criteria.version,
        "criteria_sha256": inputs.summaries["classification"].get("criteria_sha256"),
        "criteria_path": str(inputs.criteria_path),
        "criteria": table(criteria_rows(inputs.criteria)),
        "hashes": Table(["path", "sha256"], [[p, h] for p, h in input_hashes(inputs).items()]),
        "review_log": table(reviewed, REVIEW_LOG_COLUMNS),
    }


def _environment() -> jinja2.Environment:
    return jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=True,
                              undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render_report(inputs: ReportInputs, changes: CategoryChangeResult, header: ReportHeader,
                  generated_utc: datetime) -> str:
    """The report as one self-contained HTML document (spec 6.10), sections in ``SECTION_IDS`` order."""
    dig = dig_list(inputs)
    n_largest = inputs.settings.report.largest_changes_n
    moved = changes.table[changes.table.moved_more_severe]
    context = {
        "label": DRAFT_LABEL,
        "label_css": _css_string(DRAFT_LABEL),
        "plotlyjs": Markup(plotly.offline.get_plotlyjs()),
        "category_colours": CATEGORY_COLOURS,
        "head": _header(inputs, header, generated_utc),
        "summary": _summary(inputs, dig),
        "quality": _data_quality(inputs),
        "growth": _growth(inputs),
        "dig_list": table(dig, row_classes=[f"cat-{c}" for c in dig.governing_category]),
        "largest": {"n": n_largest, "growth": table(largest_growth(inputs, n_largest), LARGEST_GROWTH_COLUMNS),
                    "moved": table(moved, MOVED_COLUMNS), "n_compared": len(changes.table)},
        "charts": _charts(inputs, dig),
        "methods": _methods(inputs),
    }
    return _environment().get_template(TEMPLATE_NAME).render(context)
