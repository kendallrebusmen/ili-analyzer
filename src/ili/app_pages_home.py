"""The Home page of the review app: the loaded comparison at a glance. DRAFT output only.

``home_view`` is the pure view model (no Streamlit); ``home_page`` draws it. Every missing file or key
degrades to "not run" or a skipped fact, never an error.
"""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Literal

import streamlit as st

from ili import app_state, app_theme
from ili.app_pages_review import not_applied
from ili.report_data import pending_reviews

COUNT_KEYS = ("immediate", "60_day", "180_day", "imp_scheduled")
_PROFILES = {"conservative": "Conservative profile", "pit_to_pit": "Pit-to-pit (consultant)"}
_DEPTH = {"reported": "Reported", "reported_plus_tolerance": "Reported + tool tolerance"}


@dataclass(frozen=True)
class Attention:
    level: Literal["warn", "info", "ok"]
    title: str
    detail: str
    target: str | None


@dataclass(frozen=True)
class HomeView:
    loaded: bool
    counts: dict[str, int | None]
    min_rl_yr: float | None
    min_rl_factored_yr: float | None
    attention: list[Attention]
    facts: list[tuple[str, str]]


def _read(path: Path | None, name: str) -> dict:
    """A JSON file of the folder as a dict; {} when absent, unreadable or not an object."""
    try:
        got = app_state._json(Path(path) / name) if path is not None else {}
        return got if isinstance(got, dict) else {}
    except (ValueError, OSError):
        return {}


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _run_fact(inputs: dict, side: str) -> str | None:
    parts = [p for p in (inputs.get(f"{side}_run_date"), inputs.get(f"{side}_tool")) if p]
    return " · ".join(str(p) for p in parts) or None


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _attention(data) -> list[Attention]:
    items: list[Attention] = []
    pending = 0 if data.match_table.empty else len(pending_reviews(data.match_table))
    if pending:
        items.append(Attention("warn", f"{_plural(pending, 'match', 'matches')} waiting for review",
                               "Accept, reject or correct them in the review queue.", "Match review queue"))
    try:
        n = len(not_applied(data))
    except (AttributeError, KeyError, ValueError):
        n = 0
    if n:
        items.append(Attention("warn", f"{_plural(n, 'review decision', 'review decisions')} not applied",
                               "The last run could not use them; the queue shows why.", "Match review queue"))
    fs = data.flag_status
    n = 0 if fs.empty or "applied_status" not in fs.columns else int((fs.applied_status == "not_applied").sum())
    if n:
        items.append(Attention("warn", f"{_plural(n, 'flag', 'flags')} not applied", "The last run could not use them.",
                               "Dig list and report export"))
    if data.classification_summary.get("discovery_warning"):
        items.append(Attention("warn", "Discovery date not entered", str(data.classification_summary["discovery_warning"]),
                               "Run setup"))
    cls = data.classification
    if not cls.empty and "review_reasons" in cls.columns:
        from ili.classify import REASON_SEP
        n = sum(isinstance(c, str) and "could_be_more_severe" in c.split(REASON_SEP) for c in cls.review_reasons)
        if n:
            items.append(Attention("info", f"{_plural(n, 'anomaly', 'anomalies')} could be more severe",
                                   "Their worst case falls in a more urgent category.", "Dig list and report export"))
    missing = [name for name, frame in (("match", data.match_table), ("growth", data.growth),
                                        ("remaining life", data.remaining_life), ("classify", data.classification))
               if frame.empty]
    if missing:
        items.append(Attention("info", f"Stages not run yet: {', '.join(missing)}",
                               "Run the pipeline to fill them.", "Run setup"))
    if not cls.empty and "governing_category" in cls.columns:
        n = int((cls.governing_category == "removed").sum())
        if n:
            items.append(Attention("ok", f"{_plural(n, 'feature', 'features')} removed after the inspection",
                                   "They are out of the ranking.", None))
    return items


def home_view(data, path: Path | None) -> HomeView:
    if data is None:
        return HomeView(False, dict.fromkeys(COUNT_KEYS), None, None, [], [])
    by_cat = data.classification_summary.get("counts_by_category") if data.classification_summary else None
    by_cat = by_cat if isinstance(by_cat, dict) else None
    counts = {k: (int(by_cat.get(k, 0)) if by_cat is not None else None) for k in COUNT_KEYS}
    rl = _read(path, "remaining_life_summary.json")
    inputs = _read(path, app_state.INPUTS_FILE)
    growth = _read(path, "growth_summary.json")
    facts: list[tuple[str, str]] = []
    for label, side in (("Previous run", "prev"), ("Current run", "curr")):
        if value := _run_fact(inputs, side):
            facts.append((label, value))
    ms = data.match_summary or {}
    n_curr, n_new = _num(ms.get("n_curr_anomalies")), _num((ms.get("counts_by_match_type") or {}).get("new"))
    if n_curr is not None:
        facts.append(("Anomalies, current run", f"{int(n_curr):,}"))
        if n_new is not None and n_curr > 0:
            matched = int(n_curr - n_new)
            facts.append(("Matched to previous run", f"{matched:,} ({matched / n_curr:.0%})"))
    rate = _num((data.alignment_summary or {}).get("alignment_rate"))
    if rate is not None:
        facts.append(("Weld alignment", f"{rate:.1%} of joints matched"))
    if growth.get("governing_profile"):
        facts.append(("Growth method", _PROFILES.get(growth["governing_profile"], str(growth["governing_profile"]))))
    if rl.get("depth_basis"):
        facts.append(("Depth basis", _DEPTH.get(rl["depth_basis"], str(rl["depth_basis"]))))
    if inputs.get("settings"):
        facts.append(("Settings file", Path(str(inputs["settings"])).name))
    return HomeView(True, counts, _num(rl.get("min_rl_yr")), _num(rl.get("min_rl_factored_yr")),
                    _attention(data), facts)


def _go(page: str) -> None:
    st.session_state["page"] = page


def _kpi(label: str, value: str, note: str, alert: bool) -> str:
    cls = "nemi-kpi nemi-kpi--alert" if alert else "nemi-kpi"
    return (f'<div class="{cls}"><span class="nemi-kpi-label">{escape(label)}</span><strong>{escape(value)}</strong>'
            f'<span class="nemi-kpi-note">{escape(note)}</span></div>')


def _count_card(label: str, n: int | None, note: str) -> str:
    if n is None:
        return _kpi(label, "—", "not run", False)
    return _kpi(label, str(n), note, n > 0)


def _attention_html(item: Attention) -> str:
    return (f'<div class="nemi-attn"><span class="nemi-attn-dot nemi-attn-dot--{item.level}"></span>'
            f'<div><strong>{escape(item.title)}</strong>'
            f'<span class="nemi-attn-detail">{escape(item.detail)}</span></div></div>')


def _years(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f} yr"


def home_page() -> None:
    folder = app_state.comparison_from_session()
    app_theme.page_header("Home", app_state.comparison_context(folder))
    data = app_state.page_data(folder)  # after the header: an unreadable log stops here with the header drawn
    view = home_view(data, folder)
    if not view.loaded:
        with st.container(border=True):
            st.subheader("No comparison loaded yet")
            st.write("Choose the two inspections and run the pipeline to see the repair conditions here.")
            st.button("Start with Run setup", on_click=_go, args=("Run setup",))
        return
    st.markdown(app_theme.section_title("Repair conditions"), unsafe_allow_html=True)
    cards = [_count_card(label, view.counts[key], note) for label, key, note in
             (("Immediate", "immediate", "repair now"), ("60-day", "60_day", "from discovery"),
              ("180-day", "180_day", "from discovery"), ("Scheduled by life", "imp_scheduled", "IMP scheduled"))]
    factored = "not run" if view.min_rl_yr is None else f"factored {_years(view.min_rl_factored_yr)}"
    cards.append(_kpi("Shortest remaining life", _years(view.min_rl_yr), factored, False))
    for col, card in zip(st.columns(5), cards):
        col.markdown(card, unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True, key="home_card_attention"):
        st.markdown(app_theme.card_title("Needs your attention"), unsafe_allow_html=True)
        if not view.attention:
            st.markdown('<div class="nemi-attn-detail">Nothing needs your attention.</div>', unsafe_allow_html=True)
        for i, item in enumerate(view.attention):
            with st.container(key=f"home_attn_row_{i}"):
                text, action = st.columns([4, 1.3], vertical_alignment="center")
                text.markdown(_attention_html(item), unsafe_allow_html=True)
                if item.target:
                    action.button(app_theme.nav_label_text(item.target), key=f"attn_{i}", on_click=_go,
                                  args=(item.target,), width="stretch",
                                  help=f"Go to {app_theme.nav_label_text(item.target)}")
    with right, st.container(border=True, key="home_card_facts"):
        st.markdown(app_theme.card_title("Run facts"), unsafe_allow_html=True)
        cells = "".join(f'<div>{escape(label)}<strong class="nemi-truncate">{escape(value)}</strong></div>'
                        for label, value in view.facts)
        st.markdown(f'<div class="nemi-facts">{cells}</div>', unsafe_allow_html=True)
