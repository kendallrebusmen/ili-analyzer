# Nemi — UI Refresh and Rebrand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brand the software "Nemi" on everything users see, and refresh the Streamlit app. The new app has:
- a clean engineering look with a deep teal accent;
- a left sidebar for navigation;
- a Home overview;
- readable tables;
- one DRAFT badge.

**Architecture:**
- **Branding.** One source of truth for the brand, `src/ili/brand.py`.
- **Look.** The theme lives in two places that a test keeps in sync: tokens and CSS in `src/ili/app_theme.py`, and the matching palette in `app/.streamlit/config.toml`.
- **Shell.** `app/streamlit_app.py` draws the sidebar: wordmark, page menu, comparison name and reviewer box. Each page then calls one `page_header()` helper that draws the title, the run context and the DRAFT badge.
- **Home page.** The new `app_pages_home.py` builds its view from a pure `home_view(data, path, settings)` that can be tested without the app.
- **Tables.** `app_columns.py` gives every app table readable column names and number formats.
- **Unchanged.** The internal package `ili`, the CLI commands and the JSON `generator` strings keep their names. The reports change their name only, not their layout.

**Tech Stack:** unchanged (Streamlit 1.66, plotly, pandas, pytest with `streamlit.testing.v1.AppTest`). No new dependency, no web fonts, no network.

**Design (approved):**
- `C:\Users\krebusmen\.claude\plans\ask-me-questions-to-typed-forest.md`.
- The mockup the user approved: https://claude.ai/artifact/SmnCKkMguUCN4qJYunFT1Z (two boards, `Main.dc.html` = Home and `Review.dc.html` = Match review). Its markup is in the session scratchpad and is not committed. Match its look; the exact values are below.

**Spec:**
- `docs/SPEC.md` §2: every output carries the DRAFT label; nothing is marked approved; the app is local only.
- §6.11: the app's pages.

**Branch:** `ui-nemi` from `main` (5513baa). Tests run with `.venv/Scripts/python.exe -m pytest … -p no:cacheprovider`. The full suite takes about 10 min, so run it in the background redirected to a log file.

## Global Constraints

- **The DRAFT label is still on every app page and every download.** `DRAFT_LABEL` from `ili.align_welds`, the exact string `DRAFT – for review by qualified personnel.`, appears in each page's header badge. The download captions and dig-list caption that carry it today stay.
- **The reports keep every DRAFT marking.** The HTML banner and sections, and the PDF footer, watermark and cover box, are unchanged.
- **Local only.**
  - No `http`, `@import` or `url(` in the CSS.
  - No web fonts. The font stack is `'Segoe UI', 'Noto Sans', system-ui, -apple-system, sans-serif`.
  - The inline SVG wordmark has no `xmlns` attribute and no URL.
  - `app/.streamlit/config.toml` keeps `[server] address = "localhost"`, `[browser] gatherUsageStats = false` and `[client] toolbarMode = "viewer"`.
- **WCAG AA.** Every text/background pair in the palette is at least 4.5:1. The existing check is `tests/test_app_theme.py:45-53`; extend its pair list.
- **Palette (exact values; light / dark):**

  | Token | Light | Dark |
  |---|---|---|
  | `PRIMARY` (accent) | `#0F766E` | `#14B8A6` |
  | `LINK` | `#0F766E` | `#5EEAD4` |
  | page background | `#F6F7F8` | `#111518` |
  | surface | `#FFFFFF` | `#1A2025` |
  | sidebar | `#EEF2F3` | `#151A1E` |
  | ink | `#1F2A33` | `#E6EAED` |
  | muted | `#56636E` | `#A3AEB7` |
  | line | `#D8DEE2` | `#33404A` |
  | danger | `#B42318` | `#FF8A80` |
  | danger-soft background | `#FFF6F5` | `#3A1F1D` |
  | warning text | `#93370D` | `#FEC84B` |
  | warning background | `#FEF0C7` | `#3D2E12` |
  | success | `#067647` | `#47CD89` |

  Use the CSS `light-dark()` pattern already in `app_theme.py`.
- **Page names do not change.** The values in `PAGES` stay `"Run setup"`, `"Weld alignment"`, `"Match review queue"`, `"Growth and remaining life explorer"` and `"Dig list and report export"`. `"Home"` is added first. Tests drive pages with `at.radio(key="page").set_value(<page name>)`.
- **No behaviour change.** Every existing app behaviour stays the same: saving reviews and flags, reruns, downloads, the setup run. Only presentation changes.

## Review Focus

1. **No comparison loaded.** Every page, Home included, still renders: a clear "nothing loaded" state with a way to Run setup, never an exception. Test: Task 4 for Home, Task 6 for every page.
2. **A comparison folder from an older pipeline**, with no `remaining_life_summary.json`, no `classification_summary.json`, or a summary missing new keys. Home shows "not run yet" for that part and never raises a KeyError. Test in Task 4.
3. **Dark mode.**
  - No text is drawn in a hard-coded dark colour on the dark background.
  - Charts in the app go through `style_chart`, which removes the forced font colour.
  - Test in Task 2: `style_chart` leaves no `font.color` on the layout.
4. **The sidebar menu's pending count updates after a decision is saved.** The count comes from the same `pending_reviews` as Home, and the page reruns after a save. Test in Task 3: the label shows N, a decision is appended to the log, the app reruns, and the label shows N-1.
5. **A very long comparison folder name or line name** wraps or ellipsizes in the sidebar and header. It never pushes the DRAFT badge off-screen. The fix is CSS: `overflow-wrap: anywhere` / `text-overflow: ellipsis` with a `title` attribute carrying the full text. Test in Task 3: a 120-character name is in the rendered HTML with its `title` attribute.

---

### Task 1: The Nemi brand on everything users see

**Files:**
- Create: `src/ili/brand.py`
- Modify: `app/streamlit_app.py` (`TITLE` and the About text), `src/ili/report.py` (the header dict near line 202), `src/ili/templates/report.html.j2` (`<title>` at :22, `<h1>` at :98, the "Generated … by" line at :102), `src/ili/report_pdf.py` (cover kicker :259, `TITLE` usage in the running header :697, the "Generated by" line :437, metadata `creator` :694), `README.md:1`, `docs/SPEC.md:1`, `pyproject.toml` (`description`), `config/settings.yaml:1` comment, the `app/streamlit_app.py` module docstring
- Test: new `tests/test_brand.py`; additions to `tests/test_report_html.py` and `tests/test_report_pdf.py`

**Interfaces:**
- Produces in `brand.py`:
  - `PRODUCT_NAME = "Nemi"`;
  - `PRODUCT_TAGLINE = "ILI run comparison"`;
  - `product_label() -> str`, which returns `f"Nemi {ili.__version__}"`;
  - `wordmark_svg(size: int = 30) -> str`.

  `wordmark_svg` is an inline `<svg width=… height=… viewBox="0 0 30 30" role="img" aria-label="Nemi">` with three shapes:
  - a ring: `<circle cx="15" cy="15" r="11" fill="none" stroke-width="3.2">`, stroked with the current ink colour through `stroke="currentColor"`;
  - a teal arc: `<path d="M15 4 A11 11 0 0 1 26 15" … stroke-linecap="round">`, stroked with `var(--nemi-accent)`;
  - a centre dot: `<circle cx="15" cy="15" r="3">`, filled with `var(--nemi-accent)`.

  It has no `xmlns` attribute.
- The report header dict gains `"product": product_label()`. The manifest's `"generator"` stays `ili.report <version>`, and `tests/test_report_cli.py:62` stays green.
- **App.**
  - `TITLE = "Nemi"`, used for `page_title`.
  - About text: `f"{product_label()} — {PRODUCT_TAGLINE}. {DRAFT_LABEL}"`.
- **HTML report.**
  - `<title>` and `<h1>`: `Nemi — ILI run comparison {prev} vs {curr}`.
  - The "Generated … by" line shows `{{ head.product }}`.
- **PDF report.**
  - Cover kicker: `NEMI · INTEGRITY ASSESSMENT REPORT`.
  - Running header text: `Nemi — In-Line Inspection Run Comparison`.
  - Cover "Generated … by": `product_label()`.
  - Metadata `creator`: `product_label()`.
  - The cover's existing `TITLE` text is unchanged, so `test_report_pdf.py:62` still passes.
- **Docs and metadata.**
  - `README.md` and `docs/SPEC.md` titles: `Nemi — ILI Run-Comparison Analyzer`.
  - `pyproject.toml` description: `Nemi: ILI run-comparison analyzer (DRAFT decision aid)`.
  - Settings comment: `# Nemi settings. Spec section 7.1.`
  - `docs/plans/*` are historical and not edited.

- [ ] **Step 1: Write the failing tests.**
  - `test_brand.py`:
    - `test_product_label` equals `f"Nemi {ili.__version__}"`.
    - `test_wordmark_is_local_svg`: contains `role="img"` and `aria-label="Nemi"`, and contains neither `http` nor `xmlns`.
  - In `test_report_html.py`: the rendered report's `<title>` starts with `Nemi — ILI run comparison`, and the page contains `Nemi 0.1.0`. Use the existing report fixture.
  - In `test_report_pdf.py`: the cover page text contains `NEMI · INTEGRITY ASSESSMENT REPORT` and `Nemi 0.1.0`; every page still has `DRAFT`.
  - `test_app_title_is_nemi`: `app/streamlit_app.py`'s `TITLE == "Nemi"`. Import as `test_app_theme.py` does, or read the source.
- [ ] **Step 2:** Run those files. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4:** Run `tests/test_brand.py tests/test_report_html.py tests/test_report_pdf.py tests/test_report_cli.py tests/test_app_theme.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(brand): Nemi on the app, reports and docs`.

### Task 2: The Nemi theme (tokens, CSS, chart styling)

**Files:**
- Modify: `src/ili/app_theme.py`, `app/.streamlit/config.toml`
- Test: `tests/test_app_theme.py`

**Interfaces:**
- **Tokens.** Replace the colour constants at `app_theme.py:19-36` with the Global Constraints palette:
  - keep the existing names where they exist (`PRIMARY`, `LINK`, `INK`, `MUTED`, `LINE`, `DANGER`, and their `DARK_*` twins);
  - add `SIDEBAR`/`DARK_SIDEBAR`, `WARNING_INK`/`WARNING_BG` (and dark), `SUCCESS` (and dark), and `DANGER_SOFT` (and dark).
- **`config.toml`** mirrors the light and dark palettes in `[theme.light]` and `[theme.dark]`: `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `linkColor`, `borderColor`, `redColor`.
  - **Sidebar background.** If Streamlit 1.66 accepts per-theme sidebar options, set it there. Check with `.venv/Scripts/python.exe -c "import streamlit.config as c; print([k for k in c._config_options if 'sidebar' in k])"`. Otherwise colour `[data-testid="stSidebar"]` from the CSS variable. Record which in the task report.
  - **Comment.** The `# Enterprise Suite look` comment becomes `# Nemi look (presentation only)`.
- **CSS.** Rename `ENTERPRISE_CSS` to `APP_CSS`. CSS variables are prefixed `--nemi-`; the `--nemi-accent` variable is the teal.
  - **Remove** the radio-as-tabs rules (:89-96), the breadcrumb, and the `ili-shell` / `ili-breadcrumb` / `ili-avatar` styles. Task 3 removes their HTML.
  - **Keep and restyle** the pills, cards (`card_` keys) and the sticky action bar.
  - **Add classes** (all `nemi-`):
    - `nemi-wordmark`: the sidebar brand row;
    - `nemi-header`: a flex row with the title block left and the DRAFT badge right, wrapping on narrow screens;
    - `nemi-subtitle`;
    - `nemi-badge`: an amber pill using `WARNING_INK` on `WARNING_BG`;
    - `nemi-kpi` and `nemi-kpi--alert`: the danger-soft background with danger text, used when a repair count is above 0;
    - `nemi-attn` / `nemi-attn-dot` / `nemi-attn-dot--warn` / `--info` / `--ok`;
    - `nemi-facts`: a two-column grid;
    - `nemi-step`: a numbered round badge for Run setup steps;
    - `nemi-truncate`: `overflow-wrap: anywhere`.
  - **Sidebar menu.** Style the sidebar radio as a menu:
    - no radio dots;
    - full-width rows;
    - the selected row on a surface background with accent text and weight 600.

    Use the stable `stSidebar` / `stRadio` test ids only. Comment that this is presentation only and degrades to a plain radio if Streamlit renames them.
- **Charts.** Add `style_chart(fig: go.Figure) -> go.Figure` in `app_theme.py`, which returns the same figure restyled:
  - `layout.template = None`;
  - `paper_bgcolor` and `plot_bgcolor` = `"rgba(0,0,0,0)"`;
  - `layout.font.color` removed (set to `None`);
  - `colorway = [PRIMARY, "#1F2A33", "#B54708", "#56636E", "#14B8A6"]`;
  - axis `gridcolor` = `LINE`.

  Traces keep their data.

- [ ] **Step 1: Write the failing tests.**
  - `test_palette_matches_config`: replaces the old pinned hex values with the new palette; config.toml equals the constants.
  - `test_palette_meets_aa`: the existing AA test, with the new pairs added:
    - ink, muted and accent on page, surface and sidebar;
    - warning ink on warning background;
    - danger on danger-soft background;
    - all of the above in light and dark.
  - `test_css_local_only`: the existing check, now on `APP_CSS`.
  - `test_no_tab_css`: neither `data-selected` nor `ili-breadcrumb` appears in `APP_CSS`.
  - `test_style_chart`: a figure with `font={"color": "#2B2F36"}` and `template="simple_white"` comes back with `layout.font.color is None`, transparent backgrounds and the colorway's first entry `PRIMARY`.
- [ ] **Step 2:** Run `tests/test_app_theme.py -q`. Expected: the new tests FAIL. Old tests that pin the old palette or the shell/breadcrumb will also fail: update or remove them in this task, except the shell and breadcrumb render tests, which Task 3 rewrites.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_app_theme.py -q`. Expected: PASS. The only allowed failures are the shell/breadcrumb render tests (`test_app_renders_the_shell_with_tabs_and_no_sidebar` and the breadcrumb test); mark them `xfail(reason="rewritten in Task 3")`.
- [ ] **Step 5: Commit** `feat(app): Nemi theme tokens, CSS and chart styling`.

### Task 3: The shell: sidebar navigation and page header

**Files:**
- Modify: `app/streamlit_app.py`, `src/ili/app_theme.py` (HTML helpers), `src/ili/app_state.py`, the five page functions in `src/ili/app_pages_setup.py`, `app_pages_review.py` and `app_pages_results.py` (their first two lines only), and `NEEDS_REVIEWER` in `app_pages_review.py`
- Test: `tests/test_app_theme.py` (shell tests rewritten), `tests/test_app_review.py:348`, and the other `tests/test_app_*.py` that assert `ili-shell`

**Interfaces:**
- `app_theme.page_header(title: str, subtitle: str | None) -> None` renders, via `st.markdown(unsafe_allow_html=True)`:
  - a `<header class="nemi-header">` holding an `<h1>` title;
  - an optional `<span class="nemi-subtitle nemi-truncate" title="{subtitle}">`;
  - `<span class="nemi-badge">{DRAFT_LABEL}</span>`, with the exact label text.

  All text is HTML-escaped.
- `app_theme.sidebar_brand_html() -> str`: `<div class="nemi-wordmark">{brand.wordmark_svg()}<span>Nemi</span><span>ILI run comparison</span></div>`.
- `app_state.comparison_context(path: Path | None) -> str | None` returns `"{line_name} · {prev_run_date} → {curr_run_date}"` from the folder's `pipeline_inputs.json`.
  - A missing line name is left out.
  - It returns `None` when there is no folder or no file.
- `streamlit_app.py`:
  - `initial_sidebar_state="expanded"`.
  - In `st.sidebar`, in this order:
    - the brand HTML;
    - `st.radio("Pages", list(PAGES), key="page", format_func=nav_label, label_visibility="collapsed")`;
    - a small "Comparison" caption with the folder name (`nemi-truncate`, full path in `title`);
    - the reviewer `st.text_input("Reviewer", key=app_state.REVIEWER_KEY, placeholder="Your name (required to decide)")`.
  - The old shell bar, the breadcrumb and the `[3,1]` columns row are removed.
- `nav_label(page: str) -> str` (in `streamlit_app.py` or `app_theme.py`) gives short labels:

  | Page | Label |
  |---|---|
  | Home | Home |
  | Run setup | Run setup |
  | Weld alignment | Weld alignment |
  | Match review queue | `Match review · {n}` when n pending > 0, else `Match review` |
  | Growth and remaining life explorer | Growth & life |
  | Dig list and report export | Dig list & export |

  n comes from `len(report_data.pending_reviews(data.match_table))`, with `data` from `app_state.page_data(app_state.comparison_from_session())`. It is 0 when there is no data, and it must not raise.
- **Pages.** Each of the five pages replaces `st.title(X); st.caption(DRAFT_LABEL)` with `app_theme.page_header(X, app_state.comparison_context(path))`, where `path` is the page's comparison folder. Run setup passes `None`.
- `NEEDS_REVIEWER` text names "the Reviewer box in the sidebar". Update the test at `tests/test_app_review.py:348`: the hint mentions "Reviewer" and "sidebar".
- `PAGES` stays at five entries in this task. Task 4 adds Home.

- [ ] **Step 1: Write the failing tests.** Rewrite the shell tests in `test_app_theme.py`:
  - `test_shell_sidebar`:
    - `at.sidebar.radio` has the key `page`, default `"Run setup"`;
    - the sidebar has the reviewer text input;
    - a sidebar markdown contains `nemi-wordmark`;
    - no markdown contains `ili-shell` or `ili-breadcrumb`.
  - `test_page_header_on_every_page`: for each page, a markdown contains `nemi-header` and `DRAFT_LABEL`.
  - `test_nav_label_counts_pending`: run on a copied synthetic comparison with pending reviews (reuse the review-test fixtures in `tests/test_app_review.py`).
    - The `Match review queue` option label shows `· N`.
    - After `append_review` for one pending row and `at.run()`, it shows `· N-1`.
    - With 0 pending, the label is `Match review`.
  - `test_long_names_truncate`: `page_header("Home", "x"*120)` HTML contains `title="xxx…"` and `nemi-truncate`. Call the helper, capturing `st.markdown`, or test a pure `page_header_html(title, subtitle)` function.

  Remove the `xfail`s from Task 2. Update every other app test's `ili-shell` assertion to the header/badge assertion.
- [ ] **Step 2:** Run `tests/test_app_theme.py tests/test_app_setup.py tests/test_app_review.py tests/test_app_results.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the same files. Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): sidebar navigation and page header with one DRAFT badge`.

### Task 4: Home overview

**Files:**
- Create: `src/ili/app_pages_home.py`
- Modify: `app/streamlit_app.py` (`PAGES` gains `"Home": home_page` first, the default)
- Test: new `tests/test_app_home.py`; page-name lists in `tests/test_app_setup.py:15-16`, `test_app_review.py:16` and `test_app_results.py:14-15` gain "Home"

**Interfaces:**
- **Data classes:**
  - `@dataclass(frozen=True) class Attention: level: Literal["warn", "info", "ok"]; title: str; detail: str; target: str | None`. `target` is a page name from `PAGES`, or `None`.
  - `@dataclass(frozen=True) class HomeView`, with fields:
    - `loaded: bool`;
    - `counts: dict[str, int | None]`, with keys `immediate`, `60_day`, `180_day` and `imp_scheduled`. A value is `None` when classification has not run.
    - `min_rl_yr: float | None`;
    - `min_rl_factored_yr: float | None`;
    - `attention: list[Attention]`;
    - `facts: list[tuple[str, str]]`.
- `home_view(data: ComparisonData | None, path: Path | None) -> HomeView` is pure: no Streamlit calls. Sources:
  - **counts:** `data.classification_summary["counts_by_category"]`;
  - **minimum remaining life:** the folder's `remaining_life_summary.json` keys `min_rl_yr` and `min_rl_factored_yr`, read with `app_state`'s `_json` helper or `json`. A missing file gives `None`.
  - **attention**, in this order. Each item appears only when its condition holds:
    1. `warn` "`N` matches waiting for review": `len(pending_reviews(data.match_table))`, target "Match review queue".
    2. `warn` "`N` review decisions not applied": `app_pages_review.not_applied(data)` rows, target "Match review queue".
    3. `warn` "`N` flags not applied": `data.flag_status.applied_status == "not_applied"`, target "Dig list and report export".
    4. `warn` "Discovery date not entered": `data.classification_summary.get("discovery_warning")`, target "Run setup".
    5. `info` "`N` anomalies could be more severe": the count of `could_be_more_severe` in `data.classification.review_reasons` (the same counting as `classify._has_reason`), target "Dig list and report export".
    6. `info` "Stages not run yet: …": each of match, growth, remaining life and classify whose output is empty in `data`, target "Run setup".
    7. `ok` "`N` features removed after the inspection": `governing_category == "removed"`, target `None`.
  - **facts**, as (label, value) pairs:
    - previous run and current run: date · tool, from `pipeline_inputs.json`;
    - anomalies in the current run, and matched to the previous run with a percentage, from `match_summary.json` (use the keys it has; read it to find them);
    - weld alignment, from `alignment_summary.json`;
    - the growth method (`governing_profile`, shown as "Conservative profile" or "Pit-to-pit (consultant)"), from `growth_summary.json` in the folder;
    - the depth basis ("Reported" or "Reported + tool tolerance"), from `remaining_life_summary.json`;
    - the settings file name.

    A missing source skips that fact; it never raises.
- `home_page() -> None` draws:
  - `page_header("Home", comparison_context(path))`;
  - **when not loaded:** one card, "No comparison loaded yet", with the button `st.button("Start with Run setup", on_click=_go, args=("Run setup",))`. `_go(page)` sets `st.session_state["page"] = page`.
  - **when loaded:**
    - a **Repair conditions** row of five `nemi-kpi` cards (Immediate, 60-day, 180-day, Scheduled by life, Shortest remaining life with "factored X yr" beneath). A card with a count above 0 gets `nemi-kpi--alert`; a `None` count shows "—" with "not run".
    - Then two columns, 3:2.
      - Left: **Needs your attention**, one row per item with a coloured dot and title/detail, plus `st.button(f"Go to {nav_label(target)}", key=f"attn_{i}", on_click=_go, args=(target,))` when it has a target. With no items it says "Nothing needs your attention."
      - Right: **Run facts**, a `nemi-facts` grid.

- [ ] **Step 1: Write the failing tests** in `tests/test_app_home.py`:
  - `test_home_view_unloaded`: `home_view(None, None).loaded is False`.
  - `test_home_view_counts_match_summary`: on a classified synthetic comparison, built as `tests/test_app_results.py` builds its data (reuse its fixture), the counts equal `classification_summary.json["counts_by_category"]` for the four keys, and `min_rl_yr` equals `remaining_life_summary.json["min_rl_yr"]`.
  - `test_home_view_pending_item`: with pending matches, the first attention item is `warn` with target "Match review queue" and N in its title.
  - `test_home_view_old_folder`: delete `remaining_life_summary.json` and `classification_summary.json` from a copied folder. `home_view` returns `None` counts and `None` min RL, and an "Stages not run yet" item or no RL fact, with no exception.
  - `test_home_page_renders`: AppTest. "Home" is the default page; with no comparison the welcome text and button render. Clicking the button switches `at.radio(key="page").value` to "Run setup". On a loaded comparison it renders with no exception, and a markdown contains `nemi-kpi`.
  - Add "Home" to the page-name lists in the other app tests.
- [ ] **Step 2:** Run `tests/test_app_home.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_app_home.py tests/test_app_setup.py tests/test_app_review.py tests/test_app_results.py tests/test_app_theme.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): Home overview`.

### Task 5: Readable table columns

**Files:**
- Create: `src/ili/app_columns.py`
- Modify: the `st.dataframe` calls in the app pages:
  - the explorer table (`app_pages_results.py:58-79`);
  - the dig list and the flags table (`app_pages_results.py`);
  - the changed-sections table (`app_pages_setup.py:309-326`);
  - the review side-by-side table and the "Decisions not applied" table (`app_pages_review.py`).
- Test: new `tests/test_app_columns.py`

**Interfaces:**
- `LABELS: dict[str, tuple[str, str | None]]` maps a column to (label, number format), where `None` means not numeric. It covers every column those tables show. Examples:
  - `curr_log_dist_ft` → ("Distance (ft)", "%.1f")
  - `d_curr_pct` / `curr_depth_pct` → ("Depth now (% WT)", "%.0f")
  - `prev_depth_pct` → ("Depth before (% WT)", "%.0f")
  - `governing_mpy` → ("Growth (mpy)", "%.2f")
  - `governing_method` → ("Growth method", None)
  - `rl_yr` → ("Remaining life (yr)", "%.1f")
  - `erf` → ("ERF", "%.3f")
  - `governing_category` → ("Repair category", None)
  - `due_date` → ("Due", None)
  - `segment_id` → ("Segment", None)
  - `curr_feature_id` → ("Feature", None)
  - `match_id` → ("Match", None)

  The implementer reads each table's real columns and labels all of them in plain words with units.
- `column_config(df: pd.DataFrame) -> dict[str, st.column_config.Column]` returns one entry per column of `df` that is in `LABELS`:
  - numeric columns get `st.column_config.NumberColumn(label, format=fmt)`;
  - other columns get `st.column_config.TextColumn(label)`.

  Columns not in `LABELS` get no entry, so Streamlit shows their raw name.
- `field_label(name: str) -> str` returns `LABELS.get(name, (name.replace("_", " "), None))[0]`. The review side-by-side table uses it for its row labels.
- Every listed `st.dataframe` passes `column_config=column_config(df)` and `hide_index=True`.

- [ ] **Step 1: Write the failing tests:**
  - `test_labels_cover_explorer_columns`: every column of `explorer_rows(...)`'s output on a synthetic comparison is in `LABELS`.
  - `test_column_config_types`: a frame with `rl_yr` (float) and `governing_method` (str) gives a number column with format `%.1f` and a text column. Inspect the returned config dicts.
  - `test_unknown_column_left_raw`.
  - `test_field_label_fallback`: `field_label("foo_bar") == "foo bar"`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement**, and wire it into the tables.
- [ ] **Step 4:** Run `tests/test_app_columns.py tests/test_app_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): readable table columns`.

### Task 6: Page layouts

**Files:**
- Modify: `src/ili/app_pages_setup.py`, `app_pages_review.py`, `app_pages_results.py`
- Test: `tests/test_app_setup.py`, `tests/test_app_review.py`, `tests/test_app_results.py`

**Interfaces** (presentation only; every widget key stays the same so existing tests keep driving them):
- **Run setup.**
  - Each of the four steps and "Advanced" is a bordered container (`st.container(border=True, key="card_step_<n>")`) headed by `app_theme.step_heading(n, title, optional)`.
  - `step_heading` returns HTML with a `nemi-step` badge and the title. Optional steps say "optional".
  - Steps 3 and 4 and Advanced keep their content inside an `st.expander` within the card, collapsed.
  - The "Still needed" list and the Run button sit side by side (`st.columns([3, 1])`).
- **Weld alignment.** The four metrics sit in one `st.columns(4)` row (already true if so). The chart goes through `style_chart`.
- **Match review.**
  - The selected match shows in `st.columns([3, 2])`.
  - Left: a card with the record header, the side-by-side table and the decision action bar.
  - Right: a card with the nearby-features chart, through `style_chart`.
  - The lone "Pending review" metric goes; the record header shows "N of M pending".
  - The "Correct a decision" section moves under an `st.expander("Correct an earlier decision")`.
- **Growth & life explorer.**
  - The three filters sit in one `st.columns(3)` row.
  - "Rows shown" becomes a caption.
  - The two charts sit side by side (`st.columns(2)`), each through `style_chart`.
- **Dig list & export.** `st.tabs(["Dig list", "Manual flags", "Downloads"])`; the Rerun button goes inside the Manual flags tab, after the flag table. The DRAFT captions in the dig list and downloads stay.

- [ ] **Step 1: Write the failing tests:**
  - `test_export_has_tabs`: `at.tabs` labels equal `["Dig list", "Manual flags", "Downloads"]`.
  - `test_review_two_columns`: the review page on pending data has the chart and the Accept button. Assert existing keys are still found; layout itself is not testable beyond presence.
  - `test_every_page_renders_without_data` and `test_every_page_renders_with_data`: all six pages, no exception.
- [ ] **Step 2:** Run. Expected: the tabs test FAILS; the others may already pass.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_app_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): page layouts for the Nemi refresh`.

### Task 7: Spec, decisions, README and close-out

**Files:**
- Modify: `docs/SPEC.md` §6.11, `docs/DECISIONS.md`, `README.md` (the app section)

- [ ] **Step 1: SPEC §6.11.**
  - Add page 0, **Home**: an overview of repair conditions, items needing attention and run facts, with links to the pages.
  - State that the app is branded **Nemi**, that it shows the DRAFT label once per page in the header badge and on every download, and that navigation is a left sidebar.
- [ ] **Step 2: DECISIONS.** Add "UI refresh and Nemi brand (2026-10-29)", recording:
  - the user's choices: audience, clean engineering look with a deep teal accent, sidebar with Home, wordmark, one DRAFT badge, reports name-only;
  - the palette;
  - that the internal `ili` package, the CLI and the JSON `generator` strings keep their names (machine provenance and test-pinned);
  - that the "Enterprise Suite" look and its tab CSS are retired;
  - how the sidebar background is set (from the Task 2 report).
- [ ] **Step 3: README.** Update the app section with the Nemi name, the six pages, and how to start the app (command unchanged).
- [ ] **Step 4: Full suite** in the background, redirected to a log. Expected: all pass. Record the count.
- [ ] **Step 5: Commit** `docs: Nemi UI refresh in SPEC, DECISIONS and README`.
- [ ] **Step 6:** Final review (a fresh Opus reviewer over the branch diff against `main`), one fix pass, a "record the final-review fixes in DECISIONS" commit, a real-app check by the controller, then stop for the user's merge decision.
