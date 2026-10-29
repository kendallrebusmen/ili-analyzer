# ILI Analyzer — Milestone 12 (Parity Check Against the Consultant) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M12 parity check a repeatable program step. `python -m ili.parity` reads the consultant's workbook through a column map, compares it per anomaly with a comparison folder's outputs, and writes every difference with an explanation code. It also closes the last arithmetic gap (pipe age for half-life). Spec §9 M12 is done when the Line 930 run (consultant method, ROSEN data only) is compared and every difference is either `agree` or explained.

**What is already done (merged 2026-10-29, not in this plan):** the consultant-method growth options (`days_per_year`, `rate_decimals`, `shrinking_applies_to`, `joint_percentile_pool`), the `removed_after_run` flag, and the Line 930 settings that select the consultant's method. An ad-hoc script in the session scratchpad found 161 of 175 anomalies exact. This plan turns that script into tested code.

**Architecture:**
- **One new module, `src/ili/parity.py`.** It reads the consultant workbook through `config/consultant_maps/integrity_solutions.yaml` (sheet names, header rows, column names and vocabulary only, no data). It reads our side from the comparison folder with the existing loaders (`load_growth_table`, `load_remaining_life_table`, `load_classification_table`). It joins on current-run feature id and assigns each row one explanation code by a fixed ordered rule. The CLI writes three files into `<comparison folder>/parity/`.
- **Parity is not a pipeline stage.** It is an optional check run after `report`, because it needs a third-party file the pipeline never has.
- **`growth.age_basis`** is a fifth parity option beside the four already merged. Its default keeps every current number unchanged.

**Tech Stack:** unchanged (pandas, numpy, pydantic, openpyxl, PyYAML, pytest). No new dependency.

**Spec:** `docs/SPEC.md` §2 (synthetic only in the repo; real data needs approval, which the user gave for Line 930 on 2026-10-28/29), §6.5 (pit_to_pit and the parity options), §9 M12; `docs/DECISIONS.md` "M12: the consultant's method, reproduced (2026-10-29)".

**Branch:** `m12-parity` from `main`. Every test command uses `.venv/Scripts/python.exe -m pytest`. The full suite takes about 10 minutes.

**Decisions this plan makes where the spec is silent (confirm at plan review; each goes into DECISIONS in Task 5):**
1. **Join key.** The current-run feature id, as text. A numeric id from Excel (`143.0`) becomes `"143"`. Our cluster rows (`"B-1;B-2"`) are split, so each member id carries the cluster's values.
2. **Which consultant rows are anomalies.** The rows whose map-named category column equals the map's `anomaly_category` (`"Volumetric"` for Integrity Solutions). Fittings and other features are dropped before the join.
3. **Agreement tolerances** live in a new `parity:` settings section:
   - depth 0.05 % WT;
   - rate 0.05 mpy, so a 0.1-mpy rounding step still shows as a difference;
   - ERF 0.005;
   - remaining life 0.1 yr or 1 %, whichever is larger.
4. **Explanation codes**, assigned in this order (first that applies):
   1. `vendor_cluster_box`: theirs only, and the vendor anomaly type is in the map's `box_types`.
   2. `not_in_our_run`: any other row that is theirs only.
   3. `removed_after_run`: ours only, and classified `removed`.
   4. `not_in_their_set`: any other row that is ours only.
   5. `agree`: depth, rate, ERF and remaining life all agree.
   6. `depth_differs`.
   7. `match_differs`: one side treats the anomaly as matched to the previous run and the other does not. Theirs counts as matched when its method is `raw` or `shrinking_fixed`; ours when `history_class` is `matched`.
   8. `joint_pool_differs`: both `joint_percentile`, rates disagree.
   9. `age_basis`: both `half_life`, rates disagree.
   10. `rl_horizon`: the rates agree, the remaining lives disagree, and our `rl_governing` is `beyond_horizon`.
   11. `unexplained`: everything else. The CLI exits 0 either way, but prints and records the count.
5. **`age_basis: dates`** measures pipe age from 1 January of the install year to the current run date, in `days_per_year` years. The consultant's 30.0712 years for Line 930 is (2026-01-19 − 1996-01-01) / 365. The default `years` keeps `curr_year − install_year`.
6. **The IMP class comparison** puts the consultant's IM summary sheet beside ours, recounted from `imp_class.csv`: classified, previously repaired, unrepaired per class. It is skipped with a note when `imp_class.csv` is absent (report not run).

## Global Constraints

- All earlier constraints hold. Every output carries `DRAFT – for review by qualified personnel.`; nothing is labelled approved or final.
- **Real data stays out of the repo.** Tests use synthetic data and hand-built workbooks only. The consultant map holds column names and vocabulary, never values. Parity outputs land in the comparison folder under git-ignored `outputs/`. DECISIONS may record aggregate counts (as the M12 entry already does), never per-anomaly values.
- **No hard-coded engineering values:** the tolerances and `age_basis` are in `config/settings.yaml`.
- **Defaults change nothing:** with default settings every existing output is byte-identical on the same input. A test pins this in Task 1.
- **Traceability:** `parity_summary.json` records the sha256 of the consultant workbook and of the map, the comparison folder, and the tolerances used.

## Review Focus

1. **A mapped sheet or column missing from the workbook** → `ValueError` naming the sheet and the column. The CLI exits 2 with that message. Never a KeyError traceback. Test in Task 2.
2. **Duplicate consultant feature ids** after the category filter → `ValueError` listing the ids, never a silent many-to-one join. Test in Task 2.
3. **A growth-source label the map does not know** → method `None`; the row can still be `agree` on numbers. The summary lists the unknown labels with counts. Test in Task 2.
4. **The comparison folder lacks `growth.csv` or `remaining_life.csv`** → exit 2 naming the file. If it lacks only `imp_class.csv`, the per-anomaly table is still written and the IMP comparison is skipped with a note. Test in Task 4.
5. **Blank numbers on one side** (a dent has no rate or remaining life; a consultant row with an empty ERF): the agreement flag for that quantity is `None`, not `False`. `agree` requires each quantity to agree or be blank on both sides. Test in Task 3.

---

### Task 1: Pipe age by dates for half-life (`growth.age_basis`)

**Files:**
- Modify: `src/ili/config.py` (`GrowthCfg`), `config/settings.yaml`, `src/ili/growth.py`, `docs/SPEC.md` §6.5 parity-options list and §7.1 block
- Test: `tests/test_consultant_options.py`

**Interfaces:**
- Produces: `GrowthCfg.age_basis: Literal["years", "dates"] = "years"`; YAML line after `joint_percentile_pool`: `age_basis: years  # years | dates (the consultant): age from 1 Jan of the install year to the run date`.
- Produces: `compute_growth(..., curr_run_date: date | None = None)`, a keyword argument. `age_basis: dates` without it raises `ValueError("growth.age_basis: dates needs the current run date")`.
- Produces: `pipe_age_years(curr_year, install_year, curr_run_date, settings) -> np.ndarray` in `growth.py`. `full_life_rate` and `half_life_rate` take the age from it. The growth CLI passes the current run's date. The summary records `age_basis`.

- [ ] **Step 1: Write the failing tests:**
  - `test_age_basis_default_is_years` (`settings.growth.age_basis == "years"`).
  - `test_pipe_age_by_dates`: install 1996, run date 2026-01-19, `days_per_year=365` → `30.0712` (to 4 places). Same with `years` → `30.0`.
  - `test_half_life_uses_age_basis`: a new anomaly, 13 % of 0.375 in, install 1996, consultant settings plus `age_basis: dates` and `rate_decimals: 1` → `governing_mpy == 3.2` (3.3 with `years`).
  - `test_age_basis_dates_needs_run_date`: `pytest.raises(ValueError, match="run date")`.
- [ ] **Step 2:** Run `tests/test_consultant_options.py -q`. Expected: the four new tests FAIL.
- [ ] **Step 3: Implement** as in Interfaces. Add `age_basis` to `RECORDED_SETTINGS` in `report_data.py` beside the other growth parity keys.
- [ ] **Step 4:** Run `tests/test_consultant_options.py tests/test_growth_table.py tests/test_growth_cli.py tests/test_report_inputs.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(growth): age_basis option (pipe age from dates) for the consultant's half-life`.

### Task 2: Consultant map and workbook reader

**Files:**
- Create: `config/consultant_maps/integrity_solutions.yaml`, `src/ili/parity.py`
- Test: `tests/test_parity_read.py`

**Interfaces:**
- The map file (values are the consultant's header text, as read from their Rev 0 workbook):
  ```yaml
  consultant: integrity_solutions
  anomalies:
    sheet: Classified_Anomalies
    header_row: 1
    category_column: "Feature Category"
    anomaly_category: Volumetric
    columns:
      feature_id: "Feature ID"
      log_dist_ft: "Odometer (ft)"
      joint_id: "Joint Number"
      depth_pct: "Metal Loss Depth (%)"
      prev_depth_pct: "Prev. Anomaly Depth (%)"
      rate_mpy: "Corrosion Growth Rate (MPY)"
      rate_source: "Growth Rate Source"
      erf: "ERF"
      rl_yr: "Remaining Life (Years)"
      vendor_type: "Vendor Anomaly Type"
    rate_sources:
      "Pit to Pit Minimum": raw
      "Fixed Growth Rate": shrinking_fixed
      "Nth Percentile per Joint": joint_percentile
      "Percent Life": half_life
    box_types: ["Cluster"]
  imp_summary:
    sheet: IM Summary
    header_row: 1
    columns: {imp_class: "IMP_ Classification", n_classified: "Classified Anomalies",
              n_previously_repaired: "Previously Repaired", n_unrepaired: "Remaining Unrepaired Anomalies"}
    classes: {"Immediate": "Immediate", "60-day": "60-day", "180-day": "180-day",
              "YTC < 5 yrs": "YTC < 5 yrs", "None": "None"}
  ```
- Produces in `parity.py`:
  - `ConsultantMap`, a pydantic model of that file (strict, extra keys rejected), and `load_consultant_map(path: Path) -> ConsultantMap`.
  - `CONSULTANT_COLUMNS = ("feature_id", "log_dist_ft", "joint_id", "depth_pct", "prev_depth_pct", "rate_mpy", "method", "rate_source", "erf", "rl_yr", "vendor_type")`.
  - `@dataclass(frozen=True) class ConsultantData: anomalies: pd.DataFrame; imp_summary: pd.DataFrame | None; unknown_sources: dict[str, int]`.
  - `read_consultant(path: Path, cmap: ConsultantMap) -> ConsultantData`. `anomalies` has `CONSULTANT_COLUMNS`. `feature_id` is text, normalised as in decision 1; `method` is the mapped source, `None` for blank or unknown labels. `imp_summary` has `imp_class`, `n_classified`, `n_previously_repaired`, `n_unrepaired`, with classes mapped; it is `None` when the map has no `imp_summary` section.

- [ ] **Step 1: Write the failing tests.** A helper `_workbook(tmp_path, rows, imp_rows=None, drop=None)` writes a small `.xlsx` with openpyxl, using the map's sheet and column names, and can drop a column. Tests:
  - `test_reads_anomalies_only`: two Volumetric rows and one "Features & Fittings" row → 2 rows. `feature_id` of Excel `143` (int) and `144.0` (float) → `"143"`, `"144"`.
  - `test_rate_sources_mapped_and_unknown_counted`: sources `Pit to Pit Minimum`, `Fixed Growth Rate`, `Mystery`, blank → methods `raw`, `shrinking_fixed`, `None`, `None`; `unknown_sources == {"Mystery": 1}`.
  - `test_missing_column_names_sheet_and_column`: drop `ERF` → `pytest.raises(ValueError, match="Classified_Anomalies.*ERF")`.
  - `test_missing_sheet`: no `IM Summary` sheet while the map names it → `ValueError` matching `IM Summary`.
  - `test_duplicate_feature_ids`: two rows with id 143 → `ValueError` matching `143`.
  - `test_imp_summary_read`: five class rows → `imp_summary` with the four columns, in map class order.
  - `test_map_rejects_unknown_method`: a `rate_sources` value not in `growth.METHODS` → `ValueError` at load.
- [ ] **Step 2:** Run `tests/test_parity_read.py -q`. Expected: FAIL (no module).
- [ ] **Step 3: Implement.** Read with `pd.read_excel(path, sheet_name=..., header=header_row - 1)`.
- [ ] **Step 4:** Run `tests/test_parity_read.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(parity): consultant map and workbook reader`.

### Task 3: Per-anomaly comparison and explanation codes

**Files:**
- Modify: `src/ili/parity.py`, `src/ili/config.py` (`ParityCfg`, `Settings.parity`), `config/settings.yaml`
- Test: `tests/test_parity_compare.py`

**Interfaces:**
- Consumes: `ConsultantData` (Task 2); our tables as the loaders return them.
- Produces: settings section (YAML after `review:`, model fields with these defaults):
  ```yaml
  parity:                             # M12 comparison with an outside analysis; changes no number in the analysis
    depth_tol_pct: 0.05               # % WT
    rate_tol_mpy: 0.05                # mpy; below the consultant's 0.1-mpy rounding step
    erf_tol: 0.005
    rl_tol_yr: 0.1                    # years; the larger of this and rl_tol_frac x their value
    rl_tol_frac: 0.01
  ```
- Produces in `parity.py`:
  - `EXPLANATIONS`: the eleven codes of decision 4, in order.
  - `PARITY_COLUMNS = ("feature_id", "side", "match_id", "log_dist_ft", "depth_ours", "depth_theirs", "rate_ours", "rate_theirs", "method_ours", "method_theirs", "history_ours", "erf_ours", "erf_theirs", "rl_ours", "rl_theirs", "rl_governing_ours", "category_ours", "vendor_type_theirs", "agree_depth", "agree_rate", "agree_erf", "agree_rl", "explanation")`.
    - `side` is `both`, `theirs_only` or `ours_only`.
    - Ours come from `growth.csv`: `d_curr_pct`, `governing_mpy`, `governing_method`, `history_class`. Also from `remaining_life.csv` (`erf`, `rl_yr`, `rl_governing`) and `classification.csv` (`governing_category`).
    - `log_dist_ft` is theirs where present, else `curr_log_dist_ft`.
  - `ours_by_feature(growth: pd.DataFrame, remaining_life: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame`: one row per current-run member feature id. Cluster ids are split on `;`; growth is joined to remaining life and classification by `match_id`.
  - `compare(ours: pd.DataFrame, theirs: pd.DataFrame, settings: Settings) -> pd.DataFrame`, with columns `PARITY_COLUMNS` and rows sorted by `log_dist_ft`, then `feature_id`.
  - Agreement is `|ours − theirs| ≤ tol`. It is `None` when either side is blank. For remaining life the tolerance is `max(rl_tol_yr, rl_tol_frac × |theirs|)`, and inf compares equal to inf only.

- [ ] **Step 1: Write the failing tests** with hand-built `ours` / `theirs` frames:
  - `test_agree`: equal numbers, plus a rate that differs by 0.04 → `agree`.
  - `test_each_explanation`: one row per code (eleven rows), each built to hit exactly its rule, and the output `explanation` column equals the expected list. The `rl_horizon` row has equal rates, `rl_ours=500.0` with `rl_governing_ours="beyond_horizon"`, and `rl_theirs=612.0`.
  - `test_order_depth_before_match`: depth differs and the history differs → `depth_differs`.
  - `test_blank_quantity_is_none_not_false`: a dent row (ours with no rate or RL; theirs blank too) whose depth and ERF agree → `agree_rate is None`, `agree_rl is None`, explanation `agree`. A row with their ERF blank and ours 0.6 → `agree_erf is None`.
  - `test_cluster_members_share_values`: an ours row with feature id `"B-1;B-2"` gives two rows in `ours_by_feature`.
  - `test_settings_parity_defaults`: the five defaults load from `config/settings.yaml`.
- [ ] **Step 2:** Run `tests/test_parity_compare.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4:** Run `tests/test_parity_compare.py tests/test_config.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(parity): per-anomaly comparison with explanation codes`.

### Task 4: Summary, IMP class comparison and the CLI

**Files:**
- Modify: `src/ili/parity.py`
- Test: `tests/test_parity_cli.py`

**Interfaces:**
- Consumes: Tasks 2–3; `comparison_dir` naming; `ili.config.load_settings`.
- Produces:
  - `imp_class_comparison(ours_imp: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame`. Columns: `imp_class`, `n_classified_ours`, `n_classified_theirs`, `n_previously_repaired_ours`, `n_previously_repaired_theirs`, `n_unrepaired_ours`, `n_unrepaired_theirs`. Ours is recounted from `imp_class.csv`; the classes are in `IMP_CLASS_ORDER = ("Immediate", "60-day", "180-day", "YTC < N yrs", "None")`, with N from `report.ytc_flag_years`. Read `imp_class.csv` with `keep_default_na=False` for `imp_class`: pandas otherwise turns the class label `None` into NaN.
  - `parity_summary(table, consultant, settings, *, workbook_sha256, map_sha256, comparison) -> dict`: the DRAFT `label`, `generator`, `created_utc`, the workbook and map sha256, `comparison`, `tolerances`, `counts_by_side`, `counts_by_explanation` (every code, zeros included), `n_agree`, `n_unexplained`, `agree_counts` per quantity (true/false/none), `unknown_sources`, `imp_class_compared` (bool) and `imp_class_note`.
  - `main(argv) -> int`, `python -m ili.parity --comparison <folder> --consultant <xlsx> [--map config/consultant_maps/integrity_solutions.yaml] [--settings config/settings.yaml]`. It writes `<comparison>/parity/parity_by_anomaly.csv`, `parity_summary.json` and `parity_imp_class.csv` (the last only when compared). It prints the label, the counts by explanation and the unexplained count. Exit 0 on success, 2 on any input error, with the message on stderr.
- [ ] **Step 1: Write the failing tests:**
  - `test_cli_end_to_end`, on the `plain` fixture from `tests/test_review_flags.py` (import it, as that module imports its own helpers), copied to `tmp_path`. Run `classify` and then `report` on the copy the way `test_report_shows_review_and_flag_logs` does, so that `imp_class.csv` exists. A helper builds a consultant workbook from that folder's own `growth.csv` and `remaining_life.csv`, with three edits:
    - one anomaly's rate +1.0 mpy;
    - one anomaly removed;
    - one extra "Cluster" row.

    Asserts:
    - exit 0;
    - the three files exist;
    - `counts_by_explanation` has `vendor_cluster_box == 1`, `not_in_their_set == 1` and exactly one non-`agree` row among the shared ones;
    - the workbook sha256 equals the file's;
    - the CSV's first line is the column header and the label is in the JSON.
  - `test_cli_missing_growth_exits_2`: delete `growth.csv` → exit 2 and stderr names `growth.csv`.
  - `test_cli_without_imp_class_skips_that_table`: delete `imp_class.csv` → exit 0, no `parity_imp_class.csv`, `imp_class_compared` is false and `imp_class_note` is non-empty.
  - `test_cli_bad_workbook_exits_2`: a workbook without the mapped sheet → exit 2.
- [ ] **Step 2:** Run `tests/test_parity_cli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_parity_read.py tests/test_parity_compare.py tests/test_parity_cli.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(parity): parity summary, IMP class comparison and CLI`.

### Task 5: Line 930 parity run, docs and close-out

**Files:**
- Modify: `README.md`, `docs/SPEC.md` (§9 M12 row: done; §6.5 list gains `age_basis`), `docs/DECISIONS.md`
- Not committed: `data/raw/l930/settings_l930.yaml` gains `age_basis: dates`; outputs under `outputs/l930_vendor/`.

- [ ] **Step 1: Line 930 (real data, git-ignored).**
  - Add `age_basis: dates` to `data/raw/l930/settings_l930.yaml`.
  - Rerun the Line 930 pipeline with the command recorded in `outputs/l930_vendor/2020_rosen_vs_2026_rosen/pipeline_inputs.json`.
  - Run `python -m ili.parity --comparison outputs/l930_vendor/2020_rosen_vs_2026_rosen --consultant data/raw/l930/consultant/is_l930_2026_rev0.xlsx`.
  - Expected: no `unexplained` rows, and 0 `age_basis` rows (the 3 half-life residuals close). `match_differs` and `joint_pool_differs` remain, as known. Report the counts by explanation; if any row is `unexplained`, stop and report it rather than adjusting tolerances.
- [ ] **Step 2: Full suite.** Run `.venv/Scripts/python.exe -m pytest tests -q`. Expected: all pass. Record the count.
- [ ] **Step 3: Docs.**
  - `README.md`: a "Parity check" section with the CLI command, the three outputs and the explanation codes.
  - SPEC §9: M12 marked done. §6.5: `age_basis` added to the parity-options list.
  - DECISIONS: "M12 close-out", with decisions 1–6 above and the Line 930 counts by explanation (aggregates only).
- [ ] **Step 4: Commit** `docs: M12 parity check, decisions and README`.
- [ ] **Step 5:** Final review (fresh Opus reviewer over the branch diff against `main`), one fix pass, a "record the final-review fixes in DECISIONS" commit, then stop for the user's merge decision.
