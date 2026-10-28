# ILI Analyzer — Milestone 11 (Consultant-Practice Methods and Outputs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything the 2026-10-27 spec amendment adds, on synthetic data only. Spec §9 M11 is done when:
- `ml_origin` is ingested;
- the `half_life`, `joint_percentile` and `shrinking_fixed` rates and the `pit_to_pit` profile are computed, with tests;
- ERF, FPR, failure mode, condition dates and next-assessment projections are in the outputs;
- the report has the growth summary, IMP class summary, depth and dent bins, per-joint counts, anomalies of interest, and the orientation and elevation charts.

**Architecture:**
- **Data flow.** Origin flows from the vendor map (`ingest`) through the match table (`prev_/curr_ml_origin`) into `growth.csv`. New rates are pure vectorised functions in `growth.py` beside the existing ones. `governing_rate` dispatches on `growth.governing_profile`.
- **Remaining life.** `remaining_life.py` gains one vectorised helper, `strength_at`, for ERF/FPR now and at the next assessment. It also gains `failure_mode` and `condition_date`.
- **Report.** The new tables are builders in a new `report_tables.py`; `report_data.py` is already 444 lines. They read the tables as loaded and recompute nothing that a stage already wrote. `report.py` wires them into the template and writes the three new CSVs. New charts go in `report_charts.py`.
- **Repairs.** `repairs_between_runs.csv` now also reaches `growth`, for recoat exclusion, and `report`, for the IMP class summary and the anomalies of interest. `pipeline.py` passes it.

**Tech Stack:** unchanged (pandas, numpy, scipy, pydantic, plotly, jinja2, openpyxl, pytest). No new dependency.

**Spec:** `docs/SPEC.md` §5.1 (`ml_origin`), §6.5–6.8, §6.10, §7.1, §9 M11; `docs/DECISIONS.md` "Spec amendment: the consultant's practice (2026-10-27)".

**Branch:** `m11-consultant-practice` from `main`. Every test command uses `.venv/Scripts/python.exe -m pytest`. The full suite takes about 5 minutes.

**Decisions this plan makes where the spec is silent (confirm at plan review; each one goes into DECISIONS in Task 16):**
1. **Profiles.** `conservative` is the existing `growth.governing` lists, unchanged. `pit_to_pit` is fixed logic exactly as spec 6.5 words it, and cannot be configured as lists.
2. **pit_to_pit "grew".** A depth change Δd ≥ 0 counts as grew: `raw` floored at `min_rate_mpy`. Δd < 0 is decreased: `shrinking_fixed`.
3. **Manufacturing features still get growth rates.** They also stay in the segment and joint percentile distributions; spec 6.5 does not exclude them. "Corrosion" in the growth summary and depth-bin columns means `ml_origin != manufacturing`, so `unknown` counts as corrosion, the conservative side.
4. **A cluster's origin** is the members' common origin, else `unknown`.
5. **Failure mode** is blank when τ_80 or τ_burst is NaN, or when both are beyond the horizon (inf). ERF and FPR are blank where P_safe or MOP is not a finite positive number.
6. **IMP class "YTC < N".** It uses `rl_yr`, else `rl_upper_bound_yr` (a proven bound, as M6 does for digs). A row with neither falls in "None" and is counted in a separate `n_rl_not_evaluated` column.
7. **"Previously repaired"** means the current-run position lies in any `repairs_between_runs.csv` span, of any type, ends inclusive. The column is blank when no repairs file is given.
8. **Bin edges follow the spec wording.** Depth bins are left-closed (`[70, 80)`, `>= 80`). Dent bins are right-closed, because the regulation says "deeper than": `[0, 2]`, `(2, 6]`, `> 6`.
9. **Alignment ignores `recoat` and `sleeve` spans.** Neither changes joints. Match's `in_repaired_section` hint still uses every span.
10. **The synthetic dataset changes** (mill features, one recoat), so its feature ids renumber. The user's saved decisions in `outputs/2021_vendorA_vs_2026_vendorB/` are not touched: nothing in this plan regenerates `data/synthetic/` or runs into that folder.

## Global Constraints

- All M1–M9 constraints hold. Every output carries `DRAFT – for review by qualified personnel.` and nothing is labelled approved or final.
- **Synthetic data only** (spec §2). M12 (real data) is not part of this plan.
- **No hard-coded engineering values.** Every new threshold is in `config/settings.yaml` with units and a `PLACEHOLDER` comment where spec 7.1 has one. The keys are exactly spec 7.1's: `growth.governing_profile`, `shrinking_rate_mpy`, `joint_percentile`, `joint_percentile_min_matches`, `exclude_recoated_from_stats`; `report.ytc_flag_years`, `depth_bins_pct_wt`, `dent_bins_pct_od`, `aoi_group_ft`, `aoi_count`. Default values are copied from spec 7.1.
- **`conservative` stays the default profile**, and `depth_basis` stays `reported_plus_tolerance`. With default settings, every existing number is unchanged on the same input. A test pins this in Task 6.
- **Traceability:** every new per-feature value is a column in the stage's CSV. The report reads it from there and never recomputes it.
- **Fully local:** charts inline, nothing fetched.
- **Acceptance targets still hold** on the changed synthetic data: alignment ≥ 99%, precision ≥ 98%, recall ≥ 95%, conservatism ≥ 90%. If a test pins a count that the new synthetic features change, update the pin and say so in the task report. Never loosen a target.

## Review Focus

1. **Vendor map origin vocabulary.** An `ml_origins` key that is not a `feature_types` key, or that maps a code whose type is not `metal_loss`, is a load-time `ValueError` naming the code. It never makes an origin silently `unknown`. Test in Task 1.
2. **Joint percentile unavailable.** A joint with fewer than `joint_percentile_min_matches` matched features, or whose only matches are recoated with exclusion on, has NaN `joint_percentile`. Under `pit_to_pit` a new feature there falls back to `half_life`. If its install year is also unknown, the row is `no_governing` and flagged, never an exception. Test in Tasks 5–6.
3. **`shrinking_rate_mpy: null` under `pit_to_pit`.** Shrinking rows are `no_governing`, flagged, and get no RL (`rl_not_evaluated`). They never silently take a rate of 0. Test in Task 6.
4. **Optional inputs absent.** With no repairs file, no `elev_ft`, or `girth_weld_proximity_in` / seam settings null, the report still renders. Each affected table says why it is incomplete instead of failing or showing zeros as fact. Test in Tasks 9–14.
5. **Values exactly on a bin edge** (10.0 % and 80.0 % of wall; 2.0 % and 6.0 % of OD) land in the bin decision 8 gives. Test in Task 10.

---

### Task 1: `ml_origin` in the schema and ingestion

**Files:**
- Modify: `src/ili/schema.py`, `src/ili/ingest.py`, `config/vendor_maps/synthetic_vendor_a.yaml`, `config/vendor_maps/synthetic_vendor_b.yaml`
- Test: `tests/test_ingest_map.py`, `tests/test_ingest_standardize.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `schema.MlOrigin(StrEnum)` with `corrosion`, `manufacturing`, `unknown`. `StandardFeature.ml_origin: MlOrigin | None = None` goes directly after `surface`, so `STANDARD_COLUMNS` gains `ml_origin` there. `VendorMap.ml_origins: dict[str, MlOrigin] = {}` is keyed by vendor feature-type code and normalized like `feature_types`. The standardized CSV gains the `ml_origin` column.

- [ ] **Step 1: Write the failing tests**
  - `test_ml_origin_from_vendor_code`: in a map with `feature_types {"ML-CORR": metal_loss, "ML-MILL": metal_loss}` and `ml_origins {"ML-CORR": corrosion, "ML-MILL": manufacturing}`, rows `ML-CORR`, `ML-MILL`, `DENT` standardize to `corrosion`, `manufacturing`, None.
  - `test_ml_origin_defaults_unknown`: a metal-loss code absent from `ml_origins`, or a map without the key, gives `unknown`.
  - `test_ml_origins_key_must_be_metal_loss_code`: `ml_origins {"DENT": corrosion}` raises `ValueError` with `"DENT"` in the message. `ml_origins {"NOPE": corrosion}`, a code not in `feature_types`, raises too.
  - `test_load_standardized_without_ml_origin_column`: a standardized CSV written before M11 loads with `unknown` for metal-loss rows and None for the rest.
  - Add `ml_origin` to `EXACT` in `tests/conftest.py`, so the synthetic round-trip covers it.
- [ ] **Step 2:** Run `.venv/Scripts/python.exe -m pytest tests/test_ingest_map.py tests/test_ingest_standardize.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Add the enum and the field.
  - Add `VendorMap.ml_origins` with a `model_validator(mode="after")` that enforces Review Focus 1.
  - In `standardize`, look up `ml_origin` by the same normalized feature-type key. Then set it to None for any row whose mapped type is not `metal_loss`.
  - `load_standardized`: `ml_origin` is a text column, and the absent-column fallback is the one the tests describe.
  - Vendor maps: A gains `"Mill Anomaly": metal_loss` and `ml_origins {"Metal Loss": corrosion, "Mill Anomaly": manufacturing}`. B gains `"ML-MILL": metal_loss` and `ml_origins {"ML-CORR": corrosion, "ML-CLUS": corrosion, "ML-MILL": manufacturing}`.
- [ ] **Step 4:** Run the same tests, then `tests/test_ingest_*.py tests/test_schema.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingest): ml_origin from the vendor map (spec 5.1)`.

### Task 2: Synthetic mill features and a recoat; alignment ignores non-joint repairs

**Files:**
- Modify: `src/ili/synth.py`, `src/ili/align_welds.py`
- Test: `tests/test_synth_features.py`, `tests/test_synth_outputs.py`, `tests/test_align_robust.py`

**Interfaces:**
- Consumes: the Task 1 `ml_origin` column and vendor codes.
- Produces:
  - `SynthParams.n_mill: int = 10`. True growth is 0, depth is uniform in `MILL_DEPTH_PCT = (10.0, 25.0)` (above the threshold, so reported in both runs), and the surface is EXT or INT at random.
  - The std output's `ml_origin` is `manufacturing` for mill features, `corrosion` for other metal loss, None otherwise.
  - Vendor A writes `"Mill Anomaly"`; vendor B writes `"ML-MILL"`.
  - `repairs_between_runs.csv` gains one `recoat` row. It covers `RECOAT_JOINTS = 8` consecutive normal joints starting at the first normal joint at or after `RECOAT_AT = 0.3` of line length, clear of the replacement and of fittings. It is placed deterministically, with no new random draw. Its `repair_date` is the midpoint, the same as the replacement's.
  - `align_welds` uses only spans whose `repair_type` is `cutout` or `pipe_replacement`. A repairs table without `repair_type` keeps every span, which is the current behaviour.

- [ ] **Step 1: Write the failing tests**
  - `test_mill_features_reported_both_runs` (in the `synth` fixture): `n_mill` manufacturing features in each run, all with ground-truth growth 0.
  - `test_recoat_row_covers_metal_loss`: exactly one `recoat` row, and at least one current-run metal-loss feature lies in it.
  - `test_vendor_codes_round_trip`: ingesting the written tallies gives the generator's `ml_origin`. This is covered by `EXACT` once Task 1 has landed.
  - `test_alignment_ignores_recoat_span`: aligning with only a recoat span gives the same weld map as aligning with no repairs.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Add the mill features as one more `_reported_ml`-style block in `build_features`, with population `"mill"`, after the internal block.
  - Add the deterministic recoat row in `build_reference`.
  - Map the vendor codes in `to_vendor_a` and `to_vendor_b` by `(feature_type, ml_origin)`.
  - Filter the spans in `align_welds`.
- [ ] **Step 4:** Run `tests/test_synth_*.py tests/test_align_*.py tests/test_match_*.py tests/test_growth_score.py -q`. Expected: PASS, with the acceptance targets met. Update any pinned count the new features change and list each one in the report.
- [ ] **Step 5: Commit** `feat(synth): mill features and a recoat repair; alignment skips non-joint repairs`.

### Task 3: Match carries `ml_origin`

**Files:**
- Modify: `src/ili/match.py`
- Test: `tests/test_match_table.py`, `tests/test_match_clusters.py`

**Interfaces:**
- Produces: `"ml_origin"` appended to `SIDE_FIELDS`, so the match table has `prev_ml_origin` and `curr_ml_origin`. `load_match_table` reads both as text.

- [ ] **Step 1: Write the failing tests**
  - `test_match_table_has_origin`: a 1:1 match of two `manufacturing` features carries `manufacturing` on both sides.
  - `test_cluster_origin_common_else_unknown`: an envelope of members `{corrosion, corrosion}` is `corrosion`; of `{corrosion, manufacturing}` is `unknown` (decision 4).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement** the field and the envelope rule in `match.envelope`. Add both columns to `load_match_table`'s text columns.
- [ ] **Step 4:** Run `tests/test_match_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(match): carry ml_origin on both sides`.

### Task 4: Growth settings

**Files:**
- Modify: `config/settings.yaml`, `src/ili/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces, in `GrowthCfg`:
  - `governing_profile: Literal["conservative", "pit_to_pit"]`
  - `shrinking_rate_mpy: float | None` with `Field(ge=0)` when set
  - `joint_percentile: float` in `[0, 100]`
  - `joint_percentile_min_matches: int` with `Field(strict=True, ge=1)`
  - `exclude_recoated_from_stats: bool`
- Produces: `GrowthMethod` gains `half_life`, `joint_percentile` and `shrinking_fixed`, so `growth.governing` lists may name them.
- The YAML values and comments are copied from spec 7.1, after `default_rate_mpy`. A comment on `governing:` says it is the `conservative` profile's rule.

- [ ] **Step 1: Write the failing tests:** `test_growth_profile_values`, which checks the defaults load as in spec 7.1 and an unknown profile is rejected; `test_joint_percentile_bounds`, which rejects 101 and -1; and `test_min_matches_positive_int`, which rejects 0 and 2.5.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_config.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(config): growth profile and consultant-practice growth settings`.

### Task 5: `half_life`, `joint_percentile` and `shrinking_fixed` rates; recoat exclusion

**Files:**
- Modify: `src/ili/growth.py`, `src/ili/pipeline.py`
- Test: `tests/test_growth_methods.py`, `tests/test_growth_table.py`, `tests/test_growth_cli.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3 match columns, Task 4 settings.
- Produces:
  - `half_life_rate(d_curr_in, curr_year, install_year) -> ndarray`: mpy = d × 1000 / ((curr − install) / 2), NaN when the age is ≤ 0 or unknown.
  - `joint_percentile(raw_mpy, cls, joint_id, in_stats, settings) -> tuple[ndarray, dict]`. Per row, it is the `joint_percentile`-th percentile (linear) of the raw rates of `matched` rows with the same `joint_id` and `in_stats` True, floored at `min_rate_mpy`. It is NaN when fewer than `joint_percentile_min_matches` such rows exist or the row has no joint. Also returns the unfloored values by joint.
  - `shrinking_fixed_rate(d_prev_in, d_curr_in, cls, settings) -> ndarray`: `shrinking_rate_mpy` where the row is `matched` and d_curr < d_prev, else NaN, and NaN everywhere when the setting is null.
  - `in_recoat(pos_ft, repairs) -> ndarray[bool]`: inside any span with `repair_type == "recoat"`, ends inclusive. All False when `repairs` is None.
  - `segment_percentile(...)` gains an `in_stats` argument that keeps excluded rows out of the distribution while still giving them the rate.
  - `compute_growth(..., repairs: pd.DataFrame | None = None)`.
- New `FEATURE_COLUMNS`, inserted after `curr_log_dist_ft` and `segment_id`: `joint_id` (from `curr_us_gw_id`), `surface` (`curr_surface`), `ml_origin` (`curr_ml_origin`), `in_recoat`, `in_stats`. The three new `<method>_mpy` columns follow the existing ones, in `METHODS` order.
- `in_stats = ~(in_recoat & exclude_recoated_from_stats)`.
- The summary gains `governing_profile`, `n_excluded_recoated`, `joint_percentile`, `joint_percentile_min_matches` and `shrinking_rate_mpy`.
- The growth CLI gains `--repairs`, and `pipeline._argv("growth")` passes `--repairs` when set.

- [ ] **Step 1: Write the failing tests**
  - `test_half_life_is_twice_full_life`: d = 0.1 in at age 40 gives 5.0 mpy; age 0 gives NaN.
  - `test_joint_percentile_min_matches`: a joint with raw rates [1, 2, 3, 4] at P90 and min 3 gives 3.7 for every row in it. With min 5 it gives NaN. A negative P is floored at `min_rate_mpy`.
  - `test_shrinking_fixed_only_when_depth_drops`: rows (prev 0.10, curr 0.08) and (0.10, 0.12) at `shrinking_rate_mpy` 6 give [6, NaN]. With the setting null, both are NaN.
  - `test_recoated_rows_left_out_of_distributions`: a recoated row with raw 50 mpy does not move the segment or joint percentile when exclusion is on, and does when it is off. It gets the rate either way, and `summary["n_excluded_recoated"]` counts it.
  - `test_growth_cli_repairs_optional`: the CLI runs with and without `--repairs`.
  - `test_pipeline_passes_repairs_to_growth`.
- [ ] **Step 2:** Run `tests/test_growth_methods.py tests/test_growth_table.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions, columns and CLI flag. `load_growth_table` reads `joint_id`, `surface` and `ml_origin` as text and `in_recoat` and `in_stats` as flags.
- [ ] **Step 4:** Run `tests/test_growth_*.py tests/test_pipeline.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(growth): half-life, joint-percentile and shrinking-fixed rates; recoat exclusion`.

### Task 6: The `pit_to_pit` governing profile

**Files:**
- Modify: `src/ili/growth.py`
- Test: `tests/test_growth_methods.py`, `tests/test_growth_table.py`

**Interfaces:**
- Produces: `governing_rate(rates, cls, settings)` keeps its signature and dispatches on `governing_profile`.
- `pit_to_pit` follows spec 6.5 and decision 2:
  - `matched` with d_curr ≥ d_prev: `max(raw, min_rate_mpy)`, method `raw`.
  - `matched` with d_curr < d_prev: `shrinking_fixed`.
  - `new`: `joint_percentile` when finite, else `half_life`.
  - `no_history`: `half_life`.
  - A NaN chosen rate gives NaN and method None.
- `review_flags` sets `incomplete_rule` only under `conservative`. Under `pit_to_pit`, a fallback is by design and `no_governing` catches a missing rate.

- [ ] **Step 1: Write the failing tests**
  - `test_pit_to_pit_matched_grew_and_shrank`: raw 0.4 mpy with Δd > 0 gives `min_rate_mpy` (1.0), method `raw`. Raw 3.0 gives 3.0. Δd < 0 gives `shrinking_rate_mpy`, method `shrinking_fixed`.
  - `test_pit_to_pit_new_falls_back_to_half_life`.
  - `test_pit_to_pit_shrinking_null_is_flagged`: the row's governing rate is NaN and its `review_reasons` contains `no_governing` (Review Focus 3).
  - `test_conservative_default_unchanged`: `compute_growth` on the `synth` fixture's match table with default settings gives `governing_mpy` and `governing_method` equal to a run with the three new methods' columns dropped. This is the regression pin from Global Constraints.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_growth_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(growth): pit_to_pit governing profile (spec 6.5)`.

### Task 7: Growth summary by surface (`growth_summary.csv`)

**Files:**
- Modify: `src/ili/growth.py`
- Test: `tests/test_growth_table.py`, `tests/test_growth_cli.py`

**Interfaces:**
- Produces: `growth_summary_by_surface(features: pd.DataFrame, curr_year: int) -> pd.DataFrame`. It has one row per (`surface` in EXT, INT) × (`scope` in `all`, `excluding_recoated`), over rows with `ml_origin != "manufacturing"` (decision 3). Columns:
  - `surface`, `scope`
  - `n_features`, `n_matched`, `matched_share` (text `"n of N"`)
  - `max_depth_pct`, `max_depth_in`, `deepest_curr_feature_id`, `pipe_age_yr` (curr_year − install_year of the deepest)
  - `deepest_half_life_mpy`, `deepest_full_life_mpy`
  - `max_raw_mpy`, `mean_depth_change_pct` (mean of d_curr_pct − d_prev_pct over matched), `mean_raw_mpy`
- Blank (NaN) where there are no rows. `GrowthResult` gains `by_surface`, and `write_growth` writes `growth_summary.csv`.

- [ ] **Step 1: Write the failing tests**
  - `test_growth_summary_by_surface_values`: a hand-built table of 3 EXT corrosion rows (2 matched, one of them recoated) and 1 INT row gives exact values for both scopes. Its `matched_share` is `"2 of 3"`, then `"1 of 2"` excluding recoated.
  - `test_manufacturing_left_out`.
  - `test_empty_surface_row_blank`.
  - `test_growth_cli_writes_summary_csv`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_growth_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(growth): growth summary by surface, with and without recoated areas`.

### Task 8: ERF, FPR, failure mode, condition dates and next-assessment projections

**Files:**
- Modify: `src/ili/remaining_life.py`, `src/ili/report_data.py` (`DIG_LIST_COLUMNS`)
- Test: `tests/test_rl_solver.py`, `tests/test_rl_table.py`, `tests/test_report_tables.py`

**Interfaces:**
- Produces:
  - `strength_at(tau, d0, L0, g, gL, D, t, smys, mop, settings) -> dict[str, ndarray]`. Elementwise, it returns keys `depth_in`, `length_in`, `burst_psig`, `safe_psig`, `erf`, `fpr`. It uses `projected_depth_length`, `usable_inputs` and the configured method. ERF = MOP / P_safe and FPR = P_F / MOP, NaN per decision 5.
  - `failure_mode(tau_80, tau_burst) -> ndarray[object]` gives `leak` where τ_80 ≤ τ_burst, `burst` where τ_80 > τ_burst, and None per decision 5.
  - `condition_date(run_date, tau_yr) -> date | None` gives run_date + floor(τ × 365.25) days, None for NaN or inf.
- New `RL_FEATURE_COLUMNS`:
  - after `safe_psig`: `erf`, `fpr`;
  - after `rl_governing`: `failure_mode`;
  - after `tau_safe_yr`: `tau_80_date`, `tau_burst_date`, `tau_safe_date`;
  - after `years_to_next_assessment`: `depth_next_in`, `depth_next_pct`, `burst_next_psig`, `safe_next_psig`, `erf_next`, `fpr_next`.
- `load_remaining_life_table` reads the three condition dates as dates and `failure_mode` as text.
- `DIG_LIST_COLUMNS` gains `erf` and `fpr` after `mop_local`, `burst_next_psig`, `safe_next_psig`, `erf_next` and `fpr_next` after those, and `failure_mode` after `rl_governing`. `dig_list` looks them up by `match_id` from `remaining_life`.

- [ ] **Step 1: Write the failing tests**
  - `test_strength_at_reference_case`: with the spec 6.6 reference case (Modified B31G, P_F ≈ 1,771 psig), design factor 0.72 and MOP 1,000, at τ = 0 ERF = 1000 / (0.72 × P_F) and FPR = P_F / 1000, to 1e-9 relative.
  - `test_strength_at_projects_growth`: g = 0.01 in/yr at τ = 5 deepens by 0.05 in and lowers P_F.
  - `test_erf_blank_for_bad_mop`: MOP ≤ 0 or NaN gives NaN ERF and FPR.
  - `test_failure_mode`: pairs (3, 5) → leak, (5, 5) → leak, (6, 5) → burst, (inf, inf) → None, (NaN, 5) → None.
  - `test_condition_dates`: τ = 0 → run date; τ = 1.0 from 2026-06-15 → 2027-06-15; inf → None.
  - `test_rl_table_new_columns` on the `synth` pipeline fixture: every finite-RL row has a `failure_mode`, and `depth_next_in` ≥ `d_assessed_in`.
  - `test_dig_list_has_erf_fpr`.
- [ ] **Step 2:** Run `tests/test_rl_*.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `compute_remaining_life` calls `strength_at` once at τ = 0, for `erf` and `fpr`, and once at `years_to_next_assessment`.
- [ ] **Step 4:** Run `tests/test_rl_*.py tests/test_report_tables.py tests/test_classify*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(remaining_life): ERF, FPR, failure mode, condition dates, next-assessment projections`.

### Task 9: Report settings, repairs and weld map in the report inputs; IMP class summary

**Files:**
- Modify: `config/settings.yaml`, `src/ili/config.py`, `src/ili/report_data.py`, `src/ili/report.py`, `src/ili/pipeline.py`, `src/ili/templates/report.html.j2`
- Create: `src/ili/report_tables.py`
- Test: `tests/test_config.py`, `tests/test_report_inputs.py`, `tests/test_report_tables.py`, `tests/test_report_html.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces, in `ReportCfg` (values from spec 7.1, presentation only):
  - `ytc_flag_years: float` (> 0)
  - `depth_bins_pct_wt: list[float]` (strictly increasing, starting at 0)
  - `dent_bins_pct_od: list[float]` (same rule)
  - `aoi_group_ft: float` (≥ 0)
  - `aoi_count: _PositiveInt`
- Produces, in `ReportInputs`: `repairs: pd.DataFrame | None`, `weld_map: pd.DataFrame` (loaded with `align_welds.load_weld_map`, required) and `growth_by_surface: pd.DataFrame` (`growth_summary.csv`, required).
  - `load_report_inputs(..., repairs: Path | None)` gets a new keyword. The report CLI gains `--repairs`, and `pipeline._argv("report")` passes it.
  - `RECORDED_SETTINGS` gains `("growth", "governing_profile", "growth.governing_profile")`.
- Produces: `report_tables.imp_class_summary(classification, remaining_life, repairs, settings) -> pd.DataFrame`.
  - Rows in order: `Immediate`, `60-day`, `180-day`, `YTC < {ytc_flag_years:g} yrs`, `None`.
  - Columns: `imp_class`, `n_classified`, `n_previously_repaired`, `n_unrepaired`, `n_rl_not_evaluated`.
  - Classes follow decision 6, and repaired follows decision 7.
- Template:
  - The executive summary gains an "IMP class summary" table under "Counts by repair category", with a line saying it is a summary view and that categories drive due dates.
  - The growth section states the governing profile from `summaries["growth"]["governing_profile"]`. It shows `growth_by_surface` as a "Growth summary by surface" table, with the number of recoated features left out (`summaries["growth"]["n_excluded_recoated"]`).

- [ ] **Step 1: Write the failing tests**
  - `test_report_settings_defaults` and `test_bins_must_increase`.
  - `test_report_refuses_changed_profile`: a growth summary recording `pit_to_pit` with settings saying `conservative` gives a `ValueError` naming `growth.governing_profile`.
  - `test_imp_class_summary_rows`: hand-built rows give each class correctly. A row with no repair category, NaN RL and bound 3 lands in YTC < 5; one with neither lands in None, counted in `n_rl_not_evaluated`. A row inside a repair span counts as repaired.
  - `test_imp_class_summary_no_repairs_file`: `n_previously_repaired` and `n_unrepaired` are blank (Review Focus 4).
  - `test_report_html_has_imp_class_and_profile`: the HTML also has the growth-by-surface table.
  - `test_pipeline_passes_repairs_to_report`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_report_*.py tests/test_config.py tests/test_pipeline.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): IMP class summary; repairs, weld map and growth summary in the report inputs`.

### Task 10: Depth bins and dent bins (`anomaly_distribution.csv`)

**Files:**
- Modify: `src/ili/report_tables.py`, `src/ili/report.py`, `src/ili/templates/report.html.j2`
- Test: `tests/test_report_tables.py`, `tests/test_report_html.py`

**Interfaces:**
- Consumes: `classify.near_girth_weld`, `classify.seam_interaction`, and `match.PositionMap.from_weld_map(...).to_curr`, which places previous-run dents on the current-run reference for the OD lookup.
- Produces:
  - `depth_bins(std, settings) -> pd.DataFrame` for one run's metal loss.
    - Columns: `run_id`, `location` (`all`, `girth_weld`, `long_seam`), `bin` (labels `0-10` … `70-80`, `>=80` from `depth_bins_pct_wt`, left-closed), `ext_corrosion`, `int_corrosion`, `manufacturing`, `total`.
    - Corrosion means `ml_origin != manufacturing` on that surface. `total` includes MID and UNK.
    - `girth_weld` uses `near_girth_weld` with that run's own welds; `long_seam` uses `seam_interaction`.
    - Unknown (None) is not counted; the count of unknown rows is reported separately (below).
  - `dent_bins(std, od_in, settings) -> pd.DataFrame` for one run's dents, with `dent_depth_pct_od` = `dent_depth_in` / `od_in` × 100, right-closed per decision 8.
    - `od_in` is aligned with `std`'s rows. `anomaly_distribution` gets it with `reference.lookup(pipe, pos, "od_in", alignment.anchor_tol_ft)`. The previous run's positions are first mapped to the current run with `PositionMap.to_curr`, since reference files are keyed to current-run distance.
    - Columns: `run_id`, `bin`, `all`, `girth_weld`, `long_seam`.
  - `anomaly_distribution(inputs) -> AnomalyDistribution` holds both runs' tables and `notes: list[str]`. The notes say which locations could not be evaluated and why, e.g. `"girth-weld proximity: classification.girth_weld_proximity_in is not set"`. Each note gives the count of unknown rows.
  - `report.py` writes `anomaly_distribution.csv`: the depth and dent rows stacked, with a `table` column `depth` or `dent`.

- [ ] **Step 1: Write the failing tests**
  - `test_depth_bin_edges`: depths 9.99, 10.0, 79.99 and 80.0 fall in `0-10`, `10-20`, `70-80` and `>=80`.
  - `test_depth_bin_columns`: an EXT corrosion, an INT unknown-origin and an EXT manufacturing feature give `ext_corrosion` 1, `int_corrosion` 1, `manufacturing` 1 and `total` 3.
  - `test_dent_bin_edges`: 2.0 % OD falls in `0-2`, 2.01 in `2-6`, 6.0 in `2-6`, and 6.01 in `>6`.
  - `test_girth_weld_table_not_evaluated_note`: with proximity null, the `girth_weld` counts are 0 and a note names the setting.
  - `test_report_writes_anomaly_distribution`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement**, with a "Depth bins" and a "Dent bins" subsection in the growth and anomaly distribution section that shows the notes.
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): depth-bin and dent-bin tables (spec 6.10)`.

### Task 11: Per-joint counts (`joint_ml_counts.csv`)

**Files:**
- Modify: `src/ili/report_tables.py`, `src/ili/report.py`, `src/ili/templates/report.html.j2`
- Test: `tests/test_report_tables.py`

**Interfaces:**
- Produces: `joint_ml_counts(weld_map, prev_std, curr_std) -> pd.DataFrame`, one row per weld-map row (decision 13).
  - Columns: `section_id`, `match_type`, `prev_gw_ids`, `curr_gw_ids` (`;`-joined, from `*_gw_id` and `*_gw_id_2` when set), `curr_start_ft`, then `prev_` and `curr_` counts of `ext`, `int`, `other` and `total`, then `change_total` = curr_total − prev_total.
  - Counts come from each run's metal loss by `us_gw_id`; no anomaly matching is used.
  - An inserted or removed row has the missing side's counts blank, not 0.
- Report: two tables of `report.largest_changes_n` rows, largest `curr_total` and largest `change_total` (ties by `curr_start_ft`).

- [ ] **Step 1: Write the failing tests**
  - `test_joint_counts_one_to_one`.
  - `test_joint_counts_two_to_one_sums_both_prev_joints`.
  - `test_inserted_joint_prev_blank`.
  - `test_report_writes_joint_counts`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): per-joint metal-loss counts for both runs`.

### Task 12: Anomalies of interest (`anomalies_of_interest.csv`)

**Files:**
- Modify: `src/ili/report_tables.py`, `src/ili/report.py`, `src/ili/templates/report.html.j2`
- Test: `tests/test_report_tables.py`

**Interfaces:**
- Produces: `anomalies_of_interest(inputs) -> pd.DataFrame`.
  - Grouping: current-run metal loss sorted by `log_dist_ft`. A new group starts where the gap to the previous feature exceeds `aoi_group_ft`.
  - Each feature's RL comes from `remaining_life` through `curr_feature_id`, with cluster ids split on `;`. HCA comes from `could_affect_hca` there.
  - Columns: `rank`, `start_ft`, `end_ft`, `surfaces`, `origins` (sorted, comma-joined), `max_depth_pct`, `n_anomalies`, `min_rl_yr`, `hca` (any member True), `description`.
  - Sort by `min_rl_yr` ascending (NaN last), then `max_depth_pct` descending, then `start_ft`. Keep `aoi_count` rows.
  - `description` follows this pattern: `"Deepest {d:g}% WT, {L:g} in long, {W:g} in wide at {h}:{mm}; others {lo:g}-{hi:g}% WT; overlaps recoat repair of 2023-12-15"`. Drop the others clause for a single feature and the repair clause when nothing overlaps or no repairs file was given.
- Report: the executive summary gets an "Anomalies of interest" table after the reinspection intervals.

- [ ] **Step 1: Write the failing tests**
  - `test_aoi_grouping_gap`: features at 100.0, 101.5 and 104.0 ft with `aoi_group_ft` 2 make two groups.
  - `test_aoi_ranking`: shortest RL first; on a tie, the deepest first.
  - `test_aoi_description_text`: the exact string for a hand-built group with a recoat overlap.
  - `test_aoi_no_repairs_file`.
  - `test_report_writes_aoi`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.** Clock text is `h:mm` on the 12-hour face, with 0 shown as `12:00`.
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): anomalies of interest`.

### Task 13: Pipe summary and length check

**Files:**
- Modify: `src/ili/report_tables.py`, `src/ili/report.py`, `src/ili/templates/report.html.j2`
- Test: `tests/test_report_tables.py`

**Interfaces:**
- Produces: `pipe_summary(pipe, mop, segments, settings) -> PipeSummary`, with these fields:
  - `by_wall_grade`: a table of `wt_nom_in`, `grade` and `length_ft`.
  - `design_pressure_psig`: min and max of 2·SMYS·t·`design_factor`/D over the pipe rows.
  - `stress_at_mop_pct_smys`: min and max of MOP·D/(2·t·SMYS)·100, over the intervals formed by the union of the pipe and MOP breakpoints.
  - `install_years` and `coatings`: sorted unique values.
  - `hca_length_ft` and `hca_pct`: from segments with `could_affect_hca`.
- Produces: `length_check(prev_std, curr_std, pipe) -> pd.DataFrame` with columns `run_id`, `ili_length_ft` (max − min `log_dist_ft` of girth welds), `reference_length_ft` (max `end_ft` − min `start_ft`), `diff_ft` and `diff_pct`.
- Report: both go in the header section.

- [ ] **Step 1: Write the failing tests**
  - `test_pipe_summary_values`: a two-row pipe table (0.250 in and 0.312 in X52, 12.75 in OD) with MOP 1,440 and then 1,200 gives exact design pressures and stress range.
  - `test_length_check`.
  - `test_report_header_has_pipe_summary`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): pipe summary and length check`.

### Task 14: Orientation and elevation charts

**Files:**
- Modify: `src/ili/report_charts.py`, `src/ili/report.py`, `src/ili/templates/report.html.j2`
- Test: `tests/test_report_charts.py`

**Interfaces:**
- Produces: `orientation_charts(inputs) -> dict[str, go.Figure]` keyed `ext_corrosion`, `int_corrosion` and `manufacturing`, using the same categories as Task 10. Each chart plots clock (hours, 0–12, y axis reversed so 12 is at the top) against log distance, one trace per run, with marker size scaled by `depth_pct`.
- Produces: `elevation_chart(inputs) -> go.Figure | None`. It draws the current run's girth-weld `elev_ft` as a line against log distance, with INT metal loss as markers at its `elev_ft`. It returns None when no girth weld has `elev_ft`, and the report then says so.

- [ ] **Step 1: Write the failing tests**
  - `test_orientation_charts_one_per_category`: three figures, each with two traces.
  - `test_clock_hours`: clock_min 360 plots at 6.0.
  - `test_elevation_none_without_elev`.
  - `test_report_html_has_orientation_and_elevation`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement**, following `_figure` and the existing palette.
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): orientation and elevation charts`.

### Task 15: Methods appendix and output list

**Files:**
- Modify: `src/ili/templates/report.html.j2`, `src/ili/report.py` (`OUTPUT_NAMES`)
- Test: `tests/test_report_html.py`, `tests/test_report_cli.py`

**Interfaces:**
- The formulas section gains `half_life`, `joint_percentile`, `shrinking_fixed`, both governing profiles, ERF, FPR, failure mode and P_design / stress at MOP, worded as spec 6.5–6.7 and 6.10.
- `OUTPUT_NAMES` gains `anomaly_distribution.csv`, `joint_ml_counts.csv` and `anomalies_of_interest.csv`, all written through `_write_all`. The run manifest lists them.

- [ ] **Step 1: Write the failing tests**
  - `test_methods_lists_new_formulas`.
  - `test_report_cli_writes_all_outputs`: all seven outputs exist, and the manifest's `outputs` names each one.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `tests/test_report_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): methods appendix and outputs for M11`.

### Task 16: End to end, both profiles; DECISIONS and README

**Files:**
- Modify: `tests/test_pipeline.py`, `docs/DECISIONS.md`, `README.md`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the end-to-end tests** (the `synth` fixture, into `tmp_path`):
  - `test_pipeline_conservative_end_to_end`: every stage exits 0, and the report HTML contains each new section heading.
  - `test_pipeline_pit_to_pit_reported`: a settings copy with `governing_profile: pit_to_pit`, `depth_basis: reported`, `depth_tolerance_pct_wt: 0` and `shrinking_rate_mpy: 6` runs end to end. Its `growth_summary.json` records `pit_to_pit`, and its median `rl_yr` is at least the conservative run's, the expected direction from spec 6.5.
- [ ] **Step 2:** Run `tests/test_pipeline.py -q`. Expected: PASS, since the earlier tasks built everything. A failure here is a bug in an earlier task: fix it there.
- [ ] **Step 3:** Run the full suite with `.venv/Scripts/python.exe -m pytest tests -q`. Expected: all pass. Record the count.
- [ ] **Step 4: Docs.**
  - `docs/DECISIONS.md` gets "M11: consultant-practice methods and outputs", recording decisions 1–10 above and any made during execution.
  - `README.md` gets the new outputs, the `--repairs` flags and how to run the `pit_to_pit` profile.
- [ ] **Step 5: Commit** `docs: M11 decisions and README`.
