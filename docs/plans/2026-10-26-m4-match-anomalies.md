# ILI Analyzer — Milestone 4 (Anomaly Matching) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match every anomaly (metal loss, dent, gouge) in the current run to its counterpart in the previous run (spec §6.4). Write `match_table.csv` with `matched_1to1 / matched_cluster / new / missing`, a score, a confidence, and the review status. Reach ≥ 98% precision and ≥ 95% recall against `ground_truth.csv` on synthetic data, with the metrics in `outputs/validation_metrics.json`.

**Architecture:** `src/ili/match.py` takes the two standardized runs plus M3's weld map and works in four stages:
1. Place each anomaly in its weld-map row and map positions between runs. A `PositionMap` is piecewise linear through the ends of every matched row, which is exactly the per-joint length-ratio scaling of §6.4 step 1.
2. Form candidates (axial, clock with wraparound, surface, type) and score them. Assign one-to-one with `linear_sum_assignment` per connected component of the candidate graph.
3. Group the leftovers by the interaction rule and match group envelopes (many-to-one).
4. Classify every anomaly, add missing-reason hints and confidence, and assemble the table.

`validation.score_matching` scores the result against ground truth.

**Tech Stack:** pandas 3.x, numpy, scipy (`scipy.optimize.linear_sum_assignment`, `scipy.sparse.csgraph.connected_components`), pydantic config.

**Spec:** `docs/SPEC.md` §6.4, §7.1 (`tool`, `matching`), §8, §9 M4. Inputs: M2 standardized CSVs (`ingest.load_standardized`), M3 `weld_map.csv`, M1 `ground_truth.csv`, `reference/pipe_properties.csv`, `reference/repairs_between_runs.csv`.

## Global Constraints

- All M1–M3 constraints hold. In particular, every tuning value lives in `config/settings.yaml` with a comment, there are no magic numbers in `match.py`, and every output carries `DRAFT – for review by qualified personnel.`
- Units: inches for axial differences and sizes, feet for positions, clock minutes 0–719 with wraparound at 12:00, % of nominal wall for depth, mpy for growth, decimal years (days / 365.25) for Δt.
- Outputs go to `outputs/<prev_run_id>_vs_<curr_run_id>/`: `match_table.csv`, `match_summary.json`. Metrics merge into `outputs/validation_metrics.json` under `anomaly_matching` via `validation.write_metrics`.
- Anomaly types are `metal_loss`, `dent`, `gouge`. Every other feature type is ignored by M4. Features are placed by `log_dist_ft`, never by `us_gw_id` (M2 note: a feature may cite an excluded weld).
- Feature ids stay strings end to end.

## Review Focus

1. **A feature near a girth weld is reported in different joints by the two runs** (1 in axial noise across a weld). It must still match. Cross-row candidates are allowed between adjacent weld-map rows when both features are within `near_weld_ft` of the shared weld. Test in Task 2.
2. **A feature in an inserted/removed joint, or before the first / after the last weld.** It becomes `new`/`missing` with no crash, and a removed joint inside a repair gets `in_repaired_section`. Test in Tasks 1 and 4.
3. **A feature in a 2:1 or 1:2 joint (missed weld).** Its position scales over the merged span, so it matches as if no weld were missed. Test in Task 1. The synthetic acceptance in Task 5 also covers it: M1 puts missed welds upstream of joints that hold metal loss.
4. **A vendor cluster report vs individually reported pits, including one surviving pit.** The 1:1 pass must not pair the cluster row with one pit. A cluster report matched to a single pit is still `matched_cluster`. Test in Task 3.
5. **Equal or reversed run dates, or a missing run date.** Δt ≤ 0 makes every growth plausibility check meaningless, so it is a `ValueError`, not a division by zero. Test in Task 1.

---

### Task 1: Placement, position map, clock and Δt helpers

**Files:** Create `src/ili/match.py`; Modify `src/ili/align_welds.py` (add `load_weld_map`); Test `tests/test_match_place.py`

**Interfaces:**
- Consumes: M3 `weld_map` columns (`match_type, prev_gw_id, prev_gw_id_2, curr_gw_id, curr_gw_id_2, prev_start_ft, curr_start_ft, prev_length_ft, curr_length_ft, in_repair`); `units.CLOCK_MINUTES`.
- Produces:
  - `align_welds.load_weld_map(path) -> pd.DataFrame`, with the four gw-id columns as str/None and `in_repair`/`is_anchor` as bool.
  - `ANOMALY_TYPES = ("metal_loss", "dent", "gouge")`
  - `clock_diff_min(a, b) -> np.ndarray`: the circular difference, 0–360, NaN where either side is missing.
  - `delta_t_years(prev_std, curr_std) -> float`: (curr run_date − prev run_date).days / 365.25. Raises `ValueError` if either date is missing, a run has more than one date, or Δt ≤ 0.
  - `PositionMap.from_weld_map(wm)` with `to_prev(x_curr)` and `to_curr(x_prev)`. Knots are (prev_start, curr_start) and (prev_end, curr_end) of every row whose match type is 1:1/2:1/1:2. `np.interp` applies between knots. Outside them, extrapolate linearly with the slope between the first and last knots.
  - `place(std, wm, run, pmap) -> pd.DataFrame`, with one row per anomaly of `run` ("prev"/"curr"):
    - all standard columns
    - `wm_row`: index of the weld-map row whose own-run span `[start, start + length)` holds `log_dist_ft`, or `-1`
    - `in_matched_row`: that row is 1:1/2:1/1:2
    - `pos_prev_ft`, `pos_curr_ft`: own position, and the other run's through `pmap`
    - `edge_dist_ft`: distance to the nearer end of the row's own-run span (NaN if `wm_row == -1`)

- [ ] Write the failing tests:
  - `test_clock_diff_wraps`: (715, 5) → 10; (0, 360) → 360; (100, NaN) → NaN.
  - `test_delta_t_years`: 2021-06-15 → 2026-06-15 gives 1826/365.25. Equal dates raise `ValueError`; so do reversed dates and a `None` date.
  - `test_position_map_1to1_scales_by_joint_ratio`: weld map with one 1:1 row, prev [100, 140), curr [200, 240.4). A curr feature at 220.2 maps to prev 120.0 within 1e-9.
  - `test_position_map_2to1_spans_merged_joint` (Review Focus #3): a 2:1 row, prev [0, 80) vs curr [0, 79.2). Curr 39.6 → prev 40.0.
  - `test_place_assigns_rows_and_flags_unmatched_rows` (Review Focus #2): a feature in an `inserted` row has `in_matched_row False`. A feature before the first weld has `wm_row == -1`. A non-anomaly (valve) is absent.
  - `test_load_weld_map_keeps_string_ids`: write an M3 weld map for vendor B and load it back. `curr_gw_id` is `"1"`, not `"1.0"`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(match): placement, position map, clock and dt helpers`.

### Task 2: Candidates, scoring, one-to-one assignment

**Files:** Modify `src/ili/match.py`; Test `tests/test_match_pairs.py`

**Interfaces:**
- Consumes: Task 1 `place` frames, `clock_diff_min`, `delta_t_years`; `settings.tool`, `settings.matching`.
- Produces:
  - `depth_change_tol_pct(tool) -> float` = √2 · `depth_tolerance_pct_wt`: the tolerance of a two-run difference at the tool's `depth_certainty`.
  - `pair_metrics(prev, curr, dt_yr, settings) -> pd.DataFrame`, built from aligned arrays of equal length. Columns:
    - `axial_diff_in` = |prev.pos_prev_ft − curr.pos_prev_ft| × 12
    - `clock_diff_min`
    - `depth_change_pct` = curr − prev depth_pct
    - `implausible`: a drop below −tol, or (Δd − tol)/100 · curr `wt_nom_in` · 1000 / dt_yr > `max_plausible_growth_mpy`; False when a depth is missing
    - `score`
  - `candidates(prev, curr, dt_yr, settings) -> pd.DataFrame` with `prev_i, curr_i` (positional indices into the frames) plus the `pair_metrics` columns, for pairs where:
    - both are in matched rows
    - `axial_diff_in ≤ axial_tol_in`
    - `clock_diff_min ≤ clock_tol_min`, or either clock is missing
    - the types are equal
    - the surfaces are equal or either is `UNK`
    - neither is a vendor cluster report (`vendor_cluster_id` set; these are left to Task 3)
    - the rows are the same, or adjacent with both features within `near_weld_ft` of the shared weld
  - `assign_one_to_one(cands) -> pd.DataFrame` (the chosen rows of `cands`): `linear_sum_assignment` on each connected component of the candidate graph. Non-candidate cells cost `NONCANDIDATE_COST = 1e9`; pairs assigned at that cost are dropped.
- Score:

  ```
  a = axial_diff_in / axial_tol_in
  c = clock_diff_min / clock_tol_min
  score = sqrt((w_axial·a² + w_circ·c²) / (w_axial + w_circ)) + w_depth_change · implausible
  ```

  When a clock is missing, drop the circ term and its weight.
- Search candidates by sorting on `pos_prev_ft` and using `np.searchsorted` windows of ± `axial_tol_in`. Never build a dense n × m matrix over the whole line.

- [ ] Write the failing tests (hand-built placed frames):
  - `test_score_zero_for_identical_position`.
  - `test_clock_tolerance_wraps_at_noon`: clocks 710 and 20 are candidates (diff 30 ≤ 45).
  - `test_surface_and_type_compatibility`: EXT–INT is rejected, EXT–UNK accepted, metal_loss–dent rejected.
  - `test_implausible_drop_and_growth`: with Δt 5 yr and wt 0.25 in:
    - a drop of 15% (beyond 14.14) is flagged; a drop of 14% is not
    - Δd 80% gives (80 − 14.14) · 2.5 mils ÷ 5 = 32.9 mpy > 30, flagged; Δd 70% gives 27.9, not flagged
  - `test_assignment_is_one_to_one_and_prefers_lower_total_cost`: prev A, B and curr X, Y where A–X, A–Y, B–Y are candidates. The result is {A–X, B–Y}.
  - `test_cross_weld_candidate_only_near_weld` (Review Focus #1): adjacent rows, features 0.5 in apart straddling the weld are candidates; with `near_weld_ft` set to 0.01 they are not.
  - `test_vendor_cluster_rows_are_not_candidates`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(match): candidates, scoring, one-to-one assignment`.

### Task 3: Interaction groups and cluster matching

**Files:** Modify `src/ili/match.py`; Test `tests/test_match_clusters.py`

**Interfaces:**
- Consumes: Task 1/2 frames and functions; `settings.matching.interaction_rule`; OD per feature (Task 4 passes `od_in` as a column on the placed frames).
- Produces:
  - `interaction_groups(f) -> np.ndarray[int]`: group label per row, the connected components of "interacts". Two features interact when they are the same surface and:
    - axial edge gap = |Δpos| × 12 − (L₁ + L₂)/2 ≤ `axial_spacing_in`
    - circ edge gap = arc(Δclock) − (W₁ + W₂)/2 ≤ `circ_spacing_in`, where arc(m) = m / 720 · π · OD
  - `envelope(f) -> dict` with keys `pos_prev_ft` (centre), `pos_curr_ft`, `length_in`, `width_in`, `clock_min`, `depth_pct` (max), `surface`, `wt_nom_in`. This is the bounding box in the prev frame; circ is unwrapped relative to the first member, as in M1's vendor envelope.
  - `match_clusters(prev, curr, dt_yr, settings) -> list[tuple[list[int], list[int], pd.Series]]`, giving (prev rows, curr rows, metrics):
    - Inputs are the features left unmatched by Task 2 that are in matched rows.
    - Groups are formed per run. A group is *multi* when it has ≥ 2 members or contains a vendor cluster report.
    - Group pairs are candidates when at least one side is multi, the surfaces agree (or one is UNK), and the box gaps fit: axial gap ≤ `axial_tol_in` and circ gap in minutes ≤ `clock_tol_min` (gap = 0 when the boxes overlap).
    - Score with Task 2's formula, using the box gaps for a and c and the envelopes' max depths for implausibility. Assign with `assign_one_to_one`.

- [ ] Write the failing tests:
  - `test_pits_within_spacing_group`: three pits 0.6 in apart (centres), 0.3 in long, clocks ±8 min, OD 12.75. One group. Moving one pit 3 in away gives two groups.
  - `test_envelope_box_and_max_depth`: the length spans the outer edges; the depth is the max; clocks 715 and 5 give a box across 12:00 with centre ≈ 0.
  - `test_vendor_cluster_matches_pit_group` (Review Focus #4): a prev vendor cluster row vs 3 curr pits inside its box gives one match, 1 prev × 3 curr.
  - `test_vendor_cluster_matches_single_surviving_pit`: the same with 1 pit gives one match.
  - `test_single_vs_single_is_not_a_cluster`: two lone unmatched features 3 in apart give no match.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(match): interaction groups and cluster matching`.

### Task 4: Whole-run matching and the match table

**Files:** Modify `src/ili/match.py`; Test `tests/test_match_table.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `MatchResult(match_table: DataFrame, summary: dict)`.
  - `match_anomalies(prev_std, curr_std, weld_map, settings, pipe, repairs=None) -> MatchResult`. `pipe` is `pipe_properties.csv`. `od_in` is looked up at `pos_curr_ft`; a position outside the table is a `ValueError` naming it. `repairs` must have `start_ft, end_ft` if given, else `ValueError`.
- `match_table`: one row per match (or per unmatched feature), in prev-frame position order. Columns:
  - `match_id` (`M00001…`), `match_type`, `score`, `confidence`, `review_status`, `needs_review`, `missing_hint`, `n_prev`, `n_curr`
  - `prev_feature_id`, `curr_feature_id`: `;`-joined for clusters, in log order
  - per side `s ∈ {prev, curr}`: `{s}_feature_type, {s}_log_dist_ft, {s}_wm_row, {s}_us_gw_id, {s}_dist_from_us_gw_ft, {s}_wt_nom_in, {s}_depth_pct, {s}_dent_depth_in, {s}_length_in, {s}_width_in, {s}_clock_min, {s}_surface, {s}_seam_interaction, {s}_dent_metal_loss, {s}_vendor_cluster_id`. A cluster side shows its envelope: centre position, box length/width, max depth, envelope clock.
  - `curr_pos_in_prev_ft`, `axial_diff_in`, `clock_diff_min`, `depth_change_pct`, `implausible`
- Rules:
  - `review_status` is always `pending`.
  - `confidence` for matches: `high` if score ≤ `confidence_bands.high`, `medium` if ≤ `medium`, else `low`. It is None for new/missing.
  - `missing_hint`, for missing only, in order:
    - `in_repaired_section` when the feature's row is `removed` with `in_repair`, or `pos_curr_ft` is inside a repairs span
    - else `near_threshold` when `depth_pct < reporting_threshold_pct_wt + depth_tolerance_pct_wt`
    - else `unexplained`
  - `needs_review` = low confidence or `unexplained`.
- `summary`: run ids, Δt, anomaly counts per run, counts by match type, by confidence, by hint, `n_needs_review`.

- [ ] Write the failing tests (5-mile synthetic through `align_welds`):
  - `test_every_anomaly_in_exactly_one_row`: the split feature ids of each side cover that run's anomaly ids exactly once; no valves or welds appear.
  - `test_replaced_section_features_missing_in_repair` (Review Focus #2): the 3 truth features in the replaced joints are `missing` with `in_repaired_section`.
  - `test_columns_and_draft_fields`: the column list above; `review_status` is all `pending`; confidence is None exactly on new/missing.
  - `test_reference_errors`: pipe table not covering a feature, or repairs without `start_ft`, raise `ValueError`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(match): run matching and match table`.

### Task 5: Scoring against ground truth and acceptance

**Files:** Modify `src/ili/validation.py`; Test `tests/test_match_score.py`

**Interfaces:**
- Consumes: the Task 4 `match_table`; `ground_truth.csv` (feature ids read as str).
- Produces: `score_matching(match_table, truth) -> dict`.
  - Truth links are the (prev_feature_id, curr_feature_id) pairs of truth rows with `truth_class` in {`matched_1to1`, `matched_cluster`}, both ids set, and an anomaly type. They are deduplicated.
  - Predicted links are, for each matched row, every prev id × every curr id.
  - `precision` = |pred ∩ truth| / |pred|; `recall` = |pred ∩ truth| / |truth|.
  - `confusion`: `{run: {truth_class: {pred_class: n}}}` per reported anomaly feature id. A feature's class is its row's `match_type`; its truth class comes from any truth row holding that id. Conflicting truth classes for one id are a `ValueError`.
  - Also `n_pred_links`, `n_true_links`, `n_correct_links`, `false_links` and `missed_links` (each ≤ 20 listed), and `passes` (precision ≥ 0.98 and recall ≥ 0.95, the spec's M4 targets).
  - A table feature id absent from the truth is a `ValueError` (truth from another dataset).

- [ ] Write the failing tests:
  - `test_default_dataset_meets_acceptance`: 20-mile default seed through files (synth → ingest → align → match → score). precision ≥ 0.98, recall ≥ 0.95, and every truth `matched_cluster` report predicted `matched_cluster`.
  - `test_acceptance_across_seeds`: seeds 1–5 at 5 miles, same thresholds.
  - `test_score_detects_wrong_link`: swap the curr ids of two matched rows; precision and recall both drop.
  - `test_score_unknown_id_is_value_error`.
- [ ] Run them and see FAIL. Implement and tune only through `settings.yaml` (any changed value recorded in DECISIONS with its reason). Run and see PASS. Commit `feat(match): ground-truth scoring and acceptance`.

### Task 6: CLI, outputs, docs

**Files:** Modify `src/ili/match.py`, `README.md`, `docs/DECISIONS.md`; Test `tests/test_match_cli.py`

**Interfaces:**
- Produces:
  - `write_matches(result, out_dir) -> dict[str, Path]` writes `match_table.csv` and `match_summary.json` (summary with the DRAFT label) to `out_dir/<prev>_vs_<curr>/`.
  - `main(argv)`:
    - usage: `python -m ili.match --prev STD.csv --curr STD.csv --weld-map weld_map.csv --pipe pipe_properties.csv [--repairs CSV] [--truth ground_truth.csv] [--settings YAML] [--out DIR]`
    - prints counts by type and confidence, `n_needs_review`, and with `--truth` the precision/recall (merging `anomaly_matching` into `<out>/validation_metrics.json`)
    - exits 2 with `error:` on `ValueError`/`OSError`

- [ ] Write the failing tests:
  - `test_cli_writes_outputs_and_metrics` (5-mile fixture through ingest and align CLIs)
  - `test_cli_missing_weld_map_exit_2`
- [ ] Run them and see FAIL. Implement. Run and see PASS.
- [ ] Run the CLI on the default dataset and paste the summary. Add the README command and record every interpretation in DECISIONS:
  - the global position map
  - per-component assignment
  - excluding vendor cluster rows from 1:1
  - the score formula
  - the implausibility tolerance
  - the hint order
  - `needs_review`
- [ ] Commit `feat(match): CLI and outputs`.

### Task 7: Close-out

Run the full suite and the fresh-reviewer final review (focus on the Review Focus items, scorer leniency, dense feature fields, and clock/positions near 12:00 and line ends). Apply one fix pass, update memory, and **STOP for user review before M5**.
