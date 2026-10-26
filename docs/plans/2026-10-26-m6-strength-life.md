# ILI Analyzer — Milestone 6 (Remaining Strength and Remaining Life) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Original and Modified B31G as pure, unit-tested functions (spec §6.6), passing the reference case and the z = 20 / z = 50 boundaries. Then project each M5 growth feature forward to find τ_80, τ_burst and τ_safe, the remaining life and its governing criterion, the repair-by date, the dig flag, and per-segment reinspection intervals (spec §6.7). Write `remaining_life.csv`, `reinspection_intervals.csv` and `remaining_life_summary.json`, with an RL and governing criterion for every feature.

**Architecture:**
- `src/ili/strength.py` holds pure scalar-or-array functions: `z_param`, the Folias factors, the failure stress per method, burst and safe pressure, and assessed depth and length. It also has a method registry with an RSTRENG stub interface.
- `src/ili/remaining_life.py` does three things:
  1. Projects depth and length in time.
  2. Finds each criterion's first crossing on a fixed time grid, then refines it with `scipy.optimize.brentq`.
  3. Assembles per-feature rows from `growth.csv` plus the reference tables (pipe properties, MOP, segments, next inspection), then the segment intervals.

**Tech Stack:** numpy, scipy (`brentq`), pandas 3.x, pydantic config.

**Spec:** `docs/SPEC.md` §6.6, §6.7, §7.1 (`tool`, `strength`, `remaining_life`), §8 (B31G unit tests incl. the reference case and the z = 20 / z = 50 boundaries), §9 M6, §10.5/§10.8 (factors and assessment dates, which stay PLACEHOLDER). Inputs: M5 `growth.csv`; reference `pipe_properties.csv` (`od_in`, `smys_psi`), `mop_profile.csv`, `hca_segments.csv` (`could_affect_hca`) and `next_inspection.csv`; the M2 current standardized CSV (run date).

## Global Constraints

- All M1–M5 constraints hold. Every engineering value is in `config/settings.yaml` with a comment. There are no magic numbers in `strength.py` or `remaining_life.py`, except the spec's B31G formula constants:
  - Original: 1.1, 2/3, 0.8, z-limit 20
  - Modified: 10,000 psi, 0.85, 0.6275, 0.003375, 0.032, 3.3, z-limit 50
  - the 80 %-of-wall criterion of §6.6/§6.7

  These are named module constants, each with a comment citing the spec section. Every output carries `DRAFT – for review by qualified personnel.`
- Units:
  - D, t, d, L in inches; SMYS, flow and failure stress in psi; pressures in psig
  - growth g = `governing_mpy / 1000` in/yr; g_L = `length_rate_ipy` in/yr; time in decimal years
  - dates: run date + years × 365.25 days
- Features are the rows of M5 `growth.csv`, one per current-run metal-loss feature. Its `needs_review` and `review_reasons` carry forward, and M6 appends its own reasons.
- Reference lookups use `reference.lookup` at `curr_log_dist_ft` with `alignment.anchor_tol_ft`. Segments use `reference.assign_segment` on the hca_segments table.
- Every input and intermediate value is a column (spec §2 traceability).

## Review Focus

1. **d/t at or beyond the formula range during projection.** The projected depth can pass 0.80·t and reach t within the horizon. Pressures are computed with d/t clipped to [0, 1], so they stay finite and non-negative. τ_80 normally governs before that. Test in Tasks 1 and 2.
2. **A criterion already met at τ = 0** gives time 0, including assessed d/t ≥ 0.80 and P_safe < MOP today. **No criterion met within `horizon_yr`** gives `rl_yr` = `horizon_yr` with `rl_governing` = `beyond_horizon`. Never NaN or inf without a flag. Test in Task 2.
3. **Missing or invalid inputs:**
   - NaN governing rate (flagged in M5), or negative growth (the projection uses max(g, 0), since depth cannot shrink, and the row is already flagged `below_min_rate`)
   - NaN length, MOP or pipe values

   NaN times and RL come with a `needs_review` reason. Never a crash or a silently optimistic RL. Test in Tasks 2 and 3.
4. **The discontinuity at Original B31G z = 20.** Failure stress jumps there, so the crossing search must still find the first time P drops below MOP, not miss it or land on the wrong side of the jump. Test in Task 2.
5. **Next-assessment dates:**
   - a segment missing from `next_inspection.csv` uses the config default: HCA 5 yr capped at 68 months; non-HCA 10 yr
   - a planned HCA date later than run date + 68 months is capped
   - a feature outside every segment uses the HCA default (the conservative choice) and is flagged

   Test in Task 3.

---

### Task 1: B31G remaining strength

**Files:** Modify `config/settings.yaml`, `src/ili/config.py`; Create `src/ili/strength.py`; Tests `tests/test_strength.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - settings `strength.length_basis: reported` (`reported | reported_plus_tolerance`; spec §6.6 "length can optionally be adjusted the same way"; PLACEHOLDER) on `StrengthCfg`.
  - `assessed_depth_in(depth_pct, wt_in, settings)`: reported, or reported plus `tool.depth_tolerance_pct_wt` (% wall), capped at 100 % of wall, then converted to inches.
  - `assessed_length_in(length_in, settings)`: reported, or plus `tool.length_tolerance_in`.
  - `z_param(L, D, t)` = L² / (D·t).
  - `folias_original(z)` = √(1 + 0.8z) for z ≤ 20. For z > 20 it is not used: NaN, and the z > 20 branch takes over.
  - `folias_modified(z)` = √(1 + 0.6275z − 0.003375z²) for z ≤ 50, else 0.032z + 3.3.
  - `failure_stress(method, D, t, d, L, smys) -> (z, M, flow, S_F)` dispatches through `STRENGTH_METHODS = {"original_b31g": …, "modified_b31g": …}`. d/t is clipped to [0, 1] inside, and NaN in gives NaN out.
  - `burst_pressure(S_F, t, D)` = 2·S_F·t/D, and `safe_pressure(P_F, settings)` = P_F × `design_factor`.
  - `rstreng_effective_area(profile, D, t, smys)`: raises `NotImplementedError("RSTRENG is out of scope for v1 (spec §6.6)")`. Its docstring describes the profile input it will take. It is not in `STRENGTH_METHODS` and not selectable in config.
- [ ] Write the failing tests:
  - `test_reference_case_modified`: D 12.75, t 0.250, SMYS 52 000, d 0.125, L 4.0. z 5.0196 (±1e-3), M 2.0161 (±1e-3), S_F 45 173 (±5), P_F 1 771.5 (±1).
  - `test_reference_case_original`: M 2.2396 (±1e-3), S_F 44 803 (±5), P_F 1 757.2 (±1).
  - `test_z20_boundary_original`: at z = 20, `S_flow·(1−(2/3)(d/t))/(1−(2/3)(d/t)/√17)`. At z = 20 + 1e-9 it gives S_flow·(1 − d/t). Assert both values. (The jump is the spec's formula; record it, don't smooth it.)
  - `test_z50_boundary_modified`: M from both branches at z = 50 is √(1+31.375−8.4375) = 4.8926 vs 0.032·50+3.3 = 4.9. Assert each side and that M stays monotone across z ∈ [0, 100].
  - `test_zero_depth_gives_flow_stress`: both methods.
  - `test_very_short_length`: L → 0 gives M → 1 and S_F → S_flow for Modified.
  - `test_dt_clipped`: d = 1.2t gives a finite pressure equal to the d = t value.
  - `test_assessed_depth_and_length_basis`: 50 % + 10 = 60 % → 0.150 in; 95 % + 10 is capped at 100 %; length `reported` vs `reported_plus_tolerance` (+0.75).
  - `test_safe_pressure`: × 0.72.
  - `test_rstreng_stub_raises`.
  - `test_strength_length_basis_loads`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(strength): Original and Modified B31G`.

### Task 2: Time-to-criterion solver

**Files:** Modify `config/settings.yaml`, `src/ili/config.py`; Create `src/ili/remaining_life.py`; Test `tests/test_rl_solver.py`

**Interfaces:**
- Produces:
  - settings `remaining_life.time_step_yr: 0.05` (fallback grid step; a numerical setting, commented) on `RemainingLifeCfg`.
  - `Criterion` names `CRITERIA = ("tau_80", "tau_burst", "tau_safe")`.
  - `criterion_margins(tau, d0, L0, g, gL, D, t, smys, mop, settings) -> dict[str, np.ndarray]`. Each margin is > 0 while not met and ≤ 0 once met:
    - `tau_80`: 0.80·t − depth(τ)
    - `tau_burst`: P_F(τ) − MOP
    - `tau_safe`: P_safe(τ) − MOP
    - depth(τ) = d0 + max(g, 0)·τ and L(τ) = L0 + max(gL, 0)·τ; pressures via Task 1 with the configured method
  - `time_to_criteria(d0, L0, g, gL, D, t, smys, mop, settings) -> dict[str, float]` for one feature:
    - Met at τ = 0 gives 0.
    - Otherwise, evaluate on the grid `0, step, …, horizon_yr` (horizon included).
    - The first grid interval whose end margin is ≤ 0 is refined with `brentq` on the margin, `xtol` = 1e-6 yr.
    - Not met by the horizon gives `inf`.
    - Any NaN input in the margin gives NaN.
  - `remaining_life(times) -> (rl_yr, rl_governing)`:
    - the min over the finite times; ties go to the `CRITERIA` order
    - all `inf` gives (`horizon_yr`, `"beyond_horizon"`)
    - any NaN among the criteria gives (NaN, None)

    The last case is conservative: an RL is never claimed when one criterion can't be evaluated.
- [ ] Write the failing tests (all hand-checkable):
  - `test_tau80_linear`: t 0.25, d0 0.10, g 0.005 in/yr gives (0.2 − 0.1)/0.005 = 20.0 yr (±1e-4).
  - `test_already_met_is_zero`: d0 = 0.85t gives `tau_80` 0. And MOP above P_safe at τ = 0 gives `tau_safe` 0.
  - `test_beyond_horizon`: g 0 with a healthy feature gives all `inf`, RL `horizon_yr`, `beyond_horizon`.
  - `test_burst_crossing_matches_direct_solve`: for a Modified B31G feature, the solver's `tau_burst` is checked against an independent bisection on the closed form (±1e-4 yr), and `tau_safe` < `tau_burst`.
  - `test_original_b31g_jump_found` (Review Focus #4): a feature whose length grows through z = 20, where the jump in S_F drops P below MOP. `tau_burst` equals the time z reaches 20 (±1e-3 yr), not later.
  - `test_negative_growth_treated_as_zero` (Review Focus #3).
  - `test_nan_input_gives_nan_rl`: RL NaN, governing None.
  - `test_depth_through_wall_stays_finite` (Review Focus #1): a fast feature projected past d = t; pressures finite, RL = `tau_80`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(remaining-life): time-to-criterion solver`.

### Task 3: Remaining-life table, dig flag, reinspection intervals

**Files:** Modify `src/ili/remaining_life.py`; Test `tests/test_rl_table.py`

**Interfaces:**
- Consumes: Tasks 1–2; M5 growth features (`growth.load_growth_table` is added here, mirroring `match.load_match_table`).
- Produces:
  - `RLResult(features, intervals, summary)`.
  - `compute_remaining_life(growth, curr_run_date, pipe, mop, segments, next_inspection, settings, *, prev_run_id=None, curr_run_id=None) -> RLResult`.
- `features` has one row per growth row, in the same order. Columns:
  - `match_id, curr_feature_id, curr_log_dist_ft, segment_id, could_affect_hca`
  - `od_in, wt_in, smys_psi, mop_psig, d_reported_pct, d_assessed_in, assessed_dt, exceeds_80pct, length_in, length_assessed_in`
  - `method, z, folias_m, flow_stress_psi, failure_stress_psi, burst_psig, safe_psig`
  - `g_inyr, gl_inyr, tau_80_yr, tau_burst_yr, tau_safe_yr, rl_yr, rl_governing, rl_factor, rl_factored_yr, repair_by_date`
  - `next_assessment_date, next_assessment_source, years_to_next_assessment, dig_required_by_remaining_life`
  - `needs_review, review_reasons`
- Rules:
  - Strength columns are at τ = 0. `exceeds_80pct` is `assessed_dt ≥ 0.80`.
  - `g_inyr` = `governing_mpy`/1000. `gl_inyr` = `length_rate_ipy`.
  - `rl_factored_yr` = `rl_yr` × `rl_factor`.
  - `repair_by_date` = curr run date + `rl_factored_yr` × 365.25 days, None when RL is NaN.
  - `next_assessment_source`:
    - `planned`: from `next_inspection.csv`
    - `capped`: an HCA planned date later than run date + `hca_max_interval_months`, cut to that limit
    - `default_hca` or `default_non_hca`: from `default_next_assessment_yr`, with the HCA default also capped at the months limit
    - `default_no_segment`: a feature outside every segment; uses the HCA default and is flagged
  - `years_to_next_assessment` = (date − run date).days / 365.25.
  - `dig_required_by_remaining_life` = `rl_factored_yr` < `years_to_next_assessment`, False when RL is NaN; a NaN RL is flagged instead.
- Review reasons append to M5's `;`-joined list:
  - `rl_not_evaluated`: RL NaN
  - `exceeds_80pct`
  - `no_segment`
  - `needs_review` is True when any reason is present, M5's included.
- `intervals`: one row per segment in the segments table:
  - `segment_id, could_affect_hca, n_features, min_rl_factored_yr, regulatory_max_yr, recommended_interval_yr, governing_match_id`
  - `regulatory_max_yr` is `hca_max_interval_months`/12 for HCA, and `default_next_assessment_yr.non_hca` for non-HCA.
  - `recommended_interval_yr` = min(`min_rl_factored_yr`, `regulatory_max_yr`). A segment with no features gets the regulatory max.
  - Features with NaN RL are excluded from the min, but `n_rl_not_evaluated` is reported so they stay visible.
- `summary`: run ids, method, counts by `rl_governing`, `n_exceeds_80pct`, `n_dig_required`, `n_needs_review`, reasons counts, `min_rl_yr`, and the next-assessment source counts.
- [ ] Write the failing tests:
  - `test_every_feature_has_rl_and_governing` (the M6 acceptance, on the 5-mile synth through align, match and growth): every row has a finite `rl_yr` with a non-None `rl_governing`, or a NaN RL with `rl_not_evaluated` in its reasons.
  - `test_next_assessment_sources` (Review Focus #5), hand-built:
    - SEG-01 planned
    - SEG-03 missing in the HCA file, giving `default_hca` at 5 yr
    - a non-HCA missing segment giving `default_non_hca` at 10 yr
    - an HCA planned date 7 yr out, giving `capped` at 68 months
    - a feature outside segments, giving `default_no_segment` and flagged
  - `test_dig_flag_and_repair_by`: hand-built RL 8 yr × 0.5 = 4 < 5 gives dig True; repair_by = run date + 4 × 365.25 days.
  - `test_intervals_min_and_cap`: hand-built segment minimum and cap, an empty segment at the regulatory max, and the NaN-RL count.
  - `test_reasons_appended_to_m5`: an M5 reason is kept and M6 reasons are appended in order.
  - `test_growth_table_roundtrip`: `load_growth_table` keeps ids as str and flags as bool.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(remaining-life): remaining-life table, dig flag, intervals`.

### Task 4: CLI, outputs, docs

**Files:** Modify `src/ili/remaining_life.py`, `README.md`, `docs/DECISIONS.md`; Test `tests/test_rl_cli.py`

**Interfaces:**
- Produces:
  - `write_remaining_life(result, out_dir) -> dict[str, Path]` writes `remaining_life.csv`, `reinspection_intervals.csv` and `remaining_life_summary.json` (with the DRAFT label) into `align_welds.comparison_dir`. The JSON must be valid: no NaN, `allow_nan=False`.
  - `main(argv)`:
    - usage: `python -m ili.remaining_life --curr STD --growth growth.csv --pipe pipe_properties.csv --mop mop_profile.csv --segments hca_segments.csv --next-inspection next_inspection.csv [--settings YAML] [--out DIR]`
    - run ids come from `growth_summary.json`, or from the file's folder name, or from the curr standardized file plus the growth table's folder; choose the simplest robust source and document it
    - prints counts by governing criterion, exceeds-80 %, dig-required, `n_needs_review`, the minimum RL, and the intervals table
    - exits 2 with `error:` on `ValueError`/`OSError`
- [ ] Write the failing tests:
  - `test_cli_writes_outputs` (the 5-mile fixture through the ingest, align, match and growth CLIs; asserts the files' contents, the DRAFT label, str ids, and one interval row per segment)
  - `test_cli_missing_mop_exit_2`
- [ ] Run them and see FAIL. Implement. Run and see PASS.
- [ ] Run on the default dataset and paste the output. Add the README command (written with the Edit tool; grep for fused paths and run it as written). Record in DECISIONS:
  - the formula constants and the reference case
  - the d/t clipping
  - the z = 20 jump kept as specified
  - assessed depth and length bases
  - the solver (grid plus brentq, step, xtol)
  - beyond-horizon and NaN conventions
  - negative growth projected as 0
  - next-assessment sources and the conservative no-segment default
  - interval caps
  - review reasons
  - the RSTRENG stub
  - results
- [ ] Commit `feat(remaining-life): CLI and outputs`.

### Task 5: Close-out

Run the full suite and the fresh-reviewer final review. Focus areas:
- the Review Focus items
- B31G formulas against the spec, line by line
- unit chains (mpy → in/yr, months → yr, dates)
- solver correctness near the boundaries
- conservatism of every NaN or beyond-horizon convention
- whether any feature can silently get an optimistic RL

Apply one fix pass, update memory, and **STOP for user review before M7**.
