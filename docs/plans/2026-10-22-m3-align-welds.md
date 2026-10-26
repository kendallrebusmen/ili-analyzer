# ILI Analyzer — Milestone 3 (Girth Weld Alignment) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Map every joint in the current run to its joint in the previous run (spec §6.3). Write `weld_map.csv` with `1:1 / 2:1 / 1:2 / inserted / removed` matches and a confidence for each, report the alignment rate, and list every changed section. Score at least 99% joint accuracy against `ground_truth_welds.csv` on synthetic data.

**Architecture:** `src/ili/align_welds.py` works in five stages:
1. Build joints from each run's girth-weld rows.
2. Pair fixed-feature anchors (valve/tee/fitting) between runs and force their joints 1:1. These split the line into sections.
3. Within each section, scale the current-run lengths to the previous run's span to remove odometer drift.
4. Run a banded Needleman–Wunsch DP over the joint-length sequences, with 1:1, 2:1, 1:2, insert and delete moves. Gaps are cheaper inside repaired spans from `repairs_between_runs.csv`.
5. Collect matches into the weld map, changed sections and a summary.

Scoring against ground truth lives in a new `src/ili/validation.py`, which M4/M5 will extend; it writes `outputs/validation_metrics.json`.

**Spec:** `docs/SPEC.md` §6.3, §7.1 (`alignment`), §9 M3. Data contracts: M2 standardized CSVs, M1 `ground_truth_welds.csv`.

## Global Constraints

- All M1/M2 constraints hold. In particular, every tuning value lives in `config/settings.yaml` with a comment; there are no magic numbers in `align_welds.py`.
- Length mismatch tolerance is `max(joint_length_tol_ft, joint_length_tol_pct% × max(a, b))`.
- Outputs go to `outputs/<prev_run_id>_vs_<curr_run_id>/` (§6.10 folder convention): `weld_map.csv`, `changed_sections.csv`, `alignment_summary.json`.
- Read the standardized CSVs through one loader, so id columns stay strings (the M2 review note: vendor B's weld ids otherwise come back as `1.0`).

## Review Focus

1. **An anchor reported in only one run** (a vendor didn't call a tee). The anchor pairing must skip it, not mis-pair every later anchor. Test in Task 2.
2. **Runs that start or end at different welds** (one tally begins one joint later). Leading/trailing joints become inserted/removed and the rest still aligns. Test in Task 3.
3. **A section empty in one run** (two anchors adjacent in one run). This must not crash; everything in the other run's section becomes inserted/removed. Test in Task 3.
4. **Two consecutive missed welds** (a 3:1 situation, which the spec doesn't support). This must not crash: expect a 2:1 plus a gap, flagged low confidence. Test in Task 3.
5. **Features citing a weld excluded by M2 validation.** Alignment uses only the clean weld rows; nothing indexes by `us_gw_id` here. Covered by construction in Task 2: joints come from weld rows only.

---

### Task 1: Config keys and the standardized-CSV loader

**Files:** Modify `config/settings.yaml`, `src/ili/config.py`, `src/ili/ingest.py`; Tests `tests/test_config.py`, `tests/test_ingest_cli.py`

**Produces:**
- New `alignment` keys, each commented, with defaults **(D)**:
  - `missed_weld_penalty: 1.0` (added cost of a 2:1/1:2 match, so 1:1 wins ties)
  - `gap_penalty_repair: 0.5` (gap cost inside a repaired span)
  - `anchor_tol_ft: 100` (max position difference for pairing anchors after drift correction)
  - `band_joints: 30` (DP half-width in joints)
  - `confidence_bands: {high: 1.0, medium: 2.0}` (normalized length-mismatch cutoffs)
- `AlignmentCfg` and its tests updated.
- `ingest.load_standardized(path) -> pd.DataFrame`:
  - `feature_id`, `us_gw_id`, `vendor_cluster_id`, `comments`, `run_id` as str/None
  - `run_date` as date
  - `clock_min` as Int64
  - flags as bool/None

- [ ] Write the failing tests:
  - `test_alignment_extra_keys` (values load)
  - `test_load_standardized_roundtrip`: ingest vendor B, write, load. Assert `us_gw_id` is `"1"`, not `"1.0"`; `clock_min` dtype is `Int64`; `run_date` is a `date`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(align): alignment config keys and standardized loader`.

### Task 2: Joints and anchors

**Files:** Create `src/ili/align_welds.py`; Test `tests/test_align_joints.py`

**Produces:**
- `build_joints(std) -> pd.DataFrame`, one row per joint `[weld k, weld k+1)` in log order, with columns:
  - `joint_idx, gw_id, start_ft, end_ft, length_ft, anchor_type`
  - `anchor_type` is the type of the first anchor feature (`ANCHOR_TYPES`) whose `log_dist_ft` lies inside the joint, or None
  - The final weld closes the last joint.
- `match_anchors(prev_j, curr_j, cfg) -> list[tuple[int, int]]`, giving (prev joint_idx, curr joint_idx) pairs in order:
  - Positions are compared after a global scale (prev span / curr span, first to last weld).
  - Pairs are formed by a small DP over the two anchor lists: cost = |Δpos|; same type required; skip cost = `anchor_tol_ft`. Pairs with |Δpos| > `anchor_tol_ft` are never formed.

- [ ] Write the failing tests:
  - `test_build_joints_counts`: on synthetic prev, the number of joints = welds − 1, lengths > 0, and 10 anchor joints.
  - `test_anchor_pairs_synthetic`: all 10 pair, in order.
  - `test_anchor_missing_in_one_run` (Review Focus #1): drop one tee's feature row from curr; 9 pairs, and the pairs after the dropped one are still correct.
  - `test_anchor_type_must_match`.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(align): joints and anchor pairing`.

### Task 3: Section DP

**Files:** Modify `src/ili/align_welds.py`; Test `tests/test_align_dp.py`

**Produces:** `align_section(prev_len, curr_len, prev_repair, curr_repair, cfg) -> list[Step]`.
- Inputs are length arrays in ft (curr already drift-scaled) and boolean arrays marking joints inside a repaired span.
- `Step(match_type, prev_idx: tuple[int, ...], curr_idx: tuple[int, ...], mismatch: float, cost: float)`, in order.
- Costs:
  - `mismatch = |a − b| / tol(a, b)`, where a and b are the (summed) lengths
  - 1:1 costs `mismatch`
  - 2:1 and 1:2 cost `mismatch + missed_weld_penalty`
  - inserted and removed cost `gap_penalty`, or `gap_penalty_repair` when the joint is in a repaired span
- The DP is banded: cells with |i·m/n − j| > `band_joints` are skipped, but at least the full table when n or m is 0.
- Ties are broken in the order 1:1, 2:1, 1:2, removed, inserted, so results are deterministic.

- [ ] Write the failing tests (hand-built length lists):
  - `test_identical_sequences_all_1to1`
  - `test_curr_missed_weld_gives_2to1`: prev [40, 39, 41, 38] vs curr [40, 80, 38], giving `2:1` at prev (1, 2)
  - `test_prev_missed_weld_gives_1to2`
  - `test_inserted_and_removed_outside_repair`: an extra 25 ft joint in curr gives `inserted`
  - `test_repair_zone_prefers_gaps`: prev [40, 41, 39] in a repair vs curr [30, 31, 29, 30] in a repair gives 3 removed + 4 inserted
  - `test_offset_start` (Review Focus #2)
  - `test_empty_side` (Review Focus #3)
  - `test_double_missed_weld_no_crash` (Review Focus #4): asserts every joint is accounted for exactly once
  - `test_every_joint_used_once` (property check over random sequences with random edits)
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(align): banded sequence alignment`.

### Task 4: Whole-line alignment and scoring

**Files:** Modify `src/ili/align_welds.py`; Create `src/ili/validation.py`; Test `tests/test_align_line.py`

**Produces:**
- `AlignmentResult(weld_map: DataFrame, changed_sections: DataFrame, summary: dict)`.
- `align_welds(prev_std, curr_std, settings, repairs: DataFrame | None = None) -> AlignmentResult`.
- `weld_map` has one row per match, in line order, with these columns:
  - `section_id, match_type`
  - `prev_gw_id, prev_gw_id_2, curr_gw_id, curr_gw_id_2`; the `_2` columns are the second joint of a 2:1/1:2 match
  - `prev_start_ft, curr_start_ft, prev_length_ft, curr_length_ft, curr_length_scaled_ft`
  - `mismatch, cost, confidence, in_repair, is_anchor`
- How the line is assembled:
  - Anchor joints are forced 1:1. Sections are the runs of joints between consecutive anchor pairs, plus the head and tail.
  - Each section's curr lengths are scaled by prev span / curr span.
  - `in_repair` means the joint midpoint (curr log; prev mapped to curr log through the section scale) lies inside a repairs row.
- Confidence:
  - matches: `high` if mismatch ≤ `confidence_bands.high`, else `medium` if ≤ `medium`, else `low`
  - inserted/removed: `high` inside a repair, else `low`
  - 2:1 and 1:2 are capped at `medium`
- `changed_sections` lists each maximal run of consecutive non-1:1 rows, with columns `section_id, prev_start_ft, prev_end_ft, curr_start_ft, curr_end_ft, match_types, n_prev_joints, n_curr_joints, overlaps_repair`.
- `summary` contains:
  - joints per run and counts by match type
  - `alignment_rate`: joints (both runs) in 1:1/2:1/1:2 rows ÷ all joints
  - `alignment_rate_outside_repairs`
  - counts by confidence and the number of changed sections
- `validation.score_alignment(weld_map, prev_std, curr_std, weld_truth) -> dict`:
  - The truth groups are the segments between consecutive welds reported in both runs. The (p prev, q curr) joint counts give the type:
    - (1,1) → 1:1, (2,1) → 2:1, (1,2) → 1:2
    - a segment with an `in_replacement` weld, or any other shape, → all removed/inserted
  - A group is correct when the predicted rows reproduce exactly its joints and types.
  - Returns `joint_accuracy` (joints in correct groups ÷ all joints), `n_joints`, `wrong_groups` (≤ 20 listed), `replaced_section_flagged` (bool), and `missed_welds_handled` (all 2:1/1:2 truth groups correct).
- `validation.write_metrics(section: str, metrics: dict, path)` merges `{section: metrics}` into an existing JSON.

- [ ] Write the failing tests:
  - `test_default_dataset_meets_acceptance`: 20-mile default seed through ingest → align → score; `joint_accuracy ≥ 0.99`, replaced section flagged, missed welds handled.
  - `test_acceptance_across_seeds`: seeds 1–5 at 5 miles, same thresholds.
  - `test_no_repairs_file_still_aligns`: accuracy ≥ 0.99, with the replacement shown as a changed section.
  - `test_weld_map_covers_every_joint_once`: each joint of each run appears in exactly one row.
  - `test_score_detects_wrong_map`: corrupt one row of a correct map and accuracy drops.
- [ ] Run them and see FAIL. Implement. Run and see PASS. Commit `feat(align): line alignment, changed sections, scoring`.

### Task 5: CLI, outputs, docs

**Files:** Modify `src/ili/align_welds.py`, `README.md`, `docs/DECISIONS.md`; Test `tests/test_align_cli.py`

**Produces:**
- `write_alignment(result, out_dir) -> dict[str, Path]` writes the three files into `out_dir/<prev>_vs_<curr>/`.
- `main(argv)`:
  - usage: `python -m ili.align_welds --prev STD.csv --curr STD.csv [--repairs CSV] [--truth ground_truth_welds.csv] [--settings YAML] [--out DIR]`
  - prints the alignment rate, match counts and changed sections, plus the accuracy when `--truth` is given (and merges `weld_alignment` into `<out>/validation_metrics.json`)
  - exits 2 with `error:` on `ValueError`/`OSError`

- [ ] Write the failing tests:
  - `test_cli_writes_outputs_and_metrics` (synthetic 5-mile)
  - `test_cli_missing_file_exit_2`
- [ ] Run them and see FAIL. Implement. Run and see PASS.
- [ ] Run the CLI on the default dataset, paste the summary, update README/DECISIONS, and commit `feat(align): CLI and outputs`.

### Task 6: Close-out

Run the full suite, do the fresh-reviewer final review, apply one fix pass, update memory, and **STOP for user review before M4**.
