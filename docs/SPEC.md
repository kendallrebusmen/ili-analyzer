# Nemi — ILI Run-Comparison Analyzer: Build Spec

> **Status:** Draft spec for Claude Code. Every engineering parameter marked `PLACEHOLDER` must be confirmed by the integrity engineer before the tool is used on real data.

## 1. Purpose

Build a local Python tool that compares two in-line inspection (ILI) runs on the same hazardous liquid pipeline (49 CFR Part 195). The tool matches anomalies between runs, calculates corrosion growth rates, projects remaining life, classifies features against repair criteria, and generates a draft report and dig list for review by a qualified integrity engineer.

The tool answers four questions:

1. Which anomalies in the current run correspond to which anomalies in the previous run?
2. How fast is each anomaly, and each segment, growing?
3. When will each anomaly reach a failure or repair threshold?
4. Which anomalies must be repaired, by when, and why?

## 2. Ground rules (read before writing any code)

- **Decision aid, not decision-maker.** Every output carries a "DRAFT – for review by qualified personnel" label. The tool never marks anything as approved.
- **Synthetic data only until told otherwise.** ILI results and pipeline location data are sensitive. Work only with synthetic data until the user explicitly confirms real data is approved for use. When real data is in use, work from the schema and small samples; do not print large portions of it into the conversation.
- **Fully local.** The application makes no network calls.
- **Read-only on databases.** Never write to the PCS database or any SQL Server.
- **No hard-coded engineering values.** All tolerances, factors, thresholds, and repair criteria live in YAML config with comments and units.
- **Full traceability.** Store every input, intermediate value, and method choice per feature in the output tables, so any number in the report can be traced back to its source.
- **Units.** US customary internally: inches for wall thickness, depth, length, and width; feet for distances; psig for pressure; mils per year (mpy) for growth; decimal years for time. Convert at ingestion.
- **Work milestone by milestone** (Section 9). At the end of each milestone: run the tests, summarize what was built, list assumptions made, and stop for user review before starting the next one.
- **When the spec is ambiguous, ask rather than guess.** Record decisions in `docs/DECISIONS.md`.

## 3. Tech stack

Python 3.11+, with: pandas, numpy, scipy (`linear_sum_assignment`, `brentq`), pydantic (schema validation), PyYAML, plotly (charts), streamlit (review UI), openpyxl (Excel output), jinja2 (HTML report), pytest. PDF export is optional; if added, choose a library that installs cleanly on Windows without system dependencies. Use a virtual environment and a pinned `requirements.txt`.

## 4. Project structure

```
ili-analyzer/
├── config/
│   ├── settings.yaml
│   ├── repair_criteria.yaml
│   └── vendor_maps/
│       ├── synthetic_vendor_a.yaml
│       └── synthetic_vendor_b.yaml
├── data/
│   ├── synthetic/        # generated test data + ground truth
│   ├── raw/              # real tallies (gitignored)
│   └── reference/        # pipe properties, MOP, HCA, repairs, etc. (real files gitignored)
├── src/ili/
│   ├── schema.py
│   ├── synth.py
│   ├── ingest.py
│   ├── align_welds.py
│   ├── match.py
│   ├── growth.py
│   ├── strength.py
│   ├── remaining_life.py
│   ├── classify.py
│   ├── cp_overlay.py
│   └── report.py
├── app/streamlit_app.py
├── outputs/              # gitignored
├── tests/
├── docs/DECISIONS.md
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Inputs

### 5.1 ILI tallies (two runs)

Delivered by the vendor as Excel or CSV. Layouts differ by vendor, so each vendor gets a mapping file in `config/vendor_maps/` that defines column names, units, clock format, and how the vendor's feature-type vocabulary maps to the standard vocabulary.

After ingestion, every run uses this standard schema:

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | e.g. `2021_vendorA` |
| `run_date` | date | Date of the ILI run |
| `feature_id` | str | Vendor's feature ID |
| `feature_type` | enum | `girth_weld`, `metal_loss`, `dent`, `gouge`, `valve`, `tee`, `fitting`, `sleeve`, `casing`, `marker`, `other` |
| `log_dist_ft` | float | Odometer distance |
| `us_gw_id` | str | Upstream girth weld ID |
| `dist_from_us_gw_ft` | float | Distance from upstream girth weld |
| `joint_length_ft` | float | Length of the joint containing the feature |
| `wt_nom_in` | float | Nominal wall thickness of the joint |
| `depth_pct` | float | Metal loss depth, % of nominal wall |
| `dent_depth_in` | float | Dent depth (dents only) |
| `length_in` | float | Axial length |
| `width_in` | float | Circumferential width |
| `clock_min` | int | Minutes clockwise from 12:00, range 0–719 |
| `surface` | enum | `EXT`, `INT`, `MID`, `UNK` |
| `ml_origin` | enum | Metal loss only: `corrosion`, `manufacturing` (mill-related wall loss), or `unknown`; set from the vendor's feature type by the vendor map, default `unknown` |
| `seam_interaction` | bool | If reported by vendor |
| `dent_metal_loss` | bool | Dent with reported metal loss, cracking, or stress riser |
| `vendor_cluster_id` | str | If the vendor reports clusters |
| `lat`, `lon`, `elev_ft` | float | Optional |
| `comments` | str | Vendor comments |

Validation: required fields present, depth between 0 and 100, clock in range, girth welds monotonic in log distance. Rows that fail validation go to `outputs/validation_issues.csv`; never drop them silently.

### 5.2 Reference data (CSV)

Assume reference files are keyed to current-run log distance for v1 (see Open Questions).

- `pipe_properties.csv`: start_ft, end_ft, od_in, wt_nom_in, grade, smys_psi, seam_type, install_year, coating_type
- `mop_profile.csv`: start_ft, end_ft, mop_psig
- `hca_segments.csv`: start_ft, end_ft, segment_id, could_affect_hca (bool)
- `repairs_between_runs.csv`: start_ft, end_ft, repair_type (`sleeve`, `cutout`, `recoat`, `pipe_replacement`), repair_date, notes
- `foreign_crossings.csv` (optional): log_dist_ft, description
- `next_inspection.csv`: segment_id, planned_next_assessment_date
- `digs.csv` (optional, for back-testing): feature_id, field_depth_pct, field_length_in, dig_date
- `cp_readings.csv` (optional, Milestone 10): test_point_id, log_dist_ft or lat/lon, survey_date, on_mv, instant_off_mv. Eventually sourced from a read-only export of the PCS database.

## 6. Module specifications

### 6.1 Synthetic data generator (`synth.py`) — build first

Generate two tallies of the same line several years apart, plus ground truth, so every later module can be scored against known answers. Use a fixed random seed and expose parameters as CLI arguments.

Defaults:

- 20-mile line; joint lengths mostly 38–42 ft with some short joints
- 5 years between runs
- Odometer drift up to 0.5% plus small local slip
- One 3-joint section replaced between runs (new welds, old features removed); record it in a synthetic `repairs_between_runs.csv`
- A few girth welds missed in one run
- 500 external metal loss features in the previous run, with true growth rates from 0 to 10 mpy on a right-skewed distribution (many near zero)
- 50 new features that first appear above the reporting threshold in the current run
- A handful of internal features, dents (some on top, some with metal loss), valves, and tees as anchors
- Depth sizing noise consistent with ±10% of wall at 80% confidence (σ ≈ 7.8% of wall), applied independently to each run
- Length noise of ±0.75 in at 80% confidence; clock noise of ±15 minutes
- Reporting threshold of 10% of wall (features below it are not reported)
- Several locations where one run reports a cluster and the other reports the individual pits

Outputs: two tallies in deliberately different "vendor" formats (different column names, units, and clock formats) to exercise the vendor maps, a matching set of synthetic reference files, and `ground_truth.csv` with the true pairs, true depths, and true growth rates.

### 6.2 Ingestion (`ingest.py`)

Load each tally through its vendor map into the standard schema. Handle unit conversion and clock parsing (hh:mm, decimal hours, and degrees). Run validation. Write standardized output per run.

### 6.3 Girth weld alignment (`align_welds.py`)

**Goal:** map each joint in the current run to its joint in the previous run.

**Approach:**

1. Use valves, tees, fittings, and other fixed features as hard anchors that split the line into sections.
2. Within each section, run a global sequence alignment (Needleman–Wunsch style dynamic programming) on the joint-length sequences. Cost is based on joint length mismatch, with gap penalties for inserted or removed joints.
3. Support 2:1 and 1:2 matches, where a missed weld in one run makes two joints look like one and the summed lengths agree.
4. Use `repairs_between_runs.csv` to expect inserted or removed joints in repaired sections.

**Output:** `weld_map.csv` with prev_gw_id, curr_gw_id, match_type (`1:1`, `2:1`, `1:2`, `inserted`, `removed`), and confidence. Report the overall alignment rate and list every changed section.

**Acceptance on synthetic data:** at least 99% of joints correctly matched, the replaced section flagged, and missed welds handled.

### 6.4 Anomaly matching (`match.py`)

For each matched joint pair, plus the adjacent joint for features within `near_weld_ft` of a weld:

1. **Normalize axial position.** Scale distance from the upstream weld by the ratio of joint lengths between runs.
2. **Find candidates.** A pair is a candidate if:
   - axial difference ≤ `axial_tol_in`
   - circumferential difference ≤ `clock_tol_min`, with wraparound at 12:00
   - surfaces are compatible (EXT↔EXT, INT↔INT, UNK matches either)
   - feature types are compatible
3. **Score each candidate.** Use a weighted, normalized distance (axial and circumferential), plus a penalty for implausible depth change. Implausible means depth dropped by more than tool tolerance can explain, or grew faster than the configured maximum plausible rate.
4. **Assign one-to-one.** Run `scipy.optimize.linear_sum_assignment` on the cost matrix for each joint so no feature is matched twice.
5. **Handle clusters.** After the 1:1 pass, group remaining unmatched features in each run using the configured interaction rule. Match cluster envelopes (bounding box and maximum depth) to allow many-to-one matches.
6. **Classify every feature** as `matched_1to1`, `matched_cluster`, `new` (current run only), or `missing` (previous run only). For missing features, add a reason hint: `in_repaired_section`, `near_threshold`, or `unexplained`.
7. **Assign confidence** (`high`, `medium`, or `low`) from configured score bands. Low-confidence matches go to the review queue.

**Output:** `match_table.csv` with both runs' attributes side by side, score, confidence, match_type, and review_status (`pending`, `accepted`, `rejected`, `manual`).

**Acceptance on synthetic data:** precision of at least 98% and recall of at least 95%, measured against ground truth on features reported in both runs. Write the confusion metrics to `outputs/validation_metrics.json`.

### 6.5 Growth rates (`growth.py`)

Convert depth percentages to inches using the joint's nominal wall thickness from the current run. Compute Δt in decimal years from the two run dates.

Compute and store every method below for every feature where it applies. A config rule then selects the governing rate.

1. **`raw`**: (d_curr − d_prev) / Δt
2. **`tolerance_adjusted`** (upper bound): (Δd + k·σ_diff) / Δt
   - σ_diff = √(σ_prev² + σ_curr²)
   - Each σ comes from the tool spec. A two-sided ± tolerance at a stated certainty converts to σ with the normal z-value; for example, ±10% of wall at 80% gives σ ≈ 7.8% of wall.
   - Apply the floor `min_rate_mpy`.
3. **`segment_percentile`**: the P-th percentile of raw matched growth within a segment, applied to every feature in that segment.
   - Include negative raw values in the distribution.
   - Apply the floor `min_rate_mpy`.
4. **`new_feature`**: (d_curr − reporting_threshold) / Δt, for features new in the current run.
5. **`full_life`**: d_curr / (current run year − install_year). Also called the whole-life rate.
6. **`default_rate`**: a fixed rate from config, representing the IMP default.
7. **`half_life`**: d_curr / ((current run year − install_year) / 2). This assumes corrosion started at mid-life, so it is twice the whole-life rate.
8. **`joint_percentile`**: the `joint_percentile`-th percentile of the raw rates of matched features in the same current-run joint. It applies only when the joint has at least `joint_percentile_min_matches` matched features; otherwise it is empty. Apply the floor `min_rate_mpy`.
9. **`shrinking_fixed`**: the fixed rate `shrinking_rate_mpy`, for matched features whose reported depth decreased between runs. A depth drop is sizing noise, not negative growth, so a fixed rate stands in for the raw rate.

**Governing rate.** Config picks one of two profiles, `growth.governing_profile`; both are configurable:

`conservative` (default):

- Matched features: the maximum of `segment_percentile` and `tolerance_adjusted`.
- New features: the maximum of `new_feature` and `segment_percentile`.
- Features with no usable history: the maximum of `full_life` and `default_rate`.

`pit_to_pit` (the consultant's practice):

- Matched features whose depth grew: `raw` (pit-to-pit), floored at `min_rate_mpy`.
- Matched features whose depth decreased: `shrinking_fixed`.
- New features: `joint_percentile` when it is available, otherwise `half_life`.
- Features with no usable history: `half_life`.

Five options reproduce the consultant's arithmetic exactly (M12). Their defaults keep the behaviour above:

- `days_per_year` (365.25): the growth interval is the days between the run dates divided by this. The consultant uses 365.
- `rate_decimals` (null): the governing rate is rounded to this many places, halves away from zero. The consultant uses 1.
- `shrinking_applies_to` (`decrease`): `no_increase` also gives `shrinking_fixed` to a matched feature whose depth did not change.
- `joint_percentile_pool` (`raw`): `assigned` takes the joint percentile over the rates the matched features were given (`shrinking_fixed` included, rounded), not their raw rates. It needs `pit_to_pit`.
- `age_basis` (`years`): `dates` takes the pipe age for `full_life` and `half_life` as the days from 1 January of the install year to the current run date, divided by `days_per_year`, not as the difference of calendar years. The consultant uses `dates`.

The two profiles can differ a lot. `pit_to_pit` takes raw rates with no allowance for tool tolerance. Paired with `depth_basis: reported` (Section 6.6), it gives longer remaining lives than `conservative`. Which profile governs is an engineering decision (Open Question 4). The report always states the profile in force and the method that governed each feature.

Record which method governed each feature.

**Recoated areas.** Features inside a `recoat` repair from `repairs_between_runs.csv` still get growth rates. When `exclude_recoated_from_stats` is true, they are left out of the segment and joint percentile distributions and out of the summary statistics, and the summary says how many were left out.

**Growth summary by surface** (the consultant's CGR summary). For EXT and INT corrosion separately, report:

- the maximum reported depth, in % and in inches;
- the pipe age at the current run;
- the half-life and whole-life rates of the deepest feature;
- the share of current-run features that matched (as "n of N");
- the maximum raw (pit-to-pit) rate;
- the mean depth change and mean raw rate of matched features.

Report it twice, with and without recoated areas.

**Length growth:** a config option of `none` (default), `proportional`, or `fixed_rate`.

**Diagnostics:** raw growth histograms, percentage of negative raw rates, and per-segment statistics.

**Acceptance on synthetic data:** report the bias and RMSE of estimated versus true growth. The governing rate should be conservative (greater than or equal to the true rate) for at least 90% of features (`PLACEHOLDER` target).

### 6.6 Remaining strength (`strength.py`)

Implement both methods below. The method is selected in config (default: Modified B31G). Write each function as a pure function and unit-test it.

**Inputs:**

| Symbol | Meaning | Units |
|---|---|---|
| D | Outside diameter | in |
| t | Nominal wall thickness | in |
| d | Assessed depth | in |
| L | Axial length | in |
| SMYS | Specified minimum yield strength | psi |

Assessed depth follows `depth_basis` in config. The default is `reported_plus_tolerance`, meaning reported depth plus the tool's depth tolerance. Length can optionally be adjusted the same way.

Both methods use the same length parameter: z = L² / (D·t).

**Original B31G**

| Quantity | Formula |
|---|---|
| Flow stress | S_flow = 1.1 × SMYS |
| Failure stress, z ≤ 20 | M = √(1 + 0.8z) <br> S_F = S_flow × [1 − (2/3)(d/t)] / [1 − (2/3)(d/t)/M] |
| Failure stress, z > 20 | S_F = S_flow × (1 − d/t) |

**Modified B31G (0.85dL)**

| Quantity | Formula |
|---|---|
| Flow stress | S_flow = SMYS + 10,000 psi |
| Folias factor, z ≤ 50 | M = √(1 + 0.6275z − 0.003375z²) |
| Folias factor, z > 50 | M = 0.032z + 3.3 |
| Failure stress | S_F = S_flow × [1 − 0.85(d/t)] / [1 − 0.85(d/t)/M] |

**Pressures (both methods)**

- Predicted burst pressure: P_F = 2 · S_F · t / D
- Safe operating pressure: P_safe = P_F × `design_factor` (`PLACEHOLDER`: 0.72)
- Estimated repair factor: ERF = MOP_local / P_safe. Values above 1 mean the safe pressure is below MOP.
- Failure pressure ratio: FPR = P_F / MOP_local. Values below 1 mean the predicted burst pressure is below MOP.

`depth_basis: reported` with a zero depth tolerance reproduces the consultant's assessment, which used reported depths only.

**Edge cases:** d = 0, very short L, behavior at the z = 20 and z = 50 boundaries, and d/t ≥ 0.80. When d/t ≥ 0.80, flag the feature as exceeding 80% of wall regardless of the pressure result.

**Reference test case** (independently computed; also verify against the worked examples in ASME B31G-2012 and have the integrity engineer hand-check before relying on it):

Inputs: D = 12.75 in, t = 0.250 in, SMYS = 52,000 psi (X52), d = 50% of wall (0.125 in), L = 4.0 in.

| Method | z | M | S_F (psi) | P_F (psig) |
|---|---|---|---|---|
| Modified B31G | ≈ 5.020 | ≈ 2.016 | ≈ 45,170 | ≈ 1,771 |
| Original B31G | ≈ 5.020 | ≈ 2.240 | ≈ 44,800 | ≈ 1,757 |

RSTRENG effective area is out of scope for v1. Leave an interface for it in case the vendor supplies detailed depth profiles later.

### 6.7 Remaining life (`remaining_life.py`)

For each feature with governing growth rate g (in/yr) and length rate g_L:

- depth(τ) = d_assessed + g·τ
- L(τ) = L + g_L·τ
- MOP_local comes from `mop_profile.csv` at the feature location.

**Times to each criterion.** Solve for each with root-finding (`brentq`), with a fine time-step fallback, out to `horizon_yr`:

| Time | Criterion is met when |
|---|---|
| τ_80 | depth(τ) ≥ 0.80·t |
| τ_burst | P_F(τ) < MOP_local |
| τ_safe | P_safe(τ) < MOP_local |

If a criterion is already met at τ = 0, its time is 0.

**Remaining life:** RL = min(τ_80, τ_burst, τ_safe). Record which criterion governs.

**Predicted failure mode:** `leak` when τ_80 ≤ τ_burst, otherwise `burst`. This follows the consultant's convention: a pit that reaches 80% of wall before the predicted burst pressure falls below MOP is expected to leak.

**Projected condition at the next assessment.** At τ_next (years from the run date to the segment's next assessment date), record:

- depth(τ_next);
- P_F(τ_next) and P_safe(τ_next);
- ERF(τ_next) and FPR(τ_next).

**Condition dates.** Record run_date + τ for each of τ_80, τ_burst and τ_safe, each in its own column.

**Repair-by date:** run_date + RL × `rl_factor` (`PLACEHOLDER`: 0.5).

**Dig flag:** set `dig_required_by_remaining_life` when RL × `rl_factor` is less than the years to the next planned assessment for the feature's segment.

- Next assessment dates come from `next_inspection.csv`.
- If a date is missing, use the config defaults: HCA segments 5 years (not to exceed 68 months), non-HCA segments 10 calendar years.

**Recommended reinspection interval per segment:** the minimum of RL × `rl_factor` across the segment's features, capped at the regulatory maximum.

### 6.8 Repair classification (`classify.py`)

A rule engine reads `repair_criteria.yaml`. It evaluates rules at the current condition (τ = 0) using assessed, tolerance-adjusted values.

**Rule fields:** each rule has `id`, `category`, `applies_to` (`hca`, `non_hca`, or `all`), `feature_types`, `condition`, `description`, `citation`, and `requires_review`.

**Category values:** `immediate`, `60_day`, `180_day`, `evaluate`, or `imp_scheduled`.

**Condition expressions:** evaluate them with a safe expression evaluator, never Python `eval`.

**Dent position:** "top of pipe" means above the 4 and 8 o'clock positions, i.e. `clock_min >= 480 or clock_min <= 240`.

**Default HCA rules.** These are from 49 CFR 195.452(h)(4), reviewed October 2026. Verify them against the current eCFR and the operator's IMP before use; where the IMP is more conservative, the IMP governs.

**Immediate** (the report must state that these require a temporary pressure reduction or shutdown until repaired):

| ID | Condition | Citation |
|---|---|---|
| IMM-1 | Metal loss > 80% of nominal wall, regardless of dimensions | (h)(4)(i)(A) |
| IMM-2 | Predicted burst pressure (P_F) < MOP at the anomaly | (h)(4)(i)(B) |
| IMM-3 | Top-of-pipe dent with any indication of metal loss, cracking, or a stress riser | (h)(4)(i)(C) |
| IMM-4 | Top-of-pipe dent deeper than 6% of nominal diameter | (h)(4)(i)(D) |
| IMM-5 | Anomaly the designated evaluator judges to require immediate action (manual flag only) | (h)(4)(i)(E) |

**60-day:**

| ID | Condition | Citation |
|---|---|---|
| D60-1 | Top-of-pipe dent deeper than 3% of diameter (> 0.250 in for pipe smaller than NPS 12) | (h)(4)(ii)(A) |
| D60-2 | Bottom dent with any indication of metal loss, cracking, or a stress riser | (h)(4)(ii)(B) |

**180-day:**

| ID | Condition | Citation | Review |
|---|---|---|---|
| D180-1 | Dent deeper than 2% of diameter (0.250 in below NPS 12) that affects pipe curvature at a girth weld or longitudinal seam weld | (h)(4)(iii)(A) | Yes |
| D180-2 | Top-of-pipe dent deeper than 2% of diameter (0.250 in below NPS 12) | (h)(4)(iii)(B) | |
| D180-3 | Bottom dent deeper than 6% of diameter | (h)(4)(iii)(C) | |
| D180-4 | Remaining strength calculation shows operating pressure (P_safe) < MOP | (h)(4)(iii)(D) | |
| D180-5 | General corrosion with predicted metal loss > 50% of nominal wall | (h)(4)(iii)(E) | Yes |
| D180-6 | Predicted metal loss > 50% at a foreign pipeline crossing, in widespread circumferential corrosion, or where it could affect a girth weld (use `foreign_crossings.csv` and `girth_weld_proximity_in`) | (h)(4)(iii)(F) | Yes |
| D180-7 | Potential crack indication confirmed as a crack when excavated (manual/field entry only) | (h)(4)(iii)(G) | Yes |
| D180-8 | Corrosion of or along a longitudinal seam weld (`seam_interaction` flag, or clock overlap with the seam position) | (h)(4)(iii)(H) | Yes |
| D180-9 | Gouge or groove deeper than 12.5% of nominal wall | (h)(4)(iii)(I) | |

**Everything else:**

- Any other condition gets `evaluate`, per (h)(4)(iv).
- Features flagged by remaining life get `imp_scheduled`.
- Features flagged `removed_after_run` (a manual flag: the pipe that held them was taken out of the line after the
  inspection, e.g. a temporary or validation spool) get `removed`, whatever their rules say. A removed feature has no
  due date, needs no review and is not on the dig list. Its rule results stay in `classification.csv`, and the
  report lists it with the flag's reviewer and note.

**IMP class summary** (the consultant's layout). The report has a table with one row per class:

- Immediate, 60-day and 180-day (the categories above);
- YTC < N yrs: features with no repair category whose remaining life is under `ytc_flag_years` (years to condition);
- None: every other feature.

Each row gives the number of classified anomalies, how many were previously repaired (inside any repair in `repairs_between_runs.csv`, or removed after the inspection), and how many remain unrepaired. The table is a summary view only; the categories above still drive due dates.

**Non-HCA segments:** remediation falls under 195.401(b), via 195.416(g). Provide a separate rule set in config with operator-defined criteria and timelines from the IMP (`PLACEHOLDER`).

**Due dates:**

- Repair timelines run from the discovery date. Discovery must occur no later than 180 days after the assessment, per 195.452(h)(2).
- The user enters `discovery_date` per run.
- If it is missing, compute the latest permissible discovery date (assessment date + 180 days) and show a prominent warning.
- 60-day due date = discovery + 60 days. 180-day due date = discovery + 180 days.

### 6.9 CP overlay (`cp_overlay.py`, Milestone 10)

This module supports the data-integration and spatial-analysis requirements of 195.452(g).

- Join each anomaly to the nearest CP test point(s) within `cp_join_distance_ft`, attaching the most recent instant-off potential and its history.
- Flag features with above-average growth near test points that fail the CP criterion.
- Plot growth rate against instant-off potential, and growth and potentials together along the line.
- Source data is a CSV export from PCS for v1. A read-only SQL view can replace it later.

### 6.10 Report (`report.py`)

Write all outputs to `outputs/<prev_run>_vs_<curr_run>/`.

**`report.html`** (self-contained, charts embedded, DRAFT banner on every page) contains these sections:

1. **Header:** line name, runs compared, run dates, vendors, and tool types.
   - **Pipe summary.** Covers:
     - length by wall thickness and grade, from `pipe_properties.csv`;
     - design pressure, P_design = 2·SMYS·t·`design_factor`/D;
     - stress at MOP as a percentage of SMYS, MOP·D/(2·t·SMYS), as a range over the line;
     - install year, coating, and HCA coverage (length and percentage).
   - **Length check.** The ILI-measured length of each run compared with the reference length (the end of `pipe_properties.csv`). Show the difference in feet and in percent.
2. **Executive summary:**
   - Immediate conditions at the top, with the pressure-reduction note
   - Counts by repair category, plus the IMP class summary (Section 6.8)
   - Count of digs driven by remaining life
   - Recommended reinspection interval by segment
   - **Anomalies of interest.** Group current-run metal loss within `aoi_group_ft` of each other. List the `aoi_count` groups with the shortest remaining life, ties broken by deepest. Each row has:
     - start and end log distance;
     - surface and origin;
     - maximum depth and number of anomalies;
     - minimum remaining life (years to condition);
     - HCA;
     - a generated one-line description: the deepest feature's depth, length, width and clock position, the depth range of the others in the group, and any repair from `repairs_between_runs.csv` that overlaps it.
3. **Data quality:**
   - Validation issues
   - Weld alignment rate
   - Match statistics by type and confidence
   - Count of pending reviews. While any reviews are pending, the report shows a warning and cannot be labeled final.
4. **Growth and anomaly distribution:**
   - **Growth statistics:**
     - per-segment statistics and histograms, percentage of negative raw rates, and counts by governing method;
     - the growth summary by surface (Section 6.5), with and without recoated areas;
     - the governing profile in force.
   - **Depth bins:** counts of metal loss in each run by depth bin (`depth_bins_pct_wt`; default 10% steps to 80%, then ≥ 80%). There is one table each for all metal loss, metal loss at a girth weld (within `girth_weld_proximity_in`), and metal loss on the long seam. Columns are external corrosion, internal corrosion, manufacturing, and total.
   - **Dent bins:** dent counts by depth as a percentage of OD (`dent_bins_pct_od`), with separate columns for dents at a girth weld and on the long seam.
   - **Per-joint counts:** metal loss per current-run joint and its matched previous-run joint(s), by surface, for both runs, with the change. List the joints with the largest counts and the largest increases. These tables need the weld map and the standardized tallies, but no anomaly matching.
5. **Dig list,** ranked by category and then by remaining life, with these columns:
   - feature IDs in both runs
   - location (log distance, girth weld, distance from weld, clock, lat/lon)
   - depth and length in both runs
   - governing growth rate and method
   - P_F, P_safe, MOP, ERF, and FPR, now and projected to the next assessment
   - RL, its governing criterion, and the predicted failure mode
   - criteria triggered
   - due date
   - review flag
6. **Largest changes:** features with the highest growth rates, and features that moved to a more severe category.
7. **Charts:**
   - depth versus log distance for both runs;
   - growth versus log distance;
   - RL distribution;
   - a map if GPS is available;
   - a unity chart (ILI-predicted versus field-measured depth) when `digs.csv` is provided;
   - **orientation:** clock position versus log distance for both runs, one chart per surface and origin (external corrosion, internal corrosion, manufacturing), with marker size or colour by depth;
   - **elevation:** the elevation profile with internal metal loss marked, when `elev_ft` is available. Internal corrosion tends to collect at low points. This is a view for the engineer, not an internal corrosion model (Section 11).
8. **Methods and assumptions:**
   - Every config value used
   - Tool tolerances
   - Formulas
   - Repair criteria version
   - Full log of match review decisions

**Other files:**

| File | Contents |
|---|---|
| `dig_list.xlsx` | Frozen header row, filters, color-coded by category |
| `match_table.csv`, `growth_table.csv`, `remaining_life.csv`, `classification.csv` | Full per-feature traceability |
| `growth_summary.csv` | The growth summary by surface, with and without recoated areas |
| `anomaly_distribution.csv` | Depth-bin and dent-bin counts for both runs |
| `joint_ml_counts.csv` | Per-joint metal loss counts for both runs, by surface |
| `anomalies_of_interest.csv` | The anomalies-of-interest table |
| `run_manifest.json` | Input file hashes, config hash, code version, and timestamp |

### 6.11 Streamlit app (`app/streamlit_app.py`)

Pages:

1. **Run setup:** select the two tallies, vendor maps, and reference files, then run the pipeline.
2. **Weld alignment:** view the alignment and highlight changed sections.
3. **Match review queue:**
   - Show both features side by side, with a small plot of nearby features in both runs.
   - Offer Accept, Reject, and Manual re-pair actions.
   - Save decisions to `reviews.csv` and re-apply them automatically on every rerun.
4. **Growth and remaining life explorer:** filter by segment, category, and governing method.
5. **Dig list and report export.**

## 7. Configuration

### 7.1 `config/settings.yaml` (starting point; all `PLACEHOLDER` values need engineering sign-off)

```yaml
units: us_customary

tool:
  depth_tolerance_pct_wt: 10        # PLACEHOLDER - from vendor tool performance spec
  depth_certainty: 0.80             # PLACEHOLDER
  length_tolerance_in: 0.75         # PLACEHOLDER
  length_certainty: 0.80            # PLACEHOLDER
  reporting_threshold_pct_wt: 10    # PLACEHOLDER

alignment:
  joint_length_tol_ft: 0.5          # PLACEHOLDER
  joint_length_tol_pct: 1.0         # PLACEHOLDER
  gap_penalty: 5.0                  # tune on synthetic data

matching:
  axial_tol_in: 6                   # PLACEHOLDER
  clock_tol_min: 45                 # PLACEHOLDER
  near_weld_ft: 2
  max_plausible_growth_mpy: 30      # PLACEHOLDER
  weights: {axial: 1.0, circ: 1.0, depth_change: 0.5}
  confidence_bands: {high: 0.3, medium: 0.6}   # score cutoffs; tune on synthetic data
  interaction_rule:                 # PLACEHOLDER - use the IMP's interaction criteria
    axial_spacing_in: 1.0
    circ_spacing_in: 1.0

growth:
  k_sigma: 1.28                     # PLACEHOLDER
  min_rate_mpy: 1.0                 # PLACEHOLDER
  segment_definition: hca_segment   # hca_segment | fixed_length | reference_segments
  fixed_segment_length_mi: 1.0
  segment_percentile: 90            # PLACEHOLDER
  default_rate_mpy: null            # PLACEHOLDER - IMP default rate
  governing_profile: conservative   # conservative | pit_to_pit (consultant's practice)
  shrinking_rate_mpy: null          # PLACEHOLDER - fixed rate for matched features whose depth decreased
  joint_percentile: 90              # PLACEHOLDER - the consultant's "Nth percentile per joint"
  joint_percentile_min_matches: 3   # PLACEHOLDER
  days_per_year: 365.25             # the consultant: 365
  rate_decimals: null               # the consultant: 1
  shrinking_applies_to: decrease    # decrease | no_increase (the consultant)
  joint_percentile_pool: raw        # raw | assigned (the consultant)
  age_basis: years                  # years | dates (the consultant)
  exclude_recoated_from_stats: true
  length_growth: none               # none | proportional | fixed_rate
  length_rate_ipy: 0.0

strength:
  method: modified_b31g             # modified_b31g | original_b31g
  depth_basis: reported_plus_tolerance   # reported | reported_plus_tolerance
  design_factor: 0.72               # PLACEHOLDER

remaining_life:
  horizon_yr: 30
  rl_factor: 0.5                    # PLACEHOLDER
  default_next_assessment_yr: {hca: 5, non_hca: 10}
  hca_max_interval_months: 68

classification:
  criteria_file: config/repair_criteria.yaml
  girth_weld_proximity_in: null     # PLACEHOLDER
  crossing_proximity_ft: null       # PLACEHOLDER

cp_overlay:
  cp_join_distance_ft: 500          # PLACEHOLDER
  criterion_instant_off_mv: -850

report:
  ytc_flag_years: 5                 # PLACEHOLDER - "YTC < N yrs" row of the IMP class summary
  depth_bins_pct_wt: [0, 10, 20, 30, 40, 50, 60, 70, 80]   # last bin is >= 80
  dent_bins_pct_od: [0, 2, 6]       # last bin is > 6
  aoi_group_ft: 2.0                 # PLACEHOLDER - anomalies within this distance form one group
  aoi_count: 10
```

### 7.2 `config/repair_criteria.yaml` (format example; build out all rules from Section 6.8)

```yaml
version: "49 CFR 195.452(h)(4), reviewed Oct 2026 - VERIFY against current eCFR and IMP"
rules:
  - id: IMM-1
    category: immediate
    applies_to: hca
    feature_types: [metal_loss]
    condition: "depth_pct_assessed > 80"
    description: "Metal loss greater than 80% of nominal wall regardless of dimensions"
    citation: "195.452(h)(4)(i)(A)"
    requires_review: false

  - id: IMM-2
    category: immediate
    applies_to: hca
    feature_types: [metal_loss]
    condition: "p_fail < mop_local"
    description: "Predicted burst pressure less than MOP at the anomaly"
    citation: "195.452(h)(4)(i)(B)"
    requires_review: false

  - id: D180-4
    category: 180_day
    applies_to: hca
    feature_types: [metal_loss]
    condition: "p_safe < mop_local"
    description: "Remaining strength calculation shows operating pressure less than MOP"
    citation: "195.452(h)(4)(iii)(D)"
    requires_review: false
```

## 8. Testing and validation

- **Unit tests:**
  - B31G and Modified B31G, including the reference case and the z = 20 and z = 50 boundaries
  - Clock wraparound
  - Unit conversions
  - Each growth method
  - Each classification rule, tested just below, at, and just above its threshold
- **End-to-end on synthetic data:** write match precision and recall, growth bias and RMSE, and conservatism rate to `outputs/validation_metrics.json`, and check them against the acceptance targets above.
- **Back-test on real data (later):** compare predicted depths to field measurements in `digs.csv` with a unity chart, and report the percentage of digs within tool tolerance.
- **Engineering sign-off:** the integrity engineer reviews the formulas, config values, and repair criteria before any real-data report is used.

## 9. Milestones

Complete these in order. Stop after each one for review.

| # | Deliverable | Done when |
|---|---|---|
| M1 | Project skeleton, config files, schema, synthetic generator | Generator produces two vendor-format tallies plus ground truth; tests pass |
| M2 | Ingestion and vendor maps | Both synthetic tallies load into the standard schema; validation report produced |
| M3 | Girth weld alignment | ≥ 99% joint accuracy on synthetic data; replaced section and missed welds handled |
| M4 | Anomaly matching | ≥ 98% precision and ≥ 95% recall against ground truth; metrics file written |
| M5 | Growth rates | All methods computed; bias, RMSE, and conservatism reported |
| M6 | Remaining strength and remaining life | Reference case passes; RL and governing criterion produced for every feature |
| M7 | Repair classification | All Section 6.8 rules implemented with threshold tests; due dates calculated |
| M8 | HTML report and Excel dig list | Full report generated from synthetic data with DRAFT labels and methods appendix |
| M9 | Streamlit review UI | Review decisions persist and re-apply on rerun |
| M10 | CP overlay | Synthetic CP data joined and plotted; ready for PCS export |
| M11 | Consultant-practice methods and outputs | `ml_origin` ingested. `half_life`, `joint_percentile` and `shrinking_fixed` rates and the `pit_to_pit` profile computed, with tests. ERF, FPR, failure mode, condition dates and next-assessment projections in the outputs. Growth summary, IMP class summary, depth/dent bins, per-joint counts, anomalies of interest, orientation and elevation charts in the report, all from synthetic data |
| M12 (done) | Parity check against the consultant (real data; needs the user's explicit approval under Section 2) | The consultant's analysis of a real line rerun through the tool with the `pit_to_pit` profile and `depth_basis: reported`. Differences per anomaly listed and explained. Real data stays in gitignored folders and never enters the repo or the conversation beyond small samples |

## 10. Open questions (answer before using real data)

1. **Tally format:** Which ILI vendor(s) and tool types (MFL axial or circumferential, caliper, combo)? Provide a sample tally header row for each.
2. **Tool tolerances:** What depth and length tolerances and certainty levels are in each run's tool performance spec?
3. **Reference keying:** How are reference data (MOP, HCA, pipe properties) referenced — engineering stationing, current-run log distance, or GPS? If stationing, what is the mapping to log distance?
4. **Growth method:** Which growth method and percentile does the IMP specify? Is there an IMP default rate?
5. **Factors:** What design factor and remaining-life factor should be used?
6. **Interaction rule:** What interaction rule does the IMP use for clustering metal loss?
7. **Non-HCA criteria:** What repair criteria and timelines does the IMP set for non-HCA segments?
8. **Assessment dates:** What are the planned next assessment dates by segment?
9. **Back-testing:** Are field dig results from the previous run available?
10. **Consultant parameters:** What fixed rate (`shrinking_rate_mpy`), joint percentile, and minimum number of matches per joint did the consultant use? Does the IMP adopt them?
11. **B31.8S response:** The consultant reports an "ASME B31.8S response (years)". B31.8S is the gas-pipeline standard. Does the operator want it for a Part 195 line, and if so, by which table? It is not built until this is answered.
12. **Years-to-condition threshold:** Is 5 years the right threshold for the "YTC < N yrs" class?

## 11. Out of scope for v1

- RSTRENG effective area method
- Crack and SCC growth (fatigue analysis)
- Dent strain assessment (dents are classified by depth and flags only)
- Internal corrosion modeling. The elevation view in Section 6.10 is presentation only.
- Trending across more than two runs (but design the data model so a third run can be added later). The consultant traces each feature through four runs; this is the leading candidate for v2.
- ASME B31.8S response intervals (see Open Question 11)
- Writing to any database
- Cloud deployment
