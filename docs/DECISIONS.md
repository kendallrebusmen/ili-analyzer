# Decisions

Interpretations of `docs/SPEC.md` and choices the spec leaves open. Newest at the bottom of each milestone.

## M1: skeleton, config, schema, synthetic generator (2026-10-22)

### Project

- **Python 3.13** venv. 3.14 is the machine default, but 3.13 has the most certain wheel coverage for scipy/streamlit.
- **`requirements.txt` pins the ten top-level packages exactly** (`==`), not the full transitive freeze.
- **Synthetic data is gitignored.** It regenerates from the seed recorded in `synth_manifest.json`.
- **Synthetic reference files live in `data/synthetic/reference/`.** That keeps `data/reference/` entirely for real files, all gitignored.
- **Modules added beyond spec §4:** `units.py` (unit conversions, clock formats, tolerance→σ) and `config.py` (YAML loading and validation). Both are shared by several §4 modules.

### Conventions

- **Clock:** `clock_min` 0–719 from 12:00. hh:mm uses a 12-hour face: 0 is `12:00`, 30 is `12:30`, 60 is `1:00`. On input, `0:xx` is accepted as `12:xx`. Degrees × 2 = minutes, and decimal hours × 60 = minutes.
- **Tolerance → σ:** σ = tol / z, where z = Φ⁻¹((1 + certainty)/2). ±10% at 80% gives 7.80% of wall, ±0.75 in gives 0.585 in, and ±15 min gives 11.7 min.
- **`log_dist_ft` of a feature is its centre** and always equals its upstream weld's log distance plus `dist_from_us_gw_ft`. A cluster report's `log_dist_ft` is the centre of its envelope.
- **`repair_criteria.yaml`** holds only the three §7.2 example rules in M1. The full §6.8 set comes in M7.

### Synthetic truth (`synth.py`; all values are CLI parameters or module constants)

| Item | Choice | Why |
|---|---|---|
| Pipe | 12.75 in OD, X52 (52,000 psi), ERW, installed 1975, coal tar | Matches the §6.6 reference case |
| Wall | 0.250 in, except one 0.312 in segment over the middle ~10% (changes at welds) | Exercises per-joint wall in the depth→inch conversion |
| Joints | U(38, 42) ft; 3% short joints U(10, 30) ft | §6.1 "mostly 38–42 with some short" |
| Fittings | 4 valves and 6 tees, each its own U(3, 8) ft joint, spread evenly | Anchors for §6.3 |
| Replacement | 3 consecutive joints replaced by 4 new joints over the same span, ≥5 joints from any fitting or line end; dated mid-interval; new pipe is FBE-coated and installed in the repair year | Exercises `removed` and `inserted`. The plan said 20 joints of clearance, which won't fit on 2-mile test lines |
| Run dates | prev 2021-06-15; curr = prev + round(5 × 365.25) days = 2026-06-15 | |
| Odometer | prev drift 0; curr drift ±U(0.25%, 0.5%); cumulative slip N(0, 0.05 ft) per weld, clipped at 3σ | "Up to 0.5%", never negligible |
| Missed welds | 3 per run (0 allowed). Never adjacent to each other, never at or next to a weld the *other* run missed, never at a fitting, the replaced span, a wall-thickness change, or the line ends. Placed just upstream of a joint holding external metal loss where possible | Guarantees features whose upstream weld is wrong. A crossed miss across runs would give a joint correspondence that no §6.3 match type can express. A miss at a wall change would make the merged joint report the wrong wall |
| Growth | min(Exponential(mean 1.5 mpy), 10 mpy) | Right-skewed, many near zero |
| 500 external ML | True prev depth U(5, 55)%. Accepted only if the pre-drawn previous-run measurement is ≥ threshold, so exactly 500 are *reported* in the previous run. 3 of them sit in the replaced joints | "500 features in the previous run" read as 500 reported |
| 50 new | True prev depth U(2, 12)%; accepted only if prev measurement < threshold and curr measurement ≥ threshold | §6.1 |
| 5 severe | Prev depth U(70, 75)%, growth U(8, 10) mpy, so the current true depth is > 80% | Gives the IMM-1 and P_F rules something to fire on. The plan's 65–75%/6–10 mpy could stop at 77% |
| 8 internal | INT, prev depth U(10, 40)%, growth as external | |
| 12 dents | Half top of pipe (≥ 8:00 or ≤ 4:00), half bottom; 2 top and 2 bottom with metal loss; depths spread over 1–7% of OD; no growth | Straddles the 2/3/6% thresholds |
| Clusters | 6 sites of 2–4 pits, 0.3–0.9 in centre spacing, ±10 clock-min. Even sites are reported as one cluster in prev, odd sites in curr | §6.1. A cluster is one ML row with the bounding-box envelope, the max depth, `vendor_cluster_id`, and comment `CLUSTER` |
| Seam | Seam at 2:00 (60 min); `seam_interaction` when the true clock is within 10 min | Feeds D180-8 later |
| Noise | Depth σ 7.80% of wall (pre-drawn per run, clipped 0–100); length/width σ 0.585 in (floor 0.1 in); clock σ 11.7 min; axial σ 1.0 in; dent depth σ 0.02 in | §6.1. The plan's ±15 min clock tolerance is taken at 80% certainty, like length |
| Threshold | Metal loss below 10% measured is not reported. Dents and anchors are always reported. A dent's ML depth is reported even below the threshold | |
| GPS | Meandering route on a 40° bearing from (32.0, −84.0), rolling elevation 400 ± 60 ft | Lets the M8 map render |

### Synthetic outputs

- **`ground_truth.csv`**: one row per true feature, with true depths, growth, dimensions, and population. It also gives each run's `feature_id` and `status` (`reported`, `reported_in_cluster`, `below_threshold`, `removed_by_repair`), plus a `truth_class`:
  - `matched_1to1`, `matched_cluster`, `new`, `missing`, and `not_reported` (valid, but the generator rarely produces it)
  - **One class per vendor cluster report:** if any pit in a cluster report is matched, every pit in that report is `matched_cluster`. A pit that was below threshold in the other run keeps that in `prev_status`/`curr_status`. So each vendor feature id has exactly one truth class.
  - `wt_nom_in` is the true nominal wall of the feature's joint, so scorers can check the tally wall.
- **Counts:** every `n_*` parameter may be 0; negative counts, or a non-positive `line_miles`/`years_between`/`od_in`, raise `ValueError` naming the parameter (CLI exit code 2).
- **`ground_truth_welds.csv`** (added beyond §6.1 because M3 needs it to score joints): every true weld with each run's gw id and status (`reported`/`missed`/`not_present`). `in_replacement` marks the 5 interior welds of the replaced span (2 removed, 3 inserted); tie-in welds are not counted.
- **Reference files** are keyed to current-run log distance (§5.2) and rounded to 0.001 ft:
  - MOP: 1440 psig for the upstream half, 1200 downstream
  - 4 HCA segments `SEG-01..04`, alternating `could_affect_hca`
  - `next_inspection.csv` deliberately omits SEG-03 to exercise the defaults
  - 3 foreign crossings placed beside external ML
  - 15 digs on the deepest individually reported current features, with field depth = true + N(0, 2%)
  - `cp_readings.csv` is deferred to M10
- **Vendor files round to realistic precision.** A: 0.001 ft/in, 0.1% depth, 0.01 in length. B: 1 mm distances, 0.01 mm wall, 0.1 mm length. The standard frames in memory are unrounded, so M2's ingest oracle compares with tolerances.

### Vendor formats

| | Vendor A, previous run: `tally_prev_vendor_a.csv` | Vendor B, current run: `tally_curr_vendor_b.xlsx`, sheet `Pipe Tally` |
|---|---|---|
| Layout | Header on row 1 | Title rows 1–2 (`Synthetic Vendor B – MFL Run…`, `Run date: 2026-06-15`), header on row 3 |
| Columns | `Feature No, Event, Abs Dist (ft), US GW No, Dist to US GW (ft), Jt Len (ft), WT (in), Depth (%), Dent Depth (in), Length (in), Width (in), O'clock, Int/Ext, Seam Interaction, Dent w/ ML, Cluster ID, Latitude, Longitude, Elevation (ft), Comments` | `Log Dist [m], Feature, Feature ID, Ref GW, Rel. Dist [m], Joint Len [m], WT [mm], Peak Depth [%], Dent Depth [mm], Length [mm], Width [mm], Orientation [deg], Location, Seam, Dent ML, Cluster, Lat, Lon, Elev [m], Comment` |
| Distances | ft | m |
| Wall, dent depth, length, width | in | mm |
| Clock | hh:mm | degrees |
| Surface | `External` / `Internal` | `EXT` / `INT` |
| Event vocabulary | `Girth Weld, Metal Loss, Dent, Valve, Tee` (clusters are `Metal Loss` with a `Cluster ID`) | `GWLD, ML-CORR, ML-CLUS, DENT, VALVE, TEE` |
| Weld ids | `GW00010`, `GW00020`, … | `1`, `2`, … |
| Feature ids | `A-000001…` in log order | `B-000001…` in log order |
| Flags | `Y` / `N` / blank | `1` / `0` / blank |
| Run date | Not in file (from the manifest / vendor map) | Title row 2 |

## M2: ingestion and vendor maps (2026-10-22)

- **Vendor map format** (`config/vendor_maps/*.yaml`, validated by `ingest.VendorMap`; unknown keys are rejected):
  - `vendor`, `run_label`
  - `file` (`format` csv|excel, `sheet`, 1-based `header_row`)
  - optional `run_date` (`cell` plus a regex with one group holding an ISO date)
  - `columns` (standard field → vendor header; `run_id`/`run_date` are not mappable)
  - `units` (distance ft|m, wall in|mm, size in|mm, dent_depth in|mm, elevation ft|m)
  - `clock_format`, `feature_types`, `surfaces`, `true_values`, `false_values`
  - Vocabulary is matched after `strip()` + `casefold()`. Two keys that collide after normalization are a load error.
- **Every cell becomes the text a person sees in the file.**
  - CSV is read with `csv.reader` (`utf-8-sig` absorbs an Excel BOM).
  - Excel is read cell by cell with openpyxl (cached formula values):
    - integer cells stay `1`, not `1.0`
    - **percent-formatted cells are multiplied by 100**, so a depth typed as `23%` (stored 0.23) reads as 23
    - **time-formatted cells become `h:mm`**, so an Excel clock column parses
    - date cells become ISO dates
- **`source_row` is the physical row the record starts on**: the Excel row number, or the CSV line counting blank lines and multi-line quoted cells.
- **Structural problems are a `ValueError`, not a row issue:**
  - a mapped column missing from the file, or appearing more than once
  - a sheet that does not exist
  - a row with more non-blank values than the header has columns
- **Rows with no content at all are skipped**; there is nothing in them to lose.
- **Run date and run id.** The run date comes from `--run ... RUN_DATE`, else from the file. If both exist and disagree, ingestion fails. `run_id = <run year>_<run_label>`, which matches the spec's `2021_vendorA` example. Two runs with the same run id in one invocation are an error.
- **Clock parsing.**
  - hh:mm uses the 12-hour face (`12:30` and `0:30` are both 30 min).
  - Degrees ×2 and decimal hours ×60 give minutes.
  - The valid ranges are 0–360° and 0–<13 h. Within them a full turn wraps, so 360° is 0 and 12.5 h is 30 min, consistent with hh:mm's `12:xx`.
  - Anything outside (e.g. 400°, 360.2°, 13.0 h) is kept unwrapped and reported as `clock_out_of_range`, never wrapped silently.
- **Exclusion policy.** A row with any issue is left out of `standardized/<run_id>.csv` and listed in `validation_issues.csv` (`run_id, source_row, feature_id, field, value, reason`), one line per problem. `source_row` is the 1-based row in the original file. Reasons:
  - `missing_required`, `depth_out_of_range`, `clock_out_of_range`, `duplicate_feature_id` (later occurrences)
  - `girth_weld_not_monotonic`, `unparseable_number`, `unparseable_clock`, `unparseable_flag`
  - `unknown_feature_type`, `unknown_surface`
  - A field that failed to parse is not also reported as `missing_required`.
- **Non-monotonic welds.** The girth welds, in file order, keep the longest strictly increasing sequence of log distances; every other weld is `girth_weld_not_monotonic` and excluded. The clean set is therefore always monotonic: a swap loses one weld, a displaced block loses the block, and an odometer reset loses the shorter part. Alignment (M3) treats excluded welds as missed welds.
  - This replaces the earlier "exclude both welds of a bad pair" rule, which left displaced blocks and resets non-monotonic.
  - Features whose `us_gw_id` names an excluded weld keep that id; M3 re-derives joints from log distance.
- **Value bounds** (`value_out_of_range`), matching the schema plus physical limits:
  - wall and joint length > 0
  - length, width, dent depth, and distance from weld ≥ 0
  - latitude ±90, longitude ±180
  - any non-finite number (`inf`) fails
  - Log distance and elevation may be negative.
- **Depth range** is inclusive: 0 and 100 are valid.
- **`ingest_manifest.json`** records the DRAFT label, the UTC creation time, and per-run tally and vendor-map sha256, rows read/clean, and issue counts.
- **Outputs default to `<project>/outputs`** whatever the working directory.
- **Separate invocations accumulate.** `validation_issues.csv` and the manifest keep the entries of runs ingested earlier into the same `--out`; re-ingesting a run replaces only that run's entries. So ingesting runs one at a time never loses an earlier run's issue record.

## M3: girth weld alignment (2026-10-22)

- **Joints** are built from each run's clean girth-weld rows only (log order): joint k is [weld k, weld k+1), keyed by its upstream weld id. Features are never used to define joints, so a feature citing a weld that M2 excluded cannot affect alignment.
- **Anchors.** A joint is an anchor if a valve, tee, or fitting lies inside it. Its position is the feature's log distance.
  - Anchors are paired by a small alignment over the two anchor lists under a current → previous log map: same type required; the cost is the position difference; pairs more than `anchor_tol_ft` (100 ft) apart are never formed; and an anchor with no counterpart is skipped at a cost of `anchor_tol_ft`.
  - The map is not taken from the runs' endpoints alone, which are wrong by a joint length per joint when one run starts or ends at a different weld. Two starting maps are tried: the endpoints, and the offset most same-type anchor pairs agree on. Each is refined by pairing, refitting a least-squares line to the pairs, and pairing again until the pairs stop changing. The result with the most pairs wins (ties: smaller residual). *(Final-review fix.)*
  - The fitted map (endpoints if no anchors pair) maps previous-run positions to current-run log distance. It interpolates between paired anchors and extrapolates at the fitted scale.
  - A paired anchor bounds a section (its joints forced 1:1) only when the two anchor joints' lengths agree within the `medium` confidence band. Otherwise a weld beside the anchor was missed in one run; forcing 1:1 would turn the true 2:1 into a 1:1 plus a gap. Such an anchor joint is left to the section alignment instead. It still counts as a pair for the log map. *(Final-review fix.)*
- **Different run extents.** A joint whose midpoint lies more than `anchor_tol_ft` beyond the other run's first or last weld (through the log map) has no counterpart. It is removed/inserted outright. Within the head and tail sections, gaps before the first and after the last match cost nothing (semi-global alignment), because charged end gaps push the alignment into coincidental length matches. Swept on seeds 1–5, prev or curr, start or end, 1–100 joints: every case ≥ 0.995 joint accuracy. *(Final-review fix.)*
- **Drift correction** has two passes per section. Pass 1 aligns at the fitted line-wide scale. Pass 2 rescales by the median previous/current length ratio of the section's 1:1 matches within the `high` band, then aligns again. A span ratio spread a local odometer jump (or one missed weld's length) over every joint in the section, e.g. scale 0.884 and 20 low-confidence joints for a 150 ft jump. The median follows the section's own drift but ignores the one changed joint. *(Final-review fix.)*
- **Section alignment** is a banded Needleman–Wunsch over joint lengths. Costs are in units of the length tolerance `max(joint_length_tol_ft, joint_length_tol_pct% × length)`:
  - 1:1 costs its mismatch
  - 2:1 and 1:2 cost their summed-length mismatch + `missed_weld_penalty` (1.0)
  - an inserted or removed joint costs `gap_penalty` (5.0), or `gap_penalty_repair` (0.5) when it lies in a repaired span
  - Ties resolve in the order 1:1, 2:1, 1:2, removed, inserted.
  - The band spans the start-aligned diagonal (j = i) to the end-aligned one (j = i − (n − m)), plus `band_joints` (30) on each side. A head or tail offset of any size is searched and the end is always reachable. The band only limits mid-section excursions from both alignments. It replaced a band centred on the proportional diagonal, which cut off the true path of a head offset over 30 joints (488/600 correct at 40). *(Final-review fix.)*
  - 3:1 matches (two consecutive missed welds) are not modelled, per the spec's match types. They come out as a 2:1 plus a gap, or similar, and are visible as a changed section.
- **Repaired spans.** A joint is in a repaired span when its midpoint falls inside a `repairs_between_runs.csv` row. Previous-run positions are mapped to current-run log distance through the paired anchors. Without repair records the replacement is still found as a changed section, but its edge can shift by a joint when an old joint's length happens to match its neighbour's; this is why the spec feeds repairs in.
- **Confidence:**
  - matches: `high` when the mismatch is ≤ 1.0 tolerance, `medium` ≤ 2.0, otherwise `low`
  - 2:1/1:2 are never above `medium`
  - inserted/removed are `high` inside a repaired span and `low` elsewhere, where they need review
- **Alignment rate** = joints (both runs) in 1:1/2:1/1:2 rows ÷ all joints. It is also reported outside repaired spans.
- **The summary** also reports `n_anchor_pairs` (pairs feeding the log map), `n_anchor_bounds` (those forced 1:1 as section boundaries), `line_scale` (the fitted map's scale), and the range of section scales.
- **Changed sections** are maximal runs of consecutive non-1:1 weld-map rows.
- **Scoring** (`validation.score_alignment`):
  - Truth groups are the joints between consecutive welds reported in both runs. One previous and one current joint is 1:1, two and one is 2:1, one and two is 1:2; anything else, or a group containing a replaced weld, is all removed/inserted.
  - Joint accuracy = joints in exactly reproduced groups ÷ all joints. A group is reproduced only if each of its joints appears in exactly one weld-map row, so duplicated, contradictory, or missing rows count against it. A weld-map id that is not a joint of its run, or a run weld missing from the truth file, is a `ValueError` (CLI exit 2). *(Final-review fix: the first scorer accepted a map with every row duplicated.)*
  - Default dataset: 1.0000, with 6 missed-weld groups and the replacement all correct.
- **New settings keys** (beyond spec §7.1): `missed_weld_penalty`, `gap_penalty_repair`, `anchor_tol_ft`, `band_joints`, and `alignment.confidence_bands`. All are tuning values set on synthetic data.

## M4: anomaly matching (2026-10-26)

- **Global position map.** Piecewise linear through the ends of every matched weld-map row (equals per-joint length-ratio scaling, spec §6.4 step 1). Features are placed by log distance, never `us_gw_id`.
- **Candidates.** Same weld-map row, or adjacent matched rows when both features are within `near_weld_ft` of the shared weld (each feature's distance to its own span end facing the other row).
- **Score.** sqrt((w_axial·a² + w_circ·c²)/(w_axial + w_circ)) + w_depth_change·implausible, with a = axial diff / `axial_tol_in` and c = clock diff / `clock_tol_min`. A missing clock drops the circ term and its weight.
- **Implausibility tolerance.** A two-run difference at the tool's certainty is √2 × `depth_tolerance_pct_wt` = 14.1 % wall.
  - A drop beyond it, or (Δd − 14.1 %)·wt·1000/Δt above `max_plausible_growth_mpy`, is implausible.
  - At 80 % certainty roughly 10 % of genuine matches carry the flag (confidence drops one band). This affects the review queue, not precision/recall.
- **Assignment.** `linear_sum_assignment` per connected component of the candidate graph (equivalent to per joint, and handles near-weld pairs).
- **Vendor cluster reports** (`vendor_cluster_id` set) are excluded from the 1:1 pass, so one pit cannot take the report. They are matched in the cluster pass.
  - **Pits claimed by a report are withheld from 1:1 too.** A metal-loss feature whose box interacts with an opposite-run vendor report's box goes to the cluster pass. The test is the interaction rule (edge gaps ≤ `axial_spacing_in` / `circ_spacing_in` via the current-run OD, same surface), on the previous-run axis. Without this, a nearby individual feature could take a member pit one-to-one.
- **Metal loss only.** Interaction grouping and cluster matching apply to metal_loss (spec §10.6 frames the interaction rule as clustering metal loss).
- **Cluster pass.** Groups come from the interaction rule (edge gaps ≤ `axial_spacing_in` / `circ_spacing_in`, same surface). Group pairs need at least one multi side (≥ 2 members or a vendor report). Box gaps are scored with the same formula.
  - After the group assignment, a vendor cluster report absorbs every leftover opposite-run metal-loss group within tolerance of its envelope.
  - **Absorbed groups keep their own evidence.** Each absorbed group is scored by its box gap to the host's original (pre-absorption) opposite envelope.
    - The match score is the max over the group pair and every absorbed group. `depth_change_pct` and `implausible` (and so the depth penalty in that score) come from the final merged envelopes, because one member pit's depth says nothing against a whole envelope's maximum.
    - The tolerances are unchanged.
  - **`absorbed` column** (after `n_curr`). It is True on a cluster match that absorbed at least one group. Such a match always needs review and is capped at `medium` confidence. The summary counts `n_absorbed`.
- **`axial_diff_in` / `clock_diff_min`.** On `matched_1to1` rows these are centre-to-centre distances. On `matched_cluster` rows they are envelope box gaps (0 on overlap). An absorbed cluster shows the largest gap among the group pair and its absorbed groups.
  - Tool noise of ~1–2 in / ±25 min splits individually reported pits beyond the 1 in interaction spacing. The vendor asserts one envelope holds them. `interaction_rule` is the IMP criterion and was not widened.
- **OD for the interaction rule** comes from `pipe_properties` at the feature's current-run position. Up to `anchor_tol_ft` outside the table uses the nearest row; beyond that is an error.
  - OD is looked up only for features in matched weld-map rows, the only ones that use it. A run covering more line than the other, or than the pipe table, is fine.
  - An empty pipe table, a row without `od_in`, or a weld map missing a required column is a `ValueError`.
- **Missing hints**, in order: `in_repaired_section` (removed row in a repair, or position inside a repairs span), `near_threshold` (depth < reporting threshold + depth tolerance), `unexplained`.
- **Confidence** comes from score bands for matches only. `needs_review` = low confidence, unexplained missing, or an absorbed cluster; `review_status` starts `pending`. A deep feature that is `new` is not flagged in M4.
- **Scoring** (`validation.score_matching`): precision/recall over (prev id, curr id) links. A cluster counts every prev × curr pairing. Confusion is per run and truth class. Acceptance additionally requires every truth `matched_cluster` id predicted `matched_cluster`.
  - **The table must account for every reported anomaly exactly once.** Each run's reported ids are taken from the truth (`{run}_status` reported / reported_in_cluster, anomaly types).
  - The score counts four failures:
    - `n_unreported`: a reported id that is absent from the table.
    - `n_duplicated`: an id that appears in more than one row.
    - `n_wrong_side`: a row whose ids do not fit its type. 1:1 needs exactly one id on each side, a cluster needs both sides non-empty, `new` is curr-only and `missing` is prev-only.
    - `n_duplicate_links`: the same link produced by more than one row.
  - `passes` needs the precision/recall targets, all four counts at 0, and at least one truth link.
- **Results.**
  - Default 20-mile dataset: precision 1.000, recall 0.990 (511/516 links, no false links). All four table checks are 0. There are 3 absorbed clusters, and 42 rows need review.
  - Seeds 1–5 at 5 miles: precision 1.000, recall 0.994–0.998, table checks 0.
  - Dense stress case (5 mi, 5,000 metal-loss features, 60 cluster sites): precision 0.996, recall 0.992.
  - The remaining misses are true 1:1 pairs whose clocks differ by 47–74 min, just over `clock_tol_min` = 45 (PLACEHOLDER). Left for engineer sign-off, not tuned.
  - No settings changed in M4.

## M5: growth rates (2026-10-26)

- **Scope.** Rates go to metal_loss rows with a current side (matched_1to1, matched_cluster, new). Dents, gouges and missing rows get none. Cluster rows use the match table's envelope max depths.
- **Depth basis.** depth_in = pct/100 × the current-run joint wall (curr, else prev, else the pipe table's wt_nom_in).
  - One tool tolerance block serves both runs (spec §10.2 open). σ = `sigma_from_tolerance(tol, certainty)` and σ_diff = √2·σ = 11.0 % wall.
- **History classes.**
  - `matched`: both depths present.
  - `new`: current-only, in pipe installed on or before the previous run year.
  - `no_history`: new pipe installed after the previous run, or a matched row missing a depth.
  - Run years are the calendar years of the run dates.
- **Methods** are exactly spec §6.5. The floor at `min_rate_mpy` applies only to `tolerance_adjusted` and `segment_percentile`. `full_life` is NaN when current year − install year ≤ 0. `default_rate` is NaN while `default_rate_mpy` is null.
- **Segment percentile.** `np.percentile` (linear) over the finite matched raw rates of the segment, negatives included, then floored, and applied to every row in the segment.
  - NaN when the segment has no matched rows or the row has no segment.
  - No minimum sample size was invented. `n_matched` per segment is reported in `growth_segments.csv`.
  - `hca_segment` and `reference_segments` need `--segments`. `fixed_length` builds FIX-### segments from the current run's first girth weld.
  - A supplied segments table with a repeated `segment_id` (compared as strings) or overlapping segments (sorted by start, a start before the previous end) is a `ValueError`; it would pool or double-count features. `compute_growth` checks the table it is given as well. Each governing list needs at least one method.
- **Governing rule** lives in settings `growth.governing` (spec §6.5 default rule). The largest rate wins, ignoring NaN, and ties go to the first listed method.
- **Review rule.** Governing is not floored — a governing rate below `min_rate_mpy` is flagged, and M6 must treat flagged rows explicitly. `needs_review` is True when any reason applies; `review_reasons` (right after it) joins them with `;` in this order, None when there are none:
  - `no_governing`: the governing rate is NaN.
  - `incomplete_rule`: a method in the row's class list is NaN, so the max covered fewer rates than the rule intends. Examples: a `new` or `matched` row with no segment, or in a segment without matched rows (no percentile); any `no_history` row while `default_rate_mpy` is null.
  - `below_min_rate`: governing < `min_rate_mpy`, e.g. `new_feature` for a feature reported below the threshold (−2.0 mpy for 6 % at a 10 % threshold over 5 years).
  - `match_review`: M4's `needs_review` on the match row.
- **M4 evidence carried.** M4's `confidence`, `implausible` and `needs_review` (as `match_needs_review`) are columns after `history_class`, and are required in the match table. `implausible` is carried but does not by itself set `needs_review`; M4 decides that.
- **Missing install year.** A NaN `install_year` in the pipe-properties row a feature uses gives NaN `full_life` and `no_history` (a `new` row needs a known year), flagged `incomplete_rule`, rather than aborting the run (`reference.lookup(..., allow_nan=True)`). A position outside the table is still an error, and a NaN wall in the fallback lookup still raises.
- **Length growth.** `none` gives 0, `fixed_rate` gives `length_rate_ipy`, and `proportional` gives length × governing / depth (aspect ratio kept).
- **Diagnostics are data** (raw-rate histogram counts, % negative raw, per-segment stats). Charts come in M8.
- **Scoring** (`validation.score_growth`). True growth per row is the max `growth_mpy` over the truth rows of its current id(s), so a cluster takes its fastest pit.
  - A row with no governing rate counts as not conservative.
  - `passes` needs conservatism ≥ 0.90 (PLACEHOLDER), every truth current feature scored, and no current id scored twice.
- **Results.**
  - Default 20-mile dataset: conservatism 0.9982 (matched 1.000, new 0.982; `no_history` has no rows). Governing bias +9.41 mpy, RMSE 10.04. Raw bias −0.05 mpy, RMSE 5.49.
  - Default dataset review: 34 of 543 rows need review, all `match_review` (31 low and 3 medium M4 confidence); no `no_governing`, `incomplete_rule` or `below_min_rate`. 8 rows are `implausible` without M4 review.
  - Seeds 1–5: conservatism 0.9945–1.000; 30–48 rows need review, all `match_review`.
  - The acceptance tests also bound |raw bias| ≤ 1.0 mpy (a test bound; raw is unbiased by construction, so a units bug shows there) and need a finite governing RMSE. `score_growth` rejects a `history_class` outside matched/new/no_history.
  - No settings tuned in M5.
- **Observation for the integrity engineer.** Governing rates average about 9 mpy above true growth. Both governing methods add margin over about 5 mpy of raw-rate noise (two ±10 % wall measurements over 5 years).
  - This conservatism flows into the M6 remaining-life intervals. It depends on the PLACEHOLDER `k_sigma`, `segment_percentile` and tool tolerance.
  - The `no_history` path (`full_life`, `default_rate`) is exercised only by unit tests. The synthetic data has no features in post-run pipe.
- **Open items for the integrity engineer.**
  - New-pipe classification compares the calendar years of the runs with the pipe install year. `repairs_between_runs.csv` dates are not consulted, so `pipe_properties` must be updated for repairs.
  - Features in pipe the previous run never covered (runs of different extent) are labelled `new`. Their rate is conservative, but the label is not.
  - `fixed_length` segments start at the current run's first girth weld. Features before it or after the last weld have no segment and are now flagged `incomplete_rule`.
  - M4's `lookup_od` no longer raises on a NaN `od_in` in a row that no position uses.

## M6: remaining strength and remaining life (2026-10-26)

- **B31G formulas** are transcribed from spec §6.6 as named constants.
  - Original: flow stress 1.1·SMYS, 2/3, 0.8, z ≤ 20, else S_flow(1 − d/t).
  - Modified: flow stress SMYS + 10 000 psi, 0.85, 0.6275/0.003375 for z ≤ 50, else 0.032z + 3.3.
- **Reference case reproduced exactly.** Modified: z 5.0196, M 2.0161, S_F 45 172 psi, P_F 1 771.5 psig. Original: M 2.2396, S_F 44 801 psi, P_F 1 756.9 psig. Reviewers' scratch transcriptions of §6.6 (not in the repo) matched `ili.strength` to ≤ 2.5e-16 (relative failure stress) over 20 000 random cases of both methods.
- **Original B31G jump at z = 20** (e.g. P_F 1 627 → 1 122 psig at d/t 0.5) is the spec formula, kept and not smoothed. The integrity engineer should hand-check against ASME B31G-2012.
- **d/t clipping.** d/t is clipped to [0, 1] inside the failure-stress functions, so projections past the wall stay finite. The 80 %-of-wall flag uses the unclipped assessed d/t.
- **Assessed state.** Assessed depth = reported + tool depth tolerance (% wall), capped at 100 %. The length basis is `strength.length_basis` (default reported, PLACEHOLDER). RSTRENG is a stub interface and not selectable.
- **Solver.** The margins (0.80·t − depth, P_F − MOP, P_safe − MOP) are non-increasing in time (checked numerically, including both B31G jumps).
  - A grid of `remaining_life.time_step_yr` (0.05 yr) finds the first crossing interval. `brentq` (xtol 1e-6 yr) refines it, with a step-back so a time is never later than the true crossing. A `brentq` failure falls back to the interval start.
  - A margin of exactly 0 counts as met (spec says P_F < MOP; measure-zero, conservative).
  - Exact-zero plateaus. The margin can be exactly 0 over a stretch: P_F is constant once d/t is clipped at 1, or after the Original B31G z = 20 jump with no depth growth. `brentq` can stop anywhere on such a plateau (it returns the grid END when that margin is 0), up to one grid step late (probe: true crossing 1.013 yr, solver 1.050). When the END margin is exactly 0, or `brentq` returns a point with margin exactly 0, the solver bisects on `margin <= 0` within the interval down to the xtol and returns the lower end, which is never later than the first time the criterion is met.
  - Criteria met today are evaluated without growth, so a feature failing today gets RL 0 and a dig even if its growth rate or another input is missing.
- **Conventions.**
  - Not reached within `horizon_yr`: RL = horizon, `rl_governing` = `beyond_horizon`.
  - Any criterion NaN and none already met: RL NaN, `rl_not_evaluated`, `needs_review`.
  - Negative growth is projected as 0.
  - A NaN MOP, OD or SMYS in a used reference row gives `rl_not_evaluated` (flagged, not raised; M5 raises on a NaN wall).
  - MOP, OD or SMYS ≤ 0 is treated as missing (NaN), as is a negative length (z = L²/(D·t) would fall then rise, so the margins would not be monotone). Before, a MOP of 0 made the pressure criteria never met: RL = horizon, `beyond_horizon`, unflagged. Now the pressure criteria are NaN, the row is `rl_not_evaluated`, and τ_80 (depth and wall only) is kept, so the upper-bound rule below applies. The τ = 0 strength columns are computed from the same usable values; the input columns keep the values as read. Only a wall ≤ 0 makes every criterion NaN.
  - A negative reported depth is clipped to 0 before the depth tolerance is added.
- **Repair-by and dig.** Repair-by = current run date + RL × `rl_factor` (days floored). Dig required = RL × `rl_factor` < years to the next assessment (strict).
- **RL upper bound when RL is NaN.** RL = min of the criterion times, so any finite criterion time bounds it from above even when another criterion is NaN. Before, a NaN pressure criterion (e.g. a NaN MOP) made RL NaN and forced the dig flag False even when τ_80 was finite and small; with the 1 200 psig MOP zone of the default data set to NaN, 261 rows had NaN RL, 0 digs, and 192 provably needed one.
  - Column `rl_upper_bound_yr` (right after `rl_governing`): RL when finite (including `beyond_horizon`), else the smallest finite criterion time, blank when no criterion time is finite.
  - Dig: when RL is NaN, `rl_upper_bound_yr` × `rl_factor` < years to the next assessment. The NaN-MOP case above now gives 192 digs among the 261 NaN-RL rows. A row without a bound keeps dig False. Either way the row keeps `rl_not_evaluated`.
  - Repair-by stays None for a NaN RL: a bound is not an RL.
- **Next assessment source** per feature:
  - `planned`: the segment's date in `next_inspection.csv`.
  - `capped`: an HCA planned date beyond 68 months, cut to 68.
  - `default_hca`: 5 years, capped at 68 months.
  - `default_non_hca`: 10 years.
  - `default_no_segment`: the HCA default, flagged `no_segment` (the conservative choice).
  - A planned date on or before the run date, or a `next_inspection` segment id not in the segments table, is an error.
- **Intervals.** Per segment, min(RL × `rl_factor`, regulatory max) with the max 68/12 yr (HCA) or 10 yr (non-HCA).
  - The segment min uses RL × `rl_factor` for features with an RL and `rl_upper_bound_yr` × `rl_factor` for NaN-RL features with a bound. The segment keeps `needs_review` whenever `n_rl_not_evaluated` > 0.
  - A segment where no feature has an RL or a bound gets no interval (NaN) and `needs_review`. An empty segment gets the regulatory max.
  - Features outside every segment appear in no interval row (only the `no_segment` reason count).
- **Review reasons** appended to M5's: `rl_not_evaluated`, `exceeds_80pct`, `no_segment`. An unreadable M5 flag counts as needs review.
- **CLI.** Run ids come from `growth_summary.json` beside `--growth`. Without it, `--curr`'s run id is used for the current run and the previous is None, with a warning on stderr (`growth_summary.json not found beside <path>; previous run id unknown`). A `growth_summary.json` that is not a JSON object is an error (exit 2). The run date comes from `--curr`. All inputs are read before any output is written.
- **Summary parameters.** `remaining_life_summary.json` records every setting the results depend on: `method`, `design_factor`, `depth_basis`, `length_basis`, the tool's `depth_tolerance_pct_wt`, `depth_certainty` and `length_tolerance_in`, `time_step_yr`, `horizon_yr`, `rl_factor`, `hca_max_interval_months` and `default_next_assessment_yr`.
- **Results (default 20-mile dataset, modified B31G, RL factor 0.5, horizon 30 yr).**
  - 543 features. By `rl_governing`: `tau_80` 161, `tau_burst` 1, `tau_safe` 381, `beyond_horizon` 0, none not evaluated (so `rl_upper_bound_yr` equals `rl_yr` on every row). Unchanged by the final-review fixes: the default data has no NaN or ≤ 0 reference values and no exact-zero plateau.
  - 10 exceed 80 % of wall. 480 need a dig. 44 need review (34 M5 `match_review`, 10 `exceeds_80pct`). Minimum RL 0.00 years (114 features have RL 0: 103 `tau_safe`, 10 `tau_80`, 1 `tau_burst`).
  - Intervals: SEG-01 (HCA) 0.00 yr, SEG-02 0.00, SEG-03 (HCA) 0.00, SEG-04 0.00, against regulatory maxima 5.67 / 10 / 5.67 / 10 yr. Next assessments: 410 planned, 133 `default_hca`.
- **Observation for the integrity engineer.** The high dig count is a property of the synthetic data and PLACEHOLDER settings, not a code defect.
  - It is driven mainly by M5's conservative governing rates (≥ ~7.6 mpy, median ~9.6 mpy on a 0.25 in wall) with `rl_factor` 0.5 and next assessments 5–7 years out.
  - MOP 1 440 psig (70.6 % SMYS) puts P_safe < MOP at τ = 0 for 32 % of features in that zone (8.7 % in the 1 200 psig zone), which drives the 0.0 intervals.

## M7: repair classification (2026-10-26)

- **Expression language.** Conditions are parsed with `ast.parse(mode="eval")` and interpreted from a whitelist: and/or/not, + − × ÷, comparisons (chains, and `in` against constant lists), vocabulary names, constants. Never eval/exec/compile. Length ≤ 500 chars, depth ≤ 32. Type errors are config errors.
  - Three-valued (Kleene) logic: unknown (None) inputs give unknown unless the rest decides. An unknown rule never triggers and never clears: it is listed in `rules_not_evaluable` and sets `requires_review`.
- **Vocabulary.** 24 names: feature_type, could_affect_hca, od_in, wt_in, clock_min, top_of_pipe, depth_pct_reported, depth_pct_assessed, length_in, width_in, p_fail, p_safe, mop_local, dent_depth_in, dent_depth_pct_od, dent_metal_loss, seam_interaction, near_girth_weld, near_foreign_crossing, general_corrosion, widespread_circumferential, dig_required_by_remaining_life, manual_immediate, field_crack_confirmed.
  - Sources: the M4 match table (current side), M6 `remaining_life` (pressures, dig flag, repair-by), the current run (run date), and the pipe, segments and crossings tables.
- **Rule set.** The 16 HCA rules of spec §6.8, thresholds in the YAML. Strict operators as worded.
  - "Smaller than NPS 12" = OD < 12.75 in. Top of pipe = clock ≥ 480 or ≤ 240 (spec).
  - Dent "indication of metal loss, cracking or a stress riser" = vendor flag OR a reported metal-loss depth on the dent.
  - D180-1 "affects curvature at a weld" is approximated as near a girth weld or seam interaction (review required).
  - Non-HCA: NH- mirrors of every HCA rule, all `requires_review`, PLACEHOLDER pending the IMP (spec §10.7). Every non-HCA feature carries `nh_placeholder`.
  - IMM-1 (and NH-IMM-1) applies to metal loss and gouges: > 80 % of wall is immediate under (h)(4)(i)(A) whatever the anomaly type. Before, an 85 % gouge was only D180-9 (180-day). Needs integrity-engineer sign-off.
  - The 16 required ids cannot be weakened by editing the YAML. Each must apply to HCA features (`hca` or `all`), keep its regulatory category (IMM-* immediate, D60-* 60_day, D180-* 180_day) and cover at least the shipped feature types. Anything else is a load error naming the rule.
- **Assessed values.** Metal-loss and gouge depth = reported + tool depth tolerance (capped 100 %), the same basis as M6. Dent depth as reported (no dent tolerance in the spec).
- **PLACEHOLDER settings** (null → unknown, never guessed): `girth_weld_proximity_in`, `crossing_proximity_ft`, `seam_clock_min` / `seam_clock_tol_min`, `general_corrosion_min_dim_in`, `widespread_circ_min_width_frac`.
  - With `seam_clock_min` null, seam interaction is unknown unless the vendor flagged it, so D180-8 is not evaluable for most metal-loss features. Intended until the seam position is configured.
  - Blank crossing or weld positions are errors. A feature outside the weld span is near a weld only if within range of the nearest known one, else unknown. A missing crossings file with the proximity set → unknown (the CLI warns).
- **Category priority** immediate > 60_day > 180_day > imp_scheduled > evaluate. `imp_scheduled` when M6 flags a dig; `evaluate` otherwise ((h)(4)(iv)).
  - Unknown HCA status uses the HCA rules (`hca_assumed`). An unknown dig flag on metal loss → `evaluate` with REMAINING_LIFE not evaluable. Dents and gouges have no remaining life (not applicable).
  - Unknown top of pipe (blank clock). IMM-3/D60-2 and IMM-4/D180-3 are complementary on `top_of_pipe`, so a dent's floor is certain even when its position is not. The rules are evaluated with `top_of_pipe` True and False; the less severe outcome (ties: bottom) governs with its rule and due date, and the other side feeds the worst case. Before, such a dent came out `evaluate` with no due date.
  - Worst case. Every feature gets `worst_case_category`, `worst_case_rule` and `worst_case_due_date` (same due-date rule): the most severe of the governing category, each not-evaluable rule's category and the other side of a top-of-pipe split. Review reason `could_be_more_severe` marks features whose worst case outranks the governing category, so "could be immediate" stands apart from PLACEHOLDER noise. The summary counts it by worst category (`could_be_more_severe_by_worst_category`) and counts every review reason (`counts_by_review_reason`); the CLI prints both.
- **Discovery and due dates.** Discovery is entered per run (`--discovery-date`). Missing → assessment + 180 days with a banner warning (195.452(h)(2)). Before the assessment or more than 180 days after → error.
  - Due date = the earlier of the regulatory date (immediate: discovery; 60/180-day: discovery + 60/180 calendar days) and the feature's M6 repair-by date. `due_date_basis` records which.
  - `imp_scheduled` uses the repair-by date; `evaluate` has none. A due date before discovery → `overdue_at_discovery`.
- **Manual flags** (IMM-5, D180-7) are False until the M9 review UI supplies them.
- **Inputs and settings.**
  - Flag columns in the match, growth and remaining-life tables share one parser (`match.parse_flag_column`): true/false in any case, surrounding blanks ignored, blank → unknown, anything else an error naming the file and column. Before, an Excel re-save (TRUE/FALSE) silently made M4 flags unknown and was rejected by M5/M6.
  - Malformed YAML in any config file is an error naming the file (exit 2), not a traceback. `due_days` must hold both `60_day` and `180_day` as positive whole numbers; `discovery_max_days` must satisfy 0 < x ≤ 180.
  - The criteria file comes from `classification.criteria_file` (relative paths against the project root) unless `--criteria` is given.
  - The summary records the criteria path and sha256, the assessment date, every classification setting, whether a crossings file was given, and the assessed-depth basis (`strength.depth_basis`, `tool.depth_tolerance_pct_wt`).
- **Results (default 20-mile dataset, discovery date not entered: 2026-12-12).**
  - 555 features. By category: immediate 14, 60_day 4, 180_day 119, imp_scheduled 354, evaluate 64. The 14 immediate: 7 HCA (IMM-1 ×5, IMM-2, IMM-4) and 7 non-HCA (NH-IMM-1 ×5, NH-IMM-3 ×2).
  - 553 require review: rule_not_evaluable 543, could_be_more_severe 416, nh_placeholder 285, overdue_at_discovery 144, rule_review 76, match_review 34, exceeds_80pct 10.
  - 416 could be more severe, every one with a 180-day worst case (governing: imp_scheduled 354, evaluate 62). Driven by D180-8/NH-D180-8 (289, seam position null) and D180-5/NH-D180-5 (125, general-corrosion size null), and D180-1/NH-D180-1 (2).
  - 144 overdue at discovery. Due-date basis: regulatory 20, remaining_life 471 (64 `evaluate` rows have no due date).
  - The final-review fixes changed no category count: the default data has no gouges and no dent with a blank clock.
- **Observation for the integrity engineer.** On the synthetic stress case nearly every feature requires review. This is driven by the null PLACEHOLDER settings (seam position especially), the non-HCA placeholder rule set, and upstream M4–M6 flags. Many are overdue at the default discovery date because the synthetic remaining lives are short (see M6). Configuring the PLACEHOLDERs and entering the discovery date clears most of it.

## M8: HTML report and Excel dig list (2026-10-27)

- **Outputs.** `python -m ili.report` writes `report.html`, `dig_list.xlsx`, `category_changes.csv` and `run_manifest.json` into the comparison folder. Everything is built (HTML, workbook, manifest, category changes) before any file is written, so a failure leaves no partial outputs. Each output is then written to a temporary file beside it and moved into place with `os.replace`, `dig_list.xlsx` first: a workbook held open in Excel fails before any output is replaced. `category_changes.csv` has LF line endings, like every other CSV. The report is DRAFT always, including with no pending reviews; no wording marks it final.
- **Input checks.** The report refuses inputs that do not describe one comparison made with the settings and criteria now on disk (`ValueError` naming the file):
  - Settings: every value `remaining_life_summary.json` and `classification_summary.json` record (strength, remaining life, the classification settings, and the tool tolerances they use: `RECORDED_SETTINGS`) must equal the settings file given (numbers to a relative 1e-9, for the JSON round trip); the error names the setting, both values and the summary. Found when a settings edit after classify gave a methods table that disagreed with M6/M7 and a previous-run classification on other settings than the current one. Alignment, matching and growth settings (and `tool.length_certainty`, `tool.reporting_threshold_pct_wt`) are recorded by no summary, so they are the settings at report time; the methods caption says which are which. A key a summary does not hold is not checked.
  - Ingest: `ingest_manifest.json` must list both compared run ids; validation issues are kept for those two runs only.
  - Criteria: the criteria sha256 (raw bytes) must match the classification's. When the two differ only in line endings, the error says so (diagnostic only).
- **Config line endings.** `.gitattributes` checks `config/**/*.yaml` out with LF, so the raw-bytes hashes (`criteria_sha256`, the manifest) match the committed blob whatever `core.autocrlf` is. Found with `core.autocrlf=true`: the CRLF working copy of `repair_criteria.yaml` hashed 4e4c41..., the blob f8075c.... The hashing is unchanged (raw bytes); existing outputs classified with a CRLF copy need `ili.classify` re-run.
- **Previous-run classification (category changes).** The previous run's matched anomalies are classified with today's criteria and settings, so a change in category is a change in the feature, not in the rules.
  - The previous assessed depth is the previous reported depth at tau = 0 (plus the tool tolerance, as for the current run). Wall: M6's `wt_in` (the pipe-table wall) when M6 has the row, else the previous run's nominal wall.
  - Reference lookups (pipe, segments, crossings) use the current-run position; MOP is today's. Rule categories only (no due dates, no remaining life).
  - A value the previous run did not report leaves rules not evaluable (`prev_rules_not_evaluable`), which can make a feature look as if it moved to a more severe category when it did not (its previous category is only as severe as the data allowed). Read the column with the category.
  - `curr_rules_not_evaluable` (from the classification) follows `curr_governing_rule`, so both sides' unevaluable rules show in the moved table.
  - User decision: previous-run classification on today's basis, shown beside the current category, "moved to a more severe category" counted and listed.
- **Dig list scope and ranking.** Every classification row except `evaluate` (immediate, 60_day, 180_day, imp_scheduled). Ranked by category severity, then shortest remaining life (none last), earliest due date (none last), match id. A matched cluster's current side (`;`-joined feature ids) takes the latitude and longitude of the member nearest the row's `curr_log_dist_ft` (first listed on a tie), so it has coordinates and appears on the map.
- **Pending reviews.** Matches with `needs_review` and `review_status` pending. The CLI warns that the report is not final until each is reviewed; the DRAFT label stays either way.
- **Charts.** SVG scatter (not WebGL) so they print. Histograms start at the bin-aligned minimum so negative raw rates show. The depth chart has per-run odometer axes (each run's own log distance, not aligned). The map is offline, without a basemap (spec 2: fully local). The unity chart compares reported current depth with the field depth, within the tool tolerance; no growth to the dig date is applied. A dig without a finite reported or field depth is not compared (neither within nor outside the tolerance) and is listed by id under the chart.
- **Header fields** that the tallies do not carry (line name, tool types) come from `--line-name`, `--prev-tool` and `--curr-tool`; absent ones print "not stated". The methods table marks PLACEHOLDER settings and criteria by detecting the marker in the YAML text.
- **Print.** The DRAFT label prints in an `@page` top-left margin box on every page (Chromium and Edge; Firefox does not render margin boxes, but each section's own label still prints). The on-screen banner is sticky. No PDF export (optional in spec 3): the HTML prints with the label on every page. The dig list (32 columns) and the moved table (17) were clipped at the right edge in print, so in print they show a compact column subset (`DIG_LIST_PRINT_COLUMNS`: category, rule, match id, current feature id, log distance, upstream weld and distance from it, clock, assessed depth, RL, due date, requires review; `MOVED_PRINT_COLUMNS`: 10) with wrapping cells, and a print-only note points to `dig_list.xlsx` / `category_changes.csv` for every column. Checked at Letter landscape width: both fit and read.
- **Workbook.** Row 1 is the DRAFT label (plus the discovery warning when there is one), row 2 the header, data from row 3; panes frozen at A3 and a filter over the header and data. Rows are filled with `CATEGORY_COLOURS`, chosen for at least 4.5:1 contrast with black text. Dates are real Excel dates and blanks empty cells. `report.dig_list_sheet` is checked against Excel's rules (1-31 characters, none of `[ ] : * ? / \`) when settings load.
- **Code version.** The manifest records the package version and, from local git, the commit and whether the tree is dirty; any git failure gives null. It also lists every input's sha256, the tally and vendor-map hashes, and the settings and criteria path and sha256.
- **Results (default 20-mile dataset, with `--digs`).** The old default outputs predated a criteria edit, so `ili.classify` was re-run first (counts unchanged). Re-run again after the LF config checkout (criteria_sha256 now f8075c..., the committed blob) and the final fixes: every number below unchanged; the 3 cluster rows of the dig list now have coordinates.
  - By category: immediate 14, 60_day 4, 180_day 119, imp_scheduled 354, evaluate 64. Dig list 491 rows. Pending reviews 42.
  - 39 of 500 compared anomalies moved to a more severe category: evaluate to 180_day 33, evaluate to immediate 4, 180_day to immediate 2.
  - Back-test: 5 of 15 dug features within the tool tolerance (33 %); the tool over-reads most of the rest (up to 27 points).
  - `report.html` 5.4 MB (plotly.js inlined), `dig_list.xlsx` 114 KB.

## M9: Streamlit review UI (2026-10-27)

- **Scope.** A local Streamlit app (`app/streamlit_app.py`) and a one-command runner (`python -m ili.pipeline`). The app never edits a match row or a result file: it appends decisions to two logs, and a decision takes effect when the pipeline is rerun. Everything is DRAFT; nothing is labelled approved or final.
- **Logs (`review.py`).** `reviews.csv` (decision_id `R000001`..., decision, prev_feature_ids, curr_feature_ids, reviewer, decided_utc, note) and `feature_flags.csv` (flag_id `F000001`..., curr_feature_id, flag, value, reviewer, decided_utc, note). UTF-8, LF, a note is one line.
  - **Append-only.** No code path edits or deletes a row; a correction is a new row. A write is serialised by a process lock (Streamlit sessions are threads of one process), goes through a uniquely named temp file and `os.replace`, and numbers the row highest id + 1 (a row removed by hand never repeats an id). `decided_utc` is normalised to UTC `Z`.
  - **Keyed by feature ids, never `match_id`** (match ids renumber on every rerun). Id sets are sorted and `;`-joined, like the match table's cluster ids. A malformed row (bad decision, blank reviewer, unparseable or zone-less time, repeated id) is a `ValueError` naming the file and line; the CLIs exit 2.
- **Latest wins; supersede.** Rows are taken in `decided_utc` order (ties by id). A decision that shares a feature on the same side with an earlier effective decision supersedes it (`superseded` in `review_status.csv`). For flags the latest row per (feature, flag) decides and earlier rows are `superseded`.
- **User decisions (2026-10-27).**
  - **Reject = split and lock.** The pair's features leave automation on every rerun: the previous side is `missing` (hint `rejected_by_review`), the current side `new`, `review_status` rejected. A rejected pair is never re-paired, even when each was the other's only candidate.
  - **Manual re-pair covers 1:1 and clusters.** One feature per side, or several on either side (a manual cluster). Accept and manual make a forced row (`review_status` accepted / manual) before any automated pass.
  - **Manual flags are in M9.** `manual_immediate` and `field_crack_confirmed`, per current-run feature (a cluster's members are separate ids), read by classification on every rerun.
- **Ruling 6: an accept may confirm a one-sided row.** An accept with one empty side locks the feature out of automation. It stays `missing` (computed hint kept) or `new`, with `review_status` accepted and `needs_review` False. A reject still needs both sides.
- **Validation.** A decision that cannot apply is `not_applied` with a reason, counted in the match summary, printed by the CLI and shown in the app and report; its features stay with automation.
  - Reasons, first broken rule wins: `unknown_feature` (id not in the run), `not_anomaly`, `type_mismatch` (a 1:1 pair of two feature types), `cluster_type` (a cluster with a non-metal-loss member), `out_of_range` (members not all in the same or adjacent weld-map rows, or a feature outside every weld-map row).
  - **The matcher's own row is exempt from `out_of_range` (final review M8).** Accepting a cluster the matcher made could come back `out_of_range` when its members span more than one weld-map row (short pup joints). An accept that fails only the range check is now deferred: its features stay with automation, and when the automated passes make exactly that row (same id sets on both sides, given the other decisions in force) the accept is applied as an accepted forced row (rescored like any accept). Otherwise it stays `not_applied` / `out_of_range`. Manual re-pairs keep the range rule.
  - A reject is validated only for `unknown_feature` and `not_anomaly`: any row the matcher made can be rejected.
  - A decision naming a vendor cluster report counts as a cluster (as in M4).
  - A flag on an id absent from the run is `unknown_feature`; on a non-anomaly, `not_anomaly`.
  - Accepting an absorbed cluster rescores it envelope to envelope: `absorbed` False, and M4's medium-confidence cap no longer applies. This reverses an M4 rule, on purpose: a reviewer has looked at it.
- **Forced rows and their metrics.** An accepted or manual 1:1 row is built as the automated 1:1 rows are (`pair_metrics`), a cluster row as M4's cluster rows (member envelopes, `_envelope_pairs`). Score and confidence are the computed ones (a reviewer's pairing can still score `low`), `needs_review` is False, and growth and later stages treat it like any matched row. `--truth` metrics are computed on the reviewed match table, so a leftover `reviews.csv` changes them: score the matcher with no log in the folder.
- **Default log location; deleting outputs.** Both logs live in the comparison folder (`outputs/<prev>_vs_<curr>/`). `ili.match` and `ili.classify` read them there by default (`--reviews`, `--flags` override). The pipeline never deletes them, but deleting an output folder deletes the decisions: the README says to copy the two files first.
  - **Custom log paths are CLI-only.** `--reviews` / `--flags` exist on the `ili.match` / `ili.classify` commands only. The app writes and `ili.pipeline` reads the logs in the comparison folder, always.
- **Ruling 4: the report shows what the run applied.** The status of each log row is the one recorded by the run that used the log (`review_status.csv` from match, `flag_status.csv` from classify). A row with no recorded status shows "saved after the last run — rerun to apply" and keeps the report from being final (counted apart from the pending matches, see M1 below). A `reviews.csv` / `feature_flags.csv` whose sha256 differs from the one the match / classification summary recorded gives a report warning, not an error.
- **Ruling 5: one pipeline run at a time per process.** A non-blocking lock; a second call returns a failed `pipeline` result ("another pipeline run is in progress"). Stage output is captured in process (`redirect_stdout`), and a stage's `SystemExit` code is its exit code.
  - **Limit: both locks are per process.** The pipeline lock and the log-append lock (`review._APPEND_LOCK`) serialise the sessions of one Streamlit process. Two app processes, or the app and a command-line run, on the same comparison folder are not serialised: run one app per folder and do not run the CLI on a folder the app has open.
- **Ruling 10: latest intent wins, even when it cannot apply.** A later decision that shares a feature with an earlier one supersedes it even when the later one is itself `not_applied`: the earlier decision does not come back. Both statuses are surfaced (`superseded` for the earlier row, `not_applied` with its reason for the later) in `review_status.csv`, the app and the report, so the reviewer sees that the features went back to automation.
- **Ruling 3: `pipeline_inputs.json`.** Every run writes it into the comparison folder (right after ingest). The app reads it to rerun after a restart and never writes it; Rerun is disabled when there is neither one nor a setup in the session.
- **Ruling 2: sidebar radio routing.** The app routes with `st.sidebar.radio` (key `page`) over a dict of page functions: `AppTest` cannot drive `st.navigation` callable pages in streamlit 1.66.
- **Ruling 1: `review: nearby_window_ft`** (settings, default 10 ft) sets the review page's nearby-features window. Presentation only; it changes no number in the analysis.
- **Pages.** Run setup; Weld alignment; Match review queue (lowest confidence first, then position; keyed by feature-id pair; a "Correct a decision" view below the queue); Growth and remaining life explorer (segment, governing category and governing growth method filters, empty selection means all, with the growth and remaining-life charts of the filtered rows); Dig list and report export (the report's dig list, the flag editor, the flags in force with the status the last run recorded, Rerun, and downloads of `report.html`, `dig_list.xlsx` and `run_manifest.json`, each with a DRAFT caption). The flag editor writes one row per changed flag and needs a reviewer name.
- **Telemetry off, localhost only (controller Ruling 8, final review I1).** `app/.streamlit/config.toml` (script-level config: Streamlit reads it beside the script whatever the launch directory; it was at the project root, which only applied when launched from there) sets `browser.gatherUsageStats = false`, `server.address = "localhost"` and `server.headless = true`. The README command also passes `--server.address localhost --browser.gatherUsageStats false`. A user config, an environment variable or a flag can still override the file, so the app checks the options in force and refuses to run (an error and `st.stop`, the DRAFT label still shown) unless `server.address` is `localhost` / `127.0.0.1` and `browser.gatherUsageStats` is off. No page fetches anything from a CDN.
- **Final-review fixes (2026-10-27).**
  - **C1: a decision is bound to what was on screen.** The queue was keyed by `match_id` and the row re-read on the click's rerun; match ids renumber on every pipeline run, so a click after a rerun could save another pair's ids (reproduced: shown A-000805 / B-000826, saved A-000033 / B-000035). The queue is now keyed by the feature-id pair, and Accept / Reject / Manual are button callbacks whose arguments are the comparison folder, the ids shown and `match_table.csv`'s modification time at render. When the table changed in between, the click saves nothing and says so. After the page's Rerun (and Run setup's Run) the script reruns, so the page and the sidebar caption show the new outputs.
  - **I2 (controller Ruling 9): correction view.** The review page's "Correct a decision" lists the decisions in force (the current log reduced by `effective_reviews`: decided rows and rows saved since the last run), each with the status the last run recorded or "saved after the last run", with the same bound Accept / Reject / Manual actions. A reviewer supersedes their own decision there (a rejected pair can be re-accepted).
  - **M1: pending wording.** The report and the report CLI say "N matches pending review, plus M log rows saved after the last run" (the second part only when M > 0); a log row is not counted as a match. The not-final warning stays when either is above zero.
  - **M2: hash what was parsed.** `ili.match` / `ili.classify` record the sha256 of the log bytes they parsed (read once), not of a second read, so an append in between shows up as a changed log in the report.
  - **M3, M5: Excel.** A log append that fails with `OSError` (on Windows, `PermissionError` while the CSV is open in Excel) is a message asking to close the file. Logs are read as `utf-8-sig`, so an Excel "CSV UTF-8" save (byte-order mark, CRLF) still loads, and a header error carries the same "fix or remove the row by hand" hint as a row error.
  - **M4: unreadable folder.** A file of the comparison folder that cannot be read (a malformed log row) is an error naming the file and line on every page, not a traceback.
  - **M7: rerun setup per folder.** A setup run in this session is used for a folder only when it ran that folder; otherwise the folder's own `pipeline_inputs.json` (else Rerun is disabled).
  - **M8:** see Validation (the matcher's own row is exempt from `out_of_range`).
- **Results (default 20-mile dataset, `python -m ili.pipeline` as in the README).**
  - First run, no logs: 42 pending reviews, no decisions, categories immediate 14, 60_day 4, 180_day 119, imp_scheduled 354, evaluate 64.
  - Then one decision of each kind appended with `review.append_review` / `append_flag`: an accept and a reject of two pending paired rows, a one-sided accept of a pending `missing` row, a manual 1:1 re-pair of a nearby `missing` and `new` pair, and a `manual_immediate` flag on B-000006. Rerun: `decisions: applied 4, not_applied 0, superseded 0`; `review_status` pending 581, accepted 2 (the paired accept and the one-sided accept), rejected 2 (the pair's `missing` and `new` rows), manual 1; flags: 1 applied, 0 not applied.
  - Pending reviews 42 to 38. Match-type counts are unchanged (the reject adds one `missing` and one `new`, the manual pair removes one of each). The one-sided accept stays `missing` (hint `unexplained`).
  - Categories: immediate 15, 60_day 4, 180_day 119, imp_scheduled 352, evaluate 65. Two features moved: B-000006 imp_scheduled to immediate (the flag), and B-002088, which no decision names, imp_scheduled to evaluate. Verified against the no-decisions baseline: the decisions, chiefly the manual pair (which makes B-002255 a matched feature in SEG-03, `n_matched` 122 to 123), move the segment's percentile rate from 10.329 to 10.243 mpy, and B-002088's governing rate (segment_percentile) with it; its remaining life rises from 9.99 to 10.07 yr and its governing rule changes from REMAINING_LIFE (imp_scheduled) to EVALUATE.
  - After the final-review fixes the README command was rerun on the default dataset (no logs in `outputs/`): unchanged, and the report CLI prints "pending: 42 matches pending review (the report cannot be final)".
  - The demo ran with `--out` in a scratch folder, not the default `outputs/`, which is where a reviewer's real logs go; the default dataset's folder holds no demo decisions.

## UI: Enterprise Suite look (2026-10-27)

- **Chosen by the user** from ten mocked-up directions ("9 · Enterprise Suite"). Presentation only: no number, rule or output changes.
- **Theme.** `app/.streamlit/config.toml`: `[theme]` holds the shared radius, headings and fonts; `[theme.light]` the light palette: primary #0A6ED1, page #F5F6F7, cards #FFFFFF, text #223548, border #D9DEE3, red #AA0808, 6 px radius, smaller headings. Fonts are the system sans stack (Segoe UI first): web fonts would be a network call (spec §2).
- **Shell.** `ili.app_theme` holds the rest as one CSS block and small HTML builders: navy shell bar (product, app, DRAFT pill, reviewer initials), breadcrumb (Run comparisons / folder / page; the full path is its tooltip), record header with status pills, white cards, red-outline Reject, and an action bar that stays at the bottom of the window. Every piece of text in that HTML is escaped; the CSS has no `url()` or `@import`.
- **Layout.** The sidebar is gone: pages are tabs (the same `page` radio, drawn as tabs) and the reviewer box sits beside the breadcrumb. Streamlit's header is see-through and shows only its menu; the Deploy button stays hidden (`client.toolbarMode = "viewer"`; the app is local only).
- **Review page.** The selected match shows a header (id, confidence pill, Pending, type, score, hint); the side-by-side table shows at most three decimals and blanks for missing values (the match table keeps full precision); Accept is the primary button.
- **Dark mode (added the same day, at the user's request).** `[theme.light]` and `[theme.dark]` hold one palette each (dark: page #12171C, cards #1D232A, text #EAECEE, links #4DB1FF, border #3B4752, red #FF8888; the primary blue and the orange DRAFT pill stay). The app follows the computer's setting until the reviewer picks System, Light or Dark in the menu at the top right; Streamlit remembers that choice in the browser. The custom CSS uses `light-dark()` pairs, which resolve against the `color-scheme` Streamlit sets on the app root, so it switches with the theme and needs no Python-side detection (`st.context.theme` is documented as unreliable during a switch). Charts and tables are themed by Streamlit itself. The menu's links to Streamlit's website (Get help, Report a bug) are removed, so nothing in the app points off this computer. Tests hold every text colour to WCAG AA (4.5:1) in both themes.
- **CSS hooks are Streamlit 1.66 DOM details** (`stRadioOption`, `data-selected`, `st-key-*` classes). A Streamlit upgrade can break the look without breaking a test; re-check the pages after one.

## Spec amendment: the consultant's practice (2026-10-27)

- **Source.** The user shared the consultant's analysis workbook for a real line (values only, no formulas), so we could see their method. It was read in place on the user's drive. No value from it is in this repo; the spec quotes only method and layout. The user agreed to adopt everything recommended from the comparison.
- **What matched already.** Their Modified B31G burst and safe pressures match spec 6.6 exactly; one anomaly was recomputed by hand. Their assessment used reported depths (depth tolerance 0), which `depth_basis: reported` reproduces.
- **What the spec gains.** The amendments are in spec 5.1, 6.5–6.8, 6.10, 7.1 and 9–11:
  - **Growth (6.5):**
    - `half_life`, `joint_percentile` and `shrinking_fixed` rates;
    - a `pit_to_pit` governing profile beside the default `conservative` one;
    - recoated areas left out of the statistics;
    - a growth summary by surface.
  - **Strength and remaining life (6.6–6.7):**
    - ERF and FPR;
    - predicted failure mode (leak or burst);
    - condition dates;
    - values projected to the next assessment.
  - **Classification (6.8):** an IMP class summary with a "YTC < N yrs" row.
  - **Report (6.10):**
    - pipe summary and length check;
    - anomalies of interest;
    - depth-bin and dent-bin tables, and per-joint counts;
    - orientation and elevation charts.
  - **Schema (5.1):** an `ml_origin` field (corrosion, manufacturing, unknown).
  - **Milestones (9):** M11 builds all of this on synthetic data. M12 is the parity check against the consultant, on real data. It waits for the user's explicit approval under spec Section 2.
- **Default unchanged.** `conservative` stays the default profile. The consultant's approach (raw pit-to-pit growth on reported depths) gives longer remaining lives than ours. Which approach governs is the integrity engineer's call (Open Questions 4 and 10).
- **Not adopted.**
  - The consultant's "ASME B31.8S response (years)": B31.8S is the gas standard; see Open Question 11.
  - Trending across four runs: still out of scope, and the leading candidate for v2.
