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
