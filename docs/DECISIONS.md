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
