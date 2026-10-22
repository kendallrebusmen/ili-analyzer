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
