# ILI Run-Comparison Analyzer

> **DRAFT – for review by qualified personnel.** This tool is a decision aid, not a decision-maker. Nothing it produces is approved until a qualified integrity engineer reviews it.

Compares two in-line inspection (ILI) runs on the same 49 CFR Part 195 hazardous liquid line. It matches anomalies, calculates growth, projects remaining life, classifies against repair criteria, and drafts a report and dig list. The full spec is in [`docs/SPEC.md`](docs/SPEC.md), and interpretations and decisions are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Ground rules

- **Synthetic data only** until real data is explicitly approved for use.
- Fully local: no network calls, and read-only toward any database.
- Every engineering value lives in `config/*.yaml`. Values marked `PLACEHOLDER` need engineering sign-off.

## Status

| Milestone | State |
|---|---|
| M1 Skeleton, config, schema, synthetic generator | done |
| M2 Ingestion and vendor maps | done |
| M3 Girth weld alignment | done |
| M4 Anomaly matching | done |
| M5 Growth rates | done |
| M6 Remaining strength and life | done |
| M7 Repair classification | done |
| M8 HTML report and Excel dig list | done |
| M9 Streamlit review UI | done (awaiting review) |
| M10 | not started |
| M11 Consultant-practice methods and outputs (synthetic data) | done (awaiting review) |
| M12 Parity check against the consultant (Line 930) | done (awaiting review) |

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q            # test suite
.\.venv\Scripts\python.exe -m ili.synth                  # synthetic data -> data/synthetic/
.\.venv\Scripts\python.exe -m ili.synth --help           # every generator parameter
.\.venv\Scripts\python.exe -m ili.synth --seed 7 --line-miles 5 --out data/synthetic/small

# Ingest tallies into the standard schema -> outputs/standardized/<run_id>.csv,
# outputs/validation_issues.csv, outputs/ingest_manifest.json
.\.venv\Scripts\python.exe -m ili.ingest `
    --run data\synthetic\tally_prev_vendor_a.csv config\vendor_maps\synthetic_vendor_a.yaml 2021-06-15 `
    --run data\synthetic\tally_curr_vendor_b.xlsx config\vendor_maps\synthetic_vendor_b.yaml
```

```powershell
# Align girth welds -> outputs/<prev>_vs_<curr>/weld_map.csv, changed_sections.csv, alignment_summary.json
# (--truth scores against synthetic ground truth into outputs/validation_metrics.json)
.\.venv\Scripts\python.exe -m ili.align_welds `
    --prev outputs\standardized\2021_vendorA.csv --curr outputs\standardized\2026_vendorB.csv `
    --repairs data\synthetic\reference\repairs_between_runs.csv `
    --truth data\synthetic\ground_truth_welds.csv

# Match anomalies -> outputs/<prev>_vs_<curr>/match_table.csv, match_summary.json
# (--truth scores links into outputs/validation_metrics.json under anomaly_matching)
.\.venv\Scripts\python.exe -m ili.match `
    --prev outputs\standardized\2021_vendorA.csv --curr outputs\standardized\2026_vendorB.csv `
    --weld-map outputs\2021_vendorA_vs_2026_vendorB\weld_map.csv `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --repairs data\synthetic\reference\repairs_between_runs.csv `
    --truth data\synthetic\ground_truth.csv

# Growth rates -> outputs/<prev>_vs_<curr>/growth.csv, growth_segments.csv, growth_summary.json, growth_summary.csv
# (--truth scores conservatism into outputs/validation_metrics.json under growth)
.\.venv\Scripts\python.exe -m ili.growth `
    --prev outputs\standardized\2021_vendorA.csv --curr outputs\standardized\2026_vendorB.csv `
    --matches outputs\2021_vendorA_vs_2026_vendorB\match_table.csv `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --repairs data\synthetic\reference\repairs_between_runs.csv `
    --truth data\synthetic\ground_truth.csv

# Remaining strength and life -> outputs/<prev>_vs_<curr>/remaining_life.csv, reinspection_intervals.csv, remaining_life_summary.json
.\.venv\Scripts\python.exe -m ili.remaining_life `
    --curr outputs\standardized\2026_vendorB.csv `
    --growth outputs\2021_vendorA_vs_2026_vendorB\growth.csv `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --mop data\synthetic\reference\mop_profile.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --next-inspection data\synthetic\reference\next_inspection.csv

# Repair classification -> outputs/<prev>_vs_<curr>/classification.csv, classification_summary.json
# (add --discovery-date YYYY-MM-DD once known; without it the latest permissible date is used, with a banner)
.\.venv\Scripts\python.exe -m ili.classify `
    --curr outputs\standardized\2026_vendorB.csv `
    --matches outputs\2021_vendorA_vs_2026_vendorB\match_table.csv `
    --remaining-life outputs\2021_vendorA_vs_2026_vendorB\remaining_life.csv `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --crossings data\synthetic\reference\foreign_crossings.csv

# Report -> outputs/<prev>_vs_<curr>/report.html, report.pdf, dig_list.xlsx, category_changes.csv, run_manifest.json
# (--digs is optional field data for the back-test; --repairs marks previously repaired and recoated features;
# --line-name, --prev-tool and --curr-tool fill the header; --prepared-for and --prepared-by name the client and
# the author on the PDF cover)
# Also writes anomaly_distribution.csv (depth and dent bins), joint_ml_counts.csv (per-joint counts),
# anomalies_of_interest.csv and imp_class.csv (each classified feature's IMP class).
.\.venv\Scripts\python.exe -m ili.report `
    --prev outputs\standardized\2021_vendorA.csv `
    --curr outputs\standardized\2026_vendorB.csv `
    --comparison outputs\2021_vendorA_vs_2026_vendorB `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --mop data\synthetic\reference\mop_profile.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --crossings data\synthetic\reference\foreign_crossings.csv `
    --digs data\synthetic\reference\digs.csv `
    --repairs data\synthetic\reference\repairs_between_runs.csv
```

### One command, and the review app

```powershell
# Every stage, ingest to report, in one run (stops at the first failure; --until <stage> stops early).
# Writes outputs/<prev>_vs_<curr>/..., including pipeline_inputs.json, which the app reads to rerun.
.\.venv\Scripts\python.exe -m ili.pipeline `
    --prev-tally data\synthetic\tally_prev_vendor_a.csv --prev-map config\vendor_maps\synthetic_vendor_a.yaml `
    --prev-run-date 2021-06-15 `
    --curr-tally data\synthetic\tally_curr_vendor_b.xlsx --curr-map config\vendor_maps\synthetic_vendor_b.yaml `
    --pipe data\synthetic\reference\pipe_properties.csv --mop data\synthetic\reference\mop_profile.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --next-inspection data\synthetic\reference\next_inspection.csv `
    --repairs data\synthetic\reference\repairs_between_runs.csv `
    --crossings data\synthetic\reference\foreign_crossings.csv --digs data\synthetic\reference\digs.csv `
    --out outputs

# The review app (local only: localhost, no telemetry; app/.streamlit/config.toml sets both, and the app
# refuses to run if the options in force differ). Pages: Run setup, Weld alignment, Match review queue
# (with "Correct a decision"), Growth and remaining life explorer, Dig list and report export.
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address localhost --browser.gatherUsageStats false
```

The app's **Run setup** page is four steps:
1. **Inspections:** the two tallies and their vendor maps.
2. **Pipeline data:** pipe properties, MOP profile, HCA segments and next inspection dates.
3. **Optional data,** collapsed: repairs, foreign crossings and dig results.
4. **Report details,** collapsed: line name, discovery date, tool types, and the PDF cover's "prepared for" and "prepared by".

The output folder, settings and criteria files sit under **Advanced**.

Each pipeline-data and optional table can be chosen as a file or typed in ("Type it in"). The editor starts from the
chosen file's rows, if there is one. On **Run**, typed tables are saved as the standard CSVs in
`<output folder>/entered/`: `pipe_properties.csv`, `mop_profile.csv`, `hca_segments.csv`, `next_inspection.csv`,
`repairs_between_runs.csv` and `foreign_crossings.csv`. The pipeline reads them like any other file, and the run
manifest hashes them.

A row missing a required value (marked *) is refused with the table and row named. Blank rows are ignored.

### The consultant's practice (M11)

`growth.governing_profile` in `config/settings.yaml` picks the governing growth rule: `conservative` (the default; the
`growth.governing` lists) or `pit_to_pit` (raw pit-to-pit growth, as the consultant does it). `growth_summary.json` and
the report state the profile in force. To run `pit_to_pit` the way the consultant's assessment did, copy
`config\settings.yaml`, set `growth.governing_profile: pit_to_pit`, `strength.depth_basis: reported`,
`tool.depth_tolerance_pct_wt: 0` and `growth.shrinking_rate_mpy` (null by default, `PLACEHOLDER`: until it is set,
pit_to_pit rows whose depth decreased are flagged `needs_review`), then pass the copy:

```powershell
.\.venv\Scripts\python.exe -m ili.pipeline ... --settings config\settings_pit_to_pit.yaml --out outputs_pit_to_pit
```

`pit_to_pit` normally gives longer remaining lives than `conservative`; which one governs is the integrity engineer's
call (spec Section 10, Open Questions 4 and 10). New outputs, in the comparison folder:

| File | Contents |
|---|---|
| `growth_summary.json` / `growth.csv` | profile, `half_life`, `joint_percentile` and `shrinking_fixed` rates, recoat exclusion (`repairs_file_given`, `n_excluded_recoated`) |
| `growth_summary.csv` | the growth summary by surface (EXT, INT), with and without recoated areas |
| `remaining_life.csv` | ERF, FPR, failure mode, condition dates, values projected to the next assessment (`exceeds_80pct_next`; pressures blank once the projected depth reaches the wall) |
| `imp_class.csv` | each classified feature's IMP class, years to condition (RL, else its bound) and whether it was previously repaired; the dig list carries `imp_class` and `previously_repaired` |
| `report.html` | IMP class summary with the "YTC < N yrs" row (`report.ytc_flag_years`), pipe summary and length check, orientation and elevation charts, methods appendix |
| `anomaly_distribution.csv` | metal-loss depth bins and dent bins (`report.depth_bins_pct_wt`, `report.dent_bins_pct_od`) |
| `joint_ml_counts.csv` | metal-loss counts per joint, both runs, by surface |
| `anomalies_of_interest.csv` | the anomaly groups with the shortest remaining life (RL, else its upper bound; `min_rl_is_bound`), then the deepest (`report.aoi_group_ft`, `report.aoi_count`) |

`--repairs` is accepted by `ili.growth` and `ili.report` (and passed by `ili.pipeline`). Every step reads it the same
way: `start_ft`, `end_ft` and `repair_type` are required, and `repair_type` must be `sleeve`, `cutout`, `recoat` or
`pipe_replacement` (case, spaces and hyphens are normalised, so "Pipe Replacement" is accepted). `ml_origin` (corrosion,
manufacturing, unknown) comes from each vendor map's `ml_origins`. Decisions: `docs/DECISIONS.md` (M11).

Reviewers' decisions are saved in the comparison folder: match decisions in `reviews.csv` and manual flags in `feature_flags.csv`. Both are append-only logs (a correction is a new row; the latest wins). `ili.match` reads `reviews.csv` and `ili.classify` reads `feature_flags.csv` from the comparison folder by default (`--reviews` and `--flags` name other files on those two commands only; the app and `ili.pipeline` always use the comparison folder's logs), so a decision takes effect the next time the pipeline is rerun, from the app's Rerun button or the command above. **The logs live beside the outputs: deleting an output folder deletes the decisions.** Keep a copy of `reviews.csv` and `feature_flags.csv` before clearing outputs.

The run date is the third `--run` value. It can be left out when the vendor map says where the file states it (vendor B's title row). The run id is `<run year>_<run_label from the vendor map>`.

### The PDF deliverable (`report.pdf`)

The report step also writes `report.pdf`: the same report laid out for print as a client deliverable. It has a cover
(line, inspections, prepared for and by, key figures), a contents page, the executive summary, the pipeline summary,
the repair conditions (immediate, 60-day, 180-day) on landscape pages, growth and remaining life with charts, the
anomaly distribution, data quality, and appendices: methods and settings (PLACEHOLDER rows highlighted), input-file
fingerprints, and the anomalies scheduled by remaining life. Every page carries the DRAFT label, a page number and a
light DRAFT watermark. It renders from the same section tables as `report.html`, so the two always agree; the charts
are drawn with matplotlib and the text uses DejaVu Sans, which ships with matplotlib. Name the client and author with
`--prepared-for` and `--prepared-by` (or the Run setup fields in the app).

### Adding a vendor

Copy a file in `config/vendor_maps/` and edit:
- `columns` maps each standard field (spec §5.1) to the vendor's header; unmapped fields are left blank
- `units`, `clock_format` (`hhmm` | `degrees` | `decimal_hours` | `day_fraction`, an Excel time), and `file`
  (`csv`, `excel` for .xlsx or `xls` for the older format; sheet; 1-based header row; `skip_rows` after the header,
  e.g. a units row)
- the vendor's vocabulary for feature types, surfaces, and true/false flags. Matching ignores case and surrounding spaces. Any value not listed becomes a validation issue rather than being guessed.
- optional: `negate` (numeric fields written with the opposite sign), `fill_down` (fields written only on some rows,
  such as the joint number on the weld rows, carried down to the rows below), and `feature_id_from_row: true` (no id
  column: each feature is named `row-<n>` after its row in the vendor file)

`rosen_tally_xlsx.yaml` and `rosen_tally_xls.yaml` read ROSEN pipe tallies (the current .xlsx layout and the older
.xls one). Real tallies go in `data/raw/` and their outputs in their own folder (e.g. `--out outputs/l930`); both
are git-ignored.

### Synthetic generator outputs (`data/synthetic/`)

| File | Contents |
|---|---|
| `tally_prev_vendor_a.csv` | Previous run, "vendor A" format: feet, inches, hh:mm clock |
| `tally_curr_vendor_b.xlsx` | Current run, "vendor B" format: metres, millimetres, degrees, two title rows |
| `ground_truth.csv` | Every true feature, with its true depths/growth and its ID in each run |
| `ground_truth_welds.csv` | Every true girth weld, with its ID in each run (or missed/not present) |
| `reference/*.csv` | Pipe properties, MOP, HCA segments, repairs, crossings, next inspection, digs |
| `synth_manifest.json` | Parameters, seed, run dates, drift, and CSV hashes |

Synthetic data is gitignored; regenerate it from the seed.

## Parity check

Compares a finished comparison run with the consultant's workbook, anomaly by anomaly (milestone M12). Reads only; it changes nothing in the run.

```powershell
.\.venv\Scripts\python.exe -m ili.parity `
    --comparison outputs\<prev>_vs_<curr> --consultant <consultant_workbook>.xlsx
```

Outputs, under `<comparison folder>/parity/`:

- `parity_by_anomaly.csv`: one row per anomaly in either set, both sides' depth, rate, ERF and remaining life, the agreement flags and one explanation code.
- `parity_summary.json`: counts by explanation, by side and per-quantity agreement counts, and the tolerances used.
- `parity_imp_class.csv`: the consultant's IM summary beside ours, per class (classified, previously repaired, unrepaired). Skipped with a note when `imp_class.csv` is absent.

Explanation codes, first that applies:

- `vendor_cluster_box`: theirs only, and the vendor type is one of the map's `box_types`.
- `not_in_our_run`: any other row only the consultant has.
- `removed_after_run`: ours only, and classified `removed`.
- `not_in_their_set`: any other row only we have.
- `agree`: depth, rate, ERF and remaining life all agree within the `parity:` tolerances.
- `depth_differs`: the depths disagree.
- `match_differs`: one side matched the anomaly to the previous run and the other did not.
- `joint_pool_differs`: both used the joint percentile and the rates disagree.
- `age_basis`: both used the half-life rate and the rates disagree.
- `rl_horizon`: the rates agree, the remaining lives disagree and ours is `beyond_horizon`.
- `unexplained`: none of the above. The command still exits 0 but prints the count; investigate every one.
