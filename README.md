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

# Growth rates -> outputs/<prev>_vs_<curr>/growth.csv, growth_segments.csv, growth_summary.json
# (--truth scores conservatism into outputs/validation_metrics.json under growth)
.\.venv\Scripts\python.exe -m ili.growth `
    --prev outputs\standardized\2021_vendorA.csv --curr outputs\standardized\2026_vendorB.csv `
    --matches outputs\2021_vendorA_vs_2026_vendorB\match_table.csv `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --segments data\synthetic\reference\hca_segments.csv `
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

# Report -> outputs/<prev>_vs_<curr>/report.html, dig_list.xlsx, category_changes.csv, run_manifest.json
# (--digs is optional field data for the back-test; --line-name, --prev-tool and --curr-tool fill the header)
.\.venv\Scripts\python.exe -m ili.report `
    --prev outputs\standardized\2021_vendorA.csv `
    --curr outputs\standardized\2026_vendorB.csv `
    --comparison outputs\2021_vendorA_vs_2026_vendorB `
    --pipe data\synthetic\reference\pipe_properties.csv `
    --mop data\synthetic\reference\mop_profile.csv `
    --segments data\synthetic\reference\hca_segments.csv `
    --crossings data\synthetic\reference\foreign_crossings.csv `
    --digs data\synthetic\reference\digs.csv
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

Reviewers' decisions are saved in the comparison folder: match decisions in `reviews.csv` and manual flags in `feature_flags.csv`. Both are append-only logs (a correction is a new row; the latest wins). `ili.match` reads `reviews.csv` and `ili.classify` reads `feature_flags.csv` from the comparison folder by default (`--reviews` and `--flags` name other files on those two commands only; the app and `ili.pipeline` always use the comparison folder's logs), so a decision takes effect the next time the pipeline is rerun, from the app's Rerun button or the command above. **The logs live beside the outputs: deleting an output folder deletes the decisions.** Keep a copy of `reviews.csv` and `feature_flags.csv` before clearing outputs.

The run date is the third `--run` value. It can be left out when the vendor map says where the file states it (vendor B's title row). The run id is `<run year>_<run_label from the vendor map>`.

### Adding a vendor

Copy a file in `config/vendor_maps/` and edit:
- `columns` maps each standard field (spec §5.1) to the vendor's header; unmapped fields are left blank
- `units`, `clock_format` (`hhmm` | `degrees` | `decimal_hours`), and `file` (csv/excel, sheet, 1-based header row)
- the vendor's vocabulary for feature types, surfaces, and true/false flags. Matching ignores case and surrounding spaces. Any value not listed becomes a validation issue rather than being guessed.

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
