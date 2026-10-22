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
| M2–M10 | not started |

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
```

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
