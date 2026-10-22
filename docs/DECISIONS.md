# Decisions

Interpretations of `docs/SPEC.md` and choices the spec leaves open. Newest at the bottom of each milestone.

## M1: skeleton, config, schema, synthetic generator (2026-10-22)

### Project

- **Python 3.13** venv. 3.14 is the machine default, but 3.13 has the most certain wheel coverage for scipy/streamlit.
- **`requirements.txt` pins the ten top-level packages exactly** (`==`), not the full transitive freeze.
- **Synthetic data is gitignored.** It regenerates from the seed recorded in `synth_manifest.json`.
- **Synthetic reference files live in `data/synthetic/reference/`.** That keeps `data/reference/` entirely for real files, all gitignored.
- **Modules added beyond spec §4:** `units.py` (unit conversions, clock formats, tolerance→σ) and `config.py` (YAML loading and validation). Both are shared by several §4 modules.
