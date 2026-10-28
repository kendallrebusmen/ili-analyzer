"""Run every stage in order: ingest, align_welds, match, growth, remaining_life, classify, report.

The same commands as the README, called in process. Review decisions and manual flags in the comparison
folder are re-applied by match and classify on every run (their default log paths), so a rerun after a
review reflects it. DRAFT output only.
"""

import argparse
import contextlib
import io
import json
import sys
import threading
import traceback
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from ili import align_welds, classify, growth, ingest, match, remaining_life, report
from ili.align_welds import DRAFT_LABEL, comparison_dir
from ili.config import DEFAULT_SETTINGS

STAGES = ("ingest", "align_welds", "match", "growth", "remaining_life", "classify", "report")
INPUTS_FILE = "pipeline_inputs.json"
_PATH_FIELDS = ("prev_tally", "prev_map", "curr_tally", "curr_map", "pipe", "mop", "segments", "next_inspection",
                "repairs", "crossings", "digs", "settings", "criteria", "out")
_REQUIRED = ("prev_tally", "prev_map", "curr_tally", "curr_map", "pipe", "mop", "segments", "next_inspection",
             "out")


@dataclass(frozen=True)
class PipelineInputs:
    prev_tally: Path
    prev_map: Path
    curr_tally: Path
    curr_map: Path
    pipe: Path
    mop: Path
    segments: Path
    next_inspection: Path
    out: Path
    settings: Path = DEFAULT_SETTINGS
    prev_run_date: str | None = None
    curr_run_date: str | None = None
    repairs: Path | None = None
    crossings: Path | None = None
    digs: Path | None = None
    criteria: Path | None = None
    discovery_date: str | None = None
    line_name: str | None = None
    prev_tool: str | None = None
    curr_tool: str | None = None
    prepared_for: str | None = None
    prepared_by: str | None = None


@dataclass(frozen=True)
class StageResult:
    stage: str
    exit_code: int
    stdout: str
    stderr: str


def _optional(flag: str, value) -> list[str]:
    return [] if value is None else [flag, str(value)]


def _run_spec(tally: Path, vendor_map: Path, run_date: str | None) -> list[str]:
    return ["--run", str(tally), str(vendor_map), *([run_date] if run_date else [])]


def _argv(stage: str, i: PipelineInputs, prev_id: str | None, curr_id: str | None) -> list[str]:
    out = str(i.out)
    if stage == "ingest":
        return [*_run_spec(i.prev_tally, i.prev_map, i.prev_run_date),
                *_run_spec(i.curr_tally, i.curr_map, i.curr_run_date), "--out", out]
    std = Path(i.out) / "standardized"
    prev, curr = str(std / f"{prev_id}.csv"), str(std / f"{curr_id}.csv")
    folder = comparison_dir(i.out, prev_id, curr_id)
    settings = ["--settings", str(i.settings)]
    if stage == "align_welds":
        return ["--prev", prev, "--curr", curr, *_optional("--repairs", i.repairs), *settings, "--out", out]
    if stage == "match":
        return ["--prev", prev, "--curr", curr, "--weld-map", str(folder / "weld_map.csv"), "--pipe", str(i.pipe),
                *_optional("--repairs", i.repairs), *settings, "--out", out]
    if stage == "growth":
        return ["--prev", prev, "--curr", curr, "--matches", str(folder / "match_table.csv"),
                "--pipe", str(i.pipe), "--segments", str(i.segments), *_optional("--repairs", i.repairs), *settings, "--out", out]
    if stage == "remaining_life":
        return ["--curr", curr, "--growth", str(folder / "growth.csv"), "--pipe", str(i.pipe), "--mop", str(i.mop),
                "--segments", str(i.segments), "--next-inspection", str(i.next_inspection), *settings, "--out", out]
    if stage == "classify":
        return ["--curr", curr, "--matches", str(folder / "match_table.csv"),
                "--remaining-life", str(folder / "remaining_life.csv"), "--pipe", str(i.pipe),
                "--segments", str(i.segments), *_optional("--crossings", i.crossings),
                *_optional("--criteria", i.criteria), *_optional("--discovery-date", i.discovery_date),
                *settings, "--out", out]
    return ["--prev", prev, "--curr", curr, "--comparison", str(folder), "--pipe", str(i.pipe),
            "--mop", str(i.mop), "--segments", str(i.segments), *_optional("--crossings", i.crossings),
            *_optional("--digs", i.digs), *_optional("--repairs", i.repairs), "--ingest-dir", out, *settings, *_optional("--criteria", i.criteria),
            *_optional("--line-name", i.line_name), *_optional("--prev-tool", i.prev_tool),
            *_optional("--curr-tool", i.curr_tool), *_optional("--prepared-for", i.prepared_for),
            *_optional("--prepared-by", i.prepared_by)]


# redirect_stdout swaps process-global streams and concurrent runs would write the same folder, so one run
# at a time per process (Streamlit sessions are threads).
_RUN_LOCK = threading.Lock()
BUSY_MESSAGE = "another pipeline run is in progress; try again when it finishes"

_MODULES = {"ingest": ingest, "align_welds": align_welds, "match": match, "growth": growth,
            "remaining_life": remaining_life, "classify": classify, "report": report}


def _call(stage: str, argv: list[str]) -> StageResult:
    """Run one stage's ``main`` with stdout and stderr captured; an unexpected exception is a failed stage."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = _MODULES[stage].main(argv)
        except SystemExit as exc:  # argparse errors, sys.exit()
            if exc.code is None:
                code = 0
            elif isinstance(exc.code, int):
                code = exc.code
            else:
                code = 1
                err.write(f"{exc.code}\n")
        except Exception:
            code = 1
            err.write(traceback.format_exc())
    return StageResult(stage, int(code or 0), out.getvalue(), err.getvalue())


def _run_ids(out: Path) -> tuple[str, str]:
    runs = json.loads((Path(out) / "ingest_manifest.json").read_text(encoding="utf-8"))["runs"]
    return runs[0]["run_id"], runs[1]["run_id"]


def write_pipeline_inputs(inputs: PipelineInputs, folder: Path) -> Path:
    """Record the inputs that were run, so a later session can rerun them (a pipeline output)."""
    data = {"label": DRAFT_LABEL, "ran_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for f in fields(inputs):
        value = getattr(inputs, f.name)
        data[f.name] = str(value) if value is not None and f.name in _PATH_FIELDS else value
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / INPUTS_FILE
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def load_pipeline_inputs(path: Path) -> PipelineInputs:
    """Read a ``pipeline_inputs.json`` back; ValueError when it is not one."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        kwargs = {f.name: data[f.name] for f in fields(PipelineInputs) if f.name in data}
        kwargs = {k: Path(v) if k in _PATH_FIELDS and v is not None else v for k, v in kwargs.items()}
        return PipelineInputs(**kwargs)
    except (TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: not a pipeline inputs file ({exc})") from None


def run_pipeline(inputs: PipelineInputs, *, until: str | None = None) -> list[StageResult]:
    """Run the stages in order, stopping at the first non-zero exit (or after ``until``). One run at a
    time: while another holds the lock this returns at once with a single failed ``pipeline`` result."""
    if not _RUN_LOCK.acquire(blocking=False):
        return [StageResult("pipeline", 1, "", BUSY_MESSAGE)]
    try:
        return _run(inputs, until)
    finally:
        _RUN_LOCK.release()


def _run(inputs: PipelineInputs, until: str | None) -> list[StageResult]:
    if until is not None and until not in STAGES:
        raise ValueError(f"until {until!r} is not one of {', '.join(STAGES)}")
    results: list[StageResult] = []
    prev_id = curr_id = None
    for stage in STAGES:
        result = _call(stage, _argv(stage, inputs, prev_id, curr_id))
        results.append(result)
        if result.exit_code != 0:
            break
        if stage == "ingest":
            try:
                prev_id, curr_id = _run_ids(inputs.out)
                write_pipeline_inputs(inputs, comparison_dir(inputs.out, prev_id, curr_id))
            except (OSError, ValueError, KeyError, IndexError):
                results[-1] = StageResult(stage, 1, result.stdout, result.stderr + traceback.format_exc())
                break
        if stage == until:
            break
    return results


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ili.pipeline",
                                 description="Run every stage, ingest to report. " + DRAFT_LABEL)
    for f in fields(PipelineInputs):
        ap.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                        type=Path if f.name in _PATH_FIELDS else str, required=f.name in _REQUIRED,
                        default=DEFAULT_SETTINGS if f.name == "settings" else None)
    ap.add_argument("--until", choices=STAGES, default=None, help="stop after this stage")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = vars(_parser().parse_args(argv))
    until = args.pop("until")
    print(DRAFT_LABEL)
    for r in run_pipeline(PipelineInputs(**args), until=until):
        print(f"== {r.stage}: exit {r.exit_code}")
        print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="", file=sys.stderr)
        if r.exit_code:
            return r.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
