import hashlib
import json
from datetime import datetime, timezone

import openpyxl
import pandas as pd
import pytest

from ili import __version__, classify, config, remaining_life
from ili.align_welds import DRAFT_LABEL
from ili.report import code_version, main, write_dig_list
from ili.report_charts import CATEGORY_COLOURS
from ili.report_data import DIG_LIST_COLUMNS
from test_classify_cli import _classify_args
from test_rl_cli import _args as rl_args
from test_rl_cli import _prepare


@pytest.fixture(scope="module")
def built(synth, tmp_path_factory):
    _, data = synth
    out = tmp_path_factory.mktemp("report_cli") / "out"
    std, folder = _prepare(data, out)
    assert remaining_life.main(rl_args(std, folder, data, out)) == 0
    assert classify.main(_classify_args(std, folder, data, out)) == 0
    return data, out, std, folder


def _argv(built, *extra, mop="mop_profile.csv"):
    data, out, std, folder = built
    ref = data / "reference"
    return ["--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
            "--comparison", str(folder), "--pipe", str(ref / "pipe_properties.csv"), "--mop", str(ref / mop),
            "--segments", str(ref / "hca_segments.csv"), "--crossings", str(ref / "foreign_crossings.csv"),
            "--digs", str(ref / "digs.csv"), *extra]


def test_cli_writes_outputs(built, capsys):
    _, _, _, folder = built
    capsys.readouterr()
    assert main(_argv(built, "--line-name", "Line 7")) == 0
    printed = capsys.readouterr().out
    names = ["category_changes.csv", "report.html", "dig_list.xlsx", "run_manifest.json"]
    assert all((folder / n).is_file() for n in names)
    assert DRAFT_LABEL in printed and "pending" in printed and "dig list" in printed

    cls = pd.read_csv(folder / "classification.csv")
    dig = cls[cls.governing_category != "evaluate"]
    ws = openpyxl.load_workbook(folder / "dig_list.xlsx").active
    assert ws.title == config.load_settings().report.dig_list_sheet
    assert "DRAFT" in ws["A1"].value
    assert ws.freeze_panes == "A3"
    assert [c.value for c in ws[2]] == list(DIG_LIST_COLUMNS)
    assert ws.max_row - 2 == len(dig) > 0
    assert ws.auto_filter.ref == f"A2:{openpyxl.utils.get_column_letter(len(DIG_LIST_COLUMNS))}{ws.max_row}"
    category = ws.cell(3, 1).value
    assert ws.cell(3, 1).fill.fgColor.rgb[-6:] == CATEGORY_COLOURS[category].lstrip("#").upper()

    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert DRAFT_LABEL == manifest["label"] and manifest["generator"] == f"ili.report {__version__}"
    assert manifest["inputs"]
    for path, digest in manifest["inputs"].items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
    assert set(names) <= set(manifest["outputs"])
    assert set(manifest["code_version"]) == {"package", "git_commit", "git_dirty"}
    assert manifest["config"]["settings"]["sha256"] and manifest["config"]["criteria"]["sha256"]
    assert {r["run_id"] for r in manifest["tallies"]} == {"2021_vendorA", "2026_vendorB"}
    assert "Line 7" in (folder / "report.html").read_text(encoding="utf-8")


def test_empty_dig_list_workbook(tmp_path):
    path = tmp_path / "dig.xlsx"
    write_dig_list(pd.DataFrame(columns=list(DIG_LIST_COLUMNS)), path, config.load_settings(), DRAFT_LABEL)
    ws = openpyxl.load_workbook(path).active
    assert [c.value for c in ws[2]] == list(DIG_LIST_COLUMNS)
    assert ws.max_row == 2 and ws.freeze_panes == "A3"
    assert ws.auto_filter.ref == f"A2:{openpyxl.utils.get_column_letter(len(DIG_LIST_COLUMNS))}2"


def test_dig_list_dates_and_blanks(tmp_path):
    row = {c: None for c in DIG_LIST_COLUMNS}
    row.update(governing_category="60_day", match_id="M1", due_date="2027-01-02", p_fail=float("nan"))
    path = tmp_path / "dig.xlsx"
    write_dig_list(pd.DataFrame([row]), path, config.load_settings(), DRAFT_LABEL)
    ws = openpyxl.load_workbook(path).active
    col = {c.value: c.column for c in ws[2]}
    assert isinstance(ws.cell(3, col["due_date"]).value, datetime)
    assert ws.cell(3, col["p_fail"]).value is None and ws.cell(3, col["rl_yr"]).value is None


def test_code_version_shape():
    v = code_version()
    assert v["package"] == __version__
    assert v["git_commit"] is None or len(v["git_commit"]) == 40
    assert v["git_dirty"] is None or isinstance(v["git_dirty"], bool)


def test_cli_missing_input_exit_2(built, capsys):
    capsys.readouterr()
    assert main(_argv(built, mop="nope.csv")) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_stale_criteria_exit_2(built, tmp_path, capsys):
    stale = tmp_path / "criteria.yaml"
    stale.write_text(config.DEFAULT_CRITERIA.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    capsys.readouterr()
    assert main(_argv(built, "--criteria", str(stale))) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "changed since" in err
