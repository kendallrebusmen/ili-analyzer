"""M12 consultant map and workbook reader (synthetic workbooks only)."""
from pathlib import Path

import openpyxl
import pytest
import yaml

from ili.parity import CONSULTANT_COLUMNS, load_consultant_map, read_consultant

MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "consultant_maps" / "integrity_solutions.yaml"
CMAP = load_consultant_map(MAP_PATH)
FIELDS = ["feature_id", "log_dist_ft", "joint_id", "depth_pct", "prev_depth_pct", "rate_mpy", "rate_source",
          "erf", "rl_yr", "vendor_type"]
CLASSES = ["Immediate", "60-day", "180-day", "YTC < 5 yrs", "None"]


def _row(fid, category="Volumetric", source="Pit to Pit Minimum"):
    return {"feature_id": fid, "category": category, "log_dist_ft": 10.0, "joint_id": 5, "depth_pct": 30.0,
            "prev_depth_pct": 20.0, "rate_mpy": 4.0, "rate_source": source, "erf": 0.9, "rl_yr": 12.0,
            "vendor_type": "Corrosion"}


def _workbook(tmp_path, rows, imp_rows=None, drop=None):
    a = CMAP.anomalies
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = a.sheet
    heads = [a.category_column, *[a.columns[f] for f in FIELDS]]
    heads = [h for h in heads if h != drop]
    ws.append(heads)
    for r in rows:
        vals = {a.category_column: r["category"], **{a.columns[f]: r[f] for f in FIELDS}}
        ws.append([vals[h] for h in heads])
    if imp_rows is not None:
        m = CMAP.imp_summary
        ims = wb.create_sheet(m.sheet)
        ims.append([m.columns[f] for f in ("imp_class", "n_classified", "n_previously_repaired", "n_unrepaired")])
        for r in imp_rows:
            ims.append(list(r))
    p = tmp_path / "consultant.xlsx"
    wb.save(p)
    return p


def test_reads_anomalies_only(tmp_path):
    p = _workbook(tmp_path, [_row(143), _row(144.0), _row(9, "Features & Fittings")], imp_rows=[])
    data = read_consultant(p, CMAP)
    assert list(data.anomalies.columns) == list(CONSULTANT_COLUMNS)
    assert list(data.anomalies.feature_id) == ["143", "144"]


def test_rate_sources_mapped_and_unknown_counted(tmp_path):
    rows = [_row(1, source="Pit to Pit Minimum"), _row(2, source="Fixed Growth Rate"),
            _row(3, source="Mystery"), _row(4, source=None)]
    data = read_consultant(_workbook(tmp_path, rows, imp_rows=[]), CMAP)
    assert list(data.anomalies.method) == ["raw", "shrinking_fixed", None, None]
    assert data.unknown_sources == {"Mystery": 1}


def test_missing_column_names_sheet_and_column(tmp_path):
    p = _workbook(tmp_path, [_row(1)], imp_rows=[], drop="ERF")
    with pytest.raises(ValueError, match="Classified_Anomalies.*ERF"):
        read_consultant(p, CMAP)


def test_missing_sheet(tmp_path):
    p = _workbook(tmp_path, [_row(1)])
    with pytest.raises(ValueError, match="IM Summary"):
        read_consultant(p, CMAP)


def test_duplicate_feature_ids(tmp_path):
    p = _workbook(tmp_path, [_row(143), _row(143)], imp_rows=[])
    with pytest.raises(ValueError, match="143"):
        read_consultant(p, CMAP)


def test_imp_summary_read(tmp_path):
    imp = [(c, 10 + i, i, 10) for i, c in enumerate(CLASSES)]
    data = read_consultant(_workbook(tmp_path, [_row(1)], imp_rows=list(reversed(imp))), CMAP)
    assert list(data.imp_summary.columns) == ["imp_class", "n_classified", "n_previously_repaired", "n_unrepaired"]
    assert list(data.imp_summary.imp_class) == CLASSES


def test_no_imp_section_gives_none(tmp_path):
    cmap = CMAP.model_copy(update={"imp_summary": None})
    data = read_consultant(_workbook(tmp_path, [_row(1)]), cmap)
    assert data.imp_summary is None


def test_map_rejects_unknown_method(tmp_path):
    raw = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    raw["anomalies"]["rate_sources"]["Pit to Pit Minimum"] = "bogus"
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_consultant_map(p)


def test_blank_feature_ids_refused(tmp_path):
    rows = [_row(1), _row(None), _row(3), _row(None)]
    with pytest.raises(ValueError, match=r"Classified_Anomalies.*2 anomaly row.*blank feature id.*rows 3, 5"):
        read_consultant(_workbook(tmp_path, rows, imp_rows=[]), CMAP)


def test_header_row_must_be_positive(tmp_path):
    raw = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    raw["anomalies"]["header_row"] = 0
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception, match="header_row"):
        load_consultant_map(p)
    raw["anomalies"]["header_row"] = 1
    raw["imp_summary"]["header_row"] = 0
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception, match="header_row"):
        load_consultant_map(p)
