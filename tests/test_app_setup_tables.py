from datetime import date

import pandas as pd
import pytest

from ili.app_setup_tables import ENTERED_DIR, TABLES, blank_table, clean_table, seed_table, write_table
from ili.remaining_life import _planned_dates


def test_every_table_has_its_spec_file_name_and_required_columns():
    assert {s.file_name for s in TABLES.values()} == {
        "pipe_properties.csv", "mop_profile.csv", "hca_segments.csv", "next_inspection.csv",
        "repairs_between_runs.csv", "foreign_crossings.csv"}
    for spec in TABLES.values():
        assert set(spec.required) <= set(spec.columns)


def test_blank_rows_are_dropped_and_a_missing_value_names_the_row():
    df = pd.DataFrame({"start_ft": [0.0, None, 100.0], "end_ft": [100.0, None, None], "mop_psig": [1440, None, 1200]})
    with pytest.raises(ValueError, match=r"MOP profile: row 3 needs end_ft"):
        clean_table("mop", df)
    df.loc[2, "end_ft"] = 200.0
    assert clean_table("mop", df).start_ft.tolist() == [0.0, 100.0]


def test_repair_type_outside_the_vocabulary_is_refused():
    df = pd.DataFrame({"start_ft": [1.0], "end_ft": [2.0], "repair_type": ["patch"]})
    with pytest.raises(ValueError, match="repair_type must be one of sleeve, cutout, recoat, pipe_replacement"):
        clean_table("repairs", df)


def test_unticked_flag_alone_is_a_blank_row():
    df = pd.DataFrame({"start_ft": [None], "end_ft": [None], "segment_id": [""], "could_affect_hca": [False]})
    assert clean_table("segments", df).empty


def test_write_table_saves_the_spec_csv(tmp_path):
    df = pd.DataFrame({"segment_id": ["SEG-01", "SEG-02"], "planned_next_assessment_date": [date(2031, 6, 15), None]})
    path = write_table("next_inspection", df, tmp_path)
    assert path == tmp_path / ENTERED_DIR / "next_inspection.csv"
    assert path.read_bytes() == b"segment_id,planned_next_assessment_date\nSEG-01,2031-06-15\nSEG-02,\n"
    assert _planned_dates(pd.read_csv(path, dtype={"segment_id": str})) == {"SEG-01": date(2031, 6, 15)}


def test_write_table_keeps_years_whole_and_refuses_no_rows(tmp_path):
    df = pd.DataFrame({"start_ft": [0.0], "end_ft": [10.0], "od_in": [12.75], "wt_nom_in": [0.25], "grade": ["X52"],
                       "smys_psi": [52000.0], "install_year": [1975.0]})
    text = write_table("pipe", df, tmp_path).read_text(encoding="utf-8")
    assert ",1975," in text and "1975.0" not in text
    with pytest.raises(ValueError, match="Pipe properties: type at least one row"):
        write_table("pipe", blank_table("pipe"), tmp_path)


def test_seed_table_reads_a_file_and_falls_back_to_blank(tmp_path):
    path = tmp_path / "hca.csv"
    path.write_text("start_ft,end_ft,segment_id,could_affect_hca\n0,100,SEG-01,True\n", encoding="utf-8")
    seeded = seed_table("segments", str(path))
    assert seeded.segment_id.tolist() == ["SEG-01"] and seeded.could_affect_hca.tolist() == [True]
    assert seed_table("segments", str(tmp_path / "missing.csv")).empty
    assert seed_table("segments", "").empty


def test_every_column_has_an_editor_heading():
    from ili.app_setup_tables import COLUMN_LABELS
    assert {c for spec in TABLES.values() for c in spec.columns} <= set(COLUMN_LABELS)
