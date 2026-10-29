"""Readable table columns: labels, number formats, and coverage of the tables they are wired into."""

import pandas as pd

from ili.app_columns import LABELS, column_config, field_label
from ili.app_pages_results import explorer_rows, flag_table
from ili.app_pages_review import DECISION_COLUMNS, ROW_COLUMNS
from ili.match import SIDE_FIELDS
from ili.report_data import DIG_LIST_COLUMNS
from test_app_results import _data


def test_labels_cover_explorer_columns():
    rows = explorer_rows(_data(), [], [], [])
    assert set(rows.columns) <= set(LABELS)


def test_labels_cover_the_other_tables():
    flag_columns = flag_table(type("D", (), {"flags": pd.DataFrame()})()).columns
    wanted = {*DIG_LIST_COLUMNS, *ROW_COLUMNS, *DECISION_COLUMNS, *SIDE_FIELDS, *flag_columns, "feature_id",
              "section_id", "prev_start_ft", "prev_end_ft", "curr_start_ft", "curr_end_ft", "match_types",
              "n_prev_joints", "n_curr_joints", "overlaps_repair", "applied_status", "reason"}
    assert wanted <= set(LABELS)


def test_column_config_types():
    df = pd.DataFrame({"rl_yr": [5.0, 6.0], "governing_method": ["a", "b"], "requires_review": [True, False]})
    cfg = column_config(df)
    assert cfg["rl_yr"]["label"] == "Remaining life (yr)"
    assert cfg["rl_yr"]["type_config"]["type"] == "number"
    assert cfg["rl_yr"]["type_config"]["format"] == "%.1f"
    assert cfg["governing_method"]["label"] == "Growth method"
    assert cfg["governing_method"]["type_config"]["type"] == "text"
    assert cfg["requires_review"]["type_config"]["type"] == "checkbox"


def test_numeric_label_on_text_data_stays_text():
    cfg = column_config(pd.DataFrame({"rl_yr": pd.Series(["n/a"], dtype=object)}))
    assert cfg["rl_yr"]["type_config"]["type"] == "text"


def test_unknown_column_left_raw():
    assert column_config(pd.DataFrame({"mystery": [1], "rl_yr": [1.0]})).keys() == {"rl_yr"}


def test_field_label_fallback():
    assert field_label("foo_bar") == "foo bar"
    assert field_label("depth_pct") == "Depth (% WT)"


def test_labels_are_plain_words():
    assert all("_" not in label for label, _ in LABELS.values())
