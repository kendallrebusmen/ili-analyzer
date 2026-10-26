"""Flag columns read back by the M4-M6 table loaders: true/false in any case, blank unknown, else an error.

Excel re-saves a pandas ``True``/``False`` as ``TRUE``/``FALSE``. Before the fix those read back as unknown
(None) in the match table, silently turning a vendor flag into a not-evaluable rule.
"""

from __future__ import annotations

import pytest

from ili.growth import load_growth_table
from ili.match import load_match_table
from ili.remaining_life import load_remaining_life_table

LOADERS = [
    (load_match_table, "match_table.csv", "match_id,needs_review,curr_dent_metal_loss,curr_seam_interaction",
     "curr_dent_metal_loss"),
    (load_growth_table, "growth.csv", "match_id,needs_review,implausible,match_needs_review", "implausible"),
    (load_remaining_life_table, "remaining_life.csv",
     "match_id,needs_review,dig_required_by_remaining_life,exceeds_80pct", "dig_required_by_remaining_life"),
]


def _write(tmp_path, name, header, column, values):
    cols = header.split(",")
    lines = [header]
    for k, v in enumerate(values):
        row = {c: "False" for c in cols}
        row["match_id"] = f"M{k:05d}"
        row[column] = v
        lines.append(",".join(row[c] for c in cols))
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("loader,name,header,column", LOADERS)
def test_flags_any_case_and_blank(tmp_path, loader, name, header, column):
    values = ["True", "False", "TRUE", "FALSE", "true", "false", " True ", "", "  "]
    t = loader(_write(tmp_path, name, header, column, values))
    assert list(t[column]) == [True, False, True, False, True, False, True, None, None]
    assert all(v is None or type(v) is bool for v in t[column])


@pytest.mark.parametrize("loader,name,header,column", LOADERS)
@pytest.mark.parametrize("text", ["Y", "1", "yes", "nan", "Tru"])
def test_other_flag_text_is_an_error_naming_the_column(tmp_path, loader, name, header, column, text):
    path = _write(tmp_path, name, header, column, ["True", text])
    with pytest.raises(ValueError, match=f"column {column} has '{text}'"):
        loader(path)
