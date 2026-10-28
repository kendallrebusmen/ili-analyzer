"""Report tables of the consultant's practice (spec 6.8 IMP class summary, 6.10 report). DRAFT.

The builders read tables the earlier stages wrote (classification, match table, remaining life, repairs log)
and recompute nothing a stage wrote: a class, a remaining life or a position is looked up, never derived
again. Counts are plain integers; a value that cannot be known stays None so the writers render an empty cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ili.config import Settings

IMP_CLASS_COLUMNS = ("imp_class", "n_classified", "n_previously_repaired", "n_unrepaired", "n_rl_not_evaluated")
_CATEGORY_CLASSES = {"immediate": "Immediate", "60_day": "60-day", "180_day": "180-day"}  # spec 6.8
NO_CLASS = "None"


def _previously_repaired(position_ft: np.ndarray, repairs: pd.DataFrame) -> np.ndarray:
    """True where a current-run position (ft; NaN is never repaired) lies in any repair span, ends inclusive."""
    missing = [c for c in ("start_ft", "end_ft") if c not in repairs.columns]
    if missing:
        raise ValueError(f"repairs table is missing column(s): {', '.join(missing)}")
    hit = np.zeros(len(position_ft), dtype=bool)
    for start, end in zip(repairs.start_ft.astype(float), repairs.end_ft.astype(float)):
        hit |= (position_ft >= min(start, end)) & (position_ft <= max(start, end))
    return hit


def imp_class_summary(classification: pd.DataFrame, match_table: pd.DataFrame, remaining_life: pd.DataFrame,
                      repairs: pd.DataFrame | None, settings: Settings) -> pd.DataFrame:
    """The IMP class summary (spec 6.8), one row per class: Immediate, 60-day and 180-day by
    ``governing_category``; ``YTC < N yrs`` for any other row whose years to condition (``rl_yr``, else
    ``rl_upper_bound_yr``) is below ``report.ytc_flag_years``; ``None`` for the rest. A row with neither
    remaining life nor bound (a dent or gouge, say) falls in ``None``. ``n_rl_not_evaluated`` counts, per
    class, the rows with neither. A feature is previously repaired when its current-run position
    (``curr_log_dist_ft`` of the match table, by ``match_id``) is inside any repair span; without a repairs
    table ``n_previously_repaired`` and ``n_unrepaired`` are None. Summary view only; categories drive due dates."""
    ytc = settings.report.ytc_flag_years
    ytc_label = f"YTC < {ytc:g} yrs"
    cls = classification.reset_index(drop=True)
    rl = remaining_life.drop_duplicates("match_id").set_index("match_id")
    keys = cls.match_id.to_numpy()
    years = rl.rl_yr.astype(float).reindex(keys).to_numpy()
    bound = rl.rl_upper_bound_yr.astype(float).reindex(keys).to_numpy()
    years = np.where(np.isnan(years), bound, years)  # rl_yr (inf: no failure within the horizon), else the bound
    not_evaluated = np.isnan(years)
    ytc_hit = years < ytc  # NaN and inf are never below
    imp = np.where(cls.governing_category.isin(_CATEGORY_CLASSES), cls.governing_category.map(_CATEGORY_CLASSES),
                   np.where(ytc_hit, ytc_label, NO_CLASS))
    if repairs is None:
        repaired = None
    else:
        position = (match_table.drop_duplicates("match_id").set_index("match_id").curr_log_dist_ft
                    .astype(float).reindex(keys).to_numpy())
        repaired = _previously_repaired(position, repairs)
    rows = []
    for name in (*_CATEGORY_CLASSES.values(), ytc_label, NO_CLASS):
        in_class = imp == name
        n = int(in_class.sum())
        n_repaired = None if repaired is None else int((in_class & repaired).sum())
        rows.append({"imp_class": name, "n_classified": n, "n_previously_repaired": n_repaired,
                     "n_unrepaired": None if n_repaired is None else n - n_repaired,
                     "n_rl_not_evaluated": int((in_class & not_evaluated).sum())})
    out = pd.DataFrame(rows, columns=list(IMP_CLASS_COLUMNS))
    for col in ("n_previously_repaired", "n_unrepaired"):
        out[col] = out[col].astype(object)
    return out
