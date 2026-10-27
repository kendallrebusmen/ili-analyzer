"""Review decisions re-applied inside anomaly matching (spec 6.4, 6.11; M9 Review Focus 1-3)."""

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ili import match
from ili.align_welds import align_welds
from ili.config import load_settings
from ili.match import (ANOMALY_TYPES, HINTS, REVIEW_STATUSES, PositionMap, _with_od, delta_t_years, envelope,
                       load_match_table, match_anomalies, pair_metrics, place)
from ili.review import REVIEW_COLUMNS, REVIEWS_FILE, append_review, effective_reviews, load_reviews
from ili.synth import SynthParams, generate
from ili.validation import score_matching
from tests.test_match_cli import _prepare
from tests.test_match_table import _runs

S = load_settings()
T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
STATUS_COLUMNS = ["decision_id", "decision", "applied_status", "reason"]
TABLE_CHECKS = ("n_unreported", "n_duplicated", "n_wrong_side", "n_duplicate_links")

# sha256 of match_table.csv (LF) from the M4 matcher (commit af4819f), before decisions existed. If the
# synthetic generator or the automated matching is changed on purpose, regenerate both hashes.
M4_SYNTH_SHA256 = "441aeb62753148d9a301ca3b3c37e7369437bf3c21449eb774c230ba58463f6b"
M4_HAND_SHA256 = "e9379ca1de621502c9601ad09f6fd860f6d49552747b2c2c89716f1d2309a9e0"


def _reviews(*rows):
    """Effective decisions from (decision, prev ids, curr ids) rows, one minute apart in this order."""
    log = pd.DataFrame([{
        "decision_id": f"R{k + 1:06d}", "decision": d, "prev_feature_ids": ";".join(sorted(p)),
        "curr_feature_ids": ";".join(sorted(c)), "reviewer": "Pat",
        "decided_utc": (T0 + timedelta(minutes=k)).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": ""}
        for k, (d, p, c) in enumerate(rows)], columns=REVIEW_COLUMNS, dtype=object)
    return effective_reviews(log)


def _sha(table: pd.DataFrame) -> str:
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _ids(cell) -> list:
    return [] if cell is None else cell.split(";")


def _matched_ids(t: pd.DataFrame, side: str) -> set:
    m = t[t.match_type.isin(["matched_1to1", "matched_cluster"])]
    return {i for cell in m[f"{side}_feature_id"] for i in _ids(cell)}


def _row(t: pd.DataFrame, side: str, fid: str) -> pd.Series:
    hit = t[t[f"{side}_feature_id"].map(lambda c: fid in _ids(c))]
    assert len(hit) == 1, f"{side} {fid} is in {len(hit)} rows"
    return hit.iloc[0]


def _review_fields_only(got: pd.DataFrame, base: pd.DataFrame, rows) -> None:
    """``got`` equals ``base`` except review_status / needs_review on ``rows`` (match ids)."""
    pd.testing.assert_frame_equal(got.drop(columns=["review_status", "needs_review"]),
                                  base.drop(columns=["review_status", "needs_review"]))
    other = ~got.match_id.isin(rows)
    pd.testing.assert_frame_equal(got[other], base[other])


def _status(res) -> dict:
    return {r.decision_id: (r.applied_status, r.reason) for r in res.decision_status.itertuples()}


# ---- fixtures ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_run():
    r = generate(SynthParams(line_miles=5))
    pipe, repairs = r.reference["pipe_properties"], r.reference["repairs_between_runs"]
    wm = align_welds(r.prev_std, r.curr_std, S, repairs).weld_map

    def run(reviews=None):
        return match_anomalies(r.prev_std, r.curr_std, wm, S, pipe, repairs, reviews=reviews)
    return r, run, run()


def _hand(reviews=None, extra=True):
    """The M4 hand-built runs, plus a prev dent Pdent and a curr dent Cdent at 600 ft (they match 1:1)."""
    prev, curr, wm, pipe, repairs = _runs()
    if extra:
        prev = pd.concat([prev, prev[prev.feature_id == "P0"].assign(feature_id="Pdent", feature_type="dent",
                                                                         log_dist_ft=600.0)], ignore_index=True)
        curr = pd.concat([curr, curr[curr.feature_id == "C0"].assign(feature_id="Cdent", feature_type="dent",
                                                                         log_dist_ft=600.0)], ignore_index=True)
    return match_anomalies(prev, curr, wm, S, pipe, repairs, reviews=reviews), (prev, curr, wm, pipe)


# ---- tests -------------------------------------------------------------------------------------

def test_constants():
    assert HINTS == ("in_repaired_section", "near_threshold", "unexplained", "rejected_by_review")
    assert REVIEW_STATUSES == ("pending", "accepted", "rejected", "manual")


def test_no_reviews_is_unchanged(synth_run):
    _, run, base = synth_run
    assert _sha(base.match_table) == M4_SYNTH_SHA256
    assert _sha(_hand(extra=False)[0].match_table) == M4_HAND_SHA256
    empty = run(_reviews())
    pd.testing.assert_frame_equal(empty.match_table, base.match_table)
    for res in (base, empty):
        assert list(res.decision_status.columns) == STATUS_COLUMNS and res.decision_status.empty
        assert res.summary["decisions"] == {"applied": 0, "not_applied": 0, "superseded": 0}
        assert res.summary["counts_by_review_status"] == {"pending": len(base.match_table), "accepted": 0,
                                                          "rejected": 0, "manual": 0}
        assert res.summary["counts_by_hint"]["rejected_by_review"] == 0


def test_accept_pins_pair_and_clears_review(synth_run):
    # hand-built: P0/C0 is a medium-confidence 1:1 match
    base, _ = _hand()
    got, _ = _hand(_reviews(("accept", ["P0"], ["C0"])))
    t, b = got.match_table, base.match_table
    row = _row(t, "prev", "P0")
    assert (row.match_type, row.curr_feature_id, row.review_status) == ("matched_1to1", "C0", "accepted")
    assert row.confidence == "medium" and bool(row.needs_review) is False
    _review_fields_only(t, b, [row.match_id])
    assert _status(got) == {"R000001": ("applied", None)}
    assert got.summary["decisions"] == {"applied": 1, "not_applied": 0, "superseded": 0}
    assert got.summary["counts_by_review_status"]["accepted"] == 1

    # synth: accepting every automated match except the absorbed clusters reproduces each row exactly;
    # only the review fields change (the low-confidence ones leave the queue)
    _, run, res = synth_run
    b = res.match_table
    keep = b.match_type.isin(["matched_1to1", "matched_cluster"]) & ~b.absorbed
    assert (b.needs_review & keep).sum() > 0
    got = run(_reviews(*(("accept", _ids(p), _ids(c)) for p, c in zip(b.prev_feature_id[keep],
                                                                       b.curr_feature_id[keep]))))
    t = got.match_table
    _review_fields_only(t, b, b.match_id[keep])
    assert (t.review_status[keep] == "accepted").all() and not t.needs_review[keep].any()
    assert (t.review_status[~keep] == "pending").all() and (t.needs_review[~keep] == b.needs_review[~keep]).all()
    assert got.summary["decisions"]["applied"] == int(keep.sum())
    assert got.summary["n_needs_review"] == int(t.needs_review.sum())


def test_reject_splits_and_locks():
    base, _ = _hand()
    assert _row(base.match_table, "prev", "P0").curr_feature_id == "C0"  # each other's only candidate
    got, _ = _hand(_reviews(("reject", ["P0"], ["C0"]), ("reject", ["P1", "P2"], ["C1"])))
    t = got.match_table
    for side, fid, mtype in (("prev", "P0", "missing"), ("curr", "C0", "new"), ("prev", "P1", "missing"),
                             ("prev", "P2", "missing"), ("curr", "C1", "new")):
        row = _row(t, side, fid)
        assert row[f"{side}_feature_id"] == fid  # split: one row per feature
        assert (row.match_type, row.review_status, bool(row.needs_review)) == (mtype, "rejected", False)
        assert row.missing_hint == ("rejected_by_review" if side == "prev" else None)
        assert row.confidence is None and np.isnan(row.score)
    assert {"P0", "P1", "P2"}.isdisjoint(_matched_ids(t, "prev"))
    assert {"C0", "C1"}.isdisjoint(_matched_ids(t, "curr"))
    assert got.summary["counts_by_hint"]["rejected_by_review"] == 3
    assert got.summary["counts_by_review_status"]["rejected"] == 5
    assert got.summary["decisions"] == {"applied": 2, "not_applied": 0, "superseded": 0}
    # the rejected rows look like ordinary unmatched rows: same columns and dtypes as before
    assert t.dtypes.to_dict() == base.match_table.dtypes.to_dict()
    # the rest of the table is untouched
    others = ~t.prev_feature_id.isin(["P0", "P1", "P2"]) & ~t.curr_feature_id.isin(["C0", "C1"])
    b = base.match_table
    b_others = ~b.prev_feature_id.isin(["P0", "P2;P1"]) & ~b.curr_feature_id.isin(["C0", "C1"])
    cols = [c for c in t.columns if c != "match_id"]
    pd.testing.assert_frame_equal(t.loc[others, cols].reset_index(drop=True),
                                  b.loc[b_others, cols].reset_index(drop=True))


def test_manual_one_to_one_and_cluster():
    got, (prev, curr, wm, pipe) = _hand(_reviews(("manual", ["Punx"], ["Cnew"]),
                                                 ("manual", ["P0", "P1", "P2"], ["C0", "C1"])))
    t = got.match_table
    one = _row(t, "prev", "Punx")
    assert (one.match_type, one.curr_feature_id, one.review_status) == ("matched_1to1", "Cnew", "manual")
    assert (one.n_prev, one.n_curr) == (1, 1) and not one.needs_review and not one.absorbed
    pmap = PositionMap.from_weld_map(wm)
    pp = _with_od(place(prev, wm, "prev", pmap), pipe, S.alignment.anchor_tol_ft)
    cp = _with_od(place(curr, wm, "curr", pmap), pipe, S.alignment.anchor_tol_ft)
    dt = delta_t_years(prev, curr)
    m = pair_metrics(pp[pp.feature_id == "Punx"], cp[cp.feature_id == "Cnew"], dt, S).iloc[0]
    assert one.axial_diff_in == pytest.approx(2400.0) and one.clock_diff_min == 270
    assert one.score == pytest.approx(m.score) and one.implausible is True and one.confidence == "low"

    cl = _row(t, "prev", "P1")
    assert (cl.match_type, cl.review_status, bool(cl.needs_review)) == ("matched_cluster", "manual", False)
    assert (cl.prev_feature_id, cl.curr_feature_id) == ("P0;P2;P1", "C0;C1")  # log order, as M4 clusters
    assert (cl.n_prev, cl.n_curr) == (3, 2)
    for side, placed, ids in (("prev", pp, ["P0", "P2", "P1"]), ("curr", cp, ["C0", "C1"])):
        env = envelope(placed.set_index("feature_id").loc[ids].reset_index())
        assert cl[f"{side}_log_dist_ft"] == pytest.approx(env[f"pos_{side}_ft"])
        assert cl[f"{side}_length_in"] == pytest.approx(env["length_in"])
        assert cl[f"{side}_width_in"] == pytest.approx(env["width_in"])
        assert cl[f"{side}_depth_pct"] == env["depth_pct"]
        assert cl[f"{side}_clock_min"] == round(env["clock_min"]) % 720
    assert cl.curr_pos_in_prev_ft == pytest.approx(envelope(cp.set_index("feature_id").loc[["C0", "C1"]]
                                                            .reset_index())["pos_prev_ft"])
    assert cl.curr_vendor_cluster_id == "V1"
    assert got.summary["counts_by_review_status"]["manual"] == 2
    assert got.summary["counts_by_match_type"]["matched_cluster"] == 1
    assert {"P0", "P1", "P2", "Punx"} <= _matched_ids(t, "prev")


def test_invalid_decisions_not_applied():
    base, _ = _hand()
    got, _ = _hand(_reviews(
        ("accept", ["PX"], ["CX"]),            # R1: superseded by R7 (shares PX)
        ("manual", ["PV"], ["Cnew"]),          # R2: PV is a valve
        ("manual", ["Punx"], ["Cdent"]),       # R3: metal loss vs dent
        ("manual", ["Pdent", "Pthr"], ["C1"]),  # R4: a cluster holding a dent
        ("manual", ["Pspan"], ["C0"]),         # R5: weld-map rows 2 and 0
        ("reject", ["P0"], ["Cnope"]),         # R6: Cnope is not in the current run
        ("accept", ["PX"], ["CY"]),            # R7: unknown ids
    ))
    assert _status(got) == {
        "R000001": ("superseded", None),
        "R000002": ("not_applied", "not_anomaly"),
        "R000003": ("not_applied", "type_mismatch"),
        "R000004": ("not_applied", "cluster_type"),
        "R000005": ("not_applied", "out_of_range"),
        "R000006": ("not_applied", "unknown_feature"),
        "R000007": ("not_applied", "unknown_feature"),
    }
    assert list(got.decision_status.decision) == ["accept", "manual", "manual", "manual", "manual", "reject",
                                                   "accept"]
    assert got.summary["decisions"] == {"applied": 0, "not_applied": 6, "superseded": 1}
    # nothing applied, nothing blocked: the full automated table, unchanged
    pd.testing.assert_frame_equal(got.match_table, base.match_table)
    assert _row(got.match_table, "prev", "Pdent").curr_feature_id == "Cdent"


def _synth_decisions(t: pd.DataFrame):
    """An accept, two rejects (1:1 and cluster), a manual 1:1 swap and a manual 2:2 cluster, all on
    automated 1:1 metal-loss rows sharing a weld-map row, plus one stale decision."""
    absorbed, needs_review = t.absorbed.astype(bool), t.needs_review.astype(bool)  # also as read back
    one = t[(t.match_type == "matched_1to1") & (t.prev_feature_type == "metal_loss")
            & (t.prev_wm_row == t.curr_wm_row)]
    rows = one.groupby("prev_wm_row").filter(lambda g: len(g) >= 2)
    groups = [g for _, g in rows.groupby("prev_wm_row")][:3]
    (a, b), (c, d), (e, f) = ((g.iloc[0], g.iloc[1]) for g in groups)
    cluster = t[(t.match_type == "matched_cluster") & ~absorbed].iloc[0]
    low = t[(t.match_type == "matched_1to1") & needs_review & ~t.prev_wm_row.isin([g.iloc[0].prev_wm_row
                                                                                    for g in groups])].iloc[0]
    decisions = [
        ("accept", [low.prev_feature_id], [low.curr_feature_id]),
        ("reject", [a.prev_feature_id], [a.curr_feature_id]),
        ("reject", _ids(cluster.prev_feature_id), _ids(cluster.curr_feature_id)),
        ("manual", [c.prev_feature_id], [d.curr_feature_id]),
        ("manual", [d.prev_feature_id], [c.curr_feature_id]),
        ("manual", [e.prev_feature_id, f.prev_feature_id], [e.curr_feature_id, f.curr_feature_id]),
        ("accept", ["A-999999"], [b.curr_feature_id]),
    ]
    return decisions, (a, c, d, e, f, cluster, low)


def test_every_anomaly_once_with_reviews(synth_run):
    r, run, base = synth_run
    decisions, (a, c, d, e, f, cluster, low) = _synth_decisions(base.match_table)
    res = run(_reviews(*decisions))
    t = res.match_table
    score = score_matching(t, r.ground_truth)
    assert {k: score[k] for k in TABLE_CHECKS} == dict.fromkeys(TABLE_CHECKS, 0)
    for side, std in (("prev", r.prev_std), ("curr", r.curr_std)):
        got = [i for cell in t[f"{side}_feature_id"] for i in _ids(cell)]
        assert sorted(got) == sorted(std[std.feature_type.isin(ANOMALY_TYPES)].feature_id)
    assert res.summary["decisions"] == {"applied": 6, "not_applied": 1, "superseded": 0}
    assert _status(res)["R000007"] == ("not_applied", "unknown_feature")
    assert _row(t, "prev", low.prev_feature_id).review_status == "accepted"
    assert {a.prev_feature_id, *_ids(cluster.prev_feature_id)}.isdisjoint(_matched_ids(t, "prev"))
    assert {a.curr_feature_id, *_ids(cluster.curr_feature_id)}.isdisjoint(_matched_ids(t, "curr"))
    assert _row(t, "prev", c.prev_feature_id).curr_feature_id == d.curr_feature_id
    assert _row(t, "prev", d.prev_feature_id).curr_feature_id == c.curr_feature_id
    two = _row(t, "prev", e.prev_feature_id)
    assert (two.match_type, two.n_prev, two.n_curr, two.review_status) == ("matched_cluster", 2, 2, "manual")
    counts = res.summary["counts_by_review_status"]
    assert counts == {"pending": len(t) - 1 - 2 - (cluster.n_prev + cluster.n_curr) - 3, "accepted": 1,
                      "rejected": 2 + cluster.n_prev + cluster.n_curr, "manual": 3}
    assert not t.needs_review[t.review_status != "pending"].any()
    # Review Focus 2: the decisions change only the rows of the features they name
    named = {s: {i for dec, p, c in decisions[:-1] for i in (p if s == "prev" else c)} for s in ("prev", "curr")}

    def untouched(table):
        hit = [table[f"{s}_feature_id"].map(lambda cell, s=s: not named[s].isdisjoint(_ids(cell)))
               for s in ("prev", "curr")]
        return table[~(hit[0] | hit[1])].drop(columns="match_id").reset_index(drop=True)
    assert len(untouched(t)) == len(base.match_table) - 1 - 1 - 1 - 4  # accept, two rejects, swap, cluster
    pd.testing.assert_frame_equal(untouched(t), untouched(base.match_table))


def _cli(data, out, std, folder, *extra):
    return match.main([
        "--prev", str(std / "2021_vendorA.csv"), "--curr", str(std / "2026_vendorB.csv"),
        "--weld-map", str(folder / "weld_map.csv"),
        "--pipe", str(data / "reference" / "pipe_properties.csv"),
        "--repairs", str(data / "reference" / "repairs_between_runs.csv"),
        "--out", str(out), *extra,
    ])


def test_rerun_is_idempotent(synth, tmp_path, capsys):
    _, data = synth
    out = tmp_path / "out"
    std, folder = _prepare(data, out)
    assert _cli(data, out, std, folder) == 0
    assert (folder / "review_status.csv").read_bytes() == b"decision_id,decision,applied_status,reason\n"
    first = load_match_table(folder / "match_table.csv")
    decisions, _ = _synth_decisions(first)
    log = folder / REVIEWS_FILE
    for k, (d, p, c) in enumerate(decisions):
        append_review(log, d, p, c, "Pat", now=T0 + timedelta(minutes=k))
    capsys.readouterr()

    assert _cli(data, out, std, folder) == 0  # reviews.csv found in the comparison dir
    printed = capsys.readouterr().out
    assert "decisions: applied 6, not_applied 1, superseded 0" in printed
    assert "R000007 accept not applied: unknown_feature" in printed
    table1, status1 = (folder / "match_table.csv").read_bytes(), (folder / "review_status.csv").read_bytes()
    assert b"rejected_by_review" in table1 and b"\r" not in status1
    status = pd.read_csv(folder / "review_status.csv", dtype=str, keep_default_na=False)
    assert list(status.columns) == STATUS_COLUMNS and list(status.decision_id) == list(load_reviews(log).decision_id)

    assert _cli(data, out, std, folder) == 0
    assert (folder / "match_table.csv").read_bytes() == table1
    assert (folder / "review_status.csv").read_bytes() == status1

    elsewhere = tmp_path / "copy.csv"
    elsewhere.write_bytes(log.read_bytes())
    log.unlink()
    assert _cli(data, out, std, folder, "--reviews", str(elsewhere)) == 0
    assert (folder / "match_table.csv").read_bytes() == table1

    elsewhere.write_text(elsewhere.read_text(encoding="utf-8") + "R000008,approve,A,B,Pat,2026-10-27T12:00:00Z,\n",
                         encoding="utf-8")
    capsys.readouterr()
    assert _cli(data, out, std, folder, "--reviews", str(elsewhere)) == 2
    assert "copy.csv: line 9" in capsys.readouterr().err


def test_one_sided_accept_confirms_missing_and_new(synth_run):
    r, run, base = synth_run
    b = base.match_table
    miss = b[(b.match_type == "missing") & (b.missing_hint == "unexplained")]
    assert len(miss), "the synthetic comparison has no unexplained missing row"
    new = b[b.match_type == "new"].iloc[0]
    m = miss.iloc[0]
    assert bool(m.needs_review) is True
    res = run(_reviews(("accept", [m.prev_feature_id], []), ("accept", [], [new.curr_feature_id])))
    t = res.match_table
    got = _row(t, "prev", m.prev_feature_id)
    assert (got.match_type, got.review_status, bool(got.needs_review), got.missing_hint) == (
        "missing", "accepted", False, "unexplained")
    got = _row(t, "curr", new.curr_feature_id)
    assert (got.match_type, got.review_status, bool(got.needs_review)) == ("new", "accepted", False)
    assert m.prev_feature_id not in _matched_ids(t, "prev") and new.curr_feature_id not in _matched_ids(t, "curr")
    assert _status(res) == {"R000001": ("applied", None), "R000002": ("applied", None)}
    assert res.summary["counts_by_review_status"]["accepted"] == 2
    score = score_matching(t, r.ground_truth)
    assert {k: score[k] for k in TABLE_CHECKS} == dict.fromkeys(TABLE_CHECKS, 0)
    for side, std in (("prev", r.prev_std), ("curr", r.curr_std)):
        ids = [i for cell in t[f"{side}_feature_id"] for i in _ids(cell)]
        assert sorted(ids) == sorted(std[std.feature_type.isin(ANOMALY_TYPES)].feature_id)
    # only the two rows' review fields change
    _review_fields_only(t, b, [m.match_id, new.match_id])


def test_one_sided_accept_unknown_feature_is_not_applied(synth_run):
    _, run, _ = synth_run
    res = run(_reviews(("accept", ["A-999999"], [])))
    assert _status(res) == {"R000001": ("not_applied", "unknown_feature")}


def _split_rows(wm: pd.DataFrame, row: int, cuts_prev: list) -> pd.DataFrame:
    """``wm`` with matched row ``row`` cut into short joints at previous-run positions ``cuts_prev`` (pup
    joints). The cuts lie on the row's own linear map, so no position in either run moves."""
    r = wm.iloc[row]
    ps, pl, cs, cl = (float(r[c]) for c in ("prev_start_ft", "prev_length_ft", "curr_start_ft", "curr_length_ft"))
    edges = [0.0, *[(x - ps) / pl for x in cuts_prev], 1.0]
    pieces = [r.copy() for _ in edges[1:]]
    for piece, a, b in zip(pieces, edges, edges[1:]):
        piece["prev_start_ft"], piece["prev_length_ft"] = ps + a * pl, (b - a) * pl
        piece["curr_start_ft"], piece["curr_length_ft"] = cs + a * cl, (b - a) * cl
        piece["curr_length_scaled_ft"] = (b - a) * pl
    return pd.concat([wm.iloc[:row], pd.DataFrame(pieces), wm.iloc[row + 1:]], ignore_index=True)


def test_accepting_the_matchers_own_row_across_pup_joints_applies(synth_run):
    """M8: an accept naming exactly a row the automated passes make is applied even when its members span
    more than one weld-map row (short pup joints); any other out-of-range accept is still not applied."""
    r, _, base = synth_run
    pipe, repairs = r.reference["pipe_properties"], r.reference["repairs_between_runs"]
    wm = align_welds(r.prev_std, r.curr_std, S, repairs).weld_map
    pmap = PositionMap.from_weld_map(wm)
    prev, curr = (place(std, wm, run, pmap).set_index("feature_id") for std, run in
                  ((r.prev_std, "prev"), (r.curr_std, "curr")))
    t = base.match_table
    for row in t[t.match_type == "matched_cluster"].itertuples():
        members = [prev.loc[_ids(row.prev_feature_id)], curr.loc[_ids(row.curr_feature_id)]]
        pos = np.unique(np.concatenate([m.pos_prev_ft.to_numpy(float) for m in members]).round(3))
        rows = {int(w) for m in members for w in m.wm_row}
        if len(pos) >= 3 and len(rows) == 1:
            break
    else:
        pytest.fail("the synthetic data has no cluster with three distinct member positions in one joint")
    split = _split_rows(wm, rows.pop(), list((pos[1:] + pos[:-1]) / 2))

    def run(reviews=None):
        return match_anomalies(r.prev_std, r.curr_std, split, S, pipe, repairs, reviews=reviews)
    auto = run().match_table
    same = auto[(auto.prev_feature_id == row.prev_feature_id) & (auto.curr_feature_id == row.curr_feature_id)]
    assert len(same) == 1, "the split weld map changed the automated cluster"
    smap = PositionMap.from_weld_map(split)
    spans = [place(std, split, run, smap).set_index("feature_id").loc[_ids(cell)].wm_row.to_numpy(int)
             for std, run, cell in ((r.prev_std, "prev", row.prev_feature_id), (r.curr_std, "curr", row.curr_feature_id))]
    assert np.ptp(np.concatenate(spans)) > 1  # the members span more than two pup joints

    res = run(_reviews(("accept", _ids(row.prev_feature_id), _ids(row.curr_feature_id))))
    assert _status(res) == {"R000001": ("applied", None)}
    got = _row(res.match_table, "prev", _ids(row.prev_feature_id)[0])
    assert (got.curr_feature_id, got.review_status, got.needs_review) == (row.curr_feature_id, "accepted", False)
    assert res.summary["decisions"] == {"applied": 1, "not_applied": 0, "superseded": 0}
    # the same members minus one form no automated row: the weld-map range rule still applies
    fewer = run(_reviews(("accept", _ids(row.prev_feature_id)[:-1] or _ids(row.prev_feature_id),
                          _ids(row.curr_feature_id))))
    if len(_ids(row.prev_feature_id)) > 1:
        assert _status(fewer)["R000001"][0] == "not_applied"
