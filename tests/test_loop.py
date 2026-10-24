"""E2 loop mechanism — standing/fidelity/fade-retire/split/merge (METHODOLOGY-v3).

Doubles as the H13-H16 instantiability dry-run: every mechanism is shown to both
fire and NOT fire (a detector that cannot fail is not a test).
"""
from __future__ import annotations

from brainkit.loop import (
    LoopState, MatchRecord, Verdict, fade_retire, merge, promote, run_stream,
    should_merge, should_split, split, update,
)
from brainkit.model import brain_from_dict

BASE = {
    "id": "00000000-0000-4000-8000-000000000010",
    "title": "t", "status": "experimental", "scale": "campaign",
    "representation": "graph", "match": "likelihood", "tlp": "red",
    "created": "2026-07-10", "modified": "2026-07-10",
    "entities": [{"id": "e1", "type": "process", "pattern": {"name": "a.exe"}},
                 {"id": "e2", "type": "process", "pattern": {"name": "b.exe"}}],
    "links": [{"id": "L1", "type": "activity", "source": "e1", "target": "e2",
               "relation": "spawned", "role": "required", "volatility": "stable",
               "provenance": {"kind": "reported", "source": "t"}}],
}


def a_brain(bid="00000000-0000-4000-8000-000000000010"):
    return brain_from_dict({**BASE, "id": bid})


# --- H13: standing / fidelity --------------------------------------------------

def test_standing_rises_on_confirm_falls_on_contradict():
    s = LoopState("b")
    s = run_stream(s, [Verdict.CONFIRM] * 5, weight=2.0)
    assert s.standing == 10.0 and s.confirmed == 5 and s.total == 5
    s2 = run_stream(LoopState("b"), [Verdict.CONTRADICT] * 3, weight=2.0)
    assert s2.standing == -6.0 and s2.confirmed == 0 and s2.total == 3


def test_fidelity_is_confirmed_over_total():
    s = LoopState("b")
    s = run_stream(s, [Verdict.CONFIRM, Verdict.CONFIRM, Verdict.CONTRADICT], weight=1.0)
    assert s.total == 3 and s.confirmed == 2
    assert s.fidelity == 2 / 3
    assert LoopState("b").fidelity == 0.0  # nothing matched yet


def test_absence_does_not_move_standing_by_default_but_can_in_the_arm():
    # CL-054 default: absence never counts.
    s = run_stream(LoopState("b"), [Verdict.ABSENCE] * 4, weight=2.0)
    assert s.standing == 0.0 and s.total == 0
    # CL-062 arm: a contradiction must outweigh an absence -> absence moves a fraction.
    s2 = run_stream(LoopState("b"), [Verdict.ABSENCE] * 4, weight=2.0, absence_counts=True)
    assert s2.standing < 0.0
    # CL-062 is a PER-EVENT claim ("a contradiction must outweigh an absence", singular):
    # one contradiction's move strictly exceeds one absence's fractional move. (Cumulative
    # absences CAN sum past a contradiction — an ambiguity in CL-062 that E2 records as a
    # paper-note; the fraction only guarantees the per-event ordering.)
    one_contra = abs(update(LoopState("b"), Verdict.CONTRADICT, weight=2.0).standing)
    one_absence = abs(update(LoopState("b"), Verdict.ABSENCE, weight=2.0, absence_counts=True).standing)
    assert one_contra > one_absence


def test_status_promotes_with_confirmations():
    s = run_stream(LoopState("b"), [Verdict.CONFIRM] * 2, weight=1.0)
    assert promote(s).status == "experimental"        # < to_test
    s = run_stream(s, [Verdict.CONFIRM] * 1, weight=1.0)
    assert promote(s).status == "test"                 # >= 3
    s = run_stream(s, [Verdict.CONFIRM] * 7, weight=1.0)
    assert promote(s).status == "stable"               # >= 10


# --- H14: fade -> retire -------------------------------------------------------

def test_unconfirmed_brain_retires_confirmed_control_does_not():
    generic = run_stream(LoopState("g"), [Verdict.CONTRADICT] * 25, weight=1.0)
    assert generic.fidelity == 0.0
    assert fade_retire(generic).status == "retired"
    confirmed = run_stream(LoopState("c"), [Verdict.CONFIRM] * 25, weight=1.0)
    assert fade_retire(confirmed).status != "retired"


def test_too_few_matches_is_not_yet_retired():
    s = run_stream(LoopState("g"), [Verdict.CONTRADICT] * 5, weight=1.0)  # < min_matches
    assert fade_retire(s).status == "experimental"


# --- H15: split ----------------------------------------------------------------

def test_bimodal_brain_splits_unimodal_does_not():
    bimodal = ([MatchRecord(f"u{i}", "patternA", confirmed=True) for i in range(4)]
               + [MatchRecord(f"v{i}", "patternB", confirmed=True) for i in range(4)])
    sigs = should_split(bimodal)
    assert sigs == ["patternA", "patternB"]
    children = split(a_brain(), sigs)
    assert len(children) == 2
    assert all(any(r["relation"] == "split-from" for r in c.related) for c in children)
    assert children[0].id != children[1].id  # distinct child ids

    unimodal = [MatchRecord(f"u{i}", "patternA", confirmed=True) for i in range(8)]
    assert should_split(unimodal) == []


def test_split_needs_confirmed_support_not_just_presence():
    # two signatures but unconfirmed -> no split (a brain proves patterns by confirmation)
    recs = [MatchRecord(f"u{i}", "A") for i in range(4)] + [MatchRecord(f"v{i}", "B") for i in range(4)]
    assert should_split(recs) == []


# --- H16: merge ----------------------------------------------------------------

def test_overlapping_brains_merge_disjoint_do_not():
    a = [MatchRecord(f"u{i}", "s") for i in range(10)]
    b_same = [MatchRecord(f"u{i}", "s") for i in range(10)]
    assert should_merge(a, b_same) is True
    merged = merge(a_brain("00000000-0000-4000-8000-00000000000a"),
                   a_brain("00000000-0000-4000-8000-00000000000b"))
    assert sum(1 for r in merged.related if r["relation"] == "merged-from") == 2

    b_disjoint = [MatchRecord(f"w{i}", "s") for i in range(10)]
    assert should_merge(a, b_disjoint) is False


def test_partial_overlap_below_threshold_does_not_merge():
    a = [MatchRecord(f"u{i}", "s") for i in range(10)]
    b = [MatchRecord(f"u{i}", "s") for i in range(5)] + [MatchRecord(f"w{i}", "s") for i in range(5)]
    # Jaccard = 5 / 15 = 0.33 < 0.8
    assert should_merge(a, b) is False
