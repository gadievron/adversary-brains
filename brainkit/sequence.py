"""Sequence-aware matching (the `match: sequence` representation).

The flat co-occurrence fire rule (ADR 002) cannot demonstrate the paper's
"arrangement of individually-common pieces is the detection" claim: within any
window, common pieces co-occur routinely (exp6: a flat common-pieces brain
false-fires 6000/6000). Ordering is the missing ingredient. This module adds a
fire rule that requires a brain's required links to match in a declared TEMPORAL
ORDER on one host within a window — so an arrangement of common pieces fires on
the true ordered chain but not on the same pieces shuffled or scattered.

A sequence brain declares `match: sequence`; its required links are ordered by the
integer embedded in their link id (L0, L1, L2, ...) — there is no schema `step`
field; `_step` reads the id's digits (see its note). ``match_sequence`` fires iff there is a strictly
time-ordered assignment of one matching event per step within the window on a
single host. This is additive: `match: likelihood` brains keep the flat rule in
match.py. NOTE the `match` field is declared INTENT, not an engine dispatch key:
nothing reads it at match time — the caller chooses the entry point (match.py's
`match_brain` for the flat rule, this module's `match_sequence` for sequence).

SCOPE (what this matcher does NOT do, unlike match.py's flat rule): it evaluates
REQUIRED links only — there is no supporting-share computation, no `state`-link /
`facts` stream, no weights, and no coverage overlay, so a sequence brain never
returns SUSPENDED. Supporting links and the coverage/suspension semantics belong
to the flat `match: likelihood` path.

GREEDY, SOUND, INCOMPLETE. It reports FIRED only after actually binding one
matching event to every step in strict time order with consistent entity
bindings, so a FIRED is a witness that such an assignment exists — it is SOUND.
It is INCOMPLETE because of BINDING FORECLOSURE, not ordering: with
``bind_entities``, step 0 greedily binds the earliest step-0 event (say actor
A's), and a later step can then fail on A even though a DIFFERENT actor B has the
whole ordered chain later in the window; the single non-resetting cursor never
rewinds to try B. (Without binding, strict-subsequence greedy earliest-match is
actually complete, by an exchange argument — so the gap is the binding, not the
greed over time.) DIRECTION matters for reading exp7's number: greedy-FIRED is a
SUBSET of exhaustive-FIRED per unit, so an exhaustive matcher's BACKGROUND fire
rate can only be >= the published greedy rate — exhaustive would make the
"arrangement is rare in background" claim HARDER to support, not easier. The
shipped matcher's rate is a frozen published result; a backtracking/exhaustive
variant is deferred to a new pre-registered experiment (see docs/FOLLOWUPS.md),
not changed here. The strict `<=` tie rule (equal-timestamp events cannot satisfy
two steps) is an intentional "strictly later" ordering, not a bug.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .match import Status, entity_matches, link_matches
from .model import Brain, Link


def _step(link: Link) -> int:
    """The declared order index of a required link (from its id suffix or a
    convention). Sequence brains name required links L0,L1,... and fire in that
    order; the step is the trailing integer of the link id.

    CONVENTION (not validated): this collects ALL digits in the id, so a plain
    `L0`/`L1`/`L2` reads correctly but a decorated id like `L1-v2` would read as
    12 and reorder the chain. Sequence-brain link ids must be `L<n>` with a single
    integer and no other digits; author them accordingly."""
    digits = "".join(ch for ch in link.id if ch.isdigit())
    return int(digits) if digits else 0


def _time(event: Mapping[str, Any]) -> _dt.datetime:
    return _dt.datetime.fromisoformat(str(event["time"]).replace("Z", "+00:00"))


@dataclass(frozen=True)
class SequenceResult:
    brain_id: str
    status: Status
    order_found: tuple[str, ...]  # link ids matched, in the order satisfied
    explanation: str


def _bound_value(brain: Brain, entity_id: str, obj: Mapping[str, Any] | None) -> str | None:
    """The concrete event-side value that binds a brain entity in a matched event
    (its `value` or `name`), for cross-step binding. None if unavailable."""
    if obj is None:
        return None
    v = obj.get("value")
    if v is None:
        v = obj.get("name")
    return None if v is None else str(v).lower()


def match_sequence(brain: Brain, events: Iterable[Mapping[str, Any]],
                   bind_entities: bool = True) -> SequenceResult:
    """Fire iff the ordered required links match in strict time order, with the
    SAME concrete entity bound wherever a brain entity recurs across steps.

    Greedy earliest-match over time-sorted events; when ``bind_entities`` (the
    default and the point of the paper's "one actor's arrangement"), an event
    only satisfies a step if the values it would bind to already-bound brain
    entities agree — so an ordered triple on THREE DIFFERENT users does not
    fire, only one actor's own register -> federate -> delete does. Without
    binding, an ordered subsequence over many common events exists too often
    (exp7 probe: ~58%) — binding is what makes the arrangement rare.
    """
    ordered = sorted(brain.required_links, key=_step)
    if not ordered:
        return SequenceResult(brain.id, Status.NO_MATCH, (), "no required links to order")
    stream = sorted((e for e in events), key=_time)

    matched_ids: list[str] = []
    bindings: dict[str, str] = {}
    cursor = None
    idx = 0
    for link in ordered:
        found = False
        while idx < len(stream):
            event = stream[idx]
            idx += 1
            if cursor is not None and _time(event) <= cursor:
                continue
            if not link_matches(link, brain, event):
                continue
            if bind_entities:
                trial = dict(bindings)
                ok = True
                for brain_eid, side in ((link.source, "source"), (link.target, "target")):
                    val = _bound_value(brain, brain_eid, event.get(side))
                    if val is None:
                        continue
                    if brain_eid in trial and trial[brain_eid] != val:
                        ok = False
                        break
                    trial[brain_eid] = val
                if not ok:
                    continue
                bindings = trial
            matched_ids.append(link.id)
            cursor = _time(event)
            found = True
            break
        if not found:
            return SequenceResult(
                brain.id, Status.PARTIAL if matched_ids else Status.NO_MATCH,
                tuple(matched_ids),
                f"ordered chain broke at step {_step(link)} ({link.id}): "
                f"no later event matching the step (with entity binding)")
    return SequenceResult(
        brain.id, Status.FIRED, tuple(matched_ids),
        f"required links matched in declared order (entities bound): {' -> '.join(matched_ids)}")
