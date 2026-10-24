"""Fire semantics for brain/0.1 (appendix 1, choices pinned by ADR 002).

A brain FIRES when every visible required link is matched and the
weight-weighted share of matched visible supporting links meets its threshold.
A required link that is invisible (per the local coverage overlay) and
unmatched SUSPENDS the brain rather than failing it. Invisible supporting
links are excluded from the share denominator; with no visible supporting
links the share is 1.0 and the brain fires on its required core alone.
Required links gate, they do not grade. The rule is flat: no temporal
ordering within the match window (ADR 003).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .model import Brain, Entity, Link

#: Pattern-key operators (ADR 005). `x-*` keys are annotations, never matched.
_OPS = {
    "equals": lambda actual, expected: actual == expected,
    "contains": lambda actual, expected: expected in actual,
    "starts": lambda actual, expected: actual.startswith(expected),
    "ends": lambda actual, expected: actual.endswith(expected),
    "not-contains": lambda actual, expected: expected not in actual,
}


class Status(Enum):
    """Outcome of evaluating one brain against one evaluation unit."""

    FIRED = "FIRED"
    PARTIAL = "PARTIAL"
    NO_MATCH = "NO_MATCH"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class MatchResult:
    """One brain × one evaluation unit, with the evidence that produced it."""

    brain_id: str
    status: Status
    matched: tuple[str, ...]
    unmatched: tuple[str, ...]
    invisible_unmatched: tuple[str, ...]
    supporting_share: float
    explanation: str


def entity_matches(entity: Entity, obj: Mapping[str, Any] | None) -> bool:
    """Does an event-side object satisfy this entity?

    Concrete `value` compares case-insensitively against the object's `value`
    or `name`. A `pattern` requires every non-`x-*` constraint to hold; a
    pattern with ONLY `x-*` keys matches any object of the declared type — a
    deliberate wildcard (BRAIN-AUTHORING §4.7).
    """
    if obj is None:
        return False
    if entity.type != obj.get("type"):
        return False
    if entity.value is not None:
        # Explicit None checks, not `or`: falsy values ("" , 0) are real values,
        # and "" must not fall through to `name` (EXP-P-12c).
        candidate = obj.get("value")
        if candidate is None:
            candidate = obj.get("name")
        if candidate is None:
            return False
        return str(candidate).lower() == entity.value.lower()
    assert entity.pattern is not None  # enforced by Entity.__post_init__
    for key, expected in entity.pattern.items():
        if key.startswith("x-"):
            continue
        field_name, _, op_name = key.partition("|")
        op = _OPS[op_name or "equals"]
        actual = obj.get(field_name)
        if actual is None:
            return False
        if not op(str(actual).lower(), str(expected).lower()):
            return False
    return True


def link_matches(link: Link, brain: Brain, event: Mapping[str, Any]) -> bool:
    """One event (or state fact) satisfies one link (ADR 003 triple)."""
    if event.get("relation") != link.relation:
        return False
    return entity_matches(brain.entity(link.source), event.get("source")) and entity_matches(
        brain.entity(link.target), event.get("target")
    )


def match_brain(
    brain: Brain,
    events: Iterable[Mapping[str, Any]],
    facts: Iterable[Mapping[str, Any]] = (),
    weights: Mapping[str, float] | None = None,
    invisible: frozenset[str] = frozenset(),
) -> MatchResult:
    """Evaluate one brain against one evaluation unit (ADR 002 fire rule).

    ``events`` is the unit's activity stream (one host, one window);
    ``facts`` the global state stream; ``weights`` maps link id -> weight
    (default 1.0 each); ``invisible`` is the coverage overlay's set of
    link ids this environment cannot observe.
    """
    events = list(events)
    facts = list(facts)
    weight_of = (weights or {}).get

    matched: list[str] = []
    unmatched_visible_required: list[str] = []
    unmatched_invisible_required: list[str] = []
    for link in brain.links:
        stream = facts if link.type == "state" else events
        if any(link_matches(link, brain, item) for item in stream):
            matched.append(link.id)
        elif link.role == "required":
            if link.id in invisible:
                unmatched_invisible_required.append(link.id)
            else:
                unmatched_visible_required.append(link.id)

    matched_set = frozenset(matched)
    visible_supporting = [link for link in brain.supporting_links if link.id not in invisible]
    denominator = sum(weight_of(link.id, 1.0) for link in visible_supporting)
    numerator = sum(weight_of(link.id, 1.0) for link in visible_supporting if link.id in matched_set)
    share = (numerator / denominator) if denominator > 0 else 1.0

    unmatched = tuple(link.id for link in brain.links
                      if link.id not in matched_set and link.id not in unmatched_invisible_required)

    if unmatched_visible_required:
        status = Status.PARTIAL if matched else Status.NO_MATCH
        explanation = (f"visible required link(s) unmatched: "
                       f"{', '.join(unmatched_visible_required)}")
    elif unmatched_invisible_required:
        status = Status.SUSPENDED
        explanation = (f"required link(s) unobservable in this environment: "
                       f"{', '.join(unmatched_invisible_required)} — not seen is not not-there")
    elif share >= brain.threshold:
        status = Status.FIRED
        explanation = (f"all required links matched; supporting share "
                       f"{share:.2f} >= threshold {brain.threshold}")
    else:
        status = Status.PARTIAL
        explanation = (f"all required links matched but supporting share "
                       f"{share:.2f} < threshold {brain.threshold}")

    return MatchResult(
        brain_id=brain.id,
        status=status,
        matched=tuple(matched),
        unmatched=unmatched,
        invisible_unmatched=tuple(unmatched_invisible_required),
        supporting_share=round(share, 4),
        explanation=explanation,
    )
