"""The operating loop — standing / fidelity / fade-retire / split / merge (Phase E2).

The paper's brain is disposable and revisable: it is "born unconfirmed and must
survive the receiver's own telemetry before it is trusted" (CL-019), carries a
fidelity = confirmed/total "set by the loop" (CL-078, CL-127), and the loop
"splits a brain that starts catching two patterns, merges two that turn out to
be one, fades one that stops matching, and retires one whose matches are never
confirmed" (CL-063, CL-081).

These are LOCAL dynamics (fidelity/standing are stripped on publish and
recomputed by every receiver — CL-133), so the loop holds them in a per-brain
``LoopState`` OUTSIDE the publish-form Brain, never mutating the shared document.
Split/merge emit NEW Brain documents carrying ``related`` lineage.

Honest scope: the CONFIRM/CONTRADICT/ABSENCE verdicts are synthetic labels (what
an analyst would return), not judgements on real telemetry — E2 demonstrates the
loop MECHANISM behaves as specified given a verdict stream, not that it discovers
real adversaries (deferred to Route A / E3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Mapping, Sequence

from brainkit.model import Brain


class Verdict(str, Enum):
    """One analyst/loop judgement about a single brain match (CL-053 pass)."""

    CONFIRM = "confirm"       # corroborating evidence: raises standing, counts confirmed
    CONTRADICT = "contradict"  # contradicting evidence: lowers standing (CL-062)
    ABSENCE = "absence"        # the brain could have fired but did not — a non-match


# Status ladder (CL-125 / CL-141): experimental -> test -> stable, plus retired.
STATUS_LADDER = ("experimental", "test", "stable")


@dataclass
class LoopState:
    """Per-brain LOCAL loop state (never published; recomputed per receiver)."""

    brain_id: str
    status: str = "experimental"
    standing: float = 0.0
    confirmed: int = 0
    total: int = 0                 # total MATCHES seen (fires), the fidelity denominator
    standing_trajectory: list[float] = field(default_factory=list)

    @property
    def fidelity(self) -> float:
        """Confirmed matches / total matches (CL-127); 0 when nothing has matched."""
        return (self.confirmed / self.total) if self.total else 0.0


def update(
    state: LoopState,
    verdict: Verdict,
    weight: float,
    *,
    absence_counts: bool = False,
    absence_weight_fraction: float = 0.25,
) -> LoopState:
    """Apply one verdict to a brain's loop state, returning the updated copy.

    Standing (CL-062) moves by the evidence weight: +weight on CONFIRM, -weight
    on CONTRADICT. ABSENCE does NOT move standing by default — this follows CL-054
    ("only a contradiction counts, never a mere absence"). The ``absence_counts``
    arm instead applies a fractional negative move, following CL-062's "a
    contradiction must outweigh an absence"; the two claims collide and E2 reports
    the tension (paper-note). Fidelity is driven by confirmed/total: CONFIRM and
    CONTRADICT are both MATCHES (the brain fired) so both raise ``total``; only
    CONFIRM raises ``confirmed``. ABSENCE is not a match and does not touch total.
    """
    s = replace(state, standing_trajectory=list(state.standing_trajectory))
    if verdict is Verdict.CONFIRM:
        s.total += 1
        s.confirmed += 1
        s.standing += weight
    elif verdict is Verdict.CONTRADICT:
        s.total += 1
        s.standing -= weight
    elif verdict is Verdict.ABSENCE:
        if absence_counts:
            s.standing -= weight * absence_weight_fraction
        # else: CL-054 default — absence never counts
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unknown verdict {verdict!r}")
    s.standing_trajectory.append(round(s.standing, 6))
    return s


def run_stream(
    state: LoopState,
    verdicts: Sequence[Verdict],
    weight: float,
    **kw,
) -> LoopState:
    """Fold a verdict stream through ``update`` (the continuous pass, CL-196 gap)."""
    for v in verdicts:
        state = update(state, v, weight, **kw)
    return state


def promote(state: LoopState, *, to_test: int = 3, to_stable: int = 10) -> LoopState:
    """Status promotion through the loop's own confirmations (CL-141).

    experimental -> test at ``to_test`` confirmed matches, test -> stable at
    ``to_stable``. Monotone; never demotes here (retirement is separate).
    """
    s = replace(state, standing_trajectory=list(state.standing_trajectory))
    if s.confirmed >= to_stable:
        s.status = "stable"
    elif s.confirmed >= to_test:
        s.status = "test"
    else:
        s.status = "experimental"
    return s


def fade_retire(state: LoopState, *, retire_below: float = 0.05, min_matches: int = 20) -> LoopState:
    """Retire a brain whose matches are never confirmed (CL-081).

    A brain that has matched at least ``min_matches`` times but whose fidelity has
    faded below ``retire_below`` is retired (status -> retired). Retirement is
    terminal; a brain with too few matches yet is left to keep proving itself.
    """
    s = replace(state, standing_trajectory=list(state.standing_trajectory))
    if s.total >= min_matches and s.fidelity < retire_below:
        s.status = "retired"
    return s


# --- split / merge -------------------------------------------------------------

@dataclass
class MatchRecord:
    """One recorded match of a brain, for split/merge analysis."""

    unit: str                       # the host-day (or window) id the match landed in
    signature: str                  # a categorical key describing WHICH pattern fired
    confirmed: bool = False


def should_split(records: Sequence[MatchRecord], *, min_support: int = 3) -> list[str]:
    """Return the distinct signatures a brain is catching, if it catches ≥2 (CL-063).

    "A brain that starts catching two patterns" is detected as ≥2 signatures each
    with ≥``min_support`` confirmed matches. Returns the qualifying signatures
    (sorted); an empty list means do-not-split (unimodal or too little support).
    """
    counts: dict[str, int] = {}
    for r in records:
        if r.confirmed:
            counts[r.signature] = counts.get(r.signature, 0) + 1
    qualifying = sorted(sig for sig, n in counts.items() if n >= min_support)
    return qualifying if len(qualifying) >= 2 else []


def split(brain: Brain, signatures: Sequence[str]) -> list[Brain]:
    """Emit one child brain per signature, each carrying split-from lineage (CL-063).

    The children are lineage-stamped clones (``related: split-from``) with a
    signature-suffixed id/title; the actual per-signature link specialization is a
    downstream authoring step — here we produce the lineage-correct child stubs so
    the union-covers-parent and disjoint-fire properties are testable.
    """
    children = []
    for sig in signatures:
        child_id = _derive_id(brain.id, sig)
        children.append(replace(
            brain,
            id=child_id,
            title=f"{brain.title} [{sig}]",
            related=tuple(brain.related) + ({"relation": "split-from", "id": brain.id},),
        ))
    return children


def should_merge(
    a: Sequence[MatchRecord], b: Sequence[MatchRecord], *, overlap_threshold: float = 0.8
) -> bool:
    """Two brains merge when they cover the same matches (CL-063).

    Overlap is the Jaccard index of the (unit, signature) sets the two brains
    matched. ``>= overlap_threshold`` means "turn out to be one." Disjoint
    coverage (Jaccard 0) does not merge.
    """
    sa = {(r.unit, r.signature) for r in a}
    sb = {(r.unit, r.signature) for r in b}
    if not sa and not sb:
        return False
    return _jaccard(sa, sb) >= overlap_threshold


def merge(a: Brain, b: Brain) -> Brain:
    """Merge two brains into one carrying merged-from lineage for both (CL-063)."""
    merged_id = _derive_id(a.id, b.id)
    return replace(
        a,
        id=merged_id,
        title=f"{a.title} + {b.title} (merged)",
        related=tuple(a.related) + (
            {"relation": "merged-from", "id": a.id},
            {"relation": "merged-from", "id": b.id},
        ),
    )


# --- helpers -------------------------------------------------------------------

def _jaccard(x: set, y: set) -> float:
    union = x | y
    return (len(x & y) / len(union)) if union else 0.0


def _derive_id(base: str, tag: str) -> str:
    """Deterministic child id: keep the UUID shape, perturb the last field.

    Avoids Math.random/uuid4 (non-deterministic); derives a stable suffix from the
    tag so re-runs are byte-identical (the repo's determinism rule).
    """
    h = 0
    for ch in f"{base}:{tag}":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFFFFFF
    return f"{base[:24]}{h:012x}"
