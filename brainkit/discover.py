"""Discover — bottom-up brain formation from data that fits nothing (Phase E2).

"Discover grows a brain bottom-up from an observation that fits nothing, or that
ties two existing brains together, and watches for the rest of the pattern to
arrive" (CL-090). The paper's appendix-3 implements only the *proposing* half;
the *watching* half is unbuilt (CL-090 note, CL-076). E2 builds both:

  propose:  cluster the unmatched RESIDUAL by a structural signature; a signature
            that recurs >= min_support becomes a proposed brain (bootstrap
            maturity / experimental status). One-off noise is NOT proposed.
  watch:    the proposed brain is a real Brain document, so match_brain() fires it
            when "the rest of the pattern" (another event of that structure)
            arrives — the watch half, demonstrated by feeding a completing event.

A brain born this way is an emergent cluster (CL-056), the Discover origin of the
two-direction model (CL-086; Extract is the top-down twin). Honest scope: the
"residual" is the synthetic generator's stream minus what the shipped brains
match — Discover proposing on it demonstrates the MECHANISM, not that it finds
real novel actors (deferred to Route A / E3).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Mapping, Sequence

from brainkit.match import Status, match_brain
from brainkit.model import Brain, brain_from_dict

# A structural signature: (relation, source-type, source-key, target-type, target-key).
# It keys on the recurring STRUCTURE + concrete identifiers, so a proposed brain
# generalizes to the next event of the same shape (the watch half).
Signature = tuple


def _obj_key(obj: Mapping) -> tuple[str, str]:
    """(type, identifier) for an event endpoint; 'name' or 'value', whichever it carries."""
    ident = obj.get("name", obj.get("value", ""))
    return (obj.get("type", ""), str(ident))


def signature(event: Mapping) -> Signature:
    """Structural signature of one event triple (the clustering key)."""
    st, sid = _obj_key(event.get("source", {}))
    tt, tid = _obj_key(event.get("target", {}))
    return (event.get("relation", ""), st, sid, tt, tid)


def cluster_residual(
    residual: Sequence[Mapping], key: Callable[[Mapping], Signature] = signature
) -> Counter:
    """Count structural signatures across the unmatched residual stream."""
    return Counter(key(ev) for ev in residual)


def _brain_from_signature(sig: Signature, index: int) -> Brain:
    """Construct a proposed (bootstrap/experimental) brain that matches ``sig``."""
    relation, st, sid, tt, tid = sig
    src_field = "name" if st in ("process",) else "value"
    tgt_field = "name" if tt in ("process",) else "value"
    bid = f"d15c0000-0000-4000-8000-{index:012d}"
    return brain_from_dict({
        "id": bid,
        "title": f"discovered:{relation}:{sid}->{tid}",
        "status": "experimental",   # born unconfirmed (CL-019); bootstrap maturity
        "scale": "campaign", "representation": "graph", "match": "likelihood", "tlp": "red",
        "created": "2026-07-10", "modified": "2026-07-10",
        "entities": [
            {"id": "e_src", "type": st, "pattern": {src_field: sid}},
            {"id": "e_tgt", "type": tt, "pattern": {tgt_field: tid}},
        ],
        "links": [{"id": "L1", "type": "activity", "source": "e_src", "target": "e_tgt",
                   "relation": relation, "role": "required", "volatility": "volatile",
                   "provenance": {"kind": "discovered", "source": "discover"}}],
        "threshold": 0.7,
    })


def propose(
    residual: Sequence[Mapping],
    *,
    min_support: int = 3,
    key: Callable[[Mapping], Signature] = signature,
) -> list[Brain]:
    """Propose brains for signatures recurring >= min_support in the residual (CL-090).

    Returns bootstrap/experimental brains, one per qualifying signature, sorted for
    determinism. A stream of all-distinct (noisy) observations yields none.
    """
    counts = cluster_residual(residual, key)
    qualifying = sorted(sig for sig, n in counts.items() if n >= min_support)
    return [_brain_from_signature(sig, i) for i, sig in enumerate(qualifying)]


def watch(brain: Brain, later_events: Sequence[Mapping], weights: Mapping[str, float] | None = None) -> bool:
    """The watch half: does the proposed brain FIRE when the rest of the pattern arrives?"""
    return match_brain(brain, list(later_events), weights=weights).status is Status.FIRED
