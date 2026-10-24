"""Grounded model assessment (appendix 3 assess/synthesize; the C11 constraint).

A shortlisted brain is judged by the model, but the paper's guardrails run
DETERMINISTICALLY, outside the model, so they hold regardless of what the model
says (and are testable without one):

  * evidence that is not present in the deterministic match block is DISCARDED
    ("every model verdict must cite evidence from the deterministic match
    block"); an ungrounded verdict loses its ungrounded evidence;
  * confidence is CAPPED (default 0.4) unless at least one matched piece is
    rare (rarity >= 0.5) — "confidence is capped when no hit has rarity 0.5 or
    more"; the model ranks, it does not set calibrated probability;
  * matched-piece facts come from the deterministic screen/match, never the
    model (so a hallucinated tool/id cannot enter as evidence).

The model call itself is via llm.LLMClient (record/replay); ``assess_one``
takes an already-built LLMClient so tests can inject replay or a fake.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .llm import LLMClient, extract_json

CONFIDENCE_CAP = 0.4
RARE_THRESHOLD = 0.5

ASSESS_SYSTEM = (
    "You are a threat-intel analyst judging whether an event matches a known actor's "
    "playbook. You are given the event text and a DETERMINISTIC MATCHES block listing "
    "every piece that matched, with its rarity. Rules you must obey: (1) cite ONLY "
    "evidence that appears in the DETERMINISTIC MATCHES block; (2) return strict JSON "
    "with keys verdict (one of \"match\",\"partial\",\"no\"), confidence (0..1), evidence "
    "(list of short strings, each quoting a matched piece), reasoning (one sentence). "
    "Do not invent tools, technique ids, or indicators."
)


@dataclass(frozen=True)
class MatchedPiece:
    """One deterministically-matched piece offered to the model as evidence."""

    token: str
    rarity: float


@dataclass(frozen=True)
class Verdict:
    """A grounded verdict: model text after the deterministic guardrails."""

    brain_id: str
    verdict: str
    confidence: float
    evidence: tuple[str, ...]
    reasoning: str
    dropped_evidence: tuple[str, ...] = field(default_factory=tuple)
    confidence_capped: bool = False


def _match_block(pieces: Sequence[MatchedPiece]) -> str:
    if not pieces:
        return "(no deterministic matches)"
    return "\n".join(f"- {p.token} (rarity {p.rarity:.2f})" for p in pieces)


def ground_verdict(brain_id: str, raw: Mapping[str, Any],
                   pieces: Sequence[MatchedPiece]) -> Verdict:
    """Apply the deterministic guardrails to a raw model verdict.

    Pure function (no model): filter evidence to the match block, cap confidence
    when nothing rare matched. Tested directly, so H5 holds without an endpoint.
    """
    allowed = {p.token.lower() for p in pieces if p.token}
    raw_evidence = [str(e) for e in (raw.get("evidence") or [])]
    kept, dropped = [], []
    for item in raw_evidence:
        # Grounded iff the evidence string references a matched token as a
        # DELIMITED token, not a bare substring (EXP-3-1): a short token like
        # "ip" must not ground "description". Delimiters are string edges or
        # any non-alphanumeric, so "203.0.113.7", ".akira", and multi-word
        # tokens ("cobalt strike") all match correctly.
        low = item.lower()
        if any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", low)
               for token in allowed):
            kept.append(item)
        else:
            dropped.append(item)

    confidence = float(raw.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    has_rare = any(p.rarity >= RARE_THRESHOLD for p in pieces)
    capped = False
    if not has_rare and confidence > CONFIDENCE_CAP:
        confidence = CONFIDENCE_CAP
        capped = True

    verdict = str(raw.get("verdict", "no")).lower()
    if verdict not in {"match", "partial", "no"}:
        verdict = "no"

    return Verdict(
        brain_id=brain_id,
        verdict=verdict,
        confidence=confidence,
        evidence=tuple(kept),
        reasoning=str(raw.get("reasoning", "")),
        dropped_evidence=tuple(dropped),
        confidence_capped=capped,
    )


def assess_one(client: LLMClient, brain_id: str, event_text: str,
               pieces: Sequence[MatchedPiece]) -> Verdict:
    """Judge one shortlisted brain: model call (record/replay) then grounding."""
    messages = [
        {"role": "system", "content": ASSESS_SYSTEM},
        {"role": "user", "content":
            f"EVENT:\n{event_text}\n\nDETERMINISTIC MATCHES for brain {brain_id}:\n"
            f"{_match_block(pieces)}\n\nReturn the JSON verdict."},
    ]
    result = client.chat(messages, force_json=True)
    raw = extract_json(result.text)
    return ground_verdict(brain_id, raw, pieces)


def synthesize(verdicts: Sequence[Verdict]) -> list[Verdict]:
    """Rank verdicts by (verdict priority, confidence) — the synthesis order."""
    order = {"match": 0, "partial": 1, "no": 2}
    return sorted(verdicts, key=lambda v: (order.get(v.verdict, 3), -v.confidence, v.brain_id))
