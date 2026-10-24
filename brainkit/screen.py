"""Deterministic screen: the cheap shortlist pass before full matching.

The runtime shape of the paper's "one cheap pass" (appendix 3): index every
brain's concrete values, pattern literals, and relation verbs once, then score
an event batch against all brains and return the top-N. Scoring uses library
RARITY (1 / #brains referencing a token, floored) — the diagnosticity term of
ADR 008, kept strictly separate from the environmental weight in weights.py.

Contract (appendix 3 done-when, at 5x that scale): the full screen over a
1,000-brain library returns in under one second (exercised by exp5 at Stage 4).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .model import Brain

RARITY_FLOOR = 0.1
VALUE_HIT = 3.0
PATTERN_LITERAL_HIT = 1.0
RELATION_HIT = 0.25

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]+")


def _brain_tokens(brain: Brain) -> tuple[set[str], set[str], set[str]]:
    """(concrete values, pattern literals, relation verbs) for one brain, folded."""
    values: set[str] = set()
    literals: set[str] = set()
    relations = {link.relation.lower() for link in brain.links}
    for entity in brain.entities:
        if entity.value is not None:
            values.add(entity.value.lower())
        if entity.pattern:
            for key, val in entity.pattern.items():
                if key.startswith("x-"):
                    continue
                literals.update(_TOKEN.findall(str(val).lower()))
    return values, literals, relations


@dataclass(frozen=True)
class _Profile:
    brain_id: str
    values: frozenset[str]
    literals: frozenset[str]
    relations: frozenset[str]


@dataclass(frozen=True)
class ScreenHit:
    """One shortlisted brain with its score and the signals that earned it."""

    brain_id: str
    score: float
    signals: tuple[str, ...]


class Screen:
    """A prebuilt index over a brain library; call ``rank`` per event batch."""

    def __init__(self, brains: Sequence[Brain]):
        self._profiles: list[_Profile] = []
        for brain in brains:
            values, literals, relations = _brain_tokens(brain)
            self._profiles.append(
                _Profile(brain.id, frozenset(values), frozenset(literals), frozenset(relations))
            )
        token_brains: dict[str, set[str]] = defaultdict(set)
        for profile in self._profiles:
            for token in profile.values | profile.literals:
                token_brains[token].add(profile.brain_id)
        self._rarity = {
            token: max(1.0 / len(owners), RARITY_FLOOR)
            for token, owners in token_brains.items()
        }

    def _event_tokens(self, events: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
        tokens: set[str] = set()
        relations: set[str] = set()
        for event in events:
            relations.add(str(event.get("relation", "")).lower())
            for side in ("source", "target"):
                obj = event.get(side) or {}
                for key in ("value", "name", "command_line"):
                    if obj.get(key) is not None:
                        tokens.update(_TOKEN.findall(str(obj[key]).lower()))
        return tokens, relations

    def rank(self, events: Iterable[Mapping[str, Any]], top_n: int = 3,
             min_score: float = 0.01) -> list[ScreenHit]:
        """Score every brain against an event batch; return the top-N by score."""
        tokens, relations = self._event_tokens(list(events))
        hits: list[ScreenHit] = []
        for profile in self._profiles:
            score = 0.0
            signals: list[str] = []
            for token in profile.values & tokens:
                score += VALUE_HIT * self._rarity.get(token, RARITY_FLOOR)
                signals.append(f"value:{token}")
            for token in profile.literals & tokens:
                score += PATTERN_LITERAL_HIT * self._rarity.get(token, RARITY_FLOOR)
                signals.append(f"literal:{token}")
            for relation in profile.relations & relations:
                score += RELATION_HIT
                signals.append(f"relation:{relation}")
            if score >= min_score:
                hits.append(ScreenHit(profile.brain_id, round(score, 4), tuple(sorted(signals))))
        hits.sort(key=lambda h: (-h.score, h.brain_id))
        return hits[:top_n]
