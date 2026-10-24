"""Weight estimation from a background corpus (ADR 004) and null-model surprise.

Weight is the paper's environmental quantity: the negative log of a link's
background rate in ordinary traffic, estimated per evaluation unit — one
(host × window) — with add-one smoothing and a 30-bit cap. It is distinct
from the screen's library rarity (ADR 008) and is LOCAL: never shipped.

The null-model surprise (STATISTICS.md, corrected per EXP-0-4) is computed
here for the required-core case, where p_fire = ∏ rate(required link) is the
exact tail probability of firing under the independence null. Surprise uses
UNCAPPED rates (the cap exception).
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .match import link_matches
from .model import Brain

WEIGHT_CAP_BITS = 30.0
DEFAULT_WINDOW_HOURS = 24


def unit_key(event: Mapping[str, Any], window_hours: int = DEFAULT_WINDOW_HOURS) -> tuple[str, int]:
    """The evaluation-unit key of one event: (host, window index).

    REQUIRES a timezone-qualified ``time`` (trailing ``Z`` or an explicit offset);
    every shipped corpus/fixture satisfies this. A tz-NAIVE timestamp makes
    ``.timestamp()`` interpret the instant in the RUNNER's local timezone, so the
    same wall-clock event could land in a different 24h window on a differently
    configured machine — a nondeterministic eval-unit split. Not exercised today
    (no naive timestamp reaches this function); if a future experiment windows an
    external source that may emit naive timestamps (e.g. the Mordor adapter's raw
    Sysmon ``TimeCreated`` — see ``mordor_adapter.to_triples``), normalize to UTC
    before calling, or this becomes a real bug rather than a latent one.
    """
    moment = _dt.datetime.fromisoformat(str(event["time"]).replace("Z", "+00:00"))
    window_index = int(moment.timestamp() // (window_hours * 3600))
    return (str(event["host"]), window_index)


def group_units(
    events: Iterable[Mapping[str, Any]], window_hours: int = DEFAULT_WINDOW_HOURS
) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    """Group an event stream into evaluation units."""
    units: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        units[unit_key(event, window_hours)].append(event)
    return dict(units)


@dataclass(frozen=True)
class LinkStats:
    """Background statistics for one link, over N calibration units."""

    link_id: str
    units_matched: int
    units_total: int

    @property
    def rate(self) -> float:
        """Smoothed background rate: (count + 1) / (N + 1)."""
        return (self.units_matched + 1) / (self.units_total + 1)

    @property
    def weight(self) -> float:
        """Capped weight in bits: min(−log₂ rate, 30)."""
        return min(-math.log2(self.rate), WEIGHT_CAP_BITS)


def background_stats(
    brain: Brain,
    events: Iterable[Mapping[str, Any]],
    facts: Iterable[Mapping[str, Any]] = (),
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> dict[str, LinkStats]:
    """Per-link background stats over a calibration corpus.

    Activity links are counted per evaluation unit; state links are counted
    against the global facts stream in every unit (a standing fact has no
    window of its own — open question Q1, pinned here as visible-everywhere).
    """
    units = group_units(events, window_hours)
    total = len(units)
    facts = list(facts)
    counts: dict[str, int] = {link.id: 0 for link in brain.links}
    for link in brain.links:
        if link.type == "state":
            counts[link.id] = total if any(
                link_matches(link, brain, fact) for fact in facts
            ) else 0
            continue
        for unit_events in units.values():
            if any(link_matches(link, brain, event) for event in unit_events):
                counts[link.id] += 1
    return {
        link_id: LinkStats(link_id=link_id, units_matched=count, units_total=total)
        for link_id, count in counts.items()
    }


def weight_table(stats: Mapping[str, LinkStats]) -> dict[str, float]:
    """Link id -> weight, for match_brain's supporting-share arithmetic."""
    return {link_id: stat.weight for link_id, stat in stats.items()}


def p_fire_required_core(brain: Brain, stats: Mapping[str, LinkStats]) -> float:
    """Exact null tail probability of firing, for required-core-only brains.

    Valid only when the brain has no supporting links (or they cannot alter
    the fire decision); the general union form is Stage-2 work. Uses
    uncapped smoothed rates (the surprise identity's cap exception).
    """
    if brain.supporting_links:
        raise ValueError(
            f"brain {brain.id}: p_fire_required_core is exact only for "
            "required-core-only brains; general form not implemented in the pilot"
        )
    p = 1.0
    for link in brain.required_links:
        p *= stats[link.id].rate
    return p


def render_table(stats: Mapping[str, LinkStats]) -> str:
    """Human-checkable table: link -> count/N -> rate -> weight (bits)."""
    lines = ["link        count/N        rate        weight(bits)"]
    for link_id, stat in sorted(stats.items()):
        lines.append(
            f"{link_id:<11} {stat.units_matched}/{stat.units_total:<12} "
            f"{stat.rate:<11.6f} {stat.weight:.2f}"
        )
    return "\n".join(lines)
