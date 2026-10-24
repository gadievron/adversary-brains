"""Weight arithmetic against hand-computed cases (ADR 004; pilot scope)."""
from __future__ import annotations

import math

import pytest

from brainkit.model import brain_from_dict
from brainkit.weights import (
    LinkStats, background_stats, group_units, p_fire_required_core, unit_key,
)


def test_rate_and_weight_hand_computed():
    stat = LinkStats(link_id="L1", units_matched=0, units_total=49)
    assert stat.rate == pytest.approx(1 / 50)
    assert stat.weight == pytest.approx(-math.log2(1 / 50))  # ≈ 5.64 bits

    common = LinkStats(link_id="L2", units_matched=49, units_total=49)
    assert common.rate == pytest.approx(1.0)
    assert common.weight == pytest.approx(0.0)


def test_weight_cap_binds_for_huge_corpora():
    stat = LinkStats(link_id="L1", units_matched=0, units_total=2**40)
    assert stat.weight == 30.0


def test_rare_outweighs_common():
    rare = LinkStats("a", 1, 999)
    common = LinkStats("b", 500, 999)
    assert rare.weight > common.weight


def test_unit_key_windows():
    early = {"time": "2026-07-01T01:00:00Z", "host": "h1"}
    late = {"time": "2026-07-02T01:00:00Z", "host": "h1"}
    other = {"time": "2026-07-01T01:00:00Z", "host": "h2"}
    assert unit_key(early) != unit_key(late)          # different 24h windows
    assert unit_key(early) != unit_key(other)         # different hosts
    assert len(group_units([early, late, other])) == 3


def test_p_fire_required_core_is_product_of_rates():
    brain = brain_from_dict({
        "id": "00000000-0000-4000-8000-000000000002",
        "title": "t", "status": "experimental", "scale": "campaign",
        "representation": "graph", "match": "likelihood", "tlp": "red",
        "created": "2026-07-10", "modified": "2026-07-10",
        "entities": [
            {"id": "e1", "type": "process", "pattern": {"name": "a.exe"}},
            {"id": "e2", "type": "process", "pattern": {"name": "b.exe"}},
        ],
        "links": [
            {"id": "L1", "type": "activity", "source": "e1", "target": "e2",
             "relation": "spawned", "role": "required", "volatility": "stable",
             "provenance": {"kind": "reported", "source": "test"}},
            {"id": "L2", "type": "activity", "source": "e2", "target": "e1",
             "relation": "wrote", "role": "required", "volatility": "stable",
             "provenance": {"kind": "reported", "source": "test"}},
        ],
    })
    stats = {"L1": LinkStats("L1", 0, 99), "L2": LinkStats("L2", 9, 99)}
    assert p_fire_required_core(brain, stats) == pytest.approx((1 / 100) * (10 / 100))


def test_p_fire_refuses_supporting_links():
    brain = brain_from_dict({
        "id": "00000000-0000-4000-8000-000000000003",
        "title": "t", "status": "experimental", "scale": "campaign",
        "representation": "graph", "match": "likelihood", "tlp": "red",
        "created": "2026-07-10", "modified": "2026-07-10",
        "entities": [
            {"id": "e1", "type": "process", "pattern": {"name": "a.exe"}},
            {"id": "e2", "type": "process", "pattern": {"name": "b.exe"}},
        ],
        "links": [
            {"id": "L1", "type": "activity", "source": "e1", "target": "e2",
             "relation": "spawned", "role": "required", "volatility": "stable",
             "provenance": {"kind": "reported", "source": "test"}},
            {"id": "L2", "type": "activity", "source": "e2", "target": "e1",
             "relation": "wrote", "role": "supporting", "volatility": "stable",
             "provenance": {"kind": "reported", "source": "test"}},
        ],
    })
    with pytest.raises(ValueError, match="required-core-only"):
        p_fire_required_core(brain, {"L1": LinkStats("L1", 0, 9), "L2": LinkStats("L2", 0, 9)})


def test_background_stats_counts_per_unit():
    brain = brain_from_dict({
        "id": "00000000-0000-4000-8000-000000000004",
        "title": "t", "status": "experimental", "scale": "campaign",
        "representation": "graph", "match": "likelihood", "tlp": "red",
        "created": "2026-07-10", "modified": "2026-07-10",
        "entities": [
            {"id": "e1", "type": "process", "pattern": {"name": "x.exe"}},
            {"id": "e2", "type": "domain-name", "value": "d.example"},
        ],
        "links": [
            {"id": "L1", "type": "activity", "source": "e1", "target": "e2",
             "relation": "connected-to", "role": "required", "volatility": "stable",
             "provenance": {"kind": "reported", "source": "test"}},
        ],
    })
    hit = {"time": "2026-07-01T01:00:00Z", "host": "h1", "relation": "connected-to",
           "source": {"type": "process", "name": "x.exe"},
           "target": {"type": "domain-name", "value": "d.example"}}
    miss = {"time": "2026-07-01T01:00:00Z", "host": "h2", "relation": "connected-to",
            "source": {"type": "process", "name": "y.exe"},
            "target": {"type": "domain-name", "value": "d.example"}}
    stats = background_stats(brain, [hit, hit, miss])  # 2 units; L1 matches in 1
    assert stats["L1"].units_matched == 1 and stats["L1"].units_total == 2
