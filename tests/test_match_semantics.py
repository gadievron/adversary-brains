"""Fire-semantics unit matrix (pilot scope; extended at Stage 2).

Each case pins one ADR 002 choice with a hand-built brain — hand-checkable
against the rule: FIRED iff all visible required matched AND share >= 0.7;
invisible required unmatched -> SUSPENDED; share over VISIBLE supporting only.
"""
from __future__ import annotations

from brainkit.match import Status, entity_matches, match_brain
from brainkit.model import Entity, brain_from_dict

BASE = {
    "id": "00000000-0000-4000-8000-000000000001",
    "title": "test brain", "status": "experimental", "scale": "campaign",
    "representation": "graph", "match": "likelihood", "tlp": "red",
    "created": "2026-07-10", "modified": "2026-07-10",
}


def make_brain(links, entities=None, threshold=0.7):
    return brain_from_dict({
        **BASE,
        "entities": entities or [
            {"id": "e1", "type": "process", "pattern": {"name": "evil.exe"}},
            {"id": "e2", "type": "domain-name", "value": "c2.example"},
        ],
        "links": links,
        "threshold": threshold,
    })


def link(lid, role, relation="connected-to"):
    return {"id": lid, "type": "activity", "source": "e1", "target": "e2",
            "relation": relation, "role": role, "volatility": "stable",
            "provenance": {"kind": "reported", "source": "test"}}


def event(relation="connected-to", name="evil.exe", value="c2.example"):
    return {"time": "2026-07-01T10:00:00Z", "host": "h1", "class": "network_activity",
            "relation": relation,
            "source": {"type": "process", "name": name},
            "target": {"type": "domain-name", "value": value}}


def test_required_only_fires_on_match():
    brain = make_brain([link("L1", "required")])
    assert match_brain(brain, [event()]).status is Status.FIRED


def test_required_unmatched_is_no_match_when_nothing_matched():
    brain = make_brain([link("L1", "required")])
    assert match_brain(brain, [event(name="benign.exe")]).status is Status.NO_MATCH


def test_partial_when_some_but_not_all_required_matched():
    brain = make_brain([link("L1", "required"), link("L2", "required", relation="spawned")])
    assert match_brain(brain, [event()]).status is Status.PARTIAL


def test_invisible_required_suspends_not_fails():
    brain = make_brain([link("L1", "required"), link("L2", "required", relation="spawned")])
    result = match_brain(brain, [event()], invisible=frozenset({"L2"}))
    assert result.status is Status.SUSPENDED
    assert result.invisible_unmatched == ("L2",)


def test_supporting_share_gates_fire():
    links = [link("L1", "required"),
             link("L2", "supporting", relation="spawned"),
             link("L3", "supporting", relation="wrote")]
    brain = make_brain(links)  # share = 0/2 = 0 < 0.7
    assert match_brain(brain, [event()]).status is Status.PARTIAL
    # one supporting matched with equal weights -> 0.5 < 0.7 -> PARTIAL
    result = match_brain(brain, [event(), event(relation="spawned")])
    assert result.status is Status.PARTIAL and result.supporting_share == 0.5
    # both matched -> 1.0 -> FIRED
    assert match_brain(
        brain, [event(), event(relation="spawned"), event(relation="wrote")]
    ).status is Status.FIRED


def test_invisible_supporting_excluded_from_denominator():
    links = [link("L1", "required"),
             link("L2", "supporting", relation="spawned"),
             link("L3", "supporting", relation="wrote")]
    brain = make_brain(links)
    # L3 invisible: denominator is L2 alone; L2 matched -> share 1.0 -> FIRED
    result = match_brain(brain, [event(), event(relation="spawned")],
                         invisible=frozenset({"L3"}))
    assert result.status is Status.FIRED and result.supporting_share == 1.0


def test_weighted_share_uses_weights():
    links = [link("L1", "required"),
             link("L2", "supporting", relation="spawned"),
             link("L3", "supporting", relation="wrote")]
    brain = make_brain(links)
    weights = {"L2": 9.0, "L3": 1.0}
    result = match_brain(brain, [event(), event(relation="spawned")], weights=weights)
    assert result.status is Status.FIRED and result.supporting_share == 0.9


def test_not_contains_operator():
    entities = [
        {"id": "e1", "type": "process", "pattern": {"name": "solarwinds.businesslayerhost.exe"}},
        {"id": "e2", "type": "domain-name", "pattern": {"value|not-contains": "solarwinds"}},
    ]
    brain = make_brain([link("L1", "required")], entities=entities)
    assert match_brain(brain, [event(name="solarwinds.businesslayerhost.exe",
                                     value="c2.example")]).status is Status.FIRED
    assert match_brain(brain, [event(name="solarwinds.businesslayerhost.exe",
                                     value="telemetry.solarwinds.com")]).status is Status.NO_MATCH


def test_annotation_only_pattern_is_type_wildcard():
    assert entity_matches(
        Entity(id="e1", type="process", pattern={"x-note": "anything"}),
        {"type": "process", "name": "whatever.exe"},
    )


def test_mutation_control_relation_swap_kills_fire():
    """Tautology guard: the matcher can say no (verification-chain mechanism 5)."""
    brain = make_brain([link("L1", "required", relation="spawned")])
    assert match_brain(brain, [event(relation="connected-to")]).status is Status.NO_MATCH


def test_mutation_control_required_flip_kills_fire():
    """EXP-4-7: flipping the required link's relation to a bogus verb → no fire."""
    brain = make_brain([link("L1", "required", relation="bogus-does-not-occur")])
    assert match_brain(brain, [event()]).status is Status.NO_MATCH


def test_mutation_control_pattern_scramble_kills_fire():
    """EXP-4-7: scrambling the entity pattern (name never present) → no fire."""
    entities = [{"id": "e1", "type": "process", "pattern": {"name": "not-present.exe"}},
                {"id": "e2", "type": "domain-name", "value": "c2.example"}]
    brain = make_brain([link("L1", "required")], entities=entities)
    assert match_brain(brain, [event()]).status is Status.NO_MATCH


def test_supporting_only_brain_rejected_at_load():
    """EXP-P-8: a brain with no required-role link is rejected by the loader."""
    import pytest
    with pytest.raises(ValueError, match="role 'required'"):
        make_brain([link("L1", "supporting")])


def test_unknown_pattern_operator_rejected_at_load():
    """EXP-P-13: bad operators fail at load with the entity id, not mid-screen."""
    import pytest
    with pytest.raises(ValueError, match="unknown pattern operator"):
        make_brain([link("L1", "required")], entities=[
            {"id": "e1", "type": "process", "pattern": {"name|regex": ".*"}},
            {"id": "e2", "type": "domain-name", "value": "c2.example"},
        ])


def test_zero_weight_supporting_fires_on_core_pinned_semantics():
    """EXP-P-3 / ADR 002 addendum: all-zero-weight supporting -> share 1.0, fires on core.

    Pinned deliberately: a zero-information link (background rate ~1) cannot
    carry grade; the addendum documents the rationale and the alternative.
    """
    links = [link("L1", "required"), link("L2", "supporting", relation="spawned")]
    brain = make_brain(links)
    result = match_brain(brain, [event()], weights={"L2": 0.0})
    assert result.status is Status.FIRED and result.supporting_share == 1.0


def test_empty_value_does_not_fall_through_to_name():
    """EXP-P-12c: value "" is a real (non-matching) value, not an absent one."""
    from brainkit.model import Entity
    entity = Entity(id="e1", type="domain-name", value="x")
    assert not entity_matches(entity, {"type": "domain-name", "value": "", "name": "x"})
