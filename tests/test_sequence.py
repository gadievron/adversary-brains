"""Sequence-aware matching unit tests (E1 / H8): ordering + entity binding."""
from __future__ import annotations

from brainkit.match import Status
from brainkit.model import brain_from_dict
from brainkit.sequence import match_sequence

BASE = {"schema": "brain/0.1", "status": "test", "scale": "attack-chain",
        "representation": "graph", "match": "sequence", "tlp": "red",
        "created": "2026-07-10", "modified": "2026-07-10"}


def _brain():
    return brain_from_dict({**BASE, "id": "00000000-0000-4000-8000-00000000seqT",
        "title": "ordered chain",
        "entities": [{"id": "e1", "type": "user-account", "pattern": {"x-note": "any"}},
                     {"id": "e2", "type": "x-identity-provider", "pattern": {"x-note": "any"}}],
        "links": [
            {"id": "L0", "type": "activity", "source": "e1", "target": "e1", "relation": "reg",
             "role": "required", "volatility": "stable", "provenance": {"kind": "observed", "source": "t"}},
            {"id": "L1", "type": "activity", "source": "e2", "target": "e1", "relation": "fed",
             "role": "required", "volatility": "stable", "provenance": {"kind": "observed", "source": "t"}},
            {"id": "L2", "type": "activity", "source": "e1", "target": "e1", "relation": "del",
             "role": "required", "volatility": "stable", "provenance": {"kind": "observed", "source": "t"}}]})


def _e(t, rel, user):
    src = {"type": "x-identity-provider", "value": f"idp-{user}"} if rel == "fed" \
        else {"type": "user-account", "value": user}
    return {"time": f"2026-07-01T00:0{t}:00Z", "host": "h", "relation": rel,
            "source": src, "target": {"type": "user-account", "value": user}}


def test_ordered_same_user_fires():
    b = _brain()
    ev = [_e(0, "reg", "u"), _e(1, "fed", "u"), _e(2, "del", "u")]
    assert match_sequence(b, ev).status is Status.FIRED


def test_shuffled_order_does_not_fire():
    b = _brain()
    ev = [_e(0, "del", "u"), _e(1, "reg", "u"), _e(2, "fed", "u")]
    assert match_sequence(b, ev).status is not Status.FIRED


def test_different_users_do_not_fire_entity_binding():
    b = _brain()
    ev = [_e(0, "reg", "a"), _e(1, "fed", "b"), _e(2, "del", "c")]
    assert match_sequence(b, ev).status is not Status.FIRED


def test_binding_can_be_disabled():
    b = _brain()
    ev = [_e(0, "reg", "a"), _e(1, "fed", "b"), _e(2, "del", "c")]
    assert match_sequence(b, ev, bind_entities=False).status is Status.FIRED


def test_missing_step_is_partial():
    b = _brain()
    ev = [_e(0, "reg", "u"), _e(1, "del", "u")]  # no fed
    assert match_sequence(b, ev).status is Status.PARTIAL


def test_out_of_window_order_matters():
    """Events present but the required order impossible (del before reg only)."""
    b = _brain()
    ev = [_e(0, "del", "u"), _e(1, "reg", "u")]  # reg after del, no fed at all
    r = match_sequence(b, ev)
    assert r.status is Status.PARTIAL and "L0" in r.order_found
