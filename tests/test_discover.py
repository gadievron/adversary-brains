"""E2 Discover — bottom-up propose + watch half (H17, METHODOLOGY-v3).

Instantiability dry-run: propose fires on a recurring residual and NOT on noise;
the proposed brain then fires (watch) when the rest of the pattern arrives.
"""
from __future__ import annotations

from brainkit.discover import cluster_residual, propose, signature, watch


def _ev(relation, src_name, tgt_value, at="2026-07-01T00:00:00Z"):
    return {"time": at, "relation": relation,
            "source": {"type": "process", "name": src_name},
            "target": {"type": "domain-name", "value": tgt_value}}


def test_recurring_residual_is_proposed_noise_is_not():
    # a recurring structure (same beacon 4x) + one-off noise
    recurring = [_ev("connected-to", "novel-implant.exe", "evil-c2.example") for _ in range(4)]
    noise = [_ev("connected-to", f"proc{i}.exe", f"rand-{i}.example") for i in range(10)]
    proposed = propose(recurring + noise, min_support=3)
    assert len(proposed) == 1
    assert proposed[0].status == "experimental"          # born unconfirmed (bootstrap)
    assert "novel-implant.exe" in proposed[0].title

    # pure noise -> nothing proposed
    assert propose(noise, min_support=3) == []


def test_below_support_is_not_proposed():
    residual = [_ev("connected-to", "impl.exe", "c2.example") for _ in range(2)]  # < 3
    assert propose(residual, min_support=3) == []


def test_watch_half_fires_when_the_rest_arrives():
    residual = [_ev("connected-to", "novel-implant.exe", "evil-c2.example") for _ in range(4)]
    (brain,) = propose(residual, min_support=3)
    # the rest of the pattern arrives: a NEW event of the same structure -> FIRES
    later = [_ev("connected-to", "novel-implant.exe", "evil-c2.example",
                 at="2026-07-05T09:00:00Z")]
    assert watch(brain, later) is True
    # an unrelated event does NOT fire the proposed brain
    assert watch(brain, [_ev("connected-to", "chrome.exe", "popular.example")]) is False


def test_signature_is_structural():
    a = signature(_ev("connected-to", "x.exe", "c2.example"))
    b = signature(_ev("connected-to", "x.exe", "c2.example", at="different-time"))
    assert a == b                                        # time is not part of the structure
    assert signature(_ev("spawned", "x.exe", "c2.example")) != a  # relation matters
    counts = cluster_residual([_ev("connected-to", "x.exe", "c2.example")] * 3)
    assert counts[a] == 3
