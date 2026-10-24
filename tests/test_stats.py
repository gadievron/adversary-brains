"""Statistics against hand-computed cases (STATISTICS.md; stdlib only)."""
from __future__ import annotations

import math

import pytest

from brainkit.model import brain_from_dict
from brainkit.stats import (
    clopper_pearson, expected_false_discoveries, p_fire, reg_incomplete_beta,
    rule_of_three, separated, separation_applies, surprise_bits,
)

BASE = {
    "id": "00000000-0000-4000-8000-000000000010",
    "title": "t", "status": "experimental", "scale": "campaign",
    "representation": "graph", "match": "likelihood", "tlp": "red",
    "created": "2026-07-10", "modified": "2026-07-10",
}


def brain(links, threshold=0.7):
    return brain_from_dict({
        **BASE,
        "entities": [
            {"id": "e1", "type": "process", "pattern": {"name": "a.exe"}},
            {"id": "e2", "type": "process", "pattern": {"name": "b.exe"}},
        ],
        "links": links, "threshold": threshold,
    })


def link(lid, role, relation="spawned"):
    return {"id": lid, "type": "activity", "source": "e1", "target": "e2",
            "relation": relation, "role": role, "volatility": "stable",
            "provenance": {"kind": "reported", "source": "t"}}


# --- incomplete beta / CI --------------------------------------------------

def test_reg_incomplete_beta_symmetry_and_known_values():
    # I_x(1,1) = x (uniform); I_0.5(a,a) = 0.5 by symmetry.
    assert reg_incomplete_beta(1, 1, 0.37) == pytest.approx(0.37, abs=1e-9)
    assert reg_incomplete_beta(3, 3, 0.5) == pytest.approx(0.5, abs=1e-9)
    assert reg_incomplete_beta(2, 5, 0.5) == pytest.approx(1 - reg_incomplete_beta(5, 2, 0.5), abs=1e-9)


def test_clopper_pearson_zero_and_full():
    # 0/10: lower bound 0, upper = 1-(alpha/2)^(1/n) region; known CP upper ≈ 0.3085.
    ci = clopper_pearson(0, 10)
    assert ci.low == 0.0 and ci.high == pytest.approx(0.3085, abs=1e-3)
    # 10/10: upper 1, lower ≈ 0.6915 (symmetry of the 0/10 case).
    ci = clopper_pearson(10, 10)
    assert ci.high == 1.0 and ci.low == pytest.approx(0.6915, abs=1e-3)


def test_clopper_pearson_contains_point_estimate():
    ci = clopper_pearson(7, 20)
    assert ci.low < 0.35 < ci.high


def test_clopper_pearson_zero_trials_is_maximally_uncertain():
    ci = clopper_pearson(0, 0)
    assert ci.low == 0.0 and ci.high == 1.0


def test_rule_of_three():
    assert rule_of_three(30) == pytest.approx(0.1)
    assert rule_of_three(300) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        rule_of_three(0)


# --- p_fire ----------------------------------------------------------------

def test_p_fire_required_core_is_product():
    b = brain([link("L1", "required"), link("L2", "required", relation="wrote")])
    p = p_fire(b, {"L1": 0.01, "L2": 0.1})
    assert p == pytest.approx(0.001)


def test_p_fire_single_required_link():
    b = brain([link("L1", "required")])
    assert p_fire(b, {"L1": 1 / 51}) == pytest.approx(1 / 51)


def test_p_fire_with_supporting_union():
    # required L1 (rate .01); two supporting equal-weight links; threshold .7 =>
    # need >=1 of 2 present is NOT enough (share .5 < .7); need both (share 1.0).
    b = brain([link("L1", "required"),
               link("L2", "supporting", relation="wrote"),
               link("L3", "supporting", relation="connected-to")], threshold=0.7)
    rates = {"L1": 0.01, "L2": 0.2, "L3": 0.2}
    # equal surprise weights -> only the both-present subset reaches share 1.0
    expected = 0.01 * (0.2 * 0.2)
    assert p_fire(b, rates) == pytest.approx(expected)


def test_surprise_bits():
    assert surprise_bits(0.25) == pytest.approx(2.0)
    assert surprise_bits(0.0) == math.inf


# --- separation criterion (pilot amendment 1) ------------------------------

def test_separation_only_for_multi_link_cores():
    single = brain([link("L1", "required")])
    multi = brain([link("L1", "required"), link("L2", "required", relation="wrote")])
    assert separation_applies(single) is False
    assert separation_applies(multi) is True
    # single-link: criterion returns None (report as brittleness contrast)
    assert separated(single, 1 / 51, evaluations=300) is None
    # multi-link: 2.5e-7 passes 1/(100*300)=3.3e-5; 1/51 would fail
    assert separated(multi, 2.5e-7, evaluations=300) is True
    assert separated(multi, 1 / 51, evaluations=300) is False


def test_expected_false_discoveries_is_sum():
    assert expected_false_discoveries([0.01, 0.02, 0.03]) == pytest.approx(0.06)


def test_p_fire_refuses_too_many_supporting_links():
    """AUD-2-1: guard against O(2^n) blowup — refuse loudly past the cap."""
    from brainkit.stats import MAX_SUPPORTING_FOR_EXACT_P_FIRE
    n = MAX_SUPPORTING_FOR_EXACT_P_FIRE + 1
    entities = [{"id": f"e{i}", "type": "process", "pattern": {"name": f"p{i}.exe"}}
                for i in range(n + 1)]
    links = [{"id": "R", "type": "activity", "source": "e0", "target": "e1",
              "relation": "spawned", "role": "required", "volatility": "stable",
              "provenance": {"kind": "reported", "source": "t"}}]
    links += [{"id": f"S{i}", "type": "activity", "source": "e0", "target": f"e{i+1}",
               "relation": f"rel{i}", "role": "supporting", "volatility": "stable",
               "provenance": {"kind": "reported", "source": "t"}} for i in range(n)]
    b = brain_from_dict({**BASE, "entities": entities, "links": links})
    rates = {"R": 0.1, **{f"S{i}": 0.3 for i in range(n)}}
    with pytest.raises(ValueError, match="over the guard"):
        p_fire(b, rates)


def test_bootstrap_p_fire_ci_brackets_point_and_widens_with_spread():
    """E1: p_fire CI over calibration realizations brackets the point estimate."""
    from brainkit.stats import bootstrap_p_fire_ci
    b = brain([link("L1", "required"), link("L2", "required", relation="wrote")])
    # 5 realizations with slightly varying rates
    reals = [{"L1": r1, "L2": r2} for r1, r2 in
             [(0.01, 0.10), (0.012, 0.09), (0.008, 0.11), (0.011, 0.10), (0.009, 0.095)]]
    point, lo, hi = bootstrap_p_fire_ci(reals, b)
    assert lo <= point <= hi
    assert lo < hi  # nonzero spread across realizations
    # identical realizations -> zero-width interval at the point
    same = [{"L1": 0.01, "L2": 0.1}] * 4
    p, l2, h2 = bootstrap_p_fire_ci(same, b)
    assert l2 == h2 == p
