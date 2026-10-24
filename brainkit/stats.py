"""Statistics for the confirmatory experiments (STATISTICS.md, stdlib only).

Three tools, each unit-tested against hand-computed cases:
  * ``clopper_pearson`` — exact binomial CI via the beta-quantile relation,
    with a stdlib regularized-incomplete-beta and its inverse (no scipy).
  * ``rule_of_three`` — one-sided 95% upper bound for a zero-event outcome.
  * ``p_fire`` / ``surprise`` — the null-model tail probability of a brain
    firing (STATISTICS.md, corrected per EXP-P-1/EXP-0-4): the required core
    is a product of rates; supporting links contribute the union over
    threshold-reaching subsets. Uses UNCAPPED rates (the cap exception).

The separation criterion (p_fire <= 1/(100*E)) applies only to brains whose
required core has >= 2 links (pilot amendment 1); ``separation_applies`` and
``separated`` encode that so callers cannot misapply it to single-link brains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from .model import Brain

#: p_fire enumerates 2^(#supporting) subsets; refuse past this rather than hang
#: (AUD-2-1 / EXP-2-7). The shipped library has <=1 supporting link per brain;
#: a union-bound approximation for large supporting sets is the upgrade path.
MAX_SUPPORTING_FOR_EXACT_P_FIRE = 20

# --------------------------------------------------------------------------- #
# Regularized incomplete beta (Lentz's continued fraction) + inverse bisection #
# --------------------------------------------------------------------------- #

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function, x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b) in x, by bisection (monotone in x)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if reg_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Public estimators                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval."""

    low: float
    high: float


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Exact (Clopper-Pearson) binomial CI for a proportion."""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return Interval(0.0, 1.0)
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else _beta_quantile(alpha / 2, successes, trials - successes + 1)
    high = 1.0 if successes == trials else _beta_quantile(1 - alpha / 2, successes + 1, trials - successes)
    return Interval(low, high)


def rule_of_three(trials: int) -> float:
    """One-sided 95% upper bound on a rate given 0 events in ``trials`` (3/n)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    return 3.0 / trials


def p_fire(brain: Brain, rates: Mapping[str, float], threshold: float | None = None) -> float:
    """Exact null tail probability that ``brain`` fires in one evaluation unit.

    Under the independence null (each link present at its background rate):
    every required link must be present, AND the weight-weighted share of
    present supporting links must reach the threshold. Required core is a
    product; the supporting contribution is the union over subsets whose
    weighted share reaches threshold. Weights here are surprise weights
    (-log2 rate, UNCAPPED), consistent with STATISTICS.md.
    """
    theta = brain.threshold if threshold is None else threshold
    required = brain.required_links
    supporting = brain.supporting_links
    p_required = 1.0
    for link in required:
        p_required *= rates[link.id]

    if not supporting:
        return p_required

    if len(supporting) > MAX_SUPPORTING_FOR_EXACT_P_FIRE:
        raise ValueError(
            f"brain {brain.id}: exact p_fire enumerates 2^{len(supporting)} supporting "
            f"subsets, over the guard of {MAX_SUPPORTING_FOR_EXACT_P_FIRE}; a union-bound "
            "approximation is the upgrade path (STATISTICS.md)")

    weights = {link.id: -math.log2(rates[link.id]) for link in supporting}
    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return p_required  # zero-information supporting set -> fires on core (ADR 002)

    p_support = 0.0
    n = len(supporting)
    for k in range(n + 1):
        for present in combinations(supporting, k):
            present_ids = {link.id for link in present}
            share = sum(weights[link.id] for link in present) / total_weight
            if share < theta:
                continue
            term = 1.0
            for link in supporting:
                r = rates[link.id]
                term *= r if link.id in present_ids else (1.0 - r)
            p_support += term
    return p_required * p_support


def surprise_bits(p: float) -> float:
    """Self-information of a fire probability, in bits."""
    if p <= 0.0:
        return math.inf
    return -math.log2(p)


def separation_applies(brain: Brain) -> bool:
    """H7's separation criterion is claimed only for >=2-required-link cores
    (pilot amendment 1); single-link brains are the brittleness contrast."""
    return len(brain.required_links) >= 2


def separated(brain: Brain, p: float, evaluations: int, factor: int = 100) -> bool | None:
    """Is this fire separated from the null? None when the criterion does not
    apply (single-link core) — callers must report those as the contrast case."""
    if not separation_applies(brain):
        return None
    return p <= 1.0 / (factor * evaluations)


def expected_false_discoveries(p_fires: Sequence[float]) -> float:
    """Sum of per-evaluation fire probabilities = expected null fires (bound)."""
    return float(sum(p_fires))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence, q in [0,1]."""
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def bootstrap_p_fire_ci(
    per_realization_rates: Sequence[Mapping[str, float]],
    brain: Brain,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile CI on p_fire over independent calibration realizations (v2 §5).

    ``per_realization_rates`` is a list of {link_id -> rate} dicts, one per
    calibration realization. Returns (point_over_pooled_mean, ci_low, ci_high):
    the point uses the mean rate per link; the interval is the percentile
    interval of p_fire computed on each realization's rates. This attaches the
    uncertainty the Stage-4 audit found missing (scattered's ×1.16 separation
    margin was inside plausible calibration noise).

    Precondition: every realization dict carries the SAME link-id set (every link
    the brain scores). This is in-contract — the caller builds each realization
    from the same brain's ``background_stats`` (exp8), so the key sets are
    identical. A ragged set (a link id present in one realization, absent in
    another) is out of contract and would KeyError here and in ``p_fire``;
    it is deliberately NOT tolerated, because a missing per-link rate is not the
    same statistical object as a measured zero rate — silently substituting one
    would bias the mean. The constant-key contract is the correct guard.
    """
    if not per_realization_rates:
        raise ValueError("need at least one calibration realization")
    link_ids = set().union(*(r.keys() for r in per_realization_rates))
    mean_rates = {lid: sum(r[lid] for r in per_realization_rates) / len(per_realization_rates)
                  for lid in link_ids}
    point = p_fire(brain, mean_rates)
    samples = sorted(p_fire(brain, r) for r in per_realization_rates)
    alpha = 1.0 - confidence
    return point, percentile(samples, alpha / 2), percentile(samples, 1 - alpha / 2)
