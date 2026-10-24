"""H5 guardrails, tested deterministically — no endpoint required.

The paper's constrained-LLM claims are enforced OUTSIDE the model, so these
tests pin the mechanism directly with synthetic "raw model verdicts" as input.
The live-recorded transcripts (test_llm_replay.py) then show a real model's
raw output flowing through this same machinery.
"""
from __future__ import annotations

from brainkit.assess import (
    CONFIDENCE_CAP, MatchedPiece, Verdict, ground_verdict, synthesize,
)
from brainkit.extract import (
    build_draft, extract_iocs, extract_technique_ids, refang, validate_technique_ids,
)

RARE = [MatchedPiece("wellmess", 0.9)]
COMMODITY = [MatchedPiece("cobalt strike", 0.05), MatchedPiece("mimikatz", 0.03)]


# --- assess grounding ------------------------------------------------------

def test_ungrounded_evidence_discarded():
    raw = {"verdict": "match", "confidence": 0.3,
           "evidence": ["uses WellMess backdoor", "APT29 always targets us on Fridays"],
           "reasoning": "x"}
    v = ground_verdict("b1", raw, RARE)
    assert v.evidence == ("uses WellMess backdoor",)              # grounded kept
    assert v.dropped_evidence == ("APT29 always targets us on Fridays",)  # ungrounded dropped


def test_confidence_capped_without_rare_evidence():
    raw = {"verdict": "match", "confidence": 0.95,
           "evidence": ["cobalt strike seen", "mimikatz seen"], "reasoning": "x"}
    v = ground_verdict("b1", raw, COMMODITY)
    assert v.confidence == CONFIDENCE_CAP and v.confidence_capped is True


def test_confidence_not_capped_with_rare_evidence():
    raw = {"verdict": "match", "confidence": 0.95, "evidence": ["wellmess"], "reasoning": "x"}
    v = ground_verdict("b1", raw, RARE)
    assert v.confidence == 0.95 and v.confidence_capped is False


def test_confidence_clamped_to_unit_interval():
    assert ground_verdict("b", {"confidence": 1.7, "evidence": ["wellmess"]}, RARE).confidence == 1.0
    assert ground_verdict("b", {"confidence": -0.2, "evidence": []}, RARE).confidence == 0.0


def test_bad_verdict_value_defaults_to_no():
    assert ground_verdict("b", {"verdict": "definitely", "evidence": []}, RARE).verdict == "no"


def test_all_evidence_ungrounded_leaves_none():
    raw = {"verdict": "match", "confidence": 0.9, "evidence": ["made-up thing"]}
    v = ground_verdict("b1", raw, RARE)
    assert v.evidence == () and v.dropped_evidence == ("made-up thing",)


def test_short_token_does_not_ground_via_substring():
    """EXP-3-1: a short matched token ('ip') must NOT ground unrelated text
    ('description', 'multiple') by mere substring containment."""
    pieces = [MatchedPiece("ip", 0.9)]
    raw = {"verdict": "match", "confidence": 0.3,
           "evidence": ["see the description", "multiple hosts", "the ip 203.0.113.7"]}
    v = ground_verdict("b1", raw, pieces)
    assert v.evidence == ("the ip 203.0.113.7",)  # only the delimited-token hit survives
    assert set(v.dropped_evidence) == {"see the description", "multiple hosts"}


def test_punctuated_and_multiword_tokens_ground_correctly():
    """Delimited match still accepts real tokens with dots/spaces at word edges."""
    pieces = [MatchedPiece("203.0.113.7", 0.9), MatchedPiece("cobalt strike", 0.05),
              MatchedPiece(".akira", 0.9)]
    raw = {"verdict": "match", "confidence": 0.3, "evidence": [
        "beacon to 203.0.113.7 seen", "dropped cobalt strike loader", "wrote .akira files"]}
    v = ground_verdict("b1", raw, pieces)
    assert len(v.evidence) == 3 and v.dropped_evidence == ()


def test_synthesize_ranks_match_over_partial_over_no():
    vs = [
        Verdict("b-no", "no", 0.9, (), ""),
        Verdict("b-match-lo", "match", 0.3, (), ""),
        Verdict("b-match-hi", "match", 0.8, (), ""),
        Verdict("b-partial", "partial", 0.99, (), ""),
    ]
    assert [v.brain_id for v in synthesize(vs)] == \
        ["b-match-hi", "b-match-lo", "b-partial", "b-no"]


# --- extract guardrails ----------------------------------------------------

def test_refang():
    assert refang("hxxps://evil[.]com") == "https://evil.com"
    assert refang("1.2.3[.]4") == "1.2.3.4"


def test_extract_iocs_types_and_refang():
    text = ("beacon to hxxp://bad[.]example/x from 203.0.113.7; "
            "hash 44d88612fea8a8f36de82e1278abb02f")
    iocs = extract_iocs(text)
    kinds = {i["type"] for i in iocs}
    assert "url" in kinds and "ipv4" in kinds and "md5" in kinds
    assert any(i["value"] == "203.0.113.7" for i in iocs)
    assert any(i["value"].startswith("http://bad.example") for i in iocs)


def test_sha256_not_split_into_shorter_hashes():
    sha = "a" * 64
    iocs = extract_iocs(f"dropper {sha} observed")
    hash_vals = [i for i in iocs if i["type"] in ("md5", "sha1", "sha256")]
    assert hash_vals == [{"type": "sha256", "value": sha}]


def test_invalid_technique_id_dropped():
    valid, dropped = validate_technique_ids(["T1059.001", "T9999.999", "T1105"])
    assert valid == ["T1059.001", "T1105"] and dropped == ["T9999.999"]


def test_build_draft_drops_hallucinated_id_and_is_idempotent():
    report = ("APT used PowerShell T1059.001 and a fabricated T9999.999; "
              "c2 at evil.example, ip 198.51.100.9")
    r1 = build_draft(report, "Draft X")
    assert "T9999.999" in r1.dropped_technique_ids
    assert "T1059.001" in r1.valid_technique_ids and "T9999.999" not in r1.valid_technique_ids
    # idempotent: same input -> identical draft
    r2 = build_draft(report, "Draft X")
    assert r1.draft == r2.draft
    # the hallucinated id appears nowhere in the draft entities
    assert not any(e.get("value") == "T9999.999" for e in r1.draft["entities"])
