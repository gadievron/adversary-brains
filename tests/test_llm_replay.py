"""LLM record/replay behavior (ADR 010).

The client's mode logic, request hashing, and replay-miss error are tested with
a temp transcript dir — no endpoint, no committed transcripts needed. The
end-to-end assess-over-a-real-recorded-transcript test SKIPS until a transcript
exists (recorded live with BRAINS_LLM_ENDPOINT set); it activates automatically
once one is committed. This keeps CI green and endpoint-free while refusing to
fabricate a transcript (ADR 010: a transcript is a recorded real model output,
never hand-written).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainkit.llm import TRANSCRIPTS_DIR, LLMClient, extract_json


def test_mode_is_replay_without_endpoint():
    assert LLMClient("assess", endpoint=None).mode == "replay"
    assert LLMClient("assess", endpoint="http://localhost:11434/v1").mode == "live"


def test_replay_miss_is_an_error_never_a_silent_live_call(tmp_path):
    client = LLMClient("assess", endpoint=None, transcripts_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no transcript"):
        client.chat([{"role": "user", "content": "hi"}])


def test_replay_serves_recorded_response(tmp_path):
    client = LLMClient("assess", endpoint=None, transcripts_dir=tmp_path)
    messages = [{"role": "user", "content": "judge this"}]
    # Simulate a previously recorded transcript at the request's hash.
    from brainkit.llm import _request_hash
    h = _request_hash("assess", client.model, messages)
    (tmp_path / "assess").mkdir()
    (tmp_path / "assess" / f"{h}.json").write_text(json.dumps({
        "context": "assess", "model": client.model, "messages": messages,
        "response": '{"verdict":"match","confidence":0.3,"evidence":["wellmess"]}',
    }))
    result = client.chat(messages)
    assert result.from_replay is True
    assert extract_json(result.text)["verdict"] == "match"


def test_request_hash_is_stable_and_message_sensitive():
    from brainkit.llm import _request_hash
    a = _request_hash("assess", "m", [{"role": "user", "content": "x"}])
    b = _request_hash("assess", "m", [{"role": "user", "content": "x"}])
    c = _request_hash("assess", "m", [{"role": "user", "content": "y"}])
    assert a == b and a != c


def test_extract_json_handles_fenced_and_bare():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('prose {"a": 2} more')["a"] == 2
    with pytest.raises(ValueError):
        extract_json("no json here")


# --- end-to-end over a real recorded transcript (skips until one exists) ----

def _recorded_assess_transcripts() -> list[Path]:
    d = TRANSCRIPTS_DIR / "assess"
    return sorted(d.glob("*.json")) if d.exists() else []


@pytest.mark.skipif(not _recorded_assess_transcripts(),
                    reason="no recorded assess transcript yet — record live with "
                           "BRAINS_LLM_ENDPOINT set (ADR 010 forbids fabricating one)")
def test_recorded_verdict_flows_through_grounding():
    """When a real transcript exists, its raw verdict must parse and survive the
    grounding guardrails (evidence-filter + cap) without error."""
    from brainkit.assess import MatchedPiece, ground_verdict
    path = _recorded_assess_transcripts()[0]
    recorded = json.loads(path.read_text())
    raw = extract_json(recorded["response"])
    v = ground_verdict("b", raw, [MatchedPiece("wellmess", 0.9)])
    assert v.verdict in {"match", "partial", "no"}
    assert 0.0 <= v.confidence <= 1.0
