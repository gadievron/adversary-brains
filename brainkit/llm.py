"""OpenAI-compatible LLM client with record/replay (ADR 010, divergence D8).

The model is in the main path (assess verdicts, extract's librarian pass), but
reproducibility does not depend on it: every request is keyed by a stable hash
and, in ``live`` mode, its response is RECORDED to a committed transcript; in
``replay`` mode (the default when no endpoint is configured) the response is
served from that transcript, so reviewers and CI reproduce results with no
model. A replay cache miss is an error, never a silent live call. Hand-written
responses are forbidden — a transcript is a recorded real model output.

Endpoint: any OpenAI-compatible `/v1/chat/completions` (env BRAINS_LLM_ENDPOINT,
e.g. a local ollama at http://localhost:11434/v1); model via BRAINS_LLM_MODEL.
Temperature is pinned to 0. Uses only stdlib urllib.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "llm-transcripts"
ENDPOINT_ENV = "BRAINS_LLM_ENDPOINT"
MODEL_ENV = "BRAINS_LLM_MODEL"
DEFAULT_MODEL = "qwen2.5:1.5b"  # the model the committed transcripts were recorded with;
#                                 replay-by-hash matches them when no BRAINS_LLM_MODEL is set


def _request_hash(context: str, model: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"context": context, "model": model, "messages": messages},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class LLMResult:
    """One model turn: the raw text, plus provenance for the transcript."""

    text: str
    model: str
    from_replay: bool


class LLMClient:
    """Record (live) or replay (default) an OpenAI-compatible chat model."""

    def __init__(self, context: str, endpoint: str | None = None, model: str | None = None,
                 transcripts_dir: Path = TRANSCRIPTS_DIR):
        self.context = context
        self.endpoint = endpoint if endpoint is not None else os.environ.get(ENDPOINT_ENV)
        self.model = model or os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        self.dir = transcripts_dir / context

    @property
    def mode(self) -> str:
        return "live" if self.endpoint else "replay"

    def _transcript_path(self, req_hash: str) -> Path:
        return self.dir / f"{req_hash}.json"

    def chat(self, messages: list[dict[str, str]], *, force_json: bool = False) -> LLMResult:
        """Return the model's reply — recording it live, or replaying it."""
        req_hash = _request_hash(self.context, self.model, messages)
        path = self._transcript_path(req_hash)
        if self.endpoint:
            text, served_model = self._call_live(messages, force_json)
            self._record(path, messages, text, served_model)
            return LLMResult(text=text, model=served_model, from_replay=False)
        if not path.exists():
            raise FileNotFoundError(
                f"no transcript for {self.context}/{req_hash} and no {ENDPOINT_ENV} set: "
                "record it live (set the endpoint) before replaying")
        recorded = json.loads(path.read_text())
        return LLMResult(text=recorded["response"], model=recorded["model"], from_replay=True)

    def _call_live(self, messages: list[dict[str, str]], force_json: bool) -> tuple[str, str]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if force_json:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode()
        url = self.endpoint.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        text = payload["choices"][0]["message"]["content"]
        # Record the model the SERVER reports (ADR 010: "model name+version"),
        # which may be more specific than the requested name (EXP-3-3).
        served_model = payload.get("model") or self.model
        return text, served_model

    def _record(self, path: Path, messages: list[dict[str, str]], text: str,
                served_model: str) -> None:
        from .store import atomic_write  # local import: store has no llm dependency
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps({
            "context": self.context,
            "requested_model": self.model,
            "model": served_model,
            "messages": messages,
            "response": text,
        }, indent=2, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from a model reply (handles fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model reply: {text[:120]!r}")
    return json.loads(text[start:end + 1])
