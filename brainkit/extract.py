"""Extract: build a draft brain from a report (appendix 3 ingestion, guardrailed).

Deterministic slice (the guardrails, testable without a model):
  * ``extract_iocs`` — refang + regexes for ipv4/domain/url/md5/sha1/sha256;
  * ``extract_technique_ids`` + ``validate_technique_ids`` — pull T#### ids and
    DROP any absent from the ATT&CK index (appendix 3 step 3: "validate every
    technique id against the bundle; drop failures"), so a hallucinated id
    never enters the draft;
  * ``build_draft`` — assemble an experimental-status brain skeleton (entities
    only: IOCs as concrete-value entities, validated technique ids as
    x-mitre-attack-pattern entities); idempotent (dedupe by value). Links,
    roles, and volatility are for a human/analyst to add — the draft is a
    proposal, not a fireable brain.

Optional model slice (``librarian_pass``): an LLM proposes candidate actors /
techniques / tooling from prose; its technique ids pass the SAME validation
gate. Off unless a client is supplied; never required by a test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LLMClient, extract_json
from .store import read_yaml

TECHNIQUES_PATH = Path(__file__).resolve().parent.parent / "data" / "techniques.json"

# ATT&CK technique ids: T#### with an optional .### sub-id (EXP-3-4: matches are
# case-sensitive and a malformed sub-id like T1059.1 truncates to the base id,
# which the validation gate then accepts/drops on its own merits — under-
# extraction, the safe direction; the gate is the real guard).
_TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_IOC_RES = {
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "url": re.compile(r"\bhttps?://[^\s)>\]]+", re.I),
    "domain": re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I),
}


_HXXP_RE = re.compile(r"\bhxxp(s?)(?=\[?:)", re.I)


def refang(text: str) -> str:
    """Undo common IOC defanging (hxxp[s]://, [.], (dot), [:]).

    `hxxp` is only rewritten when it begins a scheme (followed by an optional
    `[` then `:`), so an ordinary word like "thxxput" is left alone (EXP-3-6).
    """
    text = _HXXP_RE.sub(r"http\1", text)
    return (text.replace("[.]", ".").replace("(.)", ".")
            .replace("[dot]", ".").replace("(dot)", ".").replace("[:]", ":"))


def load_technique_index(path: Path = TECHNIQUES_PATH) -> dict[str, str]:
    index = read_yaml(path)  # JSON is valid YAML
    return {k: v for k, v in index.items() if not k.startswith("_")}


@dataclass(frozen=True)
class ExtractResult:
    """A draft brain plus a record of what the guardrails dropped."""

    iocs: list[dict[str, str]]
    valid_technique_ids: list[str]
    dropped_technique_ids: list[str]
    draft: dict[str, Any]


def extract_iocs(text: str) -> list[dict[str, str]]:
    """Deterministic IOC extraction (refang first). Longer hashes win overlaps.

    PoC limits (EXP-3-5), acceptable because a draft is a proposal a human
    curates, not a fireable brain: the domain regex also matches filename-like
    tokens (``malware.exe``, ``invoice.pdf``), and the ipv4 regex does not
    range-check octets (``999.999.999.999`` extracts). URL hosts are not
    double-counted (URLs claim their span before the domain pass).
    """
    text = refang(text)
    found: dict[tuple[str, str], dict[str, str]] = {}
    claimed_spans: list[tuple[int, int]] = []
    # Hashes first (longest to shortest) so a sha256 isn't split into md5/sha1.
    for ioc_type in ("sha256", "sha1", "md5", "url", "ipv4", "domain"):
        for m in _IOC_RES[ioc_type].finditer(text):
            span = (m.start(), m.end())
            if any(s <= span[0] < e or s < span[1] <= e for s, e in claimed_spans):
                continue
            value = m.group(0)
            key = (ioc_type, value.lower())
            if key not in found:
                found[key] = {"type": ioc_type, "value": value}
                claimed_spans.append(span)
    return list(found.values())


def extract_technique_ids(text: str) -> list[str]:
    """All ATT&CK-shaped ids in the text, order-preserving and de-duplicated."""
    seen: dict[str, None] = {}
    for m in _TECH_RE.finditer(text):
        seen.setdefault(m.group(0), None)
    return list(seen)


def validate_technique_ids(ids: list[str], index: dict[str, str] | None = None
                           ) -> tuple[list[str], list[str]]:
    """Split ids into (valid, dropped) against the ATT&CK index (the gate)."""
    index = index if index is not None else load_technique_index()
    valid = [tid for tid in ids if tid in index]
    dropped = [tid for tid in ids if tid not in index]
    return valid, dropped


def build_draft(report_text: str, title: str, index: dict[str, str] | None = None
                ) -> ExtractResult:
    """Assemble a draft experimental brain from a report (deterministic path)."""
    index = index if index is not None else load_technique_index()
    iocs = extract_iocs(report_text)
    valid_ids, dropped_ids = validate_technique_ids(extract_technique_ids(report_text), index)

    stix_type_by_ioc = {"ipv4": "ipv4-addr", "domain": "domain-name", "url": "url",
                        "md5": "file", "sha1": "file", "sha256": "file"}
    entities: list[dict[str, Any]] = []
    for i, ioc in enumerate(iocs, start=1):
        stix_type = stix_type_by_ioc[ioc["type"]]
        entities.append({"id": f"e{i}", "type": stix_type, "value": ioc["value"]})
    for j, tid in enumerate(valid_ids, start=1):
        entities.append({"id": f"t{j}", "type": "x-mitre-attack-pattern",
                         "value": tid, "x-name": index[tid]})

    draft = {
        "schema": "brain/0.1",
        "title": title,
        "status": "experimental",
        "scale": "campaign",
        "representation": "graph",
        "match": "likelihood",
        "tlp": "amber",
        "entities": entities,
        "technique_ids": valid_ids,
    }
    return ExtractResult(iocs=iocs, valid_technique_ids=valid_ids,
                         dropped_technique_ids=dropped_ids, draft=draft)


def librarian_pass(client: LLMClient, report_text: str) -> dict[str, Any]:
    """Optional model pass proposing actors/techniques/tooling; ids still gated."""
    messages = [
        {"role": "system", "content":
            "Extract threat-intel structure from the report as strict JSON with keys "
            "actors (list of strings), technique_ids (list of ATT&CK ids like T1059.001), "
            "tooling (list of strings). Do not invent ids."},
        {"role": "user", "content": report_text},
    ]
    raw = extract_json(client.chat(messages, force_json=True).text)
    valid, dropped = validate_technique_ids([str(t) for t in raw.get("technique_ids", [])])
    return {
        "actors": [str(a) for a in raw.get("actors", [])],
        "tooling": [str(t) for t in raw.get("tooling", [])],
        "technique_ids": valid,
        "dropped_technique_ids": dropped,
    }
