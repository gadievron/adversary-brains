"""Coverage overlay: which links this environment can observe (D6).

Observability is LOCAL (never in a shared brain file); it lives in a coverage
map — the *result* of the paper's Coverage-map operation, an environment fact.
A brain's required link marked invisible SUSPENDS the brain rather than failing
it (fed to ``match_brain`` as its ``invisible`` set).

File shape (YAML):

    default: visible            # optional; visible if omitted
    brains:
      <brain-id>:
        <link-id>: invisible
        <link-id>: visible
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import Brain
from .store import read_yaml

_VALID = {"visible", "invisible"}


@dataclass(frozen=True)
class Coverage:
    """A parsed coverage map; ``invisible_links`` is what ``match_brain`` wants."""

    default: str = "visible"
    per_brain: dict[str, dict[str, str]] = field(default_factory=dict)

    def invisible_links(self, brain_id: str) -> frozenset[str]:
        """Explicit ``invisible`` marks for a brain (id form).

        Correct when ``default`` is ``visible`` (the common case: an overlay
        names the few links a sensor cannot see). When ``default`` is
        ``invisible`` this cannot enumerate the unmarked links — use
        :meth:`invisible_links_for`, which is given the brain, instead.
        """
        marks = self.per_brain.get(brain_id, {})
        return frozenset(lid for lid, mark in marks.items() if mark == "invisible")

    def invisible_links_for(self, brain: Brain) -> frozenset[str]:
        """The brain's link ids this environment cannot observe, honoring
        ``default`` — the set ``match_brain`` consumes. With ``default:
        visible`` this equals :meth:`invisible_links`; with ``default:
        invisible`` every link not explicitly marked ``visible`` is included."""
        return frozenset(
            link.id for link in brain.links
            if self.visibility_of(brain.id, link.id) == "invisible"
        )

    def visibility_of(self, brain_id: str, link_id: str) -> str:
        return self.per_brain.get(brain_id, {}).get(link_id, self.default)


def load_coverage(path: str | Path) -> Coverage:
    """Load and validate a coverage overlay file."""
    doc = read_yaml(path) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: coverage map must be a mapping")
    default = doc.get("default", "visible")
    if default not in _VALID:
        raise ValueError(f"{path}: default must be one of {sorted(_VALID)}, got {default!r}")
    brains = doc.get("brains") or {}
    if not isinstance(brains, dict):
        raise ValueError(f"{path}: 'brains' must be a mapping of brain-id -> marks, not {type(brains).__name__}")
    per_brain: dict[str, dict[str, str]] = {}
    for brain_id, marks in brains.items():
        if not isinstance(marks, dict):
            raise ValueError(f"{path}: brains.{brain_id} must be a mapping of link-id -> mark")
        for link_id, mark in marks.items():
            if mark not in _VALID:
                raise ValueError(
                    f"{path}: brains.{brain_id}.{link_id} must be one of {sorted(_VALID)}, "
                    f"got {mark!r}")
        per_brain[brain_id] = dict(marks)
    return Coverage(default=default, per_brain=per_brain)


def empty() -> Coverage:
    """The all-visible overlay (no coverage file supplied)."""
    return Coverage()
