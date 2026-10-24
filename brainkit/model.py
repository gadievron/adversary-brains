"""Data model for brain/0.1 documents.

Frozen dataclasses mirroring appendix 1 of the paper, plus the YAML loader.
Loader conventions (ADR 011): unquoted YAML dates are stringified to ISO
full-date form before use, so paper-style files load; calendar validity is
checked here (JSON Schema's pattern is shape-only, judge finding JUD-0-3).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Fields appendix 1 marks LOCAL — stripped on publish, rejected in shared files.
LOCAL_BRAIN_FIELDS = frozenset({"fidelity"})
LOCAL_LINK_FIELDS = frozenset({"weight", "observability", "first_seen", "last_seen"})

#: Pattern-key operators (ADR 005). Validated at load so a bad shared file fails
#: with its brain id instead of crashing the evaluation loop (EXP-P-13).
PATTERN_OPS = frozenset({"equals", "contains", "starts", "ends", "not-contains"})


@dataclass(frozen=True)
class Entity:
    """One observable: a node in the brain's graph (appendix-1 entity object)."""

    id: str
    type: str
    value: str | None = None
    pattern: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.pattern is None):
            raise ValueError(f"entity {self.id}: exactly one of value or pattern is required")
        for key in self.pattern or ():
            if key.startswith("x-"):
                continue
            _, _, op_name = key.partition("|")
            if op_name and op_name not in PATTERN_OPS:
                raise ValueError(f"entity {self.id}: unknown pattern operator {op_name!r} in {key!r}")


@dataclass(frozen=True)
class Provenance:
    """Where a link came from: observed | derived | reported, plus its source."""

    kind: str
    source: str


@dataclass(frozen=True)
class Link:
    """One typed relation between two declared entities (appendix-1 link object)."""

    id: str
    type: str  # state | activity
    source: str
    target: str
    relation: str
    role: str  # required | supporting
    volatility: str  # stable | rotating | ephemeral
    provenance: Provenance
    prior: float | None = None


@dataclass(frozen=True)
class Brain:
    """One brain document (publish form — no LOCAL fields)."""

    id: str
    title: str
    status: str
    scale: str
    representation: str
    match: str
    tlp: str
    created: str
    modified: str
    entities: tuple[Entity, ...]
    links: tuple[Link, ...]
    threshold: float = 0.7
    author: str | None = None
    references: tuple[str, ...] = ()
    related: tuple[dict[str, str], ...] = ()

    def entity(self, entity_id: str) -> Entity:
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        raise KeyError(f"brain {self.id}: link references undeclared entity {entity_id!r}")

    @property
    def required_links(self) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.role == "required")

    @property
    def supporting_links(self) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.role == "supporting")


def _stringify_dates(value: Any) -> Any:
    """Recursively turn YAML date objects into ISO strings (ADR 011)."""
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _stringify_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_dates(v) for v in value]
    return value


def _check_calendar_date(field_name: str, value: str) -> None:
    try:
        _dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: not a valid calendar date: {value!r}") from exc


def brain_from_dict(doc: dict[str, Any]) -> Brain:
    """Build a Brain from a parsed document; referential integrity enforced."""
    doc = _stringify_dates(doc)
    for field_name in ("created", "modified"):
        _check_calendar_date(field_name, doc[field_name])

    entities = tuple(
        Entity(id=e["id"], type=e["type"], value=e.get("value"), pattern=e.get("pattern"))
        for e in doc["entities"]
    )
    entity_ids = {entity.id for entity in entities}
    if len(entity_ids) != len(entities):
        raise ValueError(f"brain {doc['id']}: duplicate entity ids")

    links = []
    for raw in doc["links"]:
        for endpoint in (raw["source"], raw["target"]):
            if endpoint not in entity_ids:
                raise ValueError(
                    f"brain {doc['id']}: link {raw['id']} references undeclared entity {endpoint!r}"
                )
        links.append(
            Link(
                id=raw["id"],
                type=raw["type"],
                source=raw["source"],
                target=raw["target"],
                relation=raw["relation"],
                role=raw["role"],
                volatility=raw["volatility"],
                provenance=Provenance(**raw["provenance"]),
                prior=raw.get("prior"),
            )
        )
    link_ids = [link.id for link in links]
    if len(set(link_ids)) != len(link_ids):
        raise ValueError(f"brain {doc['id']}: duplicate link ids")
    if not any(link.role == "required" for link in links):
        # Supporting-only brains make "every required link matched" vacuously
        # true and would fire with no core at all (EXP-P-8; ADR 002 addendum).
        raise ValueError(f"brain {doc['id']}: at least one link with role 'required' is required")

    return Brain(
        id=doc["id"],
        title=doc["title"],
        status=doc["status"],
        scale=doc["scale"],
        representation=doc["representation"],
        match=doc["match"],
        tlp=doc["tlp"],
        created=doc["created"],
        modified=doc["modified"],
        entities=entities,
        links=tuple(links),
        threshold=doc.get("threshold", 0.7),
        author=doc.get("author"),
        references=tuple(doc.get("references", ())),
        related=tuple(doc.get("related", ())),
    )


def load_brain(path: str | Path) -> Brain:
    """Load one brain YAML file (publish form)."""
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a mapping document")
    return brain_from_dict(doc)


def is_study_path(path: str | Path, root: str | Path) -> bool:
    """True if `path` is inside the `study/` namespace under `root`.

    The `brains/study/` tree holds pre-Compile lifecycle-study artifacts (free-text prose `pattern`s,
    not the dict patterns the matcher/screen/schema require). The canonical gates (load_library,
    schema.validate_repo, test_contract) all exclude it via this one predicate — so a future glob of
    the brains/ tree cannot silently re-introduce the `pattern.items()` crash by forgetting the filter.
    """
    return "study" in Path(path).relative_to(Path(root)).parts


def load_library(root: str | Path) -> list[Brain]:
    """Load the canonical runnable brains under a brains/ tree, sorted by path for determinism.

    Excludes the `study/` namespace (see `is_study_path`): those are pre-Compile documentary artifacts,
    kept separate from the canonical library; including them broke screen.py/match on `pattern.items()`.

    Lifecycle `status` (experimental/test/stable/retired) does NOT gate loading: a `status: retired`
    brain still loads and can fire. This is deliberate — the library census must be reproducible from
    the tree, and retirement is a receiver/store decision, not a load-time filter. No shipped brain is
    retired today; if a retired-but-present brain should be excluded from matching, filter on
    `brain.status` at the call site rather than here (silently shrinking the census would break the
    count-based gates).
    """
    root = Path(root)
    return [load_brain(path) for path in sorted(root.rglob("*.yaml")) if not is_study_path(path, root)]
