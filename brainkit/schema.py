"""Repository validation: the appendix-1 CI contract.

Four checks, exactly as the paper's appendix-1 CI contract
specifies — schema validation, LOCAL-field rejection, id uniqueness,
related-id resolution — plus entity-ref integrity (which JSON Schema cannot
express and appendix 1 assigns to the contract). LOCAL fields are already rejected by
the schema's additionalProperties:false; they are ALSO checked explicitly so
validation output names the violation ("LOCAL field 'weight' present on link L2")
instead of emitting a generic schema error.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .model import LOCAL_BRAIN_FIELDS, LOCAL_LINK_FIELDS, _stringify_dates, is_study_path

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "brain-0.1.schema.json"


@dataclass(frozen=True)
class ValidationError:
    """One named validation failure, printable as `path: [check] message`."""

    path: str
    check: str  # schema | local-fields | id-uniqueness | related-resolution | entity-refs | extension | parse
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.check}] {self.message}"


def load_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text())


def check_calendar_dates(doc: dict[str, Any], path: str) -> list[ValidationError]:
    """Calendar-validity for created/modified so CI-green implies loadable.

    The schema's date pattern is shape-only (ADR 011 / JUD-P-3), so
    `9999-99-99` passes the regex but fails `brain_from_dict` at load. Running
    the same `date.fromisoformat` check here makes the CI contract subsume
    loadability (EA-5), with a named error instead of a downstream crash.
    """
    errors: list[ValidationError] = []
    for field in ("created", "modified"):
        value = doc.get(field)
        if not isinstance(value, str):
            continue  # a missing/non-string date is the schema check's job
        try:
            _dt.date.fromisoformat(value)
        except ValueError:
            errors.append(ValidationError(
                path, "schema", f"{field}: not a valid calendar date: {value!r}"))
    return errors


def check_local_fields(doc: dict[str, Any], path: str) -> list[ValidationError]:
    """Named LOCAL-field rejection (appendix 1: 'A validator must reject a
    shared file that contains them')."""
    errors: list[ValidationError] = []
    for field in sorted(LOCAL_BRAIN_FIELDS & doc.keys()):
        errors.append(ValidationError(path, "local-fields", f"LOCAL field '{field}' present on brain"))
    for link in doc.get("links", []):
        if isinstance(link, dict):
            for field in sorted(LOCAL_LINK_FIELDS & link.keys()):
                errors.append(ValidationError(
                    path, "local-fields",
                    f"LOCAL field '{field}' present on link {link.get('id', '?')}"))
    return errors


def check_entity_refs(doc: dict[str, Any], path: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    entities = [e for e in doc.get("entities", []) if isinstance(e, dict)]
    entity_ids = {e.get("id") for e in entities}

    # Duplicate local ids make source/target resolution ambiguous (EA-4).
    for kind, items in (("entity", entities), ("link", doc.get("links", []))):
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if item_id in seen:
                errors.append(ValidationError(
                    path, "entity-refs", f"duplicate {kind} id {item_id!r} within file"))
            elif item_id is not None:
                seen.add(item_id)

    for link in doc.get("links", []):
        if not isinstance(link, dict):
            continue
        for endpoint in ("source", "target"):
            ref = link.get(endpoint)
            if ref not in entity_ids:
                errors.append(ValidationError(
                    path, "entity-refs",
                    f"link {link.get('id', '?')} {endpoint} references undeclared entity {ref!r}"))
    return errors


def validate_file(
    path: str | Path, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[ValidationError]]:
    """Validate one brain file; returns (parsed doc or None, named errors)."""
    path_str = str(path)
    try:
        doc = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        return None, [ValidationError(path_str, "parse", f"not parseable YAML: {exc}")]
    if not isinstance(doc, dict):
        return None, [ValidationError(path_str, "parse", "not a mapping document")]
    doc = _stringify_dates(doc)

    errors = check_local_fields(doc, path_str)  # named errors first, most readable
    validator = jsonschema.Draft202012Validator(schema)
    for schema_error in sorted(validator.iter_errors(doc), key=str):
        location = "/".join(str(part) for part in schema_error.absolute_path) or "<root>"
        errors.append(ValidationError(path_str, "schema", f"{location}: {schema_error.message}"))
    errors.extend(check_calendar_dates(doc, path_str))
    errors.extend(check_entity_refs(doc, path_str))
    return doc, errors


def validate_repo(
    root: str | Path, schema_path: str | Path = DEFAULT_SCHEMA_PATH
) -> list[ValidationError]:
    """Validate every brain under a brains/ tree + the cross-file checks."""
    schema = load_schema(schema_path)
    errors: list[ValidationError] = []
    docs: dict[str, dict[str, Any]] = {}
    root = Path(root)

    # Appendix 1 pins `<slug>.yaml` (lowercase). Flag ANY file whose extension
    # is not exactly `.yaml` rather than silently skipping it, so neither a
    # stray `.yml` nor a mixed-case `.Yaml` can ride through CI unvalidated
    # (EA-1 + JUD-1: denylist replaced with an exact-suffix allowlist).
    for stray in sorted(root.rglob("*")):
        if is_study_path(stray, root):
            continue  # study/ namespace excluded from the canonical schema gate (see is_study_path)
        if stray.is_file() and stray.suffix != ".yaml":
            errors.append(ValidationError(
                str(stray), "extension",
                f"brain files must use the .yaml extension, not {stray.suffix or stray.name}"))

    for path in sorted(root.rglob("*.yaml")):
        if is_study_path(path, root):
            continue  # study/ is a pre-Compile experimental namespace (prose patterns), not the canonical schema
        doc, file_errors = validate_file(path, schema)
        errors.extend(file_errors)
        if doc is not None:
            docs[str(path)] = doc

    seen: dict[str, str] = {}
    for path, doc in docs.items():
        brain_id = doc.get("id")
        if not isinstance(brain_id, str):
            continue
        if brain_id in seen:
            errors.append(ValidationError(
                path, "id-uniqueness", f"brain id {brain_id} already used by {seen[brain_id]}"))
        else:
            seen[brain_id] = path

    for path, doc in docs.items():
        for related in doc.get("related", []):
            related_id = related.get("id") if isinstance(related, dict) else None
            if related_id not in seen:
                errors.append(ValidationError(
                    path, "related-resolution",
                    f"related id {related_id} resolves to no brain in this repository"))
    return errors


def strip_local(doc: dict[str, Any]) -> dict[str, Any]:
    """Publish form: remove LOCAL fields from a brain document (appendix 1)."""
    published = {k: v for k, v in doc.items() if k not in LOCAL_BRAIN_FIELDS}
    published["links"] = [
        {k: v for k, v in link.items() if k not in LOCAL_LINK_FIELDS}
        for link in doc.get("links", [])
    ]
    return published
