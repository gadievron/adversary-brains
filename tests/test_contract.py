"""The appendix-1 contract, tested: schema validity, LOCAL rejection with
named errors, id uniqueness, related-id resolution, entity-ref integrity,
strip_local round-trip, and the validate CLI exit codes."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from brainkit.cli import main as cli_main
from brainkit.model import is_study_path
from brainkit.schema import (
    check_local_fields, load_schema, strip_local, validate_file, validate_repo,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = load_schema()
# The six canonical SHIPPED brains. Excludes the brains/study/ namespace: those are pre-Compile
# lifecycle-study artifacts (free-text prose patterns), validated by a looser contract, not the
# canonical schema — consistent with brainkit.model.load_library / schema.validate_repo.
SHIPPED = sorted(p for p in (ROOT / "brains").rglob("*.yaml")
                 if not is_study_path(p, ROOT / "brains"))


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _set_both_value_and_pattern(doc: dict) -> None:
    """Give entity 0 both value and pattern, which the schema forbids (value XOR pattern)."""
    doc["entities"][0].update(value="x", pattern={"a": "b"})


def _set_numeric_value(doc: dict) -> None:
    """Give entity 0 a non-string (numeric) `value`, which the schema forbids.

    This pins the guarantee that makes a matcher-side `entity.value.lower()`
    crash unreachable FOR ANY CONTRACT-VALIDATED BRAIN: `value` is
    `{"type": "string"}`, so a YAML-coerced scalar (`value: 80`, `value: true`) is
    rejected by the schema, and the shipped pipeline (CI, verify_all, the authoring
    workflow) always validates before matching. `load_brain` alone does not
    validate, so a hand-built unvalidated brain could still crash — but that is not
    a shipped path. Documents why match.py does NOT defensively str()-coerce.
    """
    e = doc["entities"][0]
    e.pop("pattern", None)
    e["value"] = 80


def test_six_brains_ship():
    assert len(SHIPPED) == 6


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_every_shipped_brain_validates(path: Path):
    _, errors = validate_file(path, SCHEMA)
    assert errors == []


def test_repo_validates_end_to_end():
    assert validate_repo(ROOT / "brains") == []


def test_cli_exit_codes(tmp_path, capsys):
    assert cli_main(["validate", str(ROOT / "brains")]) == 0
    bad = load(SHIPPED[0])
    bad["fidelity"] = 0.9
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert cli_main(["validate", str(tmp_path)]) == 1
    assert "LOCAL field 'fidelity'" in capsys.readouterr().err


def test_cli_surface_is_validate_only(capsys):
    # CAP-cli-validate-only: `validate` is the ONLY registered subcommand, so argparse
    # rejects any other with exit code 2 ("invalid choice"). Asserting the code (not just
    # any SystemExit) distinguishes an unknown subcommand from a handler calling sys.exit(),
    # so this locks the argparse COMMAND SURFACE (the engine ships only validate as a CLI).
    for sub in ("screen", "match", "assess", "weights", "extract", "discover", "loop"):
        with pytest.raises(SystemExit) as exc:
            cli_main([sub, str(ROOT / "brains")])
        assert exc.value.code == 2, f"{sub!r} not an argparse invalid-choice (exit 2); got {exc.value.code}"
        assert "invalid choice" in capsys.readouterr().err


def test_cli_validate_message_does_not_overstate_coverage(capsys):
    # brains/ contains study/ prose artifacts the canonical gate deliberately SKIPS;
    # the OK line must not claim it validated "all brains under brains/" (it validates
    # the canonical brains, study/ excluded) — honest gate output, no overstated coverage.
    assert cli_main(["validate", str(ROOT / "brains")]) == 0
    out = capsys.readouterr().out.lower()
    assert "study" in out or "canonical" in out, (
        f"validate OK line overstates coverage (study/ artifacts are skipped): {out!r}"
    )


@pytest.mark.parametrize("local_field,where", [
    ("fidelity", "brain"), ("weight", "link"), ("observability", "link"),
    ("first_seen", "link"), ("last_seen", "link"),
])
def test_every_local_field_rejected_with_named_error(local_field, where):
    doc = load(SHIPPED[0])
    if where == "brain":
        doc[local_field] = "2026-01-01" if "seen" in local_field else 0.5
    else:
        doc["links"][0][local_field] = "2026-01-01" if "seen" in local_field else 0.5
    errors = check_local_fields(doc, "x.yaml")
    assert any(f"LOCAL field '{local_field}'" in e.message for e in errors)
    _, file_errors = _validate_doc(doc)
    assert any(e.check == "local-fields" for e in file_errors)


def _validate_doc(doc: dict) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.yaml"
        path.write_text(yaml.safe_dump(doc))
        return validate_file(path, SCHEMA)


@pytest.mark.parametrize("mutate,expected_check", [
    (lambda d: d.update(attribution="PLA 61398"), "schema"),          # no attribution field, by design
    (lambda d: d.update(created="yesterday"), "schema"),              # ADR 011 date pattern
    (lambda d: d.update(status="bootstrap"), "schema"),               # appendix-3 maturity value is not a status
    (lambda d: d.update(entities=d["entities"][:1]), "schema"),       # min 2 entities
    (lambda d: d.update(links=[]), "schema"),                         # min 1 link
    (_set_both_value_and_pattern, "schema"),                          # value XOR pattern
    (_set_numeric_value, "schema"),                                   # value is string-typed (crash-proofs match.py)
    (lambda d: d["links"][0].update(source="e99"), "entity-refs"),
])
def test_invalid_documents_rejected(mutate, expected_check):
    doc = copy.deepcopy(load(next(p for p in SHIPPED if p.stem.startswith("sunburst"))))
    mutate(doc)
    _, errors = _validate_doc(doc)
    assert any(e.check == expected_check for e in errors), [str(e) for e in errors]


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "a: 1\n---\nb: 2\n"],
                         ids=["empty", "list-document", "two-documents"])
def test_parse_errors_named(content, tmp_path):
    """EA-2: the parse check gets positive coverage (empty / non-mapping / multi-doc)."""
    path = tmp_path / "x.yaml"
    path.write_text(content)
    _, errors = validate_file(path, SCHEMA)
    assert any(e.check == "parse" for e in errors), [str(e) for e in errors]


def test_shape_valid_but_impossible_date_rejected():
    """EA-5: CI-green implies loadable — a shape-valid impossible date is caught."""
    doc = load(SHIPPED[0])
    doc["created"] = "9999-99-99"
    _, errors = _validate_doc(doc)
    assert any(e.check == "schema" and "calendar date" in e.message for e in errors)


def test_duplicate_entity_id_within_file_caught():
    """EA-4: duplicate local ids make source/target resolution ambiguous."""
    doc = load(SHIPPED[0])
    doc["entities"][1]["id"] = doc["entities"][0]["id"]
    _, errors = _validate_doc(doc)
    assert any(e.check == "entity-refs" and "duplicate entity id" in e.message for e in errors)


@pytest.mark.parametrize("name", ["stray.yml", "stray.Yaml", "stray.YML", "stray.json"],
                         ids=["yml", "mixed-case", "upper", "json"])
def test_wrong_extension_flagged(name, tmp_path):
    """EA-1 + JUD-1: any non-.yaml extension is flagged, not silently skipped."""
    (tmp_path / name).write_text("not: a-brain\n")
    checks = {e.check for e in validate_repo(tmp_path)}
    assert "extension" in checks


def test_dotyaml_and_nonfile_not_flagged(tmp_path):
    """The exact-suffix allowlist must not over-reach onto real brains or dirs."""
    (tmp_path / "sub").mkdir()
    a = load(SHIPPED[0])
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(a))
    assert not any(e.check == "extension" for e in validate_repo(tmp_path))


def test_duplicate_link_id_within_file_caught():
    """JUD-2: the link-id-duplicate branch of check_entity_refs is pinned."""
    doc = load(SHIPPED[0])
    doc["links"].append({**doc["links"][0], "id": doc["links"][0]["id"]})
    _, errors = _validate_doc(doc)
    assert any(e.check == "entity-refs" and "duplicate link id" in e.message for e in errors)


def test_duplicate_ids_and_dangling_related_caught(tmp_path):
    a = load(SHIPPED[0])
    b = load(SHIPPED[1])
    b["id"] = a["id"]  # collide
    a["related"] = [{"id": "00000000-0000-4000-8000-00000000dead", "type": "derived"}]
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(a))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(b))
    checks = {e.check for e in validate_repo(tmp_path)}
    assert "id-uniqueness" in checks and "related-resolution" in checks


def test_related_resolving_in_repo_is_clean(tmp_path):
    a = load(SHIPPED[0])
    b = load(SHIPPED[1])
    a["related"] = [{"id": b["id"], "type": "derived"}]
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(a))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(b))
    assert validate_repo(tmp_path) == []


def test_strip_local_roundtrip():
    doc = load(SHIPPED[0])
    doc["fidelity"] = 0.7
    doc["links"][0]["weight"] = 9.9
    doc["links"][0]["observability"] = "visible"
    published = strip_local(doc)
    assert "fidelity" not in published
    assert all(k not in published["links"][0] for k in ("weight", "observability"))
    _, errors = _validate_doc(published)
    assert errors == []
