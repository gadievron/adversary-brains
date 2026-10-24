"""Screen ranking + coverage overlay unit tests (Stage 2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from brainkit.coverage import empty, load_coverage
from brainkit.model import load_library
from brainkit.screen import Screen

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = load_library(ROOT / "brains")


def event(relation, name=None, value=None):
    obj_src = {"type": "process", "name": name} if name else {"type": "x", "value": "?"}
    return {"time": "2026-07-01T00:00:00Z", "host": "h1", "relation": relation,
            "source": obj_src,
            "target": {"type": "domain-name", "value": value or "d.example"}}


# --- screen ----------------------------------------------------------------

def test_sunburst_event_ranks_sunburst_first():
    screen = Screen(LIBRARY)
    events = [event("connected-to", name="solarwinds.businesslayerhost.exe",
                    value="evil.example")]
    hits = screen.rank(events, top_n=3)
    assert hits, "expected at least one shortlist hit"
    top = hits[0]
    brain = next(b for b in LIBRARY if b.id == top.brain_id)
    assert "sunburst" in brain.title.lower()
    assert any("solarwinds" in s for s in top.signals)


def test_screen_returns_nothing_for_unrelated_event():
    screen = Screen(LIBRARY)
    hits = screen.rank([event("logged-in", name="nothing-here.exe", value="benign.test")])
    assert hits == []


def test_rarity_prefers_distinctive_tokens():
    # A token in one brain scores higher than one shared across many.
    screen = Screen(LIBRARY)
    hits = screen.rank([event("created-port-proxy", name="netsh.exe", value="x.example")])
    ids = [h.brain_id for h in hits]
    volt = next(b for b in LIBRARY if "volt" in b.title.lower())
    assert volt.id in ids


def test_screen_deterministic():
    screen = Screen(LIBRARY)
    ev = [event("connected-to", name="solarwinds.businesslayerhost.exe", value="e.example")]
    assert screen.rank(ev) == screen.rank(ev)


def test_screen_scales_to_1000_brains_under_a_second():
    """EXP-2-4: distinct ids AND distinct value tokens so the rarity index build
    is genuinely exercised (not 1000 identical clones), and time BOTH the index
    build and the ranking — the <1s contract covers the whole cheap pass."""
    import dataclasses
    import time
    base = LIBRARY[0]
    many = []
    for i in range(1000):
        entities = tuple(
            dataclasses.replace(e, value=f"{e.value}-{i}") if e.value is not None else e
            for e in base.entities
        )
        many.append(dataclasses.replace(base, id=f"{base.id[:-4]}{i:04d}", entities=entities))
    batch = [event("connected-to", name="solarwinds.businesslayerhost.exe", value=f"e{i}.example")
             for i in range(100)]
    start = time.perf_counter()
    screen = Screen(many)          # index build over ~1000 distinct token sets
    screen.rank(batch)
    assert time.perf_counter() - start < 1.0
    assert len({p.brain_id for p in screen._profiles}) == 1000  # ids truly distinct


# --- coverage --------------------------------------------------------------

def test_empty_coverage_is_all_visible():
    cov = empty()
    assert cov.invisible_links("any-brain") == frozenset()
    assert cov.visibility_of("any-brain", "L1") == "visible"


def test_load_coverage_marks_invisible(tmp_path):
    (tmp_path / "cov.yaml").write_text(
        "brains:\n  brain-x:\n    L0: invisible\n    L1: visible\n")
    cov = load_coverage(tmp_path / "cov.yaml")
    assert cov.invisible_links("brain-x") == frozenset({"L0"})
    assert cov.visibility_of("brain-x", "L1") == "visible"
    assert cov.visibility_of("brain-x", "L9") == "visible"  # default


def test_load_coverage_rejects_bad_mark(tmp_path):
    (tmp_path / "cov.yaml").write_text("brains:\n  b:\n    L0: maybe\n")
    with pytest.raises(ValueError, match="must be one of"):
        load_coverage(tmp_path / "cov.yaml")


def test_load_coverage_rejects_bad_default(tmp_path):
    (tmp_path / "cov.yaml").write_text("default: sometimes\n")
    with pytest.raises(ValueError, match="default must be one of"):
        load_coverage(tmp_path / "cov.yaml")


def test_load_coverage_rejects_list_brains(tmp_path):
    """EXP-2-5: a list under `brains:` raises a clean ValueError, not AttributeError."""
    (tmp_path / "cov.yaml").write_text("brains:\n  - L0\n  - L1\n")
    with pytest.raises(ValueError, match="'brains' must be a mapping"):
        load_coverage(tmp_path / "cov.yaml")


def test_invisible_links_for_honors_default_invisible(tmp_path):
    """default:invisible expands to every unmarked link — the accept-but-ignore
    trap Pass C surfaced, closed by the brain-aware method."""
    brain = next(b for b in LIBRARY if "scattered" in b.title.lower())
    link_ids = {l.id for l in brain.links}
    (tmp_path / "cov.yaml").write_text(
        f"default: invisible\nbrains:\n  {brain.id}:\n    L1: visible\n")
    cov = load_coverage(tmp_path / "cov.yaml")
    # id-form under-reports (returns nothing — no explicit invisible marks)...
    assert cov.invisible_links(brain.id) == frozenset()
    # ...brain-aware form correctly returns every link except the one marked visible
    assert cov.invisible_links_for(brain) == frozenset(link_ids - {"L1"})


def test_invisible_links_for_matches_id_form_when_default_visible(tmp_path):
    brain = next(b for b in LIBRARY if "scattered" in b.title.lower())
    (tmp_path / "cov.yaml").write_text(
        f"brains:\n  {brain.id}:\n    L0: invisible\n")
    cov = load_coverage(tmp_path / "cov.yaml")
    assert cov.invisible_links_for(brain) == cov.invisible_links(brain.id) == frozenset({"L0"})
