"""Appendix-3 'done when' checklist, adapted to this repo (deterministic).

| Appendix 3 item                          | Adapted assertion here                    |
|------------------------------------------|-------------------------------------------|
| bootstrap yields folder-per-group        | all 6 shipped brains load + validate      |
| screen < 1s over the library             | screen over 1,000 brains < 1s             |
| WellMess event ranks APT29 first         | SUNBURST event ranks the sunburst brain 1 |
| invented technique id dropped at ingest  | build_draft drops T9999.999               |
| re-ingesting the same report changes ..  | build_draft is idempotent                 |
| killing mid-write leaves no partial file | atomic_write fault-injection (test_store) |
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from brainkit.extract import build_draft
from brainkit.model import load_library
from brainkit.schema import validate_repo
from brainkit.screen import Screen

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = load_library(ROOT / "brains")


def test_all_brains_load_and_validate():
    assert len(LIBRARY) == 6
    assert validate_repo(ROOT / "brains") == []


def test_screen_under_one_second_over_1000_brains():
    base = LIBRARY[0]
    many = [dataclasses.replace(
        base, id=f"{base.id[:-4]}{i:04d}",
        entities=tuple(dataclasses.replace(e, value=f"{e.value}-{i}") if e.value else e
                       for e in base.entities)) for i in range(1000)]
    batch = [{"time": "2026-07-01T00:00:00Z", "host": "h", "relation": "connected-to",
              "source": {"type": "process", "name": "solarwinds.businesslayerhost.exe"},
              "target": {"type": "domain-name", "value": f"e{i}.example"}} for i in range(100)]
    start = time.perf_counter()
    Screen(many).rank(batch)
    assert time.perf_counter() - start < 1.0


def test_distinctive_event_ranks_its_brain_first():
    """Adapted 'WellMess ranks APT29 first': a SUNBURST event ranks sunburst #1."""
    screen = Screen(LIBRARY)
    hits = screen.rank([{"time": "2026-07-01T00:00:00Z", "host": "h", "relation": "connected-to",
                         "source": {"type": "process", "name": "solarwinds.businesslayerhost.exe"},
                         "target": {"type": "domain-name", "value": "evil.example"}}])
    assert hits
    top = next(b for b in LIBRARY if b.id == hits[0].brain_id)
    assert "sunburst" in top.title.lower()


def test_invented_technique_id_dropped_at_extract():
    r = build_draft("used T1059.001 and fabricated T9999.999", "d")
    assert "T9999.999" in r.dropped_technique_ids and "T1059.001" in r.valid_technique_ids


def test_reingest_is_idempotent():
    report = "PowerShell T1059.001; c2 evil.example; ip 198.51.100.9"
    assert build_draft(report, "d").draft == build_draft(report, "d").draft
