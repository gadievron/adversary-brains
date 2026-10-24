"""Filesystem primitives: atomic writes, slugs, small YAML/JSONL IO.

Appendix 3's write contract, reimplemented from its rebuild spec: "every write
goes through a temp file and an atomic rename"; a process killed mid-write
leaves no partial state (tested by fault injection in tests/test_store_atomic.py).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def atomic_write(path: str | Path, text: str) -> None:
    """Write text to path atomically: temp file in the same dir, fsync, rename.

    Either the complete new content is visible at `path`, or the previous
    content is — never a partial file, even under mid-write crash. A hard
    SIGKILL between mkstemp and os.replace can strand a hidden
    `.<name>.<rand>.tmp` file in the directory (the except-cleanup only runs on
    catchable failures); that is contract-compatible — no partial file exists
    at any `<slug>.yaml` path, and stray temps are invisible to the
    ``*.yaml`` glob (EA-3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Leave no stray temp file behind on any failure path.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def slug(name: str) -> str:
    """Filesystem-safe slug: 'APT29 (Cozy Bear)' -> 'apt29-cozy-bear'."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return cleaned.strip("-")


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
