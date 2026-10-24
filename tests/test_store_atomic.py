"""Atomic-write contract (appendix 3 done-when: 'killing the process mid-write
leaves no partial files') — fault-injected, not narrated."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from brainkit.store import atomic_write, slug


def test_write_then_read(tmp_path):
    target = tmp_path / "brain.yaml"
    atomic_write(target, "content-1")
    assert target.read_text() == "content-1"
    atomic_write(target, "content-2")
    assert target.read_text() == "content-2"


def test_crash_before_rename_leaves_original_intact(tmp_path):
    target = tmp_path / "brain.yaml"
    atomic_write(target, "original")
    with mock.patch("brainkit.store.os.replace", side_effect=OSError("killed")):
        with pytest.raises(OSError, match="killed"):
            atomic_write(target, "partial-new")
    assert target.read_text() == "original"          # old content survives
    assert [p.name for p in tmp_path.iterdir()] == ["brain.yaml"]  # no temp litter


def test_crash_mid_write_leaves_no_partial_target(tmp_path):
    target = tmp_path / "fresh.yaml"

    real_fsync = os.fsync

    def exploding_fsync(fd):
        real_fsync(fd)
        raise OSError("killed mid-write")

    with mock.patch("brainkit.store.os.fsync", side_effect=exploding_fsync):
        with pytest.raises(OSError, match="killed"):
            atomic_write(target, "never-visible")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_slug():
    assert slug("APT29 (Cozy Bear)") == "apt29-cozy-bear"
    assert slug("  Lure chain / registrant re-use! ") == "lure-chain-registrant-re-use"
