"""Local build artifacts must not trip verify_all's dirty-tree guard.

`experiments/runlog.tree_is_dirty()` refuses to write confirmatory results when
`git status --porcelain` lists any untracked path outside `experiments/results/`.
The documented quickstart has the verifier create a venv; if that venv dir is not
git-ignored it shows up as untracked and makes `verify_all.sh` abort with
`RuntimeError: refusing to write ... from a dirty tree`. The same holds for the
paper's `.docx` exports, which the doc build writes to the repo root. Gate: the
common venv directory names AND root-level `.docx` exports are git-ignored, so
neither a verifier's venv nor a built paper ever dirties the tree.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_common_venv_dirs_are_gitignored():
    # Check a path UNDER each venv dir: a trailing-slash dir pattern (`.venv/`)
    # matches contents of the directory, which is what a real venv produces and
    # what `git status --porcelain` would otherwise report as untracked.
    for probe in (".venv/pyvenv.cfg", "venv/pyvenv.cfg"):
        r = subprocess.run(
            ["git", "check-ignore", probe],
            cwd=REPO, capture_output=True, text=True,
        )
        assert r.returncode == 0 and probe in r.stdout, (
            f"{probe!r} is NOT git-ignored (check-ignore rc={r.returncode}); "
            f"a verifier's venv would trip runlog.tree_is_dirty() and abort verify_all"
        )


def test_root_docx_paper_exports_are_gitignored():
    # The paper's docx export (a local, untracked build script) writes its `.docx`
    # to the repo root. If they are not git-ignored they show up as untracked and
    # make runlog.tree_is_dirty() abort every confirmatory experiment + verify_all.
    # A root-anchored `/*.docx` pattern ignores any root export without hiding a
    # `.docx` committed deliberately inside a subdirectory.
    for probe in ("04-academic-paper.docx", "adversary-brains-paper.docx", "any-root-export.docx"):
        r = subprocess.run(
            ["git", "check-ignore", probe],
            cwd=REPO, capture_output=True, text=True,
        )
        assert r.returncode == 0 and probe in r.stdout, (
            f"{probe!r} is NOT git-ignored (check-ignore rc={r.returncode}); "
            f"a root .docx paper export would trip runlog.tree_is_dirty() and abort verify_all"
        )


def test_gitignore_docx_rule_is_root_anchored():
    # The exclusion must NOT hide a `.docx` deliberately tracked under a subdir
    # (e.g. a fixture or vendored document) — only root-level build exports.
    r = subprocess.run(
        ["git", "check-ignore", "docs/some-tracked.docx"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 1, (
        "a subdirectory .docx is unexpectedly git-ignored; the rule should be "
        "root-anchored (`/*.docx`), not a global `*.docx`"
    )
