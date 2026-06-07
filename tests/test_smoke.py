"""
Smoke test: every Python file in the repo compiles.

This is a real gate — it catches syntax errors anywhere in the codebase on every
push, which is what makes the CI badge meaningful rather than decorative. It does
NOT import modules (that would require the full heavy dependency tree), so it runs
fast and doesn't need OpenAI keys, torch, etc.

Replace/extend with behavioural tests as modules stabilise.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "build", "dist", "cache"}


def _python_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        files.append(p)
    return sorted(files)


PY_FILES = _python_files()


@pytest.mark.parametrize("path", PY_FILES, ids=[str(p.relative_to(ROOT)) for p in PY_FILES])
def test_file_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def test_repo_has_python_files() -> None:
    # Guards against the parametrize list being silently empty.
    assert PY_FILES, "No Python files discovered — check test discovery root."
