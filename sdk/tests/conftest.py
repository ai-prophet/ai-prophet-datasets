"""Shared test fixtures: real tmp directories and real git repos.

We avoid mocking the filesystem and git — every fixture builds an actual
working tree and runs `git` against it. Tests stay fast because the
trees are tiny.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git subcommand against `repo` and return stdout."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout


def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _seed_dataset(
    repo: Path,
    *,
    name: str,
    description: str,
    release_id: str,
    release_date: str,
    release_status: str = "open",
    tasks: list[dict] | None = None,
) -> None:
    """Drop one dataset + one release into the working tree."""
    dataset_dir = repo / "datasets" / name
    _write_json(dataset_dir / "dataset.json", {"name": name, "description": description})
    release_dir = dataset_dir / "releases" / release_id
    _write_json(
        release_dir / "release.json",
        {
            "release_id": release_id,
            "release_date": release_date,
            "description": f"{name} release {release_id}",
            "status": release_status,
        },
    )
    _write_jsonl(release_dir / "tasks.jsonl", tasks or _default_tasks())


def _default_tasks() -> list[dict]:
    """A small, mixed task set: one resolved, one unresolved."""
    return [
        {
            "task_id": "t-001",
            "title": "Will it rain tomorrow?",
            "outcomes": ["Yes", "No"],
        },
        {
            "task_id": "t-002",
            "title": "Who wins the game?",
            "outcomes": ["Home", "Away"],
            "resolved_outcome": {
                "value": ["Home"],
                "resolved_at": "2026-04-01T00:00:00Z",
                "source": "test-fixture",
            },
        },
    ]


@pytest.fixture
def make_repo(tmp_path: Path) -> Iterator[callable]:
    """Factory yielding fresh git working trees rooted under `tmp_path`.

    The factory returns the repo path. By default, an `origin` bare repo
    is set up alongside so `push`/`pull` work without a real remote.
    """
    counter = {"n": 0}

    def _factory(*, init_origin: bool = True, datasets: list[dict] | None = None) -> Path:
        counter["n"] += 1
        repo = tmp_path / f"repo{counter['n']}"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
        _git(repo, "config", "commit.gpgsign", "false")

        if datasets:
            for spec in datasets:
                _seed_dataset(repo, **spec)
        else:
            _seed_dataset(
                repo,
                name="dummy",
                description="Test dummy dataset",
                release_id="2026-04-01",
                release_date="2026-04-01",
            )

        # Always seed an empty registry so locals can rebuild it.
        (repo / "registry.json").write_text(json.dumps({"datasets": []}, indent=2) + "\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "seed")

        if init_origin:
            origin = tmp_path / f"origin{counter['n']}.git"
            _git(repo, "init", "--bare", "-q", "-b", "main", str(origin))
            _git(repo, "remote", "add", "origin", str(origin))
            _git(repo, "push", "-q", "-u", "origin", "main")

        return repo

    yield _factory


@pytest.fixture
def repo(make_repo) -> Path:
    """A ready-to-use repo with the default fixture seeded."""
    return make_repo()


@pytest.fixture
def empty_repo(make_repo) -> Path:
    """A repo with no datasets seeded — useful for create_dataset tests."""
    return make_repo(datasets=[])


@pytest.fixture(autouse=True)
def _clean_pycache(tmp_path):
    """Make sure tmp paths from one test don't leak into the next via globals."""
    yield
    # tmp_path is auto-cleaned by pytest; nothing else to do.


__all__ = ["_git", "_write_json", "_write_jsonl", "_seed_dataset"]
