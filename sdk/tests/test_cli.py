"""CLI smoke tests: invoke `main()` directly with argv lists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_prophet_datasets.cli import main


def test_cli_validate_seeded_repo(repo: Path, capsys):
    """`validate` against a seeded repo prints ok and exits 0."""
    code = main(["--repo-path", str(repo), "validate"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ok" in out


def test_cli_validate_specific_release_dir(repo: Path, capsys):
    """`validate --release <dir>` works against a single release."""
    release_dir = repo / "datasets" / "dummy" / "releases" / "2026-04-01"
    code = main(["validate", "--release", str(release_dir)])
    assert code == 0


def test_cli_validate_reports_error(repo: Path, capsys):
    """A broken release returns non-zero and prints to stderr."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    tasks_path.write_text("definitely not json\n")

    code = main(["--repo-path", str(repo), "validate"])
    assert code == 1
    err = capsys.readouterr().err
    assert "error" in err


def test_cli_resolve_commits(repo: Path, capsys):
    """`resolve` produces a commit and prints its sha."""
    code = main(
        [
            "--repo-path",
            str(repo),
            "resolve",
            "dummy",
            "2026-04-01",
            "--task-id",
            "t-001",
            "--value",
            "Yes",
            "--source",
            "manual",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "committed" in out

    rows = [
        json.loads(line)
        for line in (
            repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
        )
        .read_text()
        .splitlines()
        if line.strip()
    ]
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["t-001"]["resolved_outcome"]["value"] == ["Yes"]


def test_cli_global_flag_works_after_subcommand(repo: Path, capsys):
    """Regression: `--repo-path` after the subcommand should work too."""
    code = main(
        [
            "resolve",
            "dummy",
            "2026-04-01",
            "--task-id",
            "t-001",
            "--value",
            "Yes",
            "--repo-path",
            str(repo),
        ]
    )
    assert code == 0


def test_cli_resolve_requires_repo_path(capsys):
    """`resolve` without --repo-path errors out with code 2."""
    code = main(
        [
            "resolve",
            "dummy",
            "r1",
            "--task-id",
            "x",
            "--value",
            "Yes",
        ]
    )
    assert code == 2
