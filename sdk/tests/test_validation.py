"""File-level validation tests — exercise on-disk fixtures, not just dicts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_prophet_datasets.validation import (
    SchemaError,
    load_dataset_metadata,
    load_release_metadata,
    load_tasks,
    validate_release,
    validate_tree,
)


def test_validate_seeded_repo_passes(repo: Path):
    """The fixture seed should pass full-tree validation cleanly."""
    summary = validate_tree(repo / "datasets")
    assert len(summary["datasets"]) == 1
    ds = summary["datasets"][0]
    assert ds["name"] == "dummy"
    assert len(ds["releases"]) == 1
    assert ds["releases"][0]["task_count"] == 2
    assert ds["releases"][0]["resolved_count"] == 1


def test_dataset_name_must_match_directory(repo: Path):
    """Renaming `name` inside dataset.json without moving the dir must fail."""
    dataset_json = repo / "datasets" / "dummy" / "dataset.json"
    payload = json.loads(dataset_json.read_text())
    payload["name"] = "wrong-name"
    dataset_json.write_text(json.dumps(payload))

    with pytest.raises(SchemaError, match="does not match directory"):
        load_dataset_metadata(repo / "datasets" / "dummy")


def test_release_id_must_match_directory(repo: Path):
    """Release id mismatch is caught immediately."""
    release_dir = repo / "datasets" / "dummy" / "releases" / "2026-04-01"
    release_json = release_dir / "release.json"
    payload = json.loads(release_json.read_text())
    payload["release_id"] = "different"
    release_json.write_text(json.dumps(payload))

    with pytest.raises(SchemaError, match="does not match directory"):
        load_release_metadata(release_dir)


def test_duplicate_task_ids_rejected(repo: Path):
    """Two rows with the same `task_id` is a hard failure."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    rows.append(rows[0])  # duplicate
    tasks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(SchemaError, match="duplicate task_id"):
        load_tasks(tasks_path)


def test_resolved_value_outside_outcomes_rejected(repo: Path):
    """`resolved_outcome.value` must be a subset of the task's `outcomes`."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    rows[0]["resolved_outcome"] = {"value": ["NotAnOutcome"]}
    tasks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(SchemaError, match="not in outcomes"):
        load_tasks(tasks_path)


def test_resolved_value_must_be_list_not_string(repo: Path):
    """The list-shape invariant is enforced at the file level too."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    rows[0]["resolved_outcome"] = {"value": "Yes"}  # bare string
    tasks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(SchemaError):
        load_tasks(tasks_path)


def test_invalid_jsonl_line_reports_line_number(repo: Path):
    """A bad row surfaces with file:line in the error."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    tasks_path.write_text(tasks_path.read_text() + "not valid json\n")

    with pytest.raises(SchemaError, match=r":\d+: invalid JSON"):
        load_tasks(tasks_path)


def test_empty_tasks_file_rejected(repo: Path):
    """A release with no rows is invalid."""
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    tasks_path.write_text("\n")
    with pytest.raises(SchemaError, match="no JSONL records"):
        load_tasks(tasks_path)


def test_missing_release_json_rejected(repo: Path):
    """A release directory without `release.json` is invalid."""
    release_dir = repo / "datasets" / "dummy" / "releases" / "2026-04-01"
    (release_dir / "release.json").unlink()
    with pytest.raises(SchemaError, match="missing file"):
        validate_release(release_dir)
