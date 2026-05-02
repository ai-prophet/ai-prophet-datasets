"""File-level validation: parses JSON/JSONL on disk and enforces invariants
beyond what the pydantic models check (uniqueness across rows, directory
layout, name/id consistency with directory names).

Used by the SDK's CLI and writers, and intended to be the single source
of truth that the repo's CI also calls into.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import DatasetMetadata, ReleaseMetadata, Task


class SchemaError(RuntimeError):
    """Raised when on-disk data fails validation.

    The error message includes the offending file path (and line number,
    for JSONL rows) so callers can surface it directly to the user.
    """


def load_dataset_metadata(dataset_dir: Path) -> DatasetMetadata:
    """Read and validate `<dataset_dir>/dataset.json`.

    Also enforces that `name` matches the directory name.
    """
    path = dataset_dir / "dataset.json"
    payload = _read_json(path)
    try:
        meta = DatasetMetadata.model_validate(payload)
    except ValidationError as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    if meta.name != dataset_dir.name:
        raise SchemaError(
            f"{path}: name '{meta.name}' does not match directory '{dataset_dir.name}'"
        )
    return meta


def load_release_metadata(release_dir: Path) -> ReleaseMetadata:
    """Read and validate `<release_dir>/release.json`.

    Also enforces that `release_id` matches the directory name.
    """
    path = release_dir / "release.json"
    payload = _read_json(path)
    try:
        meta = ReleaseMetadata.model_validate(payload)
    except ValidationError as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    if meta.release_id != release_dir.name:
        raise SchemaError(
            f"{path}: release_id '{meta.release_id}' does not match directory "
            f"'{release_dir.name}'"
        )
    return meta


def load_tasks(tasks_path: Path) -> list[Task]:
    """Read `tasks.jsonl`, validate every row, and enforce id uniqueness."""
    if not tasks_path.exists():
        raise SchemaError(f"missing file: {tasks_path}")

    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(tasks_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{tasks_path}:{line_no}: invalid JSON ({exc})") from exc
        try:
            task = Task.model_validate(row)
        except ValidationError as exc:
            raise SchemaError(f"{tasks_path}:{line_no}: {exc}") from exc
        if task.task_id in seen_ids:
            raise SchemaError(f"{tasks_path}:{line_no}: duplicate task_id '{task.task_id}'")
        seen_ids.add(task.task_id)
        tasks.append(task)

    if not tasks:
        raise SchemaError(f"{tasks_path}: file has no JSONL records")
    return tasks


def validate_release(release_dir: Path) -> tuple[ReleaseMetadata, list[Task]]:
    """Validate one release end-to-end (release.json + tasks.jsonl)."""
    meta = load_release_metadata(release_dir)
    tasks = load_tasks(release_dir / "tasks.jsonl")
    return meta, tasks


def validate_tree(datasets_root: Path) -> dict[str, Any]:
    """Validate every dataset and release under `datasets/`.

    Returns a structured summary mirroring `registry.json`'s shape so
    callers can write the registry directly from this output. The
    `path` of each release is expressed relative to the repo root
    (the parent of `datasets_root`).
    """
    if not datasets_root.is_dir():
        raise SchemaError(f"datasets root not found: {datasets_root}")
    repo_root = datasets_root.parent

    datasets_out: list[dict[str, Any]] = []
    for dataset_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        dataset_meta = load_dataset_metadata(dataset_dir)
        releases_out: list[dict[str, Any]] = []
        releases_root = dataset_dir / "releases"
        if releases_root.is_dir():
            for release_dir in sorted(p for p in releases_root.iterdir() if p.is_dir()):
                meta, tasks = validate_release(release_dir)
                tasks_path = release_dir / "tasks.jsonl"
                releases_out.append(
                    {
                        "id": meta.release_id,
                        "release_date": meta.release_date,
                        "status": meta.status,
                        "task_count": len(tasks),
                        "resolved_count": sum(1 for t in tasks if t.resolved_outcome is not None),
                        "path": tasks_path.relative_to(repo_root).as_posix(),
                    }
                )
        releases_out.sort(key=lambda r: r["release_date"], reverse=True)
        datasets_out.append(
            {
                "name": dataset_meta.name,
                "description": dataset_meta.description,
                "releases": releases_out,
            }
        )
    return {"datasets": datasets_out}


def _read_json(path: Path) -> Any:
    """Load a JSON file; raise SchemaError pointing at the file on failure."""
    if not path.exists():
        raise SchemaError(f"missing file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: invalid JSON ({exc})") from exc
