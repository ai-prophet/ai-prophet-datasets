#!/usr/bin/env python3
"""Validate dataset releases and rebuild the derived registry index.

The repo (`datasets/<name>/...`) is the source of truth. `registry.json`
is a denormalized cache regenerated from the tree. This script is the
only writer of `registry.json`.

Subcommands:
  validate          Lint a release directory or every release in the tree.
  rebuild-registry  Walk `datasets/` and rewrite `registry.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DATASETS_DIR = Path("datasets")
REGISTRY_PATH = Path("registry.json")

DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_RELEASE_STATUSES = {"open", "closed", "archived"}


class ValidationError(RuntimeError):
    """Raised when a dataset, release, or task row fails validation."""


@dataclass(frozen=True)
class ReleaseSummary:
    """Per-release entry as it appears in the derived registry."""

    release_id: str
    release_date: str
    status: str
    task_count: int
    resolved_count: int
    path: str

    def to_entry(self) -> dict[str, Any]:
        return {
            "id": self.release_id,
            "release_date": self.release_date,
            "status": self.status,
            "task_count": self.task_count,
            "resolved_count": self.resolved_count,
            "path": self.path,
        }


def _read_json(path: Path) -> Any:
    """Load a JSON file or raise a ValidationError pointing at it."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def _require_str(obj: dict[str, Any], key: str, *, where: Path) -> str:
    """Pull a non-empty string from a dict; raise ValidationError otherwise."""
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _validate_dataset_json(dataset_dir: Path) -> dict[str, Any]:
    """Validate `dataset.json` and return its parsed content.

    Required keys: `name` (must match the directory name), `description`.
    """
    dataset_path = dataset_dir / "dataset.json"
    payload = _read_json(dataset_path)
    if not isinstance(payload, dict):
        raise ValidationError(f"{dataset_path}: must be a JSON object")

    name = _require_str(payload, "name", where=dataset_path)
    if name != dataset_dir.name:
        raise ValidationError(
            f"{dataset_path}: name '{name}' does not match directory '{dataset_dir.name}'"
        )
    if not DATASET_NAME_RE.match(name):
        raise ValidationError(f"{dataset_path}: name '{name}' has invalid characters")

    _require_str(payload, "description", where=dataset_path)
    return payload


def _validate_release_json(release_dir: Path) -> dict[str, Any]:
    """Validate `release.json` and return its parsed content.

    Required: `release_id` (matches dir), `release_date`, `description`, `status`.
    """
    release_path = release_dir / "release.json"
    payload = _read_json(release_path)
    if not isinstance(payload, dict):
        raise ValidationError(f"{release_path}: must be a JSON object")

    release_id = _require_str(payload, "release_id", where=release_path)
    if release_id != release_dir.name:
        raise ValidationError(
            f"{release_path}: release_id '{release_id}' does not match directory '{release_dir.name}'"
        )
    if not RELEASE_ID_RE.match(release_id):
        raise ValidationError(f"{release_path}: release_id '{release_id}' has invalid characters")

    _require_str(payload, "release_date", where=release_path)
    _require_str(payload, "description", where=release_path)
    status = _require_str(payload, "status", where=release_path)
    if status not in ALLOWED_RELEASE_STATUSES:
        raise ValidationError(
            f"{release_path}: status '{status}' must be one of {sorted(ALLOWED_RELEASE_STATUSES)}"
        )
    return payload


def _validate_task_row(row: dict[str, Any], *, line_no: int, where: Path) -> bool:
    """Validate one task row; return True if it carries a resolved outcome.

    Required: `task_id`, `title`, `outcomes` (list of >=1 non-empty strings).
    Optional: `resolved_outcome` ({value in outcomes, resolved_at?, source?}).
    """
    if not isinstance(row, dict):
        raise ValidationError(f"{where}:{line_no}: row must be a JSON object")

    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValidationError(f"{where}:{line_no}: 'task_id' must be a non-empty string")

    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValidationError(f"{where}:{line_no}: 'title' must be a non-empty string")

    outcomes = row.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValidationError(f"{where}:{line_no}: 'outcomes' must be a non-empty list")
    cleaned = [o for o in outcomes if isinstance(o, str) and o.strip()]
    if len(cleaned) != len(outcomes):
        raise ValidationError(f"{where}:{line_no}: every 'outcomes' entry must be a non-empty string")

    resolved = row.get("resolved_outcome")
    if resolved is None:
        return False
    if not isinstance(resolved, dict):
        raise ValidationError(f"{where}:{line_no}: 'resolved_outcome' must be an object or omitted")
    value = resolved.get("value")
    if not isinstance(value, list) or not value:
        raise ValidationError(
            f"{where}:{line_no}: resolved_outcome.value must be a non-empty list of strings"
        )
    for v in value:
        if not isinstance(v, str) or v not in outcomes:
            raise ValidationError(
                f"{where}:{line_no}: resolved_outcome.value entry {v!r} must be one of {outcomes!r}"
            )
    if len(set(value)) != len(value):
        raise ValidationError(
            f"{where}:{line_no}: resolved_outcome.value entries must be unique"
        )
    for opt_key in ("resolved_at", "source"):
        if opt_key in resolved and not isinstance(resolved[opt_key], str):
            raise ValidationError(
                f"{where}:{line_no}: resolved_outcome.{opt_key} must be a string when present"
            )
    return True


def _validate_tasks_jsonl(tasks_path: Path) -> tuple[int, int]:
    """Validate every row of `tasks.jsonl`; return (task_count, resolved_count).

    Also enforces that `task_id` is unique within the file.
    """
    if not tasks_path.exists():
        raise ValidationError(f"missing file: {tasks_path}")

    task_count = 0
    resolved_count = 0
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(tasks_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{tasks_path}:{line_no}: invalid JSON ({exc})") from exc

        is_resolved = _validate_task_row(row, line_no=line_no, where=tasks_path)
        task_id = row["task_id"]
        if task_id in seen_ids:
            raise ValidationError(f"{tasks_path}:{line_no}: duplicate task_id '{task_id}'")
        seen_ids.add(task_id)
        task_count += 1
        if is_resolved:
            resolved_count += 1

    if task_count == 0:
        raise ValidationError(f"{tasks_path}: file has no JSONL records")
    return task_count, resolved_count


def _validate_release(release_dir: Path) -> ReleaseSummary:
    """Validate one release directory end-to-end and return its summary."""
    release_meta = _validate_release_json(release_dir)
    tasks_path = release_dir / "tasks.jsonl"
    task_count, resolved_count = _validate_tasks_jsonl(tasks_path)
    return ReleaseSummary(
        release_id=release_meta["release_id"],
        release_date=release_meta["release_date"],
        status=release_meta["status"],
        task_count=task_count,
        resolved_count=resolved_count,
        path=tasks_path.as_posix(),
    )


def _iter_dataset_dirs(root: Path) -> Iterable[Path]:
    """Yield each `datasets/<name>/` directory, sorted by name."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _iter_release_dirs(dataset_dir: Path) -> Iterable[Path]:
    """Yield each `datasets/<name>/releases/<release_id>/` directory."""
    releases_root = dataset_dir / "releases"
    if not releases_root.is_dir():
        return []
    return sorted(p for p in releases_root.iterdir() if p.is_dir())


def _build_registry(root: Path) -> dict[str, Any]:
    """Walk the tree, validate every dataset/release, return registry payload."""
    datasets_out: list[dict[str, Any]] = []
    for dataset_dir in _iter_dataset_dirs(root):
        dataset_meta = _validate_dataset_json(dataset_dir)
        releases_out: list[dict[str, Any]] = []
        for release_dir in _iter_release_dirs(dataset_dir):
            summary = _validate_release(release_dir)
            releases_out.append(summary.to_entry())
        releases_out.sort(key=lambda r: r["release_date"], reverse=True)
        datasets_out.append(
            {
                "name": dataset_meta["name"],
                "description": dataset_meta["description"],
                "releases": releases_out,
            }
        )
    return {"datasets": datasets_out}


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate one release dir, or every release if --all is passed."""
    if args.all:
        try:
            _build_registry(DATASETS_DIR)
        except ValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("ok: all datasets and releases pass validation")
        return 0

    target = Path(args.release).resolve()
    try:
        # When the user passes a release dir directly, we skip dataset.json
        # checks (it might not be the focus). For full-tree checks, use --all.
        _validate_release(target)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ok: {target}")
    return 0


def cmd_rebuild_registry(args: argparse.Namespace) -> int:
    """Regenerate `registry.json` from the dataset tree."""
    try:
        payload = _build_registry(DATASETS_DIR)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {out_path} ({len(payload['datasets'])} datasets)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Wire up the CLI subcommands."""
    parser = argparse.ArgumentParser(description="Validate datasets and rebuild registry.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate a release directory or all releases")
    grp = p_validate.add_mutually_exclusive_group(required=True)
    grp.add_argument("--release", help="Path to a release directory (datasets/<name>/releases/<id>/)")
    grp.add_argument("--all", action="store_true", help="Validate every dataset and release in the tree")
    p_validate.set_defaults(func=cmd_validate)

    p_rebuild = sub.add_parser("rebuild-registry", help="Rewrite registry.json from the tree")
    p_rebuild.add_argument("--output", default=str(REGISTRY_PATH), help="Path to registry.json")
    p_rebuild.set_defaults(func=cmd_rebuild_registry)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
