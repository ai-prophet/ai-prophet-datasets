"""Top-level entry points: Registry, Dataset, Release.

The Registry has two modes:

* **Remote (read-only)**: no `repo_path` — reads `registry.json` and
  individual `tasks.jsonl` files via raw.githubusercontent.com. The
  upstream repo is public so no auth is needed.
* **Local (read + write)**: pass `repo_path` pointing at a git working
  tree. Reads come from disk; writes mutate files and run `git` in a
  subprocess to commit. Pushing is a separate explicit step.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError as PydanticValidationError

from .models import ReleaseStatus, Task
from .validation import (
    SchemaError,
    load_dataset_metadata,
    load_release_metadata,
    load_tasks,
    validate_tree,
)

DEFAULT_REPO_URL = "https://github.com/ai-prophet/ai-prophet-datasets"
DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class ReleaseSummary:
    """One release entry as it appears in the registry index."""

    id: str
    release_date: str
    status: str
    task_count: int
    resolved_count: int
    path: str


class Registry:
    """Handle on the dataset registry — read in either mode, write in local."""

    def __init__(
        self,
        repo_path: str | os.PathLike[str] | None = None,
        *,
        repo_url: str = DEFAULT_REPO_URL,
        branch: str = DEFAULT_BRANCH,
        http_timeout: float = 30.0,
    ) -> None:
        self.repo_path: Path | None = Path(repo_path).resolve() if repo_path else None
        self.repo_url = repo_url.rstrip("/").removesuffix(".git")
        self.branch = branch
        self._http_timeout = http_timeout
        self._http_client: httpx.Client | None = None

    @property
    def is_local(self) -> bool:
        """True if this Registry is bound to a local git working tree."""
        return self.repo_path is not None

    def list_datasets(self) -> list[Dataset]:
        """List every dataset known to the registry, with release summaries."""
        index = self._load_index()
        return [self._dataset_from_index(d) for d in index["datasets"]]

    def get_dataset(self, name: str) -> Dataset:
        """Fetch a single dataset by name; raise KeyError if missing."""
        for dataset in self.list_datasets():
            if dataset.name == name:
                return dataset
        raise KeyError(f"dataset not found: {name}")

    def get_release(self, dataset: str, release_id: str) -> Release:
        """Convenience: fetch one release directly."""
        return self.get_dataset(dataset).get_release(release_id)

    def create_dataset(
        self,
        name: str,
        description: str,
        *,
        commit_message: str | None = None,
    ) -> str:
        """Create `datasets/<name>/dataset.json`. Returns the commit sha."""
        self._require_local()
        assert self.repo_path is not None
        if not name.strip():
            raise ValueError("name must be non-empty")
        if not description.strip():
            raise ValueError("description must be non-empty")

        dataset_dir = self.repo_path / "datasets" / name
        if dataset_dir.exists():
            raise FileExistsError(f"dataset already exists: {dataset_dir}")
        dataset_dir.mkdir(parents=True)

        dataset_json = dataset_dir / "dataset.json"
        _write_json_atomic(dataset_json, {"name": name, "description": description})
        load_dataset_metadata(dataset_dir)  # sanity check

        message = commit_message or f"create dataset {name}"
        return self._commit_with_registry([dataset_json], message)

    def create_release(
        self,
        dataset: str,
        release_id: str,
        tasks: list[dict[str, Any]],
        *,
        release_date: str,
        description: str,
        status: ReleaseStatus = "open",
        commit_message: str | None = None,
    ) -> str:
        """Create a new release and commit it. Returns the commit sha.

        `tasks` is a list of dict rows; each is validated through the Task
        model before being written. Fails if `release_id` already exists
        for `dataset`.
        """
        self._require_local()
        assert self.repo_path is not None

        dataset_dir = self.repo_path / "datasets" / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"dataset '{dataset}' does not exist; call create_dataset() first"
            )
        load_dataset_metadata(dataset_dir)  # ensure dataset.json is valid

        release_dir = dataset_dir / "releases" / release_id
        if release_dir.exists():
            raise FileExistsError(f"release already exists: {release_dir}")

        if not tasks:
            raise ValueError("create_release requires at least one task")

        validated_rows = _validate_rows(tasks)

        release_dir.mkdir(parents=True)
        release_json = release_dir / "release.json"
        tasks_path = release_dir / "tasks.jsonl"

        _write_json_atomic(
            release_json,
            {
                "release_id": release_id,
                "release_date": release_date,
                "description": description,
                "status": status,
            },
        )
        _write_jsonl_atomic(tasks_path, validated_rows)

        # Trip every validator end-to-end as a final guard.
        load_release_metadata(release_dir)
        load_tasks(tasks_path)

        message = commit_message or f"add release {dataset}/{release_id}"
        return self._commit_with_registry([release_json, tasks_path], message)

    def current_branch(self) -> str:
        """Return the name of the currently checked-out local branch."""
        self._require_local()
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    def sync(self) -> None:
        """`git pull --rebase origin <current-branch>`.

        Operates on whichever branch is checked out in the working tree,
        not on `self.branch` (which controls remote *reads* only).
        """
        self._require_local()
        self._git("pull", "--rebase", "origin", self.current_branch())

    def push(self) -> None:
        """`git push origin <current-branch>`.

        Pushes whatever branch is checked out in the working tree. To
        push commits to a *different* remote branch, run `git push`
        directly — the SDK keeps push behavior predictable by mirroring
        the local branch name.
        """
        self._require_local()
        self._git("push", "origin", self.current_branch())

    def close(self) -> None:
        """Close the HTTP client if one was created."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _dataset_from_index(self, payload: dict[str, Any]) -> Dataset:
        """Build a Dataset object from one entry of the registry index."""
        releases = [ReleaseSummary(**r) for r in payload.get("releases", [])]
        return Dataset(
            name=payload["name"],
            description=payload["description"],
            releases=releases,
            registry=self,
        )

    def _require_local(self) -> None:
        """Guard for write/sync ops; raises if no local clone is bound."""
        if not self.is_local:
            raise RuntimeError(
                "this operation requires a local clone — pass `repo_path=` to Registry()"
            )

    def _load_index(self) -> dict[str, Any]:
        """Return the registry index, walking the tree locally or hitting raw HTTPS."""
        if self.is_local:
            assert self.repo_path is not None
            return validate_tree(self.repo_path / "datasets")
        return json.loads(self._http_get("registry.json"))

    def _http_get(self, repo_relative_path: str) -> str:
        """Fetch a file via the GitHub Contents API at the configured branch.

        We deliberately avoid `raw.githubusercontent.com`: it serves
        responses through a CDN that caches for several minutes, so a
        freshly-pushed branch looks stale for a while. The Contents API
        with `Accept: application/vnd.github.raw` returns the file body
        directly with no base64 wrapping, and reflects pushes within
        seconds. Unauthenticated callers get 60 requests/hour, which is
        comfortably above any normal read workload here.
        """
        owner, repo = self._owner_repo()
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_relative_path}"
        client = self._client()
        response = client.get(
            url,
            params={"ref": self.branch},
            headers={"Accept": "application/vnd.github.raw"},
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        return response.text

    def _client(self) -> httpx.Client:
        """Lazy-create the HTTP client used for raw fetches."""
        if self._http_client is None:
            self._http_client = httpx.Client(follow_redirects=True)
        return self._http_client

    def _owner_repo(self) -> tuple[str, str]:
        """Parse `<owner>/<repo>` out of the configured `repo_url`."""
        parts = self.repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"cannot parse owner/repo from repo_url: {self.repo_url}")
        return parts[-2], parts[-1]

    def _git(self, *args: str) -> str:
        """Run a git subcommand against the local clone; raise on non-zero exit."""
        assert self.repo_path is not None
        proc = subprocess.run(
            ["git", "-C", str(self.repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout.strip()

    def _rebuild_registry_file(self) -> Path:
        """Regenerate `registry.json` from the local tree. Returns its path.

        Called by every write op so committed snapshots are internally
        consistent — anyone reading the repo at any sha sees a registry
        that matches the tree, without depending on CI to catch up.
        """
        assert self.repo_path is not None
        payload = validate_tree(self.repo_path / "datasets")
        registry_path = self.repo_path / "registry.json"
        text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
        registry_path.write_text(text)
        return registry_path

    def _commit_with_registry(self, paths: list[Path], message: str) -> str:
        """Refresh `registry.json`, then commit it alongside `paths`."""
        registry_path = self._rebuild_registry_file()
        return self._git_commit([*paths, registry_path], message)

    def _git_commit(self, paths: list[Path], message: str) -> str:
        """Stage `paths` and commit with `message`. Returns the new commit sha.

        If staging produces no diff (idempotent edit), returns HEAD's sha
        without creating an empty commit.
        """
        assert self.repo_path is not None
        rel_paths = [str(p.relative_to(self.repo_path)) for p in paths]
        self._git("add", "--", *rel_paths)
        diff = subprocess.run(
            ["git", "-C", str(self.repo_path), "diff", "--cached", "--quiet"],
            capture_output=True,
            text=True,
        )
        if diff.returncode == 0:
            return self._git("rev-parse", "HEAD")
        self._git("commit", "-m", message)
        return self._git("rev-parse", "HEAD")

class Dataset:
    """A named series of releases."""

    def __init__(
        self,
        name: str,
        description: str,
        releases: list[ReleaseSummary],
        registry: Registry,
    ) -> None:
        self.name = name
        self.description = description
        self.releases = releases  # already sorted by release_date desc
        self._registry = registry

    @property
    def latest(self) -> Release | None:
        """The most recent release by `release_date`, or None if empty."""
        if not self.releases:
            return None
        return self.get_release(self.releases[0].id)

    def get_release(self, release_id: str) -> Release:
        """Fetch one release by id; raise KeyError if missing."""
        for summary in self.releases:
            if summary.id == release_id:
                return Release(
                    dataset_name=self.name,
                    summary=summary,
                    registry=self._registry,
                )
        raise KeyError(f"release not found: {self.name}/{release_id}")

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, releases={len(self.releases)})"


class Release:
    """One release of a dataset — fetchable, and (locally) editable."""

    def __init__(
        self,
        dataset_name: str,
        summary: ReleaseSummary,
        registry: Registry,
    ) -> None:
        self.dataset_name = dataset_name
        self.release_id = summary.id
        self.release_date = summary.release_date
        self.status = summary.status
        self.task_count = summary.task_count
        self.resolved_count = summary.resolved_count
        self._registry = registry

    @property
    def path(self) -> str:
        """Repo-relative path to this release's `tasks.jsonl`."""
        return f"datasets/{self.dataset_name}/releases/{self.release_id}/tasks.jsonl"

    def tasks(self) -> list[Task]:
        """Load and validate every task row."""
        if self._registry.is_local:
            assert self._registry.repo_path is not None
            return load_tasks(self._registry.repo_path / self.path)
        return _parse_jsonl_text(self._registry._http_get(self.path), self.path)

    def unresolved(self) -> list[Task]:
        """Convenience: tasks whose `resolved_outcome` is None."""
        return [t for t in self.tasks() if t.resolved_outcome is None]

    def set_resolved_outcome(
        self,
        task_id: str,
        value: list[str],
        *,
        resolved_at: str | None = None,
        source: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        """Attach (or overwrite) one task's `resolved_outcome`. One commit."""
        return self.set_resolved_outcomes(
            [
                {
                    "task_id": task_id,
                    "value": value,
                    "resolved_at": resolved_at,
                    "source": source,
                }
            ],
            commit_message=commit_message,
        )

    def set_resolved_outcomes(
        self,
        updates: list[dict[str, Any]],
        *,
        commit_message: str | None = None,
    ) -> str:
        """Apply many resolutions in one commit.

        Each `update` must include `task_id` and `value` (list[str]); the
        optional `resolved_at` and `source` strings pass through.
        """
        self._registry._require_local()
        assert self._registry.repo_path is not None

        if not updates:
            raise ValueError("set_resolved_outcomes requires at least one update")

        full_path = self._registry.repo_path / self.path
        rows = _read_jsonl_rows(full_path)
        index_by_id = {row["task_id"]: row for row in rows if "task_id" in row}

        for upd in updates:
            task_id = upd.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("each update needs a non-empty 'task_id'")
            if task_id not in index_by_id:
                raise KeyError(
                    f"task_id not in release {self.dataset_name}/{self.release_id}: {task_id}"
                )
            value = upd.get("value")
            if not isinstance(value, list):
                raise TypeError(
                    f"update for {task_id}: 'value' must be a list[str], "
                    f"got {type(value).__name__}"
                )
            resolved: dict[str, Any] = {"value": list(value)}
            if upd.get("resolved_at") is not None:
                resolved["resolved_at"] = upd["resolved_at"]
            if upd.get("source") is not None:
                resolved["source"] = upd["source"]
            row = index_by_id[task_id]
            row["resolved_outcome"] = resolved
            try:
                Task.model_validate(row)
            except PydanticValidationError as exc:
                raise SchemaError(f"task_id={task_id}: {exc}") from exc

        _write_jsonl_atomic(full_path, rows)

        message = commit_message or (
            f"resolve {len(updates)} task(s) in {self.dataset_name}/{self.release_id}"
        )
        return self._registry._commit_with_registry([full_path], message)

    def delete_task(self, task_id: str, *, commit_message: str | None = None) -> str:
        """Remove one task row from the release. One commit."""
        self._registry._require_local()
        assert self._registry.repo_path is not None

        full_path = self._registry.repo_path / self.path
        rows = _read_jsonl_rows(full_path)
        new_rows = [r for r in rows if r.get("task_id") != task_id]
        if len(new_rows) == len(rows):
            raise KeyError(
                f"task_id not in release {self.dataset_name}/{self.release_id}: {task_id}"
            )
        if not new_rows:
            raise ValueError(
                "deleting this task would leave the release empty — "
                "call Release.delete() instead"
            )
        _write_jsonl_atomic(full_path, new_rows)
        message = commit_message or (
            f"remove task {task_id} from {self.dataset_name}/{self.release_id}"
        )
        return self._registry._commit_with_registry([full_path], message)

    def delete(self, *, commit_message: str | None = None) -> str:
        """Remove the entire release directory. One commit."""
        self._registry._require_local()
        assert self._registry.repo_path is not None

        release_dir = (
            self._registry.repo_path / "datasets" / self.dataset_name / "releases" / self.release_id
        )
        if not release_dir.is_dir():
            raise FileNotFoundError(f"release directory missing: {release_dir}")

        # `git rm -r` stages the directory removal; we only need to also
        # stage the regenerated registry.json. Using a targeted commit
        # (not `git add -A`) avoids sweeping in any unrelated dirty files.
        rel = release_dir.relative_to(self._registry.repo_path).as_posix()
        self._registry._git("rm", "-r", "--", rel)
        registry_path = self._registry._rebuild_registry_file()
        message = commit_message or (
            f"delete release {self.dataset_name}/{self.release_id}"
        )
        return self._registry._git_commit([registry_path], message)

    def __repr__(self) -> str:
        return (
            f"Release(dataset={self.dataset_name!r}, id={self.release_id!r}, "
            f"date={self.release_date!r}, tasks={self.task_count}, "
            f"resolved={self.resolved_count})"
        )


def _parse_jsonl_text(text: str, where: str) -> list[Task]:
    """Parse and validate JSONL text (used by the remote fetch path)."""
    out: list[Task] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{where}:{line_no}: invalid JSON ({exc})") from exc
        try:
            task = Task.model_validate(row)
        except PydanticValidationError as exc:
            raise SchemaError(f"{where}:{line_no}: {exc}") from exc
        if task.task_id in seen:
            raise SchemaError(f"{where}:{line_no}: duplicate task_id '{task.task_id}'")
        seen.add(task.task_id)
        out.append(task)
    return out


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read raw dict rows from a JSONL file (preserves field order)."""
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise SchemaError(f"{path}:{line_no}: row must be a JSON object")
        rows.append(row)
    return rows


def _validate_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a sequence of task dicts and check for duplicate `task_id`s."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tasks):
        if not isinstance(row, dict):
            raise TypeError(f"tasks[{i}]: must be a dict")
        try:
            task = Task.model_validate(row)
        except PydanticValidationError as exc:
            raise SchemaError(f"tasks[{i}]: {exc}") from exc
        if task.task_id in seen:
            raise SchemaError(f"tasks[{i}]: duplicate task_id '{task.task_id}'")
        seen.add(task.task_id)
        out.append(row)
    return out


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to `path` via tmp+rename so partial writes never linger."""
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    _atomic_write_text(path, text)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to `path` via tmp+rename."""
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    """Tmp-write + rename so a crash mid-write can't leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        shutil.move(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
