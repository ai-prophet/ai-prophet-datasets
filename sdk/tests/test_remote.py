"""Remote-mode Registry tests using httpx MockTransport.

We don't hit the real network — the transport routes raw.githubusercontent
URLs to fixture content built fresh per test. This still exercises the
real URL construction, response parsing, and validation paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ai_prophet_datasets import Registry, SchemaError


def _make_registry_with_files(files: dict[str, str]) -> Registry:
    """Build a Registry whose HTTP client serves `files` from a fake transport.

    Keys are repo-relative paths (e.g. `registry.json`,
    `datasets/foo/releases/r1/tasks.jsonl`). Missing paths return 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # URL: https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>
        parts = request.url.path.lstrip("/").split("/", 3)
        if len(parts) < 4:
            return httpx.Response(404)
        repo_relative = parts[3]
        if repo_relative not in files:
            return httpx.Response(404, text=f"missing: {repo_relative}")
        return httpx.Response(200, text=files[repo_relative])

    reg = Registry()
    reg._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return reg


def _registry_index() -> dict:
    """A small index: one dataset, one release, with both kinds of tasks."""
    return {
        "datasets": [
            {
                "name": "dummy",
                "description": "Dummy dataset",
                "releases": [
                    {
                        "id": "r1",
                        "release_date": "2026-04-01",
                        "status": "open",
                        "task_count": 2,
                        "resolved_count": 1,
                        "path": "datasets/dummy/releases/r1/tasks.jsonl",
                    }
                ],
            }
        ]
    }


def _tasks_jsonl() -> str:
    rows = [
        {"task_id": "t-001", "title": "Q1?", "outcomes": ["Yes", "No"]},
        {
            "task_id": "t-002",
            "title": "Q2?",
            "outcomes": ["Home", "Away"],
            "resolved_outcome": {"value": ["Home"]},
        },
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def test_remote_list_datasets():
    """A read-only Registry parses the index and exposes Dataset objects."""
    reg = _make_registry_with_files({"registry.json": json.dumps(_registry_index())})
    datasets = reg.list_datasets()
    assert [d.name for d in datasets] == ["dummy"]
    assert datasets[0].releases[0].task_count == 2


def test_remote_release_tasks():
    """Fetching tasks via the remote backend validates and returns Task objects."""
    reg = _make_registry_with_files(
        {
            "registry.json": json.dumps(_registry_index()),
            "datasets/dummy/releases/r1/tasks.jsonl": _tasks_jsonl(),
        }
    )
    release = reg.get_release("dummy", "r1")
    tasks = release.tasks()
    assert [t.task_id for t in tasks] == ["t-001", "t-002"]
    assert tasks[1].resolved_outcome is not None
    assert tasks[1].resolved_outcome.value == ["Home"]


def test_remote_invalid_jsonl_raises():
    """Bad JSONL content over HTTP surfaces as SchemaError, not a parse exception."""
    bad = (
        json.dumps({"task_id": "t1", "title": "Q?", "outcomes": ["A"]})
        + "\nnot valid json\n"
    )
    reg = _make_registry_with_files(
        {
            "registry.json": json.dumps(_registry_index()),
            "datasets/dummy/releases/r1/tasks.jsonl": bad,
        }
    )
    release = reg.get_release("dummy", "r1")
    with pytest.raises(SchemaError):
        release.tasks()


def test_remote_writes_refused():
    """Remote-mode Registry must refuse every write op."""
    reg = _make_registry_with_files({"registry.json": json.dumps(_registry_index())})
    with pytest.raises(RuntimeError, match="local clone"):
        reg.create_release(
            dataset="dummy",
            release_id="rX",
            tasks=[{"task_id": "x", "title": "Q?", "outcomes": ["A"]}],
            release_date="2026-05-02",
            description="x",
        )


def test_remote_404_propagates():
    """Missing remote files surface as HTTP errors."""
    reg = _make_registry_with_files({})  # no registry.json
    with pytest.raises(httpx.HTTPStatusError):
        reg.list_datasets()
