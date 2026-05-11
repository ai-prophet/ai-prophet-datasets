"""End-to-end local Registry tests against a real git working tree.

Each test builds a fresh repo with `git init`, exercises the SDK, and
verifies both file state and that commits actually landed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ai_prophet_datasets import Registry, SchemaError


def _commit_count(repo: Path) -> int:
    """Count commits on HEAD."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------- read ----------


def test_list_datasets(repo: Path):
    """Reading lists the seeded dataset and its release."""
    reg = Registry(repo_path=repo)
    datasets = reg.list_datasets()
    assert [d.name for d in datasets] == ["dummy"]
    assert [r.id for r in datasets[0].releases] == ["2026-04-01"]
    assert datasets[0].releases[0].task_count == 2
    assert datasets[0].releases[0].resolved_count == 1


def test_get_dataset_missing_raises(repo: Path):
    """`get_dataset` on an unknown name raises KeyError."""
    reg = Registry(repo_path=repo)
    with pytest.raises(KeyError):
        reg.get_dataset("does-not-exist")


def test_release_tasks_loads_and_validates(repo: Path):
    """`tasks()` returns validated Task objects in file order."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")
    tasks = release.tasks()
    assert [t.task_id for t in tasks] == ["t-001", "t-002"]
    assert tasks[0].resolved_outcome is None
    assert tasks[1].resolved_outcome is not None
    assert tasks[1].resolved_outcome.value == ["Home"]


def test_release_unresolved_filter(repo: Path):
    """`unresolved()` returns only tasks with no `resolved_outcome`."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")
    unresolved = release.unresolved()
    assert [t.task_id for t in unresolved] == ["t-001"]


def test_dataset_latest(repo: Path, make_repo):
    """`Dataset.latest` returns the most recent release by date."""
    multi = make_repo(
        datasets=[
            dict(name="ds", description="x", release_id="2026-01-01", release_date="2026-01-01"),
            dict(name="ds", description="x", release_id="2026-02-01", release_date="2026-02-01"),
        ]
    )
    reg = Registry(repo_path=multi)
    ds = reg.get_dataset("ds")
    assert ds.latest is not None
    assert ds.latest.release_id == "2026-02-01"


# ---------- write: create_dataset / create_release ----------


def test_create_dataset_then_release(empty_repo: Path):
    """A fresh dataset + release is two commits and produces valid files."""
    reg = Registry(repo_path=empty_repo)
    before = _commit_count(empty_repo)

    reg.create_dataset("new-ds", "A new dataset")
    reg.create_release(
        dataset="new-ds",
        release_id="2026-05-02",
        tasks=[
            {"task_id": "x1", "title": "Q?", "outcomes": ["Y", "N"]},
            {
                "task_id": "x2",
                "title": "Q2?",
                "outcomes": ["A", "B"],
                "resolved_outcome": {"value": ["A"]},
            },
        ],
        release_date="2026-05-02",
        description="hackathon",
    )

    assert _commit_count(empty_repo) == before + 2

    # Files on disk
    dataset_json = empty_repo / "datasets" / "new-ds" / "dataset.json"
    assert json.loads(dataset_json.read_text())["name"] == "new-ds"

    rel_dir = empty_repo / "datasets" / "new-ds" / "releases" / "2026-05-02"
    rows = _read_jsonl(rel_dir / "tasks.jsonl")
    assert [r["task_id"] for r in rows] == ["x1", "x2"]

    # And the SDK can read what it just wrote
    fresh = Registry(repo_path=empty_repo).get_release("new-ds", "2026-05-02")
    assert fresh.task_count == 2
    assert fresh.resolved_count == 1


def test_create_release_rejects_existing(repo: Path):
    """Creating a release that already exists should fail loudly."""
    reg = Registry(repo_path=repo)
    with pytest.raises(FileExistsError):
        reg.create_release(
            dataset="dummy",
            release_id="2026-04-01",
            tasks=[{"task_id": "x", "title": "Q?", "outcomes": ["Y"]}],
            release_date="2026-04-01",
            description="dup",
        )


def test_create_release_rejects_invalid_task(empty_repo: Path):
    """A bad row is caught before any files are written."""
    reg = Registry(repo_path=empty_repo)
    reg.create_dataset("ds", "x")

    bad = [{"task_id": "x", "title": "Q?", "outcomes": ["Y"], "resolved_outcome": {"value": ["Z"]}}]
    with pytest.raises(SchemaError):
        reg.create_release(
            dataset="ds",
            release_id="r1",
            tasks=bad,
            release_date="2026-05-02",
            description="x",
        )
    # Release dir should not exist after a failed validate.
    assert not (empty_repo / "datasets" / "ds" / "releases" / "r1").exists()


def test_create_dataset_requires_local(make_repo):
    """Remote-mode Registry refuses writes."""
    reg = Registry()  # no repo_path
    with pytest.raises(RuntimeError, match="local clone"):
        reg.create_dataset("foo", "bar")


# ---------- write: set_resolved_outcome / set_resolved_outcomes ----------


def test_set_resolved_outcome_creates_commit(repo: Path):
    """One resolution = one commit, file is rewritten in place."""
    reg = Registry(repo_path=repo)
    before = _commit_count(repo)

    sha = reg.get_release("dummy", "2026-04-01").set_resolved_outcome(
        task_id="t-001",
        value=["Yes"],
        resolved_at="2026-05-01T12:00:00Z",
        source="test",
    )

    assert _commit_count(repo) == before + 1
    assert len(sha) == 40

    rows = _read_jsonl(repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl")
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["t-001"]["resolved_outcome"] == {
        "value": ["Yes"],
        "resolved_at": "2026-05-01T12:00:00Z",
        "source": "test",
    }


def test_set_resolved_outcomes_batches_into_one_commit(repo: Path):
    """Batched resolutions land in a single commit."""
    reg = Registry(repo_path=repo)
    before = _commit_count(repo)

    reg.get_release("dummy", "2026-04-01").set_resolved_outcomes(
        [
            {"task_id": "t-001", "value": ["No"]},
            {"task_id": "t-002", "value": ["Away"]},  # overwrites the seeded resolution
        ]
    )

    assert _commit_count(repo) == before + 1

    rows = _read_jsonl(repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl")
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["t-001"]["resolved_outcome"]["value"] == ["No"]
    assert by_id["t-002"]["resolved_outcome"]["value"] == ["Away"]


def test_set_resolved_outcome_rejects_unknown_task(repo: Path):
    """An unknown task_id raises KeyError before any file write."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")
    with pytest.raises(KeyError):
        release.set_resolved_outcome(task_id="ghost", value=["Yes"])


def test_set_resolved_outcome_rejects_value_not_in_outcomes(repo: Path):
    """`value` must be a subset of the task's `outcomes`."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")
    with pytest.raises(SchemaError):
        release.set_resolved_outcome(task_id="t-001", value=["Maybe"])  # not in [Yes, No]


def test_set_resolved_outcome_rejects_string_value(repo: Path):
    """Bare strings for `value` raise — the SDK enforces list shape."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")
    with pytest.raises(TypeError):
        release.set_resolved_outcome(task_id="t-001", value="Yes")  # type: ignore[arg-type]


def test_set_resolved_outcome_preserves_other_fields(repo: Path, make_repo):
    """Mutating one row leaves other rows' fields untouched (key order, extras)."""
    reg = Registry(repo_path=repo)
    release = reg.get_release("dummy", "2026-04-01")

    # Add an extra unknown field to t-001 first — write a new tasks file.
    tasks_path = repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl"
    rows = _read_jsonl(tasks_path)
    rows[0]["custom_field"] = {"keep": "me"}
    tasks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    subprocess.run(["git", "-C", str(repo), "commit", "-am", "add custom"], check=True, capture_output=True)

    release.set_resolved_outcome(task_id="t-001", value=["No"])

    rows = _read_jsonl(tasks_path)
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["t-001"]["custom_field"] == {"keep": "me"}
    # The other row's resolved_outcome stayed put
    assert by_id["t-002"]["resolved_outcome"]["value"] == ["Home"]


# ---------- write: deletes ----------


def test_delete_task(repo: Path):
    """Deleting a task removes the row and commits."""
    reg = Registry(repo_path=repo)
    before = _commit_count(repo)

    reg.get_release("dummy", "2026-04-01").delete_task("t-001")

    assert _commit_count(repo) == before + 1
    rows = _read_jsonl(repo / "datasets" / "dummy" / "releases" / "2026-04-01" / "tasks.jsonl")
    assert [r["task_id"] for r in rows] == ["t-002"]


def test_delete_last_task_blocked(make_repo):
    """Deleting the only task in a release is refused — caller must delete the release."""
    single = make_repo(
        datasets=[
            dict(
                name="solo",
                description="x",
                release_id="r1",
                release_date="2026-01-01",
                tasks=[{"task_id": "only", "title": "Q?", "outcomes": ["Y"]}],
            )
        ]
    )
    reg = Registry(repo_path=single)
    release = reg.get_release("solo", "r1")
    with pytest.raises(ValueError, match="empty"):
        release.delete_task("only")


def test_delete_release(repo: Path):
    """Deleting a release removes the directory and commits."""
    reg = Registry(repo_path=repo)
    release_dir = repo / "datasets" / "dummy" / "releases" / "2026-04-01"
    assert release_dir.exists()

    reg.get_release("dummy", "2026-04-01").delete()

    assert not release_dir.exists()
    # Dataset still exists (only the release went away)
    assert (repo / "datasets" / "dummy" / "dataset.json").exists()


# ---------- registry.json freshness (dogfood regression) ----------


def test_writes_update_registry_json(empty_repo: Path):
    """Every write op should leave registry.json in sync with the tree.

    Regression: an earlier version of the SDK only touched tasks.jsonl
    on writes, so remote consumers (who read via registry.json) couldn't
    see a freshly-pushed dataset until CI rebuilt the index.
    """
    reg = Registry(repo_path=empty_repo)
    reg.create_dataset("new-ds", "x")
    after_create = json.loads((empty_repo / "registry.json").read_text())
    assert [d["name"] for d in after_create["datasets"]] == ["new-ds"]

    reg.create_release(
        dataset="new-ds",
        release_id="r1",
        tasks=[{"task_id": "t", "title": "Q?", "outcomes": ["A", "B"]}],
        release_date="2026-05-02",
        description="x",
    )
    after_release = json.loads((empty_repo / "registry.json").read_text())
    releases = after_release["datasets"][0]["releases"]
    assert releases[0]["task_count"] == 1
    assert releases[0]["resolved_count"] == 0

    reg.get_release("new-ds", "r1").set_resolved_outcome("t", value=["A"])
    after_resolve = json.loads((empty_repo / "registry.json").read_text())
    assert after_resolve["datasets"][0]["releases"][0]["resolved_count"] == 1


def test_delete_release_does_not_sweep_unrelated_dirty_files(repo: Path):
    """Regression: `Release.delete()` previously used `git add -A` and
    bundled any unrelated dirty file into the delete commit."""
    # Create an unrelated dirty file that should NOT be committed.
    stray = repo / "stray-untracked-file.txt"
    stray.write_text("do not commit me")

    reg = Registry(repo_path=repo)
    reg.get_release("dummy", "2026-04-01").delete()

    # The stray file should still be on disk, untracked.
    assert stray.exists()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    assert "stray-untracked-file.txt" in status  # still untracked


def test_push_uses_current_branch(repo: Path):
    """Regression: `Registry.push()` should push the *checked-out* branch,
    not whatever `self.branch` happens to be."""
    # Switch off the default `main` so a mismatch could surface.
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", "feature/x"],
        check=True,
        capture_output=True,
    )

    reg = Registry(repo_path=repo, branch="main")  # deliberately mismatched
    reg.get_release("dummy", "2026-04-01").set_resolved_outcome("t-001", value=["Yes"])
    reg.push()

    # Local feature/x and remote feature/x should match.
    local = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    origin_path = repo.parent / f"origin{repo.name.replace('repo', '')}.git"
    remote = subprocess.check_output(
        ["git", "-C", str(origin_path), "rev-parse", "feature/x"], text=True
    ).strip()
    assert local == remote


# ---------- push integration ----------


def test_push_propagates_to_origin(repo: Path):
    """An end-to-end commit + push lands on the bare origin."""
    reg = Registry(repo_path=repo)
    reg.get_release("dummy", "2026-04-01").set_resolved_outcome(
        task_id="t-001", value=["Yes"]
    )
    reg.push()
    # The bare origin's HEAD should now match local HEAD
    local = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    origin_path = repo.parent / f"origin{repo.name.replace('repo', '')}.git"
    remote = subprocess.check_output(
        ["git", "-C", str(origin_path), "rev-parse", "main"], text=True
    ).strip()
    assert local == remote
