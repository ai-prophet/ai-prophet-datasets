# ai-prophet-datasets SDK — Detailed Guide

This document covers programmatic usage of the SDK in depth: every
class, every method, the failure modes you should be ready to catch,
and the workflow recipes the SDK is designed around. For a quick start
see [`README.md`](README.md).

## Contents

- [Installation](#installation)
- [Two execution modes](#two-execution-modes)
- [Authentication](#authentication)
- [API reference](#api-reference)
  - [`Registry`](#registry)
  - [`Dataset`](#dataset)
  - [`Release`](#release)
  - [`Task` and `ResolvedOutcome`](#task-and-resolvedoutcome)
- [Common workflows](#common-workflows)
  - [Read: enumerate datasets and tasks](#read-enumerate-datasets-and-tasks)
  - [Write: publish a hackathon-day release](#write-publish-a-hackathon-day-release)
  - [Write: resolution-bot pattern](#write-resolution-bot-pattern)
  - [Edit: idempotent re-runs](#edit-idempotent-re-runs)
- [Concurrency, sync, and push](#concurrency-sync-and-push)
- [Validation and error reference](#validation-and-error-reference)
- [What each write op touches](#what-each-write-op-touches)
- [CLI reference](#cli-reference)

## Installation

```bash
pip install ai-prophet-datasets
# or, for local development against this repo:
pip install -e ./sdk
```

Requires Python 3.10+. Runtime deps: `pydantic>=2.0`, `httpx>=0.25`.

## Two execution modes

The same `Registry` class drives both modes; which one you get is
determined by whether you pass `repo_path`.

| | Remote mode | Local mode |
|---|---|---|
| Constructor | `Registry()` | `Registry(repo_path="/path/to/clone")` |
| Reads | GitHub Contents API (always fresh) | Direct filesystem reads + tree walk |
| Writes | Refused (RuntimeError) | One git commit per operation |
| Auth needed | No (public repo) | Whatever `git push` is already configured for |
| Use when | Hackathon participant, eval pipeline, CI consumer | Maintainer, resolution bot, anything that mutates |

You can hold a `Registry` open in a `with` block; it closes the HTTP
client on exit:

```python
with Registry() as reg:
    for d in reg.list_datasets():
        print(d.name)
```

## Authentication

**Reads** never need auth — the upstream repo is public.

**Writes** rely on whatever git already knows how to do. The SDK runs
`git` as a subprocess; it does not parse, store, or pass credentials.
That means any of the standard mechanisms work:

- An **SSH key** registered on the repo (most laptops).
- A **GitHub App installation token** (preferred for bots; org-owned,
  short-lived, scoped, easy to revoke). Mint a token at start of each
  bot run and configure git's credential helper or set
  `GIT_ASKPASS` to a script that echoes the token.
- A **fine-grained PAT** scoped to this repo with `contents: write`,
  exposed to git the same way as a GitHub App token.

When the SDK calls `Registry.push()`, the command it runs is
`git -C <repo_path> push origin <current-branch>`. If that command
works in your shell, the SDK works.

## API reference

### `Registry`

```python
Registry(
    repo_path: str | os.PathLike | None = None,
    *,
    repo_url: str = "https://github.com/ai-prophet/ai-prophet-datasets",
    branch: str = "main",
    http_timeout: float = 30.0,
)
```

- `repo_path`: path to a local git working tree. If `None`, the
  Registry is read-only.
- `repo_url`: the GitHub repo URL. Override to point at a fork.
- `branch`: branch to *read from* in remote mode. (For writes the SDK
  always uses the currently checked-out local branch — `branch` is
  irrelevant in local mode.)
- `http_timeout`: per-request timeout for remote reads, in seconds.

#### Reads

| Method | Returns | Notes |
|---|---|---|
| `list_datasets()` | `list[Dataset]` | Reads `registry.json`. Local mode walks the tree (always consistent with on-disk files); remote mode hits the Contents API. |
| `get_dataset(name)` | `Dataset` | `KeyError` if missing. |
| `get_release(dataset, release_id)` | `Release` | Convenience for `get_dataset(d).get_release(r)`. |

#### Writes (local only)

| Method | Returns | Behavior |
|---|---|---|
| `create_dataset(name, description, *, commit_message=None)` | commit sha | Creates `datasets/<name>/dataset.json`. Refuses if the directory already exists. |
| `create_release(dataset, release_id, tasks, *, release_date, description, status="open", commit_message=None)` | commit sha | Validates every row, writes `release.json` + `tasks.jsonl`. Refuses if the release already exists. |

Every write commits `registry.json` in the same commit, so any
snapshot of the repo at any sha is internally consistent.

#### Git operations (local only)

| Method | Behavior |
|---|---|
| `current_branch()` | Returns the checked-out branch name (e.g. `"main"`). |
| `sync()` | `git pull --rebase origin <current-branch>`. |
| `push()` | `git push origin <current-branch>`. |
| `close()` | Close the HTTP client (also done by `__exit__`). |

`push()` does **not** retry on conflict — see the
[concurrency section](#concurrency-sync-and-push) for the recommended
loop.

### `Dataset`

Returned by `Registry.list_datasets()` and `Registry.get_dataset()`.

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Stable identifier (matches `datasets/<name>/`). |
| `description` | `str` | One-liner from `dataset.json`. |
| `releases` | `list[ReleaseSummary]` | Sorted by `release_date` descending — `releases[0]` is the most recent. |

| Method | Returns | Notes |
|---|---|---|
| `latest` (property) | `Release | None` | `None` if the dataset has zero releases. |
| `get_release(release_id)` | `Release` | `KeyError` if missing. |

`ReleaseSummary` is a frozen dataclass with `id`, `release_date`,
`status`, `task_count`, `resolved_count`, `path` — useful for cheap
overviews without loading every task.

### `Release`

| Attribute | Type |
|---|---|
| `dataset_name` | `str` |
| `release_id` | `str` |
| `release_date` | `str` |
| `status` | `"open" | "closed" | "archived"` |
| `task_count` | `int` |
| `resolved_count` | `int` |
| `path` (property) | `str` — repo-relative path to `tasks.jsonl` |

#### Reads

| Method | Returns | Notes |
|---|---|---|
| `tasks()` | `list[Task]` | Loads the full file, validates every row. |
| `unresolved()` | `list[Task]` | Filters `tasks()` to rows with `resolved_outcome is None`. |

#### Writes (local only)

| Method | Returns | Behavior |
|---|---|---|
| `set_resolved_outcome(task_id, value, *, resolved_at=None, source=None, commit_message=None)` | commit sha | Attach (or overwrite) one task's resolution. |
| `set_resolved_outcomes(updates, *, commit_message=None)` | commit sha | Batch many resolutions into a single commit. |
| `delete_task(task_id, *, commit_message=None)` | commit sha | Remove one row. Refuses if it would empty the release — call `delete()` instead. |
| `delete(*, commit_message=None)` | commit sha | Remove the entire release directory. The dataset itself stays. |

`updates` for the batch method is a list of dicts, each shaped like:

```python
{"task_id": "...", "value": ["..."], "resolved_at": "...", "source": "..."}
```

`resolved_at` and `source` are optional.

### `Task` and `ResolvedOutcome`

Both are pydantic models (`extra="allow"` for `Task`, so unknown fields
pass through unchanged on read).

```python
class Task(BaseModel):
    task_id: str
    title: str
    outcomes: list[str]
    resolved_outcome: ResolvedOutcome | None = None
    # plus any other fields present in the row

class ResolvedOutcome(BaseModel):
    value: list[str]               # always a list, even for single resolutions
    resolved_at: str | None = None # ISO-8601 recommended
    source: str | None = None      # free-form provenance
```

Invariants the SDK enforces on every read and write:

- `task_id` non-empty, unique within the release.
- `outcomes` non-empty; every entry is a non-empty string.
- `resolved_outcome.value` is a non-empty list with no duplicates.
- Every entry of `resolved_outcome.value` must appear in `outcomes`.

`Task.to_dict()` serializes back to a JSON-friendly dict, omitting
`None` fields.

## Common workflows

### Read: enumerate datasets and tasks

Remote, no auth, fresh data:

```python
from ai_prophet_datasets import Registry

with Registry() as reg:
    for dataset in reg.list_datasets():
        print(f"{dataset.name}: {len(dataset.releases)} release(s)")
        latest = dataset.latest
        if latest is None:
            continue
        for task in latest.tasks():
            outcome = "?" if task.resolved_outcome is None else task.resolved_outcome.value
            print(f"  {task.task_id}: {task.title} -> {outcome}")
```

If you need to pin to a specific snapshot (e.g. for a reproducible
eval), pass `branch=<sha>`:

```python
reg = Registry(branch="9ca87d6a")  # any commit sha works
```

### Write: publish a hackathon-day release

The hackathon flow is one new release per day. Each release has its
own `release_id` (typically the date) and its own task set; consecutive
days are independent batches under the same dataset.

```python
from ai_prophet_datasets import Registry

reg = Registry(repo_path="/path/to/ai-prophet-datasets")

# First time only — create the dataset namespace
try:
    reg.create_dataset("hackathon-day", "Daily hackathon forecasting tasks")
except FileExistsError:
    pass  # dataset already exists from earlier days

# Today's release
reg.create_release(
    dataset="hackathon-day",
    release_id="2026-05-12",
    release_date="2026-05-12",
    description="Day 3 — markets and weather",
    tasks=load_today_tasks(),  # list[dict], your responsibility
    status="open",
)

reg.push()
```

Each call is one git commit; the two-commit sequence above is the
standard "new day" flow.

### Write: resolution-bot pattern

A bot polls upstream sources and attaches resolutions as events
resolve. The pattern is to **batch resolutions into one commit per
release per poll cycle**:

```python
from ai_prophet_datasets import Registry

reg = Registry(repo_path="/path/to/clone")

def resolve_tick():
    # Pull first so we're rebased on the latest main
    reg.sync()

    for dataset in reg.list_datasets():
        for summary in dataset.releases:
            release = dataset.get_release(summary.id)
            updates = []
            for task in release.unresolved():
                resolution = fetch_upstream_resolution(task)  # your code
                if resolution is None:
                    continue
                updates.append({
                    "task_id": task.task_id,
                    "value": resolution.value,           # list[str]
                    "resolved_at": resolution.timestamp, # ISO-8601
                    "source": resolution.source_id,
                })
            if updates:
                release.set_resolved_outcomes(
                    updates,
                    commit_message=f"bot: resolve {len(updates)} in {dataset.name}/{summary.id}",
                )

    reg.push()
```

Key points:

- `release.unresolved()` skips tasks that already have a resolution,
  so re-running the bot is safe.
- One commit per release keeps history readable and bounded — a 10-day
  hackathon with 50 resolutions/day across 3 releases is ~30 commits,
  not 1,500.
- Build all updates in memory before calling `set_resolved_outcomes`
  so partial failures don't leave the release half-written. The SDK
  validates every row before touching disk.

### Edit: idempotent re-runs

Sometimes you re-run a batch and want it to be a no-op when nothing
changed. The SDK's write methods are already idempotent at the commit
level: if the resulting file content matches what's already on disk,
`git diff --cached --quiet` returns 0 and no new commit is created.
The method still returns a sha — it's just the unchanged `HEAD`.

```python
sha_before = reg._git("rev-parse", "HEAD")  # internal, but illustrative
release.set_resolved_outcome("t-001", value=["Yes"], source="x")
release.set_resolved_outcome("t-001", value=["Yes"], source="x")  # same values
sha_after = reg._git("rev-parse", "HEAD")
assert sha_before == sha_after  # second call was a no-op
```

This makes "fix a typo and re-run the bot" safe: only the genuinely
changed releases produce new commits.

## Concurrency, sync, and push

The SDK does not lock the remote. Two writers can race and one will
get a non-fast-forward push rejection. The recommended pattern:

```python
import time
from subprocess import CalledProcessError

def push_with_retry(reg, attempts=3, base_delay=2):
    for attempt in range(attempts):
        try:
            reg.push()
            return
        except RuntimeError as exc:
            if "non-fast-forward" not in str(exc) and "rejected" not in str(exc):
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
            reg.sync()  # pulls --rebase, replays your commits on top
```

Notes:

- `reg.sync()` runs `git pull --rebase origin <current-branch>`. If
  the rebase fails because of conflicting changes to the same file,
  it'll surface as a `RuntimeError` and the working tree will be in
  rebase-in-progress state — clean it up manually with `git rebase
  --abort` (or commit a fix). The SDK deliberately does not abort for
  you, because the right resolution depends on context.
- For an unattended bot, prefer **wide commit scope** (whole release
  per cycle) over **narrow scope** (per-task commits) — fewer commits
  to rebase on contention.
- If you frequently see contention, run the bot against a dedicated
  branch and merge to `main` periodically. Each branch's
  `registry.json` is self-consistent, and CI rebuilds it on push to
  `main`.

## Validation and error reference

The SDK validates aggressively on read **and** before any file write,
so an exception surfaces at the API call rather than later as a
corrupt commit.

| Exception | When you'll see it |
|---|---|
| `KeyError` | `get_dataset` / `get_release` for a missing name; `set_resolved_outcome` / `delete_task` for a `task_id` not in the release. |
| `FileExistsError` | `create_dataset` / `create_release` when the target directory already exists. |
| `FileNotFoundError` | `create_release` when the dataset hasn't been created yet; `Release.delete` when the release directory is somehow already gone. |
| `ValueError` | Empty `tasks` list, empty name/description, deleting the only task in a release, etc. |
| `TypeError` | Passing a non-list to `resolved_outcome.value` (e.g. a bare string). |
| `SchemaError` | On-disk content fails validation (duplicate `task_id`, `value` not in `outcomes`, malformed JSONL, etc.). The message includes the offending file path and line number. |
| `RuntimeError` | Attempted a write on a remote-mode Registry; or `git` subprocess returned non-zero (the stderr is included in the message). |
| `pydantic.ValidationError` | Should be rare in practice — the SDK catches and re-raises as `SchemaError`. If you see it leak out, it's a bug. |
| `httpx.HTTPStatusError` | Remote read for a path that doesn't exist (404), or a transient API error. |

Validation that runs on every full-tree walk:

- `dataset.json` exists; `name` matches its directory.
- Each release directory has a valid `release.json` whose
  `release_id` matches the directory.
- Each release's `tasks.jsonl` parses, has ≥1 row, and every row
  passes the `Task` schema.
- `task_id` is unique within each release.

## What each write op touches

Every write produces exactly one git commit, including a refreshed
`registry.json`:

| Operation | Files in the commit |
|---|---|
| `create_dataset` | `datasets/<name>/dataset.json`, `registry.json` |
| `create_release` | `datasets/<name>/releases/<id>/release.json`, `…/tasks.jsonl`, `registry.json` |
| `set_resolved_outcome(s)` | `datasets/<name>/releases/<id>/tasks.jsonl`, `registry.json` |
| `delete_task` | `datasets/<name>/releases/<id>/tasks.jsonl`, `registry.json` |
| `Release.delete()` | the release directory (removed), `registry.json` |

`registry.json` is always rebuilt from the tree by walking
`datasets/`, so it can't drift from the underlying files within a
commit. CI's `rebuild-registry` step on `push` to `main` is effectively
a no-op on commits made by the SDK.

## CLI reference

The CLI mirrors a subset of the Python API. Global flags
(`--repo-path`, `--repo-url`, `--branch`) can appear either before or
after the subcommand.

```bash
# Reads (remote, no auth)
ai-prophet-datasets list
ai-prophet-datasets list --branch some-feature-branch
ai-prophet-datasets fetch prophet-recent v1.0.0 -o tasks.jsonl

# Reads (local)
ai-prophet-datasets --repo-path . list
ai-prophet-datasets validate --release datasets/dummy/releases/2026-03-01

# Whole-tree validation
ai-prophet-datasets --repo-path . validate

# Local writes
ai-prophet-datasets --repo-path . resolve hackathon-day 2026-05-12 \
    --task-id rain-today --value Yes --source weather.gov

# Registry rebuild (idempotent on healthy commits)
ai-prophet-datasets --repo-path . rebuild-registry
```

Exit codes: `0` success, `1` validation error, `2` usage error.

The CLI is intentionally a thin wrapper — for anything beyond the
five commands above, use the Python API directly. The CLI doesn't
have a one-shot "create release from a JSONL file" because the
programmatic version (read your input, build a list of dicts, call
`create_release`) is two lines and gives you better error handling.
