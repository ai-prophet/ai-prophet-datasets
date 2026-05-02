# ai-prophet-datasets

Python SDK for the [ai-prophet forecasting datasets registry](https://github.com/ai-prophet/ai-prophet-datasets).

## Install

```bash
pip install ai-prophet-datasets
```

## Quick start

### Read (no auth needed — the upstream repo is public)

```python
from ai_prophet_datasets import Registry

reg = Registry()                                     # defaults to ai-prophet/ai-prophet-datasets
for dataset in reg.list_datasets():
    print(dataset.name, [r.id for r in dataset.releases])

release = reg.get_release("prophet-recent", "v1.0.0")
for task in release.tasks():
    print(task.task_id, task.title, task.resolved_outcome)
```

### Write (requires a local clone)

```python
from ai_prophet_datasets import Registry

reg = Registry(repo_path="/path/to/ai-prophet-datasets")  # local mode

# Create a brand new release
reg.create_release(
    dataset="hackathon-day",
    release_id="2026-05-02",
    tasks=[
        {
            "task_id": "evt-001",
            "title": "Will it rain in Chicago tomorrow?",
            "outcomes": ["Yes", "No"],
        },
    ],
    release_date="2026-05-02",
    description="Hackathon day 2",
)

# Attach a resolution
release = reg.get_release("hackathon-day", "2026-05-02")
release.set_resolved_outcome(
    task_id="evt-001",
    value=["Yes"],
    resolved_at="2026-05-03T12:00:00Z",
    source="weather.gov",
)

# Push commits when ready
reg.push()
```

`value` is always a list of strings. Single-outcome resolutions are
wrapped (`["Yes"]`, never `"Yes"`); multi-entry lists express
multi-correct or partial resolutions.

## CLI

```bash
ai-prophet-datasets list
ai-prophet-datasets fetch prophet-recent v1.0.0 -o tasks.jsonl
ai-prophet-datasets validate --repo-path .
ai-prophet-datasets --repo-path . resolve hackathon-day 2026-05-02 \
    --task-id evt-001 --value Yes --source weather.gov
```

## What this SDK does

* **Reads** dataset metadata, release metadata, and task rows — locally
  from a clone, or remotely from `raw.githubusercontent.com`.
* **Validates** every read through pydantic models (`task_id` required
  and unique; `resolved_outcome.value ⊆ outcomes`; etc.).
* **Writes** via a local clone: each create / edit / delete is one git
  commit. Pushing is a separate explicit step (`Registry.push()`).

## What it deliberately doesn't do

* No edits to fields other than `resolved_outcome`. Renaming a title
  means publishing a new release.
* No automatic conflict resolution on push. Call `Registry.sync()`
  yourself if you need to rebase.
* No service / no database. The repo *is* the database; git is the
  audit log.
