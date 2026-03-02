# ai-prophet-datasets

Standardized dataset artifacts and registry for `mini-prophet` eval.

## Repository layout

```text
datasets/<dataset_name>/metadata.json
datasets/<dataset_name>/<version>/tasks.jsonl
```

`metadata.json` schema:

```json
{
  "description": "One-line dataset description",
  "latest": "2026-03-02"
}
```

`latest` must be an existing version folder under `datasets/<dataset_name>/`.

## Registry format

`registry.json` uses grouped datasets:

```json
{
  "datasets": [
    {
      "name": "dummy",
      "description": "Dummy benchmark dataset",
      "latest": "2026-03-02",
      "versions": [
        {
          "version": "2026-03-01",
          "git_url": "https://github.com/ai-prophet/ai-prophet-datasets.git",
          "git_ref": "<sha>",
          "path": "datasets/dummy/2026-03-01/tasks.jsonl",
          "checksum_sha256": "<sha256>"
        }
      ]
    }
  ]
}
```

## Maintainer workflow

1. Add dataset version file at `datasets/<name>/<version>/tasks.jsonl`.
2. Update `datasets/<name>/metadata.json` (`description`, `latest`).
3. Open PR. GitHub Action comments suggested grouped registry blocks.
4. Merge PR. On push to `main`, action auto-updates `registry.json` and commits.

## Manual command

```bash
python3 scripts/registry_manager.py register \
  --dataset-path datasets/<name>/<version>/tasks.jsonl \
  --registry registry.json \
  --git-ref <commit_sha>
```

This validates JSONL, reads metadata, computes checksum, and upserts the grouped registry entry.

## CI automation

Workflow: `.github/workflows/dataset-registry-sync.yml`

- `pull_request`: detect added `datasets/**/*.jsonl`, validate, and comment suggested grouped dataset blocks.
- `push` to `main`: detect added `datasets/**/*.jsonl` and apply grouped registry updates automatically.
