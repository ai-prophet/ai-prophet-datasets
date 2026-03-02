# ai-prophet-datasets

Standardized dataset registry and dataset artifacts for `mini-prophet` eval.

## Repository layout

Use this structure for all registry-managed datasets:

```text
datasets/<dataset_name>/<version>/tasks.jsonl
```

Example:

```text
datasets/dummy/2026-03-02/tasks.jsonl
```

`registry.json` stores entries consumed by `mini-prophet` dataset resolution.

## Maintainer workflow (manual)

1. Add a new dataset JSONL under `datasets/<name>/<version>/tasks.jsonl`.
2. Commit the dataset file so you have an immutable git ref.
3. Register/update `registry.json`:

```bash
python3 scripts/registry_manager.py register \
  --dataset-path datasets/<name>/<version>/tasks.jsonl \
  --registry registry.json \
  --git-ref <commit_sha> \
  --promote-latest
```

This command validates JSONL shape, computes SHA-256, and upserts `<name>@<version>` (plus optional `<name>@latest`).

## CI automation

Workflow: `.github/workflows/dataset-registry-sync.yml`

- On `pull_request` touching `datasets/**/*.jsonl`: bot detects newly added dataset files, generates a registry suggestion report, and comments it on the PR.
- On `push` to `main` touching `datasets/**/*.jsonl`: bot auto-updates `registry.json` and commits the change.

Notes:

- Auto-detection only supports canonical dataset paths under `datasets/<name>/<version>/...jsonl`.
- Push-time auto-apply writes `git_ref` as the pushed commit SHA.
- PR comments are guidance; push-to-main performs the actual automated registry update.

## mini-prophet usage

After a dataset is in registry:

```bash
prophet datasets list
prophet datasets validate -d <name>@latest
prophet eval -d <name>@latest -o runs/<name>
```
