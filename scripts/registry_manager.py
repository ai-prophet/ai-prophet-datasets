#!/usr/bin/env python3
"""Registry automation for ai-prophet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_PATH_RE = re.compile(
    r"^datasets/(?P<name>[A-Za-z0-9._-]+)/(?P<version>[A-Za-z0-9._-]+)/(?P<filename>[^/]+\.jsonl)$"
)


@dataclass(frozen=True)
class DatasetMetadata:
    description: str
    latest: str


@dataclass(frozen=True)
class DatasetVersionSuggestion:
    name: str
    version: str
    git_url: str
    git_ref: str
    path: str
    checksum_sha256: str

    def to_version_entry(self) -> dict[str, str]:
        return {
            "version": self.version,
            "git_url": self.git_url,
            "git_ref": self.git_ref,
            "path": self.path,
            "checksum_sha256": self.checksum_sha256,
        }


class RegistryError(RuntimeError):
    pass


def _git_output(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RegistryError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _normalize_git_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned.startswith("git@github.com:"):
        cleaned = "https://github.com/" + cleaned.split("git@github.com:", 1)[1]
    if cleaned.startswith("https://github.com/") and not cleaned.endswith(".git"):
        cleaned = cleaned + ".git"
    return cleaned


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_jsonl(path: Path) -> None:
    errors: list[str] = []
    line_count = 0
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        line_count += 1

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue

        if not isinstance(payload, dict):
            errors.append(f"line {line_no}: row must be a JSON object")
            continue

        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"line {line_no}: missing/invalid non-empty 'title'")

        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list):
            errors.append(f"line {line_no}: missing/invalid 'outcomes' list")
        else:
            cleaned = [x for x in outcomes if isinstance(x, str) and x.strip()]
            if len(cleaned) < 1:
                errors.append(f"line {line_no}: 'outcomes' needs at least 1 non-empty string")

        task_id = payload.get("task_id")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            errors.append(f"line {line_no}: 'task_id' must be a non-empty string when provided")

        predict_by = payload.get("predict_by")
        if predict_by is not None and (not isinstance(predict_by, str) or not predict_by.strip()):
            errors.append(f"line {line_no}: 'predict_by' must be a non-empty string when provided")

    if line_count == 0:
        errors.append("dataset file has no JSONL records")

    if errors:
        raise RegistryError("Dataset validation failed:\n- " + "\n- ".join(errors))


def _parse_dataset_path(path: str) -> tuple[str, str]:
    normalized = Path(path).as_posix().lstrip("./")
    match = DATASET_PATH_RE.match(normalized)
    if match is None:
        raise RegistryError(
            "Dataset path must match datasets/<name>/<version>/<file>.jsonl; "
            f"got '{path}'"
        )
    return match.group("name"), match.group("version")


def _load_metadata(name: str) -> DatasetMetadata:
    metadata_path = Path("datasets") / name / "metadata.json"
    if not metadata_path.exists():
        raise RegistryError(
            f"Missing metadata file for dataset '{name}': {metadata_path} "
            "(required keys: description, latest)"
        )

    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid JSON in {metadata_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RegistryError(f"Metadata at {metadata_path} must be a JSON object")

    description = payload.get("description")
    latest = payload.get("latest")

    if not isinstance(description, str) or not description.strip():
        raise RegistryError(f"{metadata_path}: 'description' must be a non-empty string")
    if not isinstance(latest, str) or not latest.strip():
        raise RegistryError(f"{metadata_path}: 'latest' must be a non-empty string")

    latest_version_dir = Path("datasets") / name / latest
    if not latest_version_dir.is_dir():
        raise RegistryError(
            f"{metadata_path}: latest='{latest}' is not a valid dataset version folder under "
            f"datasets/{name}/"
        )

    return DatasetMetadata(description=description.strip(), latest=latest.strip())


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text().strip():
        return []

    payload = json.loads(path.read_text())
    data = payload.get("datasets") if isinstance(payload, dict) else payload

    if not isinstance(data, list):
        raise RegistryError("registry.json must contain a list at top-level or at key 'datasets'")

    if data and isinstance(data[0], dict) and "versions" in data[0]:
        return data

    # Legacy flat format conversion.
    grouped: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        version = item.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        if version == "latest":
            continue

        dataset = grouped.setdefault(
            name,
            {
                "name": name,
                "description": item.get("description", "") if isinstance(item.get("description"), str) else "",
                "latest": None,
                "versions": [],
            },
        )

        dataset["versions"].append(
            {
                "version": version,
                "git_url": item.get("git_url", ""),
                "git_ref": item.get("git_ref", ""),
                "path": item.get("path", ""),
                "checksum_sha256": item.get("checksum_sha256"),
            }
        )

    return sorted(grouped.values(), key=lambda d: d["name"])


def _sort_versions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda x: str(x.get("version", "")))


def _save_registry(path: Path, datasets: list[dict[str, Any]]) -> None:
    normalized: list[dict[str, Any]] = []
    for dataset in sorted(datasets, key=lambda d: str(d.get("name", ""))):
        copy = dict(dataset)
        versions = copy.get("versions")
        if not isinstance(versions, list):
            versions = []
        copy["versions"] = _sort_versions(versions)
        normalized.append(copy)

    payload: dict[str, Any] = {"datasets": normalized}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _upsert_dataset_version(
    datasets: list[dict[str, Any]],
    *,
    suggestion: DatasetVersionSuggestion,
    metadata: DatasetMetadata,
) -> None:
    target: dict[str, Any] | None = None
    for dataset in datasets:
        if dataset.get("name") == suggestion.name:
            target = dataset
            break

    if target is None:
        target = {
            "name": suggestion.name,
            "description": metadata.description,
            "latest": metadata.latest,
            "versions": [],
        }
        datasets.append(target)

    target["description"] = metadata.description
    target["latest"] = metadata.latest

    versions = target.get("versions")
    if not isinstance(versions, list):
        versions = []
        target["versions"] = versions

    entry = suggestion.to_version_entry()
    for i, existing in enumerate(versions):
        if existing.get("version") == suggestion.version:
            versions[i] = entry
            break
    else:
        versions.append(entry)


def _added_jsonl_files(base: str, head: str) -> list[str]:
    diff = _git_output(["diff", "--name-status", "--find-renames", f"{base}..{head}"])
    out: list[str] = []
    for row in diff.splitlines():
        parts = row.split("\t")
        if not parts:
            continue

        status = parts[0]
        if status.startswith(("A", "C")) and len(parts) >= 2:
            new_path = parts[1]
        elif status.startswith("R") and len(parts) >= 3:
            new_path = parts[2]
        else:
            continue

        if new_path.endswith(".jsonl"):
            out.append(new_path)
    return sorted(set(out))


def _build_suggestion(
    *,
    dataset_path: str,
    git_ref: str,
    git_url: str,
) -> DatasetVersionSuggestion:
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise RegistryError(f"Dataset path not found: {dataset_path}")

    _validate_jsonl(path_obj)
    name, version = _parse_dataset_path(dataset_path)

    return DatasetVersionSuggestion(
        name=name,
        version=version,
        git_url=git_url,
        git_ref=git_ref,
        path=Path(dataset_path).as_posix().lstrip("./"),
        checksum_sha256=_sha256(path_obj),
    )


def _render_report(
    *,
    base: str | None,
    head: str | None,
    changed_datasets: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    lines: list[str] = []
    lines.append("## Dataset Registry Bot")
    if base and head:
        lines.append(f"Diff range: `{base}..{head}`")
    lines.append(f"Updated dataset groups: **{len(changed_datasets)}**")
    lines.append("")

    if changed_datasets:
        lines.append("Suggested dataset blocks (upsert by `name` in `registry.json`):")
        lines.append("```json")
        lines.append(json.dumps(changed_datasets, indent=2, ensure_ascii=True))
        lines.append("```")
        lines.append("")

    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    if not changed_datasets and not warnings:
        lines.append("No new dataset `.jsonl` files were detected in this diff.")

    return "\n".join(lines).rstrip() + "\n"


def _compute_changes(
    *,
    registry_path: Path,
    suggestions: list[DatasetVersionSuggestion],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = _load_registry(registry_path)
    changed_names: set[str] = set()

    for suggestion in suggestions:
        metadata = _load_metadata(suggestion.name)
        _upsert_dataset_version(registry, suggestion=suggestion, metadata=metadata)
        changed_names.add(suggestion.name)

    changed = [item for item in registry if item.get("name") in changed_names]
    changed = sorted(changed, key=lambda d: str(d.get("name", "")))
    return registry, changed


def cmd_register(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)

    try:
        git_url = _normalize_git_url(args.git_url or _git_output(["config", "--get", "remote.origin.url"]))
        git_ref = args.git_ref or _git_output(["rev-parse", "HEAD"])
        suggestion = _build_suggestion(
            dataset_path=args.dataset_path,
            git_ref=git_ref,
            git_url=git_url,
        )
        updated_registry, changed = _compute_changes(
            registry_path=registry_path,
            suggestions=[suggestion],
        )
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _save_registry(registry_path, updated_registry)

    print(f"updated {registry_path} with {suggestion.name}@{suggestion.version}")
    print("updated dataset block:")
    print(json.dumps(changed[0], indent=2, ensure_ascii=True))
    return 0


def cmd_from_diff(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    warnings: list[str] = []

    try:
        git_url = _normalize_git_url(args.git_url or _git_output(["config", "--get", "remote.origin.url"]))
        dataset_files = _added_jsonl_files(args.base, args.head)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    suggestions: list[DatasetVersionSuggestion] = []
    for dataset_file in dataset_files:
        if not dataset_file.startswith("datasets/"):
            warnings.append(
                f"Skipping '{dataset_file}': files must be under datasets/ for registry automation"
            )
            continue

        try:
            suggestions.append(
                _build_suggestion(
                    dataset_path=dataset_file,
                    git_ref=args.git_ref or args.head,
                    git_url=git_url,
                )
            )
        except RegistryError as exc:
            warnings.append(f"Skipping '{dataset_file}': {exc}")

    changed: list[dict[str, Any]] = []
    if suggestions:
        try:
            updated_registry, changed = _compute_changes(
                registry_path=registry_path,
                suggestions=suggestions,
            )
        except RegistryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        if args.apply:
            _save_registry(registry_path, updated_registry)

    report = _render_report(base=args.base, head=args.head, changed_datasets=changed, warnings=warnings)

    if args.write_report:
        Path(args.write_report).write_text(report)

    if args.write_json:
        Path(args.write_json).write_text(json.dumps(changed, indent=2, ensure_ascii=True) + "\n")

    print(report)

    if args.strict and warnings:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage ai-prophet dataset registry entries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_register = sub.add_parser("register", help="Register one dataset version into registry.json")
    p_register.add_argument("--dataset-path", required=True, help="Dataset JSONL path")
    p_register.add_argument("--registry", default="registry.json", help="Path to registry.json")
    p_register.add_argument("--git-url", default=None, help="Git URL override")
    p_register.add_argument("--git-ref", default=None, help="Git ref override")
    p_register.set_defaults(func=cmd_register)

    p_diff = sub.add_parser(
        "from-diff", help="Detect added dataset JSONLs from git diff and suggest/apply grouped registry updates"
    )
    p_diff.add_argument("--base", required=True, help="Base SHA")
    p_diff.add_argument("--head", required=True, help="Head SHA")
    p_diff.add_argument("--registry", default="registry.json", help="Path to registry.json")
    p_diff.add_argument("--git-url", default=None, help="Git URL override")
    p_diff.add_argument("--git-ref", default=None, help="Git ref override (default: head SHA)")
    p_diff.add_argument("--write-report", default=None, help="Write markdown report to file")
    p_diff.add_argument("--write-json", default=None, help="Write JSON suggestions to file")
    p_diff.add_argument("--apply", action="store_true", help="Apply suggestions directly to registry.json")
    p_diff.add_argument("--strict", action="store_true", help="Exit non-zero if warnings are present")
    p_diff.set_defaults(func=cmd_from_diff)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
