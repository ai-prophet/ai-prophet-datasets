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
class DatasetSuggestion:
    path: str
    name: str
    version: str
    description: str
    git_url: str
    git_ref: str
    checksum_sha256: str

    def to_registry_entry(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
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
            if len(cleaned) < 2:
                errors.append(f"line {line_no}: 'outcomes' needs at least 2 non-empty strings")

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


def _load_registry(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists() or not path.read_text().strip():
        return [], True

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise RegistryError("registry.json dict payload must contain list key 'datasets'")
        return datasets, True
    if isinstance(payload, list):
        return payload, False
    raise RegistryError("registry.json must be either a list or an object with a 'datasets' list")


def _sort_registry(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(entry: dict[str, Any]) -> tuple[str, int, str]:
        name = str(entry.get("name", ""))
        version = str(entry.get("version", ""))
        is_latest = 1 if version == "latest" else 0
        return name, is_latest, version

    return sorted(entries, key=_sort_key)


def _save_registry(path: Path, entries: list[dict[str, Any]], use_wrapper: bool) -> None:
    sorted_entries = _sort_registry(entries)
    payload: Any = {"datasets": sorted_entries} if use_wrapper else sorted_entries
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _upsert(entries: list[dict[str, Any]], new_entry: dict[str, Any]) -> None:
    name = new_entry["name"]
    version = new_entry["version"]

    for i, existing in enumerate(entries):
        if existing.get("name") == name and existing.get("version") == version:
            entries[i] = new_entry
            return
    entries.append(new_entry)


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
    description: str | None,
    name: str | None,
    version: str | None,
) -> DatasetSuggestion:
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise RegistryError(f"Dataset path not found: {dataset_path}")

    _validate_jsonl(path_obj)

    if name is None or version is None:
        inferred_name, inferred_version = _parse_dataset_path(dataset_path)
    else:
        inferred_name, inferred_version = name, version

    desc = description or f"{inferred_name} ({inferred_version})"

    return DatasetSuggestion(
        path=Path(dataset_path).as_posix().lstrip("./"),
        name=inferred_name,
        version=inferred_version,
        description=desc,
        git_url=git_url,
        git_ref=git_ref,
        checksum_sha256=_sha256(path_obj),
    )


def _render_report(
    *,
    base: str | None,
    head: str | None,
    suggestions: list[DatasetSuggestion],
    warnings: list[str],
) -> str:
    lines: list[str] = []
    lines.append("## Dataset Registry Bot")
    if base and head:
        lines.append(f"Diff range: `{base}..{head}`")
    lines.append(f"Detected dataset files: **{len(suggestions)}**")
    lines.append("")

    if suggestions:
        entries = [s.to_registry_entry() for s in suggestions]
        lines.append("Suggested registry entries (append/upsert these):")
        lines.append("```json")
        lines.append(json.dumps(entries, indent=2, ensure_ascii=True))
        lines.append("```")
        lines.append("")

    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    if not suggestions and not warnings:
        lines.append("No new dataset `.jsonl` files were detected in this diff.")

    return "\n".join(lines).rstrip() + "\n"


def cmd_register(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)

    try:
        git_url = _normalize_git_url(args.git_url or _git_output(["config", "--get", "remote.origin.url"]))
        git_ref = args.git_ref or _git_output(["rev-parse", "HEAD"])
        suggestion = _build_suggestion(
            dataset_path=args.dataset_path,
            git_ref=git_ref,
            git_url=git_url,
            description=args.description,
            name=args.name,
            version=args.version,
        )
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries, use_wrapper = _load_registry(registry_path)
    _upsert(entries, suggestion.to_registry_entry())

    if args.promote_latest:
        latest_entry = dict(suggestion.to_registry_entry())
        latest_entry["version"] = "latest"
        latest_entry["description"] = f"Latest {suggestion.name}"
        _upsert(entries, latest_entry)

    _save_registry(registry_path, entries, use_wrapper)

    print(f"updated {registry_path} with {suggestion.name}@{suggestion.version}")
    if args.promote_latest:
        print(f"updated {registry_path} with {suggestion.name}@latest")
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

    suggestions: list[DatasetSuggestion] = []
    for dataset_file in dataset_files:
        if not dataset_file.startswith("datasets/"):
            warnings.append(
                f"Skipping '{dataset_file}': files must be under datasets/ for registry automation"
            )
            continue

        try:
            suggestion = _build_suggestion(
                dataset_path=dataset_file,
                git_ref=args.git_ref or args.head,
                git_url=git_url,
                description=None,
                name=None,
                version=None,
            )
            suggestions.append(suggestion)
        except RegistryError as exc:
            warnings.append(f"Skipping '{dataset_file}': {exc}")

    if args.apply and suggestions:
        entries, use_wrapper = _load_registry(registry_path)
        for suggestion in suggestions:
            _upsert(entries, suggestion.to_registry_entry())
            if args.promote_latest:
                latest_entry = dict(suggestion.to_registry_entry())
                latest_entry["version"] = "latest"
                latest_entry["description"] = f"Latest {suggestion.name}"
                _upsert(entries, latest_entry)
        _save_registry(registry_path, entries, use_wrapper)

    report = _render_report(base=args.base, head=args.head, suggestions=suggestions, warnings=warnings)

    if args.write_report:
        Path(args.write_report).write_text(report)

    if args.write_json:
        Path(args.write_json).write_text(
            json.dumps([s.to_registry_entry() for s in suggestions], indent=2, ensure_ascii=True) + "\n"
        )

    print(report)

    if args.strict and warnings:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage ai-prophet dataset registry entries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_register = sub.add_parser("register", help="Register one dataset file into registry.json")
    p_register.add_argument("--dataset-path", required=True, help="Dataset JSONL path")
    p_register.add_argument("--registry", default="registry.json", help="Path to registry.json")
    p_register.add_argument("--name", default=None, help="Dataset name (default: infer from path)")
    p_register.add_argument("--version", default=None, help="Dataset version (default: infer from path)")
    p_register.add_argument("--description", default=None, help="Dataset description")
    p_register.add_argument("--git-url", default=None, help="Git URL override")
    p_register.add_argument("--git-ref", default=None, help="Git ref override")
    p_register.add_argument(
        "--promote-latest",
        action="store_true",
        help="Also upsert a <name>@latest entry that points to this dataset",
    )
    p_register.set_defaults(func=cmd_register)

    p_diff = sub.add_parser(
        "from-diff", help="Detect added dataset JSONLs from git diff and suggest/apply registry updates"
    )
    p_diff.add_argument("--base", required=True, help="Base SHA")
    p_diff.add_argument("--head", required=True, help="Head SHA")
    p_diff.add_argument("--registry", default="registry.json", help="Path to registry.json")
    p_diff.add_argument("--git-url", default=None, help="Git URL override")
    p_diff.add_argument("--git-ref", default=None, help="Git ref override (default: head SHA)")
    p_diff.add_argument("--write-report", default=None, help="Write markdown report to file")
    p_diff.add_argument("--write-json", default=None, help="Write JSON suggestions to file")
    p_diff.add_argument(
        "--promote-latest",
        action="store_true",
        help="Also upsert <name>@latest for each detected dataset",
    )
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
