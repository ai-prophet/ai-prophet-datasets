"""Command-line interface for the ai-prophet datasets SDK.

Subcommands mirror the Python API:

  list        — list datasets and releases (read).
  fetch       — download a release's tasks.jsonl to disk (read).
  validate    — lint a release dir or the whole tree (local).
  resolve     — attach a resolved_outcome to a task (local write).

Reads default to the upstream public repo over raw HTTPS and need no
auth. Writes require a local clone (`--repo-path`) and rely on whatever
`git push` is already configured to do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import DEFAULT_BRANCH, DEFAULT_REPO_URL, Registry
from .validation import SchemaError, validate_release, validate_tree


def _build_registry(args: argparse.Namespace) -> Registry:
    """Construct a Registry from CLI args."""
    return Registry(
        repo_path=args.repo_path,
        repo_url=args.repo_url or DEFAULT_REPO_URL,
        branch=args.branch or DEFAULT_BRANCH,
    )


def cmd_list(args: argparse.Namespace) -> int:
    """Print every dataset and its releases."""
    with _build_registry(args) as reg:
        for dataset in reg.list_datasets():
            print(f"{dataset.name}  —  {dataset.description}")
            for r in dataset.releases:
                print(
                    f"  {r.id}  ({r.release_date}, {r.status}, "
                    f"{r.resolved_count}/{r.task_count} resolved)"
                )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download a release's tasks.jsonl to a local file."""
    with _build_registry(args) as reg:
        release = reg.get_release(args.dataset, args.release_id)
        tasks = release.tasks()
    out = Path(args.output) if args.output else Path(f"{args.dataset}-{args.release_id}.jsonl")
    out.write_text("\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in tasks) + "\n")
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a release directory or the whole datasets/ tree."""
    try:
        if args.release:
            validate_release(Path(args.release))
            print(f"ok: {args.release}")
        else:
            root = Path(args.repo_path or ".") / "datasets"
            validate_tree(root)
            print(f"ok: {root}")
    except SchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_rebuild_registry(args: argparse.Namespace) -> int:
    """Walk the tree under `--repo-path` and rewrite `registry.json`."""
    if not args.repo_path:
        print("error: --repo-path is required for rebuild-registry", file=sys.stderr)
        return 2
    root = Path(args.repo_path)
    try:
        payload = validate_tree(root / "datasets")
    except SchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = root / "registry.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {out} ({len(payload['datasets'])} datasets)")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    """Attach a resolved_outcome to one task; commits locally."""
    if not args.repo_path:
        print("error: --repo-path is required for write operations", file=sys.stderr)
        return 2
    reg = Registry(repo_path=args.repo_path, branch=args.branch or DEFAULT_BRANCH)
    release = reg.get_release(args.dataset, args.release_id)
    sha = release.set_resolved_outcome(
        task_id=args.task_id,
        value=args.value,
        resolved_at=args.resolved_at,
        source=args.source,
    )
    print(f"committed {sha}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Wire up the CLI."""
    parser = argparse.ArgumentParser(prog="ai-prophet-datasets")
    parser.add_argument("--repo-path", default=None, help="Path to a local clone (enables writes)")
    parser.add_argument("--repo-url", default=None, help=f"Repo URL (default: {DEFAULT_REPO_URL})")
    parser.add_argument("--branch", default=None, help=f"Branch name (default: {DEFAULT_BRANCH})")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List datasets and releases")
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="Download a release's tasks.jsonl")
    p_fetch.add_argument("dataset")
    p_fetch.add_argument("release_id")
    p_fetch.add_argument("-o", "--output", default=None, help="Output file path")
    p_fetch.set_defaults(func=cmd_fetch)

    p_validate = sub.add_parser("validate", help="Validate a release dir or the whole tree")
    p_validate.add_argument("--release", default=None, help="Path to a release directory")
    p_validate.set_defaults(func=cmd_validate)

    p_rebuild = sub.add_parser("rebuild-registry", help="Rewrite registry.json from the tree")
    p_rebuild.set_defaults(func=cmd_rebuild_registry)

    p_resolve = sub.add_parser("resolve", help="Set a task's resolved_outcome and commit")
    p_resolve.add_argument("dataset")
    p_resolve.add_argument("release_id")
    p_resolve.add_argument("--task-id", required=True)
    p_resolve.add_argument(
        "--value",
        required=True,
        nargs="+",
        help="One or more outcome strings (the resolved values)",
    )
    p_resolve.add_argument("--resolved-at", default=None)
    p_resolve.add_argument("--source", default=None)
    p_resolve.set_defaults(func=cmd_resolve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
