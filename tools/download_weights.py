#!/usr/bin/env python3
"""Download TRELLIS.2, DINOv3, and background-removal weights."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "TRELLIS.2"

DEFAULT_TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
DEFAULT_DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIREFNET_REPO = "Acly/BiRefNet-GGUF"

TRELLIS_MINIMAL_PATTERNS = [
    "README*",
    "LICENSE*",
    "*.json",
    "*.txt",
    "*.safetensors",
    "ckpts/*.safetensors",
]

DINO_MINIMAL_PATTERNS = [
    "README*",
    "LICENSE*",
    "*.json",
    "*.txt",
    "model.safetensors",
]

BIREFNET_MINIMAL_PATTERNS = [
    "README*",
    "LICENSE*",
    "BiRefNet-F16.gguf",
]


@dataclass(frozen=True)
class RepoSpec:
    label: str
    repo_id: str
    local_name: str
    minimal_patterns: list[str]


def split_patterns(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    patterns: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                patterns.append(item)

    return patterns or None


def normalize_source(source: str) -> str:
    aliases = {
        "hf": "huggingface",
        "huggingface": "huggingface",
        "hugging-face": "huggingface",
        "ms": "modelscope",
        "modelscope": "modelscope",
        "model-scope": "modelscope",
    }

    normalized = aliases.get(source.lower())
    if normalized is None:
        choices = ", ".join(sorted(aliases))
        raise argparse.ArgumentTypeError(f"unknown source {source!r}; choose one of: {choices}")
    return normalized


def import_or_exit(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python package: {module_name}\n"
            f"Install it first:\n"
            f"  {install_hint}"
        ) from exc


def call_supported(fn: Callable[..., str], kwargs: dict[str, Any]) -> str:
    signature = inspect.signature(fn)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return fn(**supported)


def download_from_huggingface(
    spec: RepoSpec,
    local_dir: Path,
    revision: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    token: str | None,
    endpoint: str | None,
    local_files_only: bool,
    max_workers: int,
) -> str:
    hub = import_or_exit("huggingface_hub", 'python3 -m pip install -U "huggingface_hub[cli]"')

    return hub.snapshot_download(
        repo_id=spec.repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=token,
        endpoint=endpoint,
        local_files_only=local_files_only,
        max_workers=max_workers,
    )


def download_from_modelscope(
    spec: RepoSpec,
    local_dir: Path,
    revision: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    token: str | None,
    endpoint: str | None,
    local_files_only: bool,
    max_workers: int,
) -> str:
    modelscope = import_or_exit("modelscope", "python3 -m pip install -U modelscope")

    kwargs: dict[str, Any] = {
        "model_id": spec.repo_id,
        "repo_id": spec.repo_id,
        "revision": revision,
        "local_dir": str(local_dir),
        "allow_patterns": allow_patterns,
        "allow_file_pattern": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "ignore_file_pattern": ignore_patterns,
        "token": token,
        "endpoint": endpoint,
        "local_files_only": local_files_only,
        "max_workers": max_workers,
    }
    return call_supported(modelscope.snapshot_download, kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TRELLIS.2, DINOv3, and background-removal weights from Hugging Face or ModelScope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=normalize_source,
        default="huggingface",
        help="download source: huggingface/hf or modelscope/ms",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory that will contain TRELLIS.2-4B/, dinov3-vitl16-pretrain-lvd1689m/, and BiRefNet/",
    )
    parser.add_argument(
        "--only",
        choices=("all", "trellis", "dino", "birefnet", "background"),
        default="all",
        help="download all weights or just one group",
    )
    parser.add_argument(
        "--trellis-repo",
        default=DEFAULT_TRELLIS_REPO,
        help="TRELLIS.2 repo id on the selected source",
    )
    parser.add_argument(
        "--dino-repo",
        default=DEFAULT_DINO_REPO,
        help="DINOv3 repo id on the selected source",
    )
    parser.add_argument(
        "--birefnet-repo",
        default=DEFAULT_BIREFNET_REPO,
        help="BiRefNet background-removal repo id on the selected source",
    )
    parser.add_argument(
        "--birefnet-source",
        type=normalize_source,
        default="huggingface",
        help="download source for BiRefNet; defaults to Hugging Face even when --source is ModelScope",
    )
    parser.add_argument("--revision", help="branch, tag, or commit to download")
    parser.add_argument(
        "--full",
        action="store_true",
        help="download the full repository instead of the minimal files needed by trellis2.c",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="comma-separated glob(s) to download; can be repeated and overrides the default minimal set",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        help="comma-separated glob(s) to skip; can be repeated",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("MODELSCOPE_TOKEN"),
        help="access token; defaults to HF_TOKEN or MODELSCOPE_TOKEN when set",
    )
    parser.add_argument(
        "--endpoint",
        help="custom hub endpoint, for example https://hf-mirror.com for Hugging Face",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="use already cached files only",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="parallel download workers when supported by the selected SDK",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned downloads without downloading files",
    )
    return parser


def selected_specs(args: argparse.Namespace) -> list[RepoSpec]:
    specs = [
        RepoSpec(
            label="TRELLIS.2",
            repo_id=args.trellis_repo,
            local_name="TRELLIS.2-4B",
            minimal_patterns=TRELLIS_MINIMAL_PATTERNS,
        ),
        RepoSpec(
            label="DINOv3",
            repo_id=args.dino_repo,
            local_name="dinov3-vitl16-pretrain-lvd1689m",
            minimal_patterns=DINO_MINIMAL_PATTERNS,
        ),
        RepoSpec(
            label="BiRefNet",
            repo_id=args.birefnet_repo,
            local_name="BiRefNet",
            minimal_patterns=BIREFNET_MINIMAL_PATTERNS,
        ),
    ]

    if args.only == "trellis":
        return [specs[0]]
    if args.only == "dino":
        return [specs[1]]
    if args.only in ("birefnet", "background"):
        return [specs[2]]
    return specs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    include_patterns = split_patterns(args.include)
    ignore_patterns = split_patterns(args.ignore)
    specs = selected_specs(args)

    for spec in specs:
        local_dir = output_dir / spec.local_name
        allow_patterns = include_patterns
        if allow_patterns is None and not args.full:
            allow_patterns = spec.minimal_patterns
        source = args.birefnet_source if spec.local_name == "BiRefNet" else args.source
        downloader = download_from_huggingface if source == "huggingface" else download_from_modelscope

        print(f"[{source}] {spec.label}: {spec.repo_id}")
        print(f"  -> {local_dir}")
        if allow_patterns:
            print(f"  include: {', '.join(allow_patterns)}")
        if ignore_patterns:
            print(f"  ignore: {', '.join(ignore_patterns)}")

        if args.dry_run:
            continue

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = downloader(
            spec=spec,
            local_dir=local_dir,
            revision=args.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=args.token,
            endpoint=args.endpoint,
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
        )
        print(f"  done: {downloaded_path}")

    print("Dry run complete." if args.dry_run else "Weights are ready.")
    downloaded_names = {spec.local_name for spec in specs}
    if "TRELLIS.2-4B" in downloaded_names:
        print(f"Run with --model {output_dir / 'TRELLIS.2-4B'}")
    if "dinov3-vitl16-pretrain-lvd1689m" in downloaded_names:
        print(f"Run with --dino  {output_dir / 'dinov3-vitl16-pretrain-lvd1689m'}")
    if "BiRefNet" in downloaded_names:
        print(f"Run with --birefnet {output_dir / 'BiRefNet' / 'BiRefNet-F16.gguf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
