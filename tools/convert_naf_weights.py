#!/usr/bin/env python3
"""Safely convert the ValeoAI NAF release checkpoint to safetensors."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
from safetensors.torch import save_file


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _branch_manifest(prefix: str, kernel: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        f"{prefix}.0.weight": (128, 3, kernel, kernel),
        f"{prefix}.0.bias": (128,),
    }
    for block in (1, 2):
        block_prefix = f"{prefix}.{block}"
        shapes.update(
            {
                f"{block_prefix}.norm1.weight": (128,),
                f"{block_prefix}.norm1.bias": (128,),
                f"{block_prefix}.conv1.weight": (128, 128, kernel, kernel),
                f"{block_prefix}.conv1.bias": (128,),
                f"{block_prefix}.norm2.weight": (128,),
                f"{block_prefix}.norm2.bias": (128,),
                f"{block_prefix}.conv2.weight": (128, 128, kernel, kernel),
                f"{block_prefix}.conv2.bias": (128,),
            }
        )
    return shapes


EXPECTED_TENSOR_SHAPES = {
    **_branch_manifest("image_encoder.encoder", 1),
    **_branch_manifest("image_encoder.sem_encoder", 3),
    "image_encoder.rope.periods": (16,),
}
EXPECTED_TENSOR_COUNT = len(EXPECTED_TENSOR_SHAPES)


class ConversionError(ValueError):
    """Raised when an input is not the expected pure NAF state_dict."""


def _is_http_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


@contextmanager
def _local_checkpoint(source: str) -> Iterator[Path]:
    if not _is_http_url(source):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist or is not a file: {path}")
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="trellis2-naf-download-") as temp_dir:
        path = Path(temp_dir) / "naf_release.pth"
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "trellis2.c-naf-converter/1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output, length=DOWNLOAD_CHUNK_SIZE)
        yield path


def load_normalized_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load a pure state_dict without arbitrary pickle execution and normalize it."""
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ConversionError(f"failed to safely load {checkpoint_path}: {exc}") from exc

    if not isinstance(state_dict, Mapping):
        raise ConversionError(
            f"expected a top-level state_dict mapping, got {type(state_dict).__name__}"
        )
    for name in state_dict:
        if not isinstance(name, str):
            raise ConversionError(
                f"state_dict key must be str, got {type(name).__name__}"
            )

    actual_names = set(state_dict)
    expected_names = set(EXPECTED_TENSOR_SHAPES)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(repr(name) for name in missing))
        if extra:
            details.append("extra=" + ", ".join(repr(name) for name in extra))
        raise ConversionError("NAF tensor names mismatch: " + "; ".join(details))

    normalized: dict[str, torch.Tensor] = {}
    for name, expected_shape in EXPECTED_TENSOR_SHAPES.items():
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise ConversionError(
                f"state_dict value for {name!r} must be Tensor, got {type(value).__name__}"
            )
        if value.layout != torch.strided:
            raise ConversionError(
                f"state_dict tensor {name!r} must use strided layout, got {value.layout}"
            )
        if value.is_complex():
            raise ConversionError(f"state_dict tensor {name!r} must be real-valued")
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise ConversionError(
                f"state_dict tensor {name!r} has shape {actual_shape}, expected {expected_shape}"
            )

        try:
            tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        except Exception as exc:
            raise ConversionError(f"failed to normalize tensor {name!r}: {exc}") from exc
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ConversionError(f"failed to normalize tensor {name!r} to contiguous CPU F32")
        normalized[name] = tensor

    return normalized


def _save_atomic(
    tensors: dict[str, torch.Tensor],
    output_path: Path,
    source: str,
    force: bool,
) -> None:
    output_path = output_path.expanduser()
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (use --force): {output_path}")
    if output_path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            tensors,
            str(temporary_path),
            metadata={"format": "pt", "source": source},
        )
        if output_path.exists() and not force:
            raise FileExistsError(f"output already exists (use --force): {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_checkpoint(source: str, output_path: Path, force: bool = False) -> int:
    """Convert a local path or HTTP(S) checkpoint URL to safetensors."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (use --force): {output_path}")

    with _local_checkpoint(source) as checkpoint_path:
        tensors = load_normalized_state_dict(checkpoint_path)
    _save_atomic(tensors, output_path, source, force)
    return len(tensors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely convert ValeoAI NAF naf_release.pth from a local path or "
            "HTTP(S) URL to CPU F32 safetensors."
        )
    )
    parser.add_argument("input", help="local naf_release.pth path or HTTP(S) URL")
    parser.add_argument("output", type=Path, help="output .safetensors path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        count = convert_checkpoint(args.input, args.output, args.force)
    except (ConversionError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
