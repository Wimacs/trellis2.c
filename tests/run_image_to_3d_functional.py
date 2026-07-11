#!/usr/bin/env python3
"""Opt-in end-to-end image-to-3D generation and GLB validation.

This runner is intentionally not registered with CTest: it needs real model
weights, a DINO checkpoint, an input image, a GPU backend, and vkmesh.  It
streams the generator's output directly to the terminal and propagates the
generator's exit status on failure.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from verify_glb import GlbValidationError, verify_glb


class FunctionalTestConfigurationError(ValueError):
    """Raised when required local functional-test assets are unavailable."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _existing_path(path: Path, description: str, *, directory: bool) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise FunctionalTestConfigurationError(
            f"{description} does not exist: {path}"
        ) from error
    correct_kind = resolved.is_dir() if directory else resolved.is_file()
    if not correct_kind:
        expected = "directory" if directory else "file"
        raise FunctionalTestConfigurationError(
            f"{description} must be a {expected}: {resolved}"
        )
    return resolved


def _resolve_program(value: Path | None, description: str) -> Path | None:
    if value is None:
        return None

    raw = os.fspath(value.expanduser())
    has_path_separator = os.sep in raw or (os.altsep is not None and os.altsep in raw)
    candidate: Path | None
    if has_path_separator or value.is_absolute():
        candidate = value.expanduser()
    else:
        located = shutil.which(raw)
        candidate = Path(located) if located is not None else value.expanduser()

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FunctionalTestConfigurationError(
            f"{description} was not found: {value}"
        ) from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FunctionalTestConfigurationError(
            f"{description} is not an executable file: {resolved}"
        )
    return resolved


def _resolve_vkmesh(requested: Path | None, generator: Path) -> Path:
    if requested is not None:
        resolved = _resolve_program(requested, "vkmesh executable")
        assert resolved is not None
        return resolved

    sibling = generator.with_name("vkmesh" + generator.suffix)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling.resolve()

    located = shutil.which("vkmesh")
    if located is not None:
        resolved = _resolve_program(Path(located), "vkmesh executable")
        assert resolved is not None
        return resolved

    raise FunctionalTestConfigurationError(
        "vkmesh is required: pass --vkmesh, place it next to --executable, "
        "or make it available on PATH"
    )


@contextmanager
def _runtime_model_root(model: Path, manifest_template: Path | None) -> Iterator[Path]:
    if manifest_template is None:
        yield model
        return

    ckpts = _existing_path(model / "ckpts", "model ckpts", directory=True)
    pipeline = _existing_path(
        model / "pipeline.json", "model pipeline.json", directory=False
    )
    with tempfile.TemporaryDirectory(prefix="trellis-model-package-") as temporary:
        package_root = Path(temporary)
        shutil.copy2(manifest_template, package_root / "model.json")
        (package_root / "ckpts").symlink_to(ckpts, target_is_directory=True)
        (package_root / "pipeline.json").symlink_to(pipeline)
        print(
            f"using temporary model package {package_root} "
            f"(manifest={manifest_template}, weights={model})",
            flush=True,
        )
        yield package_root


def _build_command(
    args: argparse.Namespace,
    generator: Path,
    vkmesh: Path,
    model_root: Path,
) -> list[str]:
    command = [
        os.fspath(generator),
        "--model",
        os.fspath(model_root),
        "--dino",
        os.fspath(args.dino),
        "--image",
        os.fspath(args.image),
        "--output",
        os.fspath(args.output),
        "--backend",
        args.backend,
        "--pipeline",
        "1024_cascade",
        "--no-model-cache",
        "--mesh-postprocess",
        "--mesh-remesh",
        "--mesh-postprocess-no-simplify",
        "--vkmesh",
        os.fspath(vkmesh),
        "--texture-size",
        str(args.texture_size),
    ]
    if args.workspace_mib is not None:
        command.extend(
            ["--vkmesh-gpu-workspace-budget-mib", str(args.workspace_mib)]
        )
    if args.naf is not None:
        command.extend(["--naf", os.fspath(args.naf)])
    return command


def _generator_failure_exit_code(returncode: int) -> int:
    if returncode >= 0:
        return returncode
    # subprocess uses negative values for signals.  Convert to the conventional
    # shell status while retaining the exact signal in the diagnostic.
    return min(255, 128 + (-returncode))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an explicit real image-to-3D generation with vkmesh remeshing, "
            "then strictly validate the self-contained GLB"
        )
    )
    parser.add_argument(
        "--executable",
        required=True,
        type=Path,
        help="trellis-image-to-gltf executable (path or command on PATH)",
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="model directory containing ckpts/"
    )
    parser.add_argument(
        "--dino", required=True, type=Path, help="DINOv3 model directory"
    )
    parser.add_argument("--image", required=True, type=Path, help="input image")
    parser.add_argument(
        "--output", required=True, type=Path, help="output .glb file"
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("cuda", "vulkan"),
        help="backend compiled into the selected executable",
    )
    parser.add_argument(
        "--naf", type=Path, help="optional Pixal3D NAF safetensors override"
    )
    parser.add_argument(
        "--vkmesh",
        type=Path,
        help="vkmesh executable; otherwise use the generator sibling or PATH",
    )
    parser.add_argument(
        "--manifest-template",
        type=Path,
        help=(
            "model.json to copy into a temporary package whose ckpts and "
            "pipeline.json link to --model"
        ),
    )
    parser.add_argument(
        "--texture-size",
        type=_positive_int,
        default=1024,
        help="output texture edge (default: 1024)",
    )
    parser.add_argument(
        "--workspace-mib",
        type=_nonnegative_int,
        help="vkmesh GPU workspace cap in MiB (default: vkmesh automatic)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generator = _resolve_program(args.executable, "image-to-3D executable")
        assert generator is not None
        args.model = _existing_path(args.model, "model", directory=True)
        args.dino = _existing_path(args.dino, "DINO model", directory=True)
        args.image = _existing_path(args.image, "input image", directory=False)
        if args.naf is not None:
            args.naf = _existing_path(args.naf, "NAF weights", directory=False)
        if args.manifest_template is not None:
            args.manifest_template = _existing_path(
                args.manifest_template, "manifest template", directory=False
            )
        vkmesh = _resolve_vkmesh(args.vkmesh, generator)

        output = args.output.expanduser().resolve(strict=False)
        if output.suffix.lower() != ".glb":
            raise FunctionalTestConfigurationError(
                f"--output must use the .glb extension: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.is_symlink():
            if output.is_dir():
                raise FunctionalTestConfigurationError(
                    f"--output names an existing directory: {output}"
                )
            output.unlink()
        args.output = output

        with _runtime_model_root(args.model, args.manifest_template) as model_root:
            command = _build_command(args, generator, vkmesh, model_root)
            print(f"+ {shlex.join(command)}", flush=True)
            try:
                completed = subprocess.run(command, check=False)
            except OSError as error:
                print(f"failed to execute generator: {error}", file=sys.stderr)
                return 126
            except KeyboardInterrupt:
                print("image-to-3D generation interrupted", file=sys.stderr)
                return 130

            if completed.returncode != 0:
                if completed.returncode < 0:
                    print(
                        f"generator terminated by signal {-completed.returncode}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"generator failed with exit code {completed.returncode}",
                        file=sys.stderr,
                    )
                return _generator_failure_exit_code(completed.returncode)

            try:
                summary = verify_glb(output)
            except GlbValidationError as error:
                print(f"generated GLB is invalid: {error}", file=sys.stderr)
                return 1

            print(
                f"functional image-to-3D test passed: {output} "
                f"({summary.byte_length} bytes, {summary.mesh_count} meshes, "
                f"{summary.primitive_count} primitives, "
                f"{summary.embedded_image_count} embedded images)",
                flush=True,
            )
            return 0
    except FunctionalTestConfigurationError as error:
        print(f"functional test configuration error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"functional test setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
