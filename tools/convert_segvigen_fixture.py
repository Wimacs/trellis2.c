#!/usr/bin/env python3
"""Convert trusted local SegviGen sparse-latent fixtures to TSLAT01 v1."""

from __future__ import annotations

import argparse
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    pass


TSLAT_MAGIC = b"TSLAT01\0"
TSLAT_VERSION = 1
TSLAT_HEADER_BYTES = 80
TSLAT_COORD_FRAME_TRELLIS_DECODER_V1 = 1
TSLAT_CHANNELS = 32
SEGVIGEN_NORMALIZED_EXTENT = 0.99999
SUPPORTED_RESOLUTIONS = (512, 1024)
HEADER_STRUCT = struct.Struct("<8sIIIIQII3f3fQQ")
assert HEADER_STRUCT.size == TSLAT_HEADER_BYTES


def load_fixture(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to read a SegviGen .pth fixture") from exc

    try:
        fixture = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        fixture = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConversionError(f"failed to load trusted fixture {path}: {exc}") from exc
    if not isinstance(fixture, Mapping):
        raise ConversionError("fixture root must be a mapping")
    raw_keys = list(fixture)
    if any(not isinstance(key, str) for key in raw_keys):
        raise ConversionError("fixture mapping keys must be strings")
    keys = set(raw_keys)
    if keys != {"coords", "feats"}:
        raise ConversionError(
            "fixture must contain exactly 'coords' and 'feats'; "
            f"missing={sorted({'coords', 'feats'} - keys)}, "
            f"extra={sorted(keys - {'coords', 'feats'})}"
        )
    return fixture


def normalize_and_validate_fixture(
    fixture: Mapping[str, Any], resolution: int
) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to validate a SegviGen fixture") from exc

    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ConversionError(
            f"resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution}"
        )
    raw_keys = list(fixture)
    if any(not isinstance(key, str) for key in raw_keys):
        raise ConversionError("fixture mapping keys must be strings")
    if set(raw_keys) != {"coords", "feats"}:
        raise ConversionError("fixture must contain exactly 'coords' and 'feats'")
    coords = fixture["coords"]
    feats = fixture["feats"]
    if not isinstance(coords, torch.Tensor) or not isinstance(feats, torch.Tensor):
        raise ConversionError("fixture coords and feats must both be tensors")
    if coords.layout != torch.strided or coords.is_complex():
        raise ConversionError("fixture coords must be a real strided tensor")
    if feats.layout != torch.strided or feats.is_complex():
        raise ConversionError("fixture feats must be a real strided tensor")
    if coords.ndim != 2 or int(coords.shape[1]) != 4:
        raise ConversionError(
            f"fixture coords must have shape [N, 4], got {tuple(coords.shape)}"
        )
    if feats.ndim != 2 or int(feats.shape[1]) != TSLAT_CHANNELS:
        raise ConversionError(
            f"fixture feats must have shape [N, {TSLAT_CHANNELS}], "
            f"got {tuple(feats.shape)}"
        )
    n_coords = int(coords.shape[0])
    if n_coords <= 0 or int(feats.shape[0]) != n_coords:
        raise ConversionError("fixture coords and feats must have the same non-zero N")
    if n_coords > (1 << 63) - 1:
        raise ConversionError("fixture contains too many sparse coordinates")

    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64}
    if coords.dtype not in integer_dtypes:
        raise ConversionError(f"fixture coords must be an integer tensor, got {coords.dtype}")
    floating_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if feats.dtype not in floating_dtypes:
        raise ConversionError(f"fixture feats must be floating point, got {feats.dtype}")

    coords_i64 = coords.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if not bool(torch.all(coords_i64[:, 0] == 0).item()):
        raise ConversionError("fixture coords batch column must be zero")
    xyz = coords_i64[:, 1:]
    if not bool(torch.all((xyz >= 0) & (xyz < resolution)).item()):
        raise ConversionError(
            f"fixture xyz coordinates must be in [0, {resolution - 1}]"
        )
    if int(torch.unique(coords_i64, dim=0).shape[0]) != n_coords:
        raise ConversionError("fixture contains duplicate sparse coordinates")

    feats_f32 = feats.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(feats_f32).all().item()):
        raise ConversionError("fixture feats contain NaN or infinity")
    coords_i32 = coords_i64.to(dtype=torch.int32).contiguous()
    return coords_i32, feats_f32


def decoder_aabb_limit(resolution: int) -> float:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ConversionError(f"unsupported resolution: {resolution}")
    # Match trellis_shape_latent_decoder_aabb_limit(): FlexDualGrid vertices
    # may extend half a voxel beyond the nominal decoder cube.
    return 0.5 + 0.5 / resolution + 8.0 * (2.0**-23)


def validate_anchor_aabb(
    anchor_min: Sequence[float],
    anchor_max: Sequence[float],
    resolution: int = 512,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConversionError("NumPy is required to validate the anchor AABB") from exc

    minimum = np.asarray(anchor_min, dtype=np.float64)
    maximum = np.asarray(anchor_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ConversionError("anchor minimum and maximum must each contain three values")
    if not bool(np.isfinite(minimum).all() and np.isfinite(maximum).all()):
        raise ConversionError("anchor AABB must contain only finite values")
    if bool(np.any(maximum < minimum)):
        raise ConversionError("anchor AABB maximum must not be below its minimum")
    if not math.isfinite(float(np.max(maximum - minimum))) or float(
        np.max(maximum - minimum)
    ) <= 0.0:
        raise ConversionError("anchor AABB must have a non-zero extent")

    minimum_f32 = minimum.astype(np.float32)
    maximum_f32 = maximum.astype(np.float32)
    limit = decoder_aabb_limit(resolution)
    if bool(np.any(minimum_f32 < -limit) or np.any(maximum_f32 > limit)):
        raise ConversionError(
            "anchor AABB must be in the TRELLIS decoder frame "
            f"[-{limit}, {limit}] at resolution {resolution}"
        )
    return (
        tuple(float(value) for value in minimum_f32),
        tuple(float(value) for value in maximum_f32),
    )


def compute_normalized_anchor_aabb(
    glb_path: Path,
    resolution: int = 512,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return official SegviGen decoder-frame bounds from direct GLB world axes."""

    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise ConversionError(
            "trimesh and NumPy are required to compute an anchor from a GLB"
        ) from exc

    try:
        asset = trimesh.load(str(glb_path), force="scene")
        bounds = asset.bounds
    except Exception as exc:
        raise ConversionError(f"failed to load GLB anchor {glb_path}: {exc}") from exc
    if bounds is None:
        raise ConversionError(f"GLB anchor has no world-space geometry: {glb_path}")
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not bool(np.isfinite(bounds).all()):
        raise ConversionError("GLB world-space AABB is missing or non-finite")
    extent = bounds[1] - bounds[0]
    if bool(np.any(extent < 0.0)):
        raise ConversionError("GLB world-space AABB is invalid")
    max_extent = float(np.max(extent))
    if not math.isfinite(max_extent) or max_extent <= 0.0:
        raise ConversionError("GLB world-space AABB must have a non-zero extent")

    center = (bounds[0] + bounds[1]) * 0.5
    scale = SEGVIGEN_NORMALIZED_EXTENT / max_extent
    # SegviGen uses the glTF scene's direct world axes here. No TRELLIS output
    # axis transform is applied before centering and uniform scaling.
    normalized_min = (bounds[0] - center) * scale
    normalized_max = (bounds[1] - center) * scale
    return validate_anchor_aabb(normalized_min, normalized_max, resolution)


def _tensor_payload_bytes(coords: Any, feats: Any) -> tuple[bytes, bytes]:
    try:
        coords_array = coords.numpy().astype("<i4", copy=False)
        feats_array = feats.numpy().astype("<f4", copy=False)
    except Exception as exc:
        raise ConversionError(f"failed to materialize fixture payload: {exc}") from exc
    return coords_array.tobytes(order="C"), feats_array.tobytes(order="C")


def save_atomic(
    output_path: Path,
    coords: Any,
    feats: Any,
    resolution: int,
    anchor_min: Sequence[float],
    anchor_max: Sequence[float],
    force: bool,
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (use --force): {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise ConversionError(f"output path is not a regular file: {output_path}")
    minimum, maximum = validate_anchor_aabb(anchor_min, anchor_max, resolution)
    n_coords = int(coords.shape[0])
    coords_payload, feats_payload = _tensor_payload_bytes(coords, feats)
    expected_coords_bytes = n_coords * 4 * 4
    expected_feats_bytes = n_coords * TSLAT_CHANNELS * 4
    if len(coords_payload) != expected_coords_bytes or len(feats_payload) != expected_feats_bytes:
        raise ConversionError("fixture payload byte count does not match its tensor shapes")
    if TSLAT_HEADER_BYTES + expected_coords_bytes + expected_feats_bytes > (1 << 64) - 1:
        raise ConversionError("fixture payload is too large for TSLAT01")

    header = HEADER_STRUCT.pack(
        TSLAT_MAGIC,
        TSLAT_VERSION,
        TSLAT_HEADER_BYTES,
        resolution,
        TSLAT_CHANNELS,
        n_coords,
        TSLAT_COORD_FRAME_TRELLIS_DECODER_V1,
        0,
        *minimum,
        *maximum,
        expected_coords_bytes,
        expected_feats_bytes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            output.write(coords_payload)
            output.write(feats_payload)
            output.flush()
            os.fsync(output.fileno())
        expected_size = TSLAT_HEADER_BYTES + expected_coords_bytes + expected_feats_bytes
        if temporary_path.stat().st_size != expected_size:
            raise ConversionError("temporary TSLAT01 file has an unexpected size")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_fixture(
    input_path: Path,
    output_path: Path,
    *,
    resolution: int = 512,
    glb_path: Path | None = None,
    anchor_min: Sequence[float] | None = None,
    anchor_max: Sequence[float] | None = None,
    force: bool = False,
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path == output_path:
        raise ConversionError("input and output paths must be different")
    if glb_path is not None:
        glb_path = glb_path.expanduser().resolve()
        if output_path == glb_path:
            raise ConversionError("GLB anchor and output paths must be different")
        if anchor_min is not None or anchor_max is not None:
            raise ConversionError("use either --glb or an explicit anchor AABB, not both")
        minimum, maximum = compute_normalized_anchor_aabb(glb_path, resolution)
    else:
        if anchor_min is None or anchor_max is None:
            raise ConversionError(
                "an anchor is required: pass --glb or both --anchor-min and --anchor-max"
            )
        minimum, maximum = validate_anchor_aabb(anchor_min, anchor_max, resolution)

    fixture = load_fixture(input_path)
    coords, feats = normalize_and_validate_fixture(fixture, resolution)
    save_atomic(output_path, coords, feats, resolution, minimum, maximum, force)
    return int(coords.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a trusted local SegviGen dict(coords, feats) .pth fixture "
            "to native TSLAT01 v1."
        )
    )
    parser.add_argument("input", type=Path, help="trusted local sparse-latent .pth")
    parser.add_argument("output", type=Path, help="output .tslat path")
    parser.add_argument(
        "--resolution",
        type=int,
        choices=SUPPORTED_RESOLUTIONS,
        default=512,
        help="decoder resolution recorded in the cache (default: 512)",
    )
    parser.add_argument(
        "--glb",
        type=Path,
        help=(
            "source GLB used to compute the official direct-world-axis, centered "
            "and uniformly normalized anchor AABB"
        ),
    )
    parser.add_argument(
        "--anchor-min",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="explicit decoder-frame anchor AABB minimum",
    )
    parser.add_argument(
        "--anchor-max",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="explicit decoder-frame anchor AABB maximum",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        count = convert_fixture(
            args.input,
            args.output,
            resolution=args.resolution,
            glb_path=args.glb,
            anchor_min=args.anchor_min,
            anchor_max=args.anchor_max,
            force=args.force,
        )
    except (ConversionError, FileExistsError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} sparse tokens to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
