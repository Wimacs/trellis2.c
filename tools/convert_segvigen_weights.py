#!/usr/bin/env python3
"""Convert a released SegviGen Full Lightning checkpoint to safetensors."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConversionError(RuntimeError):
    pass


SOURCE_PREFIX = "gen3dseg.flow_model."
BLOCK_COUNT = 30
MODEL_CHANNELS = 1536
CONDITION_CHANNELS = 1024
HEADS = 12
HEAD_DIM = 128
MLP_CHANNELS = 8192
MODULATION_CHANNELS = 9216

BF16_BLOCK_TENSOR_SUFFIXES = frozenset(
    {
        "self_attn.to_qkv.weight",
        "self_attn.to_qkv.bias",
        "self_attn.to_out.weight",
        "self_attn.to_out.bias",
        "cross_attn.to_q.weight",
        "cross_attn.to_q.bias",
        "cross_attn.to_kv.weight",
        "cross_attn.to_kv.bias",
        "cross_attn.to_out.weight",
        "cross_attn.to_out.bias",
        "mlp.mlp.0.weight",
        "mlp.mlp.0.bias",
        "mlp.mlp.2.weight",
        "mlp.mlp.2.bias",
    }
)


def _expected_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "t_embedder.mlp.0.weight": (MODEL_CHANNELS, 256),
        "t_embedder.mlp.0.bias": (MODEL_CHANNELS,),
        "t_embedder.mlp.2.weight": (MODEL_CHANNELS, MODEL_CHANNELS),
        "t_embedder.mlp.2.bias": (MODEL_CHANNELS,),
        "adaLN_modulation.1.weight": (MODULATION_CHANNELS, MODEL_CHANNELS),
        "adaLN_modulation.1.bias": (MODULATION_CHANNELS,),
        "input_layer.weight": (MODEL_CHANNELS, 64),
        "input_layer.bias": (MODEL_CHANNELS,),
        "out_layer.weight": (32, MODEL_CHANNELS),
        "out_layer.bias": (32,),
    }
    for block in range(BLOCK_COUNT):
        prefix = f"blocks.{block}."
        shapes.update(
            {
                prefix + "modulation": (MODULATION_CHANNELS,),
                prefix + "norm2.weight": (MODEL_CHANNELS,),
                prefix + "norm2.bias": (MODEL_CHANNELS,),
                prefix + "self_attn.to_qkv.weight": (3 * MODEL_CHANNELS, MODEL_CHANNELS),
                prefix + "self_attn.to_qkv.bias": (3 * MODEL_CHANNELS,),
                prefix + "self_attn.q_rms_norm.gamma": (HEADS, HEAD_DIM),
                prefix + "self_attn.k_rms_norm.gamma": (HEADS, HEAD_DIM),
                prefix + "self_attn.to_out.weight": (MODEL_CHANNELS, MODEL_CHANNELS),
                prefix + "self_attn.to_out.bias": (MODEL_CHANNELS,),
                prefix + "cross_attn.to_q.weight": (MODEL_CHANNELS, MODEL_CHANNELS),
                prefix + "cross_attn.to_q.bias": (MODEL_CHANNELS,),
                prefix + "cross_attn.to_kv.weight": (2 * MODEL_CHANNELS, CONDITION_CHANNELS),
                prefix + "cross_attn.to_kv.bias": (2 * MODEL_CHANNELS,),
                prefix + "cross_attn.q_rms_norm.gamma": (HEADS, HEAD_DIM),
                prefix + "cross_attn.k_rms_norm.gamma": (HEADS, HEAD_DIM),
                prefix + "cross_attn.to_out.weight": (MODEL_CHANNELS, MODEL_CHANNELS),
                prefix + "cross_attn.to_out.bias": (MODEL_CHANNELS,),
                prefix + "mlp.mlp.0.weight": (MLP_CHANNELS, MODEL_CHANNELS),
                prefix + "mlp.mlp.0.bias": (MLP_CHANNELS,),
                prefix + "mlp.mlp.2.weight": (MODEL_CHANNELS, MLP_CHANNELS),
                prefix + "mlp.mlp.2.bias": (MODEL_CHANNELS,),
            }
        )
    return shapes


EXPECTED_TENSOR_SHAPES = _expected_tensor_shapes()
EXPECTED_TENSOR_COUNT = 640
assert len(EXPECTED_TENSOR_SHAPES) == EXPECTED_TENSOR_COUNT


def _expected_tensor_dtype_name(name: str) -> str:
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blocks":
        return "bf16" if parts[2] in BF16_BLOCK_TENSOR_SUFFIXES else "f32"
    return "f32"


EXPECTED_TENSOR_DTYPES = {
    name: _expected_tensor_dtype_name(name) for name in EXPECTED_TENSOR_SHAPES
}
assert sum(dtype == "bf16" for dtype in EXPECTED_TENSOR_DTYPES.values()) == 420
assert sum(dtype == "f32" for dtype in EXPECTED_TENSOR_DTYPES.values()) == 220


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to read the SegviGen checkpoint") from exc

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConversionError(f"failed to safely load {path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ConversionError("checkpoint root must be a mapping")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ConversionError("checkpoint must contain a state_dict mapping")
    return state


def normalize_and_validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, tensor in state.items():
        if not isinstance(raw_name, str):
            raise ConversionError("state_dict contains a non-string tensor name")
        if not raw_name.startswith(SOURCE_PREFIX):
            raise ConversionError(f"unexpected state_dict tensor {raw_name!r}")
        name = raw_name[len(SOURCE_PREFIX) :]
        if name in normalized:
            raise ConversionError(f"duplicate normalized tensor name: {name}")
        normalized[name] = tensor

    actual = set(normalized)
    expected = set(EXPECTED_TENSOR_SHAPES)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ConversionError(
            f"SegviGen flow tensor contract mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )

    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to validate tensors") from exc
    for name, expected_shape in EXPECTED_TENSOR_SHAPES.items():
        tensor = normalized[name]
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError(f"state_dict value {name!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.is_complex():
            raise ConversionError(f"unsupported tensor type for {name!r}")
        shape = tuple(int(value) for value in tensor.shape)
        if shape != expected_shape:
            raise ConversionError(
                f"tensor {name!r} has shape {shape}, expected {expected_shape}"
            )
        expected_dtype = (
            torch.bfloat16
            if EXPECTED_TENSOR_DTYPES[name] == "bf16"
            else torch.float32
        )
        if tensor.dtype != expected_dtype:
            raise ConversionError(
                f"tensor {name!r} has dtype {tensor.dtype}, expected {expected_dtype}"
            )
    return normalized


def build_inference_state(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError(f"state_dict value {name!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.is_complex():
            raise ConversionError(f"unsupported tensor type for {name!r}")
        if tensor.dtype not in (torch.float32, torch.bfloat16):
            raise ConversionError(
                f"tensor {name!r} has unsupported dtype {tensor.dtype}; expected F32 or BF16"
            )
        # SLatFlowModel converts Linear layers inside transformer blocks to
        # BF16, while global projections, LayerNorm/RMSNorm parameters, and
        # shared modulation remain F32. The Lightning state_dict captures
        # that deliberate mixed-precision contract and must be preserved.
        output[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return output


def save_atomic(
    tensors: Mapping[str, Any],
    output_path: Path,
    source: Path,
    force: bool,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ConversionError("safetensors is required to write converted weights") from exc

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (use --force): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    metadata = {
        "format": "segvigen-full-flow-v1",
        "source": source.name,
        "weight_dtype": "mixed-f32-bf16",
        "mode": "full",
        "architecture": "trellis_dit_flow",
    }
    try:
        save_file(dict(tensors), str(temporary_path), metadata=metadata)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_checkpoint(input_path: Path, output_path: Path, force: bool = False) -> int:
    if input_path.expanduser().resolve() == output_path.expanduser().resolve():
        raise ConversionError("input and output paths must be different")
    state = normalize_and_validate_state(load_checkpoint(input_path))
    inference_state = build_inference_state(state)
    save_atomic(inference_state, output_path, input_path, force)
    return len(inference_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SegviGen full_seg.ckpt to native mixed-precision safetensors."
    )
    parser.add_argument("input", type=Path, help="released full_seg.ckpt path")
    parser.add_argument("output", type=Path, help="output .safetensors path")
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        count = convert_checkpoint(args.input, args.output, args.force)
    except (ConversionError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} SegviGen Full flow tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
