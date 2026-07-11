#!/usr/bin/env python3
"""Convert an official TokenRig Lightning checkpoint to inference safetensors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConversionError(RuntimeError):
    pass


EXPECTED_MODEL_CONFIG = {
    "tokens_per_skin": 4,
    "tokens_skin_cond": 384,
}

# TokenRig overrides only hidden size and context length in its checkpoint.
# The remaining values come from the Qwen3-0.6B base config and must be made
# explicit so native inference never depends on an online Hugging Face config.
RESOLVED_QWEN_CONFIG = {
    "vocab_size": 33036,
    "hidden_size": 896,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 3192,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1.0e-6,
}

EXPECTED_TENSORS = {
    "vae.model.cond_encoder.proj_in.weight": (768, 54),
    "vae.model.cond_quant.weight": (512, 768),
    "vae.model.decoder.proj_out.weight": (1, 768),
    "vae.model.FSQ.project_out.weight": (512, 5),
    "mesh_encoder.encoder.input_proj.weight": (512, 54),
    "mesh_encoder.encoder.self_attn.resblocks.7.attn.c_qkv.weight": (1536, 512),
    "transformer.model.embed_tokens.weight": (33036, 896),
    "transformer.model.layers.27.self_attn.q_proj.weight": (2048, 896),
    "transformer.model.layers.27.self_attn.k_proj.weight": (1024, 896),
    "transformer.model.layers.27.mlp.down_proj.weight": (896, 3072),
    "transformer.model.norm.weight": (896,),
    "transformer.lm_head.weight": (33036, 896),
    "output_proj.0.weight": (896, 512),
    "output_proj.1.weight": (896,),
}

INFERENCE_PREFIXES = (
    "vae.",
    "mesh_encoder.",
    "transformer.",
    "output_proj.",
)

OUTPUT_COMPONENTS = {
    "mesh_encoder": (
        "mesh_encoder-bf16.safetensors",
        ("mesh_encoder.", "output_proj."),
    ),
    "qwen3": (
        "qwen3-tokenrig-bf16.safetensors",
        ("transformer.",),
    ),
    "skin_fsq_cvae": (
        "skin-fsq-cvae-bf16.safetensors",
        (
            "vae.model.cond_encoder.",
            "vae.model.cond_quant.",
            "vae.model.FSQ.project_out.",
            "vae.model.post_quant.",
            "vae.model.decoder.",
        ),
    ),
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _normalized_name(name: str) -> str:
    name = name.replace("_orig_mod.", "")
    if name.startswith("model."):
        name = name[len("model.") :]
    return name


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to read the Lightning checkpoint") from exc

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ConversionError(f"failed to load {path}: {exc}") from exc

    if not isinstance(checkpoint, Mapping):
        raise ConversionError("checkpoint root must be a mapping")
    state = checkpoint.get("state_dict")
    hyper = checkpoint.get("hyper_parameters")
    if not isinstance(state, Mapping) or not isinstance(hyper, Mapping):
        raise ConversionError("checkpoint must contain state_dict and hyper_parameters mappings")

    normalized: dict[str, Any] = {}
    for raw_name, tensor in state.items():
        if not isinstance(raw_name, str):
            raise ConversionError("state_dict contains a non-string tensor name")
        name = _normalized_name(raw_name)
        if name in normalized:
            raise ConversionError(f"duplicate normalized tensor name: {name}")
        normalized[name] = tensor
    return normalized, _plain(hyper)


def validate_contract(state: Mapping[str, Any], hyper: Mapping[str, Any]) -> None:
    model_config = hyper.get("model_config")
    tokenizer_config = hyper.get("tokenizer_config")
    if not isinstance(model_config, Mapping) or not isinstance(tokenizer_config, Mapping):
        raise ConversionError("TokenRig model_config/tokenizer_config are missing")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if model_config.get(key) != expected:
            raise ConversionError(
                f"unsupported model_config.{key}: {model_config.get(key)!r}, expected {expected!r}"
            )
    llm_config = model_config.get("llm")
    if not isinstance(llm_config, Mapping):
        raise ConversionError("TokenRig model_config.llm is missing")
    for key in ("hidden_size", "max_position_embeddings"):
        expected = RESOLVED_QWEN_CONFIG[key]
        if llm_config.get(key) != expected:
            raise ConversionError(
                f"unsupported model_config.llm.{key}: {llm_config.get(key)!r}, expected {expected!r}"
            )
    if tokenizer_config.get("num_discrete") != 256:
        raise ConversionError("unsupported tokenizer: num_discrete must be 256")
    if tokenizer_config.get("continuous_range") != [-1, 1]:
        raise ConversionError("unsupported tokenizer: continuous_range must be [-1, 1]")

    for name, expected_shape in EXPECTED_TENSORS.items():
        tensor = state.get(name)
        if tensor is None:
            raise ConversionError(f"checkpoint is missing required tensor {name!r}")
        shape = tuple(int(v) for v in tensor.shape)
        if shape != expected_shape:
            raise ConversionError(
                f"tensor {name!r} has shape {shape}, expected {expected_shape}"
            )

    embedding = state["transformer.model.embed_tokens.weight"]
    lm_head = state["transformer.lm_head.weight"]
    if not embedding.equal(lm_head):
        raise ConversionError("TokenRig expects tied Qwen input/output embeddings")


def build_inference_state(
    state: Mapping[str, Any],
    prefixes: tuple[str, ...] = INFERENCE_PREFIXES,
) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {}
    for name, tensor in state.items():
        if not name.startswith(prefixes):
            continue
        # Qwen ties lm_head to embed_tokens. Store it once and bind the same
        # tensor for both uses in the native runtime.
        if name == "transformer.lm_head.weight":
            continue
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError(f"state_dict value {name!r} is not a tensor")
        if tensor.is_complex() or tensor.is_sparse:
            raise ConversionError(f"unsupported tensor type for {name!r}")
        output[name] = tensor.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous().clone()
    if not output:
        raise ConversionError("no TokenRig inference tensors were selected")
    return output


def _metadata(
    hyper: Mapping[str, Any],
    source: Path,
    component: str,
) -> dict[str, str]:
    selected = {
        "model_config": hyper.get("model_config"),
        "transform_config": hyper.get("transform_config"),
        "tokenizer_config": hyper.get("tokenizer_config"),
        "resolved_qwen_config": RESOLVED_QWEN_CONFIG,
        # torch.nn.RMSNorm(eps=None) resolves to float32 epsilon in the
        # released PyTorch/CUDA kernel even when its input/output are BF16.
        "mesh_output_rms_eps": 1.1920928955078125e-7,
    }
    return {
        "format": "tokenskin-tokenrig-v1",
        "source": source.name,
        "weight_dtype": "bf16",
        "component": component,
        "config": json.dumps(selected, separators=(",", ":"), sort_keys=True),
    }


def save_atomic(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
    output_path: Path,
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
    try:
        save_file(dict(tensors), str(temporary_path), metadata=dict(metadata))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_checkpoint(
    input_path: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, int]:
    state, hyper = load_checkpoint(input_path)
    validate_contract(state, hyper)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for component, (filename, prefixes) in OUTPUT_COMPONENTS.items():
        inference_state = build_inference_state(state, prefixes)
        output_path = output_dir / filename
        save_atomic(
            inference_state,
            _metadata(hyper, input_path, component),
            output_path,
            force,
        )
        counts[component] = len(inference_state)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the official TokenRig Lightning checkpoint to native inference safetensors."
    )
    parser.add_argument("input", type=Path, help="grpo_1400.ckpt path")
    parser.add_argument("output", type=Path, help="output ckpts directory")
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        counts = convert_checkpoint(args.input, args.output, args.force)
    except (ConversionError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    total = sum(counts.values())
    detail = ", ".join(f"{name}={count}" for name, count in counts.items())
    print(f"wrote {total} TokenRig inference tensors to {args.output} ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
