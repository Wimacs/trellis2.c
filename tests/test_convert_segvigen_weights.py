#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


SOURCE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE_DIR / "tools" / "convert_segvigen_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_segvigen_weights", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class ConvertSegviGenWeightsTests(unittest.TestCase):
    def test_contract_matches_released_flow_architecture(self) -> None:
        self.assertEqual(MODULE.EXPECTED_TENSOR_COUNT, 640)
        self.assertEqual(len(MODULE.EXPECTED_TENSOR_SHAPES), 640)
        self.assertEqual(
            sum(dtype == "bf16" for dtype in MODULE.EXPECTED_TENSOR_DTYPES.values()),
            420,
        )
        self.assertEqual(
            sum(dtype == "f32" for dtype in MODULE.EXPECTED_TENSOR_DTYPES.values()),
            220,
        )
        self.assertEqual(
            MODULE.EXPECTED_TENSOR_SHAPES["input_layer.weight"],
            (1536, 64),
        )
        self.assertEqual(
            MODULE.EXPECTED_TENSOR_SHAPES["blocks.29.cross_attn.to_kv.weight"],
            (3072, 1024),
        )
        self.assertEqual(
            MODULE.EXPECTED_TENSOR_SHAPES["out_layer.weight"],
            (32, 1536),
        )
        self.assertEqual(
            MODULE.EXPECTED_TENSOR_DTYPES["blocks.0.self_attn.to_qkv.weight"],
            "bf16",
        )
        self.assertEqual(
            MODULE.EXPECTED_TENSOR_DTYPES["blocks.0.self_attn.q_rms_norm.gamma"],
            "f32",
        )

    def test_rejects_foreign_state_dict_names_before_tensor_access(self) -> None:
        with self.assertRaisesRegex(MODULE.ConversionError, "unexpected state_dict"):
            MODULE.normalize_and_validate_state({"optimizer.state": DummyTensor((1,))})

    def test_reports_missing_and_extra_flow_tensors(self) -> None:
        state = {
            MODULE.SOURCE_PREFIX + name: DummyTensor(shape)
            for name, shape in MODULE.EXPECTED_TENSOR_SHAPES.items()
        }
        del state[MODULE.SOURCE_PREFIX + "out_layer.bias"]
        state[MODULE.SOURCE_PREFIX + "unexpected.weight"] = DummyTensor((1,))
        with self.assertRaisesRegex(MODULE.ConversionError, "missing=.*out_layer.bias"):
            MODULE.normalize_and_validate_state(state)

    def test_preserves_released_mixed_precision_contract(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        state = {
            "a": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "b": torch.ones(3, dtype=torch.bfloat16),
        }
        converted = MODULE.build_inference_state(state)
        self.assertEqual(set(converted), {"a", "b"})
        self.assertEqual(converted["a"].dtype, torch.float32)
        self.assertEqual(converted["b"].dtype, torch.bfloat16)
        self.assertTrue(all(tensor.is_contiguous() for tensor in converted.values()))
        self.assertIsNot(converted["b"], state["b"])

    def test_rejects_wrong_released_tensor_dtype(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        state = {
            MODULE.SOURCE_PREFIX + name: torch.empty(
                shape,
                dtype=(
                    torch.bfloat16
                    if MODULE.EXPECTED_TENSOR_DTYPES[name] == "bf16"
                    else torch.float32
                ),
                device="meta",
            )
            for name, shape in MODULE.EXPECTED_TENSOR_SHAPES.items()
        }
        state[
            MODULE.SOURCE_PREFIX + "blocks.0.self_attn.to_qkv.weight"
        ] = torch.empty(
            MODULE.EXPECTED_TENSOR_SHAPES[
                "blocks.0.self_attn.to_qkv.weight"
            ],
            dtype=torch.float32,
            device="meta",
        )
        with self.assertRaisesRegex(MODULE.ConversionError, "dtype.*expected"):
            MODULE.normalize_and_validate_state(state)

    def test_rejects_same_input_and_output_path_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_seg.ckpt"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(
                MODULE.ConversionError, "input and output paths must be different"
            ):
                MODULE.convert_checkpoint(path, path, force=True)

    def test_atomic_safetensors_output_reopens_with_metadata(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest("safetensors is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full_seg.ckpt"
            output = root / "flow.safetensors"
            source.write_bytes(b"fixture")
            tensors = {
                "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "bf16": torch.ones(4, dtype=torch.bfloat16),
            }
            MODULE.save_atomic(tensors, output, source, force=False)
            loaded = load_file(output)
            self.assertTrue(torch.equal(loaded["f32"], tensors["f32"]))
            self.assertTrue(torch.equal(loaded["bf16"], tensors["bf16"]))
            with safe_open(output, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["format"], "segvigen-full-flow-v1")
            self.assertEqual(metadata["weight_dtype"], "mixed-f32-bf16")
            with self.assertRaises(FileExistsError):
                MODULE.save_atomic(tensors, output, source, force=False)


if __name__ == "__main__":
    unittest.main()
