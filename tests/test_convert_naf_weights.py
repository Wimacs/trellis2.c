#!/usr/bin/env python3
"""Offline tests for tools/convert_naf_weights.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import convert_naf_weights as converter  # noqa: E402


def synthetic_state_dict() -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for index, (name, shape) in enumerate(converter.EXPECTED_TENSOR_SHAPES.items()):
        value = torch.full(shape, float(index), dtype=torch.float64)
        if name == "image_encoder.encoder.1.conv1.weight":
            value = value.transpose(0, 1)
            assert not value.is_contiguous()
        elif name == "image_encoder.encoder.0.bias":
            value = value.to(torch.float16)
        elif name == "image_encoder.encoder.1.norm1.weight":
            value = value.to(torch.int64)
        state_dict[name] = value
    return state_dict


class ConvertNafWeightsTest(unittest.TestCase):
    def test_conversion_preserves_all_keys_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "naf_release.pth"
            output_path = temp / "naf_release.safetensors"
            source = synthetic_state_dict()
            torch.save(source, checkpoint_path)

            count = converter.convert_checkpoint(
                str(checkpoint_path),
                output_path,
            )

            self.assertEqual(count, converter.EXPECTED_TENSOR_COUNT)
            converted = load_file(str(output_path), device="cpu")
            self.assertEqual(set(converted), set(source))
            for name, tensor in converted.items():
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertTrue(tensor.is_contiguous())
                torch.testing.assert_close(tensor, source[name].to(torch.float32))

            with safe_open(str(output_path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["format"], "pt")
            self.assertEqual(metadata["source"], str(checkpoint_path))

    def test_rejects_non_tensor_state_dict_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "bad.pth"
            output_path = temp / "bad.safetensors"
            state_dict = synthetic_state_dict()
            state_dict["image_encoder.encoder.0.weight"] = "not a tensor"  # type: ignore[assignment]
            torch.save(state_dict, checkpoint_path)

            with self.assertRaisesRegex(converter.ConversionError, "must be Tensor"):
                converter.convert_checkpoint(str(checkpoint_path), output_path)
            self.assertFalse(output_path.exists())

    def test_existing_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "naf_release.pth"
            output_path = temp / "naf_release.safetensors"
            torch.save(synthetic_state_dict(), checkpoint_path)
            output_path.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                converter.convert_checkpoint(str(checkpoint_path), output_path)
            self.assertEqual(output_path.read_bytes(), b"existing")

            count = converter.convert_checkpoint(
                str(checkpoint_path),
                output_path,
                force=True,
            )
            self.assertEqual(count, converter.EXPECTED_TENSOR_COUNT)
            self.assertEqual(len(load_file(str(output_path))), converter.EXPECTED_TENSOR_COUNT)

    def test_rejects_missing_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "missing.pth"
            state_dict = synthetic_state_dict()
            state_dict.pop("image_encoder.rope.periods")
            torch.save(state_dict, checkpoint_path)

            with self.assertRaisesRegex(converter.ConversionError, "missing=.*rope.periods"):
                converter.convert_checkpoint(str(checkpoint_path), temp / "missing.safetensors")

    def test_rejects_extra_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "extra.pth"
            state_dict = synthetic_state_dict()
            state_dict["image_encoder.unexpected.weight"] = torch.zeros(1)
            torch.save(state_dict, checkpoint_path)

            with self.assertRaisesRegex(converter.ConversionError, "extra=.*unexpected.weight"):
                converter.convert_checkpoint(str(checkpoint_path), temp / "extra.safetensors")

    def test_rejects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint_path = temp / "wrong-shape.pth"
            state_dict = synthetic_state_dict()
            state_dict["image_encoder.rope.periods"] = torch.zeros(15)
            torch.save(state_dict, checkpoint_path)

            with self.assertRaisesRegex(converter.ConversionError, "has shape .* expected"):
                converter.convert_checkpoint(str(checkpoint_path), temp / "wrong-shape.safetensors")


if __name__ == "__main__":
    unittest.main()
