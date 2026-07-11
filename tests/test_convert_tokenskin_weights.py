#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The converter reports this dependency at runtime.
    torch = None


SOURCE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE_DIR / "tools" / "convert_tokenskin_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_tokenskin_weights", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyTensor:
    def __init__(self, shape: tuple[int, ...], equal: bool = True):
        self.shape = shape
        self._equal = equal

    def equal(self, other: object) -> bool:
        return self._equal and isinstance(other, DummyTensor)


class ConvertTokenSkinWeightsTests(unittest.TestCase):
    def test_normalizes_compiled_lightning_names(self) -> None:
        self.assertEqual(
            MODULE._normalized_name("model.vae.model._orig_mod.decoder.weight"),
            "vae.model.decoder.weight",
        )

    def test_checkpoint_contract(self) -> None:
        state = {
            name: DummyTensor(shape)
            for name, shape in MODULE.EXPECTED_TENSORS.items()
        }
        hyper = {
            "model_config": {
                **MODULE.EXPECTED_MODEL_CONFIG,
                "llm": {
                    "hidden_size": 896,
                    "max_position_embeddings": 3192,
                },
            },
            "tokenizer_config": {
                "num_discrete": 256,
                "continuous_range": [-1, 1],
            },
        }
        MODULE.validate_contract(state, hyper)

    def test_rejects_untied_qwen_head(self) -> None:
        state = {
            name: DummyTensor(shape)
            for name, shape in MODULE.EXPECTED_TENSORS.items()
        }
        state["transformer.model.embed_tokens.weight"] = DummyTensor(
            MODULE.EXPECTED_TENSORS["transformer.model.embed_tokens.weight"],
            equal=False,
        )
        hyper = {
            "model_config": {
                **MODULE.EXPECTED_MODEL_CONFIG,
                "llm": {
                    "hidden_size": 896,
                    "max_position_embeddings": 3192,
                },
            },
            "tokenizer_config": {
                "num_discrete": 256,
                "continuous_range": [-1, 1],
            },
        }
        with self.assertRaisesRegex(MODULE.ConversionError, "tied Qwen"):
            MODULE.validate_contract(state, hyper)

    def test_selects_only_inference_tensors_and_deduplicates_head(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        state = {
            "vae.x": torch.ones(2, dtype=torch.float32),
            "mesh_encoder.x": torch.ones(2, dtype=torch.float32),
            "transformer.model.embed_tokens.weight": torch.ones(2, 2),
            "transformer.lm_head.weight": torch.ones(2, 2),
            "output_proj.x": torch.ones(2, dtype=torch.float32),
            "optimizer.state": torch.ones(2, dtype=torch.float32),
        }
        converted = MODULE.build_inference_state(state)
        self.assertNotIn("transformer.lm_head.weight", converted)
        self.assertNotIn("optimizer.state", converted)
        self.assertEqual(len(converted), 4)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in converted.values()))

        mesh = MODULE.build_inference_state(
            state, ("mesh_encoder.", "output_proj.")
        )
        self.assertEqual(set(mesh), {"mesh_encoder.x", "output_proj.x"})


if __name__ == "__main__":
    unittest.main()
