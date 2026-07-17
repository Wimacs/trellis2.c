#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "download_weights.py"
SPEC = importlib.util.spec_from_file_location("download_weights", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download_weights = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_weights
SPEC.loader.exec_module(download_weights)


class DownloadWeightsTests(unittest.TestCase):
    def test_default_dino_repo_is_anonymous_mirror(self) -> None:
        self.assertEqual(
            download_weights.DEFAULT_DINO_REPO,
            "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        )

    def test_default_plan_includes_every_required_repository(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                str(SCRIPT),
                "--dry-run",
                "--output-dir",
                output_dir,
            ]
            stream = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stream):
                result = download_weights.main()

        output = stream.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("microsoft/TRELLIS.2-4B", output)
        self.assertIn("microsoft/TRELLIS-image-large", output)
        self.assertIn("camenduru/dinov3-vitl16-pretrain-lvd1689m", output)
        self.assertIn("Acly/BiRefNet-GGUF", output)
        self.assertIn("DINOv3 License (not MIT)", output)
        self.assertIn("not covered by the trellis2.c MIT License", output)
        self.assertIn("Dry run complete.", output)


if __name__ == "__main__":
    unittest.main()
