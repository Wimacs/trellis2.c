#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

try:
    import trimesh
except ImportError:
    trimesh = None


SOURCE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE_DIR / "tools" / "convert_segvigen_fixture.py"
SPEC = importlib.util.spec_from_file_location("convert_segvigen_fixture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read_tslat(path: Path) -> tuple[tuple[object, ...], bytes, bytes]:
    data = path.read_bytes()
    header = MODULE.HEADER_STRUCT.unpack_from(data)
    coords_bytes = int(header[-2])
    feats_bytes = int(header[-1])
    coords_start = MODULE.TSLAT_HEADER_BYTES
    feats_start = coords_start + coords_bytes
    return header, data[coords_start:feats_start], data[feats_start : feats_start + feats_bytes]


@unittest.skipIf(torch is None or np is None, "PyTorch and NumPy are required")
class ConvertSegviGenFixtureTests(unittest.TestCase):
    def make_fixture(self, path: Path) -> tuple[object, object]:
        coords = torch.tensor(
            [[0, 1, 2, 3], [0, 511, 0, 4], [0, 17, 18, 19]],
            dtype=torch.int64,
        )
        feats = torch.arange(3 * 32, dtype=torch.float16).reshape(3, 32) / 13.0
        torch.save({"coords": coords, "feats": feats}, path)
        return coords, feats

    def test_writes_exact_tslat01_v1_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pth"
            output = root / "fixture.tslat"
            coords, feats = self.make_fixture(source)
            count = MODULE.convert_fixture(
                source,
                output,
                anchor_min=(-0.4, -0.3, -0.2),
                anchor_max=(0.4, 0.3, 0.2),
            )

            self.assertEqual(count, 3)
            header, coords_payload, feats_payload = read_tslat(output)
            self.assertEqual(header[0], b"TSLAT01\0")
            self.assertEqual(header[1:5], (1, 80, 512, 32))
            self.assertEqual(header[5:8], (3, 1, 0))
            np.testing.assert_allclose(header[8:11], (-0.4, -0.3, -0.2), rtol=0, atol=1e-7)
            np.testing.assert_allclose(header[11:14], (0.4, 0.3, 0.2), rtol=0, atol=1e-7)
            self.assertEqual(header[14:], (3 * 4 * 4, 3 * 32 * 4))
            np.testing.assert_array_equal(
                np.frombuffer(coords_payload, dtype="<i4").reshape(3, 4),
                coords.numpy().astype(np.int32),
            )
            np.testing.assert_array_equal(
                np.frombuffer(feats_payload, dtype="<f4").reshape(3, 32),
                feats.float().numpy(),
            )
            self.assertEqual(output.stat().st_size, 80 + 3 * 4 * 4 + 3 * 32 * 4)

    def test_refuses_existing_output_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pth"
            output = root / "fixture.tslat"
            self.make_fixture(source)
            output.write_bytes(b"sentinel")
            kwargs = {
                "anchor_min": (-0.4, -0.3, -0.2),
                "anchor_max": (0.4, 0.3, 0.2),
            }
            with self.assertRaises(FileExistsError):
                MODULE.convert_fixture(source, output, **kwargs)
            self.assertEqual(output.read_bytes(), b"sentinel")
            MODULE.convert_fixture(source, output, force=True, **kwargs)
            self.assertEqual(output.read_bytes()[:8], b"TSLAT01\0")

    def test_strictly_rejects_malformed_sparse_latents(self) -> None:
        valid_coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
        valid_feats = torch.zeros((2, 32), dtype=torch.float32)
        cases = (
            (
                {"coords": valid_coords, "feats": valid_feats, "extra": torch.tensor(0)},
                "exactly",
            ),
            (
                {"coords": valid_coords.clone().fill_(1), "feats": valid_feats},
                "batch column",
            ),
            (
                {"coords": valid_coords[[0, 0]], "feats": valid_feats},
                "duplicate",
            ),
            (
                {
                    "coords": valid_coords,
                    "feats": valid_feats.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
                },
                "NaN",
            ),
        )
        for fixture, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(MODULE.ConversionError, message):
                    MODULE.normalize_and_validate_fixture(fixture, 512)

    def test_rejects_anchor_outside_decoder_frame(self) -> None:
        with self.assertRaisesRegex(MODULE.ConversionError, "decoder frame"):
            MODULE.validate_anchor_aabb((-0.6, -0.1, -0.1), (0.4, 0.1, 0.1))

    def test_anchor_limit_tracks_flex_dual_grid_resolution(self) -> None:
        for resolution in MODULE.SUPPORTED_RESOLUTIONS:
            with self.subTest(resolution=resolution):
                limit = MODULE.decoder_aabb_limit(resolution)
                minimum, maximum = MODULE.validate_anchor_aabb(
                    (-limit + 1e-7, -0.1, -0.1),
                    (limit - 1e-7, 0.1, 0.1),
                    resolution,
                )
                self.assertLess(minimum[0], -0.5)
                self.assertGreater(maximum[0], 0.5)
                with self.assertRaisesRegex(MODULE.ConversionError, "decoder frame"):
                    MODULE.validate_anchor_aabb(
                        (-limit - 1e-4, -0.1, -0.1),
                        (0.4, 0.1, 0.1),
                        resolution,
                    )

    @unittest.skipIf(trimesh is None, "trimesh is required for GLB anchor coverage")
    def test_glb_anchor_uses_direct_world_axes_and_official_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pth"
            output = root / "fixture.tslat"
            glb = root / "anchor.glb"
            self.make_fixture(source)

            mesh = trimesh.Trimesh(
                vertices=np.asarray(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 6.0]],
                    dtype=np.float64,
                ),
                faces=np.asarray([[0, 1, 2]], dtype=np.int64),
                process=False,
            )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = (10.0, -3.0, 2.0)
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="offset_triangle", transform=transform)
            glb.write_bytes(scene.export(file_type="glb"))

            MODULE.convert_fixture(source, output, glb_path=glb)
            header, _, _ = read_tslat(output)
            scale = MODULE.SEGVIGEN_NORMALIZED_EXTENT / 6.0
            np.testing.assert_allclose(
                header[8:11], (-1.0 * scale, -2.0 * scale, -3.0 * scale), atol=1e-7
            )
            np.testing.assert_allclose(
                header[11:14], (1.0 * scale, 2.0 * scale, 3.0 * scale), atol=1e-7
            )


if __name__ == "__main__":
    unittest.main()
