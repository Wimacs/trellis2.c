#!/usr/bin/env python3
"""Unit tests for the strict GLB functional-result verifier."""

from __future__ import annotations

import copy
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import verify_glb as verifier  # noqa: E402


_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def minimal_document() -> tuple[dict[str, object], bytes]:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    index_padding = b"\x00\x00"
    image_offset = len(positions) + len(indices) + len(index_padding)
    binary = positions + indices + index_padding + _PNG_1X1
    document: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "test_verify_glb"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(_PNG_1X1)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    return document, binary


def build_glb(
    document: dict[str, object],
    binary: bytes | None,
    *,
    magic: bytes = b"glTF",
    version: int = 2,
    declared_length_delta: int = 0,
) -> bytes:
    json_data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_data += b" " * (-len(json_data) % 4)
    chunks = struct.pack("<II", len(json_data), verifier.JSON_CHUNK_TYPE) + json_data
    if binary is not None:
        binary += b"\x00" * (-len(binary) % 4)
        chunks += struct.pack("<II", len(binary), verifier.BIN_CHUNK_TYPE) + binary
    total_length = 12 + len(chunks) + declared_length_delta
    return struct.pack("<4sII", magic, version, total_length) + chunks


class VerifyGlbTest(unittest.TestCase):
    def valid_fixture(self) -> tuple[dict[str, object], bytes]:
        document, binary = minimal_document()
        return copy.deepcopy(document), binary

    def assert_invalid(self, data: bytes, pattern: str) -> None:
        with self.assertRaisesRegex(verifier.GlbValidationError, pattern):
            verifier.verify_glb(data)

    def test_accepts_minimal_renderable_glb_from_bytes_and_path(self) -> None:
        document, binary = self.valid_fixture()
        data = build_glb(document, binary)
        summary = verifier.verify_glb(data)
        self.assertEqual(summary.mesh_count, 1)
        self.assertEqual(summary.primitive_count, 1)
        self.assertEqual(summary.accessor_count, 2)
        self.assertEqual(summary.pbr_material_count, 1)
        self.assertEqual(summary.embedded_image_count, 1)
        self.assertEqual(summary.byte_length, len(data))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid.glb"
            path.write_bytes(data)
            self.assertEqual(verifier.verify_glb(path), summary)

    def test_rejects_bad_magic_version_and_header_length(self) -> None:
        document, binary = self.valid_fixture()
        with self.subTest("magic"):
            self.assert_invalid(build_glb(document, binary, magic=b"BAD!"), "magic")
        with self.subTest("version"):
            self.assert_invalid(build_glb(document, binary, version=1), "version")
        with self.subTest("length"):
            self.assert_invalid(
                build_glb(document, binary, declared_length_delta=4),
                "header length",
            )

    def test_rejects_truncated_and_unaligned_chunks(self) -> None:
        document, binary = self.valid_fixture()
        data = bytearray(build_glb(document, binary))
        json_length = struct.unpack_from("<I", data, 12)[0]
        bin_header_offset = 20 + json_length
        bin_length = struct.unpack_from("<I", data, bin_header_offset)[0]
        struct.pack_into("<I", data, bin_header_offset, bin_length + 4)
        self.assert_invalid(bytes(data), "chunk.*beyond")

        data = bytearray(build_glb(document, binary))
        struct.pack_into("<I", data, 12, json_length - 1)
        self.assert_invalid(bytes(data), "unaligned")

    def test_rejects_buffer_view_beyond_declared_buffer(self) -> None:
        document, binary = self.valid_fixture()
        document["bufferViews"][0]["byteLength"] = len(binary) + 1  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), r"bufferViews\[0\].*exceeds")

    def test_rejects_accessor_beyond_buffer_view(self) -> None:
        document, binary = self.valid_fixture()
        document["accessors"][0]["count"] = 4  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), r"accessors\[0\].*beyond")

    def test_rejects_mesh_without_position_or_indices(self) -> None:
        document, binary = self.valid_fixture()
        del document["meshes"][0]["primitives"][0]["indices"]  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), "no non-empty indexed mesh")

        document, binary = self.valid_fixture()
        del document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), "no non-empty indexed mesh")

    def test_rejects_empty_or_wrongly_typed_mesh_accessor(self) -> None:
        document, binary = self.valid_fixture()
        document["accessors"][0]["type"] = "VEC2"  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), "no non-empty indexed mesh")

        document, binary = self.valid_fixture()
        document["accessors"][1]["componentType"] = 5126  # type: ignore[index]
        document["bufferViews"][1]["byteLength"] = 12  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), "no non-empty indexed mesh")

    def test_rejects_missing_pbr_material(self) -> None:
        document, binary = self.valid_fixture()
        document["materials"] = [{}]
        self.assert_invalid(build_glb(document, binary), "no PBR")

    def test_rejects_external_or_invalid_embedded_image(self) -> None:
        document, binary = self.valid_fixture()
        document["images"] = [{"uri": "texture.png"}]
        self.assert_invalid(build_glb(document, binary), "external")

        document, binary = self.valid_fixture()
        document["bufferViews"][2]["byteOffset"] += 1  # type: ignore[index,operator]
        document["bufferViews"][2]["byteLength"] -= 1  # type: ignore[index,operator]
        self.assert_invalid(build_glb(document, binary), "does not contain PNG")

    def test_rejects_unlinked_embedded_texture(self) -> None:
        document, binary = self.valid_fixture()
        document["materials"][0]["pbrMetallicRoughness"] = {}  # type: ignore[index]
        self.assert_invalid(build_glb(document, binary), "backed by an embedded image")

    def test_command_line_exit_codes(self) -> None:
        document, binary = self.valid_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.glb"
            invalid_path = Path(temp_dir) / "invalid.glb"
            valid_path.write_bytes(build_glb(document, binary))
            invalid_path.write_bytes(b"not a glb")
            success = subprocess.run(
                [sys.executable, str(TESTS_DIR / "verify_glb.py"), str(valid_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            failure = subprocess.run(
                [sys.executable, str(TESTS_DIR / "verify_glb.py"), str(invalid_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("valid GLB", success.stdout)
        self.assertEqual(failure.returncode, 1)
        self.assertIn("invalid GLB", failure.stderr)


if __name__ == "__main__":
    unittest.main()
