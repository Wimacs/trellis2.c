#!/usr/bin/env python3
"""Opt-in real-weight smoke test for the automatic SegviGen pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ENVIRONMENT_PATHS = {
    "base model": "TRELLIS_SEGVIGEN_E2E_BASE_MODEL",
    "SegviGen model": "TRELLIS_SEGVIGEN_E2E_SEGMENTATION_MODEL",
    "DINO model": "TRELLIS_SEGVIGEN_E2E_DINO",
    "input GLB": "TRELLIS_SEGVIGEN_E2E_INPUT",
}


COMPONENT_FORMATS = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
IDENTITY_MATRIX = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise AssertionError(f"{path} is too small to be a GLB")
    magic, version, total_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise AssertionError(f"{path} has an invalid GLB header")
    json_payload: bytes | None = None
    binary_payload: bytes | None = None
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise AssertionError(f"{path} has a truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise AssertionError(f"{path} has a truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            if json_payload is not None:
                raise AssertionError(f"{path} has more than one JSON chunk")
            json_payload = payload[offset:end]
        elif chunk_type == 0x004E4942:
            if binary_payload is not None:
                raise AssertionError(f"{path} has more than one BIN chunk")
            binary_payload = payload[offset:end]
        offset = end
    if json_payload is None or binary_payload is None:
        raise AssertionError(f"{path} must contain JSON and BIN chunks")
    document = json.loads(json_payload.rstrip(b" \0"))
    buffers = document.get("buffers", [])
    if (len(buffers) != 1 or buffers[0].get("uri") is not None or
            not isinstance(buffers[0].get("byteLength"), int)):
        raise AssertionError(f"{path} must use one embedded GLB buffer")
    byte_length = buffers[0]["byteLength"]
    if byte_length < 0 or byte_length > len(binary_payload):
        raise AssertionError(f"{path} declares an invalid embedded buffer length")
    return document, binary_payload[:byte_length]


def accessor_values(
    document: dict[str, Any], binary: bytes, accessor_index: int
) -> tuple[dict[str, Any], list[tuple[int | float, ...]]]:
    accessors = document.get("accessors", [])
    if (not isinstance(accessor_index, int) or accessor_index < 0 or
            accessor_index >= len(accessors)):
        raise AssertionError(f"invalid accessor index {accessor_index!r}")
    accessor = accessors[accessor_index]
    if "sparse" in accessor:
        raise AssertionError("sparse accessors are not supported by this E2E verifier")
    view_index = accessor.get("bufferView")
    views = document.get("bufferViews", [])
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise AssertionError(f"accessor {accessor_index} has no valid bufferView")
    view = views[view_index]
    if view.get("buffer", 0) != 0:
        raise AssertionError(f"accessor {accessor_index} uses a non-GLB buffer")

    component_type = accessor.get("componentType")
    value_type = accessor.get("type")
    count = accessor.get("count")
    if component_type not in COMPONENT_FORMATS or value_type not in TYPE_WIDTHS:
        raise AssertionError(f"accessor {accessor_index} has an unsupported format")
    if not isinstance(count, int) or count < 0:
        raise AssertionError(f"accessor {accessor_index} has invalid count {count!r}")
    component_format = COMPONENT_FORMATS[component_type]
    component_size = struct.calcsize("<" + component_format)
    width = TYPE_WIDTHS[value_type]
    element_size = component_size * width

    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    accessor_offset = accessor.get("byteOffset", 0)
    stride = view.get("byteStride", element_size)
    integers = (view_offset, view_length, accessor_offset, stride)
    if any(not isinstance(value, int) or value < 0 for value in integers):
        raise AssertionError(f"accessor {accessor_index} has invalid byte bounds")
    if stride < element_size or stride % component_size != 0:
        raise AssertionError(f"accessor {accessor_index} has invalid byteStride {stride}")
    view_end = view_offset + view_length
    first = view_offset + accessor_offset
    end = first if count == 0 else first + (count - 1) * stride + element_size
    if first < view_offset or end > view_end or view_end > len(binary):
        raise AssertionError(f"accessor {accessor_index} exceeds its BIN bufferView")

    unpack_format = "<" + component_format * width
    return accessor, [
        struct.unpack_from(unpack_format, binary, first + item * stride)
        for item in range(count)
    ]


def matrix_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        sum(left[row + 4 * inner] * right[inner + 4 * column] for inner in range(4))
        for column in range(4)
        for row in range(4)
    )


def node_local_matrix(node: dict[str, Any]) -> tuple[float, ...]:
    if "matrix" in node:
        matrix = node["matrix"]
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise AssertionError("node matrix must contain 16 values")
        result = tuple(float(value) for value in matrix)
    else:
        translation = node.get("translation", [0.0, 0.0, 0.0])
        rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
        scale = node.get("scale", [1.0, 1.0, 1.0])
        if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
            raise AssertionError("node TRS has an invalid width")
        tx, ty, tz = map(float, translation)
        x, y, z, w = map(float, rotation)
        sx, sy, sz = map(float, scale)
        result = (
            (1.0 - 2.0 * (y * y + z * z)) * sx,
            (2.0 * (x * y + z * w)) * sx,
            (2.0 * (x * z - y * w)) * sx,
            0.0,
            (2.0 * (x * y - z * w)) * sy,
            (1.0 - 2.0 * (x * x + z * z)) * sy,
            (2.0 * (y * z + x * w)) * sy,
            0.0,
            (2.0 * (x * z + y * w)) * sz,
            (2.0 * (y * z - x * w)) * sz,
            (1.0 - 2.0 * (x * x + y * y)) * sz,
            0.0,
            tx, ty, tz, 1.0,
        )
    if not all(math.isfinite(value) for value in result):
        raise AssertionError("node transform contains non-finite values")
    return result


def transform_position(matrix: tuple[float, ...], point: tuple[int | float, ...]) -> tuple[float, float, float]:
    x, y, z = map(float, point)
    result = (
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    )
    if not all(math.isfinite(value) for value in result):
        raise AssertionError("transformed POSITION is non-finite")
    return result


def matrix_linear_determinant(matrix: tuple[float, ...]) -> float:
    return (
        matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
        - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
        + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2])
    )


def primitive_triangles(
    document: dict[str, Any],
    binary: bytes,
    primitive: dict[str, Any],
    world: tuple[float, ...],
) -> list[tuple[tuple[float, float, float], ...]]:
    position_index = primitive.get("attributes", {}).get("POSITION")
    position_accessor, position_values = accessor_values(document, binary, position_index)
    if (position_accessor.get("componentType") != 5126 or
            position_accessor.get("type") != "VEC3" or not position_values):
        raise AssertionError("POSITION must be a non-empty float VEC3 accessor")
    positions = [transform_position(world, point) for point in position_values]

    index_accessor_index = primitive.get("indices")
    if index_accessor_index is None:
        indices = list(range(len(positions)))
    else:
        index_accessor, index_values = accessor_values(
            document, binary, index_accessor_index
        )
        if (index_accessor.get("type") != "SCALAR" or
                index_accessor.get("componentType") not in (5121, 5123, 5125)):
            raise AssertionError("indices must be an unsigned integer SCALAR accessor")
        indices = [int(value[0]) for value in index_values]
    if any(index < 0 or index >= len(positions) for index in indices):
        raise AssertionError("primitive index exceeds POSITION count")

    mode = primitive.get("mode", 4)
    elements: list[tuple[int, int, int]] = []
    if mode == 4:
        if not indices or len(indices) % 3 != 0:
            raise AssertionError("triangle index count must be positive and divisible by 3")
        elements = [tuple(indices[item:item + 3]) for item in range(0, len(indices), 3)]
    elif mode in (5, 6):
        if len(indices) < 3:
            raise AssertionError("triangle strip/fan must contain at least three indices")
        for item in range(len(indices) - 2):
            triangle = (
                [indices[item], indices[item + 1], indices[item + 2]]
                if mode == 5 else [indices[0], indices[item + 1], indices[item + 2]]
            )
            if mode == 5 and item % 2 != 0:
                triangle[0], triangle[1] = triangle[1], triangle[0]
            elements.append(tuple(triangle))
    else:
        raise AssertionError(f"unsupported non-triangle primitive mode {mode}")
    if matrix_linear_determinant(world) < 0.0:
        elements = [(a, c, b) for a, b, c in elements]
    return [tuple(positions[index] for index in triangle) for triangle in elements]


def scene_triangle_multiset(
    document: dict[str, Any], binary: bytes
) -> Counter[tuple[tuple[int, int, int], ...]]:
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    scene_index = document.get("scene", 0)
    if (not scenes or not isinstance(scene_index, int) or scene_index < 0 or
            scene_index >= len(scenes)):
        raise AssertionError("GLB has no valid default scene")
    meshes = document.get("meshes", [])
    visiting: set[int] = set()
    triangles: Counter[tuple[tuple[int, int, int], ...]] = Counter()

    def canonical_triangle(
        triangle: tuple[tuple[float, float, float], ...]
    ) -> tuple[tuple[int, int, int], ...]:
        # C and Python evaluate node transforms with slightly different float
        # rounding. Quantizing to 1e-5 still distinguishes source faces while
        # making their world-space comparison stable.
        quantized = tuple(
            tuple(int(round(component * 100000.0)) for component in point)
            for point in triangle
        )
        # A cyclic rotation is the same oriented triangle; reversing two
        # vertices is not, because that would flip the exported face normal.
        return min(
            quantized,
            (quantized[1], quantized[2], quantized[0]),
            (quantized[2], quantized[0], quantized[1]),
        )

    def visit(node_index: int, parent: tuple[float, ...]) -> None:
        if node_index in visiting:
            raise AssertionError("GLB node graph contains a cycle")
        if node_index < 0 or node_index >= len(nodes):
            raise AssertionError(f"invalid node index {node_index}")
        visiting.add(node_index)
        node = nodes[node_index]
        world = matrix_multiply(parent, node_local_matrix(node))
        mesh_index = node.get("mesh")
        if isinstance(mesh_index, int):
            if mesh_index < 0 or mesh_index >= len(meshes):
                raise AssertionError(f"invalid mesh index {mesh_index}")
            for primitive in meshes[mesh_index].get("primitives", []):
                for triangle in primitive_triangles(document, binary, primitive, world):
                    triangles[canonical_triangle(triangle)] += 1
        for child in node.get("children", []):
            visit(child, world)
        visiting.remove(node_index)

    for node in scenes[scene_index].get("nodes", []):
        visit(node, IDENTITY_MATRIX)
    return triangles


def verify_parts(
    document: dict[str, Any],
    binary: bytes,
    expected_faces: int,
    source_material_count: int,
    require_multiple_parts: bool,
) -> int:
    scenes = document.get("scenes", [])
    nodes = document.get("nodes", [])
    if (document.get("scene", 0) != 0 or len(scenes) != 1 or
            len(scenes[0].get("nodes", [])) != 1):
        raise AssertionError("parts GLB must contain one assembly root")
    root_index = scenes[0]["nodes"][0]
    if not isinstance(root_index, int) or root_index < 0 or root_index >= len(nodes):
        raise AssertionError("parts GLB has an invalid assembly root")
    root = nodes[root_index]
    if root.get("name") != "trellis_parts":
        raise AssertionError("parts GLB root must be named trellis_parts")
    children = root.get("children", [])
    if not children:
        raise AssertionError("segmentation must produce at least one retained part")
    if require_multiple_parts and len(children) <= 1:
        raise AssertionError("segmentation must produce more than one part")
    if len(nodes) != len(children) + 1 or len(document.get("meshes", [])) != len(children):
        raise AssertionError("parts GLB must have one node and one mesh per part")
    mesh_indices: list[int] = []
    part_ids: list[int] = []
    total_faces = 0
    for expected_part_id, child_index in enumerate(children):
        if not isinstance(child_index, int) or child_index < 0 or child_index >= len(nodes):
            raise AssertionError("parts GLB contains an invalid child node index")
        node = nodes[child_index]
        mesh_index = node.get("mesh")
        part_id = node.get("extras", {}).get("trellis_part_id")
        if not isinstance(mesh_index, int) or not isinstance(part_id, int):
            raise AssertionError("each part node must contain a mesh and trellis_part_id")
        if node.get("name") != f"part_{expected_part_id:04d}":
            raise AssertionError("part node names must be dense and deterministic")
        meshes = document.get("meshes", [])
        if mesh_index < 0 or mesh_index >= len(meshes):
            raise AssertionError("part node references an invalid mesh")
        primitives = document["meshes"][mesh_index].get("primitives", [])
        if not primitives:
            raise AssertionError("each part mesh must have at least one primitive")
        mesh_indices.append(mesh_index)
        part_ids.append(part_id)
        for primitive in primitives:
            material_index = primitive.get("material")
            if (not isinstance(material_index, int) or material_index < 0 or
                    material_index >= len(document.get("materials", []))):
                raise AssertionError("part primitive references an invalid material")
            total_faces += len(primitive_triangles(
                document, binary, primitive, IDENTITY_MATRIX
            ))
    if len(set(mesh_indices)) != len(children):
        raise AssertionError("each part node must own a distinct mesh")
    if sorted(part_ids) != list(range(len(children))):
        raise AssertionError("part ids must be dense and zero-based")
    output_material_count = len(document.get("materials", []))
    expected_material_counts = {
        max(source_material_count, 1),
        source_material_count + 1,
    }
    if output_material_count not in expected_material_counts:
        raise AssertionError(
            "parts must share source materials, with at most one implicit-material fallback"
        )
    if total_faces != expected_faces:
        raise AssertionError(
            f"parts contain {total_faces} faces, expected exactly {expected_faces}"
        )
    return len(children)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument(
        "--small-part-mode",
        choices=("keep", "merge", "discard"),
        default=os.getenv("TRELLIS_SEGVIGEN_E2E_SMALL_PART_MODE", "keep"),
    )
    args = parser.parse_args()

    missing = [name for name, variable in ENVIRONMENT_PATHS.items() if not os.getenv(variable)]
    if missing:
        print(
            "SKIP: set the SegviGen E2E environment variables; missing "
            + ", ".join(missing)
        )
        return 77
    paths = {
        name: Path(os.environ[variable]).expanduser().resolve()
        for name, variable in ENVIRONMENT_PATHS.items()
    }
    for name, path in paths.items():
        if not path.exists():
            raise AssertionError(f"{name} does not exist: {path}")

    input_document, input_binary = read_glb(paths["input GLB"])
    input_triangles = scene_triangle_multiset(input_document, input_binary)
    expected_faces = sum(input_triangles.values())
    keep_output = os.getenv("TRELLIS_SEGVIGEN_E2E_OUTPUT")
    with tempfile.TemporaryDirectory(prefix="trellis-segvigen-e2e-") as directory:
        output = (
            Path(keep_output).expanduser().resolve()
            if keep_output
            else Path(directory) / "parts.glb"
        )
        command = [
            str(args.cli),
            "--model", str(paths["base model"]),
            "--segmentation-model", str(paths["SegviGen model"]),
            "--dino", str(paths["DINO model"]),
            "--input", str(paths["input GLB"]),
            "--output", str(output),
            "--steps", os.getenv("TRELLIS_SEGVIGEN_E2E_STEPS", "12"),
            "--seed", "42",
            "--small-part-mode", args.small_part_mode,
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(os.getenv("TRELLIS_SEGVIGEN_E2E_TIMEOUT", "1800")),
        )
        if completed.returncode != 0:
            print(completed.stdout)
            raise AssertionError(
                f"automatic SegviGen pipeline exited {completed.returncode}"
            )
        output_document, output_binary = read_glb(output)
        output_triangles = scene_triangle_multiset(output_document, output_binary)
        output_faces = sum(output_triangles.values())
        part_count = verify_parts(
            output_document,
            output_binary,
            output_faces if args.small_part_mode == "discard" else expected_faces,
            len(input_document.get("materials", [])),
            args.small_part_mode == "keep",
        )
        if args.small_part_mode != "discard" and output_triangles != input_triangles:
            missing = sum((input_triangles - output_triangles).values())
            duplicated = sum((output_triangles - input_triangles).values())
            raise AssertionError(
                "parts do not preserve the input triangle multiset: "
                f"missing={missing} duplicated_or_new={duplicated}"
            )
        if args.small_part_mode == "discard":
            unexpected = sum((output_triangles - input_triangles).values())
            if unexpected != 0 or output_faces > expected_faces:
                raise AssertionError(
                    "discard mode emitted triangles absent from the source: "
                    f"unexpected={unexpected} output={output_faces} "
                    f"input={expected_faces}"
                )
        print(
            f"SegviGen E2E passed: input_faces={expected_faces} "
            f"output_faces={output_faces} mode={args.small_part_mode} "
            f"parts={part_count} output={output}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
