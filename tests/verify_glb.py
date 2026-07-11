#!/usr/bin/env python3
"""Strict, dependency-free validation for self-contained mesh GLB files.

The module can be imported from tests::

    summary = verify_glb(Path("result.glb"))

or invoked directly::

    python3 tests/verify_glb.py result.glb

It deliberately validates the pieces required from an image-to-3D result rather
than trying to implement every optional part of the glTF 2.0 specification.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote_to_bytes


GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942
_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<II")

_COMPONENT_SIZES = {
    5120: 1,  # BYTE
    5121: 1,  # UNSIGNED_BYTE
    5122: 2,  # SHORT
    5123: 2,  # UNSIGNED_SHORT
    5125: 4,  # UNSIGNED_INT
    5126: 4,  # FLOAT
}
_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_INDEX_COMPONENT_TYPES = {5121, 5123, 5125}


class GlbValidationError(ValueError):
    """Raised when a GLB is malformed or lacks the required renderable data."""


@dataclass(frozen=True)
class GlbSummary:
    """Small success result useful to callers and command-line output."""

    byte_length: int
    mesh_count: int
    primitive_count: int
    accessor_count: int
    pbr_material_count: int
    embedded_image_count: int


@dataclass(frozen=True)
class _Chunk:
    chunk_type: int
    data: bytes


def _fail(message: str) -> None:
    raise GlbValidationError(message)


def _array(document: Mapping[str, Any], name: str) -> list[Any]:
    value = document.get(name, [])
    if not isinstance(value, list):
        _fail(f"{name} must be an array")
    return value


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    # bool is an int subclass in Python, but is never a valid glTF index/size.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{location} must be an integer >= {minimum}")
    return value


def _index(value: Any, length: int, location: str) -> int:
    index = _integer(value, location)
    if index >= length:
        _fail(f"{location} index {index} is out of range (count={length})")
    return index


def _decode_data_uri(uri: str, location: str) -> tuple[str, bytes]:
    if not uri.startswith("data:"):
        _fail(f"{location} is external; a self-contained GLB is required")
    header, separator, payload = uri[5:].partition(",")
    if not separator:
        _fail(f"{location} is a malformed data URI")
    fields = header.split(";")
    media_type = fields[0].lower()
    try:
        if fields[-1].lower() == "base64":
            decoded = base64.b64decode(payload, validate=True)
        else:
            decoded = unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as error:
        _fail(f"{location} has invalid encoded data: {error}")
    return media_type, decoded


def _read_source(source: str | Path | bytes | bytearray | memoryview) -> bytes:
    if isinstance(source, (str, Path)):
        try:
            return Path(source).read_bytes()
        except OSError as error:
            raise GlbValidationError(f"cannot read {source}: {error}") from error
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    raise TypeError("source must be a path or bytes-like object")


def _parse_container(data: bytes) -> tuple[Mapping[str, Any], bytes | None]:
    if len(data) < _HEADER.size:
        _fail(f"GLB header is truncated ({len(data)} bytes)")
    magic, version, declared_length = _HEADER.unpack_from(data)
    if magic != GLB_MAGIC:
        _fail(f"invalid GLB magic {magic!r}; expected {GLB_MAGIC!r}")
    if version != GLB_VERSION:
        _fail(f"unsupported GLB version {version}; expected 2")
    if declared_length != len(data):
        _fail(
            f"GLB header length {declared_length} does not match file length {len(data)}"
        )

    chunks: list[_Chunk] = []
    offset = _HEADER.size
    while offset < len(data):
        if len(data) - offset < _CHUNK_HEADER.size:
            _fail(f"chunk header at byte {offset} is truncated")
        chunk_length, chunk_type = _CHUNK_HEADER.unpack_from(data, offset)
        if chunk_length % 4 != 0:
            _fail(f"chunk at byte {offset} has unaligned length {chunk_length}")
        payload_start = offset + _CHUNK_HEADER.size
        payload_end = payload_start + chunk_length
        if payload_end > len(data):
            _fail(
                f"chunk at byte {offset} ends at {payload_end}, beyond file length {len(data)}"
            )
        chunks.append(_Chunk(chunk_type, data[payload_start:payload_end]))
        offset = payload_end

    if not chunks:
        _fail("GLB has no JSON chunk")
    if chunks[0].chunk_type != JSON_CHUNK_TYPE:
        _fail("the first GLB chunk is not JSON")
    if sum(chunk.chunk_type == JSON_CHUNK_TYPE for chunk in chunks) != 1:
        _fail("GLB must contain exactly one JSON chunk")
    if sum(chunk.chunk_type == BIN_CHUNK_TYPE for chunk in chunks) > 1:
        _fail("GLB must not contain more than one BIN chunk")
    if any(chunk.chunk_type not in (JSON_CHUNK_TYPE, BIN_CHUNK_TYPE) for chunk in chunks):
        _fail("GLB contains an unsupported chunk type")
    if len(chunks) > 1 and chunks[1].chunk_type != BIN_CHUNK_TYPE:
        _fail("the optional BIN chunk must immediately follow the JSON chunk")

    try:
        document = json.loads(chunks[0].data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON chunk: {error}")
    if not isinstance(document, dict):
        _fail("the JSON chunk root must be an object")

    asset = _object(document.get("asset"), "asset")
    if asset.get("version") != "2.0":
        _fail("asset.version must be exactly '2.0'")
    binary = next(
        (chunk.data for chunk in chunks if chunk.chunk_type == BIN_CHUNK_TYPE), None
    )
    return document, binary


def _resolve_buffers(
    document: Mapping[str, Any], binary: bytes | None
) -> tuple[list[bytes], list[int]]:
    buffers = _array(document, "buffers")
    payloads: list[bytes] = []
    declared_lengths: list[int] = []
    bin_claimed = False
    for index, raw_buffer in enumerate(buffers):
        buffer = _object(raw_buffer, f"buffers[{index}]")
        length = _integer(buffer.get("byteLength"), f"buffers[{index}].byteLength")
        uri = buffer.get("uri")
        if uri is None:
            if index != 0:
                _fail("only buffers[0] may refer to the GLB BIN chunk")
            if bin_claimed:
                _fail("the GLB BIN chunk is referenced by more than one buffer")
            if binary is None:
                _fail("buffers[0] requires a BIN chunk, but none is present")
            if len(binary) < length or len(binary) > length + 3:
                _fail(
                    f"BIN chunk length {len(binary)} is incompatible with "
                    f"buffers[{index}].byteLength {length}"
                )
            payload = binary[:length]
            bin_claimed = True
        else:
            if not isinstance(uri, str):
                _fail(f"buffers[{index}].uri must be a string")
            _, payload = _decode_data_uri(uri, f"buffers[{index}].uri")
            if len(payload) != length:
                _fail(
                    f"buffers[{index}] data length {len(payload)} does not match "
                    f"byteLength {length}"
                )
        payloads.append(payload)
        declared_lengths.append(length)

    if binary is not None and not bin_claimed:
        _fail("GLB contains an unreferenced BIN chunk")
    return payloads, declared_lengths


def _validate_buffer_views(
    document: Mapping[str, Any], buffer_lengths: Sequence[int]
) -> list[tuple[int, int, int, int | None]]:
    views = _array(document, "bufferViews")
    validated: list[tuple[int, int, int, int | None]] = []
    for index, raw_view in enumerate(views):
        view = _object(raw_view, f"bufferViews[{index}]")
        buffer_index = _index(
            view.get("buffer"), len(buffer_lengths), f"bufferViews[{index}].buffer"
        )
        offset = _integer(view.get("byteOffset", 0), f"bufferViews[{index}].byteOffset")
        length = _integer(
            view.get("byteLength"), f"bufferViews[{index}].byteLength", minimum=1
        )
        end = offset + length
        if end > buffer_lengths[buffer_index]:
            _fail(
                f"bufferViews[{index}] range [{offset}, {end}) exceeds "
                f"buffers[{buffer_index}].byteLength {buffer_lengths[buffer_index]}"
            )
        raw_stride = view.get("byteStride")
        stride = None
        if raw_stride is not None:
            stride = _integer(raw_stride, f"bufferViews[{index}].byteStride", minimum=4)
            if stride > 252 or stride % 4 != 0:
                _fail(f"bufferViews[{index}].byteStride must be a multiple of 4 in [4, 252]")
        validated.append((buffer_index, offset, length, stride))
    return validated


def _accessor_element_size(component_type: int, accessor_type: str) -> int:
    component_size = _COMPONENT_SIZES[component_type]
    component_count = _TYPE_COMPONENT_COUNTS[accessor_type]
    # MAT2/MAT3 columns using 1- or 2-byte components are padded to 4-byte
    # boundaries (glTF 2.0 accessor data alignment rule).
    if accessor_type.startswith("MAT"):
        dimension = int(accessor_type[-1])
        column_size = component_size * dimension
        padded_column_size = (column_size + 3) & ~3
        return padded_column_size * dimension
    return component_size * component_count


def _validate_range(
    *,
    location: str,
    view_index: int,
    byte_offset: int,
    count: int,
    element_size: int,
    component_size: int,
    views: Sequence[tuple[int, int, int, int | None]],
    use_stride: bool,
) -> None:
    _, view_offset, view_length, view_stride = views[view_index]
    if byte_offset % component_size != 0 or (view_offset + byte_offset) % component_size != 0:
        _fail(f"{location} is not aligned to its component size {component_size}")
    stride = view_stride if use_stride and view_stride is not None else element_size
    if stride < element_size:
        _fail(f"{location} element size {element_size} exceeds byteStride {stride}")
    required_end = byte_offset + (count - 1) * stride + element_size
    if required_end > view_length:
        _fail(
            f"{location} ends at byte {required_end}, beyond "
            f"bufferViews[{view_index}].byteLength {view_length}"
        )


def _validate_accessors(
    document: Mapping[str, Any], views: Sequence[tuple[int, int, int, int | None]]
) -> list[tuple[int, str, int]]:
    accessors = _array(document, "accessors")
    validated: list[tuple[int, str, int]] = []
    for index, raw_accessor in enumerate(accessors):
        accessor = _object(raw_accessor, f"accessors[{index}]")
        component_type = accessor.get("componentType")
        if component_type not in _COMPONENT_SIZES or isinstance(component_type, bool):
            _fail(f"accessors[{index}].componentType is invalid")
        accessor_type = accessor.get("type")
        if accessor_type not in _TYPE_COMPONENT_COUNTS:
            _fail(f"accessors[{index}].type is invalid")
        count = _integer(accessor.get("count"), f"accessors[{index}].count", minimum=1)
        byte_offset = _integer(
            accessor.get("byteOffset", 0), f"accessors[{index}].byteOffset"
        )
        component_size = _COMPONENT_SIZES[component_type]
        element_size = _accessor_element_size(component_type, accessor_type)

        raw_view_index = accessor.get("bufferView")
        if raw_view_index is not None:
            view_index = _index(
                raw_view_index, len(views), f"accessors[{index}].bufferView"
            )
            _validate_range(
                location=f"accessors[{index}]",
                view_index=view_index,
                byte_offset=byte_offset,
                count=count,
                element_size=element_size,
                component_size=component_size,
                views=views,
                use_stride=True,
            )
        elif "sparse" not in accessor:
            _fail(f"accessors[{index}] has neither bufferView nor sparse storage")
        elif byte_offset != 0:
            _fail(f"accessors[{index}].byteOffset requires a bufferView")

        raw_sparse = accessor.get("sparse")
        if raw_sparse is not None:
            sparse = _object(raw_sparse, f"accessors[{index}].sparse")
            sparse_count = _integer(
                sparse.get("count"), f"accessors[{index}].sparse.count", minimum=1
            )
            if sparse_count > count:
                _fail(f"accessors[{index}].sparse.count exceeds accessor count")
            sparse_indices = _object(
                sparse.get("indices"), f"accessors[{index}].sparse.indices"
            )
            indices_component_type = sparse_indices.get("componentType")
            if (
                indices_component_type not in _INDEX_COMPONENT_TYPES
                or isinstance(indices_component_type, bool)
            ):
                _fail(f"accessors[{index}].sparse.indices.componentType is invalid")
            indices_view = _index(
                sparse_indices.get("bufferView"),
                len(views),
                f"accessors[{index}].sparse.indices.bufferView",
            )
            indices_offset = _integer(
                sparse_indices.get("byteOffset", 0),
                f"accessors[{index}].sparse.indices.byteOffset",
            )
            indices_size = _COMPONENT_SIZES[indices_component_type]
            _validate_range(
                location=f"accessors[{index}].sparse.indices",
                view_index=indices_view,
                byte_offset=indices_offset,
                count=sparse_count,
                element_size=indices_size,
                component_size=indices_size,
                views=views,
                use_stride=False,
            )
            sparse_values = _object(
                sparse.get("values"), f"accessors[{index}].sparse.values"
            )
            values_view = _index(
                sparse_values.get("bufferView"),
                len(views),
                f"accessors[{index}].sparse.values.bufferView",
            )
            values_offset = _integer(
                sparse_values.get("byteOffset", 0),
                f"accessors[{index}].sparse.values.byteOffset",
            )
            _validate_range(
                location=f"accessors[{index}].sparse.values",
                view_index=values_view,
                byte_offset=values_offset,
                count=sparse_count,
                element_size=element_size,
                component_size=component_size,
                views=views,
                use_stride=False,
            )

        validated.append((component_type, accessor_type, count))
    return validated


def _validate_embedded_images(
    document: Mapping[str, Any],
    payloads: Sequence[bytes],
    views: Sequence[tuple[int, int, int, int | None]],
) -> set[int]:
    images = _array(document, "images")
    embedded: set[int] = set()
    for index, raw_image in enumerate(images):
        image = _object(raw_image, f"images[{index}]")
        has_uri = "uri" in image
        has_view = "bufferView" in image
        if has_uri == has_view:
            _fail(f"images[{index}] must define exactly one of uri or bufferView")
        if has_view:
            view_index = _index(
                image.get("bufferView"), len(views), f"images[{index}].bufferView"
            )
            mime_type = image.get("mimeType")
            if mime_type not in ("image/png", "image/jpeg", "image/webp"):
                _fail(f"images[{index}].mimeType is missing or unsupported")
            buffer_index, offset, length, _ = views[view_index]
            image_data = payloads[buffer_index][offset : offset + length]
        else:
            uri = image.get("uri")
            if not isinstance(uri, str):
                _fail(f"images[{index}].uri must be a string")
            mime_type, image_data = _decode_data_uri(uri, f"images[{index}].uri")
            if mime_type not in ("image/png", "image/jpeg", "image/webp"):
                _fail(f"images[{index}].uri has unsupported media type {mime_type!r}")

        if not image_data:
            _fail(f"images[{index}] is empty")
        if mime_type == "image/png" and not image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            _fail(f"images[{index}] does not contain PNG data")
        if mime_type == "image/jpeg" and not image_data.startswith(b"\xff\xd8"):
            _fail(f"images[{index}] does not contain JPEG data")
        if mime_type == "image/webp" and not (
            len(image_data) >= 12
            and image_data.startswith(b"RIFF")
            and image_data[8:12] == b"WEBP"
        ):
            _fail(f"images[{index}] does not contain WebP data")
        embedded.add(index)
    return embedded


def _validate_renderable_content(
    document: Mapping[str, Any],
    accessors: Sequence[tuple[int, str, int]],
    embedded_images: set[int],
) -> tuple[int, int, int]:
    materials = _array(document, "materials")
    pbr_materials = {
        index
        for index, raw_material in enumerate(materials)
        if isinstance(raw_material, dict)
        and isinstance(raw_material.get("pbrMetallicRoughness"), dict)
    }
    if not pbr_materials:
        _fail("GLB has no PBR metallic-roughness material")

    textures = _array(document, "textures")
    embedded_texture_indices: set[int] = set()
    for index, raw_texture in enumerate(textures):
        texture = _object(raw_texture, f"textures[{index}]")
        if "source" in texture:
            source = _index(texture["source"], len(_array(document, "images")), f"textures[{index}].source")
            if source in embedded_images:
                embedded_texture_indices.add(index)
    if not embedded_images:
        _fail("GLB has no embedded image")

    def pbr_uses_embedded_texture(material_index: int) -> bool:
        material = _object(materials[material_index], f"materials[{material_index}]")
        pbr = _object(
            material.get("pbrMetallicRoughness"),
            f"materials[{material_index}].pbrMetallicRoughness",
        )
        found_texture = False
        for field in ("baseColorTexture", "metallicRoughnessTexture"):
            raw_info = pbr.get(field)
            if raw_info is None:
                continue
            info = _object(raw_info, f"materials[{material_index}].pbrMetallicRoughness.{field}")
            texture_index = _index(
                info.get("index"), len(textures), f"materials[{material_index}].{field}.index"
            )
            if texture_index in embedded_texture_indices:
                found_texture = True
        return found_texture

    meshes = _array(document, "meshes")
    primitive_count = 0
    valid_primitive_count = 0
    for mesh_index, raw_mesh in enumerate(meshes):
        mesh = _object(raw_mesh, f"meshes[{mesh_index}]")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            _fail(f"meshes[{mesh_index}].primitives must be an array")
        for primitive_index, raw_primitive in enumerate(primitives):
            primitive_count += 1
            location = f"meshes[{mesh_index}].primitives[{primitive_index}]"
            primitive = _object(raw_primitive, location)
            attributes = _object(primitive.get("attributes"), f"{location}.attributes")
            if "POSITION" not in attributes or "indices" not in primitive:
                continue
            position_index = _index(
                attributes["POSITION"], len(accessors), f"{location}.attributes.POSITION"
            )
            index_accessor_index = _index(
                primitive["indices"], len(accessors), f"{location}.indices"
            )
            position_component, position_type, position_count = accessors[position_index]
            index_component, index_type, index_count = accessors[index_accessor_index]
            if position_component != 5126 or position_type != "VEC3" or position_count == 0:
                continue
            if (
                index_component not in _INDEX_COMPONENT_TYPES
                or index_type != "SCALAR"
                or index_count == 0
            ):
                continue
            if "material" not in primitive:
                continue
            material_index = _index(
                primitive["material"], len(materials), f"{location}.material"
            )
            if material_index not in pbr_materials:
                continue
            if not pbr_uses_embedded_texture(material_index):
                continue
            valid_primitive_count += 1

    if valid_primitive_count == 0:
        _fail(
            "GLB has no non-empty indexed mesh primitive with POSITION and a PBR "
            "material backed by an embedded image"
        )
    return len(meshes), primitive_count, len(pbr_materials)


def verify_glb(
    source: str | Path | bytes | bytearray | memoryview,
) -> GlbSummary:
    """Validate *source* and return a summary, or raise GlbValidationError.

    Paths and in-memory bytes are accepted so the same entry point can be used
    by functional tests and by the standalone command-line interface.
    """

    data = _read_source(source)
    document, binary = _parse_container(data)
    payloads, buffer_lengths = _resolve_buffers(document, binary)
    views = _validate_buffer_views(document, buffer_lengths)
    accessors = _validate_accessors(document, views)
    embedded_images = _validate_embedded_images(document, payloads, views)
    mesh_count, primitive_count, pbr_material_count = _validate_renderable_content(
        document, accessors, embedded_images
    )
    return GlbSummary(
        byte_length=len(data),
        mesh_count=mesh_count,
        primitive_count=primitive_count,
        accessor_count=len(accessors),
        pbr_material_count=pbr_material_count,
        embedded_image_count=len(embedded_images),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate a self-contained, renderable glTF 2.0 GLB"
    )
    parser.add_argument("glb", type=Path, help="GLB file to validate")
    args = parser.parse_args(argv)
    try:
        summary = verify_glb(args.glb)
    except GlbValidationError as error:
        print(f"invalid GLB: {error}", file=sys.stderr)
        return 1
    print(
        f"valid GLB: {args.glb} ({summary.byte_length} bytes, "
        f"{summary.mesh_count} meshes, {summary.primitive_count} primitives, "
        f"{summary.pbr_material_count} PBR materials, "
        f"{summary.embedded_image_count} embedded images)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
