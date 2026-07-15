#!/usr/bin/env python3
"""Scalable quality and regression checks for TRLMESH1 ``.meshbin`` files.

The checker memory-maps mesh payloads, scans geometry in chunks, and partitions
face/edge keys into temporary on-disk buckets.  Duplicate and topology counts
are exact; the entire ``3 * face_count`` edge array is never resident in RAM.

Typical use::

    python tools/vkmesh/meshbin_validate.py model.meshbin
    python tools/vkmesh/meshbin_validate.py model.meshbin --components
    python tools/vkmesh/meshbin_validate.py input.meshbin \
        --compare output.meshbin --samples 200000 --json report.json
    python tools/vkmesh/meshbin_validate.py output.meshbin \
        --fail-on nonfinite,indices,degenerate,duplicate,nonmanifold,winding

``--compare`` reports a symmetric sampled-surface distance.  Points are sampled
from triangles in proportion to area, then queried against the other mesh's
surface samples with SciPy cKDTree.  It is a deterministic, bounded-memory
regression metric, not an exact triangle-BVH Hausdorff distance.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
    raise SystemExit("meshbin_validate.py requires NumPy") from exc


HEADER = struct.Struct("<8sQQII")
MAGIC = b"TRLMESH1"
FACE_KEY_DTYPE = np.dtype([("a", "<u4"), ("b", "<u4"), ("c", "<u4")])
U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
FAIL_CHOICES = {
    "nonfinite",
    "indices",
    "degenerate",
    "duplicate",
    "boundary",
    "nonmanifold",
    "winding",
    "unused",
}


class MeshBinError(RuntimeError):
    """Invalid or unsupported meshbin file."""


@dataclass
class MeshBin:
    path: Path
    n_vertices: int
    n_faces: int
    flags: int
    reserved: int
    file_bytes: int
    expected_bytes: int
    vertices: np.memmap
    faces: np.memmap
    uvs: np.memmap | None

    @classmethod
    def open(cls, path_value: str | os.PathLike[str]) -> "MeshBin":
        path = Path(path_value).resolve()
        try:
            file_bytes = path.stat().st_size
            with path.open("rb") as stream:
                raw_header = stream.read(HEADER.size)
        except OSError as exc:
            raise MeshBinError(f"cannot open {path}: {exc}") from exc
        if len(raw_header) != HEADER.size:
            raise MeshBinError(f"{path}: truncated meshbin header")
        magic, n_vertices, n_faces, flags, reserved = HEADER.unpack(raw_header)
        if magic != MAGIC:
            raise MeshBinError(f"{path}: expected magic {MAGIC!r}, got {magic!r}")
        if n_vertices == 0 or n_faces == 0:
            raise MeshBinError(f"{path}: empty meshes are not accepted by vkmesh")
        if n_vertices > 0x7FFFFFFF:
            raise MeshBinError(
                f"{path}: {n_vertices} vertices cannot be addressed by signed int32 faces"
            )
        if flags & ~1:
            raise MeshBinError(f"{path}: unsupported meshbin flags 0x{flags:x}")

        vertex_bytes = int(n_vertices) * 3 * 4
        face_bytes = int(n_faces) * 3 * 4
        uv_bytes = int(n_vertices) * 2 * 4 if flags & 1 else 0
        expected_bytes = HEADER.size + vertex_bytes + face_bytes + uv_bytes
        if file_bytes < expected_bytes:
            raise MeshBinError(
                f"{path}: truncated payload ({file_bytes} bytes, expected {expected_bytes})"
            )
        try:
            vertices = np.memmap(
                path,
                mode="r",
                dtype="<f4",
                offset=HEADER.size,
                shape=(int(n_vertices), 3),
            )
            faces = np.memmap(
                path,
                mode="r",
                dtype="<i4",
                offset=HEADER.size + vertex_bytes,
                shape=(int(n_faces), 3),
            )
            uvs = None
            if flags & 1:
                uvs = np.memmap(
                    path,
                    mode="r",
                    dtype="<f4",
                    offset=HEADER.size + vertex_bytes + face_bytes,
                    shape=(int(n_vertices), 2),
                )
        except (OSError, ValueError) as exc:
            raise MeshBinError(f"{path}: failed to memory-map payload: {exc}") from exc
        return cls(
            path=path,
            n_vertices=int(n_vertices),
            n_faces=int(n_faces),
            flags=int(flags),
            reserved=int(reserved),
            file_bytes=file_bytes,
            expected_bytes=expected_bytes,
            vertices=vertices,
            faces=faces,
            uvs=uvs,
        )

    def close(self) -> None:
        for array_name in ("vertices", "faces", "uvs"):
            array = getattr(self, array_name)
            mmap = getattr(array, "_mmap", None) if array is not None else None
            if mmap is not None:
                mmap.close()


def iter_ranges(count: int, chunk: int) -> Iterable[tuple[int, int]]:
    for start in range(0, count, chunk):
        yield start, min(start + chunk, count)


def human_bytes(value: int | float) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or suffix == "TiB":
            return f"{amount:.1f} {suffix}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def peak_rss_bytes() -> int | None:
    """Return process peak working set on Windows or max RSS on POSIX."""
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            get_memory_info = kernel32.K32GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory_info.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = get_memory_info(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
        except (AttributeError, OSError):
            return None
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError):
        return None


def mix_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer, used only to balance exact disk buckets."""
    x = np.asarray(values, dtype=np.uint64).copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x &= U64_MASK
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x &= U64_MASK
    x ^= x >> np.uint64(31)
    return x


class BucketWriter:
    """Hash partition records so each exact sort has a bounded working set."""

    def __init__(self, directory: Path, prefix: str, bucket_count: int):
        self.directory = directory
        self.prefix = prefix
        self.bucket_count = bucket_count
        self.paths = [directory / f"{prefix}_{index:04x}.bin" for index in range(bucket_count)]
        self.touched: set[int] = set()
        self.bytes_written = 0

    def add(self, records: np.ndarray, buckets: np.ndarray) -> None:
        if records.size == 0:
            return
        order = np.argsort(buckets, kind="stable")
        ordered_buckets = np.asarray(buckets[order])
        ordered_records = np.ascontiguousarray(records[order])
        changes = np.flatnonzero(ordered_buckets[1:] != ordered_buckets[:-1]) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), changes))
        ends = np.concatenate((changes, np.array([records.size], dtype=np.int64)))
        for start, end in zip(starts.tolist(), ends.tolist()):
            bucket = int(ordered_buckets[start])
            with self.paths[bucket].open("ab", buffering=0) as stream:
                ordered_records[start:end].tofile(stream)
            self.touched.add(bucket)
        self.bytes_written += int(records.nbytes)


def reduce_face_buckets(writer: BucketWriter) -> dict[str, int]:
    duplicate_groups = 0
    duplicate_instances = 0
    maximum_multiplicity = 1
    unique_faces = 0
    for bucket in sorted(writer.touched):
        path = writer.paths[bucket]
        records = np.fromfile(path, dtype=FACE_KEY_DTYPE)
        records.sort(order=("a", "b", "c"), kind="quicksort")
        change = np.empty(records.size, dtype=np.bool_)
        change[0] = True
        change[1:] = (
            (records["a"][1:] != records["a"][:-1])
            | (records["b"][1:] != records["b"][:-1])
            | (records["c"][1:] != records["c"][:-1])
        )
        starts = np.flatnonzero(change)
        counts = np.diff(np.append(starts, records.size))
        duplicated = counts > 1
        duplicate_groups += int(np.count_nonzero(duplicated))
        duplicate_instances += int(np.sum(counts[duplicated] - 1, dtype=np.int64))
        maximum_multiplicity = max(maximum_multiplicity, int(counts.max(initial=1)))
        unique_faces += int(starts.size)
    return {
        "unique_canonical_faces": unique_faces,
        "duplicate_face_groups": duplicate_groups,
        "duplicate_face_instances": duplicate_instances,
        "maximum_face_multiplicity": maximum_multiplicity,
    }


def reduce_edge_buckets(writer: BucketWriter) -> dict[str, int]:
    unique_edges = 0
    boundary_edges = 0
    manifold_edges = 0
    nonmanifold_edges = 0
    winding_conflict_edges = 0
    nonmanifold_direction_imbalanced_edges = 0
    maximum_incidence = 0
    incidence_3 = 0
    incidence_4 = 0
    incidence_5_plus = 0
    for bucket in sorted(writer.touched):
        path = writer.paths[bucket]
        packed = np.fromfile(path, dtype="<u8")
        packed.sort(kind="quicksort")
        edge_ids = packed >> np.uint64(1)
        change = np.empty(packed.size, dtype=np.bool_)
        change[0] = True
        change[1:] = edge_ids[1:] != edge_ids[:-1]
        starts = np.flatnonzero(change)
        counts = np.diff(np.append(starts, packed.size))
        directions = np.add.reduceat(packed & np.uint64(1), starts).astype(np.int64)
        unique_edges += int(starts.size)
        boundary_edges += int(np.count_nonzero(counts == 1))
        manifold_edges += int(np.count_nonzero(counts == 2))
        nonmanifold_edges += int(np.count_nonzero(counts > 2))
        winding_conflict_edges += int(
            np.count_nonzero((counts == 2) & ((directions == 0) | (directions == 2)))
        )
        nonmanifold = counts > 2
        # More than one traversal of either direction is unavoidable for an edge
        # with incidence > 2; this metric highlights directional imbalance.
        nonmanifold_direction_imbalanced_edges += int(
            np.count_nonzero(nonmanifold & (np.abs(2 * directions - counts) > 1))
        )
        incidence_3 += int(np.count_nonzero(counts == 3))
        incidence_4 += int(np.count_nonzero(counts == 4))
        incidence_5_plus += int(np.count_nonzero(counts >= 5))
        maximum_incidence = max(maximum_incidence, int(counts.max(initial=0)))
    return {
        "unique_edges": unique_edges,
        "boundary_edges": boundary_edges,
        "manifold_incidence_2_edges": manifold_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "winding_conflict_edges": winding_conflict_edges,
        "nonmanifold_direction_imbalanced_edges": nonmanifold_direction_imbalanced_edges,
        "incidence_3_edges": incidence_3,
        "incidence_4_edges": incidence_4,
        "incidence_5_plus_edges": incidence_5_plus,
        "maximum_edge_incidence": maximum_incidence,
    }


def scan_vertices(mesh: MeshBin, vertex_chunk: int) -> dict[str, Any]:
    finite_vertices = 0
    nonfinite_values = 0
    bbox_min = np.full(3, np.inf, dtype=np.float64)
    bbox_max = np.full(3, -np.inf, dtype=np.float64)
    for start, end in iter_ranges(mesh.n_vertices, vertex_chunk):
        vertices = np.asarray(mesh.vertices[start:end])
        finite_values = np.isfinite(vertices)
        finite_rows = np.all(finite_values, axis=1)
        finite_vertices += int(np.count_nonzero(finite_rows))
        nonfinite_values += int(vertices.size - np.count_nonzero(finite_values))
        if np.any(finite_rows):
            valid = vertices[finite_rows].astype(np.float64, copy=False)
            bbox_min = np.minimum(bbox_min, np.min(valid, axis=0))
            bbox_max = np.maximum(bbox_max, np.max(valid, axis=0))
    if finite_vertices == 0:
        bbox_min[:] = np.nan
        bbox_max[:] = np.nan
        diagonal = math.nan
    else:
        diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    result: dict[str, Any] = {
        "finite_vertices": finite_vertices,
        "nonfinite_vertices": mesh.n_vertices - finite_vertices,
        "nonfinite_coordinate_values": nonfinite_values,
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_diagonal": diagonal,
    }
    if mesh.uvs is not None:
        finite_uv_vertices = 0
        nonfinite_uv_values = 0
        for start, end in iter_ranges(mesh.n_vertices, vertex_chunk):
            uvs = np.asarray(mesh.uvs[start:end])
            finite_values = np.isfinite(uvs)
            finite_uv_vertices += int(np.count_nonzero(np.all(finite_values, axis=1)))
            nonfinite_uv_values += int(uvs.size - np.count_nonzero(finite_values))
        result.update(
            {
                "finite_uv_vertices": finite_uv_vertices,
                "nonfinite_uv_vertices": mesh.n_vertices - finite_uv_vertices,
                "nonfinite_uv_values": nonfinite_uv_values,
            }
        )
    return result


def triangle_double_areas(
    mesh: MeshBin, valid_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return double areas and a finite-position mask for valid-index faces."""
    p0 = np.asarray(mesh.vertices[valid_faces[:, 0]], dtype=np.float64)
    p1 = np.asarray(mesh.vertices[valid_faces[:, 1]], dtype=np.float64)
    p2 = np.asarray(mesh.vertices[valid_faces[:, 2]], dtype=np.float64)
    finite = np.all(np.isfinite(p0), axis=1)
    finite &= np.all(np.isfinite(p1), axis=1)
    finite &= np.all(np.isfinite(p2), axis=1)
    areas = np.full(valid_faces.shape[0], np.nan, dtype=np.float64)
    if np.any(finite):
        edge1 = p1[finite] - p0[finite]
        edge2 = p2[finite] - p0[finite]
        cross_x = edge1[:, 1] * edge2[:, 2] - edge1[:, 2] * edge2[:, 1]
        cross_y = edge1[:, 2] * edge2[:, 0] - edge1[:, 0] * edge2[:, 2]
        cross_z = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        areas[finite] = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    return areas, finite


def write_face_and_edge_keys(
    valid_faces: np.ndarray,
    index_degenerate: np.ndarray,
    face_writer: BucketWriter,
    edge_writer: BucketWriter,
) -> int:
    canonical = np.sort(valid_faces.astype("<u4", copy=False), axis=1)
    records = canonical.view(FACE_KEY_DTYPE).reshape(-1)
    face_hash = canonical[:, 0].astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
    face_hash ^= canonical[:, 1].astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    face_hash ^= canonical[:, 2].astype(np.uint64) * np.uint64(0x165667B19E3779F9)
    face_buckets = mix_u64(face_hash) & np.uint64(face_writer.bucket_count - 1)
    face_writer.add(records, face_buckets.astype(np.uint32, copy=False))

    topology_faces = valid_faces[~index_degenerate].astype(np.uint32, copy=False)
    count = topology_faces.shape[0]
    if count == 0:
        return 0
    source = np.empty(count * 3, dtype=np.uint32)
    target = np.empty(count * 3, dtype=np.uint32)
    source[:count], target[:count] = topology_faces[:, 0], topology_faces[:, 1]
    source[count : 2 * count], target[count : 2 * count] = (
        topology_faces[:, 1],
        topology_faces[:, 2],
    )
    source[2 * count :], target[2 * count :] = topology_faces[:, 2], topology_faces[:, 0]
    low = np.minimum(source, target).astype(np.uint64)
    high = np.maximum(source, target).astype(np.uint64)
    edge_id = (low << np.uint64(31)) | high
    packed = ((edge_id << np.uint64(1)) | (source > target).astype(np.uint64)).astype(
        "<u8", copy=False
    )
    edge_buckets = mix_u64(edge_id) & np.uint64(edge_writer.bucket_count - 1)
    edge_writer.add(packed, edge_buckets.astype(np.uint32, copy=False))
    return int(packed.size)


def compute_components(
    mesh: MeshBin,
    used_vertices: np.ndarray,
    face_chunk: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Count vertex-connected components with vectorized hooking and compression."""
    parent = np.arange(mesh.n_vertices, dtype=np.int32)
    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        hooked = False
        for start, end in iter_ranges(mesh.n_faces, face_chunk):
            faces = np.asarray(mesh.faces[start:end])
            valid = np.all((faces >= 0) & (faces < mesh.n_vertices), axis=1)
            faces = faces[valid]
            for left, right in ((0, 1), (1, 2), (2, 0)):
                if faces.size == 0:
                    continue
                roots_left = parent[faces[:, left]]
                roots_right = parent[faces[:, right]]
                high = np.maximum(roots_left, roots_right)
                low = np.minimum(roots_left, roots_right)
                if np.any(low < parent[high]):
                    hooked = True
                np.minimum.at(parent, high, low)
        while True:
            grandparents = parent[parent]
            if np.array_equal(grandparents, parent):
                break
            parent[:] = grandparents
        if not hooked:
            converged = True
            break
    roots = used_vertices & (parent == np.arange(mesh.n_vertices, dtype=np.int32))
    return {
        "definition": "vertex-connected through valid faces",
        "count": int(np.count_nonzero(roots)),
        "iterations": iterations,
        "converged": converged,
        "max_iterations": max_iterations,
        "parent_bytes": int(parent.nbytes),
    }


def analyze_mesh(mesh: MeshBin, args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    vertex_metrics = scan_vertices(mesh, args.vertex_chunk)
    timings["vertex_scan"] = time.perf_counter() - stage_start
    bbox_diagonal = float(vertex_metrics["bbox_diagonal"])
    area_tolerance = (
        args.relative_area_epsilon * bbox_diagonal * bbox_diagonal
        if math.isfinite(bbox_diagonal)
        else math.nan
    )

    used_vertices = np.zeros(mesh.n_vertices, dtype=np.bool_)
    invalid_faces = 0
    invalid_index_values = 0
    valid_faces_total = 0
    index_degenerate_faces = 0
    finite_geometry_faces = 0
    nonfinite_geometry_faces = 0
    zero_area_faces = 0
    geometric_degenerate_faces = 0
    geometric_degenerate_distinct_index_faces = 0
    edge_records = 0
    surface_area = 0.0
    area_compensation = 0.0

    temp_parent = Path(args.temp_dir).resolve() if args.temp_dir else None
    with tempfile.TemporaryDirectory(prefix="meshbin_validate_", dir=temp_parent) as temp_name:
        temp_path = Path(temp_name)
        face_writer = BucketWriter(temp_path, "face", args.buckets)
        edge_writer = BucketWriter(temp_path, "edge", args.buckets)

        stage_start = time.perf_counter()
        for start, end in iter_ranges(mesh.n_faces, args.face_chunk):
            faces = np.asarray(mesh.faces[start:end])
            index_valid_values = (faces >= 0) & (faces < mesh.n_vertices)
            valid = np.all(index_valid_values, axis=1)
            invalid_faces += int(faces.shape[0] - np.count_nonzero(valid))
            invalid_index_values += int(faces.size - np.count_nonzero(index_valid_values))
            valid_faces = faces[valid]
            valid_faces_total += int(valid_faces.shape[0])
            if valid_faces.size == 0:
                continue
            used_vertices[valid_faces.reshape(-1)] = True
            index_degenerate = (
                (valid_faces[:, 0] == valid_faces[:, 1])
                | (valid_faces[:, 1] == valid_faces[:, 2])
                | (valid_faces[:, 2] == valid_faces[:, 0])
            )
            index_degenerate_faces += int(np.count_nonzero(index_degenerate))
            edge_records += write_face_and_edge_keys(
                valid_faces,
                index_degenerate,
                face_writer,
                edge_writer,
            )

            double_areas, finite_geometry = triangle_double_areas(mesh, valid_faces)
            finite_geometry_faces += int(np.count_nonzero(finite_geometry))
            nonfinite_geometry_faces += int(finite_geometry.size - np.count_nonzero(finite_geometry))
            finite_areas = double_areas[finite_geometry]
            if finite_areas.size:
                zero_area_faces += int(np.count_nonzero(finite_areas == 0.0))
                below = finite_areas <= area_tolerance
                geometric_degenerate_faces += int(np.count_nonzero(below))
                geometric_degenerate_distinct_index_faces += int(
                    np.count_nonzero(below & ~index_degenerate[finite_geometry])
                )
                # Kahan accumulation over chunk sums keeps area stable across chunk sizes.
                chunk_area = float(np.sum(finite_areas, dtype=np.float64)) * 0.5
                corrected = chunk_area - area_compensation
                updated = surface_area + corrected
                area_compensation = (updated - surface_area) - corrected
                surface_area = updated
        timings["face_scan_and_partition"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        duplicate_metrics = reduce_face_buckets(face_writer)
        topology_metrics = reduce_edge_buckets(edge_writer)
        timings["exact_bucket_reduction"] = time.perf_counter() - stage_start
        temp_bytes = face_writer.bytes_written + edge_writer.bytes_written

        component_metrics = None
        if args.components:
            stage_start = time.perf_counter()
            component_metrics = compute_components(
                mesh,
                used_vertices,
                args.face_chunk,
                args.component_max_iterations,
            )
            timings["components"] = time.perf_counter() - stage_start

    used_count = int(np.count_nonzero(used_vertices))
    topology_face_count = valid_faces_total - index_degenerate_faces
    euler_characteristic = (
        used_count - int(topology_metrics["unique_edges"]) + topology_face_count
    )
    quality = {
        "valid_index_faces": valid_faces_total,
        "invalid_index_faces": invalid_faces,
        "invalid_index_values": invalid_index_values,
        "index_degenerate_faces": index_degenerate_faces,
        "finite_geometry_faces": finite_geometry_faces,
        "nonfinite_geometry_faces": nonfinite_geometry_faces,
        "zero_area_faces": zero_area_faces,
        "geometric_degenerate_faces": geometric_degenerate_faces,
        "geometric_degenerate_distinct_index_faces": geometric_degenerate_distinct_index_faces,
        "relative_area_epsilon": float(args.relative_area_epsilon),
        "double_area_tolerance": float(area_tolerance),
        "used_vertices": used_count,
        "unused_vertices": mesh.n_vertices - used_count,
    }
    topology_metrics.update(
        {
            "faces_in_topology": topology_face_count,
            "directed_edge_records": edge_records,
            "ignored_invalid_or_index_degenerate_faces": invalid_faces
            + index_degenerate_faces,
            "euler_characteristic_using_used_vertices": euler_characteristic,
            "closed_oriented_edge_manifold": bool(
                topology_metrics["boundary_edges"] == 0
                and topology_metrics["nonmanifold_edges"] == 0
                and topology_metrics["winding_conflict_edges"] == 0
            ),
        }
    )
    timings["total"] = time.perf_counter() - total_start
    report: dict[str, Any] = {
        "schema_version": 1,
        "path": str(mesh.path),
        "format": {
            "magic": MAGIC.decode("ascii"),
            "flags": mesh.flags,
            "has_uvs": bool(mesh.flags & 1),
            "reserved": mesh.reserved,
            "file_bytes": mesh.file_bytes,
            "expected_bytes": mesh.expected_bytes,
            "trailing_bytes": mesh.file_bytes - mesh.expected_bytes,
        },
        "counts": {"vertices": mesh.n_vertices, "faces": mesh.n_faces},
        "vertices": vertex_metrics,
        "geometry": {
            "surface_area": surface_area,
            "surface_area_is_finite": math.isfinite(surface_area),
        },
        "quality": quality,
        "duplicates": duplicate_metrics,
        "topology": topology_metrics,
        "components": component_metrics,
        "performance": {
            "timings_seconds": timings,
            "temporary_bytes_written": temp_bytes,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "face_chunk": args.face_chunk,
            "vertex_chunk": args.vertex_chunk,
            "buckets": args.buckets,
        },
    }
    return report


def surface_sample_points(
    mesh: MeshBin,
    total_area: float,
    sample_count: int,
    seed: int,
    face_chunk: int,
) -> np.ndarray:
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise MeshBinError(f"{mesh.path}: cannot sample a mesh with area {total_area}")
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.random(sample_count, dtype=np.float64) * total_area)
    points = np.empty((sample_count, 3), dtype=np.float64)
    next_target = 0
    accumulated = 0.0
    for start, end in iter_ranges(mesh.n_faces, face_chunk):
        faces = np.asarray(mesh.faces[start:end])
        valid = np.all((faces >= 0) & (faces < mesh.n_vertices), axis=1)
        areas = np.zeros(faces.shape[0], dtype=np.float64)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            valid_faces = faces[valid]
            double_areas, finite = triangle_double_areas(mesh, valid_faces)
            good_indices = valid_indices[finite]
            areas[good_indices] = double_areas[finite] * 0.5
        cumulative = np.cumsum(areas, dtype=np.float64)
        chunk_area = float(cumulative[-1]) if cumulative.size else 0.0
        upper = accumulated + chunk_area
        target_end = int(np.searchsorted(targets, upper, side="left"))
        if target_end > next_target:
            local_targets = targets[next_target:target_end] - accumulated
            local_faces = np.searchsorted(cumulative, local_targets, side="right")
            selected_faces = faces[local_faces]
            p0 = np.asarray(mesh.vertices[selected_faces[:, 0]], dtype=np.float64)
            p1 = np.asarray(mesh.vertices[selected_faces[:, 1]], dtype=np.float64)
            p2 = np.asarray(mesh.vertices[selected_faces[:, 2]], dtype=np.float64)
            root = np.sqrt(rng.random(target_end - next_target, dtype=np.float64))
            second = rng.random(target_end - next_target, dtype=np.float64)
            w0 = 1.0 - root
            w1 = root * (1.0 - second)
            w2 = root * second
            points[next_target:target_end] = (
                w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2
            )
            next_target = target_end
        accumulated = upper
        if next_target == sample_count:
            break
    if next_target != sample_count:
        # The analysis and sampling passes use the same formula, but their chunk
        # summation order can differ by a few ulps.  Retry missing tail targets at
        # the final positive triangle rather than silently returning garbage.
        raise MeshBinError(
            f"{mesh.path}: sampled {next_target}/{sample_count} points; "
            "surface-area accumulation was inconsistent"
        )
    return points


def distance_statistics(distances: np.ndarray, scale: float) -> dict[str, float]:
    squared_mean = float(np.mean(distances * distances, dtype=np.float64))
    result = {
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances, dtype=np.float64)),
        "rms": math.sqrt(squared_mean),
        "p50": float(np.percentile(distances, 50)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }
    if math.isfinite(scale) and scale > 0.0:
        result.update({f"relative_{key}": value / scale for key, value in list(result.items())})
    return result


def batched_nearest_distances(
    tree: Any,
    points: np.ndarray,
    batch: int,
    workers: int,
) -> np.ndarray:
    distances = np.empty(points.shape[0], dtype=np.float64)
    for start, end in iter_ranges(points.shape[0], batch):
        queried, _ = tree.query(points[start:end], k=1, workers=workers)
        distances[start:end] = queried
    return distances


def compare_sampled_surfaces(
    mesh_a: MeshBin,
    report_a: dict[str, Any],
    mesh_b: MeshBin,
    report_b: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise MeshBinError("--compare requires SciPy (scipy.spatial.cKDTree)") from exc
    timings: dict[str, float] = {}
    started = time.perf_counter()
    points_a = surface_sample_points(
        mesh_a,
        float(report_a["geometry"]["surface_area"]),
        args.samples,
        args.seed,
        args.face_chunk,
    )
    timings["sample_input"] = time.perf_counter() - started
    started = time.perf_counter()
    points_b = surface_sample_points(
        mesh_b,
        float(report_b["geometry"]["surface_area"]),
        args.samples,
        args.seed,
        args.face_chunk,
    )
    timings["sample_compare"] = time.perf_counter() - started

    started = time.perf_counter()
    tree_b = cKDTree(points_b, compact_nodes=True, balanced_tree=True)
    distances_a_to_b = batched_nearest_distances(
        tree_b, points_a, args.distance_batch, args.workers
    )
    timings["query_input_to_compare"] = time.perf_counter() - started
    del tree_b

    started = time.perf_counter()
    tree_a = cKDTree(points_a, compact_nodes=True, balanced_tree=True)
    distances_b_to_a = batched_nearest_distances(
        tree_a, points_b, args.distance_batch, args.workers
    )
    timings["query_compare_to_input"] = time.perf_counter() - started
    scale = float(report_a["vertices"]["bbox_diagonal"])
    forward = distance_statistics(distances_a_to_b, scale)
    reverse = distance_statistics(distances_b_to_a, scale)
    symmetric = {
        "mean": 0.5 * (forward["mean"] + reverse["mean"]),
        "rms": math.sqrt(0.5 * (forward["rms"] ** 2 + reverse["rms"] ** 2)),
        "p99_max_direction": max(forward["p99"], reverse["p99"]),
        "sampled_hausdorff": max(forward["max"], reverse["max"]),
    }
    if math.isfinite(scale) and scale > 0.0:
        symmetric.update(
            {f"relative_{key}": value / scale for key, value in list(symmetric.items())}
        )
    timings["total"] = sum(timings.values())
    return {
        "method": "area_weighted_surface_samples_to_sample_cKDTree",
        "note": "sampled approximation; not exact point-to-triangle Hausdorff distance",
        "samples_per_mesh": args.samples,
        "seed": args.seed,
        "normalization_scale_input_bbox_diagonal": scale,
        "input_to_compare": forward,
        "compare_to_input": reverse,
        "symmetric": symmetric,
        "timings_seconds": timings,
        "peak_process_rss_bytes": peak_rss_bytes(),
    }


def print_report(report: dict[str, Any], label: str) -> None:
    counts = report["counts"]
    vertices = report["vertices"]
    geometry = report["geometry"]
    quality = report["quality"]
    duplicates = report["duplicates"]
    topology = report["topology"]
    performance = report["performance"]
    print(f"[{label}] {report['path']}")
    print(
        f"  mesh: V={counts['vertices']:,} F={counts['faces']:,} "
        f"file={human_bytes(report['format']['file_bytes'])} UV={report['format']['has_uvs']}"
    )
    print(
        "  geometry: "
        f"bbox={vertices['bbox_min']}..{vertices['bbox_max']} "
        f"diag={vertices['bbox_diagonal']:.9g} area={geometry['surface_area']:.9g}"
    )
    print(
        "  validity: "
        f"nonfinite_vertices={vertices['nonfinite_vertices']:,} "
        f"invalid_faces={quality['invalid_index_faces']:,} "
        f"index_degenerate={quality['index_degenerate_faces']:,} "
        f"geometric_degenerate={quality['geometric_degenerate_faces']:,} "
        f"unused_vertices={quality['unused_vertices']:,}"
    )
    print(
        "  duplicates: "
        f"groups={duplicates['duplicate_face_groups']:,} "
        f"extra_instances={duplicates['duplicate_face_instances']:,} "
        f"max_multiplicity={duplicates['maximum_face_multiplicity']:,}"
    )
    print(
        "  edges: "
        f"unique={topology['unique_edges']:,} boundary={topology['boundary_edges']:,} "
        f"nonmanifold={topology['nonmanifold_edges']:,} "
        f"winding_conflict={topology['winding_conflict_edges']:,} "
        f"max_incidence={topology['maximum_edge_incidence']:,} "
        f"closed_oriented={topology['closed_oriented_edge_manifold']}"
    )
    if report["components"] is not None:
        components = report["components"]
        print(
            "  components: "
            f"count={components['count']:,} iterations={components['iterations']} "
            f"converged={components['converged']}"
        )
    peak = performance["peak_process_rss_bytes"]
    peak_text = human_bytes(peak) if peak is not None else "unavailable"
    print(
        f"  performance: total={performance['timings_seconds']['total']:.3f}s "
        f"peak_RSS={peak_text} temp_IO={human_bytes(performance['temporary_bytes_written'])}"
    )
    print(
        "    stages: "
        + ", ".join(
            f"{name}={seconds:.3f}s"
            for name, seconds in performance["timings_seconds"].items()
            if name != "total"
        )
    )


def print_comparison(comparison: dict[str, Any]) -> None:
    forward = comparison["input_to_compare"]
    reverse = comparison["compare_to_input"]
    symmetric = comparison["symmetric"]
    print("[surface comparison]")
    print(
        f"  method: {comparison['method']}, samples={comparison['samples_per_mesh']:,}; "
        f"{comparison['note']}"
    )
    print(
        "  input -> compare: "
        f"mean={forward['mean']:.9g} rms={forward['rms']:.9g} "
        f"p99={forward['p99']:.9g} max={forward['max']:.9g}"
    )
    print(
        "  compare -> input: "
        f"mean={reverse['mean']:.9g} rms={reverse['rms']:.9g} "
        f"p99={reverse['p99']:.9g} max={reverse['max']:.9g}"
    )
    print(
        "  symmetric: "
        f"mean={symmetric['mean']:.9g} rms={symmetric['rms']:.9g} "
        f"p99_max={symmetric['p99_max_direction']:.9g} "
        f"sampled_hausdorff={symmetric['sampled_hausdorff']:.9g}"
    )


def parse_fail_on(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = result - FAIL_CHOICES
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown --fail-on values: {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(sorted(FAIL_CHOICES))}"
        )
    return result


def report_failures(report: dict[str, Any], fail_on: set[str], label: str) -> list[str]:
    vertices = report["vertices"]
    quality = report["quality"]
    duplicates = report["duplicates"]
    topology = report["topology"]
    checks = {
        "nonfinite": int(vertices["nonfinite_vertices"])
        + int(vertices.get("nonfinite_uv_vertices", 0)),
        "indices": int(quality["invalid_index_faces"]),
        "degenerate": int(quality["geometric_degenerate_faces"]),
        "duplicate": int(duplicates["duplicate_face_instances"]),
        "boundary": int(topology["boundary_edges"]),
        "nonmanifold": int(topology["nonmanifold_edges"]),
        "winding": int(topology["winding_conflict_edges"]),
        "unused": int(quality["unused_vertices"]),
    }
    return [f"{label}:{name}={checks[name]}" for name in sorted(fail_on) if checks[name] > 0]


def json_safe(value: Any) -> Any:
    """Convert non-finite diagnostic floats to JSON null without hiding them in text output."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def power_of_two(text: str) -> int:
    value = positive_int(text)
    if value & (value - 1):
        raise argparse.ArgumentTypeError("bucket count must be a power of two")
    if value > 65536:
        raise argparse.ArgumentTypeError("bucket count must not exceed 65536")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Low-memory exact topology/quality checks and optional sampled-surface "
            "regression metrics for TRLMESH1 meshbin files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="input TRLMESH1 .meshbin")
    parser.add_argument("--compare", metavar="OUTPUT", help="also validate and compare this mesh")
    parser.add_argument(
        "--components",
        action="store_true",
        help=(
            "compute vertex-connected components (extra passes; 4 bytes/vertex persistent "
            "parent plus a transient compression copy)"
        ),
    )
    parser.add_argument(
        "--component-max-iterations",
        type=positive_int,
        default=64,
        help="maximum vectorized union/compression iterations",
    )
    parser.add_argument(
        "--face-chunk", type=positive_int, default=500_000, help="faces processed per chunk"
    )
    parser.add_argument(
        "--vertex-chunk",
        type=positive_int,
        default=1_000_000,
        help="vertices processed per chunk",
    )
    parser.add_argument(
        "--buckets",
        type=power_of_two,
        default=64,
        help="exact-sort temporary partitions; increase for extremely large meshes",
    )
    parser.add_argument(
        "--temp-dir", help="parent directory for temporary exact-sort bucket files"
    )
    parser.add_argument(
        "--relative-area-epsilon",
        type=float,
        default=1e-12,
        help="degenerate threshold: double triangle area <= epsilon * bbox_diagonal^2",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=100_000,
        help="area-weighted surface samples per mesh for --compare",
    )
    parser.add_argument("--seed", type=int, default=1337, help="surface sampling seed")
    parser.add_argument(
        "--distance-batch",
        type=positive_int,
        default=65_536,
        help="KD-tree query points per batch",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="SciPy cKDTree query workers (-1 uses all cores)",
    )
    parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        metavar="CHECKS",
        help=(
            "comma-separated nonzero metrics that cause exit 1: "
            + ",".join(sorted(FAIL_CHOICES))
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="equivalent to --fail-on all available quality checks, including boundary/unused",
    )
    parser.add_argument("--max-distance-rms", type=float, help="fail above symmetric sampled RMS")
    parser.add_argument(
        "--max-distance-p99", type=float, help="fail above worst directional sampled p99"
    )
    parser.add_argument(
        "--max-distance-max", type=float, help="fail above sampled Hausdorff maximum"
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write complete JSON report; use '-' for JSON-only stdout",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress human-readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.relative_area_epsilon) or args.relative_area_epsilon < 0.0:
        parser.error("--relative-area-epsilon must be finite and non-negative")
    try:
        fail_on = parse_fail_on(args.fail_on)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.strict:
        fail_on = set(FAIL_CHOICES)
    for option_name in ("max_distance_rms", "max_distance_p99", "max_distance_max"):
        value = getattr(args, option_name)
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"--{option_name.replace('_', '-')} must be finite and non-negative")
    if args.compare is None and any(
        getattr(args, name) is not None
        for name in ("max_distance_rms", "max_distance_p99", "max_distance_max")
    ):
        parser.error("distance thresholds require --compare")

    meshes: list[MeshBin] = []
    try:
        input_mesh = MeshBin.open(args.input)
        meshes.append(input_mesh)
        input_report = analyze_mesh(input_mesh, args)
        compare_mesh = None
        compare_report = None
        comparison = None
        if args.compare:
            compare_mesh = MeshBin.open(args.compare)
            meshes.append(compare_mesh)
            compare_report = analyze_mesh(compare_mesh, args)
            gc.collect()
            comparison = compare_sampled_surfaces(
                input_mesh,
                input_report,
                compare_mesh,
                compare_report,
                args,
            )
    except (MeshBinError, OSError, ValueError, MemoryError) as exc:
        print(f"meshbin_validate: error: {exc}", file=sys.stderr)
        return 2
    finally:
        for mesh in meshes:
            mesh.close()

    failures = report_failures(input_report, fail_on, "input")
    if compare_report is not None:
        failures.extend(report_failures(compare_report, fail_on, "compare"))
    if comparison is not None:
        symmetric = comparison["symmetric"]
        thresholds = (
            ("rms", args.max_distance_rms),
            ("p99_max_direction", args.max_distance_p99),
            ("sampled_hausdorff", args.max_distance_max),
        )
        for metric, limit in thresholds:
            if limit is not None and float(symmetric[metric]) > limit:
                failures.append(f"distance:{metric}={symmetric[metric]:.9g}>{limit:.9g}")

    output = {
        "schema_version": 1,
        "input": input_report,
        "compare": compare_report,
        "surface_comparison": comparison,
        "fail_on": sorted(fail_on),
        "failures": failures,
        "passed": not failures,
    }
    json_stdout = args.json == "-"
    if not args.quiet and not json_stdout:
        print_report(input_report, "input")
        if compare_report is not None:
            print_report(compare_report, "compare")
        if comparison is not None:
            print_comparison(comparison)
        if failures:
            print("[FAILED] " + "; ".join(failures))
        elif fail_on or comparison is not None:
            print("[PASSED]")
    serialized = json.dumps(json_safe(output), indent=2, sort_keys=True, allow_nan=False)
    if json_stdout:
        print(serialized)
    elif args.json:
        try:
            Path(args.json).write_text(serialized + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"meshbin_validate: cannot write JSON report: {exc}", file=sys.stderr)
            return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
