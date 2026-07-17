# vkmesh

`vkmesh` is the Vulkan mesh postprocessor used by `trellis2.c`. It is designed
for large, imperfect triangle meshes produced by 3D generation models and is
available both as an integrated C path and as a standalone `.meshbin` tool.

## Features

- Parallel QEM edge-collapse simplification with edge-length and skinny-triangle
  penalties, link-condition checks, flip guards, and compact GPU output.
- Sparse narrow-band unsigned-distance remeshing followed by simple Dual
  Contouring. Distance queries and large-grid corner/DC/topology work run on
  Vulkan; BVH construction and some sparse-grid orchestration remain on the CPU.
- Hole filling, duplicate/degenerate face removal, non-manifold sheet splitting,
  small-component removal, and consistent face orientation.
- GPU unsigned-distance queries for arbitrary point sets.
- xatlas UV unwrap for standalone `.meshbin` output.
- Reusable Vulkan workspaces with an automatic memory budget capped at 2048 MiB
  by default and an explicit override.

In the TRELLIS.2 image-to-GLB pipeline, the usual order is:

```text
decoded mesh
  -> fill small holes
  -> preserve the projection source used by texture baking
  -> narrow-band remesh
  -> simplify to the requested face target
  -> remove small components
  -> xatlas UV unwrap and Vulkan PBR texture bake
  -> GLB
```

`vkmesh` owns the geometry stages. UV packing and PBR baking for production GLB
export are performed by the image-to-3D exporter.

## Build

Requirements are CMake 3.22+, a C/C++ compiler, and a Vulkan SDK containing the
loader, headers, and `glslc`. The selected device must support Vulkan 1.2, a
compute queue, and 16-bit storage buffers. OpenMP is used when available.

From the repository root:

```sh
cmake -S . -B build-vulkan \
  -DTRELLIS2_C_BACKEND=vulkan \
  -DTRELLIS2_C_GGML_BACKEND=vulkan \
  -DTRELLIS2_C_BUILD_TOOLS=ON \
  -DTRELLIS2_C_BUILD_TESTS=ON

cmake --build build-vulkan --config Release \
  --target vkmesh trellis2_c_vkmesh_tests
```

The executable is normally under `build-vulkan/Release/` on Windows and
`build-vulkan/` with a single-config generator. Use `vkmesh --probe-vulkan-json`
to list and validate available Vulkan devices.

## Usage

The standalone tool reads and writes the repository's `TRLMESH1` `.meshbin`
format: vertex positions, triangle indices, and optional UVs.

Run the TRELLIS preset, remesh at resolution 512, and simplify to 500,000 faces:

```sh
vkmesh \
  --input raw.meshbin \
  --output processed.meshbin \
  --postprocess \
  --target-faces 500000 \
  --remesh-resolution 512 \
  --no-uv-unwrap \
  --device 0
```

`--postprocess` enables remesh and UV unwrap by default. Passing
`--target-faces` enables simplification; `--no-uv-unwrap` keeps the result
geometry-only. For a broad cleanup pass without remeshing:

```sh
vkmesh --input dirty.meshbin --output clean.meshbin --cleanup
```

Individual stages such as `--fill-holes`, `--repair-non-manifold-edges`,
`--remove-small-components`, `--remesh`, `--simplify`, and `--uv-unwrap` can be
selected separately. Run `vkmesh --help` for all thresholds and weights.

Limit vkmesh-owned Vulkan allocations when sharing the GPU with model weights:

```sh
vkmesh ... --gpu-workspace-budget-mib 1024
```

The equivalent image-to-GLB controls are `--mesh-postprocess`, `--mesh-remesh`,
`--mesh-postprocess-simplify`, `--mesh-decimation-target`, and
`--vkmesh-gpu-workspace-budget-mib`.

## Performance

The following are local Release measurements on an RTX 4090 D under Windows
WDDM. Wall time covers the complete standalone operation, including host
orchestration, submissions, output compaction, readback, and `.meshbin` writing,
but excludes model inference. `Workspace` is vkmesh's own tracked peak and does
not include the Vulkan driver or context.

| Workload | Result | Wall time | Workspace |
|---|---:|---:|---:|
| Simplify 5.423M faces to about 1M | 0.984-0.985M faces | 0.72-0.94 s | 613.7 MiB |
| Simplify 1.992M faces to about 1M | 0.957-0.958M faces | 0.38-0.47 s | 224.3 MiB |
| Remesh512 on the 1.992M-face fixture | 6.439M faces | 1.22-1.42 s | 601.6 MiB |
| Fill holes on a 3.979M-face TRELLIS mesh | 3,556 loops / +23,702 faces | 0.54 s | 339.8 MiB |

Simplification is sub-second on both measured fixtures. Remeshing is still the
largest of these isolated geometry stages, and its time, output size, and memory
depend primarily on surface extent, narrow-band occupancy, and
`--remesh-resolution`, rather than only on the input face count. Results vary by
driver, GPU, topology, storage speed, and workspace budget. Parallel atomic
selection can also produce small run-to-run differences in the final face
count.

## Current limitations

- Simple Dual Contouring can produce closed meshes that still contain
  incidence-four non-manifold edges. Use `--repair-non-manifold-edges` only when
  strict sheet splitting is more important than preserving the reconstructed
  surface exactly.
- Simplification is best-effort: topology guards can stop above the requested
  target, while a parallel collapse batch can finish slightly below it. Dirty
  generated meshes should normally be remeshed before they are simplified.
- The pipeline does not provide PaMO-style self-intersection rollback or safe
  projection, so it does not guarantee an intersection-free or strictly
  manifold result.
- Production UV charting and packing use CPU xatlas. Vulkan acceleration starts
  at UV-space PBR rasterization and continues through dilation and empty-texel
  filling.
- The workspace budget covers allocations owned by vkmesh, not inference model
  weights or every driver allocation.

For scalable topology checks:

```sh
python tools/vkmesh/meshbin_validate.py processed.meshbin \
  --fail-on boundary,degenerate,duplicate,indices,nonfinite,unused,winding
```

Use `--strict` when the application requires non-manifold edges to fail as well.

## License

The vkmesh code and shaders authored for this repository are covered by the
project's [MIT License](../../LICENSE). xatlas and the other dependencies keep
their own licenses. Model weights used by the surrounding generation pipeline
are separate from vkmesh and are not covered by the project MIT License; see
[the third-party and model notices](../../THIRD_PARTY_NOTICES.md).
