# trellis2.c tools

Download the default TRELLIS.2 and DINOv3 weights from Hugging Face:

```sh
python3 tools/download_weights.py --source huggingface
```

Or download the same layout from ModelScope:

```sh
python3 tools/download_weights.py --source modelscope
```

The downloader writes to `../TRELLIS.2/` by default. The GUI reads the same
folder layout: `TRELLIS.2-4B/`, `dinov3-vitl16-pretrain-lvd1689m/`, and
`BiRefNet/`. Use `--output-dir`, `--only trellis|dino|birefnet`, `--revision`,
`--include`, or `--full` when you need a custom layout or a non-default set of
files.

`trellis-gui` is the default local GUI. It opens without arguments; choose the
weights folder in the window, or pass it up front:

```sh
../build/trellis-gui --weights ../TRELLIS.2
```

`trellis-image-to-gltf` is the default terminal inference app:

```sh
../build/trellis-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image ../assets/example_image/T.png \
  --gltf benchmark_outputs/output.glb
```

It loads an input image, runs DINO conditioning, sparse-structure flow,
structured-latent shape flow, shape decode, texture decode, TRELLIS topology
postprocess, and writes a GLB or glTF in one command. It does not open raylib.
Sparse coords and DINO condition data are passed directly in memory, so no stage
handoff files are written by default. The CLI defaults to the PyTorch app-style
pipeline for the detected family (TRELLIS.2 `512`, Pixal3D `1024_cascade`) and
vkmesh remesh postprocess without simplification. If no output path is passed,
it writes `output.glb`. WebP inputs are converted to a
temporary PNG because the current
stb_image loader does not decode WebP directly.
For opaque input, both model families use an auto-discovered BiRefNet model when
available. Pixal3D requires foreground isolation; TRELLIS.2 remains able to run
without BiRefNet for compatibility.

Pixal3D checkpoints are auto-detected and use the same command. Convert the NAF
release checkpoint once, then pass it with the Pixal3D model directory:

```sh
python3 tools/convert_naf_weights.py \
  /path/to/naf_release.pth \
  ../Pixal3D/Pixal3D/ckpts/naf_release.safetensors

../build/trellis-image-to-gltf \
  --model ../Pixal3D/Pixal3D \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image ../assets/example_image/T.png \
  --gltf benchmark_outputs/pixal3d.glb
```

Pixal3D also supports `1536_cascade`. `--naf` falls back to
`TRELLIS_NAF_PATH` and then `ckpts/naf_release.safetensors`. Use
`--no-model-cache` or `--model-cache-budget-mib N` on memory-constrained GPUs.
The `--fov`, `--camera-distance`, and `--mesh-scale` flags control Pixal3D
projection. If distance is omitted or zero, it is fitted to FOV and mesh scale
as `1 / (2 * mesh_scale * tan(fov / 2))`; a positive distance remains explicit.
This path does not estimate FOV from the image. A foreground-isolated RGBA input
is used directly. For opaque input, BiRefNet is discovered from
`TRELLIS_BIREFNET_PATH`, the model directory, or a `BiRefNet` directory beside
DINO; `--birefnet FILE` remains an explicit override.

`trellis_image_to_gltf.c` is intentionally thin: it parses arguments and calls
`trellis_pipeline_image_to_gltf_ex()` from `src/tasks/image_to_3d/image_to_3d.c`.
The legacy `trellis_pipeline_image_to_gltf()` entry point remains available and
uses the fitted default Pixal3D camera plus automatic NAF path discovery.

`vkmesh` runs the Vulkan compute mesh postprocess path. The TRELLIS preset
fills small holes, remeshes with narrow-band dual contouring by default, and
unwraps UVs by default. Pass an explicit simplify target when you want face
decimation.
The standalone remesher lives under `tools/vkmesh/` and its compute shaders
live under `tools/vkmesh/shaders/`. The image-to-3D glTF exporter and its
texture bake shaders live under `src/tasks/image_to_3d/export/`.

```sh
../build/trellis-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --birefnet ../TRELLIS.2/BiRefNet/model.gguf \
  --image ../assets/example_image/T.png \
  --gltf benchmark_outputs/output_post.glb \
  --backend vulkan \
  --mesh-postprocess \
  --mesh-remesh
```

In the full pipeline, `vkmesh` cleans topology before PBR voxel baking, so the
glTF exporter unwraps and bakes textures on the processed mesh. Use
`--no-mesh-postprocess` for raw/debug exports. In Vulkan
builds, UV-space rasterization and PBR voxel sampling run through the Vulkan
bake pipeline, then seam dilation and empty texel fill run as compute passes.
Use a `.glb` output path to embed geometry and PNG textures in one binary file;
`.gltf` output paths keep writing external `.bin` and `.png` files.
BiRefNet follows the same `--backend` and `--device` settings as the rest of the
image-to-3D pipeline. Use standalone `vkmesh --postprocess --no-uv-unwrap` for
geometry-only meshbin output, `--cleanup` for a single primitive cleanup pass, or
individual flags such as `--fill-holes`, `--repair-non-manifold-edges`, and
`--remove-small-components` when debugging one stage at a time.
Vulkan inference calls the integrated vkmesh C API. CUDA builds place the
standalone `vkmesh` beside `trellis-image-to-gltf`, and the pipeline finds that
sibling automatically; `PATH` and `--vkmesh FILE` are only fallbacks for a
custom layout.
vkmesh keeps its Vulkan buffer workspace bounded. By default it derives a
conservative budget from `VK_EXT_memory_budget` (with a 2048 MiB ceiling), keeps
source geometry/BVH resident once, and streams distance-query points through a
reusable batch buffer. Override the cap with
`--vkmesh-gpu-workspace-budget-mib N` in `trellis-image-to-gltf`,
`--gpu-workspace-budget-mib N` in standalone `vkmesh`, or
`TRELLIS_VKMESH_GPU_WORKSPACE_BUDGET_MIB`. The limit covers vkmesh
`VkDeviceMemory` workspace, not model weights owned by the rest of the process.
UV parameterization and packing use CPU xatlas. For large meshes, the glTF
exporter first builds manifold face adjacency on the CPU, grows connected local
chunks, and adds each chunk to xatlas as its own mesh. Vulkan acceleration starts
at UV-space PBR rasterization and continues through texture dilation and empty
texel fill. `TRELLIS_GLTF_UV_CHART_FACES` controls the target faces per xatlas
input mesh, and `TRELLIS_GLTF_UV_CONE_DEGREES` controls connected chunk growth
from the default 90 degree cone threshold.

`trellis-birefnet-rgba` runs only the BiRefNet background-removal model and
writes an RGBA PNG:

```sh
../build/trellis-birefnet-rgba \
  --birefnet ../TRELLIS.2/BiRefNet/model.gguf \
  --image ../example_image/T.png \
  --out /tmp/T_rgba.png
```

By default it uses the compiled graph backend (`cuda` in CUDA builds,
`vulkan` in Vulkan builds). Use `--backend cuda|vulkan|cpu --device N` to test
a specific graph backend.

Pipeline code lives under `src/`:

- `src/runtime/trellis_runtime.c`: CUDA backend setup, logging/progress, model path/load helpers.
- `src/architectures/dinov3/dinov3.c`: DINOv3 image encoder weight binding and graph definition.
- `src/ops/sparse/cuda/forward.cu`: CUDA forward paths for sparse 3D decoder networks.
- `src/architectures/dit_flow/dit_flow.c`: reusable pure-ggml DiT/SLatFlowModel binding and graph definition.
- `src/architectures/sparse_structure_decoder/decoder.c`: sparse-structure VAE decoder weight binding.
- `src/architectures/sparse_unet_decoder/decoder.c`: reusable SparseUnetVaeDecoder binding for shape and texture decoders.
- `src/ops/sparse/cuda/kernels.cu`: internal CUDA kernels for sparse shape decoding.
- `src/tasks/image_to_3d/stages/sparse_structure.c`: image -> sparse coords + DINO condition.
- `src/tasks/image_to_3d/stages/structured_latent.c`: sparse coords + condition -> shape/texture SLat.
- `src/tasks/image_to_3d/image_to_3d.c`: image -> textured GLB/glTF orchestration.
- `tools/debug/trellis_checkpoint_validate.c`: checkpoint contract validation for debug tools/tests.
- `tools/debug/trellis_sparse_reference.c`: CPU sparse reference ops for tests/debug.

Debug helpers:

- `trellis_tool_cli.h`: terminal logging and diffusion.cpp-style progress output.
- `trellis_tool_model.h`: shared safetensors-to-CUDA tensor-store loading helper.
- `trellis_image_to_gltf.c`: one-shot image-to-GLB/glTF CLI entry point.
- `debug/trellis_infer.c`: legacy/debug sparse-structure image/DINO/flow/voxel decode CLI.

Standalone debug tools:

- `trellis_info.c`: inspect checkpoint manifests.
- `debug/trellis_sparse_structure.c`: sparse-structure checkpoint and schedule validation.
- `debug/trellis_structured_latent.c`: structured-latent tensor sampler/decode debug CLI.
- `debug/trellis_verify.c`: small numeric/file verification helper.
