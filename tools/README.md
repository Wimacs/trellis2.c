# trellis2.c tools

Download the default TRELLIS.2 and DINOv3 weights from Hugging Face:

```sh
python3 tools/download_weights.py --source huggingface
```

Or download the same layout from ModelScope:

```sh
python3 tools/download_weights.py --source modelscope
```

The downloader writes to `../TRELLIS.2/` by default. Use `--output-dir`,
`--only trellis|dino`, `--revision`, `--include`, or `--full` when you need a
custom layout or a non-default set of files.

`trellis-image-to-obj` is the default terminal inference app:

```sh
../build/trellis-image-to-obj \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image ../assets/example_image/T.png \
  --obj benchmark_outputs/output.obj
```

It loads an input image, runs DINO conditioning, sparse-structure flow,
structured-latent shape flow, shape decode, and writes an OBJ in one command.
It does not open raylib. Sparse coords and DINO condition data are passed
directly in memory, so no stage handoff files are written by default.
WebP inputs are converted to a temporary PNG because the current
stb_image loader does not decode WebP directly.

`trellis-image-to-obj.c` is intentionally thin: it parses arguments and calls
`trellis_pipeline_image_to_obj()` from `src/pipeline/trellis_pipeline.c`.

`vkmesh` runs the Vulkan compute mesh postprocess path. The TRELLIS preset
mirrors the standard PyTorch/o-voxel export cleanup order: fill small holes,
simplify toward `3 * target`, remove duplicate/non-manifold/small-component
artifacts, fill again, simplify to the final target, repeat cleanup, fill once
more, unify winding, then unwrap UVs by default.
The implementation lives under `tools/vkmesh/`; compute shaders, including the
glTF texture bake/dilate/fill shaders, live under `tools/vkmesh/shaders/`.

```sh
../build/trellis-image-to-obj \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --birefnet ../TRELLIS.2/BiRefNet/model.gguf \
  --image ../assets/example_image/T.png \
  --gltf benchmark_outputs/output_post.gltf \
  --backend vulkan \
  --mesh-postprocess \
  --mesh-postprocess-no-simplify \
  --mesh-decimation-target 1000000 \
  --vkmesh ../build/vkmesh
```

In the full pipeline, `vkmesh` cleans topology before PBR voxel baking, so the
glTF exporter unwraps and bakes textures on the processed mesh. In Vulkan
builds, UV-space rasterization and PBR voxel sampling run through the Vulkan
bake pipeline, then seam dilation and empty texel fill run as compute passes.
BiRefNet follows the same `--backend` and `--device` settings as the rest of the
image-to-3D pipeline. Use standalone `vkmesh --postprocess --no-uv-unwrap` for
geometry-only OBJ output, `--cleanup` for a single primitive cleanup pass, or
individual flags such as `--fill-holes`, `--repair-non-manifold-edges`, and
`--remove-small-components` when debugging one stage at a time.
CuMesh-style UV unwrap is hybrid rather than fully GPU-resident: chart
clustering is the GPU-accelerated part, then each chart is copied to CPU xatlas
for parameterization and packing. The vkmesh exporter now follows that shape:
large meshes first run a Vulkan chart-key prepass and then add the resulting
smaller chart meshes to xatlas. `TRELLIS_GLTF_UV_CHART_FACES` controls the
target faces per chart, and `TRELLIS_GLTF_UV_CHART_GRID` can override the
spatial grid used by the current prepass.

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
- `src/model/trellis_dino.c`: DINOv3 image encoder weight binding and graph definition.
- `src/model/trellis_cuda_forward.cu`: CUDA forward paths for sparse 3D decoder networks.
- `src/model/trellis_slat_flow_model.c`: reusable pure-ggml DiT/SLatFlowModel binding and graph definition.
- `src/model/trellis_sparse_structure_decoder.c`: sparse-structure VAE decoder weight binding.
- `src/model/trellis_sparse_unet_vae_decoder.c`: reusable SparseUnetVaeDecoder binding for shape and texture decoders.
- `src/model/trellis_cuda_kernels.cu`: internal CUDA kernels for sparse shape decoding.
- `src/pipeline/trellis_sparse_structure_pipeline.c`: image -> sparse coords + DINO condition.
- `src/pipeline/trellis_structured_latent_pipeline.c`: sparse coords + condition -> shape/texture SLat.
- `src/pipeline/trellis_pipeline.c`: image -> colored OBJ orchestration.
- `tools/debug/trellis_checkpoint_validate.c`: checkpoint contract validation for debug tools/tests.
- `tools/debug/trellis_sparse_reference.c`: CPU sparse reference ops for tests/debug.

Debug/live helpers:

- `trellis_tool_cli.h`: terminal logging and diffusion.cpp-style progress output.
- `trellis_tool_model.h`: shared safetensors-to-CUDA tensor-store loading helper.
- `trellis_tool_live.h`: in-memory sparse-structure and structured-latent live interfaces.
- `trellis_image_to_obj.c`: one-shot image-to-OBJ CLI entry point.
- `debug/trellis_infer.c`: legacy/debug sparse-structure image/DINO/flow/voxel decode CLI.
- `viewers/trellis_structured_latent_mesh_live_viewer.c`: structured-latent shape flow, per-step mesh decode,
  and raylib mesh upload. It also builds as the standalone
  `trellis-structured-latent-mesh-live-viewer` debug CLI.

Raylib live viewer:

```sh
../build/trellis-live --display :1
```

It is now an optional visualization path rather than the default inference
entry point.

Standalone debug tools:

- `trellis_info.c`: inspect checkpoint manifests.
- `debug/trellis_sparse_structure.c`: sparse-structure checkpoint and schedule validation.
- `debug/trellis_structured_latent.c`: structured-latent tensor sampler/decode debug CLI.
- `viewers/trellis_voxel_viewer.c`: replay written voxel snapshots.
- `viewers/trellis_mesh_viewer.c`: replay written mesh snapshots.
- `debug/trellis_verify.c`: small numeric/file verification helper.
