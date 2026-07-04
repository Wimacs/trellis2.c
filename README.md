# trellis2.c

`trellis2.c` is a native C + ggml + CUDA inference project for TRELLIS.2 image-to-3D generation.

The goal is to run the TRELLIS.2 pipeline without Python or PyTorch:

- encode an input image with DINOv3
- run sparse-structure flow denoising
- decode sparse-structure voxel snapshots for debug/inspection
- run structured-latent shape flow denoising
- decode mesh snapshots for structured-latent debug/inspection

The main inference entry point is `trellis-image-to-obj`, a thin terminal CLI
over the pipeline in `src/`. It reports model loading and sampler step
progress without opening a GUI. Raylib viewers are optional replay/live debug
tools.

## Status

Implemented:

- CUDA-backed ggml runtime setup
- safetensors loading with terminal progress
- DINOv3 image conditioning
- sparse-structure flow and decoder
- structured-latent shape flow
- shape decoder with FlexiDualGrid mesh extraction
- one-shot image-to-OBJ pipeline
- CLI sparse-structure image-to-voxel debug inference
- optional raylib voxel and mesh visualization
- custom CUDA kernels for the parts not covered by ggml

Still in progress:

- Python parity tuning
- performance optimization for sparse convolution and decode
- texture/PBR output
- full end-to-end model quality validation

## Dependencies

- CMake
- CUDA Toolkit
- a CUDA GPU
- Git submodules

`ggml` is included as a submodule from:

```sh
https://github.com/Wimacs/ggml.git
```

`raylib` is included as a submodule for visualization.

Clone with submodules:

```sh
git clone --recursive git@github.com:Wimacs/trellis2.c.git
```

## Build

```sh
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

Place the TRELLIS.2 and DINOv3 checkpoints next to the project, then run one-shot OBJ inference:

```sh
./build/trellis-image-to-obj \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image ../assets/example_image/T.png \
  --obj benchmark_outputs/output.obj
```

`trellis-image-to-obj` runs sparse-structure flow, structured-latent shape flow,
and shape decode in one terminal command. It does not open a GUI, and sparse
coords plus DINO condition data are passed directly in memory. Only the final OBJ
is written by default; WebP inputs are converted to a temporary PNG because the
current stb_image loader does not decode WebP directly.

## Useful Tools

- `trellis-infer`: terminal sparse-structure image-to-voxel debug inference
- `trellis-image-to-obj`: one-shot terminal image-to-OBJ executable
- `trellis-live`: optional image-to-3D live viewer
- `trellis-info`: inspect checkpoint coverage
- `trellis-sparse-structure`: sparse-structure validation helpers
- `trellis-structured-latent`: structured-latent tensor and decoder helpers
- `trellis-voxel-viewer`: replay voxel snapshots
- `trellis-mesh-viewer`: replay mesh snapshots

## Layout

```text
include/       public C API
src/           C runtime implementation
src/model      network definitions, weight binding, and model CUDA forward paths
src/pipeline   image-to-OBJ orchestration and in-memory pipeline steps
src/runtime    CUDA backend setup, logging/progress, model path/load helpers
src/io         safetensors parser and ggml tensor-store loading
src/mesh       FlexiDualGrid mesh extraction
src/model      ggml network definitions plus CUDA sparse-conv model forward paths
src/pipeline   samplers and end-to-end CLI orchestration
tools/         thin CLIs, inspect/debug tools, and optional raylib viewers
tests/         unit tests
docs/          implementation notes
3rd/           ggml and raylib submodules
```

Primary pipeline files:

- `src/runtime/trellis_runtime.c`: CUDA backend setup, logging/progress, model path/load helpers
- `src/model/trellis_dino.c`: DINOv3 image encoder weight binding and graph definition
- `src/model/trellis_cuda_forward.cu`: CUDA forward paths for sparse 3D decoder networks
- `src/model/trellis_slat_flow_model.c`: reusable pure-ggml DiT/SLatFlowModel binding and graph definition
- `src/model/trellis_sparse_structure_decoder.c`: sparse-structure VAE decoder weight binding
- `src/model/trellis_sparse_unet_vae_decoder.c`: reusable SparseUnetVaeDecoder binding for shape and texture decoders
- `src/pipeline/trellis_sparse_structure_pipeline.c`: image -> sparse coords + DINO condition
- `src/pipeline/trellis_structured_latent_pipeline.c`: sparse coords + condition -> denormalized shape/texture SLat
- `src/pipeline/trellis_pipeline.c`: image -> colored OBJ orchestration
- `tools/debug/trellis_checkpoint_validate.c`: checkpoint contract validation for debug tools/tests
- `tools/debug/trellis_sparse_reference.c`: CPU sparse reference ops for tests/debug
