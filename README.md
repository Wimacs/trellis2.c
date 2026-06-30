# trellis2.c

`trellis2.c` is a CUDA-only C/ggml port-in-progress for TRELLIS.2 inference.

The primary app is now the native C+ggml+CUDA live pipeline:

```sh
trellis2.c/build/trellis-live --display :1
```

With no extra arguments it opens a raylib window, reads
`assets/example_image/T.png`, runs stage1 image-to-sparse-structure denoising,
decodes and visualizes voxels after every stage1 step, then feeds the final
coords/condition into stage2 shape denoising and decodes/visualizes a mesh after
every stage2 step. It defaults to `TRELLIS.2-4B`,
`dinov3-vitl16-pretrain-lvd1689m`, and 12 Euler steps.

`trellis-live` preloads all safetensors into resident CUDA tensor stores before
creating the raylib window, so the X11 window does not appear frozen while large
checkpoint files are being read.

For an SSH smoke test that still exercises the live per-step decode paths but
uses smaller debug settings:

```sh
trellis2.c/build/trellis-live \
  --display :1 \
  --image assets/example_image/T.png \
  --cond-resolution 64 \
  --steps 12 \
  --flow-blocks 0 \
  --decode-max-input-tokens 32 \
  --max-faces 20000 \
  --hold 0.02
```

The rest of the tree builds the pieces that app uses:

- ggml CUDA backend bootstrap and graph execution helpers
- safetensors metadata loader with F32/F16/BF16/I32/I64/U8/BOOL support
- dense token graph builders used by the TRELLIS.2 DiT/flow models:
  - linear
  - LayerNorm
  - RMSNorm and multi-head RMSNorm
  - GELU(tanh) feed-forward block
  - timestep embedding MLP
  - FlashAttention-based self/cross attention
  - modulated cross-attention transformer block skeleton
  - TRELLIS 3D RoPE phase reference generation
  - standalone CUDA RoPE apply kernel for `[head_dim, tokens, heads, batch]`
  - Euler flow and CFG host-side sampler math
  - reference-compatible CFG rescale and timestep schedules
- standalone CUDA kernels for the dense sparse-structure decoder:
  - NCDHW Conv3D with stride/pad/dilation/bias
  - 3D pixel shuffle
- SparseStructureFlowModel and SparseStructureDecoder checkpoint contract checks
- host reference implementations for TRELLIS sparse coordinate transforms:
  - sparse downsample mean
  - spatial-to-channel
  - channel-to-spatial
- native C+ggml+CUDA stage1 image condition, sparse-structure flow, and dense
  Conv3D decoder path used by `trellis-live`
- native C+ggml+CUDA stage2 shape-SLat tensor sampler and FlexiDualGrid shape
  decoder used by `trellis-live`
- native C/raylib voxel and mesh snapshot viewers built from `3rd/raylib`
- native C/raylib live stage2 mesh viewer that denoises, decodes, and uploads
  every step to GPU indexed buffers
- unit tests that compare ggml CUDA outputs with host reference math

The current implementation is intentionally explicit about what remains:

| TRELLIS.2 component | Status |
| --- | --- |
| SparseStructureFlowModel dense DiT core | Wired into the native stage1 live path with ggml CUDA graph execution, CFG, schedule, RoPE phases, and debug block controls |
| SLatFlowModel sparse DiT core | Wired into the native stage2 live path with sparse coordinate helpers plus dense token operators |
| SparseStructureDecoder dense Conv3D VAE | Wired into the stage1 live path; each denoise step decodes to sparse voxel coords |
| Shape SLat flow | Checkpoint contract and C+ggml+CUDA 512/1024 tensor sampler implemented |
| Shape/Tex sparse VAE decoders | Shape FlexiDualGrid decoder implemented with CUDA neighbor maps, rulebooks, ggml matmul sparse conv, and C debug dumps; texture pending |
| DINOv3 image encoder | Native image preprocessing, patch embedding, 2D RoPE, and transformer forward are wired into stage1; Python parity is still being tightened |
| O-Voxel mesh/PBR postprocess | Pending |

Build:

```sh
cmake -S trellis2.c -B trellis2.c/build -DGGML_CUDA=ON
cmake --build trellis2.c/build -j
ctest --test-dir trellis2.c/build --output-on-failure
```

Inspect a checkpoint manifest:

```sh
trellis2.c/build/trellis-info TRELLIS.2-4B
```

Validate the stage1 checkpoint contract and print the reference sampler
schedule:

```sh
trellis2.c/build/trellis-stage1 --model TRELLIS.2-4B --validate
trellis2.c/build/trellis-stage1 --timesteps 12 --rescale-t 5.0
```

Validate the stage2 shape-SLat checkpoint contract and print the shape sampler
schedule:

```sh
trellis2.c/build/trellis-stage2 --model TRELLIS.2-4B --validate --resolution 512
trellis2.c/build/trellis-stage2 --model TRELLIS.2-4B --validate --resolution 1024
trellis2.c/build/trellis-stage2 --timesteps 12 --rescale-t 3.0
```

The following binaries are lower-level debug tools. They remain useful for
checkpoint validation, dumps, and replay, but `trellis-live` is the normal
end-to-end viewer.

Run the stage1 image/voxel debug entry point and write snapshot files:

```sh
trellis2.c/build/trellis-infer \
  --model TRELLIS.2-4B \
  --dino dinov3-vitl16-pretrain-lvd1689m \
  --stage1-image-voxels \
  --image /path/to/input.png \
  --out benchmark_outputs/stage1_snapshots \
  --steps 12
```

Replay compatible voxel snapshots with the native C/raylib viewer built from
`trellis2.c/3rd/raylib`:

```sh
trellis2.c/build/trellis-voxel-viewer \
  --snapshot-dir benchmark_outputs/stage1_snapshots \
  --source x_t \
  --display :1
```

Run C+ggml+CUDA shape-SLat denoising from tensor inputs:

```sh
trellis2.c/build/trellis-stage2 \
  --model TRELLIS.2-4B \
  --shape-flow-512 \
  --coords-i32 benchmark_outputs/stage2_shape_smoke/shape_512_coords.i32 \
  --input-cond-f32 benchmark_outputs/stage2_shape_smoke/cond_512.f32 \
  --input-neg-cond-f32 benchmark_outputs/stage2_shape_smoke/neg_cond_512.f32 \
  --input-noise-f32 benchmark_outputs/stage2_shape_smoke/shape_512_noise.f32 \
  --out benchmark_outputs/c_shape_flow \
  --steps 12
```

Decode a denormalized shape SLat tensor with the C+CUDA FlexiDualGrid decoder:

```sh
trellis2.c/build/trellis-stage2 \
  --model TRELLIS.2-4B \
  --shape-decode \
  --coords-i32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.coords.i32 \
  --input-slat-f32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.feats.f32 \
  --out benchmark_outputs/c_shape_decode
```

Replay exported mesh frames with the native C/raylib viewer:

```sh
trellis2.c/build/trellis-mesh-viewer \
  --snapshot-dir benchmark_outputs/stage2_shape_live_viewer \
  --source all \
  --display :1 \
  --max-faces 0 \
  --mesh-upload-mode gpu_indexed \
  --mesh-style solid
```

Run native C/raylib live shape denoising and decode each step to mesh:

```sh
trellis2.c/build/trellis-stage2-mesh-live-viewer \
  --model TRELLIS.2-4B \
  --coords-i32 benchmark_outputs/stage2_shape_smoke/shape_512_coords.i32 \
  --input-cond-f32 benchmark_outputs/stage2_shape_smoke/cond_512.f32 \
  --input-neg-cond-f32 benchmark_outputs/stage2_shape_smoke/neg_cond_512.f32 \
  --input-noise-f32 benchmark_outputs/stage2_shape_smoke/shape_512_noise.f32 \
  --display :1 \
  --steps 12 \
  --source pred_x0 \
  --max-faces 0 \
  --mesh-upload-mode gpu_indexed \
  --mesh-style solid
```

The default live mesh renderer uses a single GPU indexed VBO/EBO and a GLSL
shader for flat face lighting. Legacy raylib chunk upload is still available
with `--mesh-upload-mode expanded` or `--mesh-upload-mode indexed`, but it is
much slower for full 512 meshes. For a quick end-to-end viewer smoke that still
runs the C denoise/decode/mesh path every step, add
`--flow-blocks 0 --decode-max-input-tokens 32 --max-faces 20000 --no-final`.
Those debug limits intentionally show only a partial mesh preview.

The snapshot formats are intentionally simple (`manifest.json`, F32 tensor
dumps, compressed int32 voxel coords, raw mesh sidecars, ASCII PLY fallback,
`frames.tsv`, and `mesh_frames.tsv`) so the C binaries can write and read the
same files as CUDA graph coverage expands. On SSH sessions without `DISPLAY`,
the viewers can use `--display :1`; the native viewers also auto-detect
`/tmp/.X11-unix/X*` when no display is set.
