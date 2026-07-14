<p align="center">
  <samp>
&nbsp;&nbsp;_______&nbsp;____&nbsp;&nbsp;_____&nbsp;_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;___&nbsp;____&nbsp;&nbsp;____&nbsp;&nbsp;&nbsp;&nbsp;____<br>
&nbsp;|__&nbsp;&nbsp;&nbsp;__|&nbsp;&nbsp;_&nbsp;\|&nbsp;____|&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;|_&nbsp;_/&nbsp;___||___&nbsp;\&nbsp;&nbsp;/&nbsp;___|<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;|&nbsp;|_)&nbsp;|&nbsp;&nbsp;_|&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|\___&nbsp;\&nbsp;&nbsp;__)&nbsp;||&nbsp;|<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;_&nbsp;&lt;|&nbsp;|___|&nbsp;|___|&nbsp;|___&nbsp;|&nbsp;|&nbsp;___)&nbsp;|/&nbsp;__/&nbsp;|&nbsp;|___<br>
&nbsp;&nbsp;&nbsp;&nbsp;|_|&nbsp;&nbsp;|_|&nbsp;\_\_____|_____|_____|___|____/|_____(_)____|<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;trellis2.c&nbsp;image-to-3D&nbsp;pipeline
  </samp>
</p>

<p align="center">
  <img src="img.png" alt="trellis2.c local workspace">
</p>

Native TRELLIS.2 and Pixal3D image-to-3D inference plus TokenSkin mesh rigging,
with CUDA and Vulkan support. Each model has one family-pinned executable:
`trellis2-image-to-gltf`, `trellis2-texture-mesh`, `trellis2-segment-mesh`,
`pixal3d-image-to-gltf`, or `tokenskin-rig`.

## Build

Clone with submodules:

```sh
git clone --recursive git@github.com:Wimacs/trellis2.c.git
cd trellis2.c
```

If the repository was cloned without `--recursive`, run:

```sh
git submodule update --init --recursive
```

Both inference backends build vkmesh and the Vulkan texture baker by default,
so the Vulkan SDK (headers, loader, and `glslc`) is a build dependency even for
CUDA. This keeps remesh and texture export available to the normal one-command
inference path without installing a separate vkmesh executable.

CUDA:

```sh
cmake -S . -B build -DTRELLIS2_C_BACKEND=cuda
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Vulkan:

```sh
cmake -S . -B build-vulkan -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-vulkan -j
```

Windows CUDA:

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 -DTRELLIS2_C_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-win --config Release
ctest --test-dir build-win -C Release --output-on-failure
```

Windows Vulkan:

Install the Vulkan SDK first.

```powershell
cmake -S . -B build-win-vulkan -G "Visual Studio 17 2022" -A x64 -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-win-vulkan --config Release
ctest --test-dir build-win-vulkan -C Release --output-on-failure
```

## Download Weights

Hugging Face:

```sh
python3 tools/download_weights.py --source huggingface
```

ModelScope:

```sh
python3 tools/download_weights.py --source modelscope
```

This downloads TRELLIS.2, DINOv3, and the BiRefNet background-removal model:

The default Hugging Face download uses the public
`camenduru/dinov3-vitl16-pretrain-lvd1689m` mirror because the official Meta
repository requires accepting an access agreement. The required model files
are identical; use `--dino-repo` to override the source.

```text
../TRELLIS.2/
|-- TRELLIS.2-4B/
|-- dinov3-vitl16-pretrain-lvd1689m/
`-- BiRefNet/
    `-- BiRefNet-F16.gguf
```

## Run

Linux:

```sh
./build/trellis2-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --pipeline 1024 \
  --gltf output.glb
```

The TRELLIS.2 executable defaults to the `512` profile and accepts
`--pipeline 1024` for direct 1024-resolution generation. It rejects Pixal3D
model packages before loading the image or initializing a GPU. Both model CLIs use
vkmesh for hole filling and remeshing, with simplification disabled by default.
For opaque input, both also use an auto-discovered BiRefNet model when available;
Pixal3D requires it, while TRELLIS.2 can still run without it.

Shape generation can be run independently from material generation.  This also
persists the exact foreground condition image and the pre-remesh shape latent so
that a later material task for the same asset can reuse both:

```sh
./build/trellis2-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --shape-only \
  --prepared-image-output prepared.png \
  --shape-latent-output shape.tslat \
  --output shape.glb
```

`--shape-only` skips the texture flow and texture decoder.  The `.tslat` file is
an asset-bound cache: keep it with the generated shape and pass it only for that
shape or a topology-only remesh of it.

### TRELLIS.2 existing-mesh material generation

`trellis2-texture-mesh` applies the released TRELLIS.2 material pipeline to an
existing triangle mesh. It normalizes the mesh, converts it to a Flexible Dual
Grid and runs `FlexiDualGridVaeEncoder` when no compatible shape cache is
provided. It conditions the texture flow on the resulting or cached shape
latent and reference image, then bakes base-color and metallic-roughness
textures into a self-contained GLB.

CUDA and Vulkan use the same command contract; select the executable from the
corresponding build directory:

```sh
./build/trellis2-texture-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input input.glb \
  --image reference.png \
  --output textured-cuda.glb

./build-vulkan/trellis2-texture-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input input.glb \
  --image reference.png \
  --output textured-vulkan.glb
```

To texture a generated or topology-remeshed shape without repeating background
removal or shape encoding, reuse the two artifacts produced above:

```sh
./build/trellis2-texture-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input remeshed-shape.glb \
  --image prepared.png --image-prepared \
  --shape-latent shape.tslat \
  --output textured.glb
```

An unavailable, invalid, resolution-mismatched, or geometrically incompatible
cache is ignored and the current mesh is encoded instead.  Use
`--shape-latent-output FILE` to persist that fallback encoding.

The standard TRELLIS.2 download contains the shape encoder checkpoint. Opaque
images use the same automatic BiRefNet discovery as image-to-3D, so neither a
backend flag nor an extra preprocessing command is required. The tested default
uses a 512 Flexible Dual Grid, 12 texture-flow steps, and 1024-pixel PBR maps.
This task rebuilds a static textured mesh: source nodes, materials, UVs, skins,
animations, and VRM extensions are not preserved.

### SegviGen Full automatic mesh decomposition

`trellis2-segment-mesh` runs the released SegviGen Full network as a separate
`mesh_segmentation` task. Convert the trusted Lightning checkpoint once; the
converter keeps the checkpoint's intentional BF16/F32 split and rejects any
missing, extra, renamed, shape-incompatible, or dtype-incompatible tensor. The
conversion command requires Python `torch` and `safetensors`:

```sh
python3 tools/convert_segvigen_weights.py \
  /path/to/full_seg.ckpt \
  /path/to/SegviGen/segvigen_full_mixed.safetensors
cp models/segvigen/model.json /path/to/SegviGen/model.json
```

The normal command needs only the input mesh and model roots. It renders the
fixed 40-degree SegviGen condition view, samples the source glTF PBR material,
encodes shape and source texture SLat, runs the 2N-token paired flow for 12
steps, transfers decoded colors to faces, welds glTF seams for connectivity,
and writes one GLB node+mesh per physical part:

```sh
./build/trellis2-segment-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --segmentation-model /path/to/SegviGen \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input input.glb \
  --output parts.glb
```

The automatic condition renderer is deterministic and CPU-only: it matches the
released camera and glTF-to-Blender axis transform, samples base-color/alpha
materials per fragment, and resolves a 2x supersampled raster. It does not try
to reproduce Blender Cycles path tracing; pass the released Cycles render with
`--condition-image ... --condition-prepared` when exact 2D conditioning parity
is required. PNG/JPEG glTF textures are supported, while KTX2/BasisU material
textures currently require prior transcoding.

The output is one self-contained assembly GLB with a `trellis_parts` root and
dense `part_0000`, `part_0001`, ... children. Every source face is emitted
exactly once. Boundary vertices are duplicated between parts and each child
receives a distinct categorical PBR material; source hierarchy, skins,
animations, and appearance are not copied to this segmentation-preview asset.
No cut or cap geometry is synthesized, so the separated surface meshes are not
guaranteed to be watertight manufacturing solids.

For reproducible experiments, `--condition-image FILE --condition-prepared`
can replace the automatic render. `--shape-latent`, `--texture-latent`, and
`--segmentation-latent` reuse native `TSLAT01` stages; the corresponding
`--*-latent-output` flags persist them. Trusted official `.pth` fixtures can be
converted with:

```sh
python3 tools/convert_segvigen_fixture.py \
  official_slat.pth native_slat.tslat \
  --glb input.glb
```

These cache flags are diagnostics: ordinary automatic decomposition does not
require them. Cache files do not embed a cryptographic identity of the source
mesh, so only reuse a cache with the GLB from which it was produced. The
`--segmentation-latent` postprocess-only path skips condition encoding and the
paired flow, so `--dino` may be omitted for that path. Base shape/texture
decoders remain required because labels still have to be decoded and projected;
the shape encoder is also required unless `--shape-latent` is supplied at the
same time.
Only the released `512_full` profile is accepted. The interactive and guided
checkpoint contracts are deliberately not inferred from Full weights.

An opt-in real-weight regression runs the complete automatic path and verifies
that the output has multiple independent part meshes and preserves every input
face exactly once. Configure/build normally, set
`TRELLIS_SEGVIGEN_E2E_BASE_MODEL`,
`TRELLIS_SEGVIGEN_E2E_SEGMENTATION_MODEL`, `TRELLIS_SEGVIGEN_E2E_DINO`, and
`TRELLIS_SEGVIGEN_E2E_INPUT`, then run. Optional
`TRELLIS_SEGVIGEN_E2E_OUTPUT`, `TRELLIS_SEGVIGEN_E2E_STEPS`, and
`TRELLIS_SEGVIGEN_E2E_TIMEOUT` preserve the result or override the 12-step and
1800-second defaults:

```sh
ctest --test-dir build -R mesh_segmentation_e2e --output-on-failure
```

### Pixal3D

Pixal3D uses its own `pixal3d-image-to-gltf` executable. Its projected image
conditioning also needs the ValeoAI NAF checkpoint, converted once from the
release `.pth` file to safetensors (this command needs Python `torch` and
`safetensors`):

```sh
python3 tools/convert_naf_weights.py \
  https://github.com/valeoai/NAF/releases/download/model/naf_release.pth \
  ../Pixal3D/Pixal3D/ckpts/naf_release.safetensors
```

Run the 1024 cascade on CUDA or Vulkan with:

```sh
./build/pixal3d-image-to-gltf \
  --model ../Pixal3D/Pixal3D \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --gltf pixal3d.glb
```

`1536_cascade` is also supported. `--naf` may be omitted when
`ckpts/naf_release.safetensors` exists below the model directory, or when
`TRELLIS_NAF_PATH` is set. If GPU memory is constrained, use
`--no-model-cache` or a finite `--model-cache-budget-mib` value. Pixal3D camera
projection defaults to horizontal FOV `0.857556` and mesh scale `1`; when
`--camera-distance` is omitted or zero, distance is fitted as
`1 / (2 * mesh_scale * tan(fov / 2))` (`1.09375` for those defaults). An
explicit positive `--camera-distance` is preserved. This fits distance to the
selected FOV but does not estimate FOV from the image. A transparent foreground
mask is used directly. For opaque input, BiRefNet is discovered from
`TRELLIS_BIREFNET_PATH`, the model directory, or a `BiRefNet` directory beside
DINO; `--birefnet FILE` remains an explicit override.
NAF and camera projection options are exposed only by the Pixal3D executable.

C callers that need Pixal3D overrides can initialize
`trellis_pixal3d_options` with `TRELLIS_PIXAL3D_OPTIONS_INIT` and call
`trellis_pipeline_pixal3d_image_to_gltf()`; non-positive `camera_distance`
selects the same FOV/mesh-scale fit. TRELLIS.2 callers can use
`trellis_pipeline_trellis2_image_to_gltf()`. The generic entry points remain as
library compatibility APIs, but no command-line executable auto-dispatches
between families. To request shape-only output or persisted prepared-image and
latent artifacts, initialize `trellis_image_to_gltf_feature_options` with
`TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_INIT` and use the corresponding
family-specific `_ex` entry point. Keeping these features in a separate,
versioned struct preserves the ABI of the original image-to-glTF options.

### TokenSkin mesh rigging

TokenSkin has a separate `tokenskin-rig` executable and does not add a mode to
either image-to-3D CLI. Convert the official TokenRig Lightning checkpoint once
(this command needs Python `torch` and `safetensors`):

```sh
python3 tools/convert_tokenskin_weights.py \
  /path/to/grpo_1400.ckpt \
  models/tokenskin/ckpts
```

The same three-argument inference command works for a CUDA build:

```sh
./build/tokenskin-rig \
  --model models/tokenskin \
  --input input.glb \
  --output rigged-cuda.glb
```

and a Vulkan build:

```sh
./build-vulkan/tokenskin-rig \
  --model models/tokenskin \
  --input input.glb \
  --output rigged-vulkan.glb
```

No backend-specific environment variable or extra inference flag is required.
The input may be GLB or glTF; the output is a self-contained rigged GLB with a
generated skeleton, inverse bind matrices, joints, and skin weights. The
current path preserves the flattened world-space mesh geometry and topology,
plus standard primitive material assignments, UV sets, materials, textures,
samplers, and embedded image payloads. If source appearance cannot be copied,
rigging still succeeds with a default white PBR material. Source node structure,
morph targets, and animations are not preserved.

Pixal3D defaults to BF16-style block rounding and BF16 Flash Attention.
On CUDA with NVIDIA Ampere or newer GPUs, BF16 K/V select ggml's streaming vector kernel:
Q/K dot products, online softmax state, and V accumulation stay in F32, and KV
tail rows are bounds checked. This avoids the BF16-to-F16 narrowing and F16
accumulator overflow of ggml's current MMA kernel. The base TRELLIS.2 package
keeps the faster F16 MMA path for its ordinary sparse and 512-resolution flows.
Its 1024 shape/texture components and SegviGen Full use strict BF16 Flash
because real long-sequence regression testing exposes the same F16 accumulator
overflow there. `--no-ggml-flash-attn`
explicitly selects SDPA; that path can require
quadratic score memory for long sparse sequences. The package-level policies
are instance scoped, so loading Trellis2 and Pixal3D in one process does not
change either model's attention mode.

On supported NVIDIA Vulkan devices that expose F16 KHR cooperative matrices
with F32 accumulators but do not expose native BF16 cooperative-matrix
operands, strict BF16 attention automatically lowers BF16 K/V to the D128 F16
MMA path. QK and P×V accumulate in F32, and a power-of-two V scale is applied
independently to each 16-channel panel before restoring its F32 contribution.
No model command needs an opt-in environment variable; set
`GGML_VK_BF16_F16_MMA=0` only to disable this lowering for backend diagnostics.
The capability-gated path does not change the ordinary Trellis2 F16 path.

Windows:

```powershell
.\build-win\Release\trellis2-image-to-gltf.exe `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --image example_image\T.png `
  --gltf output.glb
```

For a Windows Vulkan build, use:

```powershell
.\build-win-vulkan\Release\trellis2-image-to-gltf.exe `
  --backend vulkan `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --image example_image\T.png `
  --gltf output.glb
```
