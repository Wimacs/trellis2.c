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

Native TRELLIS.2 and Pixal3D image-to-3D inference tool with CUDA and Vulkan
support. Each model has a family-pinned command: `trellis2-image-to-gltf` or
`pixal3d-image-to-gltf`.

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
  --gltf output.glb
```

The TRELLIS.2 executable is fixed to the 512 profile. It rejects Pixal3D model
packages before loading the image or initializing a GPU. Both model CLIs use
vkmesh for hole filling and remeshing, with simplification disabled by default.
For opaque input, both also use an auto-discovered BiRefNet model when available;
Pixal3D requires it, while TRELLIS.2 can still run without it.

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
between families.

Pixal3D defaults to BF16-style block rounding and BF16 Flash Attention.
On CUDA with NVIDIA Ampere or newer GPUs, BF16 K/V select ggml's streaming vector kernel:
Q/K dot products, online softmax state, and V accumulation stay in F32, and KV
tail rows are bounds checked. This avoids the BF16-to-F16 narrowing and F16
accumulator overflow of ggml's current MMA kernel. TRELLIS.2 keeps the faster
F16 MMA path for its sparse and 512-resolution flows, while its 1024 shape and
texture components use strict BF16 Flash because real long-sequence regression
testing exposes the same F16 accumulator overflow there. `--no-ggml-flash-attn`
explicitly selects SDPA; that path can require
quadratic score memory for long sparse sequences. The package-level policies
are instance scoped, so loading Trellis2 and Pixal3D in one process does not
change either model's attention mode.

On Vulkan devices that expose F16 KHR cooperative matrices with F32
accumulators but do not expose native BF16 cooperative-matrix operands, the
strict BF16 mode continues to use the range-safe scalar-F32 streaming kernel by
default. `GGML_VK_BF16_F16_MMA=1` enables an NVIDIA-only D128 experiment that
stages BF16 K/V as F16, uses F32 accumulation for both QK and P×V, and applies
a power-of-two V scale independently to each 16-channel panel of every KV tile
before restoring its F32 contribution. This avoids F16 V-staging overflow and
does not change the ordinary Trellis2 F16 path. It is intentionally opt-in:
Q/K still have F16 operand range, softmax probabilities are staged as F16, and
very wide value ranges inside one V panel can underflow its smaller values.
Consequently, this is not equivalent to native BF16 Tensor Core operands for
arbitrary inputs and should be revalidated before enabling it for another model
family.

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
