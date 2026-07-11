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
support. The main command is `trellis-image-to-gltf`.

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
./build/trellis-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --birefnet ../TRELLIS.2/BiRefNet/BiRefNet-F16.gguf \
  --image example_image/T.png \
  --gltf output.glb
```

### Pixal3D

Pixal3D uses the same executable and is detected from the checkpoint tensor
layout. Its projected image conditioning also needs the ValeoAI NAF checkpoint,
converted once from the release `.pth` file to safetensors (this command needs
Python `torch` and `safetensors`):

```sh
python3 tools/convert_naf_weights.py \
  https://github.com/valeoai/NAF/releases/download/model/naf_release.pth \
  ../Pixal3D/Pixal3D/ckpts/naf_release.safetensors
```

Run the 1024 cascade on CUDA or Vulkan with:

```sh
./build/trellis-image-to-gltf \
  --model ../Pixal3D/Pixal3D \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --naf ../Pixal3D/Pixal3D/ckpts/naf_release.safetensors \
  --image example_image/T.png \
  --gltf pixal3d.glb \
  --pipeline 1024_cascade
```

`1536_cascade` is also supported. `--naf` may be omitted when
`ckpts/naf_release.safetensors` exists below the model directory, or when
`TRELLIS_NAF_PATH` is set. If GPU memory is constrained, use
`--no-model-cache` or a finite `--model-cache-budget-mib` value. Pixal3D camera
projection defaults to the reference horizontal FOV `0.857556`, distance `2`,
and mesh scale `1`; override them with `--fov`, `--camera-distance`, and
`--mesh-scale`. These are explicit projection parameters, not automatic camera
estimation. The input must already contain a transparent foreground mask, or
`--birefnet FILE` must be supplied so Pixal3D can apply its subject crop.
TRELLIS.2 checkpoints ignore the NAF and camera options.

C callers that need explicit Pixal3D overrides can initialize
`trellis_pixal3d_options` with `TRELLIS_PIXAL3D_OPTIONS_INIT` and call
`trellis_pipeline_image_to_gltf_ex()`; the legacy entry point keeps the default
camera and automatic NAF lookup.

Pixal3D defaults to BF16-style block rounding and BF16 Flash Attention.
On NVIDIA Ampere or newer GPUs, BF16 K/V select ggml's streaming vector kernel:
Q/K dot products, online softmax state, and V accumulation stay in F32, and KV
tail rows are bounds checked. This avoids the BF16-to-F16 narrowing and F16
accumulator overflow of ggml's current MMA kernel. TRELLIS.2 remains on its
existing F16 MMA Flash path, so its numerical and performance policy is
unchanged. `--no-ggml-flash-attn` explicitly selects SDPA; that path can require
quadratic score memory for long sparse sequences. The package-level policies
are instance scoped, so loading Trellis2 and Pixal3D in one process does not
change either model's attention mode.

Windows:

```powershell
.\build-win\Release\trellis-image-to-gltf.exe `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```

For a Windows Vulkan build, use:

```powershell
.\build-win-vulkan\Release\trellis-image-to-gltf.exe `
  --backend vulkan `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```
