# trellis2.c

Native TRELLIS.2 image-to-3D inference tool with CUDA and Vulkan support. The main command is `trellis-image-to-gltf`.

## Build

Clone submodules first:

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
