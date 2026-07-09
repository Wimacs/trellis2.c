# trellis2.c

`trellis2.c` 是 TRELLIS.2 图像转 3D 的原生推理工具，支持 CUDA 和 Vulkan。主要命令是 `trellis-image-to-gltf`。

## 编译

先拉取子模块：

```sh
git submodule update --init --recursive
```

CUDA：

```sh
cmake -S . -B build -DTRELLIS2_C_BACKEND=cuda
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Vulkan：

```sh
cmake -S . -B build-vulkan -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-vulkan -j
```

Windows CUDA：

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 -DTRELLIS2_C_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-win --config Release
ctest --test-dir build-win -C Release --output-on-failure
```

Windows Vulkan：

先安装 Vulkan SDK。

```powershell
cmake -S . -B build-win-vulkan -G "Visual Studio 17 2022" -A x64 -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-win-vulkan --config Release
ctest --test-dir build-win-vulkan -C Release --output-on-failure
```

## 下载权重

Hugging Face：

```sh
python3 tools/download_weights.py --source huggingface
```

ModelScope：

```sh
python3 tools/download_weights.py --source modelscope
```

会下载 TRELLIS.2、DINOv3 和 BiRefNet 背景去除模型：

```text
../TRELLIS.2/
|-- TRELLIS.2-4B/
|-- dinov3-vitl16-pretrain-lvd1689m/
`-- BiRefNet/
    `-- BiRefNet-F16.gguf
```

## 使用

Linux：

```sh
./build/trellis-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --birefnet ../TRELLIS.2/BiRefNet/BiRefNet-F16.gguf \
  --image example_image/T.png \
  --gltf output.glb
```

Windows：

```powershell
.\build-win\Release\trellis-image-to-gltf.exe `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```

Windows Vulkan 构建使用：

```powershell
.\build-win-vulkan\Release\trellis-image-to-gltf.exe `
  --backend vulkan `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```
