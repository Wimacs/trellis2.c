<p align="center">
  <samp>
&nbsp;&nbsp;_______&nbsp;____&nbsp;&nbsp;_____&nbsp;_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;___&nbsp;____&nbsp;&nbsp;____&nbsp;&nbsp;&nbsp;&nbsp;____<br>
&nbsp;|__&nbsp;&nbsp;&nbsp;__|&nbsp;&nbsp;_&nbsp;\|&nbsp;____|&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;|_&nbsp;_/&nbsp;___||___&nbsp;\&nbsp;&nbsp;/&nbsp;___|<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;|&nbsp;|_)&nbsp;|&nbsp;&nbsp;_|&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|\___&nbsp;\&nbsp;&nbsp;__)&nbsp;||&nbsp;|<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;_&nbsp;&lt;|&nbsp;|___|&nbsp;|___|&nbsp;|___&nbsp;|&nbsp;|&nbsp;___)&nbsp;|/&nbsp;__/&nbsp;|&nbsp;|___<br>
&nbsp;&nbsp;&nbsp;&nbsp;|_|&nbsp;&nbsp;|_|&nbsp;\_\_____|_____|_____|___|____/|_____(_)____|<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;trellis2.c&nbsp;3D&nbsp;Local&nbsp;Gen&nbsp;All-in-One
  </samp>
</p>

<p align="center">
  <img src="teaser.png" alt="trellis2.c pipeline 工作区">
</p>

`trellis2.c` 是一套全本地运行的一体化 3D 生成工具，集成 TRELLIS.2、
Pixal3D、SegViGen 和 TokenSkin，包含图生 3D、网格贴图、全自动拆件和自动绑骨，
推理完全运行在原生 CUDA 或 Vulkan 上，不依赖 Python 或 PyTorch runtime。

## 编译

依赖：

- Git 和 CMake 3.22 或更高版本。
- C/C++ 工具链：Linux 使用 GCC/Clang，Windows 使用 Visual Studio 2022。
- Vulkan SDK，包括 loader、headers 和 `glslc`。
- 编译 CUDA 后端时需要 CUDA Toolkit。
- Linux 编译随附的 raylib GUI 时需要 OpenGL 和 X11 开发包。

两种推理后端都需要 Vulkan SDK，因为 pipeline 工具还会编译 Vulkan 网格后处理器和纹理烘焙器。

连同所有子模块一起克隆：

```sh
git clone --recursive https://github.com/Wimacs/trellis2.c.git
cd trellis2.c
```

如果克隆时没有下载子模块：

```sh
git submodule update --init --recursive
```

### Linux

CUDA：

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRELLIS2_C_BACKEND=cuda
cmake --build build-cuda -j
```

Vulkan：

```sh
cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-vulkan -j
```

### Windows

在 Visual Studio Developer Shell 中运行。

CUDA：

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 `
  -DTRELLIS2_C_BACKEND=cuda
cmake --build build-win --config Release --parallel
```

Vulkan：

```powershell
cmake -S . -B build-win-vulkan -G "Visual Studio 17 2022" -A x64 `
  -DTRELLIS2_C_BACKEND=vulkan
cmake --build build-win-vulkan --config Release --parallel
```

CUDA 默认编译 compute capability 8.9。其他代际 GPU 请设置
`-DCMAKE_CUDA_ARCHITECTURES=<SM>`。只编译 CLI 时可加入
`-DTRELLIS2_C_BUILD_RAYLIB_VIEWER=OFF`；不需要测试程序时可加入
`-DTRELLIS2_C_BUILD_TESTS=OFF`。

下面的示例使用 `./build-cuda/`。Vulkan 构建改用 `./build-vulkan/`；
Windows CUDA 和 Vulkan 分别改用 `./build-win/Release/` 和
`./build-win-vulkan/Release/`。后端已经编译进可执行文件，不需要额外的后端参数。

## Pipelines

示例假设对应的模型包已经下载或转换完成。TRELLIS.2、DINOv3 和 BiRefNet
可以使用 `tools/download_weights.py` 创建的目录；NAF、SegViGen 和 TokenSkin
的转换脚本位于 `tools/`。

### `trellis2-image-to-gltf`

使用 TRELLIS.2 从单张图片生成带纹理 3D 资产。完整流程包括前景处理、DINOv3
图像编码、形状生成与解码、拓扑清理、PBR 纹理生成以及 GLB/glTF 导出。
默认 profile 为 `512`，`--pipeline 1024` 使用 1024 profile；
`--shape-only` 可以跳过纹理生成。

```sh
./build-cuda/trellis2-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --pipeline 1024 \
  --output output.glb
```

不透明输入会在可用时使用自动发现的 BiRefNet；透明 RGBA 输入直接使用。

### `pixal3d-image-to-gltf`

使用 Pixal3D 的 NAF 和投影图像条件生成带纹理 3D 资产。默认使用
`1024_cascade`，可通过 `--pipeline 1536_cascade` 使用更大的 cascade。
NAF 会从 `MODEL/ckpts/naf_release.safetensors` 自动发现，也可通过 `--naf` 指定。

```sh
./build-cuda/pixal3d-image-to-gltf \
  --model ../Pixal3D/Pixal3D \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --output pixal3d.glb
```

不透明输入还需要可自动发现的 BiRefNet，或显式传入 `--birefnet FILE`；
透明 RGBA 输入不需要。

### `trellis2-texture-mesh`

根据参考图为已有三角网格生成新的 PBR 材质。pipeline 会把网格编码为
TRELLIS.2 shape latent，运行纹理 flow 和 decoder，再写出新的静态 GLB。
由于资产会被重建，原节点层级、材质、UV、skin、animation 和 VRM 扩展不会保留。

```sh
./build-cuda/trellis2-texture-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input input.glb \
  --image reference.png \
  --output textured.glb
```

不透明参考图需要 BiRefNet；已完成前景处理的图片可配合 `--image-prepared` 使用。

### `trellis2-segment-mesh`

使用 SegViGen Full 全自动拆分静态 GLB/glTF。没有指定条件图时，pipeline
会从固定视角自动渲染条件图，预测零件标签、拆分连通组件，并输出一个 assembly
GLB，其中每个物理零件都是可单独选择的 node 和 mesh。默认输出保证每个源面恰好
出现一次，并保留可映射的标准顶点属性、材质和纹理。

```sh
./build-cuda/trellis2-segment-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --segmentation-model ../SegviGen \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input input.glb \
  --output parts.glb
```

`--small-part-mode keep|merge|discard` 控制断开的小壳；`discard` 会主动删除其面。
pipeline 不生成切面或封口，因此拆出的零件不保证水密。无法正确保留 skin、
animation、morph、instancing 或扩展的输入会被拒绝。

### `tokenskin-rig`

使用 TokenSkin 为已有 GLB/glTF 自动绑骨。输出 GLB 包含生成的骨架、joint、
inverse bind matrix 和 skin weight。展平后的世界空间几何和标准 PBR 外观会保留；
原节点层级、morph target 和已有 animation 不会保留。

```sh
./build-cuda/tokenskin-rig \
  --model models/tokenskin \
  --input input.glb \
  --output rigged.glb
```
