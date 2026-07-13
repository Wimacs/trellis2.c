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
  <img src="img.png" alt="trellis2.c 本地工作区">
</p>

`trellis2.c` 支持 TRELLIS.2、Pixal3D 图像转 3D 和 TokenSkin 网格权重绑定的
原生 CUDA/Vulkan 推理。每个模型使用一个独立可执行文件：
`trellis2-image-to-gltf`、`pixal3d-image-to-gltf` 和 `tokenskin-rig`。

## 编译

带子模块克隆：

```sh
git clone --recursive git@github.com:Wimacs/trellis2.c.git
cd trellis2.c
```

如果克隆时没有加 `--recursive`，再执行：

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
./build/trellis2-image-to-gltf \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --birefnet ../TRELLIS.2/BiRefNet/BiRefNet-F16.gguf \
  --image example_image/T.png \
  --pipeline 1024 \
  --gltf output.glb
```

TRELLIS.2 命令默认使用 512 profile，并支持通过 `--pipeline 1024` 直接进行
1024 分辨率生成；它会在读取图片和初始化 GPU 前拒绝 Pixal3D 模型包。

形状生成可以和材质生成分开执行。下面的命令只生成网格，同时保存实际用于
条件输入的去背 RGBA 图片，以及重拓扑前的 shape latent：

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

`--shape-only` 会跳过 texture flow 和 texture decoder。`.tslat` 是与资产绑定的
缓存，只应和同一个形状或仅修改拓扑的重拓扑版本一起保存和复用。

### 为现有 TRELLIS.2 网格生成材质

`trellis2-texture-mesh` 可以为现有三角网格生成 PBR 材质。没有兼容缓存时，
它会将网格转换成 Flexible Dual Grid，再运行 `FlexiDualGridVaeEncoder`；有缓存
时则直接复用 shape latent。之后根据参考图片生成并烘焙 base color 和
metallic-roughness 纹理，输出自包含 GLB。

```sh
./build/trellis2-texture-mesh \
  --model ../TRELLIS.2/TRELLIS.2-4B \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --input remeshed-shape.glb \
  --image prepared.png --image-prepared \
  --shape-latent shape.tslat \
  --output textured.glb
```

缓存不存在、损坏、分辨率不符或与当前网格几何不兼容时，程序会回退到重新编码
当前网格。可使用 `--shape-latent-output FILE` 保存回退后得到的新缓存。该任务会
重建静态纹理网格，不保留输入的节点、材质、UV、蒙皮、动画或 VRM 扩展。

Pixal3D 默认使用 `1024_cascade`，命令为：

```sh
./build/pixal3d-image-to-gltf \
  --model ../Pixal3D/Pixal3D \
  --dino ../TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m \
  --image example_image/T.png \
  --gltf pixal3d.glb
```

Pixal3D 专用命令提供 `--naf`、`--fov`、`--camera-distance`、
`--mesh-scale` 和 cascade pipeline；TRELLIS.2 的 `--pipeline` 仅接受
`512` 或 `1024`。两个命令共享底层 task、算子、vkmesh 补洞和 remesh 实现。

### TokenSkin 网格权重绑定

TokenSkin 使用独立的 `tokenskin-rig`，不会作为模式混入两个图像转 3D
命令。先将官方 TokenRig Lightning checkpoint 转换为三个推理权重文件；
转换脚本需要 Python `torch` 和 `safetensors`：

```sh
python3 tools/convert_tokenskin_weights.py \
  /path/to/grpo_1400.ckpt \
  models/tokenskin/ckpts
```

CUDA 构建直接运行：

```sh
./build/tokenskin-rig \
  --model models/tokenskin \
  --input input.glb \
  --output rigged-cuda.glb
```

Vulkan 构建使用相同参数，不需要额外环境变量或推理选项：

```sh
./build-vulkan/tokenskin-rig \
  --model models/tokenskin \
  --input input.glb \
  --output rigged-vulkan.glb
```

输入可以是 GLB 或 glTF，输出是包含骨骼、逆绑定矩阵、关节索引和蒙皮权重的
自包含 GLB。当前实现会保留展平到世界坐标的网格几何、拓扑、标准 glTF PBR
材质、UV、采样器以及图片和纹理数据；不会保留源节点结构、morph target、
动画或 VRM 扩展。若源外观无法安全复制，仍会输出有效的绑定模型，并回退为
不透明白色 PBR 材质。

Windows：

```powershell
.\build-win\Release\trellis2-image-to-gltf.exe `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```

Windows Vulkan 构建使用：

```powershell
.\build-win-vulkan\Release\trellis2-image-to-gltf.exe `
  --backend vulkan `
  --model ..\TRELLIS.2\TRELLIS.2-4B `
  --dino ..\TRELLIS.2\dinov3-vitl16-pretrain-lvd1689m `
  --birefnet ..\TRELLIS.2\BiRefNet\BiRefNet-F16.gguf `
  --image example_image\T.png `
  --gltf output.glb
```
