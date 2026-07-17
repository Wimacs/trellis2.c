# Third-party notices

Last reviewed: 2026-07-17.

The MIT License in [`LICENSE`](LICENSE) covers the original software and
documentation authored for trellis2.c. It does not relicense third-party code,
model weights, datasets, input files, sample assets, or generated outputs.
Those materials remain subject to their own terms. The license text shipped by
each upstream source or model revision is authoritative.

This document is an engineering inventory, not legal advice.

## Source dependencies

| Component | Location | License | License material |
|---|---|---|---|
| cgltf | `3rd/cgltf` | MIT | [`3rd/cgltf/LICENSE`](3rd/cgltf/LICENSE) |
| Eigen | `3rd/eigen` | Primarily MPL-2.0, with file-specific compatible licenses | [`COPYING.README`](3rd/eigen/COPYING.README), [`COPYING.MPL2`](3rd/eigen/COPYING.MPL2), and the other `COPYING.*` files |
| ggml | `3rd/ggml` | MIT | [`3rd/ggml/LICENSE`](3rd/ggml/LICENSE) |
| meshoptimizer | `3rd/meshoptimizer` | MIT | [`3rd/meshoptimizer/LICENSE.md`](3rd/meshoptimizer/LICENSE.md) |
| raylib | `3rd/raylib` | zlib/libpng | [`3rd/raylib/LICENSE`](3rd/raylib/LICENSE); bundled GLFW and other external sources retain the notices in `3rd/raylib/src/external` |
| stb | `3rd/stb` | MIT or public domain; trellis2.c uses the MIT option for notices | [`3rd/stb/LICENSE`](3rd/stb/LICENSE) |
| xatlas | `3rd/xatlas` | MIT, with embedded MIT/BSD components | [`3rd/xatlas/LICENSE`](3rd/xatlas/LICENSE) and [`licenses/xatlas-embedded-LICENSES.txt`](licenses/xatlas-embedded-LICENSES.txt) |
| O-Voxel flexible-dual-grid port | `src/ops/mesh/mesh_to_flexible_dual_grid.cpp` | MIT | The source header and [`licenses/O-Voxel-LICENSE.txt`](licenses/O-Voxel-LICENSE.txt) |

Uses meshoptimizer. Copyright (c) 2016-2026, Arseny Kapoulkine.

Eigen's MPL-2.0 applies to Eigen files; it does not change the license of the
rest of trellis2.c. If an Eigen file is modified, review the MPL-2.0
file-level obligations. Binary distributors should preserve the Eigen license
texts and make the corresponding, pinned Eigen source available.

The release build disables the meshoptimizer demo, raylib examples/games, and
Eigen benchmarks. Keep those directories out of binary packages: they contain
separately licensed or provenance-sensitive benchmark code and sample assets,
including GPL, Creative Commons, freeware, and unspecified terms.

Release archives may also contain redistributable platform runtime libraries,
such as CUDA, Vulkan-loader, compiler, or C/C++ runtime binaries. These are not
covered by either the trellis2.c MIT License or the `3rd/` notices. A release
distributor must comply with the terms supplied by the corresponding SDK or
runtime vendor.

## Model weights

No model weights are tracked in this repository or covered by its MIT License.
The release packages created by `tools/package_windows_release.ps1` and
`tools/package_linux_release.sh` contain a downloader, but no weights. The
downloader preserves upstream `README*` and `LICENSE*` files where they are
available; do not remove them.

The following inventory describes the default and optional sources known to
the project at the review date. Always verify the exact repository and revision
you download, because mirrors, conversions, and combined checkpoints do not
erase or replace upstream terms.

| Model or checkpoint | Project source/use | License note |
|---|---|---|
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | Default shape and PBR generation weights | The upstream model card identifies the release as MIT. Preserve the downloaded model card/license. |
| [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | Legacy sparse-structure decoder files copied into `TRELLIS.2-4B/ckpts` | The upstream model card identifies the repository as MIT. The downloader fetches only the required checkpoint files, so retain this source record and verify the selected revision. |
| [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | Image encoder; the default downloader uses a public mirror for transport | **Meta DINOv3 License, not MIT.** It includes redistribution, notice, use, reverse-engineering, trade-control, and restricted-end-use conditions. Redistribution must include the DINOv3 agreement. A mirror or conversion does not relicense the weights. |
| [BiRefNet-GGUF](https://huggingface.co/Acly/BiRefNet-GGUF) | Optional/default background removal | The conversion model card is marked MIT and identifies the original BiRefNet model. Verify both the converted and upstream terms for the revision used. |
| [Pixal3D](https://github.com/TencentARC/Pixal3D) and NAF checkpoints | Optional Pixal3D pipeline; supplied separately | The current official Pixal3D project is MIT, but third-party, mirrored, NAF-only, and combined checkpoints can have additional terms. Follow the exact checkpoint source and its NOTICE/license files. |
| SegViGen checkpoints | Optional mesh segmentation; supplied or converted separately | Conversion does not relicense a checkpoint. Preserve and follow the exact model-card/license files from the selected source and revision. |
| TokenSkin/TokenRig checkpoints | Optional rigging; supplied or converted separately | The official project and every base/combined checkpoint can have distinct terms. Conversion to the trellis2.c format does not relicense Qwen, Michelangelo, or other embedded weights. |

Users are responsible for having the necessary rights to their input images,
meshes, datasets, and other assets. trellis2.c does not claim ownership of user
outputs and does not grant rights that are controlled by an input or model
provider. Review the applicable model and input terms before commercial use or
redistribution.

## Binary distribution

The release packaging scripts copy this file, the project `LICENSE`, and the
direct dependency license texts into each package. If dependencies, runtime
DLLs/shared libraries, model sources, or sample assets change, update the
inventory and the package license bundle before publishing a release.
