# Runtime architecture

The source tree separates a model package, a task, an architecture, and an
operator.  They are related, but they are not interchangeable:

```text
models/<package>/model.json     runtime data and component wiring
src/runtime/                    package loading and static registries
src/tasks/<task>/               typed request/result orchestration
src/architectures/<name>/       graph construction and weight contracts
src/weights/                    model-neutral tensor storage and loading
src/ops/                        reusable CPU/CUDA/Vulkan operators
apps/ and tools/                user-facing entry points
```

`models/` never contains C implementations.  A package selects registered
tasks and architecture implementations by stable identifiers, maps semantic
component roles to weight files, and supplies per-component execution policy.
The package root is therefore portable and can be inspected without loading
multi-gigabyte weights.

## Dependency direction

```text
apps/tools -> tasks -> architectures -> ops
                  \-> weights ------/
       runtime registry and package metadata are shared by tasks/architectures
```

An operator must not inspect a checkpoint to decide which model is running.
An architecture validates and binds its own tensors.  A task adapter owns
model-family workflow differences.  For example, the Pixal3D image-to-3D
adapter owns projected DINO/NAF conditioning, its foreground requirement,
camera defaults, and cascade-coordinate quantization.  TRELLIS.2 uses a
different adapter while sharing most architectures and operators.

Execution choices such as explicit SDPA versus Flash Attention and activation
dtype belong to a package/component instance.  They are not process-global
model identity.  This lets two package instances use different safe numeric
policies in the same process.  Flash K/V dtype is part of that policy: legacy
Flash means F16, while BF16 Flash is represented by a separate ABI-compatible
mode. Pixal3D requests strict BF16 Flash for every flow. Trellis2 keeps the
faster F16 MMA lowering for sparse and 512-resolution components, but its 1024
shape and texture components request BF16 after long-sequence functional tests
proved that F16 accumulation can overflow there. Backend selection is driven
by each component contract. A strict BF16 request must not fall through to the
ordinary unscaled F16 attention path.

On NVIDIA Vulkan devices without native BF16 cooperative-matrix operands, the
backend automatically selects the validated D128 lowering when all required
capabilities match. It stages BF16 K/V through F16 cooperative-matrix operands,
keeps QK and P×V accumulation in F32, and applies an independent power-of-two V
scale to every 16-channel panel. No model command requires an environment
variable; `GGML_VK_BF16_F16_MMA=0` is only a backend-diagnostic opt-out. The
dtype/head-size/capability gate leaves every ordinary F16 attention path
unchanged. New model families still need to validate Q/K range, softmax-tail
sensitivity, and within-panel V exponent span; native BF16 cooperative-matrix
operands remain the preferred path.

## Adding another task

A task has its own typed input/output contract and registry descriptor.  It may
reuse existing architectures and operators or introduce new ones.  A future
AI weight-binding task, for example, can consume an image plus a mesh and
produce a rigging result without depending on the image-to-3D mesh decoder,
texture baker, or glTF exporter.

The extension sequence is:

1. Register the task descriptor and implement `src/tasks/<task>/`.
2. Register any new architecture descriptors and weight-binding contracts.
3. Add reusable operators only where the architecture cannot be expressed by
   existing operators.
4. Add a model package whose manifest maps task profiles and component roles
   to those registered identifiers.
5. Add task-level contract tests and at least one end-to-end package test.

The legacy `trellis_pipeline_image_to_gltf()` and
`trellis_pipeline_image_to_gltf_ex()` symbols remain compatibility wrappers
around the registered `image_to_3d` task.

## Model-pinned command line tools

The library may host multiple families for the same task, but a user-facing
inference executable belongs to exactly one model family.  The current
model-pinned entry points are:

```text
trellis2-image-to-gltf  -> family=trellis2, profile=512
trellis2-texture-mesh   -> family=trellis2, task=mesh_texturing
trellis2-segment-mesh   -> family=trellis2, task=mesh_segmentation, profile=512_full
pixal3d-image-to-gltf   -> family=pixal3d, profile=1024_cascade
tokenskin-rig           -> family=tokenskin, task=mesh_rigging
```

The two image-to-3D tools reuse the registered `image_to_3d` task and its
operators.  They do not select a family from `argv[0]`, expose a `--family`
switch, or silently dispatch from model metadata.  Their model-pinned library
wrappers validate the package family before image loading or backend
initialization.  A model for a different task receives another task-specific
executable instead of adding modes to an existing tool. Existing-mesh
texturing follows that rule even though it reuses TRELLIS.2 image conditioning,
texture flow, texture decoding, and PBR export. TokenSkin likewise reuses the
runtime and operators where appropriate, but its mesh-rigging pipeline and CLI
remain separate from the image-to-3D and mesh-texturing executables.

SegviGen follows the same separation. Its package owns only the Full paired
flow, while the base TRELLIS.2 package owns DINO-independent SLat
normalization, shape/texture encoders, and decoders. The mesh-segmentation task
duplicates sparse coordinates into a dynamic/fixed 2N transformer sequence,
integrates only the first N rows, then turns decoded categorical colors into
connected face instances and a multi-node GLB. That task-specific pairing and
postprocess do not add a SegviGen mode to the generic image-to-3D task.
