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
by each component contract and must never silently narrow a strict BF16 request
to F16.

That rule also applies to backend-specific acceleration experiments. On a
Vulkan implementation without native BF16 cooperative-matrix operands, strict
BF16 Flash remains on the scalar-F32 streaming lowering unless the process
explicitly opts into `GGML_VK_BF16_F16_MMA=1`. The opt-in lowering is
capability-gated and isolated from package policy: it may stage BF16 K/V through
scaled F16 cooperative-matrix operands with F32 QK/P×V accumulators, but it
must not silently become the default or alter another model instance's F16
attention path. New model families must validate their Q/K range, softmax-tail
sensitivity, and the within-panel V exponent span before opting into that
lowering; native BF16 cooperative-matrix operands remain the preferred path.

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
