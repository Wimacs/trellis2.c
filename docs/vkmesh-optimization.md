# vkmesh GPU mesh post-processing optimization

This report records the implementation scope, optimization order, and measured
quality/performance of the `vkmesh` branch. The measurements were made on an
RTX 4090 D with the Vulkan device and model assets used by this workspace.

The short conclusion is:

- `vkmesh` now provides a practical low-memory post-processing path for the
  TRELLIS.2 generator.
- On the real 5.423M-face remesh fixture, simplify fell from 70.38 seconds to
  3.39--3.46 seconds while its workspace fell from 918.1 MiB to 577.6 MiB.
- The final HEAD completed the real image-to-GLB pipeline in 129.97 seconds and
  produced a 452,629-vertex / 953,782-face GLB.
- CuMesh remains much faster inside its CUDA kernels, especially for remesh,
  but its measured transient VRAM is substantially higher and its simplifier
  permits topology defects that `vkmesh` now rejects.
- This is not a complete PaMO implementation. In particular, `vkmesh` does not
  provide PaMO's self-intersection rollback or collision-aware Safe Projection.

## Role in the TRELLIS.2 pipeline

The production shape-only path is:

```text
TRELLIS.2 decoder mesh
  -> fill small boundary loops
  -> preserve pre-remesh projection source when texturing is enabled
  -> sparse narrow-band UDF
  -> Dual Contouring remesh
  -> parallel edge-collapse simplify with relative-degenerate rejection
  -> optionally remove remaining degenerates; remove small manifold-sheet components
  -> one final GPU-to-host download
  -> GLB export
```

Remesh is deliberately before simplify. The decoder mesh is open and highly
non-manifold; remesh converts it into a closed, much cleaner surface on which
topology-constrained collapses can make reliable progress. Directly simplifying
an arbitrary dirty mesh remains best-effort.

The C API path keeps remesh output on the device for simplify and final cleanup.
It does not run the broad cleanup sequence (duplicate removal, non-manifold
splitting, repeated fill, orientation) after remesh because experiments showed
that those generic operations damaged this generator's reconstructed surface.

## CuMesh and PaMO scope mapping

Primary references:

- [CuMesh](https://github.com/JeffreyXiang/CuMesh), MIT
- [PaMO project](https://seonghunn.github.io/pamo/)
- [PaMO code](https://github.com/SarahWeiii/pamo), AGPL-3.0
- [PaMO paper](https://arxiv.org/abs/2509.05595)

| Capability | vkmesh | CuMesh | PaMO |
|---|---|---|---|
| Narrow-band UDF | Vulkan distance queries; CPU sparse-grid orchestration | CUDA/Torch | CUDA voxel hierarchy |
| Dual extraction | Simple DC; CPU topology construction | CUDA DC + Torch topology | PDMC / improved DualMC |
| Parallel QEM collapse | Vulkan compute | CUDA | CUDA |
| Edge-length and skinny penalties | Yes | Yes | Yes |
| Local link/topology condition | Strict local guard | No equivalent complete guard | Protected by stronger global checks |
| Triangle self-intersection detection and collapse rollback | No | No | Yes |
| Collision-aware Safe Projection | No | No | Yes |
| Nearest-surface vertex blend | Optional | Optional | Not equivalent to its Safe Projection |
| Duplicate/degenerate/component cleanup | Yes | Yes | Not its main API surface |
| Non-manifold splitting, winding, hole fill | Yes | Yes | Stage 1 targets a clean reconstructed mesh |
| UV unwrap | CPU xatlas | xatlas plus GPU clustering | No |

The precise description is therefore:

> `vkmesh` implements a CuMesh-style narrow-band remesh and a PaMO-inspired
> parallel edge-collapse backbone, with additional local topology and geometry
> guards. It does not implement PaMO's intersection-free rollback or Safe
> Projection stages.

The pre-remesh `projection_mesh_out` used by TRELLIS texture baking is a sampling
source, not PaMO Safe Projection. `vkmesh_remesh_project_back` is only a
per-vertex closest-point blend and has no collision barrier or self-intersection
guarantee.

## Parallel algorithms

### Simplify iteration

Every simplify iteration performs these bulk-parallel operations:

1. Count vertex-face degrees and build a CSR adjacency with a hierarchical
   prefix scan.
2. Expand every triangle into three canonical 64-bit undirected-edge keys.
3. Sort keys using workgroup-local sorting followed by batched merge passes,
   then compact unique edges and identify boundary vertices.
4. Accumulate a 10-value symmetric QEM for every vertex.
5. Evaluate every unique edge in parallel. The cost combines QEM, edge length,
   and skinny-triangle penalties. Candidate rejection covers self-edges,
   non-finite/collinear output, normal flips, relative-area degeneracy, retained
   duplicate faces, and the manifold link condition.
6. Atomically propagate each edge cost to incident faces. An edge is selected
   only when it is the best candidate throughout both endpoint stars, producing
   independent local collapse regions.
7. Collapse selected edges, mark removed faces, and compact faces and vertices
   with GPU scans.
8. Increase the admissible cost threshold when progress becomes small. Stop on
   target, explicit step limit, repeated no-progress, or threshold exhaustion.

This has the same broad independent-region structure as the CuMesh/PaMO
simplifier. The stricter local link condition is a deliberate quality/speed
tradeoff: it prevents duplicate faces and topology destruction, but its CSR
neighborhood scans cost time and can reject most candidates on severely dirty
meshes.

### Remesh

The remesher builds a triangle BVH, refines a sparse narrow band around the
surface, evaluates unsigned distances on active grid vertices, subtracts the
band epsilon, solves simple Dual Contouring vertices, and emits triangles from
active-cell topology. Distance evaluation is Vulkan-parallel, but BVH build,
sparse-grid refinement/hash, Dual Contouring orchestration, and topology output
still contain substantial CPU work. This is now the largest remaining runtime
bottleneck.

### Cleanup

Cleanup uses sorted canonical face/edge records and scans for duplicate and
degenerate removal, boundary discovery, non-manifold sheet splitting, hole
filling, and orientation propagation. Small-component removal intentionally
uses manifold-sheet connectivity (only incidence-two edges join faces), matching
the desired behavior for deleting artifacts attached through non-manifold
junctions.

Component area is deterministic: the GPU writes one value per face, the host
performs a bulk double-precision reduction by component root, and one compact
keep mask is uploaded. This replaced a contended bounded float-CAS loop that
could silently lose area contributions and produce scheduling-dependent output.

## Why the old implementation was slow

The dominant causes were algorithmic and synchronization overhead, not simply
"Vulkan is slow":

- Global bitonic sorting cost `O(N log^2 N)` and required hundreds of dependent
  dispatch/submit/wait operations on multi-million-edge meshes.
- Hillis--Steele scans moved `O(N log N)` data and submitted one dispatch per
  stride.
- Generic edge records used 20 bytes, were padded to a power of two, and needed
  an equally large ping-pong buffer.
- Buffers that shaders immediately overwrote were nevertheless zero-filled from
  the CPU.
- Construction buffers and scratch regions lived longer than their consumers.
- Remesh, simplify, and cleanup downloaded and re-uploaded meshes between stages.
- Soft fallback could rerun an expensive completed device stage after a fatal
  Vulkan/OOM failure.
- Ordinary standalone dispatch still records, submits, and waits on a fence.
  Simplify now batches its dependent sequences, but sort passes and count-driven
  allocation boundaries remain a gap versus CUDA stream scheduling and CUB.

## Dispatch batching and explicit buffer reuse

Commit `8511745` keeps the implementation local to `vkmesh.c` and leaves the
public API unchanged. It deliberately uses a fixed-size command array rather
than a general command-graph or allocator abstraction:

- One batch records at most 32 compute dispatches, inserts conservative
  compute-to-compute barriers, and waits on one fence.
- Descriptor sets are allocated once with the Vulkan context and reused after
  each synchronous batch completes.
- Prefix-scan levels, degree/seed setup, adjacency construction, the 12-kernel
  QEM/candidate/collapse sequence, and remap/vertex compaction are submitted in
  small explicit batches.
- A single phase buffer first stores sorted edge keys and is destructively
  repurposed as packed simplify scratch only after unique-edge compaction has
  completed. It persists across productive rounds.
- The sort's dead temporary becomes unique-edge storage. After collapse,
  adjacency, boundary, and counter buffers become compaction outputs instead of
  triggering equivalent new allocations.
- Vertex-map assignment remains a separate submission: its actual vertex count
  is read before allocating the output vertex buffer. This preserves the low
  memory peak when cleanup removes most vertices.
- A no-progress round releases the persistent phase buffer, preventing a stale
  large allocation from overlapping the next round on pathological `V >> F`
  input.

The resulting code has explicit ownership-transfer points and one fixed batch
type; it does not add a public workspace-pool or scheduler interface.

## Optimization order and staged commits

The work was ordered by expected benefit and risk:

| Order | Change | Commit | Result |
|---:|---|---|---|
| 1 | Prefer device-local workspace and enforce a budget | `1d349dc` | Better bandwidth and bounded allocation |
| 2 | Bound simplify lifetime/progress | `2257416` | Prevent unbounded rounds/workspace |
| 3 | Add scalable topology/quality validator | `09ff1fa` | Multi-million-face exact regression checks |
| 4 | Replace bitonic sort with batched merge sort | `ca276f2` | `O(N log N)` sort and far fewer submissions |
| 5 | Replace serial/stride scans with hierarchical scans | `d73691d` | Linear-work prefix scans |
| 6 | Trim scratch and reject invalid collapses | `4ee8d51` | Lower peak and safer output |
| 7 | Skip redundant workspace clears | `a65c84d` | Less host-visible memory traffic |
| 8 | Compact simplify keys to 64 bits | `7638c30` | 60% smaller edge-sort records |
| 9 | Enforce link condition and retained-face safety | `435d9d5` | No duplicate faces from tested clean input |
| 10 | Propagate fatal Vulkan errors | `5a26a27` | No expensive invalid fallback rerun |
| 11 | Enforce relative degeneracy | `dff9a77` | Scale-aware geometry validity |
| 12 | Keep remesh result on device through finalization | `fd677df` | One final download; narrower cleanup |
| 13 | Report an unmet simplify target | `11e7262`, `db87de3` | Explicit best-effort termination reason |
| 14 | Batch dependent dispatches and reuse phase/compaction buffers | `8511745` | Fewer submit/waits and allocation/map stalls |

## Benchmark methodology

All comparisons below use device 0 on the same RTX 4090 D. NVML figures are the
maximum process-visible device usage minus the median idle usage sampled around
the run. `vkmesh workspace` is the allocator's own peak and excludes the Vulkan
driver context. CuMesh timings distinguish the CUDA simplify/remesh call from
Python process startup where available.

The principal real fixture is deterministic TRELLIS.2 512 output from `T.png`:

- Raw decoder: 1,833,567 vertices / 3,979,440 faces.
- Remesh512: 2,687,875 vertices / 5,423,072 faces.
- RTX 4090 D, Vulkan/CUDA device 0.
- Pipeline: 12 steps, seed 1, noise seed 18, shape-only.

Large artifacts and sampled resource traces remain under the ignored local
`benchmark_outputs/vkmesh/` directory. The tracked validator is
`tools/vkmesh/meshbin_validate.py`.

## Results

### Real 5.423M-face simplify

| Implementation | Wall/process | Core simplify | Peak signal | Output faces | Duplicate extras |
|---|---:|---:|---:|---:|---:|
| Old vkmesh | 70.38 s | included | 918.1 MiB workspace; 979.5 MiB NVML | 946,502 | 1,064 |
| Topology-guard baseline | 9.84--15.12 s | included | 577.6 MiB workspace; 636--728 MiB NVML | about 945k--958k | 0 |
| Final vkmesh, batched/reused | 3.39--3.46 s | included | 577.6 MiB workspace | 957,252--958,320 | 0 |
| CuMesh | 3.09 s | 0.903 s | 1,267 MiB NVML | 962,082 | 1,021 |

Relative to the original 70.38-second implementation, the final path is
20.3--20.8x faster and uses 37.1% less tracked workspace. Relative to the best
9.84-second topology-guard baseline immediately before this change, it is about
2.9x faster. CuMesh's 0.903-second CUDA core is still about 3.8x faster, while
the complete CuMesh Python process (3.09 seconds) is now within about 10% of the
complete vkmesh executable. CuMesh's observed transient NVML peak remains about
twice vkmesh's measured approximately-616 MiB incremental peak on this fixture.

### Same clean approximately-2M-face input

The common input contains 972,492 vertices / 1,992,306 faces and was generated
from the real remesh fixture with the current topology-safe simplifier.

| Operation | Implementation | Core/wall | Workspace / allocations | NVML delta | Output |
|---|---|---:|---:|---:|---:|
| simplify to 1M | vkmesh | 1.15--1.18 s | 211.2 MiB workspace | about 269 MiB | about 454k V / 956k F |
| simplify to 1M | CuMesh | 0.227 s core; 2.37 s process | 50.7 MiB Torch-tracked (extension temporaries are not fully tracked) | 687.2 MiB | 458,520 V / 931,354 F |
| remesh512 | vkmesh | 33.50 s | 103.1 MiB workspace | 190.9 MiB | 3,193,463 V / 6,438,676 F |
| remesh512 | CuMesh | 1.864 s core; 4.49 s process | 2,519.9 MiB Torch allocated / 2,726 MiB reserved | 3,102.1 MiB | 3,193,463 V / 6,438,676 F |

The remesh outputs have exactly the same counts because both implementations use
the same narrow-band/DC construction at the matched scale. CuMesh is about 18x
faster in the remesh core, while vkmesh uses about one-sixteenth of the measured
incremental VRAM. This is the clearest current speed/memory tradeoff.

On the same 2M->1M simplify comparison, sampled surface fidelity was nearly
identical. Final vkmesh emitted no duplicate or unused geometry; one selected
run retained two geometric degenerates from an input containing five. Atomic
candidate selection is intentionally nondeterministic, and other runs retained
zero. CuMesh emitted 1,021 duplicate instances, four geometric degenerates, and
1,645 unused vertices. CuMesh's lower non-manifold-edge count must not be read
as a strict topology improvement because invalid duplicate-producing collapses
also change that count.

The final binary passed the 100,000-sample regression thresholds on both
fixtures. Symmetric RMS / worst directional p99 / sampled Hausdorff were
0.004325 / 0.008426 / 0.012707 for 2M, and 0.004323 / 0.008345 / 0.013937 for
5.423M. Both outputs had zero invalid indices, duplicate faces, boundary edges,
winding conflicts, and unused vertices.

### Complete isolated post-process

| Version | Input -> remesh -> final | Wall | Workspace | NVML delta | CPU peak private |
|---|---|---:|---:|---:|---:|
| Old | 3.979M -> 5.423M -> 936,578 F | 162.90 s | 918.1 MiB | 928.2 MiB | 1,233.3 MiB |
| Pre-batching optimized | 3.979M -> 5.423M -> 949,986 F | 102.37 s | 577.6 MiB | 644.6 MiB | 986.1 MiB |

That historical complete mesh-processor run was 37.2% faster, used 37.1% less
tracked workspace, and used 30.6% less measured incremental VRAM than the old
path. The current change further reduces its simplify portion; remesh remains
the dominant stage and the complete isolated sequence was not resampled.

### TRELLIS.2 end-to-end verification

The final branch HEAD ran:

```powershell
build-vkmesh-opt\Release\trellis2-image-to-gltf.exe `
  --model <trellis2-model-package> `
  --dino <dinov3-directory> `
  --birefnet <BiRefNet-F16.gguf> `
  --image example_image\T.png `
  --output trellis_t512_shape_vkmesh_final_head.glb `
  --pipeline 512 --shape-only --steps 12 --seed 1 --noise-seed 18 `
  --backend vulkan --device 0 --no-model-cache `
  --mesh-postprocess --mesh-remesh --mesh-postprocess-simplify `
  --mesh-decimation-target 1000000 `
  --mesh-remesh-resolution 512 --mesh-remesh-band 1 `
  --mesh-remesh-project 0 --vkmesh-gpu-workspace-budget-mib 2048
```

Current batched/reused result:

- Return code: 0.
- Wall: 129.97 seconds.
- Raw: 1,833,567 V / 3,979,440 F.
- Remesh: 2,687,875 V / 5,423,072 F.
- Final GLB: 452,629 V / 953,782 F.
- vkmesh workspace peak: 577.6 MiB.

A preceding instrumented run before dispatch batching took 126.77 seconds and
measured 5,893.5 MiB whole-process NVML delta plus 2,824.4/6,995.5 MiB peak CPU
working-set/private memory. Those whole-process figures include DINO, BiRefNet,
TRELLIS weights, inference workspaces, model loading, and remesh. Their normal
run-to-run variation is larger than the approximately-6-second simplify saving,
so the isolated 5.423M-face measurements are the useful performance comparison.
The successful end-to-end rerun is the integration/compatibility check.

The preceding instrumented GLB's container length/chunks, accessors, finite
positions, and index bounds were independently parsed. Exact scalable topology
validation reported:

| Metric | Instrumented GLB |
|---|---:|
| Non-finite vertices | 0 |
| Invalid/index-degenerate faces | 0 |
| Geometric degenerate faces | 0 |
| Unused vertices | 0 |
| Duplicate face extras | 0 |
| Boundary edges | 0 |
| Non-manifold edges | 22,694 |
| Winding-conflict edges | 0 |
| Surface area | 7.87099 |

The remesh input already has 23,569 non-manifold edges. The guarded simplifier
does not claim to repair all of them, but it avoids introducing the duplicate
faces used by the old/CuMesh outputs to reach a similar target.

After inverting the documented GLB axis rotation `(x,y,z)->(x,z,-y)`, a
200,000-sample surface comparison against remesh512 produced:

- symmetric relative mean: 0.2106% of the reference bounding-box diagonal;
- symmetric relative RMS: 0.2334%;
- worst directional p99: 0.4537%;
- sampled Hausdorff: 0.7782%.

This is a sampled approximation, not an exact triangle-to-triangle Hausdorff or
self-intersection test.

Release builds and both focused tests passed after the final edit. Khronos core
and synchronization validation reported zero VUIDs or synchronization hazards,
including a 174k-face multi-level-scan simplify to 50k faces.

## Memory guidance for a 2M-face asset

For a clean approximately-2M-face mesh on this GPU:

- vkmesh simplify-only: use about 0.21 GiB tracked workspace and expect about
  0.27 GiB incremental NVML usage.
- CuMesh simplify-only: the same process showed about 0.69 GiB incremental NVML
  usage despite only 50.7 MiB being visible to PyTorch's tensor allocator.
- vkmesh remesh512-only: about 0.10 GiB tracked workspace / 0.19 GiB NVML, but
  about 33.5 seconds.
- CuMesh remesh512-only: about 3.10 GiB NVML / 2.52 GiB Torch allocation, but
  only 1.86 seconds in the core call.

For a full sequence, stages are sequential rather than additive. Peak is driven
by whichever of remesh output or simplify scratch is larger. The real TRELLIS
raw->remesh5.4M->simplify sequence peaked at 577.6 MiB workspace / 644.6 MiB
isolated NVML. Resolution and active narrow-band voxel count matter more to
remesh memory than the original face count, so a blanket estimate based only on
"2M input faces" is not reliable.

## Explicit limits

- Simplify is best-effort. A highly non-manifold 7.964M-face fixture stopped at
  about 5.685M faces after threshold exhaustion. It returned a usable mesh and a
  structured `target_not_reached` warning; it did not silently claim success.
- There is no triangle-triangle self-intersection detector in the validator or
  simplifier, so there is no PaMO-style intersection-free guarantee.
- There is no collapse rollback and no collision-aware Safe Projection.
- Non-manifold stars with expensive/invalid local topology can prevent the
  target from being reached.
- Only the focused vkmesh tests were built and run in the incremental build:
  `trellis2_c_vkmesh_tests` and `trellis2_c_vkmesh_cli_smoke`, both passing.
  This report does not claim that every repository test target was built.

## Highest-return next work

1. Move sparse-grid filtering/hash, DC vertex solving, and topology construction
   from the CPU to Vulkan. Remesh is now the largest end-to-end bottleneck.
2. Replace the remaining merge-sort passes with a Vulkan radix sort and compact
   run-length encoding, provided profiling shows a net win on the target GPU.
3. Move large scratch to device-local-only memory with small staging/readback
   buffers for GPUs without a large host-visible BAR heap. Keep actual-count
   readbacks where they prevent oversized output allocation.
4. Reduce full CSR/edge/QEM reconstruction frequency or maintain incremental
   topology.
5. Optimize link-condition neighborhood queries without weakening the output
   invariant.
6. If PaMO-equivalent guarantees become a product requirement, add a triangle
   BVH self-intersection pass with rollback, then a collision-aware Safe
   Projection stage. These are correctness features with non-trivial time and
   memory cost, not small simplify optimizations.
