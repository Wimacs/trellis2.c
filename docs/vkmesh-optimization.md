# vkmesh GPU mesh post-processing optimization

This report records the implementation scope, optimization order, and measured
quality/performance of the `vkmesh` branch. The measurements were made on an
RTX 4090 D with the Vulkan device and model assets used by this workspace.

The short conclusion is:

- `vkmesh` now provides a practical low-memory post-processing path for the
  TRELLIS.2 generator.
- On the real 5.423M-face remesh fixture, simplify fell from 70.38 seconds to
  1.065--1.110 seconds while its workspace fell from 918.1 MiB to 613.7 MiB.
- The final HEAD completed the real image-to-GLB pipeline in 115.38 seconds and
  produced a 465,664-vertex / 979,846-face GLB.
- CuMesh remains much faster for remesh. On the clean 2M fixture its measured
  0.227-second CUDA call is 2.9--3.6x faster than vkmesh's measured wall time;
  on the 5.423M fixture its 0.903-second CUDA call is now close to vkmesh wall.
  CuMesh used 1,267 MiB measured transient VRAM versus vkmesh's final 674 MiB
  NVML delta, and permits topology defects that `vkmesh` rejects.
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
3. Sort large key arrays with eight stable 8-bit LSD radix passes, then compact
   unique edges and identify boundary vertices. Small, oversized, or
   memory-constrained arrays retain the batched merge fallback.
4. Accumulate a 10-value symmetric QEM for every vertex.
5. Evaluate every unique edge in parallel. QEM plus edge length is first tested
   as a lower bound against the current threshold; impossible candidates skip
   the expensive neighborhood work. Remaining candidates add the skinny
   penalty and run all self-edge, non-finite/collinear, normal-flip,
   relative-area, retained-duplicate, and manifold-link guards.
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

The radix path keeps keys as `uvec2`, processing the four low-word bytes and
then the four high-word bytes so no shader `int64` feature is required. Each
pass records histogram, digit-row scan, digit-offset, and stable scatter; all
eight passes fit one 32-dispatch command buffer. The histogram aliases a dead
CSR/boundary buffer, and a budget check selects merge sort instead when the
temporary would endanger later simplify allocations.

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
  Simplify now batches dependent sequences, including all eight radix passes,
  but count-driven allocation/readback boundaries remain a gap versus CUDA
  stream scheduling and CUB.

## Dispatch batching and explicit buffer reuse

The batching and workspace changes keep the implementation local to `vkmesh.c`
and leave the public API unchanged. They deliberately use fixed-size command
arrays and explicitly named buffers rather than a general command graph or
allocator abstraction:

- One batch records at most 32 compute dispatches, inserts conservative
  compute-to-compute barriers, and waits on one fence.
- Descriptor sets are allocated once with the Vulkan context and reused after
  each synchronous batch completes.
- Prefix-scan levels, CSR construction, the QEM/candidate/collapse sequence,
  prefix compaction, and all 32 radix dispatches are submitted in explicit
  batches.
- A single phase buffer first stores sorted edge keys and is destructively
  repurposed as packed simplify scratch only after unique-edge compaction has
  completed. It persists across productive rounds.
- The sort's dead temporary becomes unique-edge storage. After collapse,
  adjacency, boundary, and counter buffers become compaction outputs instead of
  triggering equivalent new allocations.
- Stable face/vertex prefix scans replace the old global compaction atomics and
  CAS vertex map. The final counts are read together after one fence.
- A no-progress round releases the persistent phase buffer, preventing a stale
  large allocation from overlapping the next round on pathological `V >> F`
  input.

The resulting code has explicit ownership-transfer points and one fixed batch
type; it does not add a public workspace-pool or scheduler interface.
`TRELLIS_VKMESH_PROFILE=1` enables accumulated per-pipeline GPU timestamps in
the existing command buffers; it is off by default and adds no extra submit or
fence wait.

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
| 15 | Add optional per-pipeline Vulkan timestamps | `5433019` | Separate GPU kernel time from host allocation/submit cost |
| 16 | Persist explicitly named simplify workspace buffers | `80d0b74` | Remove repeated allocation/map churn without a generic pool |
| 17 | Build CSR in one batched pass | `c1acf7a` | Remove total-adjacency readback and redundant buffers |
| 18 | Compact output with stable prefix scans | `251c04d` | Remove global face atomic, vertex CAS, and three submissions |
| 19 | Add 8-bit LSD radix sort with merge fallback | `5f85583` | Cut 5.423M edge-sort GPU time from about 245 ms to about 151 ms |
| 20 | Reject over-threshold base costs before link checks | `971afc3` | Preserve guards while cutting link/propagation work |

## Benchmark methodology

All comparisons below use device 0 on the same RTX 4090 D. NVML figures are the
maximum total-device `memory.used` minus the median idle usage sampled around
the run; per-process memory is unavailable under this WDDM configuration.
`vkmesh workspace` is the allocator's own peak and excludes the Vulkan driver
context. The NVML delta can include unrelated-process noise. CuMesh timings
distinguish the CUDA simplify/remesh call from Python process startup where
available.

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
| Batched/reused vkmesh (`8511745`) | 3.39--3.46 s | included | 577.6 MiB workspace | 957,252--958,320 | 0 |
| Final vkmesh, radix + early reject | 1.065--1.110 s | included | 613.7 MiB workspace; 674 MiB NVML delta | 983,722--984,062 | 0 |
| CuMesh | 3.09 s | 0.903 s | 1,267 MiB NVML | 962,082 | 1,021 |

Relative to the original 70.38-second implementation, the final path is
63.4--66.1x faster and uses 33.2% less tracked workspace. Relative to the
batched/reused checkpoint it is another 3.1--3.2x faster. CuMesh's measured
0.903-second CUDA call is only 15--19% below the complete vkmesh simplify wall
time on this larger fixture, although the implementations retain different face
counts and topology policies. CuMesh's observed 1,267 MiB transient NVML peak
is about 1.9x vkmesh's final 674 MiB delta on this fixture.

Timestamp profiling explains the remaining work. On the final 5.423M run, all
GPU dispatches totalled 238.8 ms: radix histogram/scan/offset/scatter consumed
23.1/12.1/0.7/114.7 ms, edge cost 32.5 ms, cost propagation 7.3 ms, and best-edge
selection 3.9 ms. Host-visible workspace traffic, descriptor/command setup,
count readbacks, mesh I/O, and driver scheduling account for the rest of wall
time. The radix scatter's local 1024-item bitonic rank is now the largest single
GPU kernel family.

Two smaller experiments were deliberately not retained. Raising the large
edge-cost shader from 128 to 256 threads increased its measured 5.423M time from
32.5 to 34.1 ms. Removing barriers between logically disjoint packed-scratch
regions either regressed wall time or fell within noise, while making descriptor
range synchronization harder to audit. The final code keeps 128 threads and
the conservative barriers.

### Same clean approximately-2M-face input

The common input contains 972,492 vertices / 1,992,306 faces and was generated
from the real remesh fixture with the current topology-safe simplifier.

| Operation | Implementation | Core/wall | Workspace / allocations | NVML delta | Output |
|---|---|---:|---:|---:|---:|
| simplify to 1M | vkmesh | 0.657--0.807 s | 224.3 MiB workspace | 288 MiB | about 455k V / 957k F |
| simplify to 1M | CuMesh | 0.227 s core; 2.37 s process | 50.7 MiB Torch-tracked (extension temporaries are not fully tracked) | 687.2 MiB | 458,520 V / 931,354 F |
| remesh512 | vkmesh | 33.50 s | 103.1 MiB workspace | 190.9 MiB | 3,193,463 V / 6,438,676 F |
| remesh512 | CuMesh | 1.864 s core; 4.49 s process | 2,519.9 MiB Torch allocated / 2,726 MiB reserved | 3,102.1 MiB | 3,193,463 V / 6,438,676 F |

The remesh outputs have exactly the same counts because both implementations use
the same narrow-band/DC construction at the matched scale. CuMesh is about 18x
faster in the remesh core, while vkmesh uses about one-sixteenth of the measured
incremental VRAM. This is the clearest current speed/memory tradeoff.

On the same 2M->1M simplify comparison, sampled surface fidelity remained
nearly identical. The selected final vkmesh run emitted no degenerate,
duplicate, boundary, winding-conflict, or unused geometry. Atomic candidate
selection remains intentionally nondeterministic. CuMesh emitted 1,021
duplicate instances, four geometric degenerates, and 1,645 unused vertices.
CuMesh's lower non-manifold-edge count must not be read as a strict topology
improvement because invalid duplicate-producing collapses also change it.

The final binary passed the 100,000-sample regression thresholds on both
fixtures. Symmetric RMS / worst directional p99 / sampled Hausdorff were
0.004317 / 0.008410 / 0.013875 for 2M, and 0.004305 / 0.008383 / 0.013655 for
5.423M. Relative to the input bounding-box diagonal, RMS / p99 / maximum were
0.308% / 0.600% / 0.990% and 0.307% / 0.598% / 0.974%, respectively. Both
outputs had zero invalid indices, geometric degenerates, duplicate faces,
boundary edges, winding conflicts, and unused vertices.

### Complete isolated post-process

| Version | Input -> remesh -> final | Wall | Workspace | NVML delta | CPU peak private |
|---|---|---:|---:|---:|---:|
| Old | 3.979M -> 5.423M -> 936,578 F | 162.90 s | 918.1 MiB | 928.2 MiB | 1,233.3 MiB |
| Pre-batching optimized | 3.979M -> 5.423M -> 949,986 F | 102.37 s | 577.6 MiB | 644.6 MiB | 986.1 MiB |

That historical complete mesh-processor run was 37.2% faster, used 37.1% less
tracked workspace, and used 30.6% less measured incremental VRAM than the old
path. The current changes further reduce its simplify portion; remesh remains
the dominant stage. The isolated remesh sequence was not resampled, but the
full image-to-GLB path below was.

### TRELLIS.2 end-to-end verification

The final branch HEAD ran:

```powershell
build-vkmesh-opt\Release\trellis2-image-to-gltf.exe `
  --model <trellis2-model-package> `
  --dino <dinov3-directory> `
  --birefnet <BiRefNet-F16.gguf> `
  --image example_image\T.png `
  --output build-vkmesh-opt\bench-results\final_vkmesh_end_to_end.glb `
  --pipeline 512 --shape-only --steps 12 --seed 1 --noise-seed 18 `
  --backend vulkan --device 0 --no-model-cache `
  --mesh-postprocess --mesh-remesh --mesh-postprocess-simplify `
  --mesh-decimation-target 1000000 `
  --mesh-remesh-resolution 512 --mesh-remesh-band 1 `
  --mesh-remesh-project 0 --vkmesh-gpu-workspace-budget-mib 2048
```

Final radix/early-reject result:

- Return code: 0.
- Wall: 115.38 seconds.
- Raw: 1,833,567 V / 3,979,440 F.
- Remesh: 2,687,875 V / 5,423,072 F.
- Simplify: 468,643 V / 984,608 F.
- Final component cleanup and GLB: 465,664 V / 979,846 F.
- vkmesh workspace peak: 613.7 MiB.

A preceding instrumented run before dispatch batching took 126.77 seconds and
measured 5,893.5 MiB whole-process NVML delta plus 2,824.4/6,995.5 MiB peak CPU
working-set/private memory. Those whole-process figures include DINO, BiRefNet,
TRELLIS weights, inference workspaces, model loading, and remesh. Their normal
run-to-run variation is large enough that the isolated 5.423M-face measurements
are the useful simplify comparison. The successful end-to-end rerun is the
integration/compatibility check.

The final GLB's 22,935,704-byte container, chunks, accessors, finite positions,
and index bounds were independently parsed. Exact topology validation reported:

| Metric | Final GLB |
|---|---:|
| Non-finite vertices | 0 |
| Invalid/index-degenerate faces | 0 |
| Geometric degenerate faces | 0 |
| Unused vertices | 0 |
| Duplicate face extras | 0 |
| Boundary edges | 0 |
| Non-manifold edges | 22,684 |
| Winding-conflict edges | 0 |

The remesh input already has 23,569 non-manifold edges. The guarded simplifier
does not claim to repair all of them, but it avoids introducing the duplicate
faces used by the old/CuMesh outputs to reach a similar target.

The earlier 200,000-sample GLB-to-remesh comparison remains useful as an
integration-scale reference after inverting the documented GLB axis rotation
`(x,y,z)->(x,z,-y)`:

- symmetric relative mean: 0.2106% of the reference bounding-box diagonal;
- symmetric relative RMS: 0.2334%;
- worst directional p99: 0.4537%;
- sampled Hausdorff: 0.7782%.

This is a sampled approximation, not an exact triangle-to-triangle Hausdorff or
self-intersection test.

Release builds and both focused tests passed after the final edit. Khronos core
and synchronization validation reported zero VUIDs or synchronization hazards
for the focused suite and a full 2M-to-1M simplify. The focused suite includes
a 174k-face multi-level-scan simplify to 50k faces.

## Memory guidance for a 2M-face asset

For a clean approximately-2M-face mesh on this GPU:

- vkmesh simplify-only: 224.3 MiB tracked workspace / 288 MiB incremental NVML.
- CuMesh simplify-only: the same process showed 687.2 MiB incremental NVML
  usage despite only 50.7 MiB being visible to PyTorch's tensor allocator.
- vkmesh remesh512-only: 103.1 MiB tracked workspace / 190.9 MiB NVML, but about
  33.5 seconds.
- CuMesh remesh512-only: 3,102.1 MiB NVML / 2,519.9 MiB Torch allocation, but
  only 1.86 seconds in the core call.

For a full sequence, stages are sequential rather than simply additive. Peak is
the larger maximum live footprint of the remesh and simplify stages, including
each stage's live mesh plus scratch. The real TRELLIS raw->remesh5.4M->simplify
sequence now peaks at 613.7 MiB tracked workspace. An isolated simplify run
measured 674 MiB incremental NVML; the earlier pre-radix build measured 644.6
MiB at 577.6 MiB tracked workspace. Resolution and active narrow-band voxel
count matter more to remesh memory than the original face count, so a blanket
estimate based only on "2M input faces" is not reliable.

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
2. Replace radix scatter's 1024-item local bitonic rank with a stable subgroup
   ballot/block-radix rank. It accounts for about 115 of 151 radix milliseconds
   on the 5.423M fixture; keep the portable bitonic path as fallback.
3. Move large scratch to device-local-only memory with small staging/readback
   buffers for GPUs without a large host-visible BAR heap. Keep actual-count
   readbacks where they prevent oversized output allocation.
4. Reduce full CSR/edge/QEM reconstruction frequency or maintain incremental
   topology.
5. Optimize the remaining link-condition neighborhood queries without weakening
   the output invariant. Base-cost threshold rejection already avoids them for
   candidates that cannot collapse in the current round.
6. If PaMO-equivalent guarantees become a product requirement, add a triangle
   BVH self-intersection pass with rollback, then a collision-aware Safe
   Projection stage. These are correctness features with non-trivial time and
   memory cost, not small simplify optimizations.
