# vkmesh GPU mesh post-processing optimization

This report records the implementation scope, optimization order, and measured
quality/performance of the `vkmesh` branch. The measurements were made on an
RTX 4090 D with the Vulkan device and model assets used by this workspace.

The short conclusion is:

- `vkmesh` now provides a practical low-memory post-processing path for the
  TRELLIS.2 generator.
- On the real 5.423M-face remesh fixture, simplify fell from 70.38 seconds to
  9.84--15.12 seconds while its workspace fell from 918.1 MiB to 577.6 MiB.
- The final HEAD completed the real image-to-GLB pipeline and produced a valid
  451,720-vertex / 951,990-face GLB with no invalid, duplicate, degenerate, or
  boundary faces/edges.
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
- Every ordinary `vkmesh_dispatch` still records, submits, and waits on a fence;
  this remains a performance gap versus CUDA stream scheduling and CUB.

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
| New vkmesh, topology guard | 9.84--15.12 s | included | 577.6 MiB workspace; 636--728 MiB NVML | about 945k--958k | 0 |
| CuMesh | 3.09 s | 0.903 s | 1,267 MiB NVML | 962,082 | 1,021 |

GPU clock/power state caused the new vkmesh spread; a same-state frozen guard
run was 15.12 seconds versus 14.34 seconds for the final code, so the later
diagnostic/finalizer work did not regress simplify.

Relative to the old implementation, the guarded path is 4.65--7.15x faster and
uses 37.1% less tracked workspace. CuMesh's kernel is still much faster, while
its observed transient NVML peak is about twice the new vkmesh peak on this
fixture.

### Same clean approximately-2M-face input

The common input contains 972,492 vertices / 1,992,306 faces and was generated
from the real remesh fixture with the current topology-safe simplifier.

| Operation | Implementation | Core/wall | Workspace / allocations | NVML delta | Output |
|---|---|---:|---:|---:|---:|
| simplify to 1M | vkmesh | 2.33 s | 211.2 MiB workspace | 266.7 MiB | 454,606 V / 956,534 F |
| simplify to 1M | CuMesh | 0.227 s core; 2.37 s process | 50.7 MiB Torch-tracked (extension temporaries are not fully tracked) | 687.2 MiB | 458,520 V / 931,354 F |
| remesh512 | vkmesh | 33.50 s | 103.1 MiB workspace | 190.9 MiB | 3,193,463 V / 6,438,676 F |
| remesh512 | CuMesh | 1.864 s core; 4.49 s process | 2,519.9 MiB Torch allocated / 2,726 MiB reserved | 3,102.1 MiB | 3,193,463 V / 6,438,676 F |

The remesh outputs have exactly the same counts because both implementations use
the same narrow-band/DC construction at the matched scale. CuMesh is about 18x
faster in the remesh core, while vkmesh uses about one-sixteenth of the measured
incremental VRAM. This is the clearest current speed/memory tradeoff.

On the same 2M->1M simplify comparison, sampled surface fidelity was nearly
identical. vkmesh emitted no duplicate, degenerate, or unused geometry; CuMesh
emitted 1,021 duplicate instances, four geometric degenerates, and 1,645 unused
vertices. CuMesh's lower non-manifold-edge count must not be read as a strict
topology improvement because invalid duplicate-producing collapses also change
that count.

### Complete isolated post-process

| Version | Input -> remesh -> final | Wall | Workspace | NVML delta | CPU peak private |
|---|---|---:|---:|---:|---:|
| Old | 3.979M -> 5.423M -> 936,578 F | 162.90 s | 918.1 MiB | 928.2 MiB | 1,233.3 MiB |
| New | 3.979M -> 5.423M -> 949,986 F | 102.37 s | 577.6 MiB | 644.6 MiB | 986.1 MiB |

The complete mesh processor is 37.2% faster, tracked workspace is 37.1% lower,
and measured incremental VRAM is 30.6% lower. Remesh is now the dominant stage;
most of the achieved runtime gain came from simplify and eliminating stage
round-trips.

### Final-HEAD TRELLIS.2 end-to-end run

The final branch HEAD, after merging current `main`, ran:

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

Measured result:

- Return code: 0.
- Wall: 126.77 seconds.
- Raw: 1,833,567 V / 3,979,440 F.
- Remesh: 2,687,875 V / 5,423,072 F.
- Final GLB: 451,720 V / 951,990 F.
- vkmesh workspace peak: 577.6 MiB.
- Whole model-process NVML delta: 5,893.5 MiB.
- Whole process CPU peak working set/private: 2,824.4 / 6,995.5 MiB.

The whole-process figure includes DINO/BiRefNet/TRELLIS weights and inference
workspaces and is not a vkmesh requirement. It also varies more with background
GPU state; the isolated 644.6 MiB result is the useful mesh-processor peak.

The GLB container length/chunks, accessors, finite positions, and index bounds
were independently parsed. Exact scalable topology validation reported:

| Metric | Final HEAD |
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
2. Record more of a simplify iteration into fewer command buffers; ordinary
   dispatch currently incurs a fence wait.
3. Add a persistent buffer pool and device-local-only workspaces with small
   staging/readback buffers for GPUs without a large host-visible BAR heap.
4. Reduce full CSR/edge/QEM reconstruction frequency or maintain incremental
   topology.
5. Optimize link-condition neighborhood queries without weakening the output
   invariant.
6. If PaMO-equivalent guarantees become a product requirement, add a triangle
   BVH self-intersection pass with rollback, then a collision-aware Safe
   Projection stage. These are correctness features with non-trivial time and
   memory cost, not small simplify optimizations.
