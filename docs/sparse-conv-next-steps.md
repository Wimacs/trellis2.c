# Sparse Conv Next Steps

Current status:
- Submanifold sparse conv now builds a neighbor map and a compact per-offset rulebook.
- Shape decoder default path uses per-offset rulebook + ggml CUDA matmul + scatter-add.
- Full shape decode improved from about 39.0s to about 11.3s on the smoke input.
- ConvNeXt sparse convs cache ggml matmul workspaces per `(Ci, Co, pair_count)` within each decoder level.
- Sparse conv weights are pre-packed once per source weight pointer into 27 contiguous `[Ci, Co]` offset matrices.
- `TRELLIS_SPARSE_CONV_BACKEND=map` keeps the previous neighbor-map scalar channel-loop path for A/B checks.

Next target:
- Reduce the remaining 27 per-offset graph executions, gathers, and scatters.

Highest-impact work:
1. Batch offsets when possible.
   - Instead of 27 separate ggml graph executions, group offsets with similar pair counts.
   - Options:
     - one batched ggml matmul graph if ggml shape support is practical,
     - or concatenate gathered pairs for several offsets and run fewer larger GEMMs, with offset-specific scatter metadata.

2. Replace scatter-add atomics with deterministic segmented write when submanifold structure allows it.
   - For each offset, each destination row appears at most once.
   - Across offsets, multiple writes hit the same destination/channel.
   - Current implementation launches scatter after each offset and uses `+=`.
   - Faster path: accumulate into per-offset output slabs then reduce across 27 offsets, or write one fused reduction kernel over offsets.

3. Fuse gather + GEMM for common channel sizes after the ggml baseline is stable.
   - Common `Ci/Co`: 1024, 512, 256, 128 and C2S variants.
   - A custom tiled CUDA kernel can load source rows through rulebook and compute `Co` tiles directly, avoiding materialized gathered features.
   - This is the real FlexGEMM-style path, but should come after graph/weight caching so the remaining bottleneck is clear.

Correctness checks to keep:
- `ctest --test-dir trellis2.c/build --output-on-failure`
- Level-1 decode rulebook vs map exact coords and feature cosine/mean-abs.
- Full decode coords IoU against `TRELLIS_SPARSE_CONV_BACKEND=map`, currently about `0.9999957`.

Useful benchmark commands:

```sh
TRELLIS_PROFILE=1 /usr/bin/time -f 'wall=%e user=%U sys=%S' \
  ./trellis2.c/build/trellis-stage2 \
  --model TRELLIS.2-4B \
  --shape-decode \
  --coords-i32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.coords.i32 \
  --input-slat-f32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.feats.f32 \
  --out /tmp/trellis_rulebook_full
```

```sh
TRELLIS_SPARSE_CONV_BACKEND=map /usr/bin/time -f 'wall=%e user=%U sys=%S' \
  ./trellis2.c/build/trellis-stage2 \
  --model TRELLIS.2-4B \
  --shape-decode \
  --coords-i32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.coords.i32 \
  --input-slat-f32 benchmark_outputs/stage2_shape_smoke/slat_shape_512_final_x_t_step_001.feats.f32 \
  --out /tmp/trellis_map_full_current
```
