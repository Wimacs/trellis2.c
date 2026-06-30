# TRELLIS.2 Operator Map

This file tracks the source checkpoint operator coverage in the C + ggml + CUDA port.

## Dense DiT / flow operators

| Source op | C/ggml implementation | Test |
| --- | --- | --- |
| `nn.Linear` | `trellis_ggml_linear`, `ggml_mul_mat` + optional bias repeat | `test_linear_cuda` |
| `LayerNorm32` / `F.layer_norm` | `trellis_ggml_layer_norm`, `ggml_norm` + affine | `test_norms_cuda` |
| `MultiHeadRMSNorm` | `trellis_ggml_multihead_rms_norm`, `ggml_rms_norm` + gamma | `test_norms_cuda` |
| `nn.GELU(approximate="tanh")` | `ggml_gelu` | `test_feed_forward_cuda` |
| `F.silu` | `ggml_silu` | `test_timestep_mlp_cuda` |
| `TimestepEmbedder.timestep_embedding` | `ggml_timestep_embedding` | `test_timestep_mlp_cuda` |
| `TimestepEmbedder.timestep_embedding` scalar reference | `trellis_timestep_embedding_f32` | `test_stage1_sampler_math` |
| `FlowEulerSampler` rescaled `t_pairs` | `trellis_flow_timestep_pairs_f32` | `test_stage1_sampler_math` |
| full self/cross attention | `ggml_flash_attn_ext` through `trellis_ggml_sdpa` | `test_attention_cuda` |
| `ModulatedTransformerCrossBlock` AdaLN/gate skeleton | `trellis_ggml_modulated_cross_block` | graph builder coverage through compile/tests |
| `RotaryPositionEmbedder` 3D phases | `trellis_rope_3d_phases_f32` | `test_stage1_sampler_math` |
| RoPE apply for ggml attention layout `[head_dim, tokens, heads, batch]` | `trellis_ggml_apply_rope_adjacent`, `trellis_cuda_apply_rope_f32` | `test_rope_adjacent_ggml_cuda`, `test_custom_cuda_kernels` |
| full stage1 flow graph with real checkpoint weights | `trellis_dit_flow_forward`, `trellis-infer --dry-forward` | manual real-weight run |
| flow Euler step | `trellis_flow_euler_step_f32` | `test_flow_and_sparse_host` |
| classifier-free guidance combine | `trellis_flow_cfg_combine_f32` | `test_flow_and_sparse_host` |
| classifier-free guidance rescale | `trellis_flow_cfg_rescale_combine_f32` | `test_stage1_sampler_math` |

## Dense Conv3D decoder kernels

| Source op | CUDA implementation | Test |
| --- | --- | --- |
| `nn.Conv3d` in `SparseStructureDecoder` | `trellis_cuda_conv3d_f32` direct NCDHW kernel with stride/pad/dilation/bias | `test_custom_cuda_kernels` |
| `pixel_shuffle_3d(x, 2)` in `UpsampleBlock3d` | `trellis_cuda_pixel_shuffle_3d_f32` | `test_custom_cuda_kernels` |
| `ChannelLayerNorm3d` | `trellis_cuda_channel_layer_norm_3d_f32` | `test_custom_cuda_kernels` |
| `SiLU` and residual add | `trellis_cuda_silu_f32`, `trellis_cuda_add_f32` | `test_custom_cuda_kernels` |
| SparseStructureDecoder real-weight path | `trellis_ss_decoder_forward_f32_host`, `trellis-infer --dry-decode` | manual real-weight run |

## DINOv3 Image Encoder

| Source op | CUDA implementation | Test |
| --- | --- | --- |
| ViT patch embedding | `trellis_cuda_dino_patch_embed_f32`, `trellis_dino_patch_embed_f32_host` | `test_custom_cuda_kernels`, `trellis-infer --dry-dino-patch` |
| DINOv3 checkpoint binding | `trellis_dino_vit_bind_weights` | manual real-weight run |

## Sparse tensor coordinate operators

| Source op | C implementation | Test |
| --- | --- | --- |
| `SparseDownsample(..., mode="mean")` | `trellis_sparse_downsample_mean_host` | `test_flow_and_sparse_host` |
| `SparseSpatial2Channel` | `trellis_sparse_spatial2channel_host` | `test_flow_and_sparse_host` |
| `SparseChannel2Spatial` | `trellis_sparse_channel2spatial_host` | `test_flow_and_sparse_host` |

These are currently host reference implementations used to lock down semantics.
The CUDA sparse kernels should match these functions byte-for-byte for coords and
within tolerance for features.

## Sparse decoder CUDA kernels

| Source op | CUDA implementation | Test |
| --- | --- | --- |
| `SparseConv3d` submanifold conv, flex_gemm weight layout `[Co,Kd,Kh,Kw,Ci]` | `trellis_cuda_sparse_subm_conv3d_f32` with CUDA coord hash lookup | `test_custom_cuda_kernels` |

## Checkpoint I/O

| Source format | C implementation | Test |
| --- | --- | --- |
| safetensors header | `trellis_safetensors_open` | `test_safetensors` |
| F32/F16/BF16 tensor reads as F32 | `trellis_safetensors_read_f32` | `test_safetensors` |
| TRELLIS.2-4B manifest inspection | `trellis-info` | manual run |
| SparseStructureFlowModel checkpoint contract | `trellis_ss_flow_validate_checkpoint`, `trellis-stage1 --validate` | optional real-checkpoint test |
| SparseStructureDecoder checkpoint contract | `trellis_ss_decoder_validate_checkpoint`, `trellis-stage1 --validate` | optional real-checkpoint test |
| Shape SLat flow checkpoint contract | `trellis_shape_slat_flow_validate_checkpoint`, `trellis-stage2 --validate` | optional real-checkpoint test |
| FlexiDualGridVaeDecoder checkpoint contract | `trellis_shape_decoder_validate_checkpoint`, `trellis-stage2 --validate-shape-decoder` | optional real-checkpoint test |

Linear weights are stored as `[out, in]` in the checkpoint and are bound to ggml
tensors with shape `[in, out]` without moving elements; the contiguous memory
order is already the one `ggml_mul_mat` expects. Conv3D kernels stored as
`[out, in, kd, kh, kw]` are folded into ggml's 4D tensor limit as
`[kw, kh, kd, in * out]` while preserving raw contiguous storage for the direct
CUDA kernel.

## Remaining custom CUDA work

1. Complete the DINOv3 transformer graph after patch embedding.
2. Add image file loading, resize, RGB conversion, and normalization in C.
3. Run the full 12-step stage1 sampler at 16^3 latent resolution with CFG.
4. Optimize Conv3D for full 16->64 decoder runs; the current direct kernel is correctness-first.
5. Convert decoder logits to voxel coordinate buffers and stream them to the raylib viewer.
6. Varlen FlashAttention packing for later sparse SLat stages.
7. Wire the FlexiDualGridVaeDecoder graph around the tested sparse submanifold conv, then optimize it into a gather/GEMM/scatter path.
8. O-Voxel flexible dual-grid mesh conversion in C/CUDA.

## Component status

`trellis_component_status_at()` exposes the same status to tools. The current
code is a tested C + ggml + CUDA foundation, not yet a completed end-to-end
TRELLIS.2 image-to-3D binary.
