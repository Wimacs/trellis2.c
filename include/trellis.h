#ifndef TRELLIS2_C_TRELLIS_H
#define TRELLIS2_C_TRELLIS_H

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRELLIS_MAX_DIMS 8

typedef enum trellis_status {
    TRELLIS_STATUS_OK = 0,
    TRELLIS_STATUS_ERROR = 1,
    TRELLIS_STATUS_INVALID_ARGUMENT = 2,
    TRELLIS_STATUS_IO_ERROR = 3,
    TRELLIS_STATUS_PARSE_ERROR = 4,
    TRELLIS_STATUS_OUT_OF_MEMORY = 5,
    TRELLIS_STATUS_CUDA_UNAVAILABLE = 6,
    TRELLIS_STATUS_NOT_FOUND = 7,
    TRELLIS_STATUS_NOT_IMPLEMENTED = 8,
} trellis_status;

const char * trellis_status_string(trellis_status status);

typedef struct trellis_cuda_context {
    ggml_backend_t backend;
    int device;
} trellis_cuda_context;

trellis_status trellis_cuda_init(trellis_cuda_context * ctx, int device);
void trellis_cuda_free(trellis_cuda_context * ctx);
ggml_gallocr_t trellis_cuda_new_graph_allocator(const trellis_cuda_context * ctx);
trellis_status trellis_cuda_compute_graph(const trellis_cuda_context * ctx, struct ggml_cgraph * graph);

typedef enum trellis_dtype {
    TRELLIS_DTYPE_UNKNOWN = 0,
    TRELLIS_DTYPE_F32,
    TRELLIS_DTYPE_F16,
    TRELLIS_DTYPE_BF16,
    TRELLIS_DTYPE_I64,
    TRELLIS_DTYPE_I32,
    TRELLIS_DTYPE_U8,
    TRELLIS_DTYPE_BOOL,
} trellis_dtype;

const char * trellis_dtype_name(trellis_dtype dtype);
size_t trellis_dtype_size(trellis_dtype dtype);

typedef struct trellis_safetensor_meta {
    char * name;
    trellis_dtype dtype;
    int n_dims;
    int64_t shape[TRELLIS_MAX_DIMS];
    uint64_t data_begin;
    uint64_t data_end;
} trellis_safetensor_meta;

typedef struct trellis_safetensors {
    char * path;
    char * header_json;
    uint64_t header_size;
    uint64_t data_base_offset;
    size_t n_tensors;
    trellis_safetensor_meta * tensors;
} trellis_safetensors;

typedef struct trellis_tensor_store_entry {
    char * name;
    struct ggml_tensor * tensor;
} trellis_tensor_store_entry;

typedef struct trellis_tensor_store {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    trellis_tensor_store_entry * entries;
    size_t n_entries;
    size_t capacity;
} trellis_tensor_store;

trellis_status trellis_safetensors_open(const char * path, trellis_safetensors * out);
void trellis_safetensors_close(trellis_safetensors * st);
const trellis_safetensor_meta * trellis_safetensors_find(const trellis_safetensors * st, const char * name);
uint64_t trellis_safetensor_nelements(const trellis_safetensor_meta * meta);
trellis_status trellis_safetensors_read_f32(
    const trellis_safetensors * st,
    const trellis_safetensor_meta * meta,
    float * dst,
    size_t dst_count);

trellis_status trellis_tensor_store_init(
    trellis_tensor_store * store,
    size_t graph_tensors,
    size_t tensor_data_bytes);

void trellis_tensor_store_free(trellis_tensor_store * store);

const struct ggml_tensor * trellis_tensor_store_get_const(
    const trellis_tensor_store * store,
    const char * name);

struct ggml_tensor * trellis_tensor_store_get(
    trellis_tensor_store * store,
    const char * name);

trellis_status trellis_tensor_store_load_safetensors_f32(
    trellis_tensor_store * store,
    const trellis_cuda_context * cuda,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors);

#define TRELLIS_DIT_FLOW_BLOCKS 30

typedef struct trellis_dit_flow_block_weights {
    struct ggml_tensor * modulation;
    struct ggml_tensor * norm2_gamma;
    struct ggml_tensor * norm2_beta;

    struct ggml_tensor * self_qkv_w;
    struct ggml_tensor * self_qkv_b;
    struct ggml_tensor * self_q_rms_gamma;
    struct ggml_tensor * self_k_rms_gamma;
    struct ggml_tensor * self_out_w;
    struct ggml_tensor * self_out_b;

    struct ggml_tensor * cross_q_w;
    struct ggml_tensor * cross_q_b;
    struct ggml_tensor * cross_kv_w;
    struct ggml_tensor * cross_kv_b;
    struct ggml_tensor * cross_q_rms_gamma;
    struct ggml_tensor * cross_k_rms_gamma;
    struct ggml_tensor * cross_out_w;
    struct ggml_tensor * cross_out_b;

    struct ggml_tensor * mlp_fc1_w;
    struct ggml_tensor * mlp_fc1_b;
    struct ggml_tensor * mlp_fc2_w;
    struct ggml_tensor * mlp_fc2_b;
} trellis_dit_flow_block_weights;

typedef struct trellis_dit_flow_weights {
    int in_channels;
    int out_channels;
    int model_channels;
    int cond_channels;
    int time_frequency_dim;
    int heads;
    int head_dim;
    int mlp_channels;
    int mod_channels;
    int n_blocks;
    int debug_block_parts;
    int debug_disable_rope;
    int emulate_bf16_blocks;
    float final_norm_eps;

    struct ggml_tensor * input_w;
    struct ggml_tensor * input_b;
    struct ggml_tensor * t_embedder_0_w;
    struct ggml_tensor * t_embedder_0_b;
    struct ggml_tensor * t_embedder_2_w;
    struct ggml_tensor * t_embedder_2_b;
    struct ggml_tensor * adaln_w;
    struct ggml_tensor * adaln_b;
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;

    trellis_dit_flow_block_weights blocks[TRELLIS_DIT_FLOW_BLOCKS];
} trellis_dit_flow_weights;

trellis_status trellis_dit_flow_bind_weights(
    trellis_tensor_store * store,
    int in_channels,
    int out_channels,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_ss_flow_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_shape_slat_flow_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size);

struct ggml_tensor * trellis_dit_flow_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * timesteps,
    struct ggml_tensor * context,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dit_flow_weights * weights);

typedef struct trellis_ss_decoder_resblock_weights {
    struct ggml_tensor * norm1_gamma;
    struct ggml_tensor * norm1_beta;
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * norm2_gamma;
    struct ggml_tensor * norm2_beta;
    struct ggml_tensor * conv2_w;
    struct ggml_tensor * conv2_b;
    int channels;
} trellis_ss_decoder_resblock_weights;

typedef struct trellis_ss_decoder_weights {
    struct ggml_tensor * input_w;
    struct ggml_tensor * input_b;
    trellis_ss_decoder_resblock_weights middle[2];
    trellis_ss_decoder_resblock_weights block0;
    trellis_ss_decoder_resblock_weights block1;
    struct ggml_tensor * up0_w;
    struct ggml_tensor * up0_b;
    trellis_ss_decoder_resblock_weights block3;
    trellis_ss_decoder_resblock_weights block4;
    struct ggml_tensor * up1_w;
    struct ggml_tensor * up1_b;
    trellis_ss_decoder_resblock_weights block6;
    trellis_ss_decoder_resblock_weights block7;
    struct ggml_tensor * out_norm_gamma;
    struct ggml_tensor * out_norm_beta;
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;
} trellis_ss_decoder_weights;

trellis_status trellis_ss_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_ss_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_ss_decoder_forward_f32_host(
    const trellis_ss_decoder_weights * weights,
    const float * latent,
    int device,
    int batch,
    int latent_size,
    float ** logits_out,
    int * output_size);

#define TRELLIS_SHAPE_DECODER_LEVELS 5
#define TRELLIS_SHAPE_DECODER_MAX_BLOCKS 16

typedef struct trellis_shape_decoder_convnext_block_weights {
    struct ggml_tensor * conv_w;
    struct ggml_tensor * conv_b;
    struct ggml_tensor * norm_gamma;
    struct ggml_tensor * norm_beta;
    struct ggml_tensor * mlp0_w;
    struct ggml_tensor * mlp0_b;
    struct ggml_tensor * mlp2_w;
    struct ggml_tensor * mlp2_b;
    int channels;
} trellis_shape_decoder_convnext_block_weights;

typedef struct trellis_shape_decoder_c2s_block_weights {
    struct ggml_tensor * norm1_gamma;
    struct ggml_tensor * norm1_beta;
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * conv2_w;
    struct ggml_tensor * conv2_b;
    struct ggml_tensor * to_subdiv_w;
    struct ggml_tensor * to_subdiv_b;
    int in_channels;
    int out_channels;
} trellis_shape_decoder_c2s_block_weights;

typedef struct trellis_shape_decoder_weights {
    struct ggml_tensor * from_latent_w;
    struct ggml_tensor * from_latent_b;
    struct ggml_tensor * output_w;
    struct ggml_tensor * output_b;
    int levels;
    int channels[TRELLIS_SHAPE_DECODER_LEVELS];
    int blocks_per_level[TRELLIS_SHAPE_DECODER_LEVELS];
    trellis_shape_decoder_convnext_block_weights
        blocks[TRELLIS_SHAPE_DECODER_LEVELS][TRELLIS_SHAPE_DECODER_MAX_BLOCKS];
    trellis_shape_decoder_c2s_block_weights up_blocks[TRELLIS_SHAPE_DECODER_LEVELS - 1];
} trellis_shape_decoder_weights;

typedef struct trellis_shape_decoder_debug_options {
    const char * dump_dir;
} trellis_shape_decoder_debug_options;

trellis_status trellis_shape_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_shape_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_shape_decoder_forward_f32_host(
    const trellis_shape_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

trellis_status trellis_shape_decoder_forward_f32_host_debug(
    const trellis_shape_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    const trellis_shape_decoder_debug_options * debug,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

#define TRELLIS_DINO_VIT_LAYERS 24

typedef struct trellis_dino_vit_block_weights {
    struct ggml_tensor * norm1_gamma;
    struct ggml_tensor * norm1_beta;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * k_w;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;
    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;
    struct ggml_tensor * layer_scale1;
    struct ggml_tensor * norm2_gamma;
    struct ggml_tensor * norm2_beta;
    struct ggml_tensor * mlp_up_w;
    struct ggml_tensor * mlp_up_b;
    struct ggml_tensor * mlp_down_w;
    struct ggml_tensor * mlp_down_b;
    struct ggml_tensor * layer_scale2;
} trellis_dino_vit_block_weights;

typedef struct trellis_dino_vit_weights {
    struct ggml_tensor * cls_token;
    struct ggml_tensor * mask_token;
    struct ggml_tensor * register_tokens;
    struct ggml_tensor * patch_w;
    struct ggml_tensor * patch_b;
    trellis_dino_vit_block_weights blocks[TRELLIS_DINO_VIT_LAYERS];
    struct ggml_tensor * norm_gamma;
    struct ggml_tensor * norm_beta;
    int hidden_size;
    int intermediate_size;
    int patch_size;
    int heads;
    int head_dim;
    int register_tokens_count;
} trellis_dino_vit_weights;

trellis_status trellis_dino_vit_bind_weights(
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_dino_patch_embed_f32_host(
    const trellis_dino_vit_weights * weights,
    const float * image,
    int device,
    int batch,
    int image_h,
    int image_w,
    float ** tokens_out,
    int * n_patches_out);

struct ggml_tensor * trellis_dino_vit_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * tokens,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dino_vit_weights * weights);

trellis_status trellis_dino_vit_forward_f32_host(
    const trellis_cuda_context * cuda,
    const trellis_dino_vit_weights * weights,
    const float * tokens,
    const float * cos_phase,
    const float * sin_phase,
    int batch,
    int n_tokens,
    float ** tokens_out);

trellis_status trellis_dino_rope_2d_phases_f32(
    int n_special_tokens,
    int patches_h,
    int patches_w,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* ggml graph builders for TRELLIS.2 dense-token operators.
 *
 * Convention:
 *   - Dense token features are [channels, tokens, batch].
 *   - Linear weights use ggml's [in_channels, out_channels] layout, matching
 *     checkpoint linear weights after the importer adjusts the logical shape.
 *   - Attention tensors are [head_dim, tokens, heads, batch].
 */
struct ggml_tensor * trellis_ggml_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias);

struct ggml_tensor * trellis_ggml_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps);

struct ggml_tensor * trellis_ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

struct ggml_tensor * trellis_ggml_bf16_roundtrip(
    struct ggml_context * ctx,
    struct ggml_tensor * x);

void trellis_ggml_set_flash_attn_enabled(int enabled);

struct ggml_tensor * trellis_ggml_multihead_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

struct ggml_tensor * trellis_ggml_feed_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

struct ggml_tensor * trellis_ggml_timestep_mlp(
    struct ggml_context * ctx,
    struct ggml_tensor * timesteps,
    int frequency_dim,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

struct ggml_tensor * trellis_ggml_sdpa(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale);

struct ggml_tensor * trellis_ggml_apply_rope_adjacent(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

struct ggml_tensor * trellis_ggml_self_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

struct ggml_tensor * trellis_ggml_self_attention_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

struct ggml_tensor * trellis_ggml_cross_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

void trellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0);

void trellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred);

void trellis_flow_cfg_rescale_combine_f32(
    const float * x_t,
    const float * pred_pos,
    const float * pred_neg,
    size_t batch,
    size_t sample_stride,
    float sigma_min,
    float t,
    float guidance_strength,
    float guidance_rescale,
    float * pred);

trellis_status trellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count);

void trellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding);

trellis_status trellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

trellis_status trellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* CUDA custom kernels used where ggml's CUDA backend does not expose a matching
 * TRELLIS.2 operator yet. Device variants expect CUDA device pointers on the
 * current device and use the default stream. Host variants are correctness and
 * tooling helpers: they copy host buffers to the selected device, run the same
 * kernel, synchronize, and copy results back.
 *
 * Tensor memory layout follows contiguous NCDHW:
 *   x:      [batch, in_channels, in_d, in_h, in_w]
 *   weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
 *   y:      [batch, out_channels, out_d, out_h, out_w]
 */
trellis_status trellis_cuda_conv3d_f32(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

trellis_status trellis_cuda_conv3d_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

/* Pixel shuffle input is [batch, out_channels * scale^3, d, h, w],
 * output is [batch, out_channels, d * scale, h * scale, w * scale].
 */
trellis_status trellis_cuda_pixel_shuffle_3d_f32(
    const float * x_dev,
    float * y_dev,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale);

trellis_status trellis_cuda_pixel_shuffle_3d_f32_host(
    const float * x,
    float * y,
    int device,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale);

trellis_status trellis_cuda_channel_layer_norm_3d_f32(
    const float * x_dev,
    const float * gamma_dev,
    const float * beta_dev,
    float * y_dev,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps);

trellis_status trellis_cuda_channel_layer_norm_3d_f32_host(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int device,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps);

trellis_status trellis_cuda_silu_f32(
    const float * x_dev,
    float * y_dev,
    size_t n);

trellis_status trellis_cuda_silu_f32_host(
    const float * x,
    float * y,
    int device,
    size_t n);

trellis_status trellis_cuda_add_f32(
    const float * a_dev,
    const float * b_dev,
    float * y_dev,
    size_t n);

trellis_status trellis_cuda_add_f32_host(
    const float * a,
    const float * b,
    float * y,
    int device,
    size_t n);

/* Sparse row-wise linear used by the shape decoder.
 * x/y are row-major [n, channels]. weight is row-major [out_channels, in_channels].
 */
trellis_status trellis_cuda_sparse_linear_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int64_t n,
    int in_channels,
    int out_channels);

trellis_status trellis_cuda_dino_patch_embed_f32(
    const float * image_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * tokens_dev,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size);

trellis_status trellis_cuda_dino_patch_embed_f32_host(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int device,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size);

/* Submanifold sparse 3D convolution used by FlexiDualGridVaeDecoder.
 * coords are [n,4] = batch,x,y,z and feats/out are row-major [n,channels].
 * weight follows the TRELLIS flex_gemm checkpoint layout [out,kd,kh,kw,in].
 */
trellis_status trellis_cuda_sparse_subm_conv3d_f32(
    const int32_t * coords_dev,
    const float * feats_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * out_dev,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

trellis_status trellis_cuda_sparse_subm_conv3d_f32_host(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    const float * bias,
    float * out,
    int device,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

/* RoPE layout matches trellis_ggml_self_attention internals:
 *   x/out: [head_dim, tokens, heads, batch]
 *   cos/sin: [tokens, head_dim / 2]
 */
trellis_status trellis_cuda_apply_rope_f32(
    const float * x_dev,
    const float * cos_dev,
    const float * sin_dev,
    float * y_dev,
    int batch,
    int tokens,
    int heads,
    int head_dim);

trellis_status trellis_cuda_apply_rope_f32_host(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int device,
    int batch,
    int tokens,
    int heads,
    int head_dim);

typedef struct trellis_ggml_modulated_cross_block_params {
    struct ggml_tensor * block_modulation; /* [6 * channels] */
    struct ggml_tensor * norm2_gamma;      /* [channels] */
    struct ggml_tensor * norm2_beta;       /* [channels] */

    struct ggml_tensor * self_qkv_w;
    struct ggml_tensor * self_qkv_b;
    struct ggml_tensor * self_q_rms_gamma;
    struct ggml_tensor * self_k_rms_gamma;
    struct ggml_tensor * self_out_w;
    struct ggml_tensor * self_out_b;

    struct ggml_tensor * cross_q_w;
    struct ggml_tensor * cross_q_b;
    struct ggml_tensor * cross_kv_w;
    struct ggml_tensor * cross_kv_b;
    struct ggml_tensor * cross_q_rms_gamma;
    struct ggml_tensor * cross_k_rms_gamma;
    struct ggml_tensor * cross_out_w;
    struct ggml_tensor * cross_out_b;

    struct ggml_tensor * mlp_fc1_w;
    struct ggml_tensor * mlp_fc1_b;
    struct ggml_tensor * mlp_fc2_w;
    struct ggml_tensor * mlp_fc2_b;
    int debug_parts;
    int emulate_bf16;
} trellis_ggml_modulated_cross_block_params;

struct ggml_tensor * trellis_ggml_modulated_cross_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params);

struct ggml_tensor * trellis_ggml_modulated_cross_block_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

typedef struct trellis_sparse_tensor_host {
    int32_t * coords; /* [n, 4] = batch, x, y, z */
    float * feats;    /* [n, channels] */
    int64_t n;
    int64_t channels;
} trellis_sparse_tensor_host;

typedef struct trellis_mesh_host {
    float * vertices; /* [n_vertices, 3] */
    int32_t * faces;  /* [n_faces, 3] */
    int64_t n_vertices;
    int64_t n_faces;
} trellis_mesh_host;

void trellis_sparse_tensor_free(trellis_sparse_tensor_host * tensor);
void trellis_mesh_free(trellis_mesh_host * mesh);

trellis_status trellis_sparse_downsample_mean_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

trellis_status trellis_sparse_spatial2channel_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

trellis_status trellis_sparse_channel2spatial_host(
    const trellis_sparse_tensor_host * input,
    const uint8_t * subdivision,
    int factor,
    trellis_sparse_tensor_host * output);

trellis_status trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    trellis_mesh_host * mesh_out);

typedef enum trellis_model_component {
    TRELLIS_COMPONENT_SPARSE_STRUCTURE_FLOW = 0,
    TRELLIS_COMPONENT_SPARSE_STRUCTURE_DECODER,
    TRELLIS_COMPONENT_SHAPE_SLAT_FLOW,
    TRELLIS_COMPONENT_TEX_SLAT_FLOW,
    TRELLIS_COMPONENT_SHAPE_SLAT_DECODER,
    TRELLIS_COMPONENT_TEX_SLAT_DECODER,
    TRELLIS_COMPONENT_DINOV3_IMAGE_ENCODER,
    TRELLIS_COMPONENT_OVOXEL_POSTPROCESS,
} trellis_model_component;

typedef struct trellis_component_status {
    trellis_model_component component;
    const char * name;
    bool implemented;
    const char * notes;
} trellis_component_status;

size_t trellis_component_status_count(void);
const trellis_component_status * trellis_component_status_at(size_t index);

typedef struct trellis_checkpoint_report {
    size_t expected_tensors;
    size_t actual_tensors;
    size_t found_tensors;
    size_t missing_tensors;
    size_t shape_mismatches;
    size_t dtype_mismatches;
    size_t extra_tensors;
    uint64_t expected_elements;
    uint64_t expected_bytes;
    char first_issue[256];
} trellis_checkpoint_report;

void trellis_checkpoint_report_clear(trellis_checkpoint_report * report);

trellis_status trellis_ss_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_shape_slat_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_ss_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_shape_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_make_model_path(
    const char * model_dir,
    const char * relative_path,
    char * dst,
    size_t dst_size);

#ifdef __cplusplus
}
#endif

#endif
