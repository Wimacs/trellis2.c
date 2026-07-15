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

#ifndef TRELLIS_DEFAULT_GGML_BACKEND
#define TRELLIS_DEFAULT_GGML_BACKEND "cuda"
#endif

#ifndef TRELLIS_DEFAULT_BACKEND
#define TRELLIS_DEFAULT_BACKEND TRELLIS_DEFAULT_GGML_BACKEND
#endif

#ifndef TRELLIS_HAS_VKMESH_C_API
#define TRELLIS_HAS_VKMESH_C_API 0
#endif

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

/* Initializes process-wide runtime services. This function is thread-safe and
 * idempotent; applications may call it explicitly during startup. Public
 * pipeline and backend entry points also call it defensively. */
void trellis_runtime_init(void);

typedef enum trellis_backend_kind {
    TRELLIS_BACKEND_CPU = 0,
    TRELLIS_BACKEND_CUDA = 1,
    TRELLIS_BACKEND_VULKAN = 2,
} trellis_backend_kind;

typedef enum trellis_sparse_backend_kind {
    TRELLIS_SPARSE_BACKEND_CUDA = 0,
    TRELLIS_SPARSE_BACKEND_CPU = 1,
    TRELLIS_SPARSE_BACKEND_VULKAN = 2,
} trellis_sparse_backend_kind;

typedef struct trellis_backend_context {
    ggml_backend_t backend;
    trellis_backend_kind kind;
    int device;
} trellis_backend_context;

typedef trellis_backend_context trellis_cuda_context;

const char * trellis_backend_kind_name(trellis_backend_kind kind);
trellis_status trellis_backend_kind_from_name(const char * name, trellis_backend_kind * kind_out);
trellis_status trellis_backend_init(trellis_backend_context * ctx, trellis_backend_kind kind, int device);
void trellis_backend_free(trellis_backend_context * ctx);
ggml_gallocr_t trellis_backend_new_graph_allocator(const trellis_backend_context * ctx);
trellis_status trellis_backend_compute_graph(const trellis_backend_context * ctx, struct ggml_cgraph * graph);

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
    TRELLIS_DTYPE_C64,
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
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors);

trellis_status trellis_tensor_store_load_safetensors(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors);

typedef struct trellis_tensor_store_load_progress {
    const char * path;
    const char * tensor_name;
    size_t tensor_index;
    size_t tensor_count;
    uint64_t bytes_loaded;
    uint64_t total_bytes;
} trellis_tensor_store_load_progress;

typedef void (*trellis_tensor_store_load_progress_callback)(
    const trellis_tensor_store_load_progress * progress,
    void * user_data);

trellis_status trellis_tensor_store_load_safetensors_f32_ex(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data);

trellis_status trellis_tensor_store_load_safetensors_ex(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data);

trellis_status trellis_tensor_store_load_gguf(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * gguf_path,
    size_t * loaded_tensors);

typedef enum trellis_log_level {
    TRELLIS_LOG_DEBUG = 0,
    TRELLIS_LOG_INFO = 1,
    TRELLIS_LOG_WARN = 2,
    TRELLIS_LOG_ERROR = 3,
} trellis_log_level;

typedef struct trellis_model_load_result {
    size_t tensors;
    uint64_t bytes;
    double seconds;
} trellis_model_load_result;

void trellis_set_verbose(int verbose);
void trellis_log(trellis_log_level level, const char * fmt, ...);

void trellis_progress_bytes(
    const char * label,
    int step,
    int steps,
    uint64_t bytes_processed,
    uint64_t bytes_total,
    int64_t elapsed_us);

void trellis_progress_steps(
    const char * label,
    int step,
    int steps,
    int64_t step_us,
    const char * detail);

typedef struct trellis_progress_step_event {
    const char * label;
    int step;
    int steps;
    int64_t step_us;
    const char * detail;
} trellis_progress_step_event;

typedef void (*trellis_progress_step_callback)(
    const trellis_progress_step_event * event,
    void * user_data);

void trellis_set_progress_step_callback(
    trellis_progress_step_callback callback,
    void * user_data);

int trellis_load_tensor_store_f32(
    const trellis_backend_context * backend,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_model_load_result * result);

int trellis_load_tensor_store(
    const trellis_backend_context * backend,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_model_load_result * result);

#define TRELLIS_DEBUG(...) trellis_log(TRELLIS_LOG_DEBUG, __VA_ARGS__)
#define TRELLIS_INFO(...) trellis_log(TRELLIS_LOG_INFO, __VA_ARGS__)
#define TRELLIS_WARN(...) trellis_log(TRELLIS_LOG_WARN, __VA_ARGS__)
#define TRELLIS_ERROR(...) trellis_log(TRELLIS_LOG_ERROR, __VA_ARGS__)

#define TRELLIS_DIT_FLOW_BLOCKS 30

struct trellis_ggml_attention_policy;

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

trellis_status trellis_tex_slat_flow_bind_weights(
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

struct ggml_tensor * trellis_dit_flow_forward_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * timesteps,
    struct ggml_tensor * context,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dit_flow_weights * weights,
    const struct trellis_ggml_attention_policy * attention_policy);

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
    const trellis_backend_context * backend,
    int batch,
    int latent_size,
    float ** logits_out,
    int * output_size);

#define TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS 5
#define TRELLIS_SPARSE_UNET_VAE_DECODER_MAX_BLOCKS 16
#define TRELLIS_SPARSE_UNET_VAE_DECODER_UP_LEVELS (TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS - 1)
#define TRELLIS_SHAPE_DECODER_LEVELS TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS
#define TRELLIS_SHAPE_DECODER_MAX_BLOCKS TRELLIS_SPARSE_UNET_VAE_DECODER_MAX_BLOCKS

typedef struct trellis_sparse_unet_vae_decoder_convnext_block_weights {
    const float * conv_w;
    const float * conv_b;
    const float * norm_gamma;
    const float * norm_beta;
    const float * mlp0_w;
    const float * mlp0_b;
    const float * mlp2_w;
    const float * mlp2_b;
    int channels;
} trellis_sparse_unet_vae_decoder_convnext_block_weights;

typedef trellis_sparse_unet_vae_decoder_convnext_block_weights
    trellis_sparse_unet_convnext_block_weights;

typedef struct trellis_sparse_unet_vae_decoder_c2s_block_weights {
    const float * norm1_gamma;
    const float * norm1_beta;
    const float * conv1_w;
    const float * conv1_b;
    const float * conv2_w;
    const float * conv2_b;
    const float * to_subdiv_w;
    const float * to_subdiv_b;
    int in_channels;
    int out_channels;
} trellis_sparse_unet_vae_decoder_c2s_block_weights;

typedef struct trellis_sparse_unet_vae_decoder_weights {
    const float * from_latent_w;
    const float * from_latent_b;
    const float * output_w;
    const float * output_b;
    int latent_channels;
    int out_channels;
    int pred_subdiv;
    int levels;
    int channels[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS];
    int blocks_per_level[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS];
    trellis_sparse_unet_vae_decoder_convnext_block_weights
        blocks[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS][TRELLIS_SPARSE_UNET_VAE_DECODER_MAX_BLOCKS];
    trellis_sparse_unet_vae_decoder_c2s_block_weights
        up_blocks[TRELLIS_SPARSE_UNET_VAE_DECODER_UP_LEVELS];
} trellis_sparse_unet_vae_decoder_weights;

typedef trellis_sparse_unet_vae_decoder_convnext_block_weights
    trellis_shape_decoder_convnext_block_weights;
typedef trellis_sparse_unet_vae_decoder_c2s_block_weights
    trellis_shape_decoder_c2s_block_weights;
typedef trellis_sparse_unet_vae_decoder_weights
    trellis_shape_decoder_weights;

typedef struct trellis_sparse_c2s_guide_level {
    int32_t * coords_bxyz; /* [n_coords, 4] */
    int32_t * parent;      /* [n_coords] row in previous level */
    int32_t * subidx;      /* [n_coords] channel-to-spatial child index [0, 7] */
    void * device_map;     /* Optional backend-owned device-resident map. */
    trellis_sparse_backend_kind device_backend_kind;
    int device;
    int64_t n_coords;
} trellis_sparse_c2s_guide_level;

typedef struct trellis_sparse_c2s_guides {
    trellis_sparse_c2s_guide_level levels[TRELLIS_SPARSE_UNET_VAE_DECODER_UP_LEVELS];
    int n_levels;
} trellis_sparse_c2s_guides;

void trellis_sparse_c2s_guides_free(trellis_sparse_c2s_guides * guides);

typedef struct trellis_shape_decoder_debug_options {
    const char * dump_dir;
} trellis_shape_decoder_debug_options;

trellis_status trellis_sparse_unet_vae_decoder_bind_weights(
    trellis_tensor_store * store,
    int out_channels,
    bool pred_subdiv,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_shape_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_shape_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_flexi_dual_grid_vae_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_tex_slat_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_sparse_unet_vae_decoder_forward_f32_host(
    const trellis_sparse_unet_vae_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    const trellis_sparse_c2s_guides * guide_subs,
    trellis_sparse_c2s_guides * return_subs,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

typedef struct trellis_sparse_unet_vae_decoder_forward_options {
    trellis_sparse_backend_kind backend_kind;
    int device;
    int max_levels;
    void * sparse_backend;
    const trellis_sparse_c2s_guides * guide_subs;
    trellis_sparse_c2s_guides * return_subs;
} trellis_sparse_unet_vae_decoder_forward_options;

trellis_status trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
    const trellis_sparse_unet_vae_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    const trellis_sparse_unet_vae_decoder_forward_options * options,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

#define TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS 5
#define TRELLIS_SPARSE_UNET_VAE_ENCODER_MAX_BLOCKS 16
#define TRELLIS_SPARSE_UNET_VAE_ENCODER_DOWN_LEVELS (TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS - 1)

typedef trellis_sparse_unet_convnext_block_weights
    trellis_sparse_unet_vae_encoder_convnext_block_weights;

typedef struct trellis_sparse_unet_vae_encoder_s2c_block_weights {
    const float * norm1_gamma;
    const float * norm1_beta;
    const float * conv1_w;
    const float * conv1_b;
    const float * conv2_w;
    const float * conv2_b;
    int in_channels;
    int out_channels;
} trellis_sparse_unet_vae_encoder_s2c_block_weights;

typedef struct trellis_sparse_unet_vae_encoder_weights {
    const float * input_w;
    const float * input_b;
    const float * to_latent_w;
    const float * to_latent_b;
    int in_channels;
    int latent_channels;
    int levels;
    int channels[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS];
    int blocks_per_level[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS];
    trellis_sparse_unet_vae_encoder_convnext_block_weights
        blocks[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS][TRELLIS_SPARSE_UNET_VAE_ENCODER_MAX_BLOCKS];
    trellis_sparse_unet_vae_encoder_s2c_block_weights
        down_blocks[TRELLIS_SPARSE_UNET_VAE_ENCODER_DOWN_LEVELS];
} trellis_sparse_unet_vae_encoder_weights;

trellis_status trellis_sparse_unet_vae_encoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_encoder_weights * weights,
    char * first_issue,
    size_t first_issue_size);

typedef struct trellis_sparse_unet_vae_encoder_forward_options {
    trellis_sparse_backend_kind backend_kind;
    int device;
    int max_levels;
    void * sparse_backend;
    trellis_sparse_c2s_guides * return_subs;
} trellis_sparse_unet_vae_encoder_forward_options;

trellis_status trellis_sparse_unet_vae_encoder_forward_backend_f32_host(
    const trellis_sparse_unet_vae_encoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    const trellis_sparse_unet_vae_encoder_forward_options * options,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

/* CUDA implementation is provided by the CUDA sparse backend. */
trellis_status trellis_sparse_unet_vae_encoder_forward_f32_host(
    const trellis_sparse_unet_vae_encoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    trellis_sparse_c2s_guides * return_subs,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out);

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

struct ggml_tensor * trellis_dino_patch_embedding_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * image,
    const trellis_dino_vit_weights * weights);

struct ggml_tensor * trellis_dino_image_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * image,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dino_vit_weights * weights);

struct ggml_tensor * trellis_dino_vit_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * tokens,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dino_vit_weights * weights);

trellis_status trellis_dino_image_forward_f32_host(
    const trellis_backend_context * backend,
    const trellis_dino_vit_weights * weights,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    const float * cos_phase,
    const float * sin_phase,
    float ** tokens_out,
    int * n_tokens_out);

/* 2D rotary phase table for DINO image patch tokens. */
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

#define TRELLIS_BIREFNET_LAYERS 4

typedef struct trellis_birefnet_params {
    int image_size;
    int image_multiple;
    int image_w;
    int image_h;
    int embed_dim;
    int window_size;
    int layer_depths[TRELLIS_BIREFNET_LAYERS];
    int layer_heads[TRELLIS_BIREFNET_LAYERS];
    int layer_features[TRELLIS_BIREFNET_LAYERS];
} trellis_birefnet_params;

typedef struct trellis_birefnet_model {
    trellis_backend_context backend;
    trellis_tensor_store store;
    trellis_birefnet_params params;
} trellis_birefnet_model;

trellis_status trellis_birefnet_load_gguf(
    trellis_birefnet_model * model,
    const char * gguf_path);

trellis_status trellis_birefnet_load_gguf_with_backend(
    trellis_birefnet_model * model,
    const char * gguf_path,
    trellis_backend_kind backend_kind,
    int device);

void trellis_birefnet_free(trellis_birefnet_model * model);

trellis_status trellis_birefnet_compute_mask_u8(
    trellis_birefnet_model * model,
    const unsigned char * rgba,
    int width,
    int height,
    unsigned char ** mask_out);

typedef struct trellis_mesh_host {
    float * vertices; /* [n_vertices, 3] */
    float * vertex_colors; /* optional [n_vertices, 3] in linear 0..1 RGB */
    int32_t * faces;  /* [n_faces, 3] */
    int64_t n_vertices;
    int64_t n_faces;
} trellis_mesh_host;

void trellis_mesh_free(trellis_mesh_host * mesh);

/* Options matching o-voxel's mesh_to_flexible_dual_grid conversion. The
 * input mesh is expressed in the same coordinate system as the AABB. */
typedef struct trellis_flexible_dual_grid_options {
    int32_t grid_size[3];
    float aabb_min[3];
    float aabb_max[3];
    float face_weight;
    float boundary_weight;
    float regularization_weight;
} trellis_flexible_dual_grid_options;

typedef struct trellis_flexible_dual_grid {
    int32_t * coords;         /* [n, 4] batch,x,y,z; batch is zero */
    float * dual_vertices;    /* [n, 3], relative to aabb_min */
    uint8_t * intersected;    /* [n, 3] x/y/z edge-intersection flags */
    int64_t n;
} trellis_flexible_dual_grid;

/* Initializes the TRELLIS.2 texturing defaults: a 512^3 grid over
 * [-0.5, 0.5]^3 with QEF weights 1.0, 0.2 and 0.01. */
void trellis_flexible_dual_grid_options_default(
    trellis_flexible_dual_grid_options * options);

/* Converts a raw triangle mesh to the Flexible Dual Grid representation used
 * by FlexiDualGridVaeEncoder. dual_vertices use AABB-relative coordinates, so
 * the encoder-local feature for axis a is
 * dual_vertices[a] / voxel_size[a] - coords[a + 1]. */
trellis_status trellis_mesh_to_flexible_dual_grid_host(
    const float * vertices,
    int64_t n_vertices,
    const int32_t * faces,
    int64_t n_faces,
    const trellis_flexible_dual_grid_options * options,
    trellis_flexible_dual_grid * grid_out);

void trellis_flexible_dual_grid_free(trellis_flexible_dual_grid * grid);

typedef struct trellis_vkmesh_postprocess_options {
    int decimation_target;       /* default 1000000 */
    int no_simplify;             /* skip simplify passes, matching vkmesh --no-simplify */
    int run_degenerate_cleanup;  /* default 0, matching current TRELLIS pipeline */
    int simplify_steps;          /* 0 means vkmesh default/unbounded */
    int remesh;                  /* run CuMesh-style narrow-band dual-contouring remesh */
    int remesh_resolution;       /* default 1024 when remesh is enabled */
    float max_hole_perimeter;    /* default 0.03 */
    float degenerate_abs;        /* default 1e-24 */
    float degenerate_rel;        /* default 1e-12 */
    float min_component_area;    /* default 1e-5 */
    float lambda_edge_length;    /* default 1e-2 */
    float lambda_skinny;         /* default 1e-3 */
    float simplify_threshold;    /* default 1e-8 */
    float remesh_band;           /* default 1.0 */
    float remesh_project;        /* default 0.0 for TRELLIS.2 examples */
    int device;                  /* Vulkan physical-device index, default 0 */
    int gpu_workspace_budget_mib; /* 0=automatic, hard cap for vkmesh VkDeviceMemory workspace */
} trellis_vkmesh_postprocess_options;

trellis_status trellis_vkmesh_postprocess(
    const trellis_mesh_host * mesh,
    trellis_mesh_host * mesh_out,
    trellis_mesh_host * projection_mesh_out,
    const trellis_vkmesh_postprocess_options * options);

/* Extracts a mesh from FlexiDualGrid decoder logits. */
trellis_status trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    trellis_mesh_host * mesh_out);

typedef struct trellis_image_to_gltf_options {
    const char * model_dir;
    const char * dino_dir;
    const char * birefnet_path;
    const char * image_path;
    const char * gltf_path;
    const char * flow_override_path;
    const char * decoder_override_path;
    const char * backend;
    const char * pipeline_type;
    int device;
    int sparse_structure_steps;
    int structured_latent_steps;
    int latent_size;
    int resolution;
    int cond_resolution;
    int sparse_resolution;
    uint32_t seed;
    uint32_t noise_seed;
    float rescale_t;
    float guidance_strength;
    float guidance_rescale;
    float guidance_min;
    float guidance_max;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    int emulate_bf16_blocks;
    int use_ggml_flash_attn;
    int no_ggml_flash_attn;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    int max_num_tokens;
    int texture_size;
    int mesh_postprocess;
    int mesh_postprocess_no_simplify;
    int mesh_postprocess_decimation_target;
    int mesh_remesh;
    int mesh_remesh_resolution;
    float mesh_remesh_band;
    float mesh_remesh_project;
    int model_cache;
    int model_cache_budget_mib;
    const char * vkmesh_path;
    int vkmesh_gpu_workspace_budget_mib; /* 0=automatic */
} trellis_image_to_gltf_options;

/* Optional image-to-glTF features and persisted intermediate artifacts.  This
 * is deliberately separate from trellis_image_to_gltf_options so callers
 * compiled against the original, unversioned options layout remain ABI-safe. */
typedef struct trellis_image_to_gltf_feature_options {
    size_t struct_size;           /* set to sizeof(trellis_image_to_gltf_feature_options) */
    uint32_t version;             /* set to TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION */
    int shape_only;               /* skip texture flow/decoder and export an untextured mesh */
    const char * prepared_image_output_path; /* optional RGBA PNG used for conditioning */
    const char * shape_latent_output_path; /* optional reusable TRELLIS.2 shape SLat cache */
} trellis_image_to_gltf_feature_options;

#define TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION 1u
#define TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_V1_SIZE \
    (offsetof(trellis_image_to_gltf_feature_options, shape_latent_output_path) + \
     sizeof(const char *))
#define TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_INIT \
    { sizeof(trellis_image_to_gltf_feature_options), \
      TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION, 0, NULL, NULL }

typedef struct trellis_pixal3d_options {
    size_t struct_size;           /* set to sizeof(trellis_pixal3d_options) */
    const char * naf_path;        /* NULL/empty: search model_dir/ckpts automatically */
    float camera_angle_x;         /* horizontal FOV in radians; <=0 uses 0.857556 */
    float camera_distance;        /* <=0 derives from FOV/mesh_scale; >0 is explicit */
    float mesh_scale;             /* <=0 uses 1 */
} trellis_pixal3d_options;

#define TRELLIS_PIXAL3D_OPTIONS_V1_SIZE \
    (offsetof(trellis_pixal3d_options, mesh_scale) + sizeof(float))
#define TRELLIS_PIXAL3D_OPTIONS_INIT \
    { sizeof(trellis_pixal3d_options), NULL, 0.0f, 0.0f, 0.0f }

/* Legacy entry point: Pixal3D uses the default camera and automatic NAF lookup. */
trellis_status trellis_pipeline_image_to_gltf(const trellis_image_to_gltf_options * options);

/* Extended entry point for an explicit NAF path or Pixal3D camera parameters. */
trellis_status trellis_pipeline_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options);

/* Model-pinned entry points used by per-family CLIs.  They reject a model
 * package from another family before image loading or backend initialization. */
trellis_status trellis_pipeline_trellis2_image_to_gltf(
    const trellis_image_to_gltf_options * options);

trellis_status trellis_pipeline_trellis2_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_image_to_gltf_feature_options * feature_options);

trellis_status trellis_pipeline_pixal3d_image_to_gltf(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options);

trellis_status trellis_pipeline_pixal3d_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options,
    const trellis_image_to_gltf_feature_options * feature_options);

typedef struct trellis_mesh_texturing_options {
    size_t struct_size;              /* set to sizeof(trellis_mesh_texturing_options) */
    const char * model_dir;          /* TRELLIS.2 model package root */
    const char * dino_dir;           /* DINOv3 image encoder directory */
    const char * input_path;         /* source .glb/.gltf triangle mesh */
    const char * image_path;         /* material reference image */
    const char * output_path;        /* self-contained textured .glb */
    const char * birefnet_path;      /* optional opaque-image foreground model override */
    const char * encoder_path;       /* optional shape encoder checkpoint override */
    const char * texture_flow_path;  /* optional texture SLat flow override */
    const char * texture_decoder_path; /* optional texture decoder override */
    const char * backend;            /* NULL uses the backend compiled into this binary */
    int device;                      /* backend device index, default 0 */
    int resolution;                  /* Flexible Dual Grid resolution: 512 or 1024 */
    int texture_size;                /* output PBR texture edge, default 1024 */
    int steps;                       /* texture flow Euler steps, default 12 */
    uint32_t seed;                   /* texture latent noise seed, default 42 */
    int flow_blocks_override;        /* debug: -1 runs all texture flow blocks */
    int flow_block_parts_override;   /* debug: -1 runs each complete block */
    int flow_no_rope;                /* debug sparse-RoPE bypass */
    int emulate_bf16_blocks;         /* debug explicit BF16 activation round trips */
    int no_flash_attn;               /* debug explicit-attention fallback */
    int image_prepared;              /* image_path is the final condition; skip BiRefNet */
    const char * shape_latent_path;   /* optional reusable shape SLat cache input */
    const char * shape_latent_output_path; /* optional cache written after mesh encoding */
} trellis_mesh_texturing_options;

#define TRELLIS_MESH_TEXTURING_OPTIONS_V1_SIZE \
    (offsetof(trellis_mesh_texturing_options, no_flash_attn) + sizeof(int))
#define TRELLIS_MESH_TEXTURING_OPTIONS_V2_SIZE \
    (offsetof(trellis_mesh_texturing_options, image_prepared) + sizeof(int))
#define TRELLIS_MESH_TEXTURING_OPTIONS_V3_SIZE \
    (offsetof(trellis_mesh_texturing_options, shape_latent_output_path) + sizeof(const char *))
#define TRELLIS_MESH_TEXTURING_OPTIONS_INIT \
    { sizeof(trellis_mesh_texturing_options), NULL, NULL, NULL, NULL, NULL, \
      NULL, NULL, NULL, NULL, NULL, 0, 512, 1024, 12, 42u, -1, -1, 0, 0, 0, 0, NULL, NULL }

/* TRELLIS.2-only existing-mesh material generation. The input geometry is
 * aligned to a compatible cached shape SLat when supplied; otherwise it is
 * normalized, converted to a Flexible Dual Grid, and encoded. The shape SLat
 * then conditions the released texture flow/decoder. */
trellis_status trellis_pipeline_trellis2_texture_mesh(
    const trellis_mesh_texturing_options * options);

typedef enum trellis_mesh_segmentation_small_part_mode {
    /* Leave disconnected micro-shells unchanged. */
    TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP = 0,
    /* Attach automatically detected micro-shells to the nearest retained part. */
    TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE = 1,
    /* Remove automatically detected micro-shell faces from the output GLB. */
    TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD = 2,
} trellis_mesh_segmentation_small_part_mode;

typedef struct trellis_mesh_segmentation_options {
    size_t struct_size;              /* set to sizeof(trellis_mesh_segmentation_options) */
    const char * model_dir;          /* base TRELLIS.2 model package root */
    const char * segmentation_model_dir; /* SegviGen Full model package root */
    const char * dino_dir;           /* DINOv3 directory; NULL only with segmentation_latent_path */
    const char * input_path;         /* source .glb/.gltf triangle mesh */
    const char * output_path;        /* self-contained multi-part .glb */
    const char * condition_image_path; /* optional prepared/render reference image */
    const char * birefnet_path;      /* optional foreground model override */
    const char * shape_encoder_path; /* optional base shape encoder override */
    const char * texture_encoder_path; /* optional base texture encoder override */
    const char * segmentation_flow_path; /* optional SegviGen flow override */
    const char * shape_decoder_path; /* optional base shape decoder override */
    const char * texture_decoder_path; /* optional base texture decoder override */
    const char * shape_latent_path;  /* optional trusted/debug TSLAT cache input */
    const char * texture_latent_path; /* optional trusted/debug TSLAT cache input */
    const char * segmentation_latent_path; /* optional generated label TSLAT input */
    const char * rendered_condition_output_path; /* optional automatic render dump */
    const char * shape_latent_output_path; /* optional shape encoder cache output */
    const char * texture_latent_output_path; /* optional texture encoder cache output */
    const char * segmentation_latent_output_path; /* optional generated label SLat dump */
    const char * backend;            /* NULL uses the backend compiled into this binary */
    int device;                      /* backend device index, default 0 */
    int resolution;                  /* SegviGen Full is pinned to 512 */
    int steps;                       /* paired flow Euler steps, default 12 */
    uint32_t seed;                   /* generated segmentation latent seed, default 42 */
    int min_component_faces;         /* absorb islands; micro-shell face threshold, default 16 */
    int min_palette_voxels;          /* ignore smaller decoded color bins, default 16 */
    float palette_merge_distance;    /* RGB Euclidean radius, default 32/255 */
    int condition_image_prepared;    /* skip BiRefNet for condition_image_path */
    int flow_blocks_override;        /* debug: -1 runs all flow blocks */
    int flow_block_parts_override;   /* debug: -1 runs each complete block */
    int flow_no_rope;                /* debug sparse-RoPE bypass */
    int emulate_bf16_blocks;         /* debug explicit BF16 activation round trips */
    int no_flash_attn;               /* debug explicit-attention fallback */
    trellis_mesh_segmentation_small_part_mode small_part_mode; /* default KEEP; V2 */
} trellis_mesh_segmentation_options;

#define TRELLIS_MESH_SEGMENTATION_OPTIONS_V1_SIZE \
    (offsetof(trellis_mesh_segmentation_options, no_flash_attn) + sizeof(int))
#define TRELLIS_MESH_SEGMENTATION_OPTIONS_V2_SIZE \
    (offsetof(trellis_mesh_segmentation_options, small_part_mode) + \
     sizeof(trellis_mesh_segmentation_small_part_mode))
#define TRELLIS_MESH_SEGMENTATION_OPTIONS_INIT \
    { sizeof(trellis_mesh_segmentation_options), NULL, NULL, NULL, NULL, NULL, \
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
      NULL, NULL, NULL, 0, 512, 12, 42u, 16, 16, (32.0f / 255.0f), 0, -1, -1, 0, 0, 0, \
      TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP }

/* SegviGen Full automatic mesh decomposition. The model predicts categorical
 * voxel colors from a paired generated/source texture SLat. Colors are
 * transferred to faces, separated into connected physical instances, and
 * written as one independently selectable glTF node+mesh per part. */
trellis_status trellis_pipeline_trellis2_segment_mesh(
    const trellis_mesh_segmentation_options * options);

typedef struct trellis_tokenskin_rig_options {
    size_t struct_size;              /* set to sizeof(trellis_tokenskin_rig_options) */
    const char * model_dir;          /* TokenSkin model package root */
    const char * input_path;         /* input .glb or .gltf mesh */
    const char * output_path;        /* output self-contained rigged .glb */
    const char * backend;            /* NULL uses the backend compiled into this binary */
    int device;                      /* backend device index, default 0 */
    uint32_t seed;                   /* preprocessing and autoregressive sampling seed */
    int sample_count;                /* surface samples >=2048, reference default 54000 */
    int max_length;                  /* Qwen total length including 512 mesh tokens, default 2048 */
    int top_k;                       /* released checkpoint default 10 */
    float top_p;                     /* reference demo default 0.95 */
    float temperature;               /* released checkpoint default 1.5 */
    float repetition_penalty;        /* reference demo default 2 */
    int num_beams;                   /* reference demo default 10; supported range 1..16 */
    int official_eos_compat;         /* reproduce the released SkinTokens EOS/FSQ alias */
    int no_flash_attn;               /* debug fallback; model package defaults to FlashAttention */
} trellis_tokenskin_rig_options;

#define TRELLIS_TOKENSKIN_MAX_BEAMS 16

#define TRELLIS_TOKENSKIN_RIG_OPTIONS_V1_SIZE \
    (offsetof(trellis_tokenskin_rig_options, no_flash_attn) + sizeof(int))
#define TRELLIS_TOKENSKIN_RIG_OPTIONS_INIT \
    { sizeof(trellis_tokenskin_rig_options), NULL, NULL, NULL, NULL, 0, 1u, \
      54000, 2048, 10, 0.95f, 1.5f, 2.0f, 10, 1, 0 }

/* TokenSkin-only mesh rigging entry point. It rejects any other model family
 * or task before parsing the input mesh or initializing a GPU backend. */
trellis_status trellis_pipeline_tokenskin_rig(
    const trellis_tokenskin_rig_options * options);

#include "trellis_ggml_layers.h"
#include "trellis_flow_sampler.h"

typedef enum trellis_model_component {
    TRELLIS_COMPONENT_SPARSE_STRUCTURE_FLOW = 0,
    TRELLIS_COMPONENT_SPARSE_STRUCTURE_DECODER,
    TRELLIS_COMPONENT_SHAPE_SLAT_FLOW,
    TRELLIS_COMPONENT_TEX_SLAT_FLOW,
    TRELLIS_COMPONENT_SHAPE_SLAT_DECODER,
    TRELLIS_COMPONENT_TEX_SLAT_DECODER,
    TRELLIS_COMPONENT_DINOV3_IMAGE_ENCODER,
    TRELLIS_COMPONENT_BIREFNET_BACKGROUND_REMOVAL,
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

trellis_status trellis_make_model_path(
    const char * model_dir,
    const char * relative_path,
    char * dst,
    size_t dst_size);

#ifdef __cplusplus
}
#endif

#endif
