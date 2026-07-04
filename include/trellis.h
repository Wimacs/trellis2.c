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
    const trellis_cuda_context * cuda,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data);

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

int trellis_load_tensor_store_f32(
    const trellis_cuda_context * cuda,
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
    const trellis_cuda_context * cuda,
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
    const trellis_cuda_context * cuda,
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

typedef struct trellis_mesh_host {
    float * vertices; /* [n_vertices, 3] */
    float * vertex_colors; /* optional [n_vertices, 3] in linear 0..1 RGB */
    int32_t * faces;  /* [n_faces, 3] */
    int64_t n_vertices;
    int64_t n_faces;
} trellis_mesh_host;

void trellis_mesh_free(trellis_mesh_host * mesh);

/* Extracts a mesh from FlexiDualGrid decoder logits. */
trellis_status trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    trellis_mesh_host * mesh_out);

typedef struct trellis_image_to_obj_options {
    const char * model_dir;
    const char * dino_dir;
    const char * image_path;
    const char * obj_path;
    const char * gltf_path;
    const char * flow_override_path;
    const char * decoder_override_path;
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
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    int texture_size;
} trellis_image_to_obj_options;

trellis_status trellis_pipeline_image_to_obj(const trellis_image_to_obj_options * options);

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
