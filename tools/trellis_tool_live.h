#ifndef TRELLIS2_C_TOOLS_TRELLIS_TOOL_LIVE_H
#define TRELLIS2_C_TOOLS_TRELLIS_TOOL_LIVE_H

#include "trellis.h"

#include <stdint.h>

typedef struct trellis_tool_stage1_frame {
    int step;
    int steps;
    float t;
    float t_prev;
    int resolution;
    int64_t n_coords;
    const int32_t * coords_xyz;
} trellis_tool_stage1_frame;

typedef int (*trellis_tool_stage1_frame_callback)(
    const trellis_tool_stage1_frame * frame,
    void * user_data);

typedef struct trellis_tool_stage1_weights {
    trellis_tensor_store dino_store;
    trellis_dino_vit_weights dino;
    int has_dino;
    trellis_tensor_store flow_store;
    trellis_dit_flow_weights flow;
    int has_flow;
    trellis_tensor_store decoder_store;
    trellis_ss_decoder_weights decoder;
    int has_decoder;
} trellis_tool_stage1_weights;

int trellis_tool_stage1_weights_load(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * dino_dir,
    trellis_tool_stage1_weights * weights);

void trellis_tool_stage1_weights_free(trellis_tool_stage1_weights * weights);

typedef struct trellis_tool_stage1_image_options {
    const char * model_dir;
    const char * dino_dir;
    const char * image_path;
    const char * input_cond_path;
    const char * input_neg_cond_path;
    const char * input_latent_path;
    const char * out_dir;
    const char * dump_dir;
    int latent_size;
    int steps;
    int cond_resolution;
    int sparse_resolution;
    uint32_t seed;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    float voxel_threshold;
    const trellis_tool_stage1_weights * weights;
} trellis_tool_stage1_image_options;

typedef struct trellis_tool_stage1_result {
    int32_t * coords_bxyz;
    int64_t n_coords;
    int resolution;
    float * cond;
    int cond_tokens;
} trellis_tool_stage1_result;

void trellis_tool_stage1_result_free(trellis_tool_stage1_result * result);

int trellis_tool_run_stage1_image(
    const trellis_cuda_context * cuda,
    const trellis_tool_stage1_image_options * options,
    trellis_tool_stage1_frame_callback frame_callback,
    void * frame_user_data,
    trellis_tool_stage1_result * result);

typedef struct trellis_tool_stage2_weights {
    trellis_tensor_store flow_store;
    trellis_dit_flow_weights flow;
    int has_flow;
    trellis_tensor_store decoder_store;
    trellis_shape_decoder_weights decoder;
    int has_decoder;
    float slat_mean[32];
    float slat_std[32];
    int has_normalization;
} trellis_tool_stage2_weights;

int trellis_tool_stage2_weights_load(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    int resolution,
    const char * flow_override_path,
    const char * decoder_override_path,
    trellis_tool_stage2_weights * weights);

void trellis_tool_stage2_weights_free(trellis_tool_stage2_weights * weights);

typedef struct trellis_tool_stage2_live_options {
    const char * model_dir;
    const char * flow_override_path;
    const char * decoder_override_path;
    const int32_t * coords_bxyz;
    int64_t n_coords;
    const float * cond;
    int cond_tokens;
    const float * neg_cond;
    const float * noise;
    uint32_t noise_seed;
    const char * source;
    const char * display;
    const char * xauthority;
    int device;
    int resolution;
    int steps;
    float rescale_t;
    float guidance_strength;
    float guidance_rescale;
    float guidance_min;
    float guidance_max;
    int decode_initial;
    int decode_final;
    int width;
    int height;
    int max_faces;
    int mesh_chunk_faces;
    const char * mesh_upload_mode;
    const char * mesh_style;
    float hold;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    int emulate_bf16_blocks;
    int use_ggml_flash_attn;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    int use_existing_window;
    const trellis_cuda_context * cuda;
    const trellis_tool_stage2_weights * weights;
} trellis_tool_stage2_live_options;

int trellis_tool_run_stage2_mesh_live(const trellis_tool_stage2_live_options * options);

#endif
