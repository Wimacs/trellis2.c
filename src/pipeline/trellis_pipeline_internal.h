#ifndef TRELLIS2_C_SRC_TRELLIS_PIPELINE_INTERNAL_H
#define TRELLIS2_C_SRC_TRELLIS_PIPELINE_INTERNAL_H

#include "trellis.h"

typedef struct trellis_sparse_structure_result {
    int32_t * coords_bxyz; /* [n_coords, 4] = batch, x, y, z */
    int64_t n_coords;
    int resolution;
    float * cond;          /* [cond_tokens, 1024] */
    int cond_tokens;
} trellis_sparse_structure_result;

typedef struct trellis_sparse_structure_options {
    const char * model_dir;
    const char * dino_dir;
    const char * image_path;
    int latent_size;
    int steps;
    int cond_resolution;
    int sparse_resolution;
    uint32_t seed;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    float voxel_threshold;
    const trellis_cuda_context * cuda;
} trellis_sparse_structure_options;

void trellis_sparse_structure_result_free(trellis_sparse_structure_result * result);

trellis_status trellis_pipeline_run_sparse_structure(
    const trellis_sparse_structure_options * options,
    trellis_sparse_structure_result * result);

typedef struct trellis_structured_latent {
    int32_t * coords_bxyz; /* [n_coords, 4] */
    int64_t n_coords;
    int resolution;
    float * feats;         /* denormalized [n_coords, channels] */
    int channels;
} trellis_structured_latent;

typedef trellis_structured_latent trellis_shape_latent;

void trellis_structured_latent_free(trellis_structured_latent * latent);
void trellis_shape_latent_free(trellis_shape_latent * latent);

typedef struct trellis_structured_latent_options {
    const char * model_dir;
    const char * flow_override_path;
    trellis_model_component flow_component;
    const char * label;
    const char * normalization_key;
    const int32_t * coords_bxyz;
    int64_t n_coords;
    const float * cond;
    int cond_tokens;
    const float * neg_cond;
    const float * noise;
    const float * concat_cond;
    int concat_channels;
    const float * concat_mean;
    const float * concat_std;
    uint32_t noise_seed;
    int resolution;
    int steps;
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
    const trellis_cuda_context * cuda;
} trellis_structured_latent_options;

trellis_status trellis_pipeline_run_structured_latent(
    const trellis_structured_latent_options * options,
    trellis_structured_latent * latent_out);

typedef struct trellis_pipeline_mesh_options {
    const char * model_dir;
    const char * decoder_override_path;
    const trellis_structured_latent * latent;
    int resolution;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    const trellis_cuda_context * cuda;
} trellis_pipeline_mesh_options;

trellis_status trellis_pipeline_decode_shape_latent_mesh(
    const trellis_pipeline_mesh_options * options,
    trellis_sparse_c2s_guides * subs_out,
    trellis_mesh_host * mesh_out);

typedef struct trellis_pbr_voxels {
    int32_t * coords_bxyz; /* [n_coords, 4] */
    float * attrs;         /* [n_coords, channels], usually RGB metallic roughness alpha */
    int64_t n_coords;
    int channels;
    int resolution;
} trellis_pbr_voxels;

void trellis_pbr_voxels_free(trellis_pbr_voxels * voxels);

typedef struct trellis_pipeline_texture_options {
    const char * model_dir;
    const trellis_structured_latent * latent;
    const trellis_sparse_c2s_guides * guide_subs;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    const trellis_cuda_context * cuda;
} trellis_pipeline_texture_options;

trellis_status trellis_pipeline_decode_texture_latent_voxels(
    const trellis_pipeline_texture_options * options,
    trellis_pbr_voxels * voxels_out);

trellis_status trellis_pipeline_apply_pbr_voxels_to_mesh(
    const trellis_pbr_voxels * voxels,
    trellis_mesh_host * mesh);

trellis_status trellis_pipeline_write_obj(const char * path, const trellis_mesh_host * mesh);

trellis_status trellis_pipeline_write_gltf(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size);

#endif
