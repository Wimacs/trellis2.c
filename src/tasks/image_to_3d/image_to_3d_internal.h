#ifndef TRELLIS2_C_IMAGE_TO_3D_INTERNAL_H
#define TRELLIS2_C_IMAGE_TO_3D_INTERNAL_H

#include "trellis.h"
#include "trellis_ggml_layers.h"
#include "gltf_axes.h"
#include "../../architectures/dit_flow/projected_dit_flow.h"
#include "../../architectures/pixal_naf/pixal_naf.h"

typedef struct trellis_prepared_condition_image {
    char source_path[4096];
    char converted_path[4096];
    char foreground_path[4096];
} trellis_prepared_condition_image;

trellis_status trellis_pipeline_prepare_condition_image(
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * birefnet_path,
    trellis_backend_kind backend_kind,
    int device,
    int require_foreground,
    trellis_prepared_condition_image * prepared_out);

trellis_status trellis_pipeline_prepare_condition_image_for_image_to_gltf(
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * birefnet_path,
    trellis_backend_kind backend_kind,
    int device,
    int require_foreground,
    trellis_prepared_condition_image * prepared_out);

void trellis_pipeline_prepared_condition_image_free(
    trellis_prepared_condition_image * prepared);

typedef enum trellis_pipeline_cache_entry_kind {
    TRELLIS_PIPELINE_CACHE_ENTRY_DINO = 0,
    TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW = 1,
    TRELLIS_PIPELINE_CACHE_ENTRY_SS_DECODER = 2,
    TRELLIS_PIPELINE_CACHE_ENTRY_SPARSE_UNET_DECODER = 3,
    TRELLIS_PIPELINE_CACHE_ENTRY_PIXAL_NAF = 4,
} trellis_pipeline_cache_entry_kind;

typedef struct trellis_pipeline_cache_entry {
    const char * name;
    trellis_pipeline_cache_entry_kind kind;
    int loaded;
    int pinned;
    uint64_t last_used;
    size_t budget_bytes;
    char path[4096];
    trellis_tensor_store store;
    union {
        trellis_dino_vit_weights dino;
        trellis_dit_flow_model flow_model;
        trellis_ss_decoder_weights ss_decoder;
        trellis_sparse_unet_vae_decoder_weights sparse_unet_decoder;
        trellis_pixal_naf_ggml_weights pixal_naf;
    } weights;
} trellis_pipeline_cache_entry;

typedef struct trellis_pipeline_model_cache {
    const trellis_backend_context * backend;
    const trellis_backend_context * decoder_weight_backend;
    trellis_backend_context cpu_decoder_weight_backend;
    int owns_cpu_decoder_weight_backend;
    size_t budget_bytes;
    size_t used_budget_bytes;
    uint64_t clock;

    trellis_pipeline_cache_entry dino;
    trellis_pipeline_cache_entry sparse_structure_flow;
    trellis_pipeline_cache_entry sparse_structure_decoder;
    trellis_pipeline_cache_entry shape_flow_512;
    trellis_pipeline_cache_entry shape_flow_1024;
    trellis_pipeline_cache_entry texture_flow_512;
    trellis_pipeline_cache_entry texture_flow_1024;
    trellis_pipeline_cache_entry shape_decoder;
    trellis_pipeline_cache_entry texture_decoder;
    trellis_pipeline_cache_entry pixal_naf;
} trellis_pipeline_model_cache;

trellis_status trellis_pipeline_model_cache_init(
    trellis_pipeline_model_cache * cache,
    const trellis_backend_context * backend,
    int use_cpu_decoder_weight_backend,
    size_t budget_bytes);

void trellis_pipeline_model_cache_free(trellis_pipeline_model_cache * cache);
void trellis_pipeline_model_cache_unpin_all(trellis_pipeline_model_cache * cache);
void trellis_pipeline_model_cache_evict_unpinned(trellis_pipeline_model_cache * cache);

trellis_status trellis_pipeline_model_cache_get_dino(
    trellis_pipeline_model_cache * cache,
    const char * dino_dir,
    const trellis_dino_vit_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_sparse_structure_flow(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const trellis_dit_flow_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_sparse_structure_flow_model(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_dit_flow_model ** model_out);

trellis_status trellis_pipeline_model_cache_get_sparse_structure_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_ss_decoder_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_slat_flow(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    const char * label,
    const trellis_dit_flow_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_slat_flow_model(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    const char * label,
    const trellis_dit_flow_model ** model_out);

trellis_status trellis_pipeline_model_cache_get_shape_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_sparse_unet_vae_decoder_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_texture_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_sparse_unet_vae_decoder_weights ** weights_out);

trellis_status trellis_pipeline_model_cache_get_pixal_naf(
    trellis_pipeline_model_cache * cache,
    const char * path,
    const trellis_pixal_naf_ggml_weights ** weights_out);

typedef struct trellis_sparse_structure_result {
    int32_t * coords_bxyz; /* [n_coords, 4] = batch, x, y, z */
    int64_t n_coords;
    int resolution;
    float * cond;          /* [cond_tokens, 1024] */
    int cond_tokens;
    int projected_conditioning;
} trellis_sparse_structure_result;

typedef struct trellis_sparse_structure_options {
    const char * model_dir;
    const char * flow_path;
    const char * decoder_path;
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
    int projected_conditioning;
    int emulate_bf16_blocks;
    trellis_ggml_attention_policy attention_policy;
    /* Legacy fallback when attention_policy was left zero-initialized. */
    int use_ggml_flash_attn;
    float voxel_threshold;
    float camera_angle_x;
    float camera_distance;
    float mesh_scale;
    const trellis_backend_context * backend;
    const trellis_cuda_context * cuda;
    trellis_pipeline_model_cache * cache;
} trellis_sparse_structure_options;

void trellis_sparse_structure_result_free(trellis_sparse_structure_result * result);

trellis_status trellis_pipeline_run_sparse_structure(
    const trellis_sparse_structure_options * options,
    trellis_sparse_structure_result * result);

typedef struct trellis_image_condition_result {
    float * cond; /* [cond_tokens, 1024] */
    int cond_tokens;
    int resolution;
    float * projected; /* optional [projected_tokens, projected_channels] */
    int64_t projected_tokens;
    int projected_channels;
} trellis_image_condition_result;

typedef struct trellis_image_condition_options {
    const char * model_dir;
    const char * dino_dir;
    const char * image_path;
    const char * naf_path;
    int cond_resolution;
    /* Zero keeps the image-to-3D family default. Mesh texturing uses the
     * released pipeline's alpha > 204, crop scale 1.0 profile. */
    int foreground_alpha_threshold;
    float foreground_crop_scale;
    int projection_grid_resolution;
    int projection_channels;
    int naf_target_resolution;
    const int32_t * projection_coords_bxyz;
    int64_t projection_n_coords;
    float camera_angle_x;
    float camera_distance;
    float mesh_scale;
    const trellis_backend_context * backend;
    const trellis_cuda_context * cuda;
    void * sparse_backend; /* optional backend-native fused Pixal3D NAF executor */
    trellis_pipeline_model_cache * cache;
} trellis_image_condition_options;

void trellis_image_condition_result_free(trellis_image_condition_result * result);

trellis_status trellis_pipeline_run_image_condition(
    const trellis_image_condition_options * options,
    trellis_image_condition_result * result);

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
    const float * projected_cond;
    const float * neg_projected_cond;
    int64_t projected_tokens;
    int projected_channels;
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
    trellis_ggml_attention_policy attention_policy;
    /* Legacy fallback when attention_policy was left zero-initialized. */
    int use_ggml_flash_attn;
    const trellis_backend_context * backend;
    const trellis_cuda_context * cuda;
    trellis_pipeline_model_cache * cache;
} trellis_structured_latent_options;

trellis_status trellis_pipeline_run_structured_latent(
    const trellis_structured_latent_options * options,
    trellis_structured_latent * latent_out);

typedef enum trellis_cascade_coord_quantization {
    TRELLIS_CASCADE_COORD_QUANTIZE_TRELLIS = 0,
    TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL = 1,
} trellis_cascade_coord_quantization;

trellis_status trellis_pipeline_quantize_cascade_coords(
    const int32_t * decoder_coords,
    int64_t decoder_n,
    int lr_resolution,
    int hr_resolution,
    trellis_cascade_coord_quantization quantization,
    int32_t ** coords_out,
    int64_t * n_out);

typedef struct trellis_pipeline_mesh_options {
    const char * model_dir;
    const char * decoder_override_path;
    const trellis_structured_latent * latent;
    int resolution;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    const trellis_cuda_context * cuda;
    trellis_sparse_backend_kind sparse_backend_kind;
    int sparse_device;
    void * sparse_backend;
    trellis_pipeline_model_cache * cache;
} trellis_pipeline_mesh_options;

trellis_status trellis_pipeline_decode_shape_latent_mesh(
    const trellis_pipeline_mesh_options * options,
    trellis_sparse_c2s_guides * subs_out,
    trellis_mesh_host * mesh_out);

trellis_status trellis_pipeline_decode_shape_latent_decoder_coords(
    const trellis_pipeline_mesh_options * options,
    int32_t ** coords_out,
    int64_t * n_out);

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
    const char * decoder_override_path;
    const trellis_structured_latent * latent;
    const trellis_sparse_c2s_guides * guide_subs;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    const trellis_cuda_context * cuda;
    trellis_sparse_backend_kind sparse_backend_kind;
    int sparse_device;
    void * sparse_backend;
    trellis_pipeline_model_cache * cache;
} trellis_pipeline_texture_options;

trellis_status trellis_pipeline_decode_texture_latent_voxels(
    const trellis_pipeline_texture_options * options,
    trellis_pbr_voxels * voxels_out);

trellis_status trellis_pipeline_dump_raw_mesh_if_requested(
    const char * dump_dir,
    const trellis_mesh_host * mesh);

trellis_status trellis_pipeline_postprocess_mesh_with_vkmesh(
    trellis_mesh_host * mesh,
    trellis_mesh_host * projection_mesh_out,
    const char * vkmesh_path,
    int decimation_target,
    int no_simplify,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project,
    int device,
    int gpu_workspace_budget_mib);

trellis_status trellis_pipeline_dump_material_inputs_if_requested(
    const char * dump_dir,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels);

trellis_status trellis_pipeline_apply_pbr_voxels_to_mesh(
    const trellis_pbr_voxels * voxels,
    trellis_mesh_host * mesh);

/* Applies the selected rigid output-axis transform to parallel xyz arrays.
 * Normals receive exactly the same rotation as positions; neither transform
 * changes handedness, so triangle winding must remain unchanged. */
trellis_status trellis_pipeline_transform_gltf_vectors(
    trellis_pipeline_gltf_coordinate_transform transform,
    float * positions_xyz,
    float * normals_xyz,
    size_t vector_count);

trellis_status trellis_pipeline_write_gltf(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    int device);

trellis_status trellis_pipeline_write_gltf_ex(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    int device,
    trellis_pipeline_gltf_coordinate_transform coordinate_transform);

#endif
