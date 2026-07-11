#ifndef TRELLIS2_C_MESH_RIGGING_PREPROCESS_H
#define TRELLIS2_C_MESH_RIGGING_PREPROCESS_H

#include "gltf_io.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRELLIS_MESH_RIGGING_DEFAULT_SURFACE_SAMPLES 54000u
#define TRELLIS_MESH_RIGGING_DEFAULT_MESH_TOKENS 512u
#define TRELLIS_MESH_RIGGING_DEFAULT_SKIN_TOKENS 384u
#define TRELLIS_MESH_RIGGING_DEFAULT_FPS_CANDIDATE_MULTIPLIER 4u

typedef struct trellis_mesh_rigging_preprocess_options {
    size_t struct_size;
    size_t surface_sample_count;
    size_t mesh_token_count;
    size_t skin_token_count;
    size_t fps_candidate_multiplier;
    uint64_t seed;
} trellis_mesh_rigging_preprocess_options;

#define TRELLIS_MESH_RIGGING_PREPROCESS_OPTIONS_INIT \
    { sizeof(trellis_mesh_rigging_preprocess_options), \
      TRELLIS_MESH_RIGGING_DEFAULT_SURFACE_SAMPLES, \
      TRELLIS_MESH_RIGGING_DEFAULT_MESH_TOKENS, \
      TRELLIS_MESH_RIGGING_DEFAULT_SKIN_TOKENS, \
      TRELLIS_MESH_RIGGING_DEFAULT_FPS_CANDIDATE_MULTIPLIER, 0u }

/* Uniform normalization used by the reference predict transform:
 *   normalized = (world - center) * scale
 * where the largest AABB extent maps to length 2.  Both matrices are
 * column-major so they can be copied directly into glTF matrices. */
typedef struct trellis_mesh_rigging_normalization {
    float source_min[3];
    float source_max[3];
    float center[3];
    float scale;
    float inverse_scale;
    float normalized_from_world[16];
    float world_from_normalized[16];
} trellis_mesh_rigging_normalization;

typedef struct trellis_mesh_rigging_preprocessed {
    trellis_mesh_rigging_normalization normalization;
    float * normalized_vertices; /* [vertex_count, 3] */
    size_t vertex_count;

    float * sample_positions;    /* [sample_count, 3], normalized space */
    float * sample_normals;      /* [sample_count, 3] */
    uint32_t * sample_triangles; /* [sample_count], source flattened triangle */
    float * sample_barycentric;  /* [sample_count, 3] */
    size_t sample_count;

    /* Candidate indices and final FPS indices both address sample_positions.
     * The final counts are normally 512 for Michelangelo and 384 for the skin
     * VAE condition encoder. */
    uint32_t * mesh_candidate_indices;
    uint32_t * mesh_fps_indices;
    size_t mesh_candidate_count;
    size_t mesh_fps_count;

    uint32_t * skin_candidate_indices;
    uint32_t * skin_fps_indices;
    size_t skin_candidate_count;
    size_t skin_fps_count;
} trellis_mesh_rigging_preprocessed;

#define TRELLIS_MESH_RIGGING_PREPROCESSED_INIT \
    { { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, 0, 0, { 0 }, { 0 } }, \
      NULL, 0, NULL, NULL, NULL, NULL, 0, NULL, NULL, 0, 0, NULL, NULL, 0, 0 }

trellis_status trellis_mesh_rigging_preprocess(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_preprocess_options * options,
    trellis_mesh_rigging_preprocessed * output,
    char * error_out,
    size_t error_size);

void trellis_mesh_rigging_preprocessed_free(
    trellis_mesh_rigging_preprocessed * output);

void trellis_mesh_rigging_normalize_point(
    const trellis_mesh_rigging_normalization * normalization,
    const float world[3],
    float normalized[3]);

void trellis_mesh_rigging_denormalize_point(
    const trellis_mesh_rigging_normalization * normalization,
    const float normalized[3],
    float world[3]);

#ifdef __cplusplus
}
#endif

#endif
