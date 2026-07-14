#ifndef TRELLIS2_C_MESH_SEGMENTATION_MATERIAL_FEATURES_H
#define TRELLIS2_C_MESH_SEGMENTATION_MATERIAL_FEATURES_H

#include "trellis.h"
#include "../mesh_rigging/gltf_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque, read-only view of a glTF asset's material data.  The condition
 * renderer keeps one sampler open for the whole raster pass so embedded or
 * external images are decoded at most once. */
typedef struct trellis_mesh_segmentation_material_sampler {
    void * implementation;
} trellis_mesh_segmentation_material_sampler;

#define TRELLIS_MESH_SEGMENTATION_MATERIAL_SAMPLER_INIT { NULL }

typedef struct trellis_mesh_segmentation_surface_sample {
    /* Linear-light RGB, after applying glTF baseColorFactor and texture. */
    float base_color[3];
    /* Effective coverage after glTF OPAQUE/MASK/BLEND handling. */
    float alpha;
    int unlit;
} trellis_mesh_segmentation_surface_sample;

/* Opens the same cgltf/image sampling path used by the source-texture
 * encoder, but exposes linear-light base color and effective alpha for a
 * renderer.  The asset must have been flattened from source_path by gltf_io.
 */
trellis_status trellis_mesh_segmentation_material_sampler_open(
    const char * source_path,
    const trellis_mesh_rigging_asset * asset,
    trellis_mesh_segmentation_material_sampler * sampler_out);

trellis_status trellis_mesh_segmentation_material_sampler_sample(
    trellis_mesh_segmentation_material_sampler * sampler,
    size_t face,
    const float barycentric[3],
    trellis_mesh_segmentation_surface_sample * sample_out);

void trellis_mesh_segmentation_material_sampler_close(
    trellis_mesh_segmentation_material_sampler * sampler);

/* Samples glTF metallic-roughness materials at the Flexible Dual Grid surface
 * points. normalized_mesh must preserve the flattened asset's vertex and face
 * order and use SegviGen's [-0.5, 0.5] frame. FDG dual_vertices are relative
 * to the default AABB minimum, so each sampled point is dual_vertex - 0.5.
 *
 * The returned allocation contains grid->n rows of:
 *   baseColor.r, baseColor.g, baseColor.b, metallic, roughness, alpha
 * after each channel has been mapped from [0, 1] to [-1, 1]. The caller owns
 * the allocation and releases it with free(). */
trellis_status trellis_mesh_segmentation_material_features(
    const char * source_path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_host * normalized_mesh,
    const trellis_flexible_dual_grid * grid,
    float ** features_out);

#ifdef __cplusplus
}
#endif

#endif
