#ifndef TRELLIS2_C_MESH_SEGMENTATION_LABELS_H
#define TRELLIS2_C_MESH_SEGMENTATION_LABELS_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Transfers decoded SegviGen voxel RGB to the normalized source mesh, builds
 * a frequency-ordered merged palette, and returns one semantic label per
 * face. normalized_positions use the same [-0.5,0.5] coordinate frame as the
 * decoder. Voxel attrs are [n_voxels, voxel_channels], with RGB first. */
trellis_status trellis_mesh_segmentation_labels_from_voxels(
    const float * normalized_positions,
    size_t vertex_count,
    const uint32_t * triangles,
    size_t face_count,
    const int32_t * voxel_coords_bxyz,
    const float * voxel_attrs,
    int64_t n_voxels,
    int voxel_channels,
    int resolution,
    size_t min_palette_voxels,
    float palette_merge_distance,
    uint32_t ** semantic_labels_out,
    size_t * semantic_count_out);

#ifdef __cplusplus
}
#endif

#endif
