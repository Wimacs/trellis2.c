#ifndef TRELLIS2_C_MESH_SEGMENTATION_PARTITION_H
#define TRELLIS2_C_MESH_SEGMENTATION_PARTITION_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Turns semantic face labels into physical parts. Faces with the same label
 * are joined only across shared edges, so repeated semantic parts (for
 * example four chair legs) remain independently selectable instances. Small
 * connected islands may be absorbed into their largest edge neighbor. */
trellis_status trellis_mesh_segmentation_partition_faces(
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    uint32_t ** part_ids_out,
    size_t * part_count_out);

/* Geometry-aware variant used for glTF input. UV/material seams often duplicate
 * vertex indices at exactly the same position; welding them before edge
 * connectivity prevents one physical surface from fragmenting into hundreds
 * of artificial parts. The source triangles and face order remain unchanged. */
trellis_status trellis_mesh_segmentation_partition_faces_geometric(
    const float * positions,
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    float weld_tolerance,
    uint32_t ** part_ids_out,
    size_t * part_count_out);

#ifdef __cplusplus
}
#endif

#endif
