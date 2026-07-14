#ifndef TRELLIS2_C_MESH_SEGMENTATION_PARTS_GLB_H
#define TRELLIS2_C_MESH_SEGMENTATION_PARTS_GLB_H

#include "gltf_io.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Writes one self-contained GLB with an assembly root and one independently
 * selectable node+mesh per dense part id. Geometry remains in the flattened
 * source glTF world coordinate system. Each input face is emitted exactly
 * once, and vertices shared across part boundaries are duplicated. */
trellis_status trellis_mesh_segmentation_write_parts_glb(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const uint32_t * face_part_ids,
    size_t face_count,
    size_t part_count,
    const char * const * part_names,
    char * error_out,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
