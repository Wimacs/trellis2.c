#ifndef TRELLIS2_C_MESH_SEGMENTATION_NORMALIZATION_H
#define TRELLIS2_C_MESH_SEGMENTATION_NORMALIZATION_H

#include "image_to_3d_internal.h"
#include "gltf_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Converts flattened glTF world positions to the coordinate frame consumed by
 * SegviGen. With a decoder-frame cache anchor, both direct world axes and the
 * inverse TRELLIS glTF export transform are evaluated and the compatible frame
 * is selected. */
trellis_status trellis_mesh_segmentation_normalize_asset_mesh(
    const trellis_mesh_rigging_asset * asset,
    const trellis_shape_latent_cache_info * cache_anchor,
    trellis_mesh_host * mesh_out);

#ifdef __cplusplus
}
#endif

#endif
