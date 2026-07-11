#ifndef TRELLIS2_C_MESH_RIGGING_RIGGED_GLTF_H
#define TRELLIS2_C_MESH_RIGGING_RIGGED_GLTF_H

#include "preprocess.h"
#include "tokenizer.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dense skin predictions.  values is row-major [point_count, joint_count].
 *
 * When sample_positions is NULL, rows must correspond one-to-one with the
 * flattened asset vertices.  When it is non-NULL, it contains normalized
 * model-space positions [point_count, 3]; weights are transferred to every
 * asset vertex using exact k-nearest-neighbour inverse-distance interpolation.
 * This mirrors the sampled-to-original-vertex step in the TokenSkin reference
 * pipeline without exposing a spatial-index implementation to callers. */
typedef struct trellis_mesh_rigging_dense_skin_weights {
    const float * values;
    const float * sample_positions;
    size_t point_count;
    size_t joint_count;
    /* Zero selects the reference default of 8.  Values above 8 are rejected. */
    size_t interpolation_neighbors;
} trellis_mesh_rigging_dense_skin_weights;

#define TRELLIS_MESH_RIGGING_DENSE_SKIN_WEIGHTS_INIT \
    { NULL, NULL, 0, 0, 0 }

/* Writes a self-contained glTF 2.0 binary file.
 *
 * The flattened world-space geometry is emitted unchanged, with one output
 * primitive for every asset primitive range.  Joint heads are converted from
 * normalized model space back to the asset's world space.  Every vertex is
 * reduced to its four strongest non-negative influences and renormalized;
 * an all-zero row deterministically falls back to joint zero with weight one.
 *
 * Texture coordinates, source materials/textures, morph targets, animation,
 * and arbitrary source-node properties are not present in the flattened
 * inference asset and therefore cannot be reproduced by this writer.  Every
 * primitive receives a shared opaque white, non-metallic default PBR material
 * so the resulting asset remains immediately renderable and spec-complete. */
trellis_status trellis_mesh_rigging_write_rigged_glb(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_tokenskin_skeleton * skeleton,
    const trellis_mesh_rigging_dense_skin_weights * dense_weights,
    char * error_out,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
