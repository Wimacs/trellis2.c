#ifndef TRELLIS2_C_MESH_RIGGING_INTERPOLATE_H
#define TRELLIS2_C_MESH_RIGGING_INTERPOLATE_H

#include "trellis.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matches Asset.from_data(): k=min(8,N), inverse (Euclidean) distance with
 * epsilon 1e-8, then a weighted sum for every joint. */
trellis_status trellis_tokenskin_interpolate_skin_8nn(
    const float * sampled_positions,
    const float * sampled_skin,
    size_t sample_count,
    size_t joint_count,
    const float * query_positions,
    size_t query_count,
    float ** query_skin_out);

#ifdef __cplusplus
}
#endif

#endif
