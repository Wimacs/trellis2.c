#ifndef TRELLIS2_C_MESH_RIGGING_TOKENSKIN_FEATURES_H
#define TRELLIS2_C_MESH_RIGGING_TOKENSKIN_FEATURES_H

#include "preprocess.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TRELLIS_TOKENSKIN_FEATURE_CHANNELS = 54,
};

typedef enum trellis_tokenskin_feature_kind {
    TRELLIS_TOKENSKIN_FEATURE_MICHELANGELO = 0,
    TRELLIS_TOKENSKIN_FEATURE_FSQ_CVAE = 1,
} trellis_tokenskin_feature_kind;

/* Both functions preserve the exact PyTorch flattening order:
 * xyz, xyz-major sine frequencies, xyz-major cosine frequencies, normals. */
void trellis_tokenskin_encode_michelangelo_feature(
    const float position[3],
    const float normal[3],
    float feature[TRELLIS_TOKENSKIN_FEATURE_CHANNELS]);

void trellis_tokenskin_encode_fsq_cvae_feature(
    const float position[3],
    const float normal[3],
    float feature[TRELLIS_TOKENSKIN_FEATURE_CHANNELS]);

trellis_status trellis_tokenskin_build_features(
    const float * positions,
    const float * normals,
    size_t count,
    trellis_tokenskin_feature_kind kind,
    float ** features_out);

trellis_status trellis_tokenskin_gather_features(
    const float * features,
    size_t feature_count,
    const uint32_t * indices,
    size_t index_count,
    float ** gathered_out);

#ifdef __cplusplus
}
#endif

#endif
