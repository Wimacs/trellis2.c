#include "tokenskin_features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void encode_fourier(
    const float position[3],
    const float normal[3],
    int include_pi,
    int use_pmpe,
    float feature[TRELLIS_TOKENSKIN_FEATURE_CHANNELS]) {
    float sine[24];
    float cosine[24];
    for (int axis = 0; axis < 3; ++axis) {
        for (int frequency_index = 0; frequency_index < 8; ++frequency_index) {
            const int index = axis * 8 + frequency_index;
            float frequency = ldexpf(1.0f, frequency_index);
            if (include_pi) frequency *= (float) M_PI;
            const float angle = position[axis] * frequency;
            if (use_pmpe) {
                const float ratio = (float) (frequency_index + 1) / 8.0f;
                const float phase_base = powf(8.0f, 1.0f - ratio) + ratio;
                const float phase = position[axis] * (float) M_PI * 0.5f +
                    phase_base * (float) (2.0 * M_PI);
                sine[index] = sinf(angle) + sinf(phase);
                cosine[index] = cosf(angle) + cosf(phase);
            } else {
                sine[index] = sinf(angle);
                cosine[index] = cosf(angle);
            }
        }
    }
    memcpy(feature, position, 3 * sizeof(float));
    memcpy(feature + 3, sine, 24 * sizeof(float));
    memcpy(feature + 27, cosine, 24 * sizeof(float));
    memcpy(feature + 51, normal, 3 * sizeof(float));
}

void trellis_tokenskin_encode_michelangelo_feature(
    const float position[3],
    const float normal[3],
    float feature[TRELLIS_TOKENSKIN_FEATURE_CHANNELS]) {
    if (position == NULL || normal == NULL || feature == NULL) return;
    encode_fourier(position, normal, 0, 0, feature);
}

void trellis_tokenskin_encode_fsq_cvae_feature(
    const float position[3],
    const float normal[3],
    float feature[TRELLIS_TOKENSKIN_FEATURE_CHANNELS]) {
    if (position == NULL || normal == NULL || feature == NULL) return;
    encode_fourier(position, normal, 1, 1, feature);
}

trellis_status trellis_tokenskin_build_features(
    const float * positions,
    const float * normals,
    size_t count,
    trellis_tokenskin_feature_kind kind,
    float ** features_out) {
    if (positions == NULL || normals == NULL || count == 0 ||
        features_out == NULL ||
        (kind != TRELLIS_TOKENSKIN_FEATURE_MICHELANGELO &&
         kind != TRELLIS_TOKENSKIN_FEATURE_FSQ_CVAE) ||
        count > SIZE_MAX / (TRELLIS_TOKENSKIN_FEATURE_CHANNELS * sizeof(float))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    float * result = (float *) malloc(
        count * TRELLIS_TOKENSKIN_FEATURE_CHANNELS * sizeof(float));
    if (result == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (size_t i = 0; i < count; ++i) {
        float * output = result + i * TRELLIS_TOKENSKIN_FEATURE_CHANNELS;
        if (kind == TRELLIS_TOKENSKIN_FEATURE_MICHELANGELO) {
            trellis_tokenskin_encode_michelangelo_feature(
                positions + i * 3u, normals + i * 3u, output);
        } else {
            trellis_tokenskin_encode_fsq_cvae_feature(
                positions + i * 3u, normals + i * 3u, output);
        }
    }
    *features_out = result;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tokenskin_gather_features(
    const float * features,
    size_t feature_count,
    const uint32_t * indices,
    size_t index_count,
    float ** gathered_out) {
    if (features == NULL || feature_count == 0 || indices == NULL ||
        index_count == 0 || gathered_out == NULL ||
        index_count > SIZE_MAX /
            (TRELLIS_TOKENSKIN_FEATURE_CHANNELS * sizeof(float))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    float * result = (float *) malloc(
        index_count * TRELLIS_TOKENSKIN_FEATURE_CHANNELS * sizeof(float));
    if (result == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (size_t i = 0; i < index_count; ++i) {
        if ((size_t) indices[i] >= feature_count) {
            free(result);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        memcpy(
            result + i * TRELLIS_TOKENSKIN_FEATURE_CHANNELS,
            features + (size_t) indices[i] * TRELLIS_TOKENSKIN_FEATURE_CHANNELS,
            TRELLIS_TOKENSKIN_FEATURE_CHANNELS * sizeof(float));
    }
    *gathered_out = result;
    return TRELLIS_STATUS_OK;
}
