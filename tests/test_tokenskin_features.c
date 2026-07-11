#include "tokenskin_features.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void require_close(float actual, float expected, float tolerance, const char * label) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g expected %.9g\n", label, actual, expected);
        exit(1);
    }
}

static void check_selected(
    const float feature[54],
    const float expected[20],
    float expected_sum,
    const char * label) {
    static const int indices[20] = {
        0, 1, 2, 3, 4, 10, 11, 18, 19, 26,
        27, 28, 34, 35, 42, 43, 50, 51, 52, 53,
    };
    float sum = 0.0f;
    for (int i = 0; i < 54; ++i) sum += feature[i];
    for (int i = 0; i < 20; ++i) {
        require_close(feature[indices[i]], expected[i], 3.0e-5f, label);
    }
    require_close(sum, expected_sum, 1.0e-4f, label);
}

int main(void) {
    const float position[3] = {0.25f, -0.5f, 0.75f};
    const float normal[3] = {0.1f, 0.2f, 0.3f};
    const float michelangelo_expected[20] = {
        0.25f, -0.5f, 0.75f, 0.247403964f, 0.47942555f,
        0.551426709f, -0.47942555f, -0.920026064f, 0.681638777f,
        0.983587742f, 0.968912423f, 0.87758255f, 0.83422333f,
        0.87758255f, 0.391857237f, 0.731688857f, -0.180430457f,
        0.1f, 0.2f, 0.3f,
    };
    const float fsq_expected[20] = {
        0.25f, -0.5f, 0.75f, 1.49206018f, 1.42195773f,
        0.382686704f, -0.127217054f, -0.707112372f, 0.824062765f,
        0.923880637f, 0.0875518918f, 0.906615496f, 1.92387938f,
        0.488108516f, 1.70710683f, -1.70024395f, 1.38268268f,
        0.1f, 0.2f, 0.3f,
    };
    float feature[54];
    trellis_tokenskin_encode_michelangelo_feature(position, normal, feature);
    check_selected(feature, michelangelo_expected, 3.48054647f, "Michelangelo");
    trellis_tokenskin_encode_fsq_cvae_feature(position, normal, feature);
    check_selected(feature, fsq_expected, 23.6180477f, "FSQ-CVAE");

    const float positions[6] = {0.25f, -0.5f, 0.75f, -0.25f, 0.0f, 1.0f};
    const float normals[6] = {0.1f, 0.2f, 0.3f, 0.0f, 1.0f, 0.0f};
    float * features = NULL;
    if (trellis_tokenskin_build_features(
            positions,
            normals,
            2,
            TRELLIS_TOKENSKIN_FEATURE_MICHELANGELO,
            &features) != TRELLIS_STATUS_OK || features == NULL) {
        return 1;
    }
    const uint32_t indices[2] = {1, 0};
    float * gathered = NULL;
    if (trellis_tokenskin_gather_features(features, 2, indices, 2, &gathered) !=
            TRELLIS_STATUS_OK || gathered == NULL) {
        free(features);
        return 1;
    }
    require_close(gathered[0], -0.25f, 0.0f, "gather first");
    require_close(gathered[54], 0.25f, 0.0f, "gather second");
    free(gathered);
    free(features);
    puts("TokenSkin feature tests passed");
    return 0;
}
