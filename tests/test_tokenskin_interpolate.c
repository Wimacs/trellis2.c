#include "interpolate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void require_close(float actual, float expected, float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "got %.9g expected %.9g\n", actual, expected);
        exit(1);
    }
}

int main(void) {
    const float points[12] = {
        0, 0, 0,
        1, 0, 0,
        2, 0, 0,
        3, 0, 0,
    };
    const float skin[8] = {
        1, 0,
        0.75f, 0.25f,
        0.25f, 0.75f,
        0, 1,
    };
    const float queries[6] = {0, 0, 0, 1.5f, 0, 0};
    float * result = NULL;
    if (trellis_tokenskin_interpolate_skin_8nn(
            points, skin, 4, 2, queries, 2, &result) != TRELLIS_STATUS_OK) {
        return 1;
    }
    require_close(result[0], 1.0f, 1.0e-6f);
    require_close(result[1], 0.0f, 1.0e-6f);
    require_close(result[2], 0.5f, 1.0e-6f);
    require_close(result[3], 0.5f, 1.0e-6f);
    free(result);
    puts("TokenSkin interpolation tests passed");
    return 0;
}
