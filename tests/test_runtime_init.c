#include "trellis.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static trellis_status call_pipeline(const char * mode) {
    if (strcmp(mode, "image") == 0) {
        return trellis_pipeline_image_to_gltf(NULL);
    }
    if (strcmp(mode, "texture") == 0) {
        return trellis_pipeline_trellis2_texture_mesh(NULL);
    }
    if (strcmp(mode, "tokenskin") == 0) {
        return trellis_pipeline_tokenskin_rig(NULL);
    }
    return TRELLIS_STATUS_ERROR;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s image|texture|tokenskin\n", argv[0]);
        return 2;
    }

    const trellis_status status = call_pipeline(argv[1]);
    if (status != TRELLIS_STATUS_INVALID_ARGUMENT) {
        fprintf(
            stderr,
            "%s pipeline returned %s instead of invalid argument\n",
            argv[1],
            trellis_status_string(status));
        return 1;
    }

    const int64_t first_us = ggml_time_us();
    const int64_t second_us = ggml_time_us();
    if (first_us < 0 || second_us < first_us) {
        fprintf(
            stderr,
            "%s pipeline left the runtime timer invalid: first=%lld second=%lld\n",
            argv[1],
            (long long) first_us,
            (long long) second_us);
        return 1;
    }

    return 0;
}
