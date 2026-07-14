#include "trellis.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE * output, const char * executable) {
    fprintf(
        output,
        "Usage:\n"
        "  %s --model DIR --dino DIR --input mesh.glb --image reference.png "
        "--output textured.glb [options]\n\n"
        "Textures an existing mesh with the TRELLIS.2 shape-conditioned material "
        "pipeline. Other model families are rejected before GPU initialization.\n\n"
        "Options:\n"
        "  --model DIR             TRELLIS.2 model directory\n"
        "  --dino DIR              DINOv3 image encoder directory\n"
        "  --input FILE            Input .glb/.gltf triangle mesh\n"
        "  --image FILE            Material reference image\n"
        "  --image-prepared        Treat --image as the final condition; disable BiRefNet\n"
        "  --shape-latent FILE     Reuse a compatible TRELLIS shape SLat cache\n"
        "  --shape-latent-output FILE Write cache after mesh encoding when reuse misses\n"
        "  --output FILE           Output textured GLB (default textured.glb)\n"
        "  --birefnet FILE         Override auto-discovered BiRefNet GGUF\n"
        "  --shape-encoder FILE    Override FlexiDualGridVaeEncoder weights\n"
        "  --texture-flow FILE     Override texture SLat flow weights\n"
        "  --texture-decoder FILE  Override texture SparseUnet decoder weights\n"
        "  --resolution N          Flexible dual grid resolution: 512 or 1024 "
        "(default 512)\n"
        "  --texture-size N        Output PBR texture edge (default 1024)\n"
        "  --steps N               Texture flow Euler steps (default 12)\n"
        "  --seed N                Texture latent noise seed (default 42)\n"
        "  --backend NAME          Backend compiled into this binary (default "
            TRELLIS_DEFAULT_BACKEND ")\n"
        "  --device N              Backend device index (default 0)\n"
        "  --no-flash-attn         Debug explicit-attention fallback\n"
        "  --verbose               Enable debug logging\n"
        "  --help                  Show this help\n",
        executable);
}

static const char * argument_value(int argc, char ** argv, int * index) {
    if (*index + 1 >= argc) return NULL;
    return argv[++*index];
}

static int parse_int(const char * text, int * output) {
    if (text == NULL || output == NULL || text[0] == '\0') return 0;
    errno = 0;
    char * end = NULL;
    const long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) return 0;
    *output = (int) value;
    return 1;
}

static int parse_u32(const char * text, uint32_t * output) {
    if (text == NULL || output == NULL || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    char * end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value > UINT32_MAX) return 0;
    *output = (uint32_t) value;
    return 1;
}

int main(int argc, char ** argv) {
    /* Prepared-image execution can reach performance timers before ggml_init(). */
    trellis_runtime_init();

    trellis_mesh_texturing_options options = TRELLIS_MESH_TEXTURING_OPTIONS_INIT;
    options.output_path = "textured.glb";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            options.dino_dir = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--input") == 0) {
            options.input_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--image") == 0) {
            options.image_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--image-prepared") == 0) {
            options.image_prepared = 1;
        } else if (strcmp(argv[i], "--shape-latent") == 0) {
            options.shape_latent_path = argument_value(argc, argv, &i);
            if (options.shape_latent_path == NULL ||
                options.shape_latent_path[0] == '\0') goto bad_args;
        } else if (strcmp(argv[i], "--shape-latent-output") == 0) {
            options.shape_latent_output_path = argument_value(argc, argv, &i);
            if (options.shape_latent_output_path == NULL ||
                options.shape_latent_output_path[0] == '\0') goto bad_args;
        } else if (strcmp(argv[i], "--output") == 0 ||
                   strcmp(argv[i], "--glb") == 0) {
            options.output_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--birefnet") == 0) {
            options.birefnet_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--shape-encoder") == 0) {
            options.encoder_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--texture-flow") == 0) {
            options.texture_flow_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--texture-decoder") == 0) {
            options.texture_decoder_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--resolution") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.texture_size)) goto bad_args;
        } else if (strcmp(argv[i], "--steps") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.steps)) goto bad_args;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u32(argument_value(argc, argv, &i), &options.seed)) goto bad_args;
        } else if (strcmp(argv[i], "--backend") == 0) {
            options.backend = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.device)) goto bad_args;
        } else if (strcmp(argv[i], "--no-flash-attn") == 0) {
            options.no_flash_attn = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            goto bad_args;
        }
    }

    if (options.model_dir == NULL || options.dino_dir == NULL ||
        options.input_path == NULL || options.image_path == NULL ||
        options.output_path == NULL) goto bad_args;

    const trellis_status status = trellis_pipeline_trellis2_texture_mesh(&options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "trellis2-texture-mesh failed: %s",
            trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(stderr, argv[0]);
    return 2;
}
