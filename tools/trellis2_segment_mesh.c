#include "trellis.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE * output, const char * executable) {
    fprintf(
        output,
        "Usage:\n"
        "  %s --model DIR --segmentation-model DIR [--dino DIR] --input mesh.glb "
        "[--output parts.glb] [options]\n\n"
        "Runs SegviGen Full inside the TRELLIS.2 runtime and writes one GLB "
        "containing one selectable node+mesh per physical part. Without "
        "--condition-image, a deterministic fixed-camera condition image is "
        "rendered automatically.\n\n"
        "Required:\n"
        "  --model DIR                 Base TRELLIS.2 package (codecs/normalization)\n"
        "  --segmentation-model DIR    SegviGen Full package\n"
        "  --dino DIR                  DINOv3 directory (unless --segmentation-latent)\n"
        "  --input FILE                Input .glb/.gltf triangle mesh\n\n"
        "Output and conditioning:\n"
        "  --output FILE               Multi-part GLB (default parts.glb)\n"
        "  --condition-image FILE      Optional reference/render image\n"
        "  --condition-prepared        Skip BiRefNet for --condition-image\n"
        "  --rendered-condition FILE   Keep the automatic condition PNG\n"
        "  --birefnet FILE             Optional BiRefNet override\n\n"
        "Model/cache overrides:\n"
        "  --shape-encoder FILE        Base shape encoder checkpoint\n"
        "  --texture-encoder FILE      Base source-texture encoder checkpoint\n"
        "  --segmentation-flow FILE    SegviGen paired-flow checkpoint\n"
        "  --shape-decoder FILE        Base shape decoder checkpoint\n"
        "  --texture-decoder FILE      Base texture/label decoder checkpoint\n"
        "  --shape-latent FILE         Trusted/debug shape TSLAT cache\n"
        "  --texture-latent FILE       Trusted/debug source-texture TSLAT cache\n"
        "  --segmentation-latent FILE  Reuse a generated label TSLAT for postprocess\n"
        "  --shape-latent-output FILE  Save encoded shape TSLAT cache\n"
        "  --texture-latent-output FILE Save encoded source-texture TSLAT cache\n\n"
        "  --segmentation-latent-output FILE Save generated label TSLAT for comparison\n\n"
        "Inference/postprocess:\n"
        "  --steps N                   Paired flow steps (default 12)\n"
        "  --seed N                    Segmentation latent seed (default 42)\n"
        "  --min-component-faces N     Absorb smaller islands (default 16)\n"
        "  --min-palette-voxels N      Ignore smaller color bins (default 16)\n"
        "  --palette-merge-distance F  RGB merge radius (default 0.12549)\n"
        "  --backend NAME              Compiled backend (default "
            TRELLIS_DEFAULT_BACKEND ")\n"
        "  --device N                  Backend device index (default 0)\n"
        "  --no-flash-attn             Debug explicit-attention fallback\n"
        "  --verbose                   Enable debug logging\n"
        "  --help                      Show this help\n",
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

static int parse_float(const char * text, float * output) {
    if (text == NULL || output == NULL || text[0] == '\0') return 0;
    errno = 0;
    char * end = NULL;
    const float value = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(value)) return 0;
    *output = value;
    return 1;
}

static int argument_ranges_are_valid(
    const trellis_mesh_segmentation_options * options) {
    return options != NULL &&
        options->device >= 0 &&
        options->steps > 0 && options->steps <= 1000 &&
        options->min_component_faces >= 0 &&
        options->min_palette_voxels > 0 &&
        isfinite(options->palette_merge_distance) &&
        options->palette_merge_distance >= 0.0f &&
        options->palette_merge_distance <= 1.732051f;
}

int main(int argc, char ** argv) {
    trellis_runtime_init();
    trellis_mesh_segmentation_options options =
        TRELLIS_MESH_SEGMENTATION_OPTIONS_INIT;
    options.output_path = "parts.glb";

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--model") == 0) {
            options.model_dir = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--segmentation-model") == 0) {
            options.segmentation_model_dir = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--dino") == 0) {
            options.dino_dir = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--input") == 0) {
            options.input_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--output") == 0 ||
                   strcmp(argv[index], "--glb") == 0) {
            options.output_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--condition-image") == 0) {
            options.condition_image_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--condition-prepared") == 0) {
            options.condition_image_prepared = 1;
        } else if (strcmp(argv[index], "--rendered-condition") == 0) {
            options.rendered_condition_output_path =
                argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--birefnet") == 0) {
            options.birefnet_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--shape-encoder") == 0) {
            options.shape_encoder_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--texture-encoder") == 0) {
            options.texture_encoder_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--segmentation-flow") == 0) {
            options.segmentation_flow_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--shape-decoder") == 0) {
            options.shape_decoder_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--texture-decoder") == 0) {
            options.texture_decoder_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--shape-latent") == 0) {
            options.shape_latent_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--texture-latent") == 0) {
            options.texture_latent_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--segmentation-latent") == 0) {
            options.segmentation_latent_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--shape-latent-output") == 0) {
            options.shape_latent_output_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--texture-latent-output") == 0) {
            options.texture_latent_output_path = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--segmentation-latent-output") == 0) {
            options.segmentation_latent_output_path =
                argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--steps") == 0) {
            if (!parse_int(argument_value(argc, argv, &index), &options.steps)) goto bad_args;
        } else if (strcmp(argv[index], "--seed") == 0) {
            if (!parse_u32(argument_value(argc, argv, &index), &options.seed)) goto bad_args;
        } else if (strcmp(argv[index], "--min-component-faces") == 0) {
            if (!parse_int(
                    argument_value(argc, argv, &index),
                    &options.min_component_faces)) goto bad_args;
        } else if (strcmp(argv[index], "--min-palette-voxels") == 0) {
            if (!parse_int(
                    argument_value(argc, argv, &index),
                    &options.min_palette_voxels)) goto bad_args;
        } else if (strcmp(argv[index], "--palette-merge-distance") == 0) {
            if (!parse_float(
                    argument_value(argc, argv, &index),
                    &options.palette_merge_distance)) goto bad_args;
        } else if (strcmp(argv[index], "--backend") == 0) {
            options.backend = argument_value(argc, argv, &index);
        } else if (strcmp(argv[index], "--device") == 0) {
            if (!parse_int(argument_value(argc, argv, &index), &options.device)) goto bad_args;
        } else if (strcmp(argv[index], "--no-flash-attn") == 0) {
            options.no_flash_attn = 1;
        } else if (strcmp(argv[index], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[index], "--help") == 0 ||
                   strcmp(argv[index], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[index]);
            goto bad_args;
        }
    }
    if (options.model_dir == NULL || options.segmentation_model_dir == NULL ||
        (options.dino_dir == NULL && options.segmentation_latent_path == NULL) ||
        options.input_path == NULL ||
        options.output_path == NULL) goto bad_args;
    if (!argument_ranges_are_valid(&options)) {
        fprintf(
            stderr,
            "invalid argument range: --device must be >= 0, --steps must be "
            "1..1000, component thresholds must be non-negative (palette > 0), "
            "and --palette-merge-distance must be 0..sqrt(3)\n");
        goto bad_args;
    }

    const trellis_status status =
        trellis_pipeline_trellis2_segment_mesh(&options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "trellis2-segment-mesh failed: %s",
            trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(stderr, argv[0]);
    return 2;
}
