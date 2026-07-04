#include "trellis.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --model DIR --dino DIR --image FILE --obj FILE [options]\n"
        "\n"
        "Runs image -> sparse structure -> shape SLat -> mesh/subs -> texture SLat -> colored OBJ.\n"
        "\n"
        "Options:\n"
        "  --model DIR             TRELLIS.2 model directory containing ckpts/\n"
        "  --dino DIR              DINOv3 image encoder directory containing model.safetensors\n"
        "  --image FILE            Input image. PNG/JPEG load directly; WebP is converted with ffmpeg first.\n"
        "  --obj FILE              Output OBJ path\n"
        "  --device N              CUDA device, default 0\n"
        "  --steps N               Sparse-structure and structured-latent Euler steps, default 12\n"
        "  --sparse-structure-steps N Override sparse-structure steps\n"
        "  --structured-latent-steps N Override shape and texture SLat steps\n"
        "  --seed N                Sparse-structure latent seed, default 1\n"
        "  --noise-seed N          Structured-latent noise seed, default 18\n"
        "  --latent-size N         Sparse-structure latent grid edge, default 16\n"
        "  --resolution N          Shape checkpoint/mesh resolution, 512 or 1024, default 512\n"
        "  --cond-resolution N     DINO input square edge, default 512\n"
        "  --sparse-resolution N   Sparse-structure output edge, default 32\n"
        "  --flow PATH             Override shape SLat flow safetensors path\n"
        "  --decoder PATH          Override FlexiDualGridVaeDecoder safetensors path\n"
        "  --rescale-t X           Shape SLat timestep rescale factor, default 3.0\n"
        "  --guidance-strength X   Shape SLat CFG strength, default 7.5\n"
        "  --guidance-rescale X    Shape SLat CFG rescale, default 0.5\n"
        "  --guidance-min X        Shape SLat CFG interval min, default 0.6\n"
        "  --guidance-max X        Shape SLat CFG interval max, default 1.0\n"
        "  --flow-blocks N         Debug: run only first N transformer blocks in both flows\n"
        "  --flow-block-parts N    Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope          Debug: disable sparse RoPE\n"
        "  --emulate-bf16-blocks   Debug: round structured-latent block activations like reference bf16 flow\n"
        "  --use-ggml-flash-attn   Debug: use ggml flash attention in structured-latent flow\n"
        "  --decode-max-levels N   Debug: run only first N shape decoder levels, default full\n"
        "  --decode-max-input-tokens N Debug: truncate shape decoder input tokens\n"
        "  --verbose               Print debug logs\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int parse_int_arg(const char * text, int * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (int) v;
    return 1;
}

static int parse_i64_arg(const char * text, int64_t * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    long long v = strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (int64_t) v;
    return 1;
}

static int parse_u32_arg(const char * text, uint32_t * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    unsigned long v = strtoul(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (uint32_t) v;
    return 1;
}

static int parse_float_arg(const char * text, float * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    float v = strtof(text, &end);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = v;
    return 1;
}

int main(int argc, char ** argv) {
    trellis_image_to_obj_options options;
    memset(&options, 0, sizeof(options));
    options.device = 0;
    options.sparse_structure_steps = 12;
    options.structured_latent_steps = 12;
    options.latent_size = 16;
    options.resolution = 512;
    options.cond_resolution = 512;
    options.sparse_resolution = 32;
    options.seed = 1u;
    options.noise_seed = 18u;
    options.rescale_t = 3.0f;
    options.guidance_strength = 7.5f;
    options.guidance_rescale = 0.5f;
    options.guidance_min = 0.6f;
    options.guidance_max = 1.0f;
    options.flow_blocks_override = -1;
    options.flow_block_parts_override = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            options.dino_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--image") == 0) {
            options.image_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--obj") == 0) {
            options.obj_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--flow") == 0) {
            options.flow_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decoder") == 0) {
            options.decoder_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.device)) goto bad_args;
        } else if (strcmp(argv[i], "--steps") == 0) {
            int steps = 0;
            if (!parse_int_arg(arg_value(argc, argv, &i), &steps)) goto bad_args;
            options.sparse_structure_steps = steps;
            options.structured_latent_steps = steps;
        } else if (strcmp(argv[i], "--sparse-structure-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.sparse_structure_steps)) goto bad_args;
        } else if (strcmp(argv[i], "--structured-latent-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.structured_latent_steps)) goto bad_args;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options.seed)) goto bad_args;
        } else if (strcmp(argv[i], "--noise-seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options.noise_seed)) goto bad_args;
        } else if (strcmp(argv[i], "--latent-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.latent_size)) goto bad_args;
        } else if (strcmp(argv[i], "--resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--cond-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.cond_resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--sparse-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.sparse_resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.rescale_t)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-strength") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_strength)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-rescale") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_rescale)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-min") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_min)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-max") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_max)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.flow_blocks_override)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.flow_block_parts_override)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            options.flow_no_rope = 1;
        } else if (strcmp(argv[i], "--emulate-bf16-blocks") == 0) {
            options.emulate_bf16_blocks = 1;
        } else if (strcmp(argv[i], "--use-ggml-flash-attn") == 0) {
            options.use_ggml_flash_attn = 1;
        } else if (strcmp(argv[i], "--decode-max-levels") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.decode_max_levels)) goto bad_args;
        } else if (strcmp(argv[i], "--decode-max-input-tokens") == 0) {
            if (!parse_i64_arg(arg_value(argc, argv, &i), &options.decode_max_input_tokens)) goto bad_args;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            TRELLIS_ERROR("unknown option: %s", argv[i]);
            goto bad_args;
        }
    }

    if (options.model_dir == NULL || options.dino_dir == NULL ||
        options.image_path == NULL || options.obj_path == NULL ||
        options.sparse_structure_steps <= 0 || options.structured_latent_steps <= 0 ||
        options.latent_size <= 0 || options.cond_resolution <= 0 ||
        options.sparse_resolution <= 0 || (options.resolution != 512 && options.resolution != 1024)) {
        goto bad_args;
    }

    trellis_status status = trellis_pipeline_image_to_obj(&options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("trellis-image-to-obj failed: %s", trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(argv[0]);
    return 2;
}
