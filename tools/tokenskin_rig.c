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
        "  %s --model DIR --input FILE.glb --output rigged.glb [options]\n\n"
        "Runs only TokenSkin/TokenRig mesh rigging. Model packages from Trellis.2,\n"
        "Pixal3D, or another task are rejected before mesh parsing or GPU startup.\n\n"
        "Options:\n"
        "  --model DIR              TokenSkin package containing model.json and ckpts/\n"
        "  --input FILE             Input .glb/.gltf mesh\n"
        "  --output FILE            Output rigged .glb (default tokenskin-rigged.glb)\n"
        "  --backend NAME           Backend compiled into this binary (default "
            TRELLIS_DEFAULT_BACKEND ")\n"
        "  --device N               Backend device index (default 0)\n"
        "  --seed N                 Sampling seed (default 1)\n"
        "  --samples N              Surface samples, at least 2048 (released model: 54000)\n"
        "  --max-length N           Total Qwen context including mesh prefix (default 2048)\n"
        "  --top-k N                Top-k sampling (released checkpoint: 10)\n"
        "  --top-p X                Nucleus probability (default 0.95)\n"
        "  --temperature X          Sampling temperature (released checkpoint: 1.5)\n"
        "  --repetition-penalty X   Generated-token penalty (default 2)\n"
        "  --num-beams N            Beam-sampling width 1..16 (released checkpoint: 10)\n"
        "  --greedy                 Deterministic single-beam argmax\n"
        "  --official-eos-compat    Preserve released EOS/FSQ alias behavior (default)\n"
        "  --corrected-eos           Generate all four real FSQ tokens per joint\n"
        "  --no-flash-attn          Debug explicit-attention fallback\n"
        "  --verbose                Enable debug logging\n"
        "  --help                   Show this help\n",
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

int main(int argc, char ** argv) {
    trellis_tokenskin_rig_options options = TRELLIS_TOKENSKIN_RIG_OPTIONS_INIT;
    options.output_path = "tokenskin-rigged.glb";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--input") == 0) {
            options.input_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--output") == 0 ||
                   strcmp(argv[i], "--glb") == 0) {
            options.output_path = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--backend") == 0) {
            options.backend = argument_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.device)) goto bad_args;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u32(argument_value(argc, argv, &i), &options.seed)) goto bad_args;
        } else if (strcmp(argv[i], "--samples") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.sample_count)) goto bad_args;
        } else if (strcmp(argv[i], "--max-length") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.max_length)) goto bad_args;
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.top_k)) goto bad_args;
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (!parse_float(argument_value(argc, argv, &i), &options.top_p)) goto bad_args;
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (!parse_float(argument_value(argc, argv, &i), &options.temperature)) goto bad_args;
        } else if (strcmp(argv[i], "--repetition-penalty") == 0) {
            if (!parse_float(
                    argument_value(argc, argv, &i),
                    &options.repetition_penalty)) goto bad_args;
        } else if (strcmp(argv[i], "--num-beams") == 0) {
            if (!parse_int(argument_value(argc, argv, &i), &options.num_beams)) goto bad_args;
        } else if (strcmp(argv[i], "--greedy") == 0) {
            options.temperature = 0.0f;
            options.num_beams = 1;
        } else if (strcmp(argv[i], "--official-eos-compat") == 0) {
            options.official_eos_compat = 1;
        } else if (strcmp(argv[i], "--corrected-eos") == 0) {
            options.official_eos_compat = 0;
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
    if (options.model_dir == NULL || options.input_path == NULL ||
        options.output_path == NULL) goto bad_args;
    const trellis_status status = trellis_pipeline_tokenskin_rig(&options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("tokenskin-rig failed: %s", trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(stderr, argv[0]);
    return 2;
}
