#include "trellis.h"
#include "trellis_checkpoint_validate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --model TRELLIS.2-4B --validate\n"
        "  %s --timesteps 12 --rescale-t 5.0\n"
        "\n"
        "Options:\n"
        "  --model DIR        TRELLIS.2 model directory containing ckpts/\n"
        "  --flow PATH        Override sparse-structure flow safetensors path\n"
        "  --decoder PATH     Override sparse-structure decoder safetensors path\n"
        "  --validate         Validate sparse-structure flow + decoder checkpoint contracts\n"
        "  --timesteps N      Print Euler timestep pairs\n"
        "  --rescale-t X      Timestep rescale factor, default 5.0\n"
        "  --cuda-check       Initialize ggml CUDA backend and report availability\n",
        argv0, argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static void print_report(const char * label, trellis_status status, const trellis_checkpoint_report * report) {
    printf("%s: %s\n", label, trellis_status_string(status));
    printf("  tensors: expected=%zu actual=%zu found=%zu missing=%zu extra=%zu\n",
        report->expected_tensors,
        report->actual_tensors,
        report->found_tensors,
        report->missing_tensors,
        report->extra_tensors);
    printf("  mismatches: shape=%zu dtype=%zu\n", report->shape_mismatches, report->dtype_mismatches);
    printf("  expected: elements=%llu bytes=%llu\n",
        (unsigned long long) report->expected_elements,
        (unsigned long long) report->expected_bytes);
    if (report->first_issue[0] != '\0') {
        printf("  first issue: %s\n", report->first_issue);
    }
}

int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * flow_path = NULL;
    const char * decoder_path = NULL;
    int validate = 0;
    int cuda_check = 0;
    int timesteps = 0;
    float rescale_t = 5.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--flow") == 0) {
            flow_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decoder") == 0) {
            decoder_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--validate") == 0) {
            validate = 1;
        } else if (strcmp(argv[i], "--cuda-check") == 0) {
            cuda_check = 1;
        } else if (strcmp(argv[i], "--timesteps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            timesteps = atoi(v);
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            rescale_t = (float) atof(v);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (cuda_check) {
        trellis_cuda_context cuda;
        trellis_status status = trellis_cuda_init(&cuda, 0);
        printf("cuda: %s\n", trellis_status_string(status));
        trellis_cuda_free(&cuda);
        if (status != TRELLIS_STATUS_OK) {
            return 1;
        }
    }

    if (timesteps > 0) {
        float * pairs = (float *) calloc((size_t) timesteps * 2u, sizeof(float));
        if (pairs == NULL) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        trellis_status status = trellis_flow_timestep_pairs_f32(timesteps, rescale_t, pairs, (size_t) timesteps * 2u);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "timesteps: %s\n", trellis_status_string(status));
            free(pairs);
            return 1;
        }
        for (int i = 0; i < timesteps; ++i) {
            printf("%02d %.9g %.9g\n", i + 1, pairs[2 * i + 0], pairs[2 * i + 1]);
        }
        free(pairs);
    }

    if (validate) {
        char flow_buf[4096];
        char dec_buf[4096];
        if (flow_path == NULL) {
            if (model_dir == NULL) {
                usage(argv[0]);
                return 2;
            }
            trellis_status pstatus = trellis_make_model_path(
                model_dir,
                "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
                flow_buf,
                sizeof(flow_buf));
            if (pstatus != TRELLIS_STATUS_OK) {
                fprintf(stderr, "flow path: %s\n", trellis_status_string(pstatus));
                return 1;
            }
            flow_path = flow_buf;
        }
        if (decoder_path == NULL) {
            if (model_dir == NULL) {
                usage(argv[0]);
                return 2;
            }
            trellis_status pstatus = trellis_make_model_path(
                model_dir,
                "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                dec_buf,
                sizeof(dec_buf));
            if (pstatus != TRELLIS_STATUS_OK) {
                fprintf(stderr, "decoder path: %s\n", trellis_status_string(pstatus));
                return 1;
            }
            decoder_path = dec_buf;
        }

        trellis_checkpoint_report flow_report;
        trellis_checkpoint_report dec_report;
        trellis_status flow_status = trellis_ss_flow_validate_checkpoint(flow_path, &flow_report);
        trellis_status dec_status = trellis_ss_decoder_validate_checkpoint(decoder_path, &dec_report);
        print_report("sparse-structure flow", flow_status, &flow_report);
        print_report("sparse-structure decoder", dec_status, &dec_report);
        if (flow_status != TRELLIS_STATUS_OK || dec_status != TRELLIS_STATUS_OK) {
            return 1;
        }
    }

    if (!validate && !cuda_check && timesteps <= 0) {
        usage(argv[0]);
        return 2;
    }
    return 0;
}
