#include "trellis.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_component_status(void) {
    printf("TRELLIS.2 component coverage:\n");
    for (size_t i = 0; i < trellis_component_status_count(); ++i) {
        const trellis_component_status * s = trellis_component_status_at(i);
        printf("  [%s] %s - %s\n", s->implemented ? "partial" : "todo", s->name, s->notes);
    }
}

static void inspect_safetensors(const char * path) {
    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    if (status != TRELLIS_STATUS_OK) {
        printf("  %s: %s\n", path, trellis_status_string(status));
        return;
    }
    uint64_t total = 0;
    uint64_t by_dtype[TRELLIS_DTYPE_BOOL + 1];
    memset(by_dtype, 0, sizeof(by_dtype));
    for (size_t i = 0; i < st.n_tensors; ++i) {
        uint64_t n = trellis_safetensor_nelements(&st.tensors[i]);
        total += n;
        if (st.tensors[i].dtype <= TRELLIS_DTYPE_BOOL) {
            by_dtype[st.tensors[i].dtype] += n;
        }
    }
    printf("  %s: %zu tensors, %" PRIu64 " scalars", path, st.n_tensors, total);
    for (int dtype = TRELLIS_DTYPE_F32; dtype <= TRELLIS_DTYPE_BOOL; ++dtype) {
        if (by_dtype[dtype] != 0) {
            printf(", %s=%" PRIu64, trellis_dtype_name((trellis_dtype) dtype), by_dtype[dtype]);
        }
    }
    printf("\n");
    trellis_safetensors_close(&st);
}

int main(int argc, char ** argv) {
    print_component_status();

    if (argc < 2) {
        printf("\nUsage: %s /path/to/TRELLIS.2-4B\n", argv[0]);
        return 0;
    }

    const char * files[] = {
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    };

    printf("\nCheckpoint manifests under %s:\n", argv[1]);
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        size_t len = strlen(argv[1]) + 1 + strlen(files[i]) + 1;
        char * path = (char *) malloc(len);
        if (path == NULL) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        snprintf(path, len, "%s/%s", argv[1], files[i]);
        inspect_safetensors(path);
        free(path);
    }
    return 0;
}
