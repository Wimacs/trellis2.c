#include "trellis.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern unsigned char * stbi_load(char const * filename, int * x, int * y, int * comp, int req_comp);
extern void stbi_image_free(void * retval_from_stbi_load);
extern int stbi_write_png(char const * filename, int w, int h, int comp, const void * data, int stride_in_bytes);

static void usage(FILE * out, const char * argv0) {
    fprintf(out,
        "Usage:\n"
        "  %s --birefnet FILE --image FILE --out FILE [options]\n"
        "\n"
        "Runs BiRefNet background removal on one RGB/RGBA image and writes an RGBA PNG.\n"
        "\n"
        "Options:\n"
        "  --birefnet FILE   BiRefNet GGUF model\n"
        "  --image FILE      Input RGB/RGBA image readable by stb_image\n"
        "  --out FILE        Output RGBA PNG path\n"
        "  --backend NAME    Backend for BiRefNet graph: cpu, cuda, or vulkan; default cpu\n"
        "  --device N        Backend device index, default 0\n"
        "  --verbose         Print debug logs\n"
        "  --help, -h        Show this help\n",
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

int main(int argc, char ** argv) {
    const char * birefnet_path = NULL;
    const char * image_path = NULL;
    const char * out_path = NULL;
    trellis_backend_kind backend_kind = TRELLIS_BACKEND_CPU;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--birefnet") == 0 || strcmp(argv[i], "--model") == 0) {
            birefnet_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--image") == 0 || strcmp(argv[i], "--input") == 0) {
            image_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "--output") == 0) {
            out_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--backend") == 0) {
            const char * name = arg_value(argc, argv, &i);
            if (trellis_backend_kind_from_name(name, &backend_kind) != TRELLIS_STATUS_OK) {
                fprintf(stderr, "invalid backend: %s\n", name == NULL ? "" : name);
                goto bad_args;
            }
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &device) || device < 0) {
                goto bad_args;
            }
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

    if (birefnet_path == NULL || image_path == NULL || out_path == NULL ||
        birefnet_path[0] == '\0' || image_path[0] == '\0' || out_path[0] == '\0') {
        goto bad_args;
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(image_path, &width, &height, &comp, 4);
    if (rgba == NULL || width <= 0 || height <= 0) {
        fprintf(stderr, "failed to load image: %s\n", image_path);
        stbi_image_free(rgba);
        return 1;
    }

    TRELLIS_INFO(
        "BiRefNet tool: %dx%d image, input components=%d, backend=%s",
        width,
        height,
        comp,
        trellis_backend_kind_name(backend_kind));

    trellis_birefnet_model model;
    memset(&model, 0, sizeof(model));
    trellis_status status = trellis_birefnet_load_gguf_with_backend(&model, birefnet_path, backend_kind, device);

    unsigned char * mask = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_birefnet_compute_mask_u8(&model, rgba, width, height, &mask);
    }

    if (status == TRELLIS_STATUS_OK) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = (size_t) y * (size_t) width + (size_t) x;
                rgba[i * 4u + 3u] =
                    (unsigned char) (((unsigned int) rgba[i * 4u + 3u] * (unsigned int) mask[i] + 127u) / 255u);
            }
        }
        if (!stbi_write_png(out_path, width, height, 4, rgba, width * 4)) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }

    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO("BiRefNet tool: wrote RGBA PNG: %s", out_path);
    } else {
        TRELLIS_ERROR("BiRefNet tool failed: %s", trellis_status_string(status));
    }

    free(mask);
    trellis_birefnet_free(&model);
    stbi_image_free(rgba);
    return status == TRELLIS_STATUS_OK ? 0 : 1;

bad_args:
    usage(stderr, argv[0]);
    return 2;
}
