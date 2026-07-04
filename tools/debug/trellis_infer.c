#include "trellis.h"
#include "trellis_tool_cli.h"
#include "trellis_tool_live.h"
#include "trellis_tool_model.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../3rd/raylib/src/external/stb_image.h"

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --model TRELLIS.2-4B --load-stage1-flow\n"
        "  %s --model TRELLIS.2-4B --load-shape-flow-512\n"
        "  %s --model TRELLIS.2-4B --dino dinov3-vitl16-pretrain-lvd1689m --stage1-image-voxels --image input.png --out frames\n"
        "\n"
        "This is the C + ggml + CUDA inference entry point.\n"
        "It runs from the terminal, reports model loading progress, and prints\n"
        "sampler steps without opening the raylib live viewer.\n"
        "\n"
        "Options:\n"
        "  --model DIR              TRELLIS.2 model directory\n"
        "  --dino DIR               DINOv3 model directory, default dinov3-vitl16-pretrain-lvd1689m\n"
        "  --device N               CUDA device, default 0\n"
        "  --load-dino              Load DINOv3 image encoder model.safetensors\n"
        "  --load-stage1-flow       Load ss_flow_img_dit_1_3B_64_bf16.safetensors\n"
        "  --load-stage1-decoder    Load ss_dec_conv3d_16l8_fp16.safetensors\n"
        "  --load-shape-flow-512    Load slat_flow_img2shape_dit_1_3B_512_bf16.safetensors\n"
        "  --load-shape-flow-1024   Load slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors\n"
        "  --stage1-smoke-voxels    Run flow->decoder smoke pipeline and write voxel frames\n"
        "  --stage1-image-voxels    Run image-conditioned DINO->flow->decoder stage1 voxel frames\n"
        "  --image FILE             Input image for --stage1-image-voxels (PNG/JPEG/BMP/etc. via stb_image)\n"
        "  --cond-resolution N      DINO input square edge, default 512 for image stage\n"
        "  --sparse-resolution N    Output sparse voxel edge, default 32; use 64 for raw decoder\n"
        "  --input-cond-f32 FILE    Use reference cond tensor [1,N,1024] and skip C DINO\n"
        "  --input-neg-cond-f32 FILE Use reference neg_cond tensor [1,N,1024], default zeros\n"
        "  --input-latent-f32 FILE  Use reference initial latent [1,8,D,H,W] instead of C RNG\n"
        "  --save-cond-f32 FILE     Write final DINO cond tensor [T,1024] for stage2\n"
        "  --save-final-coords-i32 FILE Write final sparse coords [N,4] batch,d,h,w for stage2\n"
        "  --flow-blocks N          Debug: run only the first N flow transformer blocks\n"
        "  --flow-block-parts N     Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope           Debug: disable flow self-attention RoPE\n"
        "  --dump-intermediates DIR Write C inference tensor dumps and checks to DIR\n"
        "  --out DIR                Output snapshot directory for voxel frames\n"
        "  --steps N                Smoke sampler Euler steps, default 1\n"
        "  --seed N                 Deterministic latent seed for image stage, default 1\n"
        "  --voxel-threshold X      Decoder logit threshold, default 0\n"
        "  --dry-forward            Run a small real-weight CUDA forward smoke test\n"
        "  --dry-decode             Run a small real-weight decoder smoke test\n"
        "  --dry-dino-patch         Run a real-weight DINO image encoder ggml smoke test\n"
        "  --tokens N               Dry forward latent tokens, default 4\n"
        "  --cond-tokens N          Dry forward condition tokens, default 2\n"
        "  --latent-size N          Dry decoder latent edge size, default 1\n"
        "  --batch N                Dry forward batch, default 1\n"
        "  --verbose                Show detailed per-phase logs\n",
        argv0, argv0, argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int file_size_bytes(const char * path, size_t * size_out) {
    if (path == NULL || size_out == NULL) {
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long n = ftell(f);
    fclose(f);
    if (n < 0) {
        return 0;
    }
    *size_out = (size_t) n;
    return 1;
}

static int read_f32_file_exact(const char * path, float * dst, size_t count) {
    if (path == NULL || dst == NULL) {
        return 0;
    }
    size_t bytes = 0;
    if (!file_size_bytes(path, &bytes) || bytes != count * sizeof(float)) {
        fprintf(stderr, "f32 input: %s has unexpected size\n", path);
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "f32 input: failed to open %s\n", path);
        return 0;
    }
    const size_t got = fread(dst, sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "f32 input: short read %s\n", path);
        return 0;
    }
    return 1;
}

static int read_f32_file_alloc(const char * path, float ** data_out, size_t * count_out) {
    if (path == NULL || data_out == NULL || count_out == NULL) {
        return 0;
    }
    *data_out = NULL;
    *count_out = 0;
    size_t bytes = 0;
    if (!file_size_bytes(path, &bytes) || (bytes % sizeof(float)) != 0) {
        fprintf(stderr, "f32 input: %s is not a whole f32 tensor\n", path);
        return 0;
    }
    const size_t count = bytes / sizeof(float);
    float * data = (float *) malloc(count * sizeof(float));
    if (data == NULL && count != 0) {
        return 0;
    }
    if (!read_f32_file_exact(path, data, count)) {
        free(data);
        return 0;
    }
    *data_out = data;
    *count_out = count;
    return 1;
}

static int write_binary_file_exact(const char * path, const void * data, size_t elem_size, size_t count) {
    if (path == NULL || data == NULL || elem_size == 0) {
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "write: failed to open %s\n", path);
        return 0;
    }
    const size_t wrote = fwrite(data, elem_size, count, f);
    fclose(f);
    if (wrote != count) {
        fprintf(stderr, "write: short write %s\n", path);
        return 0;
    }
    return 1;
}

static int choose_path(
    const char * model_dir,
    const char * rel,
    char * dst,
    size_t dst_size) {
    trellis_status status = trellis_make_model_path(model_dir, rel, dst, dst_size);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "path: %s\n", trellis_status_string(status));
        return 0;
    }
    return 1;
}

static int dry_forward_flow(
    const trellis_cuda_context * cuda,
    const char * label,
    const trellis_dit_flow_weights * weights,
    int tokens,
    int cond_tokens,
    int batch) {
    if (cuda == NULL || weights == NULL || tokens <= 0 || cond_tokens <= 0 || batch <= 0) {
        return 1;
    }

    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false) + 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: graph ctx allocation failed\n", label);
        return 1;
    }

    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->in_channels, tokens, batch);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch);
    struct ggml_tensor * c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->cond_channels, cond_tokens, batch);
    struct ggml_tensor * cos_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, weights->head_dim / 2, tokens, 1);
    struct ggml_tensor * sin_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, weights->head_dim / 2, tokens, 1);
    struct ggml_tensor * y = trellis_dit_flow_forward(ctx, x, t, c, cos_phase, sin_phase, weights);
    if (x == NULL || t == NULL || c == NULL || cos_phase == NULL || sin_phase == NULL || y == NULL) {
        fprintf(stderr, "%s: dry forward graph construction failed\n", label);
        ggml_free(ctx);
        return 1;
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        fprintf(stderr, "%s: graph allocation failed\n", label);
        ggml_free(ctx);
        return 1;
    }
    ggml_build_forward_expand(graph, y);

    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(cuda);
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "%s: CUDA graph allocation failed\n", label);
        if (alloc != NULL) {
            ggml_gallocr_free(alloc);
        }
        ggml_free(ctx);
        return 1;
    }

    const size_t x_count = (size_t) weights->in_channels * (size_t) tokens * (size_t) batch;
    const size_t c_count = (size_t) weights->cond_channels * (size_t) cond_tokens * (size_t) batch;
    const size_t phase_count = (size_t) (weights->head_dim / 2) * (size_t) tokens;
    float * x_data = (float *) malloc(x_count * sizeof(float));
    float * c_data = (float *) calloc(c_count, sizeof(float));
    float * t_data = (float *) malloc((size_t) batch * sizeof(float));
    float * cos_data = (float *) malloc(phase_count * sizeof(float));
    float * sin_data = (float *) calloc(phase_count, sizeof(float));
    float * out = (float *) malloc((size_t) weights->out_channels * (size_t) tokens * (size_t) batch * sizeof(float));
    if (x_data == NULL || c_data == NULL || t_data == NULL || cos_data == NULL || sin_data == NULL || out == NULL) {
        fprintf(stderr, "%s: host buffer allocation failed\n", label);
        free(x_data);
        free(c_data);
        free(t_data);
        free(cos_data);
        free(sin_data);
        free(out);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 1;
    }

    for (size_t i = 0; i < x_count; ++i) {
        x_data[i] = 0.01f * (float) ((int) (i % 17) - 8);
    }
    for (int i = 0; i < batch; ++i) {
        t_data[i] = 1.0f;
    }
    for (size_t i = 0; i < phase_count; ++i) {
        cos_data[i] = 1.0f;
    }

    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(t, t_data, 0, ggml_nbytes(t));
    ggml_backend_tensor_set(c, c_data, 0, ggml_nbytes(c));
    ggml_backend_tensor_set(cos_phase, cos_data, 0, ggml_nbytes(cos_phase));
    ggml_backend_tensor_set(sin_phase, sin_data, 0, ggml_nbytes(sin_phase));

    trellis_status status = trellis_cuda_compute_graph(cuda, graph);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "%s: dry forward compute failed: %s\n", label, trellis_status_string(status));
        free(x_data);
        free(c_data);
        free(t_data);
        free(cos_data);
        free(sin_data);
        free(out);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    printf("  dry forward ok: tokens=%d cond_tokens=%d batch=%d output ne=[%lld,%lld,%lld] first=",
        tokens,
        cond_tokens,
        batch,
        (long long) y->ne[0],
        (long long) y->ne[1],
        (long long) y->ne[2]);
    const int n_print = weights->out_channels < 6 ? weights->out_channels : 6;
    for (int i = 0; i < n_print; ++i) {
        printf("%s%.6g", i == 0 ? "" : ",", out[i]);
    }
    printf("\n");

    free(x_data);
    free(c_data);
    free(t_data);
    free(cos_data);
    free(sin_data);
    free(out);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return 0;
}

static int make_join_path(const char * dir, const char * file, char * dst, size_t dst_size) {
    if (dir == NULL || file == NULL || dst == NULL || dst_size == 0) {
        return 0;
    }
    int n = snprintf(dst, dst_size, "%s/%s", dir, file);
    return n >= 0 && (size_t) n < dst_size;
}

static int load_decoder_store(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    int dry_decode,
    int dry_latent_size,
    int dry_batch) {
    char path[4096];
    if (!choose_path(model_dir, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", path, sizeof(path))) {
        return 1;
    }

    trellis_tensor_store store;
    if (!trellis_tool_load_tensor_store_f32(
            cuda,
            "stage1 sparse-structure decoder",
            path,
            false,
            64,
            &store,
            NULL)) {
        return 1;
    }

    trellis_ss_decoder_weights weights;
    char issue[256];
    trellis_status status = trellis_ss_decoder_bind_weights(&store, &weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("stage1 sparse-structure decoder: bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        trellis_tensor_store_free(&store);
        return 1;
    }

    TRELLIS_TOOL_INFO("stage1 sparse-structure decoder: bindings ok input=8->512 middle=2 upsample=2 output=32->1");
    if (dry_decode) {
        if (dry_latent_size <= 0 || dry_batch <= 0) {
            fprintf(stderr, "stage1 sparse-structure decoder: invalid dry decode dimensions\n");
            trellis_tensor_store_free(&store);
            return 1;
        }
        const size_t latent_count = (size_t) dry_batch * 8u *
            (size_t) dry_latent_size * (size_t) dry_latent_size * (size_t) dry_latent_size;
        float * latent = (float *) malloc(latent_count * sizeof(float));
        if (latent == NULL) {
            fprintf(stderr, "stage1 sparse-structure decoder: dry decode host allocation failed\n");
            trellis_tensor_store_free(&store);
            return 1;
        }
        for (size_t i = 0; i < latent_count; ++i) {
            latent[i] = 0.02f * (float) ((int) (i % 11) - 5);
        }
        float * logits = NULL;
        int output_size = 0;
        status = trellis_ss_decoder_forward_f32_host(&weights, latent, cuda, dry_batch, dry_latent_size, &logits, &output_size);
        free(latent);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "stage1 sparse-structure decoder: dry decode failed: %s\n", trellis_status_string(status));
            free(logits);
            trellis_tensor_store_free(&store);
            return 1;
        }
        size_t logits_count = (size_t) dry_batch * (size_t) output_size * (size_t) output_size * (size_t) output_size;
        size_t positive = 0;
        for (size_t i = 0; i < logits_count; ++i) {
            if (logits[i] > 0.0f) {
                ++positive;
            }
        }
        printf("  dry decode ok: latent=%d output=%d logits=%zu positive=%zu first=",
            dry_latent_size,
            output_size,
            logits_count,
            positive);
        const size_t n_print = logits_count < 6 ? logits_count : 6;
        for (size_t i = 0; i < n_print; ++i) {
            printf("%s%.6g", i == 0 ? "" : ",", logits[i]);
        }
        printf("\n");
        free(logits);
    }
    trellis_tensor_store_free(&store);
    return 0;
}

static int load_dino_store(
    const trellis_cuda_context * cuda,
    const char * dino_dir,
    int dry_dino_patch) {
    char path[4096];
    if (!make_join_path(dino_dir, "model.safetensors", path, sizeof(path))) {
        TRELLIS_TOOL_ERROR("dino: invalid model path");
        return 1;
    }

    trellis_tensor_store store;
    if (!trellis_tool_load_tensor_store_f32(cuda, "dino", path, true, 64, &store, NULL)) {
        return 1;
    }

    trellis_dino_vit_weights weights;
    char issue[256];
    trellis_status status = trellis_dino_vit_bind_weights(&store, &weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("dino: bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        trellis_tensor_store_free(&store);
        return 1;
    }

    TRELLIS_TOOL_INFO("dino: bindings ok layers=%d hidden=%d heads=%d head_dim=%d patch=%d registers=%d",
        TRELLIS_DINO_VIT_LAYERS,
        weights.hidden_size,
        weights.heads,
        weights.head_dim,
        weights.patch_size,
        weights.register_tokens_count);
    if (dry_dino_patch) {
        enum { IMAGE_H = 32, IMAGE_W = 32, BATCH = 1 };
        const size_t image_count = (size_t) BATCH * 3u * IMAGE_H * IMAGE_W;
        float * image = (float *) malloc(image_count * sizeof(float));
        if (image == NULL) {
            fprintf(stderr, "dino: dry image host allocation failed\n");
            trellis_tensor_store_free(&store);
            return 1;
        }
        for (size_t i = 0; i < image_count; ++i) {
            image[i] = 0.01f * (float) ((int) (i % 23) - 11);
        }
        float * tokens = NULL;
        int n_tokens = 0;
        status = trellis_dino_image_forward_f32_host(
            cuda, &weights, image, BATCH, IMAGE_H, IMAGE_W, NULL, NULL, &tokens, &n_tokens);
        free(image);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "dino: dry image encoder failed: %s\n", trellis_status_string(status));
            free(tokens);
            trellis_tensor_store_free(&store);
            return 1;
        }
        printf("  dry dino image ok: image=%dx%d tokens=%d hidden=%d first=",
            IMAGE_H,
            IMAGE_W,
            n_tokens,
            weights.hidden_size);
        for (int i = 0; i < 6; ++i) {
            printf("%s%.6g", i == 0 ? "" : ",", tokens[i]);
        }
        printf("\n");
        free(tokens);
    }
    trellis_tensor_store_free(&store);
    return 0;
}

static int mkdir_p(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) {
        return 0;
    }
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (tmp[0] != '\0' && mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return 0;
            }
            tmp[i] = '/';
        }
    }
    return mkdir(tmp, 0775) == 0 || errno == EEXIST;
}

static int write_voxel_frame(
    const char * out_dir,
    const float * logits,
    int resolution,
    float threshold,
    int step,
    const char * source,
    int append,
    int * out_count) {
    if (out_dir == NULL || logits == NULL || resolution <= 0 || step < 0 ||
        source == NULL || source[0] == '\0' || out_count == NULL) {
        return 0;
    }
    *out_count = 0;
    if (!mkdir_p(out_dir)) {
        fprintf(stderr, "failed to create %s\n", out_dir);
        return 0;
    }

    const int total = resolution * resolution * resolution;
    for (int i = 0; i < total; ++i) {
        if (logits[i] > threshold) {
            *out_count += 1;
        }
    }

    char coord_file[128];
    int name_n = snprintf(coord_file, sizeof(coord_file), "coords_%03d_%s.i32", step, source);
    if (name_n < 0 || (size_t) name_n >= sizeof(coord_file)) {
        return 0;
    }
    char coords_path[4096];
    if (!make_join_path(out_dir, coord_file, coords_path, sizeof(coords_path))) {
        return 0;
    }
    FILE * coords = fopen(coords_path, "wb");
    if (coords == NULL) {
        fprintf(stderr, "failed to open %s\n", coords_path);
        return 0;
    }
    for (int d = 0; d < resolution; ++d) {
        for (int h = 0; h < resolution; ++h) {
            for (int w = 0; w < resolution; ++w) {
                const int idx = (d * resolution + h) * resolution + w;
                if (logits[idx] > threshold) {
                    /* Match torch.argwhere(logits > 0)[:, [D, H, W]] from the
                     * original TRELLIS pipeline. The viewers map this order to
                     * raylib coordinates consistently. */
                    int32_t c[3] = { (int32_t) d, (int32_t) h, (int32_t) w };
                    if (fwrite(c, sizeof(int32_t), 3, coords) != 3) {
                        fclose(coords);
                        return 0;
                    }
                }
            }
        }
    }
    fclose(coords);

    char frames_path[4096];
    if (!make_join_path(out_dir, "frames.tsv", frames_path, sizeof(frames_path))) {
        return 0;
    }
    FILE * frames = fopen(frames_path, append ? "a" : "w");
    if (frames == NULL) {
        fprintf(stderr, "failed to open %s\n", frames_path);
        return 0;
    }
    if (!append) {
        fprintf(frames, "step\tsource\tresolution\tcount\tfile\n");
    }
    fprintf(frames, "%d\t%s\t%d\t%d\t%s\n", step, source, resolution, *out_count, coord_file);
    fclose(frames);
    return 1;
}

static int collect_voxel_coords_xyz(
    const float * logits,
    int resolution,
    float threshold,
    int32_t ** coords_out,
    int64_t * n_out) {
    if (logits == NULL || resolution <= 0 || coords_out == NULL || n_out == NULL) {
        return 0;
    }
    *coords_out = NULL;
    *n_out = 0;
    const int64_t total = (int64_t) resolution * (int64_t) resolution * (int64_t) resolution;
    int64_t count = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (logits[i] > threshold) {
            ++count;
        }
    }
    int32_t * coords = (int32_t *) malloc((size_t) count * 3u * sizeof(int32_t));
    if (coords == NULL && count != 0) {
        return 0;
    }
    int64_t row = 0;
    for (int d = 0; d < resolution; ++d) {
        for (int h = 0; h < resolution; ++h) {
            for (int w = 0; w < resolution; ++w) {
                const int64_t idx = ((int64_t) d * resolution + h) * (int64_t) resolution + w;
                if (logits[idx] <= threshold) {
                    continue;
                }
                coords[(size_t) row * 3u + 0u] = (int32_t) d;
                coords[(size_t) row * 3u + 1u] = (int32_t) h;
                coords[(size_t) row * 3u + 2u] = (int32_t) w;
                ++row;
            }
        }
    }
    *coords_out = coords;
    *n_out = count;
    return 1;
}

static int set_stage1_result(
    trellis_tool_stage1_result * result,
    const int32_t * coords_xyz,
    int64_t n_coords,
    int resolution,
    const float * cond,
    int cond_tokens) {
    if (result == NULL) {
        return 1;
    }
    if (coords_xyz == NULL || n_coords < 0 || cond == NULL || cond_tokens <= 0) {
        return 0;
    }
    trellis_tool_stage1_result_free(result);
    int32_t * coords_bxyz = (int32_t *) malloc((size_t) n_coords * 4u * sizeof(int32_t));
    float * cond_copy = (float *) malloc((size_t) cond_tokens * 1024u * sizeof(float));
    if ((coords_bxyz == NULL && n_coords != 0) || cond_copy == NULL) {
        free(coords_bxyz);
        free(cond_copy);
        return 0;
    }
    for (int64_t i = 0; i < n_coords; ++i) {
        coords_bxyz[(size_t) i * 4u + 0u] = 0;
        coords_bxyz[(size_t) i * 4u + 1u] = coords_xyz[(size_t) i * 3u + 0u];
        coords_bxyz[(size_t) i * 4u + 2u] = coords_xyz[(size_t) i * 3u + 1u];
        coords_bxyz[(size_t) i * 4u + 3u] = coords_xyz[(size_t) i * 3u + 2u];
    }
    memcpy(cond_copy, cond, (size_t) cond_tokens * 1024u * sizeof(float));
    result->coords_bxyz = coords_bxyz;
    result->n_coords = n_coords;
    result->resolution = resolution;
    result->cond = cond_copy;
    result->cond_tokens = cond_tokens;
    return 1;
}

void trellis_tool_stage1_result_free(trellis_tool_stage1_result * result) {
    if (result == NULL) {
        return;
    }
    free(result->coords_bxyz);
    free(result->cond);
    memset(result, 0, sizeof(*result));
}

static int write_voxel_snapshot(
    const char * out_dir,
    const float * logits,
    int resolution,
    float threshold,
    int * out_count) {
    return write_voxel_frame(out_dir, logits, resolution, threshold, 0, "x_t", 0, out_count);
}

static int max_pool_logits_to_resolution(
    const float * logits,
    int input_resolution,
    float threshold,
    int output_resolution,
    float ** pooled_out) {
    if (logits == NULL || input_resolution <= 0 || output_resolution <= 0 || pooled_out == NULL ||
        input_resolution % output_resolution != 0) {
        return 0;
    }
    *pooled_out = NULL;
    if (input_resolution == output_resolution) {
        return 1;
    }
    const int ratio = input_resolution / output_resolution;
    const size_t out_count = (size_t) output_resolution * (size_t) output_resolution * (size_t) output_resolution;
    float * pooled = (float *) malloc(out_count * sizeof(float));
    if (pooled == NULL) {
        return 0;
    }
    for (int d = 0; d < output_resolution; ++d) {
        for (int h = 0; h < output_resolution; ++h) {
            for (int w = 0; w < output_resolution; ++w) {
                int occupied = 0;
                for (int rd = 0; rd < ratio && !occupied; ++rd) {
                    for (int rh = 0; rh < ratio && !occupied; ++rh) {
                        for (int rw = 0; rw < ratio; ++rw) {
                            const int sd = d * ratio + rd;
                            const int sh = h * ratio + rh;
                            const int sw = w * ratio + rw;
                            const int idx = (sd * input_resolution + sh) * input_resolution + sw;
                            if (logits[idx] > threshold) {
                                occupied = 1;
                                break;
                            }
                        }
                    }
                }
                pooled[(size_t) (d * output_resolution + h) * (size_t) output_resolution + (size_t) w] =
                    occupied ? 1.0f : 0.0f;
            }
        }
    }
    *pooled_out = pooled;
    return 1;
}

typedef struct tensor_dump_stats {
    size_t count;
    size_t finite_count;
    size_t nan_count;
    size_t inf_count;
    float min_value;
    float max_value;
    double mean;
    double l2;
    uint64_t fnv64;
} tensor_dump_stats;

typedef struct tensor_dump_context {
    const char * dir;
    FILE * manifest;
    int enabled;
    int next_index;
} tensor_dump_context;

static uint64_t fnv1a_bytes(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static tensor_dump_stats tensor_stats_f32(const float * data, size_t count) {
    tensor_dump_stats s;
    memset(&s, 0, sizeof(s));
    s.count = count;
    s.min_value = 0.0f;
    s.max_value = 0.0f;
    s.fnv64 = data == NULL ? 0u : fnv1a_bytes(data, count * sizeof(float));
    double sum = 0.0;
    double l2 = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float v = data[i];
        if (isnan(v)) {
            ++s.nan_count;
            continue;
        }
        if (!isfinite(v)) {
            ++s.inf_count;
            continue;
        }
        if (s.finite_count == 0 || v < s.min_value) {
            s.min_value = v;
        }
        if (s.finite_count == 0 || v > s.max_value) {
            s.max_value = v;
        }
        ++s.finite_count;
        sum += (double) v;
        l2 += (double) v * (double) v;
    }
    s.mean = s.finite_count == 0 ? 0.0 : sum / (double) s.finite_count;
    s.l2 = sqrt(l2);
    return s;
}

static int dump_context_open(tensor_dump_context * dump, const char * dir) {
    if (dump == NULL) {
        return 0;
    }
    memset(dump, 0, sizeof(*dump));
    if (dir == NULL || dir[0] == '\0') {
        return 1;
    }
    if (!mkdir_p(dir)) {
        fprintf(stderr, "intermediate dump: failed to create %s\n", dir);
        return 0;
    }
    char manifest_path[4096];
    if (!make_join_path(dir, "intermediates.tsv", manifest_path, sizeof(manifest_path))) {
        return 0;
    }
    FILE * f = fopen(manifest_path, "w");
    if (f == NULL) {
        fprintf(stderr, "intermediate dump: failed to open %s\n", manifest_path);
        return 0;
    }
    fprintf(f, "index\tname\tshape\tcount\tfinite\tnan\tinf\tmin\tmean\tmax\tl2\tfnv64\tfile\n");
    dump->dir = dir;
    dump->manifest = f;
    dump->enabled = 1;
    dump->next_index = 0;
    return 1;
}

static void dump_context_close(tensor_dump_context * dump) {
    if (dump != NULL && dump->manifest != NULL) {
        fclose(dump->manifest);
        dump->manifest = NULL;
    }
}

static int check_or_dump_tensor_f32(
    tensor_dump_context * dump,
    const char * name,
    const char * shape,
    const float * data,
    size_t count) {
    if (name == NULL || data == NULL) {
        fprintf(stderr, "tensor check: invalid tensor %s\n", name == NULL ? "(null)" : name);
        return 0;
    }
    tensor_dump_stats s = tensor_stats_f32(data, count);
    if (s.finite_count != s.count || s.nan_count != 0 || s.inf_count != 0) {
        fprintf(stderr,
            "tensor check failed: %s shape=%s count=%zu finite=%zu nan=%zu inf=%zu hash=%016llx\n",
            name,
            shape == NULL ? "" : shape,
            s.count,
            s.finite_count,
            s.nan_count,
            s.inf_count,
            (unsigned long long) s.fnv64);
        return 0;
    }
    if (dump == NULL || !dump->enabled || dump->manifest == NULL) {
        return 1;
    }

    char file_name[512];
    int n = snprintf(file_name, sizeof(file_name), "%03d_%s.f32", dump->next_index, name);
    if (n < 0 || (size_t) n >= sizeof(file_name)) {
        return 0;
    }
    for (char * p = file_name; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\' || *p == ':' || *p == '\t' || *p == ' ') {
            *p = '_';
        }
    }
    char path[4096];
    if (!make_join_path(dump->dir, file_name, path, sizeof(path))) {
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "intermediate dump: failed to open %s\n", path);
        return 0;
    }
    const size_t wrote = fwrite(data, sizeof(float), count, f);
    fclose(f);
    if (wrote != count) {
        fprintf(stderr, "intermediate dump: short write %s\n", path);
        return 0;
    }

    fprintf(
        dump->manifest,
        "%d\t%s\t%s\t%zu\t%zu\t%zu\t%zu\t%.9g\t%.17g\t%.9g\t%.17g\t%016llx\t%s\n",
        dump->next_index,
        name,
        shape == NULL ? "" : shape,
        s.count,
        s.finite_count,
        s.nan_count,
        s.inf_count,
        s.min_value,
        s.mean,
        s.max_value,
        s.l2,
        (unsigned long long) s.fnv64,
        file_name);
    fflush(dump->manifest);
    dump->next_index += 1;
    return 1;
}

static void shape_2d(char * dst, size_t dst_size, int a, int b) {
    snprintf(dst, dst_size, "%dx%d", a, b);
}

static void shape_3d(char * dst, size_t dst_size, int a, int b, int c) {
    snprintf(dst, dst_size, "%dx%dx%d", a, b, c);
}

static void shape_4d(char * dst, size_t dst_size, int a, int b, int c, int d) {
    snprintf(dst, dst_size, "%dx%dx%dx%d", a, b, c, d);
}

static void shape_5d(char * dst, size_t dst_size, int a, int b, int c, int d, int e) {
    snprintf(dst, dst_size, "%dx%dx%dx%dx%d", a, b, c, d, e);
}

static void ncdhw_to_token_channels(
    const float * src,
    float * dst,
    int channels,
    int size) {
    for (int d = 0; d < size; ++d) {
        for (int h = 0; h < size; ++h) {
            for (int w = 0; w < size; ++w) {
                const size_t token = ((size_t) d * (size_t) size + (size_t) h) * (size_t) size + (size_t) w;
                for (int c = 0; c < channels; ++c) {
                    const size_t src_i =
                        (((size_t) c * (size_t) size + (size_t) d) * (size_t) size + (size_t) h) *
                            (size_t) size +
                        (size_t) w;
                    dst[token * (size_t) channels + (size_t) c] = src[src_i];
                }
            }
        }
    }
}

static void token_channels_to_ncdhw(
    const float * src,
    float * dst,
    int channels,
    int size) {
    for (int d = 0; d < size; ++d) {
        for (int h = 0; h < size; ++h) {
            for (int w = 0; w < size; ++w) {
                const size_t token = ((size_t) d * (size_t) size + (size_t) h) * (size_t) size + (size_t) w;
                for (int c = 0; c < channels; ++c) {
                    const size_t dst_i =
                        (((size_t) c * (size_t) size + (size_t) d) * (size_t) size + (size_t) h) *
                            (size_t) size +
                        (size_t) w;
                    dst[dst_i] = src[token * (size_t) channels + (size_t) c];
                }
            }
        }
    }
}

static trellis_status run_flow_once_host(
    const trellis_cuda_context * cuda,
    const trellis_dit_flow_weights * weights,
    const float * latent,
    const float * context_data,
    const float * cos_data,
    const float * sin_data,
    float timestep,
    int latent_size,
    int cond_tokens,
    float * out_pred) {
    if (cuda == NULL || weights == NULL || latent == NULL || context_data == NULL ||
        cos_data == NULL || sin_data == NULL || out_pred == NULL ||
        latent_size <= 0 || cond_tokens <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int batch = 1;
    const int tokens = latent_size * latent_size * latent_size;
    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false) + 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->in_channels, tokens, batch);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch);
    struct ggml_tensor * c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->cond_channels, cond_tokens, batch);
    struct ggml_tensor * cos_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, weights->head_dim / 2, tokens, 1);
    struct ggml_tensor * sin_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, weights->head_dim / 2, tokens, 1);
    struct ggml_tensor * y = trellis_dit_flow_forward(ctx, x, t, c, cos_phase, sin_phase, weights);
    if (x == NULL || t == NULL || c == NULL || cos_phase == NULL || sin_phase == NULL || y == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_ERROR;
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, y);

    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(cuda);
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc != NULL) {
            ggml_gallocr_free(alloc);
        }
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    ggml_backend_tensor_set(x, latent, 0, ggml_nbytes(x));
    if (weights->n_blocks > 0) {
        const float model_timestep = 1000.0f * timestep;
        ggml_backend_tensor_set(t, &model_timestep, 0, ggml_nbytes(t));
        const int debug_parts = weights->debug_block_parts < 0 ? 3 : weights->debug_block_parts;
        if (debug_parts >= 1 && !weights->debug_disable_rope) {
            ggml_backend_tensor_set(cos_phase, cos_data, 0, ggml_nbytes(cos_phase));
            ggml_backend_tensor_set(sin_phase, sin_data, 0, ggml_nbytes(sin_phase));
        }
        if (debug_parts >= 2) {
            ggml_backend_tensor_set(c, context_data, 0, ggml_nbytes(c));
        }
    }

    trellis_status status = trellis_cuda_compute_graph(cuda, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(y, out_pred, 0, ggml_nbytes(y));
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return status;
}

static int load_flow_for_stage1(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights) {
    char path[4096];
    if (!choose_path(model_dir, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", path, sizeof(path))) {
        return 0;
    }
    if (!trellis_tool_load_tensor_store_f32(
            cuda,
            "stage1 sparse-structure flow",
            path,
            true,
            64,
            store,
            NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_ss_flow_bind_weights(store, weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("stage1 sparse-structure flow: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_TOOL_INFO("stage1 sparse-structure flow: ready blocks=%d channels=%d",
        weights->n_blocks, weights->in_channels);
    return 1;
}

static int load_decoder_for_stage1(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    trellis_tensor_store * store,
    trellis_ss_decoder_weights * weights) {
    char path[4096];
    if (!choose_path(model_dir, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", path, sizeof(path))) {
        return 0;
    }
    if (!trellis_tool_load_tensor_store_f32(
            cuda,
            "stage1 sparse-structure decoder",
            path,
            false,
            64,
            store,
            NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_ss_decoder_bind_weights(store, weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("stage1 sparse-structure decoder: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_TOOL_INFO("stage1 sparse-structure decoder: ready");
    return 1;
}

static int load_dino_for_stage1(
    const trellis_cuda_context * cuda,
    const char * dino_dir,
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights);

int trellis_tool_stage1_weights_load(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * dino_dir,
    trellis_tool_stage1_weights * weights) {
    if (weights == NULL) {
        return 0;
    }
    memset(weights, 0, sizeof(*weights));
    if (cuda == NULL || model_dir == NULL || dino_dir == NULL) {
        return 0;
    }
    if (!load_dino_for_stage1(cuda, dino_dir, &weights->dino_store, &weights->dino)) {
        trellis_tool_stage1_weights_free(weights);
        return 0;
    }
    weights->has_dino = 1;
    if (!load_flow_for_stage1(cuda, model_dir, &weights->flow_store, &weights->flow)) {
        trellis_tool_stage1_weights_free(weights);
        return 0;
    }
    weights->has_flow = 1;
    if (!load_decoder_for_stage1(cuda, model_dir, &weights->decoder_store, &weights->decoder)) {
        trellis_tool_stage1_weights_free(weights);
        return 0;
    }
    weights->has_decoder = 1;
    return 1;
}

void trellis_tool_stage1_weights_free(trellis_tool_stage1_weights * weights) {
    if (weights == NULL) {
        return;
    }
    trellis_tensor_store_free(&weights->decoder_store);
    trellis_tensor_store_free(&weights->flow_store);
    trellis_tensor_store_free(&weights->dino_store);
    memset(weights, 0, sizeof(*weights));
}

static int copy_tensor_f32(const struct ggml_tensor * tensor, float * dst, size_t count) {
    if (tensor == NULL || dst == NULL || count > (size_t) ggml_nelements(tensor)) {
        return 0;
    }
    ggml_backend_tensor_get(tensor, dst, 0, count * sizeof(float));
    return 1;
}

static void linear_vec_f32(
    const float * x,
    const float * w,
    const float * b,
    int in,
    int out,
    float * y) {
    for (int o = 0; o < out; ++o) {
        float acc = b == NULL ? 0.0f : b[o];
        for (int i = 0; i < in; ++i) {
            acc += w[(size_t) o * (size_t) in + (size_t) i] * x[i];
        }
        y[o] = acc;
    }
}

static int dump_flow_time_mod_debug(
    tensor_dump_context * dump,
    const trellis_dit_flow_weights * flow,
    float timestep) {
    if (dump == NULL || !dump->enabled || flow == NULL) {
        return 1;
    }
    const int freq_dim = flow->time_frequency_dim;
    const int channels = flow->model_channels;
    const int mod_channels = flow->mod_channels;
    float * t_freq = (float *) malloc((size_t) freq_dim * sizeof(float));
    float * w0 = (float *) malloc((size_t) freq_dim * (size_t) channels * sizeof(float));
    float * b0 = (float *) malloc((size_t) channels * sizeof(float));
    float * h0 = (float *) malloc((size_t) channels * sizeof(float));
    float * w1 = (float *) malloc((size_t) channels * (size_t) channels * sizeof(float));
    float * b1 = (float *) malloc((size_t) channels * sizeof(float));
    float * t_emb = (float *) malloc((size_t) channels * sizeof(float));
    float * wad = (float *) malloc((size_t) channels * (size_t) mod_channels * sizeof(float));
    float * bad = (float *) malloc((size_t) mod_channels * sizeof(float));
    float * mod6 = (float *) malloc((size_t) mod_channels * sizeof(float));
    float * block_mod = (float *) malloc((size_t) mod_channels * sizeof(float));
    float * block_param = (float *) malloc((size_t) mod_channels * sizeof(float));
    if (t_freq == NULL || w0 == NULL || b0 == NULL || h0 == NULL || w1 == NULL || b1 == NULL ||
        t_emb == NULL || wad == NULL || bad == NULL || mod6 == NULL || block_mod == NULL || block_param == NULL) {
        free(t_freq); free(w0); free(b0); free(h0); free(w1); free(b1);
        free(t_emb); free(wad); free(bad); free(mod6); free(block_mod); free(block_param);
        return 0;
    }
    trellis_timestep_embedding_f32(&timestep, 1, freq_dim, 10000.0f, t_freq);
    int ok = copy_tensor_f32(flow->t_embedder_0_w, w0, (size_t) freq_dim * (size_t) channels) &&
        copy_tensor_f32(flow->t_embedder_0_b, b0, (size_t) channels) &&
        copy_tensor_f32(flow->t_embedder_2_w, w1, (size_t) channels * (size_t) channels) &&
        copy_tensor_f32(flow->t_embedder_2_b, b1, (size_t) channels) &&
        copy_tensor_f32(flow->adaln_w, wad, (size_t) channels * (size_t) mod_channels) &&
        copy_tensor_f32(flow->adaln_b, bad, (size_t) mod_channels) &&
        copy_tensor_f32(flow->blocks[0].modulation, block_param, (size_t) mod_channels);
    if (ok) {
        linear_vec_f32(t_freq, w0, b0, freq_dim, channels, h0);
        for (int i = 0; i < channels; ++i) {
            h0[i] = h0[i] / (1.0f + expf(-h0[i]));
        }
        linear_vec_f32(h0, w1, b1, channels, channels, t_emb);
        for (int i = 0; i < channels; ++i) {
            h0[i] = t_emb[i] / (1.0f + expf(-t_emb[i]));
        }
        linear_vec_f32(h0, wad, bad, channels, mod_channels, mod6);
        for (int i = 0; i < mod_channels; ++i) {
            block_mod[i] = block_param[i] + mod6[i];
        }
        char shape[64];
        shape_2d(shape, sizeof(shape), 1, freq_dim);
        ok = check_or_dump_tensor_f32(dump, "flow_t_freq", shape, t_freq, (size_t) freq_dim);
        shape_2d(shape, sizeof(shape), 1, channels);
        ok = ok && check_or_dump_tensor_f32(dump, "flow_t_emb_base", shape, t_emb, (size_t) channels);
        shape_2d(shape, sizeof(shape), 1, mod_channels);
        ok = ok && check_or_dump_tensor_f32(dump, "flow_mod6", shape, mod6, (size_t) mod_channels);
        ok = ok && check_or_dump_tensor_f32(dump, "flow_block0_mod", shape, block_mod, (size_t) mod_channels);
    }
    free(t_freq); free(w0); free(b0); free(h0); free(w1); free(b1);
    free(t_emb); free(wad); free(bad); free(mod6); free(block_mod); free(block_param);
    return ok;
}

static int load_dino_for_stage1(
    const trellis_cuda_context * cuda,
    const char * dino_dir,
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights) {
    char path[4096];
    if (!make_join_path(dino_dir, "model.safetensors", path, sizeof(path))) {
        TRELLIS_TOOL_ERROR("stage1 image: invalid dino model path");
        return 0;
    }
    if (!trellis_tool_load_tensor_store_f32(cuda, "stage1 dino image encoder", path, true, 64, store, NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_dino_vit_bind_weights(store, weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("stage1 dino image encoder: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_TOOL_INFO("stage1 dino image encoder: ready");
    return 1;
}

static float lcg_uniform(uint32_t * state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float) ((*state >> 8) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

static void fill_gaussian_latent(float * dst, size_t count, uint32_t seed) {
    if (dst == NULL) {
        return;
    }
    uint32_t state = seed == 0 ? 1u : seed;
    const float two_pi = 6.2831853071795864769f;
    for (size_t i = 0; i < count; i += 2) {
        float u1 = lcg_uniform(&state);
        float u2 = lcg_uniform(&state);
        if (u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        const float r = sqrtf(-2.0f * logf(u1));
        dst[i] = r * cosf(two_pi * u2);
        if (i + 1 < count) {
            dst[i + 1] = r * sinf(two_pi * u2);
        }
    }
}

static float clampf_local(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float sinc_f32(float x) {
    const float ax = fabsf(x);
    if (ax < 1e-6f) {
        return 1.0f;
    }
    const float pix = 3.14159265358979323846f * x;
    return sinf(pix) / pix;
}

static float lanczos3_f32(float x) {
    const float ax = fabsf(x);
    if (ax >= 3.0f) {
        return 0.0f;
    }
    return sinc_f32(x) * sinc_f32(x / 3.0f);
}

static int preprocess_image_for_dino(
    const char * image_path,
    int edge,
    float ** image_out,
    int * src_w_out,
    int * src_h_out) {
    if (image_path == NULL || edge <= 0 || image_out == NULL) {
        return 0;
    }
    *image_out = NULL;
    if (src_w_out != NULL) *src_w_out = 0;
    if (src_h_out != NULL) *src_h_out = 0;

    int src_w = 0;
    int src_h = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(image_path, &src_w, &src_h, &comp, 4);
    if (rgba == NULL || src_w <= 0 || src_h <= 0) {
        fprintf(stderr, "stage1 image: failed to load image %s\n", image_path);
        stbi_image_free(rgba);
        return 0;
    }

    float * image = (float *) malloc(3u * (size_t) edge * (size_t) edge * sizeof(float));
    if (image == NULL) {
        stbi_image_free(rgba);
        return 0;
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    int crop_left = 0;
    int crop_top = 0;
    int crop_size = src_w > src_h ? src_w : src_h;
    int use_alpha_composite = 0;
    int min_x = src_w, min_y = src_h, max_x = -1, max_y = -1;
    for (int yy = 0; yy < src_h; ++yy) {
        for (int xx = 0; xx < src_w; ++xx) {
            const unsigned char * p = rgba + (((size_t) yy * (size_t) src_w + (size_t) xx) * 4u);
            if (p[3] != 255) {
                use_alpha_composite = 1;
            }
            if (p[3] > 8) {
                if (xx < min_x) min_x = xx;
                if (yy < min_y) min_y = yy;
                if (xx + 1 > max_x) max_x = xx + 1;
                if (yy + 1 > max_y) max_y = yy + 1;
            }
        }
    }
    if (use_alpha_composite && max_x > min_x && max_y > min_y) {
        const float center_x = 0.5f * (float) (min_x + max_x);
        const float center_y = 0.5f * (float) (min_y + max_y);
        int side = max_x - min_x > max_y - min_y ? max_x - min_x : max_y - min_y;
        side = (int) floorf((float) side * 1.16f + 0.5f);
        if (side < 1) side = 1;
        const int max_side = 2 * (src_w > src_h ? src_w : src_h);
        if (side > max_side) side = max_side;
        crop_left = (int) floorf(center_x - 0.5f * (float) side + 0.5f);
        crop_top = (int) floorf(center_y - 0.5f * (float) side + 0.5f);
        crop_size = side;
    } else {
        crop_left = 0;
        crop_top = 0;
        crop_size = src_w;
        if (src_h > crop_size) crop_size = src_h;
    }
    for (int y = 0; y < edge; ++y) {
        for (int x = 0; x < edge; ++x) {
            const float sx = ((float) x + 0.5f) * (float) crop_size / (float) edge - 0.5f;
            const float sy = ((float) y + 0.5f) * (float) crop_size / (float) edge - 0.5f;
            const int x_begin = (int) floorf(sx - 3.0f + 1.0f);
            const int x_end = (int) floorf(sx + 3.0f);
            const int y_begin = (int) floorf(sy - 3.0f + 1.0f);
            const int y_end = (int) floorf(sy + 3.0f);
            float acc[3] = {0.0f, 0.0f, 0.0f};
            float wsum = 0.0f;
            for (int yy = y_begin; yy <= y_end; ++yy) {
                if (yy < 0 || yy >= crop_size) {
                    continue;
                }
                const float wy = lanczos3_f32(sy - (float) yy);
                if (wy == 0.0f) {
                    continue;
                }
                for (int xx = x_begin; xx <= x_end; ++xx) {
                    if (xx < 0 || xx >= crop_size) {
                        continue;
                    }
                    const float wx = lanczos3_f32(sx - (float) xx);
                    const float w = wx * wy;
                    if (w == 0.0f) {
                        continue;
                    }
                    const int src_x = crop_left + xx;
                    const int src_y = crop_top + yy;
                    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
                        const unsigned char * p = rgba + (((size_t) src_y * (size_t) src_w + (size_t) src_x) * 4u);
                        const float alpha = use_alpha_composite ? (float) p[3] / 255.0f : 1.0f;
                        for (int c = 0; c < 3; ++c) {
                            acc[c] += w * (float) p[c] * alpha;
                        }
                    }
                    wsum += w;
                }
            }
            for (int c = 0; c < 3; ++c) {
                float v = wsum == 0.0f ? 0.0f : acc[c] / wsum;
                v = clampf_local(v, 0.0f, 255.0f);
                const float rgb = v / 255.0f;
                image[((size_t) c * (size_t) edge + (size_t) y) * (size_t) edge + (size_t) x] =
                    (rgb - mean[c]) / std[c];
            }
        }
    }

    stbi_image_free(rgba);
    if (src_w_out != NULL) *src_w_out = src_w;
    if (src_h_out != NULL) *src_h_out = src_h;
    *image_out = image;
    return 1;
}

static int run_dino_condition(
    const trellis_cuda_context * cuda,
    const char * dino_dir,
    const char * image_path,
    int cond_resolution,
    tensor_dump_context * dump,
    const trellis_dino_vit_weights * preloaded_dino,
    float ** context_out,
    int * cond_tokens_out) {
    if (cuda == NULL || dino_dir == NULL || image_path == NULL ||
        cond_resolution <= 0 || context_out == NULL || cond_tokens_out == NULL) {
        return 0;
    }
    *context_out = NULL;
    *cond_tokens_out = 0;

    trellis_tensor_store dino_store;
    memset(&dino_store, 0, sizeof(dino_store));
    trellis_dino_vit_weights dino_local;
    const trellis_dino_vit_weights * dino = preloaded_dino;
    int owns_dino_store = 0;
    int rc = 0;
    float * image = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    float * output_tokens = NULL;

    if (dino == NULL) {
        if (!load_dino_for_stage1(cuda, dino_dir, &dino_store, &dino_local)) {
            goto cleanup;
        }
        dino = &dino_local;
        owns_dino_store = 1;
    }
    if (cond_resolution % dino->patch_size != 0) {
        fprintf(stderr, "stage1 image: --cond-resolution must be divisible by DINO patch size %d\n", dino->patch_size);
        goto cleanup;
    }

    int src_w = 0;
    int src_h = 0;
    if (!preprocess_image_for_dino(image_path, cond_resolution, &image, &src_w, &src_h)) {
        goto cleanup;
    }
    TRELLIS_TOOL_INFO("stage1 image: preprocessed %s from %dx%d to %dx%d",
        image_path, src_w, src_h, cond_resolution, cond_resolution);
    char shape[64];
    shape_4d(shape, sizeof(shape), 1, 3, cond_resolution, cond_resolution);
    if (!check_or_dump_tensor_f32(dump, "dino_image_nchw", shape, image,
            3u * (size_t) cond_resolution * (size_t) cond_resolution)) {
        goto cleanup;
    }

    const int patches_h = cond_resolution / dino->patch_size;
    const int patches_w = cond_resolution / dino->patch_size;
    const int n_special = 1 + dino->register_tokens_count;
    const int n_patches = patches_h * patches_w;
    const int total_tokens = n_special + n_patches;
    const size_t token_count = (size_t) total_tokens * (size_t) dino->hidden_size;
    cos_phase = (float *) malloc((size_t) total_tokens * (size_t) dino->head_dim * sizeof(float));
    sin_phase = (float *) malloc((size_t) total_tokens * (size_t) dino->head_dim * sizeof(float));
    if (cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "stage1 image: dino host allocation failed\n");
        goto cleanup;
    }

    trellis_status status = trellis_dino_rope_2d_phases_f32(
        n_special,
        patches_h,
        patches_w,
        dino->head_dim,
        1.0f,
        100.0f,
        cos_phase,
        sin_phase,
        (size_t) total_tokens * (size_t) dino->head_dim);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "stage1 image: dino rope phases failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }
    shape_3d(shape, sizeof(shape), total_tokens, dino->head_dim, 1);
    if (!check_or_dump_tensor_f32(dump, "dino_rope_cos", shape, cos_phase,
            (size_t) total_tokens * (size_t) dino->head_dim) ||
        !check_or_dump_tensor_f32(dump, "dino_rope_sin", shape, sin_phase,
            (size_t) total_tokens * (size_t) dino->head_dim)) {
        goto cleanup;
    }

    TRELLIS_TOOL_INFO("stage1 image: running DINO image encoder graph tokens=%d layers=%d",
        total_tokens, TRELLIS_DINO_VIT_LAYERS);
    int got_tokens = 0;
    status = trellis_dino_image_forward_f32_host(
        cuda, dino, image, 1, cond_resolution, cond_resolution, cos_phase, sin_phase, &output_tokens, &got_tokens);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "stage1 image: dino image encoder failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }
    if (got_tokens != total_tokens) {
        fprintf(stderr, "stage1 image: dino token count mismatch got %d expected %d\n", got_tokens, total_tokens);
        goto cleanup;
    }
    shape_3d(shape, sizeof(shape), 1, total_tokens, dino->hidden_size);
    if (!check_or_dump_tensor_f32(dump, "dino_context_tokens", shape, output_tokens, token_count)) {
        goto cleanup;
    }

    TRELLIS_TOOL_INFO("stage1 image: DINO %s %dx%d -> %dx%d patches=%d tokens=%d hidden=%d",
        image_path, src_w, src_h, cond_resolution, cond_resolution, n_patches, total_tokens, dino->hidden_size);
    *context_out = output_tokens;
    *cond_tokens_out = total_tokens;
    output_tokens = NULL;
    rc = 1;

cleanup:
    free(image);
    free(cos_phase);
    free(sin_phase);
    free(output_tokens);
    if (owns_dino_store) {
        trellis_tensor_store_free(&dino_store);
    }
    return rc;
}

static int run_stage1_image_voxels(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * input_cond_path,
    const char * input_neg_cond_path,
    const char * input_latent_path,
    const char * out_dir,
    const char * dump_dir,
    int latent_size,
    int steps,
    int cond_resolution,
    int sparse_resolution,
    uint32_t seed,
    int flow_blocks_override,
    int flow_block_parts_override,
    int flow_no_rope,
    float threshold,
    const trellis_tool_stage1_weights * preloaded_weights,
    trellis_tool_stage1_frame_callback frame_callback,
    void * frame_user_data,
    trellis_tool_stage1_result * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (cuda == NULL || model_dir == NULL || dino_dir == NULL ||
        (image_path == NULL && input_cond_path == NULL) ||
        latent_size <= 0 || steps <= 0 || cond_resolution <= 0 || sparse_resolution <= 0) {
        return 1;
    }

    tensor_dump_context dump;
    if (!dump_context_open(&dump, dump_dir)) {
        return 1;
    }

    TRELLIS_TOOL_INFO(
        "stage1 image: cli inference latent=%d^3 steps=%d cond_resolution=%d sparse_resolution=%d seed=%u",
        latent_size,
        steps,
        cond_resolution,
        sparse_resolution,
        (unsigned) seed);

    float * context = NULL;
    int cond_tokens = 0;
    if (input_cond_path != NULL) {
        size_t context_values = 0;
        if (!read_f32_file_alloc(input_cond_path, &context, &context_values)) {
            dump_context_close(&dump);
            return 1;
        }
        if ((context_values % 1024u) != 0) {
            fprintf(stderr, "stage1 image: --input-cond-f32 count %zu is not divisible by 1024\n", context_values);
            free(context);
            dump_context_close(&dump);
            return 1;
        }
        cond_tokens = (int) (context_values / 1024u);
        char shape[64];
        shape_3d(shape, sizeof(shape), 1, cond_tokens, 1024);
        if (!check_or_dump_tensor_f32(&dump, "input_cond_context", shape, context, context_values)) {
            free(context);
            dump_context_close(&dump);
            return 1;
        }
        TRELLIS_TOOL_INFO("stage1 image: using reference cond %s tokens=%d hidden=1024", input_cond_path, cond_tokens);
    } else {
        const trellis_dino_vit_weights * dino_weights =
            preloaded_weights != NULL && preloaded_weights->has_dino ? &preloaded_weights->dino : NULL;
        if (!run_dino_condition(cuda, dino_dir, image_path, cond_resolution, &dump, dino_weights, &context, &cond_tokens)) {
            free(context);
            dump_context_close(&dump);
            return 1;
        }
    }

    trellis_tensor_store flow_store;
    trellis_tensor_store decoder_store;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_dit_flow_weights flow;
    trellis_ss_decoder_weights decoder;
    memset(&flow, 0, sizeof(flow));
    memset(&decoder, 0, sizeof(decoder));
    int owns_flow_store = 0;
    int owns_decoder_store = 0;
    int rc = 1;

    float * neg_context = NULL;
    float * latent = NULL;
    float * pred_pos = NULL;
    float * pred_neg = NULL;
    float * pred = NULL;
    float * next = NULL;
    float * x0 = NULL;
    float * latent_ncdhw = NULL;
    float * input_latent_ncdhw = NULL;
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;

    if (preloaded_weights != NULL && preloaded_weights->has_flow && preloaded_weights->has_decoder) {
        flow = preloaded_weights->flow;
        decoder = preloaded_weights->decoder;
    } else {
        if (!load_flow_for_stage1(cuda, model_dir, &flow_store, &flow)) {
            goto cleanup;
        }
        owns_flow_store = 1;
        if (!load_decoder_for_stage1(cuda, model_dir, &decoder_store, &decoder)) {
            goto cleanup;
        }
        owns_decoder_store = 1;
    }
    if (flow_blocks_override >= 0) {
        if (flow_blocks_override > flow.n_blocks) {
            fprintf(stderr, "stage1 image: --flow-blocks %d exceeds checkpoint blocks %d\n",
                flow_blocks_override,
                flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = flow_blocks_override;
        TRELLIS_TOOL_INFO("stage1 image: debug flow blocks override=%d", flow.n_blocks);
    }
    if (flow_block_parts_override >= 0) {
        if (flow_block_parts_override > 3) {
            fprintf(stderr, "stage1 image: --flow-block-parts must be in [0,3]\n");
            goto cleanup;
        }
        flow.debug_block_parts = flow_block_parts_override;
        TRELLIS_TOOL_INFO("stage1 image: debug flow block parts override=%d", flow.debug_block_parts);
    }
    if (flow_no_rope) {
        flow.debug_disable_rope = 1;
        TRELLIS_TOOL_INFO("stage1 image: debug flow RoPE disabled");
    }
    if (flow.cond_channels != 1024) {
        fprintf(stderr, "stage1 image: unexpected flow cond channels %d\n", flow.cond_channels);
        goto cleanup;
    }

    const int tokens = latent_size * latent_size * latent_size;
    const size_t latent_count = (size_t) flow.in_channels * (size_t) tokens;
    const size_t context_count = (size_t) flow.cond_channels * (size_t) cond_tokens;
    const size_t phase_count = (size_t) (flow.head_dim / 2) * (size_t) tokens;
    if (input_neg_cond_path != NULL) {
        neg_context = (float *) malloc(context_count * sizeof(float));
        if (neg_context == NULL || !read_f32_file_exact(input_neg_cond_path, neg_context, context_count)) {
            fprintf(stderr, "stage1 image: failed to read --input-neg-cond-f32 %s\n", input_neg_cond_path);
            goto cleanup;
        }
    } else {
        neg_context = (float *) calloc(context_count, sizeof(float));
    }
    latent = (float *) malloc(latent_count * sizeof(float));
    pred_pos = (float *) malloc(latent_count * sizeof(float));
    pred_neg = (float *) malloc(latent_count * sizeof(float));
    pred = (float *) malloc(latent_count * sizeof(float));
    next = (float *) malloc(latent_count * sizeof(float));
    x0 = (float *) malloc(latent_count * sizeof(float));
    latent_ncdhw = (float *) malloc(latent_count * sizeof(float));
    input_latent_ncdhw = (float *) malloc(latent_count * sizeof(float));
    pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (neg_context == NULL || latent == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL ||
        next == NULL || x0 == NULL || latent_ncdhw == NULL || input_latent_ncdhw == NULL ||
        pairs == NULL || cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "stage1 image: host allocation failed\n");
        goto cleanup;
    }

    if (input_latent_path != NULL) {
        if (!read_f32_file_exact(input_latent_path, input_latent_ncdhw, latent_count)) {
            fprintf(stderr, "stage1 image: failed to read --input-latent-f32 %s\n", input_latent_path);
            goto cleanup;
        }
        ncdhw_to_token_channels(input_latent_ncdhw, latent, flow.in_channels, latent_size);
        TRELLIS_TOOL_INFO("stage1 image: using reference latent %s layout=NCDHW -> token-major", input_latent_path);
    } else {
        fill_gaussian_latent(latent, latent_count, seed);
        token_channels_to_ncdhw(latent, input_latent_ncdhw, flow.in_channels, latent_size);
    }
    trellis_status status = trellis_flow_timestep_pairs_f32(steps, 5.0f, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_phases_f32(latent_size, flow.head_dim, 1.0f, 10000.0f, cos_phase, sin_phase, phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "stage1 image: schedule/rope failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }
    if (!dump_flow_time_mod_debug(&dump, &flow, 1000.0f * pairs[0])) {
        fprintf(stderr, "stage1 image: flow time modulation debug dump failed\n");
        goto cleanup;
    }
    char shape[96];
    shape_3d(shape, sizeof(shape), 1, tokens, flow.in_channels);
    if (!check_or_dump_tensor_f32(&dump, "flow_latent_initial", shape, latent, latent_count)) {
        goto cleanup;
    }
    shape_5d(shape, sizeof(shape), 1, flow.in_channels, latent_size, latent_size, latent_size);
    if (!check_or_dump_tensor_f32(&dump, "flow_latent_initial_ncdhw", shape, input_latent_ncdhw, latent_count)) {
        goto cleanup;
    }
    shape_3d(shape, sizeof(shape), 1, cond_tokens, flow.cond_channels);
    if (!check_or_dump_tensor_f32(&dump, "flow_cond_context", shape, context, context_count) ||
        !check_or_dump_tensor_f32(&dump, "flow_neg_context", shape, neg_context, context_count)) {
        goto cleanup;
    }
    shape_3d(shape, sizeof(shape), tokens, flow.head_dim / 2, 1);
    if (!check_or_dump_tensor_f32(&dump, "flow_rope_cos", shape, cos_phase, phase_count) ||
        !check_or_dump_tensor_f32(&dump, "flow_rope_sin", shape, sin_phase, phase_count)) {
        goto cleanup;
    }

    for (int step = 0; step < steps; ++step) {
        const int64_t step_start_us = ggml_time_us();
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        TRELLIS_TOOL_DEBUG("stage1 image: step %d/%d running conditional flow t=%.6g", step + 1, steps, t);
        status = run_flow_once_host(cuda, &flow, latent, context, cos_phase, sin_phase, t, latent_size, cond_tokens, pred_pos);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "stage1 image: conditional flow step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto cleanup;
        }
        char tensor_name[128];
        shape_3d(shape, sizeof(shape), 1, tokens, flow.out_channels);
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_flow_pred_pos", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, pred_pos, latent_count)) {
            goto cleanup;
        }
        TRELLIS_TOOL_DEBUG("stage1 image: step %d/%d running unconditional flow t=%.6g", step + 1, steps, t);
        status = run_flow_once_host(cuda, &flow, latent, neg_context, cos_phase, sin_phase, t, latent_size, cond_tokens, pred_neg);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "stage1 image: unconditional flow step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto cleanup;
        }
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_flow_pred_neg", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, pred_neg, latent_count)) {
            goto cleanup;
        }

        if (t >= 0.6f && t <= 1.0f) {
            trellis_flow_cfg_rescale_combine_f32(latent, pred_pos, pred_neg, 1, latent_count, 1e-5f, t, 7.5f, 0.7f, pred);
        } else {
            memcpy(pred, pred_pos, latent_count * sizeof(float));
        }
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_flow_pred_cfg", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, pred, latent_count)) {
            goto cleanup;
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_pred_x0", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, x0, latent_count)) {
            goto cleanup;
        }
        memcpy(latent, next, latent_count * sizeof(float));
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_latent", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, latent, latent_count)) {
            goto cleanup;
        }
        token_channels_to_ncdhw(latent, latent_ncdhw, flow.in_channels, latent_size);
        shape_5d(shape, sizeof(shape), 1, flow.in_channels, latent_size, latent_size, latent_size);
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_latent_ncdhw", step + 1);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, latent_ncdhw, latent_count)) {
            goto cleanup;
        }

        const int need_decode =
            dump.enabled ||
            frame_callback != NULL ||
            (out_dir != NULL && out_dir[0] != '\0') ||
            (result != NULL && step + 1 == steps);
        if (!need_decode) {
            char progress_detail[128];
            snprintf(
                progress_detail,
                sizeof(progress_detail),
                "t=%.6g->%.6g flow-only",
                t,
                t_prev);
            trellis_tool_progress_steps(
                "stage1 image",
                step + 1,
                steps,
                ggml_time_us() - step_start_us,
                progress_detail);
            continue;
        }

        float * logits = NULL;
        int output_size = 0;
        TRELLIS_TOOL_DEBUG("stage1 image: step %d/%d decoding sparse structure", step + 1, steps);
        status = trellis_ss_decoder_forward_f32_host(&decoder, latent_ncdhw, cuda, 1, latent_size, &logits, &output_size);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "stage1 image: decoder step %d failed: %s\n", step + 1, trellis_status_string(status));
            free(logits);
            goto cleanup;
        }
        const size_t logits_count = (size_t) output_size * (size_t) output_size * (size_t) output_size;
        shape_5d(shape, sizeof(shape), 1, 1, output_size, output_size, output_size);
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_decoder_logits_%d3", step + 1, output_size);
        if (!check_or_dump_tensor_f32(&dump, tensor_name, shape, logits, logits_count)) {
            free(logits);
            goto cleanup;
        }
        float * frame_logits = NULL;
        int frame_resolution = output_size;
        float frame_threshold = threshold;
        if (sparse_resolution != output_size) {
            if (!max_pool_logits_to_resolution(logits, output_size, threshold, sparse_resolution, &frame_logits)) {
                fprintf(stderr, "stage1 image: cannot pool decoder output %d^3 to sparse resolution %d^3\n",
                    output_size,
                    sparse_resolution);
                free(logits);
                goto cleanup;
            }
            frame_resolution = sparse_resolution;
            frame_threshold = 0.5f;
        }
        int32_t * frame_coords = NULL;
        int64_t frame_coord_count = 0;
        if (!collect_voxel_coords_xyz(
                frame_logits != NULL ? frame_logits : logits,
                frame_resolution,
                frame_threshold,
                &frame_coords,
                &frame_coord_count)) {
            fprintf(stderr, "stage1 image: voxel coord extraction failed at step %d\n", step + 1);
            free(frame_logits);
            free(logits);
            goto cleanup;
        }
        int voxel_count = frame_coord_count > INT32_MAX ? INT32_MAX : (int) frame_coord_count;
        if (frame_callback != NULL) {
            trellis_tool_stage1_frame frame;
            memset(&frame, 0, sizeof(frame));
            frame.step = step + 1;
            frame.steps = steps;
            frame.t = t;
            frame.t_prev = t_prev;
            frame.resolution = frame_resolution;
            frame.n_coords = frame_coord_count;
            frame.coords_xyz = frame_coords;
            if (!frame_callback(&frame, frame_user_data)) {
                free(frame_coords);
                free(frame_logits);
                free(logits);
                goto cleanup;
            }
        }
        if (step + 1 == steps &&
            !set_stage1_result(result, frame_coords, frame_coord_count, frame_resolution, context, cond_tokens)) {
            fprintf(stderr, "stage1 image: failed to retain final coords/condition for stage2\n");
            free(frame_coords);
            free(frame_logits);
            free(logits);
            goto cleanup;
        }
        if (out_dir != NULL && out_dir[0] != '\0') {
            int written_voxels = 0;
            if (!write_voxel_frame(
                    out_dir,
                    frame_logits != NULL ? frame_logits : logits,
                    frame_resolution,
                    frame_threshold,
                    step + 1,
                    "x_t",
                    step != 0,
                    &written_voxels)) {
                fprintf(stderr, "stage1 image: voxel frame write failed at step %d\n", step + 1);
                free(frame_coords);
                free(frame_logits);
                free(logits);
                goto cleanup;
            }
            voxel_count = written_voxels;
        }
        float logit_min = logits_count == 0 ? 0.0f : logits[0];
        float logit_max = logits_count == 0 ? 0.0f : logits[0];
        double logit_sum = 0.0;
        for (size_t i = 0; i < logits_count; ++i) {
            if (logits[i] < logit_min) logit_min = logits[i];
            if (logits[i] > logit_max) logit_max = logits[i];
            logit_sum += logits[i];
        }
        const double logit_mean = logits_count == 0 ? 0.0 : logit_sum / (double) logits_count;
        free(frame_coords);
        free(frame_logits);
        free(logits);
        char progress_detail[256];
        snprintf(
            progress_detail,
            sizeof(progress_detail),
            "t=%.6g->%.6g voxels=%d %d^3 logits[min=%.6g mean=%.6g max=%.6g]",
            t,
            t_prev,
            voxel_count,
            frame_resolution,
            logit_min,
            logit_mean,
            logit_max);
        trellis_tool_progress_steps(
            "stage1 image",
            step + 1,
            steps,
            ggml_time_us() - step_start_us,
            progress_detail);
    }

    if (out_dir != NULL && out_dir[0] != '\0') {
        TRELLIS_TOOL_INFO("stage1 image: wrote %d decoded voxel frames to %s (threshold=%.6g)", steps, out_dir, threshold);
    } else {
        TRELLIS_TOOL_INFO("stage1 image: decoded %d voxel frames (threshold=%.6g)", steps, threshold);
    }
    rc = 0;

cleanup:
    if (rc != 0 && result != NULL) {
        trellis_tool_stage1_result_free(result);
    }
    free(context);
    free(neg_context);
    free(latent);
    free(pred_pos);
    free(pred_neg);
    free(pred);
    free(next);
    free(x0);
    free(latent_ncdhw);
    free(input_latent_ncdhw);
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    if (owns_decoder_store) {
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_flow_store) {
        trellis_tensor_store_free(&flow_store);
    }
    dump_context_close(&dump);
    return rc;
}

static int run_stage1_smoke_voxels(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * out_dir,
    int latent_size,
    int cond_tokens,
    int steps,
    float threshold) {
    if (cuda == NULL || model_dir == NULL || out_dir == NULL ||
        latent_size <= 0 || cond_tokens <= 0 || steps <= 0) {
        return 1;
    }

    trellis_tensor_store flow_store;
    trellis_tensor_store decoder_store;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_dit_flow_weights flow;
    trellis_ss_decoder_weights decoder;
    int rc = 1;

    if (!load_flow_for_stage1(cuda, model_dir, &flow_store, &flow) ||
        !load_decoder_for_stage1(cuda, model_dir, &decoder_store, &decoder)) {
        goto cleanup;
    }

    const int tokens = latent_size * latent_size * latent_size;
    const size_t latent_count = (size_t) flow.in_channels * (size_t) tokens;
    const size_t context_count = (size_t) flow.cond_channels * (size_t) cond_tokens;
    const size_t phase_count = (size_t) (flow.head_dim / 2) * (size_t) tokens;
    float * latent = (float *) malloc(latent_count * sizeof(float));
    float * pred = (float *) malloc(latent_count * sizeof(float));
    float * next = (float *) malloc(latent_count * sizeof(float));
    float * x0 = (float *) malloc(latent_count * sizeof(float));
    float * context = (float *) calloc(context_count, sizeof(float));
    float * pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    float * cos_phase = (float *) malloc(phase_count * sizeof(float));
    float * sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (latent == NULL || pred == NULL || next == NULL || x0 == NULL || context == NULL ||
        pairs == NULL || cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "stage1 smoke: host allocation failed\n");
        free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
        goto cleanup;
    }

    for (size_t i = 0; i < latent_count; ++i) {
        latent[i] = 0.35f * sinf(0.013f * (float) (i + 1)) + 0.15f * cosf(0.031f * (float) (i + 7));
    }
    trellis_status status = trellis_flow_timestep_pairs_f32(steps, 5.0f, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_phases_f32(latent_size, flow.head_dim, 1.0f, 10000.0f, cos_phase, sin_phase, phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "stage1 smoke: schedule/rope failed: %s\n", trellis_status_string(status));
        free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
        goto cleanup;
    }

    for (int step = 0; step < steps; ++step) {
        const int64_t step_start_us = ggml_time_us();
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        status = run_flow_once_host(cuda, &flow, latent, context, cos_phase, sin_phase, t, latent_size, cond_tokens, pred);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "stage1 smoke: flow step %d failed: %s\n", step + 1, trellis_status_string(status));
            free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
            goto cleanup;
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);
        memcpy(latent, next, latent_count * sizeof(float));
        char detail[96];
        snprintf(detail, sizeof(detail), "t=%.6g->%.6g", t, t_prev);
        trellis_tool_progress_steps("stage1 smoke", step + 1, steps, ggml_time_us() - step_start_us, detail);
    }

    float * logits = NULL;
    int output_size = 0;
    status = trellis_ss_decoder_forward_f32_host(&decoder, latent, cuda, 1, latent_size, &logits, &output_size);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "stage1 smoke: decoder failed: %s\n", trellis_status_string(status));
        free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
        goto cleanup;
    }

    int voxel_count = 0;
    if (!write_voxel_snapshot(out_dir, logits, output_size, threshold, &voxel_count)) {
        fprintf(stderr, "stage1 smoke: voxel snapshot write failed\n");
        free(logits);
        free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
        goto cleanup;
    }
    TRELLIS_TOOL_INFO("stage1 smoke: wrote %d voxels at %d^3 to %s (threshold=%.6g)",
        voxel_count, output_size, out_dir, threshold);
    free(logits);
    free(latent); free(pred); free(next); free(x0); free(context); free(pairs); free(cos_phase); free(sin_phase);
    rc = 0;

cleanup:
    trellis_tensor_store_free(&decoder_store);
    trellis_tensor_store_free(&flow_store);
    return rc;
}

int trellis_tool_run_stage1_image(
    const trellis_cuda_context * cuda,
    const trellis_tool_stage1_image_options * options,
    trellis_tool_stage1_frame_callback frame_callback,
    void * frame_user_data,
    trellis_tool_stage1_result * result) {
    if (options == NULL) {
        return 1;
    }
    return run_stage1_image_voxels(
        cuda,
        options->model_dir,
        options->dino_dir,
        options->image_path,
        options->input_cond_path,
        options->input_neg_cond_path,
        options->input_latent_path,
        options->out_dir,
        options->dump_dir,
        options->latent_size,
        options->steps,
        options->cond_resolution,
        options->sparse_resolution,
        options->seed,
        options->flow_blocks_override,
        options->flow_block_parts_override,
        options->flow_no_rope,
        options->voxel_threshold,
        options->weights,
        frame_callback,
        frame_user_data,
        result);
}

static int load_store(
    const trellis_cuda_context * cuda,
    const char * label,
    const char * path,
    trellis_status (*bind_fn)(trellis_tensor_store *, trellis_dit_flow_weights *, char *, size_t),
    int dry_forward,
    int dry_tokens,
    int dry_cond_tokens,
    int dry_batch) {
    trellis_tensor_store store;
    if (!trellis_tool_load_tensor_store_f32(cuda, label, path, true, 64, &store, NULL)) {
        return 1;
    }

    const struct ggml_tensor * input_w = trellis_tensor_store_get_const(&store, "input_layer.weight");
    const struct ggml_tensor * input_b = trellis_tensor_store_get_const(&store, "input_layer.bias");
    const struct ggml_tensor * out_w = trellis_tensor_store_get_const(&store, "out_layer.weight");
    if (input_w != NULL && input_b != NULL) {
        TRELLIS_TOOL_INFO("%s: input_layer.weight ne=[%lld,%lld] input_layer.bias ne=[%lld]",
            label,
            (long long) input_w->ne[0],
            (long long) input_w->ne[1],
            (long long) input_b->ne[0]);
    }
    if (out_w != NULL) {
        TRELLIS_TOOL_INFO("%s: out_layer.weight ne=[%lld,%lld]",
            label,
            (long long) out_w->ne[0],
            (long long) out_w->ne[1]);
    }

    if (bind_fn != NULL) {
        trellis_dit_flow_weights weights;
        char issue[256];
        trellis_status status = bind_fn(&store, &weights, issue, sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_TOOL_ERROR("%s: bind failed: %s%s%s",
                label,
                trellis_status_string(status),
                issue[0] == '\0' ? "" : " ",
                issue);
            trellis_tensor_store_free(&store);
            return 1;
        }
        TRELLIS_TOOL_INFO("%s: bindings ok blocks=%d channels=%d heads=%d head_dim=%d in=%d out=%d cond=%d",
            label,
            weights.n_blocks,
            weights.model_channels,
            weights.heads,
            weights.head_dim,
            weights.in_channels,
            weights.out_channels,
            weights.cond_channels);
        if (dry_forward && dry_forward_flow(cuda, label, &weights, dry_tokens, dry_cond_tokens, dry_batch) != 0) {
            trellis_tensor_store_free(&store);
            return 1;
        }
    }

    trellis_tensor_store_free(&store);
    return 0;
}

#ifndef TRELLIS_TOOL_LIBRARY
int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * dino_dir = "dinov3-vitl16-pretrain-lvd1689m";
    int device = 0;
    int load_dino = 0;
    int load_stage1_flow = 0;
    int load_stage1_decoder = 0;
    int load_shape_512 = 0;
    int load_shape_1024 = 0;
    int stage1_smoke_voxels = 0;
    int stage1_image_voxels = 0;
    const char * image_path = NULL;
    const char * input_cond_path = NULL;
    const char * input_neg_cond_path = NULL;
    const char * input_latent_path = NULL;
    const char * save_cond_path = NULL;
    const char * save_final_coords_path = NULL;
    const char * out_dir = "benchmark_outputs/c_stage1_voxels";
    const char * dump_dir = NULL;
    int cond_resolution = 512;
    int sparse_resolution = 32;
    int steps = 1;
    int steps_set = 0;
    int latent_size_set = 0;
    uint32_t seed = 1u;
    int flow_blocks_override = -1;
    int flow_block_parts_override = -1;
    int flow_no_rope = 0;
    float voxel_threshold = 0.0f;
    int dry_forward = 0;
    int dry_decode = 0;
    int dry_dino_patch = 0;
    int dry_tokens = 4;
    int dry_cond_tokens = 2;
    int dry_latent_size = 1;
    int dry_batch = 1;
    int verbose = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            dino_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            device = atoi(v);
        } else if (strcmp(argv[i], "--load-dino") == 0) {
            load_dino = 1;
        } else if (strcmp(argv[i], "--load-stage1-flow") == 0) {
            load_stage1_flow = 1;
        } else if (strcmp(argv[i], "--load-stage1-decoder") == 0) {
            load_stage1_decoder = 1;
        } else if (strcmp(argv[i], "--load-shape-flow-512") == 0) {
            load_shape_512 = 1;
        } else if (strcmp(argv[i], "--load-shape-flow-1024") == 0) {
            load_shape_1024 = 1;
        } else if (strcmp(argv[i], "--stage1-smoke-voxels") == 0) {
            stage1_smoke_voxels = 1;
        } else if (strcmp(argv[i], "--stage1-image-voxels") == 0) {
            stage1_image_voxels = 1;
        } else if (strcmp(argv[i], "--image") == 0) {
            image_path = arg_value(argc, argv, &i);
            if (image_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--input-cond-f32") == 0) {
            input_cond_path = arg_value(argc, argv, &i);
            if (input_cond_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--input-neg-cond-f32") == 0) {
            input_neg_cond_path = arg_value(argc, argv, &i);
            if (input_neg_cond_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--input-latent-f32") == 0) {
            input_latent_path = arg_value(argc, argv, &i);
            if (input_latent_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--save-cond-f32") == 0) {
            save_cond_path = arg_value(argc, argv, &i);
            if (save_cond_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--save-final-coords-i32") == 0) {
            save_final_coords_path = arg_value(argc, argv, &i);
            if (save_final_coords_path == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            flow_blocks_override = atoi(v);
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            flow_block_parts_override = atoi(v);
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            flow_no_rope = 1;
        } else if (strcmp(argv[i], "--cond-resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            cond_resolution = atoi(v);
        } else if (strcmp(argv[i], "--sparse-resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            sparse_resolution = atoi(v);
        } else if (strcmp(argv[i], "--dump-intermediates") == 0) {
            dump_dir = arg_value(argc, argv, &i);
            if (dump_dir == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--out") == 0) {
            out_dir = arg_value(argc, argv, &i);
            if (out_dir == NULL) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            steps = atoi(v);
            steps_set = 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            seed = (uint32_t) strtoul(v, NULL, 10);
        } else if (strcmp(argv[i], "--voxel-threshold") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            voxel_threshold = strtof(v, NULL);
        } else if (strcmp(argv[i], "--dry-forward") == 0) {
            dry_forward = 1;
        } else if (strcmp(argv[i], "--dry-decode") == 0) {
            dry_decode = 1;
        } else if (strcmp(argv[i], "--dry-dino-patch") == 0) {
            dry_dino_patch = 1;
        } else if (strcmp(argv[i], "--tokens") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            dry_tokens = atoi(v);
        } else if (strcmp(argv[i], "--cond-tokens") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            dry_cond_tokens = atoi(v);
        } else if (strcmp(argv[i], "--latent-size") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            dry_latent_size = atoi(v);
            latent_size_set = 1;
        } else if (strcmp(argv[i], "--batch") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            dry_batch = atoi(v);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    trellis_tool_set_verbose(verbose);

    const int need_model_dir = load_stage1_flow || load_stage1_decoder || load_shape_512 || load_shape_1024 ||
        stage1_smoke_voxels || stage1_image_voxels;
    if ((need_model_dir && model_dir == NULL) ||
        (stage1_image_voxels && image_path == NULL && input_cond_path == NULL) ||
        (!load_dino && !load_stage1_flow && !load_stage1_decoder && !load_shape_512 && !load_shape_1024 &&
         !stage1_smoke_voxels && !stage1_image_voxels)) {
        usage(argv[0]);
        return 2;
    }

    trellis_cuda_context cuda;
    trellis_status status = trellis_cuda_init(&cuda, device);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "cuda: %s\n", trellis_status_string(status));
        return 1;
    }

    int rc = 0;
    char path[4096];
    if (load_dino) {
        if (load_dino_store(&cuda, dino_dir, dry_dino_patch) != 0) {
            rc = 1;
        }
    }
    if (load_stage1_flow) {
        if (!choose_path(model_dir, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", path, sizeof(path)) ||
            load_store(&cuda, "stage1 sparse-structure flow", path, trellis_ss_flow_bind_weights,
                dry_forward, dry_tokens, dry_cond_tokens, dry_batch) != 0) {
            rc = 1;
        }
    }
    if (load_stage1_decoder) {
        if (load_decoder_store(&cuda, model_dir, dry_decode, dry_latent_size, dry_batch) != 0) {
            rc = 1;
        }
    }
    if (load_shape_512) {
        if (!choose_path(model_dir, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", path, sizeof(path)) ||
            load_store(&cuda, "shape SLat flow 512", path, trellis_shape_slat_flow_bind_weights,
                dry_forward, dry_tokens, dry_cond_tokens, dry_batch) != 0) {
            rc = 1;
        }
    }
    if (load_shape_1024) {
        if (!choose_path(model_dir, "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", path, sizeof(path)) ||
            load_store(&cuda, "shape SLat flow 1024", path, trellis_shape_slat_flow_bind_weights,
                dry_forward, dry_tokens, dry_cond_tokens, dry_batch) != 0) {
            rc = 1;
        }
    }
    if (stage1_smoke_voxels) {
        if (run_stage1_smoke_voxels(&cuda, model_dir, out_dir, dry_latent_size, dry_cond_tokens, steps, voxel_threshold) != 0) {
            rc = 1;
        }
    }
    if (stage1_image_voxels) {
        const int image_latent_size = latent_size_set ? dry_latent_size : 16;
        const int image_steps = steps_set ? steps : 12;
        trellis_tool_stage1_result result;
        memset(&result, 0, sizeof(result));
        trellis_tool_stage1_result * result_ptr =
            (save_cond_path != NULL || save_final_coords_path != NULL) ? &result : NULL;
        if (run_stage1_image_voxels(
                &cuda,
                model_dir,
                dino_dir,
                image_path,
                input_cond_path,
                input_neg_cond_path,
                input_latent_path,
                out_dir,
                dump_dir,
                image_latent_size,
                image_steps,
                cond_resolution,
                sparse_resolution,
                seed,
                flow_blocks_override,
                flow_block_parts_override,
                flow_no_rope,
                voxel_threshold,
                NULL,
                NULL,
                NULL,
                result_ptr) != 0) {
            rc = 1;
        } else if (result_ptr != NULL) {
            if (save_cond_path != NULL &&
                !write_binary_file_exact(
                    save_cond_path,
                    result.cond,
                    sizeof(float),
                    (size_t) result.cond_tokens * 1024u)) {
                rc = 1;
            } else if (save_cond_path != NULL) {
                TRELLIS_TOOL_INFO("stage1 image: wrote cond %s tokens=%d hidden=1024", save_cond_path, result.cond_tokens);
            }
            if (save_final_coords_path != NULL &&
                !write_binary_file_exact(
                    save_final_coords_path,
                    result.coords_bxyz,
                    sizeof(int32_t),
                    (size_t) result.n_coords * 4u)) {
                rc = 1;
            } else if (save_final_coords_path != NULL) {
                TRELLIS_TOOL_INFO("stage1 image: wrote final coords %s n=%lld", save_final_coords_path, (long long) result.n_coords);
            }
        }
        trellis_tool_stage1_result_free(&result);
    }

    trellis_cuda_free(&cuda);
    return rc;
}
#endif
