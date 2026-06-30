#include "trellis.h"

#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --model TRELLIS.2-4B --validate [--resolution 512|1024]\n"
        "  %s --model TRELLIS.2-4B --validate-shape-decoder\n"
        "  %s --timesteps 12 --rescale-t 3.0\n"
        "  %s --model TRELLIS.2-4B --shape-flow-512 --coords-i32 coords.i32 --input-cond-f32 cond.f32 --input-neg-cond-f32 neg.f32 --input-noise-f32 noise.f32 --out DIR\n"
        "  %s --model TRELLIS.2-4B --shape-decode --coords-i32 shape_coords.i32 --input-slat-f32 shape_final_latent.f32 --out DIR\n"
        "\n"
        "Options:\n"
        "  --model DIR             TRELLIS.2 model directory containing ckpts/\n"
        "  --flow PATH             Override shape SLat flow safetensors path\n"
        "  --decoder PATH          Override shape decoder safetensors path\n"
        "  --resolution N          Shape flow checkpoint resolution, default 512\n"
        "  --validate              Validate stage2 shape-SLat flow checkpoint contract\n"
        "  --validate-shape-decoder Validate FlexiDualGrid shape decoder checkpoint contract\n"
        "  --shape-flow-512        Run C+ggml+CUDA shape SLat flow sampler with 512 checkpoint\n"
        "  --shape-flow-1024       Run C+ggml+CUDA shape SLat flow sampler with 1024 checkpoint\n"
        "  --shape-decode          Run C+CUDA FlexiDualGrid shape decoder to raw sparse 7-channel output\n"
        "  --coords-i32 FILE       Sparse coords, [N,4] batch,d,h,w or [N,3] d,h,w\n"
        "  --coords-xyz            Force --coords-i32 to be [N,3] without batch column\n"
        "  --input-cond-f32 FILE   Positive DINO cond tensor [1,T,1024]\n"
        "  --input-neg-cond-f32 FILE Negative DINO cond tensor [1,T,1024], default zeros\n"
        "  --input-noise-f32 FILE  Initial sparse latent feats [N,32]\n"
        "  --input-slat-f32 FILE   Denormalized shape SLat feats [N,32] for --shape-decode\n"
        "  --out DIR               Output tensor dump directory, default benchmark_outputs/c_shape_flow\n"
        "  --steps N               Euler steps, default 12\n"
        "  --rescale-t X           Timestep rescale factor, default 3.0\n"
        "  --guidance-strength X   CFG strength, default 7.5\n"
        "  --guidance-rescale X    CFG rescale, default 0.5\n"
        "  --guidance-min X        CFG interval min, default 0.6\n"
        "  --guidance-max X        CFG interval max, default 1.0\n"
        "  --flow-blocks N         Debug: run only first N transformer blocks\n"
        "  --flow-block-parts N    Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope          Debug: disable sparse RoPE\n"
        "  --emulate-bf16-blocks   Debug: round block activations like reference bf16 shape flow\n"
        "  --use-ggml-flash-attn   Debug: use ggml flash attention instead of explicit SDPA\n"
        "  --decode-max-levels N   Debug: run only first N shape decoder levels, default full\n"
        "  --decode-max-input-tokens N Debug: truncate input sparse tensor before decode\n"
        "  --decode-dump-dir DIR   Debug: dump shape decoder intermediate tensors\n"
        "  --timesteps N           Print Euler timestep pairs\n"
        "  --cuda-check            Initialize ggml CUDA backend and report availability\n",
        argv0, argv0, argv0, argv0, argv0);
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

static int mkdir_p(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(tmp)) {
        return 0;
    }
    for (char * p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return 0;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return 0;
    }
    return 1;
}

static int make_join_path(const char * dir, const char * name, char * dst, size_t dst_size) {
    if (dir == NULL || name == NULL || dst == NULL || dst_size == 0) {
        return 0;
    }
    int n = snprintf(dst, dst_size, "%s/%s", dir, name);
    return n >= 0 && (size_t) n < dst_size;
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

static char * read_text_file_null_terminated(const char * path, size_t * size_out) {
    if (path == NULL) {
        return NULL;
    }
    size_t size = 0;
    if (!file_size_bytes(path, &size)) {
        return NULL;
    }
    char * data = (char *) malloc(size + 1u);
    if (data == NULL) {
        return NULL;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        free(data);
        return NULL;
    }
    const size_t got = fread(data, 1, size, f);
    fclose(f);
    if (got != size) {
        free(data);
        return NULL;
    }
    data[size] = '\0';
    if (size_out != NULL) {
        *size_out = size;
    }
    return data;
}

static const char * skip_json_ws(const char * p) {
    while (p != NULL && *p != '\0' && isspace((unsigned char) *p)) {
        ++p;
    }
    return p;
}

static int parse_json_float_array_32(const char * start, const char * key, float out[32]) {
    if (start == NULL || key == NULL || out == NULL) {
        return 0;
    }
    const char * p = strstr(start, key);
    if (p == NULL) {
        return 0;
    }
    p = strchr(p, '[');
    if (p == NULL) {
        return 0;
    }
    ++p;
    for (int i = 0; i < 32; ++i) {
        p = skip_json_ws(p);
        if (p == NULL || *p == '\0') {
            return 0;
        }
        char * end = NULL;
        out[i] = strtof(p, &end);
        if (end == p) {
            return 0;
        }
        p = skip_json_ws(end);
        if (i < 31) {
            if (p == NULL || *p != ',') {
                return 0;
            }
            ++p;
        }
    }
    p = skip_json_ws(p);
    return p != NULL && *p == ']';
}

static int load_shape_slat_normalization(const char * model_dir, float mean[32], float std[32]) {
    if (model_dir == NULL || mean == NULL || std == NULL) {
        return 0;
    }
    char path[4096];
    trellis_status pstatus = trellis_make_model_path(model_dir, "pipeline.json", path, sizeof(path));
    if (pstatus != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow normalization path: %s\n", trellis_status_string(pstatus));
        return 0;
    }
    char * json = read_text_file_null_terminated(path, NULL);
    if (json == NULL) {
        fprintf(stderr, "shape flow: failed to read %s\n", path);
        return 0;
    }
    const char * section = strstr(json, "\"shape_slat_normalization\"");
    int ok = 0;
    if (section != NULL &&
        parse_json_float_array_32(section, "\"mean\"", mean) &&
        parse_json_float_array_32(section, "\"std\"", std)) {
        ok = 1;
    }
    if (!ok) {
        fprintf(stderr, "shape flow: failed to parse shape_slat_normalization from %s\n", path);
    }
    free(json);
    return ok;
}

static void denormalize_slat_f32(
    const float * norm,
    int64_t tokens,
    int channels,
    const float * mean,
    const float * std,
    float * out) {
    if (norm == NULL || mean == NULL || std == NULL || out == NULL || tokens <= 0 || channels <= 0) {
        return;
    }
    for (int64_t i = 0; i < tokens; ++i) {
        for (int c = 0; c < channels; ++c) {
            out[(size_t) i * (size_t) channels + (size_t) c] =
                norm[(size_t) i * (size_t) channels + (size_t) c] * std[c] + mean[c];
        }
    }
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
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        free(data);
        return 0;
    }
    const size_t got = fread(data, sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        free(data);
        return 0;
    }
    *data_out = data;
    *count_out = count;
    return 1;
}

static int read_f32_file_exact(const char * path, float * dst, size_t count) {
    float * tmp = NULL;
    size_t got = 0;
    if (!read_f32_file_alloc(path, &tmp, &got)) {
        return 0;
    }
    if (got != count) {
        fprintf(stderr, "f32 input: %s has %zu values, expected %zu\n", path, got, count);
        free(tmp);
        return 0;
    }
    memcpy(dst, tmp, count * sizeof(float));
    free(tmp);
    return 1;
}

static int read_coords_i32_alloc(const char * path, int force_xyz, int32_t ** coords_out, int64_t * n_out) {
    if (path == NULL || coords_out == NULL || n_out == NULL) {
        return 0;
    }
    *coords_out = NULL;
    *n_out = 0;
    size_t bytes = 0;
    if (!file_size_bytes(path, &bytes) || (bytes % sizeof(int32_t)) != 0) {
        fprintf(stderr, "coords input: %s is not a whole i32 tensor\n", path);
        return 0;
    }
    const size_t count = bytes / sizeof(int32_t);
    int32_t * raw = (int32_t *) malloc(count * sizeof(int32_t));
    if (raw == NULL && count != 0) {
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        free(raw);
        return 0;
    }
    const size_t got = fread(raw, sizeof(int32_t), count, f);
    fclose(f);
    if (got != count) {
        free(raw);
        return 0;
    }

    if (!force_xyz && (count % 4u) == 0) {
        *coords_out = raw;
        *n_out = (int64_t) (count / 4u);
        return 1;
    }
    if ((count % 3u) != 0) {
        fprintf(stderr, "coords input: %s must be [N,4] or [N,3]\n", path);
        free(raw);
        return 0;
    }
    const size_t n = count / 3u;
    int32_t * coords = (int32_t *) malloc(n * 4u * sizeof(int32_t));
    if (coords == NULL && n != 0) {
        free(raw);
        return 0;
    }
    for (size_t i = 0; i < n; ++i) {
        coords[4u * i + 0u] = 0;
        coords[4u * i + 1u] = raw[3u * i + 0u];
        coords[4u * i + 2u] = raw[3u * i + 1u];
        coords[4u * i + 3u] = raw[3u * i + 2u];
    }
    free(raw);
    *coords_out = coords;
    *n_out = (int64_t) n;
    return 1;
}

static int write_f32_file(const char * out_dir, const char * name, const float * data, size_t count) {
    if (out_dir == NULL || name == NULL || data == NULL) {
        return 0;
    }
    if (!mkdir_p(out_dir)) {
        return 0;
    }
    char path[4096];
    if (!make_join_path(out_dir, name, path, sizeof(path))) {
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return 0;
    }
    const size_t wrote = fwrite(data, sizeof(float), count, f);
    fclose(f);
    return wrote == count;
}

static int write_i32_file(const char * out_dir, const char * name, const int32_t * data, size_t count) {
    if (out_dir == NULL || name == NULL || data == NULL) {
        return 0;
    }
    if (!mkdir_p(out_dir)) {
        return 0;
    }
    char path[4096];
    if (!make_join_path(out_dir, name, path, sizeof(path))) {
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return 0;
    }
    const size_t wrote = fwrite(data, sizeof(int32_t), count, f);
    fclose(f);
    return wrote == count;
}

static int write_manifest_line(FILE * f, const char * name, const char * shape, const char * file) {
    if (f == NULL || name == NULL || shape == NULL || file == NULL) {
        return 0;
    }
    return fprintf(f, "%s\t%s\t%s\n", name, shape, file) > 0;
}

static int load_shape_flow(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_dit_flow_weights * flow) {
    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow: open %s failed: %s\n", path, trellis_status_string(status));
        return 0;
    }
    const size_t graph_tensors = st.n_tensors + 64;
    trellis_safetensors_close(&st);
    status = trellis_tensor_store_init(store, graph_tensors, 0);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow: store init failed: %s\n", trellis_status_string(status));
        return 0;
    }
    status = trellis_tensor_store_load_safetensors_f32(store, cuda, path, true, NULL);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow: load failed: %s\n", trellis_status_string(status));
        return 0;
    }
    char issue[256];
    status = trellis_shape_slat_flow_bind_weights(store, flow, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow: bind failed: %s%s%s\n",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        return 0;
    }
    return 1;
}

static int load_shape_decoder(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_shape_decoder_weights * decoder) {
    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape decoder: open %s failed: %s\n", path, trellis_status_string(status));
        return 0;
    }
    const size_t graph_tensors = st.n_tensors + 64;
    trellis_safetensors_close(&st);
    status = trellis_tensor_store_init(store, graph_tensors, 0);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape decoder: store init failed: %s\n", trellis_status_string(status));
        return 0;
    }
    status = trellis_tensor_store_load_safetensors_f32(store, cuda, path, true, NULL);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape decoder: load failed: %s\n", trellis_status_string(status));
        return 0;
    }
    char issue[256];
    status = trellis_shape_decoder_bind_weights(store, decoder, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape decoder: bind failed: %s%s%s\n",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        return 0;
    }
    return 1;
}

static trellis_status run_flow_once_tokens_host(
    const trellis_cuda_context * cuda,
    const trellis_dit_flow_weights * weights,
    const float * latent,
    const float * context_data,
    const float * cos_data,
    const float * sin_data,
    float timestep,
    int64_t tokens,
    int cond_tokens,
    float * out_pred) {
    if (cuda == NULL || weights == NULL || latent == NULL || context_data == NULL ||
        cos_data == NULL || sin_data == NULL || out_pred == NULL || tokens <= 0 || cond_tokens <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int batch = 1;
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

static void cfg_rescale_combine_population_f32(
    const float * x_t,
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float sigma_min,
    float t,
    float guidance_strength,
    float guidance_rescale,
    float * pred) {
    if (x_t == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL || n == 0) {
        return;
    }
    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    const float one_minus_sigma_min = 1.0f - sigma_min;
    for (size_t i = 0; i < n; ++i) {
        pred[i] = guidance_strength * pred_pos[i] + (1.0f - guidance_strength) * pred_neg[i];
    }
    if (guidance_rescale <= 0.0f) {
        return;
    }

    double mean_pos = 0.0;
    double mean_cfg = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float x0_pos = one_minus_sigma_min * x_t[i] - sigma_t * pred_pos[i];
        const float x0_cfg = one_minus_sigma_min * x_t[i] - sigma_t * pred[i];
        mean_pos += x0_pos;
        mean_cfg += x0_cfg;
    }
    mean_pos /= (double) n;
    mean_cfg /= (double) n;

    double var_pos = 0.0;
    double var_cfg = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x0_pos = (double) (one_minus_sigma_min * x_t[i] - sigma_t * pred_pos[i]) - mean_pos;
        const double x0_cfg = (double) (one_minus_sigma_min * x_t[i] - sigma_t * pred[i]) - mean_cfg;
        var_pos += x0_pos * x0_pos;
        var_cfg += x0_cfg * x0_cfg;
    }
    const double std_pos = sqrt(var_pos / (double) n);
    const double std_cfg = sqrt(var_cfg / (double) n);
    if (std_cfg <= 0.0) {
        return;
    }
    const float ratio = (float) (std_pos / std_cfg);
    for (size_t i = 0; i < n; ++i) {
        const float x0_cfg = one_minus_sigma_min * x_t[i] - sigma_t * pred[i];
        const float x0_rescaled = x0_cfg * ratio;
        const float x0 = guidance_rescale * x0_rescaled + (1.0f - guidance_rescale) * x0_cfg;
        pred[i] = (one_minus_sigma_min * x_t[i] - x0) / sigma_t;
    }
}

static int run_shape_flow(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * flow_override_path,
    int checkpoint_resolution,
    const char * coords_path,
    int coords_force_xyz,
    const char * cond_path,
    const char * neg_cond_path,
    const char * noise_path,
    const char * out_dir,
    int steps,
    float rescale_t,
    float guidance_strength,
    float guidance_rescale,
    float guidance_min,
    float guidance_max,
    int flow_blocks_override,
    int flow_block_parts_override,
    int flow_no_rope,
    int emulate_bf16_blocks,
    int use_ggml_flash_attn) {
    if (cuda == NULL || model_dir == NULL || coords_path == NULL || cond_path == NULL ||
        noise_path == NULL || out_dir == NULL || steps <= 0) {
        return 1;
    }

    char flow_path[4096];
    if (flow_override_path != NULL) {
        snprintf(flow_path, sizeof(flow_path), "%s", flow_override_path);
    } else {
        const char * rel = checkpoint_resolution == 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
        trellis_status pstatus = trellis_make_model_path(model_dir, rel, flow_path, sizeof(flow_path));
        if (pstatus != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape flow path: %s\n", trellis_status_string(pstatus));
            return 1;
        }
    }

    trellis_tensor_store flow_store;
    memset(&flow_store, 0, sizeof(flow_store));
    trellis_dit_flow_weights flow;
    int rc = 1;
    int32_t * coords = NULL;
    float * cond = NULL;
    float * neg_cond = NULL;
    float * latent = NULL;
    float * pred_pos = NULL;
    float * pred_neg = NULL;
    float * pred = NULL;
    float * next = NULL;
    float * x0 = NULL;
    float * denorm = NULL;
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    float slat_mean[32];
    float slat_std[32];
    FILE * manifest = NULL;

    int64_t n_coords = 0;
    size_t cond_values = 0;
    if (!read_coords_i32_alloc(coords_path, coords_force_xyz, &coords, &n_coords)) {
        goto cleanup;
    }
    if (!read_f32_file_alloc(cond_path, &cond, &cond_values)) {
        goto cleanup;
    }
    if ((cond_values % 1024u) != 0) {
        fprintf(stderr, "shape flow: cond count %zu is not divisible by 1024\n", cond_values);
        goto cleanup;
    }
    const int cond_tokens = (int) (cond_values / 1024u);

    if (!load_shape_flow(cuda, flow_path, &flow_store, &flow)) {
        goto cleanup;
    }
    if (flow.in_channels != 32 || flow.out_channels != 32) {
        fprintf(stderr, "shape flow: denormalization expects 32 channels, got in=%d out=%d\n",
            flow.in_channels,
            flow.out_channels);
        goto cleanup;
    }
    if (!load_shape_slat_normalization(model_dir, slat_mean, slat_std)) {
        goto cleanup;
    }
    if (flow_blocks_override >= 0) {
        if (flow_blocks_override > flow.n_blocks) {
            fprintf(stderr, "shape flow: --flow-blocks %d exceeds checkpoint blocks %d\n",
                flow_blocks_override,
                flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = flow_blocks_override;
        printf("shape flow: debug flow blocks override=%d\n", flow.n_blocks);
    }
    if (flow_block_parts_override >= 0) {
        if (flow_block_parts_override > 3) {
            fprintf(stderr, "shape flow: --flow-block-parts must be in [0,3]\n");
            goto cleanup;
        }
        flow.debug_block_parts = flow_block_parts_override;
        printf("shape flow: debug flow block parts override=%d\n", flow.debug_block_parts);
    }
    if (flow_no_rope) {
        flow.debug_disable_rope = 1;
        printf("shape flow: debug sparse RoPE disabled\n");
    }
    if (emulate_bf16_blocks) {
        flow.emulate_bf16_blocks = 1;
        printf("shape flow: bf16 block activation round-trip enabled\n");
    }
    trellis_ggml_set_flash_attn_enabled(use_ggml_flash_attn);
    if (use_ggml_flash_attn) {
        printf("shape flow: ggml flash attention enabled\n");
    }

    const size_t latent_count = (size_t) n_coords * (size_t) flow.in_channels;
    const size_t context_count = (size_t) cond_tokens * (size_t) flow.cond_channels;
    const size_t phase_count = (size_t) n_coords * (size_t) (flow.head_dim / 2);
    if (cond_values != context_count) {
        fprintf(stderr, "shape flow: unexpected cond size\n");
        goto cleanup;
    }
    neg_cond = (float *) calloc(context_count, sizeof(float));
    latent = (float *) malloc(latent_count * sizeof(float));
    pred_pos = (float *) malloc(latent_count * sizeof(float));
    pred_neg = (float *) malloc(latent_count * sizeof(float));
    pred = (float *) malloc(latent_count * sizeof(float));
    next = (float *) malloc(latent_count * sizeof(float));
    x0 = (float *) malloc(latent_count * sizeof(float));
    denorm = (float *) malloc(latent_count * sizeof(float));
    pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (neg_cond == NULL || latent == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL ||
        next == NULL || x0 == NULL || denorm == NULL || pairs == NULL || cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "shape flow: host allocation failed\n");
        goto cleanup;
    }
    if (neg_cond_path != NULL && !read_f32_file_exact(neg_cond_path, neg_cond, context_count)) {
        goto cleanup;
    }
    if (!read_f32_file_exact(noise_path, latent, latent_count)) {
        goto cleanup;
    }

    trellis_status status = trellis_flow_timestep_pairs_f32(steps, rescale_t, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_sparse_phases_f32(
            coords,
            n_coords,
            flow.head_dim,
            1.0f,
            10000.0f,
            cos_phase,
            sin_phase,
            phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow: schedule/rope failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }

    if (!mkdir_p(out_dir)) {
        fprintf(stderr, "shape flow: failed to create %s\n", out_dir);
        goto cleanup;
    }
    char manifest_path[4096];
    if (!make_join_path(out_dir, "tensors.tsv", manifest_path, sizeof(manifest_path))) {
        goto cleanup;
    }
    manifest = fopen(manifest_path, "w");
    if (manifest == NULL) {
        goto cleanup;
    }
    fprintf(manifest, "name\tshape\tfile\n");
    write_i32_file(out_dir, "shape_coords.i32", coords, (size_t) n_coords * 4u);
    write_manifest_line(manifest, "shape_coords", "Nx4", "shape_coords.i32");
    write_f32_file(out_dir, "shape_noise.f32", latent, latent_count);
    write_manifest_line(manifest, "shape_noise", "Nx32", "shape_noise.f32");
    denormalize_slat_f32(latent, n_coords, flow.in_channels, slat_mean, slat_std, denorm);
    write_f32_file(out_dir, "shape_noise_denorm.f32", denorm, latent_count);
    write_manifest_line(manifest, "shape_noise_denorm", "Nx32", "shape_noise_denorm.f32");
    write_f32_file(out_dir, "shape_rope_cos.f32", cos_phase, phase_count);
    write_manifest_line(manifest, "shape_rope_cos", "Nx64", "shape_rope_cos.f32");
    write_f32_file(out_dir, "shape_rope_sin.f32", sin_phase, phase_count);
    write_manifest_line(manifest, "shape_rope_sin", "Nx64", "shape_rope_sin.f32");

    for (int step = 0; step < steps; ++step) {
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        status = run_flow_once_tokens_host(cuda, &flow, latent, cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_pos);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape flow: conditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto cleanup;
        }
        status = run_flow_once_tokens_host(cuda, &flow, latent, neg_cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_neg);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape flow: unconditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto cleanup;
        }

        if (t >= guidance_min && t <= guidance_max) {
            cfg_rescale_combine_population_f32(
                latent,
                pred_pos,
                pred_neg,
                latent_count,
                1e-5f,
                t,
                guidance_strength,
                guidance_rescale,
                pred);
        } else {
            memcpy(pred, pred_pos, latent_count * sizeof(float));
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);

        char name[128];
        char shape[64];
        snprintf(shape, sizeof(shape), "%lldx%d", (long long) n_coords, flow.out_channels);
#define DUMP_STEP(suffix, data_ptr) do { \
            snprintf(name, sizeof(name), "step_%03d_%s.f32", step + 1, suffix); \
            if (!write_f32_file(out_dir, name, data_ptr, latent_count)) goto cleanup; \
            char tensor_name[128]; \
            snprintf(tensor_name, sizeof(tensor_name), "step_%03d_%s", step + 1, suffix); \
            write_manifest_line(manifest, tensor_name, shape, name); \
        } while (0)
        DUMP_STEP("flow_pred_pos", pred_pos);
        DUMP_STEP("flow_pred_neg", pred_neg);
        DUMP_STEP("flow_pred_cfg", pred);
        DUMP_STEP("pred_x0", x0);
        denormalize_slat_f32(x0, n_coords, flow.out_channels, slat_mean, slat_std, denorm);
        DUMP_STEP("pred_x0_denorm", denorm);
#undef DUMP_STEP
        memcpy(latent, next, latent_count * sizeof(float));
        snprintf(name, sizeof(name), "step_%03d_latent.f32", step + 1);
        if (!write_f32_file(out_dir, name, latent, latent_count)) {
            goto cleanup;
        }
        char tensor_name[128];
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_latent", step + 1);
        write_manifest_line(manifest, tensor_name, shape, name);
        denormalize_slat_f32(latent, n_coords, flow.in_channels, slat_mean, slat_std, denorm);
        snprintf(name, sizeof(name), "step_%03d_latent_denorm.f32", step + 1);
        if (!write_f32_file(out_dir, name, denorm, latent_count)) {
            goto cleanup;
        }
        snprintf(tensor_name, sizeof(tensor_name), "step_%03d_latent_denorm", step + 1);
        write_manifest_line(manifest, tensor_name, shape, name);
        fflush(manifest);

        printf("shape flow %d: step %d/%d t=%.6g -> %.6g tokens=%lld\n",
            checkpoint_resolution,
            step + 1,
            steps,
            t,
            t_prev,
            (long long) n_coords);
    }

    write_f32_file(out_dir, "shape_final_latent_norm.f32", latent, latent_count);
    write_manifest_line(manifest, "shape_final_latent_norm", "Nx32", "shape_final_latent_norm.f32");
    denormalize_slat_f32(latent, n_coords, flow.in_channels, slat_mean, slat_std, denorm);
    write_f32_file(out_dir, "shape_final_latent.f32", denorm, latent_count);
    write_manifest_line(manifest, "shape_final_latent", "Nx32", "shape_final_latent.f32");
    printf("shape flow %d: wrote %d steps for %lld tokens to %s\n",
        checkpoint_resolution,
        steps,
        (long long) n_coords,
        out_dir);
    rc = 0;

cleanup:
    if (manifest != NULL) {
        fclose(manifest);
    }
    trellis_ggml_set_flash_attn_enabled(0);
    free(coords);
    free(cond);
    free(neg_cond);
    free(latent);
    free(pred_pos);
    free(pred_neg);
    free(pred);
    free(next);
    free(x0);
    free(denorm);
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    trellis_tensor_store_free(&flow_store);
    return rc;
}

static int run_shape_decode(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    const char * decoder_override_path,
    const char * coords_path,
    int coords_force_xyz,
    const char * slat_path,
    const char * out_dir,
    int max_levels,
    int64_t max_input_tokens,
    const char * dump_dir) {
    if (cuda == NULL || model_dir == NULL || coords_path == NULL || slat_path == NULL || out_dir == NULL) {
        return 1;
    }

    char decoder_path[4096];
    if (decoder_override_path != NULL) {
        snprintf(decoder_path, sizeof(decoder_path), "%s", decoder_override_path);
    } else {
        trellis_status pstatus = trellis_make_model_path(
            model_dir,
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            decoder_path,
            sizeof(decoder_path));
        if (pstatus != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape decoder path: %s\n", trellis_status_string(pstatus));
            return 1;
        }
    }

    int rc = 1;
    int32_t * coords = NULL;
    float * slat = NULL;
    int32_t * out_coords = NULL;
    float * out_feats = NULL;
    FILE * manifest = NULL;
    trellis_tensor_store decoder_store;
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_shape_decoder_weights decoder;
    memset(&decoder, 0, sizeof(decoder));

    int64_t n_coords = 0;
    size_t slat_values = 0;
    if (!read_coords_i32_alloc(coords_path, coords_force_xyz, &coords, &n_coords)) {
        goto cleanup;
    }
    if (!read_f32_file_alloc(slat_path, &slat, &slat_values)) {
        goto cleanup;
    }
    if (n_coords <= 0 || slat_values != (size_t) n_coords * 32u) {
        fprintf(stderr, "shape decoder: slat values=%zu expected=%zu for %lld coords\n",
            slat_values,
            (size_t) n_coords * 32u,
            (long long) n_coords);
        goto cleanup;
    }
    int64_t n_use = n_coords;
    if (max_input_tokens > 0 && max_input_tokens < n_use) {
        n_use = max_input_tokens;
        printf("shape decoder: truncating input tokens %lld -> %lld\n",
            (long long) n_coords,
            (long long) n_use);
    }

    if (!load_shape_decoder(cuda, decoder_path, &decoder_store, &decoder)) {
        goto cleanup;
    }
    if (dump_dir != NULL && dump_dir[0] != '\0' && !mkdir_p(dump_dir)) {
        fprintf(stderr, "shape decoder: failed to create dump dir %s\n", dump_dir);
        goto cleanup;
    }

    int64_t n_out = 0;
    int channels_out = 0;
    trellis_shape_decoder_debug_options debug_options;
    memset(&debug_options, 0, sizeof(debug_options));
    debug_options.dump_dir = dump_dir;
    trellis_status status = trellis_shape_decoder_forward_f32_host_debug(
        &decoder,
        coords,
        slat,
        n_use,
        cuda->device,
        max_levels,
        dump_dir == NULL || dump_dir[0] == '\0' ? NULL : &debug_options,
        &out_coords,
        &out_feats,
        &n_out,
        &channels_out);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape decoder: forward failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }

    if (!mkdir_p(out_dir)) {
        fprintf(stderr, "shape decoder: failed to create %s\n", out_dir);
        goto cleanup;
    }
    char manifest_path[4096];
    if (!make_join_path(out_dir, "shape_decoder_tensors.tsv", manifest_path, sizeof(manifest_path))) {
        goto cleanup;
    }
    manifest = fopen(manifest_path, "w");
    if (manifest == NULL) {
        goto cleanup;
    }
    fprintf(manifest, "name\tshape\tfile\n");
    write_i32_file(out_dir, "shape_decoder_coords.i32", out_coords, (size_t) n_out * 4u);
    write_manifest_line(manifest, "shape_decoder_coords", "Nx4", "shape_decoder_coords.i32");
    write_f32_file(out_dir, "shape_decoder_feats.f32", out_feats, (size_t) n_out * (size_t) channels_out);
    char shape[64];
    snprintf(shape, sizeof(shape), "%lldx%d", (long long) n_out, channels_out);
    write_manifest_line(manifest, "shape_decoder_feats", shape, "shape_decoder_feats.f32");
    printf("shape decoder: wrote %lld tokens x %d channels to %s\n",
        (long long) n_out,
        channels_out,
        out_dir);
    rc = 0;

cleanup:
    if (manifest != NULL) {
        fclose(manifest);
    }
    free(coords);
    free(slat);
    free(out_coords);
    free(out_feats);
    trellis_tensor_store_free(&decoder_store);
    return rc;
}

int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * flow_path = NULL;
    const char * decoder_path = NULL;
    const char * coords_path = NULL;
    const char * cond_path = NULL;
    const char * neg_cond_path = NULL;
    const char * noise_path = NULL;
    const char * slat_path = NULL;
    const char * out_dir = "benchmark_outputs/c_shape_flow";
    const char * decode_dump_dir = NULL;
    int validate = 0;
    int validate_shape_decoder = 0;
    int cuda_check = 0;
    int timesteps = 0;
    int resolution = 512;
    int shape_flow = 0;
    int shape_decode = 0;
    int coords_force_xyz = 0;
    int steps = 12;
    int flow_blocks_override = -1;
    int flow_block_parts_override = -1;
    int flow_no_rope = 0;
    int emulate_bf16_blocks = 0;
    int use_ggml_flash_attn = 0;
    int decode_max_levels = 0;
    int64_t decode_max_input_tokens = 0;
    float rescale_t = 3.0f;
    float guidance_strength = 7.5f;
    float guidance_rescale = 0.5f;
    float guidance_min = 0.6f;
    float guidance_max = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--flow") == 0) {
            flow_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decoder") == 0) {
            decoder_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            resolution = atoi(v);
        } else if (strcmp(argv[i], "--validate") == 0) {
            validate = 1;
        } else if (strcmp(argv[i], "--validate-shape-decoder") == 0) {
            validate_shape_decoder = 1;
        } else if (strcmp(argv[i], "--shape-flow-512") == 0) {
            shape_flow = 1;
            resolution = 512;
        } else if (strcmp(argv[i], "--shape-flow-1024") == 0) {
            shape_flow = 1;
            resolution = 1024;
        } else if (strcmp(argv[i], "--shape-decode") == 0) {
            shape_decode = 1;
        } else if (strcmp(argv[i], "--coords-i32") == 0) {
            coords_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--coords-xyz") == 0) {
            coords_force_xyz = 1;
        } else if (strcmp(argv[i], "--input-cond-f32") == 0) {
            cond_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--input-neg-cond-f32") == 0) {
            neg_cond_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--input-noise-f32") == 0) {
            noise_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--input-slat-f32") == 0) {
            slat_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--out") == 0) {
            out_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            steps = atoi(v);
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            rescale_t = (float) atof(v);
        } else if (strcmp(argv[i], "--guidance-strength") == 0) {
            const char * v = arg_value(argc, argv, &i);
            guidance_strength = v == NULL ? guidance_strength : (float) atof(v);
        } else if (strcmp(argv[i], "--guidance-rescale") == 0) {
            const char * v = arg_value(argc, argv, &i);
            guidance_rescale = v == NULL ? guidance_rescale : (float) atof(v);
        } else if (strcmp(argv[i], "--guidance-min") == 0) {
            const char * v = arg_value(argc, argv, &i);
            guidance_min = v == NULL ? guidance_min : (float) atof(v);
        } else if (strcmp(argv[i], "--guidance-max") == 0) {
            const char * v = arg_value(argc, argv, &i);
            guidance_max = v == NULL ? guidance_max : (float) atof(v);
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            const char * v = arg_value(argc, argv, &i);
            flow_blocks_override = v == NULL ? flow_blocks_override : atoi(v);
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            const char * v = arg_value(argc, argv, &i);
            flow_block_parts_override = v == NULL ? flow_block_parts_override : atoi(v);
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            flow_no_rope = 1;
        } else if (strcmp(argv[i], "--emulate-bf16-blocks") == 0) {
            emulate_bf16_blocks = 1;
        } else if (strcmp(argv[i], "--use-ggml-flash-attn") == 0) {
            use_ggml_flash_attn = 1;
        } else if (strcmp(argv[i], "--decode-max-levels") == 0) {
            const char * v = arg_value(argc, argv, &i);
            decode_max_levels = v == NULL ? decode_max_levels : atoi(v);
        } else if (strcmp(argv[i], "--decode-max-input-tokens") == 0) {
            const char * v = arg_value(argc, argv, &i);
            decode_max_input_tokens = v == NULL ? decode_max_input_tokens : (int64_t) atoll(v);
        } else if (strcmp(argv[i], "--decode-dump-dir") == 0) {
            decode_dump_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--cuda-check") == 0) {
            cuda_check = 1;
        } else if (strcmp(argv[i], "--timesteps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            timesteps = atoi(v);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    trellis_cuda_context cuda;
    memset(&cuda, 0, sizeof(cuda));
    if (cuda_check || shape_flow || shape_decode) {
        trellis_status status = trellis_cuda_init(&cuda, 0);
        printf("cuda: %s\n", trellis_status_string(status));
        if (status != TRELLIS_STATUS_OK) {
            return 1;
        }
        if (cuda_check && !shape_flow) {
            trellis_cuda_free(&cuda);
            return 0;
        }
    }

    if (timesteps > 0) {
        float * pairs = (float *) calloc((size_t) timesteps * 2u, sizeof(float));
        if (pairs == NULL) {
            fprintf(stderr, "out of memory\n");
            trellis_cuda_free(&cuda);
            return 1;
        }
        trellis_status status = trellis_flow_timestep_pairs_f32(timesteps, rescale_t, pairs, (size_t) timesteps * 2u);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "timesteps: %s\n", trellis_status_string(status));
            free(pairs);
            trellis_cuda_free(&cuda);
            return 1;
        }
        for (int i = 0; i < timesteps; ++i) {
            printf("%02d %.9g %.9g\n", i + 1, pairs[2 * i + 0], pairs[2 * i + 1]);
        }
        free(pairs);
    }

    if (validate) {
        char flow_buf[4096];
        if (flow_path == NULL) {
            if (model_dir == NULL) {
                usage(argv[0]);
                trellis_cuda_free(&cuda);
                return 2;
            }
            const char * rel = NULL;
            if (resolution == 512) {
                rel = "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
            } else if (resolution == 1024) {
                rel = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors";
            } else {
                fprintf(stderr, "--resolution must be 512 or 1024\n");
                trellis_cuda_free(&cuda);
                return 2;
            }
            trellis_status pstatus = trellis_make_model_path(model_dir, rel, flow_buf, sizeof(flow_buf));
            if (pstatus != TRELLIS_STATUS_OK) {
                fprintf(stderr, "flow path: %s\n", trellis_status_string(pstatus));
                trellis_cuda_free(&cuda);
                return 1;
            }
            flow_path = flow_buf;
        }

        trellis_checkpoint_report flow_report;
        trellis_status flow_status = trellis_shape_slat_flow_validate_checkpoint(flow_path, &flow_report);
        print_report("shape SLat flow", flow_status, &flow_report);
        if (flow_status != TRELLIS_STATUS_OK) {
            trellis_cuda_free(&cuda);
            return 1;
        }
    }

    if (validate_shape_decoder) {
        if (model_dir == NULL) {
            usage(argv[0]);
            trellis_cuda_free(&cuda);
            return 2;
        }
        char decoder_path[4096];
        trellis_status pstatus = trellis_make_model_path(
            model_dir,
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            decoder_path,
            sizeof(decoder_path));
        if (pstatus != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape decoder path: %s\n", trellis_status_string(pstatus));
            trellis_cuda_free(&cuda);
            return 1;
        }
        trellis_checkpoint_report decoder_report;
        trellis_status decoder_status = trellis_shape_decoder_validate_checkpoint(decoder_path, &decoder_report);
        print_report("shape decoder", decoder_status, &decoder_report);
        if (decoder_status != TRELLIS_STATUS_OK) {
            trellis_cuda_free(&cuda);
            return 1;
        }
    }

    if (shape_flow) {
        if (model_dir == NULL || coords_path == NULL || cond_path == NULL || noise_path == NULL) {
            usage(argv[0]);
            trellis_cuda_free(&cuda);
            return 2;
        }
        int rc = run_shape_flow(
            &cuda,
            model_dir,
            flow_path,
            resolution,
            coords_path,
            coords_force_xyz,
            cond_path,
            neg_cond_path,
            noise_path,
            out_dir,
            steps,
            rescale_t,
            guidance_strength,
            guidance_rescale,
            guidance_min,
            guidance_max,
            flow_blocks_override,
            flow_block_parts_override,
            flow_no_rope,
            emulate_bf16_blocks,
            use_ggml_flash_attn);
        trellis_cuda_free(&cuda);
        return rc;
    }

    if (shape_decode) {
        if (model_dir == NULL || coords_path == NULL || slat_path == NULL) {
            usage(argv[0]);
            trellis_cuda_free(&cuda);
            return 2;
        }
        int rc = run_shape_decode(
            &cuda,
            model_dir,
            decoder_path,
            coords_path,
            coords_force_xyz,
            slat_path,
            out_dir,
            decode_max_levels,
            decode_max_input_tokens,
            decode_dump_dir);
        trellis_cuda_free(&cuda);
        return rc;
    }

    trellis_cuda_free(&cuda);
    if (!validate && !validate_shape_decoder && !cuda_check && !shape_decode && timesteps <= 0) {
        usage(argv[0]);
        return 2;
    }
    return 0;
}
