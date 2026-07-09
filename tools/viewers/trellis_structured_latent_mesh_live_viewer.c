#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_tool_live.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "external/glad.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>
#include <unistd.h>
#endif

typedef struct loaded_mesh {
    Mesh * meshes;
    int count;
    int capacity;
    int ready;
    int gpu_ready;
    unsigned int vao_id;
    unsigned int vbo_id;
    unsigned int ebo_id;
    int gpu_vertex_count;
    int gpu_index_count;
    int source_vertices;
    int source_faces;
    int drawn_faces;
} loaded_mesh;

typedef struct mesh_shader {
    Shader shader;
    int loc_mvp;
    int loc_base_color;
    int loc_wire_color;
    int loc_light_dir;
    int loc_wire_mode;
} mesh_shader;

typedef struct mesh_step_timing {
    double denorm_s;
    double decoder_s;
    double fdg_s;
    double upload_s;
} mesh_step_timing;

typedef struct face3 {
    int v[3];
} face3;

static const float MESH_BASE_COLOR[4] = {0.72f, 0.76f, 0.82f, 1.0f};
static const float MESH_AMBIENT_LIGHT = 0.36f;
static const float MESH_DIFFUSE_LIGHT = 0.64f;
static const Vector3 MESH_LIGHT_DIR = {-0.449609f, 0.719374f, 0.519548f};

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --model TRELLIS.2-4B --coords-i32 shape_coords.i32 --input-cond-f32 cond.f32 --input-neg-cond-f32 neg.f32 --input-noise-f32 noise.f32\n"
        "\n"
        "Runs C+ggml+CUDA stage2 shape denoising and decodes every step to a raylib mesh.\n"
        "\n"
        "Options:\n"
        "  --model DIR             TRELLIS.2 model directory containing ckpts/\n"
        "  --flow PATH             Override shape SLat flow safetensors path\n"
        "  --decoder PATH          Override shape decoder safetensors path\n"
        "  --resolution N          Shape checkpoint/output grid resolution, default 512\n"
        "  --coords-i32 FILE       Sparse coords, [N,4] batch,x,y,z or [N,3] x,y,z\n"
        "  --coords-xyz            Force --coords-i32 to be [N,3]\n"
        "  --input-cond-f32 FILE   Positive DINO cond tensor [1,T,1024]\n"
        "  --input-neg-cond-f32 FILE Negative DINO cond tensor [1,T,1024], default zeros\n"
        "  --input-noise-f32 FILE  Initial sparse latent feats [N,32]\n"
        "  --steps N               Euler steps, default 12\n"
        "  --rescale-t X           Timestep rescale factor, default 3.0\n"
        "  --guidance-strength X   CFG strength, default 7.5\n"
        "  --guidance-rescale X    CFG rescale, default 0.5\n"
        "  --guidance-min X        CFG interval min, default 0.6\n"
        "  --guidance-max X        CFG interval max, default 1.0\n"
        "  --source NAME           pred_x0 or x_t, default pred_x0\n"
        "  --no-initial           Do not decode initial x_t before step 1\n"
        "  --no-final              Do not decode final_x_t after the last step\n"
        "  --display DISPLAY       X11 display, e.g. :1. Auto-detected when unset\n"
        "  --xauthority PATH       Optional Xauthority file for the desktop session\n"
        "  --width N               Window width, default 1280\n"
        "  --height N              Window height, default 800\n"
        "  --max-faces N           Draw at most N sampled faces per frame; 0 draws all, default 0\n"
        "  --mesh-chunk-faces N    Faces per legacy GPU mesh chunk, default 21000\n"
        "  --mesh-upload-mode M    gpu_indexed, expanded, or indexed; default gpu_indexed\n"
        "  --mesh-style STYLE      solid, wire, or solid_wire; default solid\n"
        "  --hold SECONDS          Minimum seconds to display each decoded step, default 0.35\n"
        "  --flow-blocks N         Debug: run only first N transformer blocks\n"
        "  --flow-block-parts N    Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope          Debug: disable sparse RoPE\n"
        "  --emulate-bf16-blocks   Debug: round block activations like reference bf16 shape flow\n"
        "  --use-ggml-flash-attn   Debug: use ggml flash attention instead of explicit SDPA\n"
        "  --decode-max-levels N   Debug: run only first N shape decoder levels, default full\n"
        "  --decode-max-input-tokens N Debug: truncate input sparse tensor before decode\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int path_exists(const char * path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static int env_is_empty(const char * name) {
    const char * value = getenv(name);
    return value == NULL || value[0] == '\0';
}

static void detect_local_x11_display(char * dst, size_t dst_size) {
#ifdef _WIN32
    if (dst != NULL && dst_size > 0) {
        dst[0] = '\0';
    }
#else
    int best = -1;
    DIR * dir = opendir("/tmp/.X11-unix");
    if (dir != NULL) {
        struct dirent * ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] != 'X') {
                continue;
            }
            char * end = NULL;
            long display = strtol(ent->d_name + 1, &end, 10);
            if (end != NULL && *end == '\0' && display >= 0 && display <= 9999) {
                if (best < 0 || display < best) {
                    best = (int) display;
                }
            }
        }
        closedir(dir);
    }
    snprintf(dst, dst_size, ":%d", best >= 0 ? best : 0);
#endif
}

static int detect_xauthority(char * dst, size_t dst_size) {
#ifdef _WIN32
    (void) dst;
    (void) dst_size;
    return 0;
#else
    const char * home = getenv("HOME");
    if (home != NULL && home[0] != '\0') {
        char home_auth[PATH_MAX];
        int n = snprintf(home_auth, sizeof(home_auth), "%s/.Xauthority", home);
        if (n >= 0 && (size_t) n < sizeof(home_auth) && path_exists(home_auth)) {
            snprintf(dst, dst_size, "%s", home_auth);
            return 1;
        }
    }
    char gdm_auth[PATH_MAX];
    int n = snprintf(gdm_auth, sizeof(gdm_auth), "/run/user/%ld/gdm/Xauthority", (long) getuid());
    if (n >= 0 && (size_t) n < sizeof(gdm_auth) && path_exists(gdm_auth)) {
        snprintf(dst, dst_size, "%s", gdm_auth);
        return 1;
    }
    return 0;
#endif
}

static void configure_display_env(const char * display, const char * xauthority) {
#ifdef _WIN32
    (void) display;
    (void) xauthority;
#else
    if (display != NULL && display[0] != '\0') {
        trellis_setenv("DISPLAY", display, 1);
    } else if (env_is_empty("DISPLAY")) {
        char detected[32];
        detect_local_x11_display(detected, sizeof(detected));
        trellis_setenv("DISPLAY", detected, 1);
    }

    if (xauthority != NULL && xauthority[0] != '\0') {
        trellis_setenv("XAUTHORITY", xauthority, 1);
    } else if (env_is_empty("XAUTHORITY")) {
        char detected[PATH_MAX];
        if (detect_xauthority(detected, sizeof(detected))) {
            trellis_setenv("XAUTHORITY", detected, 1);
        }
    }
#endif
}

static int file_size_bytes(const char * path, size_t * size_out) {
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

static char * read_text_file_null_terminated(const char * path) {
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
    return data;
}

static const char * skip_json_ws(const char * p) {
    while (p != NULL && *p != '\0' && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
        ++p;
    }
    return p;
}

static int parse_json_float_array_32(const char * start, const char * key, float out[32]) {
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
    char path[4096];
    trellis_status pstatus = trellis_make_model_path(model_dir, "pipeline.json", path, sizeof(path));
    if (pstatus != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape flow normalization path: %s\n", trellis_status_string(pstatus));
        return 0;
    }
    char * json = read_text_file_null_terminated(path);
    if (json == NULL) {
        fprintf(stderr, "shape flow: failed to read %s\n", path);
        return 0;
    }
    const char * section = strstr(json, "\"shape_slat_normalization\"");
    int ok = section != NULL &&
        parse_json_float_array_32(section, "\"mean\"", mean) &&
        parse_json_float_array_32(section, "\"std\"", std);
    if (!ok) {
        fprintf(stderr, "shape flow: failed to parse shape_slat_normalization from %s\n", path);
    }
    free(json);
    return ok;
}

static int read_f32_file_alloc(const char * path, float ** data_out, size_t * count_out) {
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

static void denormalize_slat_f32(
    const float * norm,
    int64_t tokens,
    int channels,
    const float * mean,
    const float * std,
    float * out) {
    for (int64_t i = 0; i < tokens; ++i) {
        for (int c = 0; c < channels; ++c) {
            out[(size_t) i * (size_t) channels + (size_t) c] =
                norm[(size_t) i * (size_t) channels + (size_t) c] * std[c] + mean[c];
        }
    }
}

static int load_shape_flow(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_dit_flow_weights * flow) {
    fprintf(stderr, "shape live: loading shape flow from %s\n", path);
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
    status = trellis_tensor_store_load_safetensors(store, cuda, path, true, NULL);
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
    fprintf(stderr, "shape live: shape flow ready blocks=%d channels=%d\n",
        flow->n_blocks, flow->in_channels);
    return 1;
}

static int load_shape_decoder(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_shape_decoder_weights * decoder) {
    fprintf(stderr, "shape live: loading shape decoder from %s\n", path);
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
    fprintf(stderr, "shape live: shape decoder ready\n");
    return 1;
}

int trellis_tool_stage2_weights_load(
    const trellis_cuda_context * cuda,
    const char * model_dir,
    int resolution,
    const char * flow_override_path,
    const char * decoder_override_path,
    trellis_tool_stage2_weights * weights) {
    if (weights == NULL) {
        return 0;
    }
    memset(weights, 0, sizeof(*weights));
    if (cuda == NULL || model_dir == NULL || resolution <= 0) {
        return 0;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    char flow_path[4096];
    if (flow_override_path != NULL) {
        snprintf(flow_path, sizeof(flow_path), "%s", flow_override_path);
    } else {
        const char * rel = resolution == 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
        status = trellis_make_model_path(model_dir, rel, flow_path, sizeof(flow_path));
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: flow path failed: %s\n", trellis_status_string(status));
            return 0;
        }
    }
    char decoder_path[4096];
    if (decoder_override_path != NULL) {
        snprintf(decoder_path, sizeof(decoder_path), "%s", decoder_override_path);
    } else {
        status = trellis_make_model_path(model_dir, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", decoder_path, sizeof(decoder_path));
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: decoder path failed: %s\n", trellis_status_string(status));
            return 0;
        }
    }

    if (!load_shape_flow(cuda, flow_path, &weights->flow_store, &weights->flow)) {
        trellis_tool_stage2_weights_free(weights);
        return 0;
    }
    weights->has_flow = 1;
    if (!load_shape_decoder(cuda, decoder_path, &weights->decoder_store, &weights->decoder)) {
        trellis_tool_stage2_weights_free(weights);
        return 0;
    }
    weights->has_decoder = 1;
    if (!load_shape_slat_normalization(model_dir, weights->slat_mean, weights->slat_std)) {
        trellis_tool_stage2_weights_free(weights);
        return 0;
    }
    weights->has_normalization = 1;
    return 1;
}

void trellis_tool_stage2_weights_free(trellis_tool_stage2_weights * weights) {
    if (weights == NULL) {
        return;
    }
    trellis_tensor_store_free(&weights->decoder_store);
    trellis_tensor_store_free(&weights->flow_store);
    memset(weights, 0, sizeof(*weights));
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
        mean_pos += one_minus_sigma_min * x_t[i] - sigma_t * pred_pos[i];
        mean_cfg += one_minus_sigma_min * x_t[i] - sigma_t * pred[i];
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

static int sampled_index(int i, int count, int max_count) {
    if (max_count <= 0 || count <= max_count) {
        return i;
    }
    if (max_count <= 1) {
        return 0;
    }
    return (int) llround((double) i * (double) (count - 1) / (double) (max_count - 1));
}

static Vector3 transformed_vertex(const float * vertices, int index) {
    const float * src = &vertices[(size_t) index * 3u];
    Vector3 out = {src[0] * 2.0f, src[2] * 2.0f, src[1] * 2.0f};
    return out;
}

static Vector3 normal_for_triangle(Vector3 a, Vector3 b, Vector3 c) {
    Vector3 ab = Vector3Subtract(b, a);
    Vector3 ac = Vector3Subtract(c, a);
    return Vector3Normalize(Vector3CrossProduct(ab, ac));
}

static unsigned char color_byte(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    return (unsigned char) (value * 255.0f + 0.5f);
}

static void write_shaded_color(unsigned char * dst, float shade) {
    if (!isfinite(shade)) {
        shade = MESH_AMBIENT_LIGHT;
    }
    dst[0] = color_byte(MESH_BASE_COLOR[0] * shade);
    dst[1] = color_byte(MESH_BASE_COLOR[1] * shade);
    dst[2] = color_byte(MESH_BASE_COLOR[2] * shade);
    dst[3] = color_byte(MESH_BASE_COLOR[3]);
}

static const char * MESH_GPU_VS =
    "#version 330\n"
    "in vec3 vertexPosition;\n"
    "uniform mat4 mvp;\n"
    "out vec3 fragPos;\n"
    "void main() {\n"
    "    vec3 p = vec3(vertexPosition.x * 2.0, vertexPosition.z * 2.0, vertexPosition.y * 2.0);\n"
    "    fragPos = p;\n"
    "    gl_Position = mvp * vec4(p, 1.0);\n"
    "}\n";

static const char * MESH_GPU_FS =
    "#version 330\n"
    "in vec3 fragPos;\n"
    "out vec4 finalColor;\n"
    "uniform vec4 baseColor;\n"
    "uniform vec4 wireColor;\n"
    "uniform vec3 lightDir;\n"
    "uniform int wireMode;\n"
    "void main() {\n"
    "    if (wireMode != 0) {\n"
    "        finalColor = wireColor;\n"
    "        return;\n"
    "    }\n"
    "    vec3 dx = dFdx(fragPos);\n"
    "    vec3 dy = dFdy(fragPos);\n"
    "    vec3 n = normalize(cross(dx, dy));\n"
    "    float ndl = abs(dot(n, normalize(lightDir)));\n"
    "    float shade = 0.36 + 0.64 * ndl;\n"
    "    finalColor = vec4(baseColor.rgb * shade, baseColor.a);\n"
    "}\n";

static int load_mesh_shader(mesh_shader * out) {
    if (out == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->shader = LoadShaderFromMemory(MESH_GPU_VS, MESH_GPU_FS);
    if (out->shader.id == 0) {
        return 0;
    }
    out->loc_mvp = out->shader.locs[SHADER_LOC_MATRIX_MVP];
    out->loc_base_color = GetShaderLocation(out->shader, "baseColor");
    out->loc_wire_color = GetShaderLocation(out->shader, "wireColor");
    out->loc_light_dir = GetShaderLocation(out->shader, "lightDir");
    out->loc_wire_mode = GetShaderLocation(out->shader, "wireMode");
    return out->loc_mvp >= 0 && out->loc_base_color >= 0 &&
        out->loc_wire_color >= 0 && out->loc_light_dir >= 0 &&
        out->loc_wire_mode >= 0;
}

static void unload_mesh_shader(mesh_shader * shader) {
    if (shader != NULL && shader->shader.id != 0) {
        UnloadShader(shader->shader);
        memset(shader, 0, sizeof(*shader));
    }
}

static void set_mesh_shader_common(const mesh_shader * shader, int wire_mode) {
    float base_color[4] = {MESH_BASE_COLOR[0], MESH_BASE_COLOR[1], MESH_BASE_COLOR[2], 1.0f};
    float wire_color[4] = {74.0f / 255.0f, 210.0f / 255.0f, 178.0f / 255.0f, 1.0f};
    float light_dir[3] = {MESH_LIGHT_DIR.x, MESH_LIGHT_DIR.y, MESH_LIGHT_DIR.z};
    Matrix mvp = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
    rlSetUniformMatrix(shader->loc_mvp, mvp);
    rlSetUniform(shader->loc_base_color, base_color, RL_SHADER_UNIFORM_VEC4, 1);
    rlSetUniform(shader->loc_wire_color, wire_color, RL_SHADER_UNIFORM_VEC4, 1);
    rlSetUniform(shader->loc_light_dir, light_dir, RL_SHADER_UNIFORM_VEC3, 1);
    rlSetUniform(shader->loc_wire_mode, &wire_mode, RL_SHADER_UNIFORM_INT, 1);
}

static int append_uploaded_mesh(loaded_mesh * out, Mesh mesh, int faces) {
    if (out->count == out->capacity) {
        int next_capacity = out->capacity == 0 ? 8 : out->capacity * 2;
        Mesh * next = (Mesh *) realloc(out->meshes, (size_t) next_capacity * sizeof(Mesh));
        if (next == NULL) {
            return 0;
        }
        out->meshes = next;
        out->capacity = next_capacity;
    }
    out->meshes[out->count++] = mesh;
    out->drawn_faces += faces;
    out->ready = 1;
    return 1;
}

static int upload_expanded_chunk(
    const float * vertices,
    const face3 * faces,
    int face_count,
    loaded_mesh * out) {
    const int vertex_count = face_count * 3;
    float * vertex_data = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    unsigned char * color_data = (unsigned char *) malloc((size_t) vertex_count * 4u);
    if (vertex_data == NULL || color_data == NULL) {
        free(vertex_data);
        free(color_data);
        return 0;
    }

    for (int i = 0; i < face_count; ++i) {
        Vector3 tri[3];
        for (int k = 0; k < 3; ++k) {
            tri[k] = transformed_vertex(vertices, faces[i].v[k]);
            float * dst = &vertex_data[(size_t) (i * 3 + k) * 3u];
            dst[0] = tri[k].x;
            dst[1] = tri[k].y;
            dst[2] = tri[k].z;
        }
        Vector3 normal = normal_for_triangle(tri[0], tri[1], tri[2]);
        float intensity = fabsf(Vector3DotProduct(normal, MESH_LIGHT_DIR));
        float shade = MESH_AMBIENT_LIGHT + MESH_DIFFUSE_LIGHT * intensity;
        for (int k = 0; k < 3; ++k) {
            write_shaded_color(&color_data[(size_t) (i * 3 + k) * 4u], shade);
        }
    }

    Mesh mesh = {0};
    mesh.vertexCount = vertex_count;
    mesh.triangleCount = face_count;
    mesh.vertices = vertex_data;
    mesh.colors = color_data;
    UploadMesh(&mesh, false);
    if (!append_uploaded_mesh(out, mesh, face_count)) {
        UnloadMesh(mesh);
        return 0;
    }
    return 1;
}

static int upload_indexed_chunk(
    const float * vertices,
    int vertex_source_count,
    const face3 * faces,
    int face_count,
    int * local_map,
    loaded_mesh * out) {
    const int max_unique = face_count * 3;
    int * unique = (int *) malloc((size_t) max_unique * sizeof(int));
    unsigned short * indices = (unsigned short *) malloc((size_t) face_count * 3u * sizeof(unsigned short));
    if (unique == NULL || indices == NULL) {
        free(unique);
        free(indices);
        return 0;
    }

    int unique_count = 0;
    for (int i = 0; i < face_count; ++i) {
        for (int k = 0; k < 3; ++k) {
            int id = faces[i].v[k];
            if (id < 0 || id >= vertex_source_count) {
                free(unique);
                free(indices);
                return 0;
            }
            int local = local_map[id];
            if (local < 0) {
                if (unique_count >= 65535) {
                    free(unique);
                    free(indices);
                    return 0;
                }
                local = unique_count;
                local_map[id] = local;
                unique[unique_count++] = id;
            }
            indices[(size_t) i * 3u + (size_t) k] = (unsigned short) local;
        }
    }

    float * vertex_data = (float *) malloc((size_t) unique_count * 3u * sizeof(float));
    unsigned char * color_data = (unsigned char *) malloc((size_t) unique_count * 4u);
    Vector3 * normals = (Vector3 *) calloc((size_t) unique_count, sizeof(Vector3));
    if (vertex_data == NULL || color_data == NULL || normals == NULL) {
        free(vertex_data);
        free(color_data);
        free(normals);
        for (int i = 0; i < unique_count; ++i) local_map[unique[i]] = -1;
        free(unique);
        free(indices);
        return 0;
    }

    for (int i = 0; i < unique_count; ++i) {
        Vector3 v = transformed_vertex(vertices, unique[i]);
        vertex_data[(size_t) i * 3u + 0u] = v.x;
        vertex_data[(size_t) i * 3u + 1u] = v.y;
        vertex_data[(size_t) i * 3u + 2u] = v.z;
    }
    for (int i = 0; i < face_count; ++i) {
        int ia = indices[(size_t) i * 3u + 0u];
        int ib = indices[(size_t) i * 3u + 1u];
        int ic = indices[(size_t) i * 3u + 2u];
        Vector3 a = {vertex_data[(size_t) ia * 3u + 0u], vertex_data[(size_t) ia * 3u + 1u], vertex_data[(size_t) ia * 3u + 2u]};
        Vector3 b = {vertex_data[(size_t) ib * 3u + 0u], vertex_data[(size_t) ib * 3u + 1u], vertex_data[(size_t) ib * 3u + 2u]};
        Vector3 c = {vertex_data[(size_t) ic * 3u + 0u], vertex_data[(size_t) ic * 3u + 1u], vertex_data[(size_t) ic * 3u + 2u]};
        Vector3 normal = normal_for_triangle(a, b, c);
        normals[ia] = Vector3Add(normals[ia], normal);
        normals[ib] = Vector3Add(normals[ib], normal);
        normals[ic] = Vector3Add(normals[ic], normal);
    }
    for (int i = 0; i < unique_count; ++i) {
        Vector3 normal = Vector3Normalize(normals[i]);
        float intensity = fabsf(Vector3DotProduct(normal, MESH_LIGHT_DIR));
        float shade = MESH_AMBIENT_LIGHT + MESH_DIFFUSE_LIGHT * intensity;
        write_shaded_color(&color_data[(size_t) i * 4u], shade);
        local_map[unique[i]] = -1;
    }

    Mesh mesh = {0};
    mesh.vertexCount = unique_count;
    mesh.triangleCount = face_count;
    mesh.vertices = vertex_data;
    mesh.indices = indices;
    mesh.colors = color_data;
    UploadMesh(&mesh, false);
    int ok = append_uploaded_mesh(out, mesh, face_count);
    if (!ok) {
        UnloadMesh(mesh);
    }
    free(normals);
    free(unique);
    return ok;
}

static int upload_gpu_indexed_mesh_host(
    const trellis_mesh_host * mesh,
    int max_faces,
    loaded_mesh * out) {
    memset(out, 0, sizeof(*out));
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > INT_MAX || mesh->n_faces > INT_MAX) {
        return 0;
    }

    const int vertex_count = (int) mesh->n_vertices;
    const int source_face_count = (int) mesh->n_faces;
    int draw_faces = source_face_count;
    if (max_faces > 0 && draw_faces > max_faces) {
        draw_faces = max_faces;
    }
    const int64_t vertex_bytes64 = (int64_t) vertex_count * 3ll * (int64_t) sizeof(float);
    const int64_t index_count64 = (int64_t) draw_faces * 3ll;
    const int64_t index_bytes64 = index_count64 * (int64_t) sizeof(uint32_t);
    if (vertex_bytes64 <= 0 || vertex_bytes64 > INT_MAX ||
        index_count64 <= 0 || index_count64 > INT_MAX ||
        index_bytes64 <= 0 || index_bytes64 > INT_MAX) {
        return 0;
    }

    const void * index_src = mesh->faces;
    uint32_t * sampled_indices = NULL;
    int written_faces = draw_faces;
    if (draw_faces != source_face_count) {
        sampled_indices = (uint32_t *) malloc((size_t) index_count64 * sizeof(uint32_t));
        if (sampled_indices == NULL) {
            return 0;
        }
        written_faces = 0;
        for (int i = 0; i < draw_faces; ++i) {
            int idx = sampled_index(i, source_face_count, draw_faces);
            const int32_t * f = mesh->faces + (size_t) idx * 3u;
            if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
                f[0] >= vertex_count || f[1] >= vertex_count || f[2] >= vertex_count) {
                continue;
            }
            sampled_indices[(size_t) written_faces * 3u + 0u] = (uint32_t) f[0];
            sampled_indices[(size_t) written_faces * 3u + 1u] = (uint32_t) f[1];
            sampled_indices[(size_t) written_faces * 3u + 2u] = (uint32_t) f[2];
            ++written_faces;
        }
        if (written_faces == 0) {
            free(sampled_indices);
            return 0;
        }
        index_src = sampled_indices;
    }

    out->vao_id = rlLoadVertexArray();
    if (out->vao_id == 0 || !rlEnableVertexArray(out->vao_id)) {
        free(sampled_indices);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->vbo_id = rlLoadVertexBuffer(mesh->vertices, (int) vertex_bytes64, false);
    rlEnableVertexBuffer(out->vbo_id);
    rlSetVertexAttribute(0, 3, RL_FLOAT, false, 3 * (int) sizeof(float), 0);
    rlEnableVertexAttribute(0);
    out->ebo_id = rlLoadVertexBufferElement(index_src, written_faces * 3 * (int) sizeof(uint32_t), false);
    rlDisableVertexArray();
    rlDisableVertexBuffer();
    rlDisableVertexBufferElement();
    free(sampled_indices);

    if (out->vbo_id == 0 || out->ebo_id == 0) {
        if (out->vao_id != 0) rlUnloadVertexArray(out->vao_id);
        if (out->vbo_id != 0) rlUnloadVertexBuffer(out->vbo_id);
        if (out->ebo_id != 0) rlUnloadVertexBuffer(out->ebo_id);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->ready = 1;
    out->gpu_ready = 1;
    out->gpu_vertex_count = vertex_count;
    out->gpu_index_count = written_faces * 3;
    out->source_vertices = vertex_count;
    out->source_faces = source_face_count;
    out->drawn_faces = written_faces;
    out->count = 1;
    return 1;
}

static void unload_loaded_mesh(loaded_mesh * mesh) {
    if (mesh != NULL && mesh->gpu_ready) {
        if (mesh->vao_id != 0) rlUnloadVertexArray(mesh->vao_id);
        if (mesh->vbo_id != 0) rlUnloadVertexBuffer(mesh->vbo_id);
        if (mesh->ebo_id != 0) rlUnloadVertexBuffer(mesh->ebo_id);
        memset(mesh, 0, sizeof(*mesh));
    } else if (mesh != NULL && mesh->ready) {
        for (int i = 0; i < mesh->count; ++i) {
            UnloadMesh(mesh->meshes[i]);
        }
        free(mesh->meshes);
        memset(mesh, 0, sizeof(*mesh));
    }
}

static int upload_mesh_host(
    const trellis_mesh_host * mesh,
    int max_faces,
    int mesh_chunk_faces,
    const char * upload_mode,
    loaded_mesh * out) {
    memset(out, 0, sizeof(*out));
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > INT_MAX || mesh->n_faces > INT_MAX) {
        return 0;
    }
    if (strcmp(upload_mode, "gpu_indexed") == 0) {
        return upload_gpu_indexed_mesh_host(mesh, max_faces, out);
    }
    const int vertex_count = (int) mesh->n_vertices;
    const int source_face_count = (int) mesh->n_faces;
    int draw_faces = source_face_count;
    if (max_faces > 0 && draw_faces > max_faces) {
        draw_faces = max_faces;
    }
    face3 * selected_faces = (face3 *) malloc((size_t) draw_faces * sizeof(face3));
    if (selected_faces == NULL && draw_faces != 0) {
        return 0;
    }
    int written = 0;
    for (int i = 0; i < draw_faces; ++i) {
        int idx = sampled_index(i, source_face_count, draw_faces);
        const int32_t * f = mesh->faces + (size_t) idx * 3u;
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= vertex_count || f[1] >= vertex_count || f[2] >= vertex_count) {
            continue;
        }
        selected_faces[written].v[0] = (int) f[0];
        selected_faces[written].v[1] = (int) f[1];
        selected_faces[written].v[2] = (int) f[2];
        ++written;
    }
    if (written == 0) {
        free(selected_faces);
        return 0;
    }

    int chunk_size = mesh_chunk_faces <= 0 ? 21000 : mesh_chunk_faces;
    if (chunk_size > 21000) {
        chunk_size = 21000;
    }
    out->source_vertices = vertex_count;
    out->source_faces = source_face_count;
    int ok = 1;
    int * local_map = NULL;
    if (strcmp(upload_mode, "indexed") == 0) {
        local_map = (int *) malloc((size_t) vertex_count * sizeof(int));
        if (local_map == NULL) {
            ok = 0;
        } else {
            for (int i = 0; i < vertex_count; ++i) local_map[i] = -1;
        }
    }
    for (int start = 0; ok && start < written; start += chunk_size) {
        int count = written - start;
        if (count > chunk_size) count = chunk_size;
        if (strcmp(upload_mode, "indexed") == 0) {
            ok = upload_indexed_chunk(mesh->vertices, vertex_count, selected_faces + start, count, local_map, out);
        } else {
            ok = upload_expanded_chunk(mesh->vertices, selected_faces + start, count, out);
        }
    }
    free(local_map);
    free(selected_faces);
    if (!ok) {
        unload_loaded_mesh(out);
    }
    return ok;
}

static void draw_loaded_meshes(
    const loaded_mesh * loaded,
    Material material,
    const mesh_shader * shader,
    const char * style) {
    if (loaded == NULL || !loaded->ready || loaded->count <= 0) {
        return;
    }
    if (loaded->gpu_ready) {
        if (shader == NULL || shader->shader.id == 0 || loaded->vao_id == 0 || loaded->gpu_index_count <= 0) {
            return;
        }
        rlDrawRenderBatchActive();
        rlDisableBackfaceCulling();
        rlEnableShader(shader->shader.id);
        rlEnableVertexArray(loaded->vao_id);
        if (strcmp(style, "wire") != 0) {
            set_mesh_shader_common(shader, 0);
            glDrawElements(GL_TRIANGLES, loaded->gpu_index_count, GL_UNSIGNED_INT, 0);
        }
        if (strcmp(style, "solid") != 0) {
            set_mesh_shader_common(shader, 1);
            rlEnableWireMode();
            glDrawElements(GL_TRIANGLES, loaded->gpu_index_count, GL_UNSIGNED_INT, 0);
            rlDisableWireMode();
        }
        rlDisableVertexArray();
        rlDisableShader();
        rlEnableBackfaceCulling();
        return;
    }
    rlDisableBackfaceCulling();
    if (strcmp(style, "wire") != 0) {
        material.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
        for (int i = 0; i < loaded->count; ++i) {
            DrawMesh(loaded->meshes[i], material, MatrixIdentity());
        }
    }
    if (strcmp(style, "solid") != 0) {
        material.maps[MATERIAL_MAP_DIFFUSE].color = (Color) {74, 210, 178, 255};
        rlEnableWireMode();
        for (int i = 0; i < loaded->count; ++i) {
            DrawMesh(loaded->meshes[i], material, MatrixIdentity());
        }
        rlDisableWireMode();
    }
    rlEnableBackfaceCulling();
}

static void update_camera_from_input(float * yaw, float * pitch, float * distance) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.008f;
        *pitch -= delta.y * 0.008f;
        if (*pitch < -1.35f) *pitch = -1.35f;
        if (*pitch > 1.35f) *pitch = 1.35f;
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        *distance *= powf(0.9f, wheel);
        if (*distance < 1.2f) *distance = 1.2f;
        if (*distance > 8.0f) *distance = 8.0f;
    }
}

static void draw_scene(
    const loaded_mesh * loaded,
    Material material,
    const mesh_shader * shader,
    const char * mesh_style,
    const char * title,
    const char * detail,
    float * yaw,
    float * pitch,
    float * distance) {
    update_camera_from_input(yaw, pitch, distance);
    float radius_xz = cosf(*pitch) * *distance;
    Camera3D camera = {
        .position = {sinf(*yaw) * radius_xz, sinf(*pitch) * *distance, cosf(*yaw) * radius_xz},
        .target = {0.0f, 0.0f, 0.0f},
        .up = {0.0f, 1.0f, 0.0f},
        .fovy = 45.0f,
        .projection = CAMERA_PERSPECTIVE,
    };
    BeginDrawing();
    ClearBackground((Color) {18, 20, 24, 255});
    BeginMode3D(camera);
    draw_loaded_meshes(loaded, material, shader, mesh_style);
    EndMode3D();
    DrawRectangle(0, 0, GetScreenWidth(), 90, (Color) {0, 0, 0, 175});
    DrawText(title, 18, 16, 22, RAYWHITE);
    DrawText(detail, 18, 50, 16, (Color) {195, 204, 216, 255});
    DrawText("live stage2 denoise/decode; drag rotate; wheel zoom; Esc close", 18, GetScreenHeight() - 30, 16, (Color) {180, 188, 200, 255});
    DrawFPS(GetScreenWidth() - 96, 14);
    EndDrawing();
}

static int hold_frame(
    const loaded_mesh * loaded,
    Material material,
    const mesh_shader * shader,
    const char * mesh_style,
    const char * title,
    const char * detail,
    float hold,
    float * yaw,
    float * pitch,
    float * distance) {
    double until = GetTime() + (double) hold;
    do {
        if (WindowShouldClose()) {
            return 0;
        }
        draw_scene(loaded, material, shader, mesh_style, title, detail, yaw, pitch, distance);
    } while (GetTime() < until);
    return 1;
}

static int decode_upload_step(
    const trellis_cuda_context * cuda,
    const trellis_shape_decoder_weights * decoder,
    const int32_t * coords,
    int64_t n_coords,
    const float * latent_norm,
    const float * mean,
    const float * std,
    int resolution,
    int decode_max_levels,
    int64_t decode_max_input_tokens,
    int max_faces,
    int mesh_chunk_faces,
    const char * mesh_upload_mode,
    loaded_mesh * loaded,
    int64_t * raw_vertices,
    int64_t * raw_faces,
    mesh_step_timing * timing) {
    if (timing != NULL) {
        memset(timing, 0, sizeof(*timing));
    }
    double t0 = GetTime();
    const int64_t n_use = decode_max_input_tokens > 0 && decode_max_input_tokens < n_coords ? decode_max_input_tokens : n_coords;
    float * denorm = (float *) malloc((size_t) n_use * 32u * sizeof(float));
    int32_t * dec_coords = NULL;
    float * dec_feats = NULL;
    int64_t n_dec = 0;
    int channels_dec = 0;
    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (denorm == NULL) {
        return 0;
    }
    denormalize_slat_f32(latent_norm, n_use, 32, mean, std, denorm);
    double t1 = GetTime();
    trellis_status status = trellis_shape_decoder_forward_f32_host(
        decoder,
        coords,
        denorm,
        n_use,
        cuda->device,
        decode_max_levels,
        &dec_coords,
        &dec_feats,
        &n_dec,
        &channels_dec);
    free(denorm);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape live decode: decoder failed: %s\n", trellis_status_string(status));
        free(dec_coords);
        free(dec_feats);
        return 0;
    }
    double t2 = GetTime();
    if (channels_dec != 7) {
        fprintf(stderr, "shape live decode: decoder produced %d channels, need 7 for mesh\n", channels_dec);
        free(dec_coords);
        free(dec_feats);
        return 0;
    }
    status = trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
        dec_coords,
        dec_feats,
        n_dec,
        channels_dec,
        resolution,
        &mesh);
    free(dec_coords);
    free(dec_feats);
    double t3 = GetTime();
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape live decode: flexible dual grid mesh failed: %s\n", trellis_status_string(status));
        return 0;
    }
    unload_loaded_mesh(loaded);
    int ok = upload_mesh_host(&mesh, max_faces, mesh_chunk_faces, mesh_upload_mode, loaded);
    double t4 = GetTime();
    if (raw_vertices != NULL) *raw_vertices = mesh.n_vertices;
    if (raw_faces != NULL) *raw_faces = mesh.n_faces;
    if (timing != NULL) {
        timing->denorm_s = t1 - t0;
        timing->decoder_s = t2 - t1;
        timing->fdg_s = t3 - t2;
        timing->upload_s = t4 - t3;
    }
    trellis_mesh_free(&mesh);
    return ok;
}

static float stage2_lcg_uniform(uint32_t * state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float) ((*state >> 8) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

static void stage2_fill_gaussian_latent(float * dst, size_t count, uint32_t seed) {
    uint32_t state = seed == 0 ? 1u : seed;
    const float two_pi = 6.2831853071795864769f;
    for (size_t i = 0; i < count; i += 2) {
        float u1 = stage2_lcg_uniform(&state);
        float u2 = stage2_lcg_uniform(&state);
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

int trellis_tool_run_stage2_mesh_live(const trellis_tool_stage2_live_options * options) {
    if (options == NULL || options->model_dir == NULL || options->coords_bxyz == NULL ||
        options->n_coords <= 0 || options->cond == NULL || options->cond_tokens <= 0 ||
        options->steps <= 0 || options->resolution <= 0) {
        return 2;
    }
    const char * source = options->source == NULL ? "pred_x0" : options->source;
    const char * mesh_upload_mode = options->mesh_upload_mode == NULL ? "gpu_indexed" : options->mesh_upload_mode;
    const char * mesh_style = options->mesh_style == NULL ? "solid" : options->mesh_style;
    if ((strcmp(source, "pred_x0") != 0 && strcmp(source, "x_t") != 0) ||
        (strcmp(mesh_upload_mode, "gpu_indexed") != 0 && strcmp(mesh_upload_mode, "expanded") != 0 && strcmp(mesh_upload_mode, "indexed") != 0) ||
        (strcmp(mesh_style, "solid") != 0 && strcmp(mesh_style, "wire") != 0 && strcmp(mesh_style, "solid_wire") != 0)) {
        return 2;
    }

    if (!options->use_existing_window) {
        configure_display_env(options->display, options->xauthority);
    }
    trellis_cuda_context owned_cuda;
    memset(&owned_cuda, 0, sizeof(owned_cuda));
    const trellis_cuda_context * cuda = options->cuda;
    int owns_cuda = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (cuda == NULL) {
        status = trellis_cuda_init(&owned_cuda, options->device);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "CUDA init failed: %s\n", trellis_status_string(status));
            return 1;
        }
        cuda = &owned_cuda;
        owns_cuda = 1;
    }

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
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    float slat_mean[32];
    float slat_std[32];
    trellis_tensor_store flow_store;
    trellis_tensor_store decoder_store;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_dit_flow_weights flow;
    trellis_shape_decoder_weights decoder;
    memset(&flow, 0, sizeof(flow));
    memset(&decoder, 0, sizeof(decoder));
    int owns_weight_stores = 0;

    const int64_t n_coords = options->n_coords;
    const int cond_tokens = options->cond_tokens;
    coords = (int32_t *) malloc((size_t) n_coords * 4u * sizeof(int32_t));
    cond = (float *) malloc((size_t) cond_tokens * 1024u * sizeof(float));
    if (coords == NULL || cond == NULL) {
        fprintf(stderr, "shape live: input copy allocation failed\n");
        goto cleanup;
    }
    memcpy(coords, options->coords_bxyz, (size_t) n_coords * 4u * sizeof(int32_t));
    memcpy(cond, options->cond, (size_t) cond_tokens * 1024u * sizeof(float));

    char flow_path[4096];
    if (options->flow_override_path != NULL) {
        snprintf(flow_path, sizeof(flow_path), "%s", options->flow_override_path);
    } else {
        const char * rel = options->resolution == 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
        status = trellis_make_model_path(options->model_dir, rel, flow_path, sizeof(flow_path));
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    char decoder_path[4096];
    if (options->decoder_override_path != NULL) {
        snprintf(decoder_path, sizeof(decoder_path), "%s", options->decoder_override_path);
    } else {
        status = trellis_make_model_path(options->model_dir, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", decoder_path, sizeof(decoder_path));
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    if (options->weights != NULL &&
        options->weights->has_flow &&
        options->weights->has_decoder &&
        options->weights->has_normalization) {
        flow = options->weights->flow;
        decoder = options->weights->decoder;
        memcpy(slat_mean, options->weights->slat_mean, sizeof(slat_mean));
        memcpy(slat_std, options->weights->slat_std, sizeof(slat_std));
    } else {
        if (!load_shape_flow(cuda, flow_path, &flow_store, &flow)) goto cleanup;
        owns_weight_stores = 1;
        if (!load_shape_decoder(cuda, decoder_path, &decoder_store, &decoder)) goto cleanup;
        if (!load_shape_slat_normalization(options->model_dir, slat_mean, slat_std)) goto cleanup;
    }
    if (flow.in_channels != 32 || flow.out_channels != 32) {
        fprintf(stderr, "shape live: expected shape flow channels 32->32\n");
        goto cleanup;
    }
    if (options->flow_blocks_override >= 0) {
        if (options->flow_blocks_override > flow.n_blocks) {
            fprintf(stderr, "shape live: --flow-blocks %d exceeds checkpoint blocks %d\n",
                options->flow_blocks_override,
                flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = options->flow_blocks_override;
    }
    if (options->flow_block_parts_override >= 0) {
        flow.debug_block_parts = options->flow_block_parts_override;
    }
    if (options->flow_no_rope) {
        flow.debug_disable_rope = 1;
    }
    if (options->emulate_bf16_blocks) {
        flow.emulate_bf16_blocks = 1;
    }
    trellis_ggml_set_flash_attn_enabled(options->use_ggml_flash_attn);

    const size_t latent_count = (size_t) n_coords * (size_t) flow.in_channels;
    const size_t context_count = (size_t) cond_tokens * (size_t) flow.cond_channels;
    const size_t phase_count = (size_t) n_coords * (size_t) (flow.head_dim / 2);
    if (context_count != (size_t) cond_tokens * 1024u) {
        fprintf(stderr, "shape live: unexpected cond size\n");
        goto cleanup;
    }
    neg_cond = (float *) calloc(context_count, sizeof(float));
    latent = (float *) malloc(latent_count * sizeof(float));
    pred_pos = (float *) malloc(latent_count * sizeof(float));
    pred_neg = (float *) malloc(latent_count * sizeof(float));
    pred = (float *) malloc(latent_count * sizeof(float));
    next = (float *) malloc(latent_count * sizeof(float));
    x0 = (float *) malloc(latent_count * sizeof(float));
    pairs = (float *) malloc((size_t) options->steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (neg_cond == NULL || latent == NULL || pred_pos == NULL || pred_neg == NULL ||
        pred == NULL || next == NULL || x0 == NULL || pairs == NULL || cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "shape live: host allocation failed\n");
        goto cleanup;
    }
    if (options->neg_cond != NULL) {
        memcpy(neg_cond, options->neg_cond, context_count * sizeof(float));
    }
    if (options->noise != NULL) {
        memcpy(latent, options->noise, latent_count * sizeof(float));
    } else {
        stage2_fill_gaussian_latent(latent, latent_count, options->noise_seed);
    }
    status = trellis_flow_timestep_pairs_f32(options->steps, options->rescale_t, pairs, (size_t) options->steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_sparse_phases_f32(coords, n_coords, flow.head_dim, 1.0f, 10000.0f, cos_phase, sin_phase, phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape live: schedule/rope failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }

    int opened_window = 0;
    if (!options->use_existing_window) {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
        InitWindow(options->width, options->height, "TRELLIS.2 live image-to-3D");
        SetTargetFPS(60);
        opened_window = 1;
    }
    Material material = LoadMaterialDefault();
    material.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    mesh_shader shader;
    memset(&shader, 0, sizeof(shader));
    loaded_mesh loaded;
    memset(&loaded, 0, sizeof(loaded));
    if (strcmp(mesh_upload_mode, "gpu_indexed") == 0 && !load_mesh_shader(&shader)) {
        fprintf(stderr, "shape live: failed to load gpu indexed mesh shader\n");
        goto viewer_done;
    }
    float yaw = 0.75f;
    float pitch = 0.35f;
    float distance = 3.2f;
    char title[256];
    char detail[512];

    snprintf(title, sizeof(title), "stage2 shape loading");
    snprintf(detail, sizeof(detail), "tokens=%lld cond_tokens=%d; loading CUDA graphs", (long long) n_coords, cond_tokens);
    hold_frame(&loaded, material, &shader, mesh_style, title, detail, 0.1f, &yaw, &pitch, &distance);

    if (options->decode_initial && !WindowShouldClose()) {
        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (decode_upload_step(cuda, &decoder, coords, n_coords, latent, slat_mean, slat_std, options->resolution,
                options->decode_max_levels, options->decode_max_input_tokens, options->max_faces,
                options->mesh_chunk_faces, mesh_upload_mode, &loaded, &raw_v, &raw_f, &mesh_timing)) {
            snprintf(title, sizeof(title), "stage2 x_t step 0/%d", options->steps);
            snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn; %lld vertices; initial noise",
                loaded.drawn_faces < raw_f ? "sampled" : "all",
                loaded.drawn_faces, (long long) raw_f, (long long) raw_v);
            printf("shape live: step 0/%d source=x_t mesh=%lld vertices %lld faces drawn=%d/%lld decode=%.3fs fdg=%.3fs upload=%.3fs\n",
                options->steps, (long long) raw_v, (long long) raw_f,
                loaded.drawn_faces, (long long) raw_f,
                mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
            if (!hold_frame(&loaded, material, &shader, mesh_style, title, detail, options->hold, &yaw, &pitch, &distance)) goto viewer_done;
        }
    }

    for (int step = 0; step < options->steps && !WindowShouldClose(); ++step) {
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        snprintf(title, sizeof(title), "stage2 denoising step %d/%d", step + 1, options->steps);
        snprintf(detail, sizeof(detail), "running flow t=%.6g -> %.6g; previous mesh remains visible", t, t_prev);
        draw_scene(&loaded, material, &shader, mesh_style, title, detail, &yaw, &pitch, &distance);

        double flow_t0 = GetTime();
        status = run_flow_once_tokens_host(cuda, &flow, latent, cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_pos);
        double flow_t1 = GetTime();
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: conditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto viewer_done;
        }
        status = run_flow_once_tokens_host(cuda, &flow, latent, neg_cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_neg);
        double flow_t2 = GetTime();
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: unconditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto viewer_done;
        }
        if (t >= options->guidance_min && t <= options->guidance_max) {
            cfg_rescale_combine_population_f32(
                latent, pred_pos, pred_neg, latent_count, 1e-5f, t,
                options->guidance_strength, options->guidance_rescale, pred);
        } else {
            memcpy(pred, pred_pos, latent_count * sizeof(float));
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);
        double flow_t3 = GetTime();
        const float * view_latent = strcmp(source, "x_t") == 0 ? next : x0;

        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (!decode_upload_step(cuda, &decoder, coords, n_coords, view_latent, slat_mean, slat_std, options->resolution,
                options->decode_max_levels, options->decode_max_input_tokens, options->max_faces,
                options->mesh_chunk_faces, mesh_upload_mode, &loaded, &raw_v, &raw_f, &mesh_timing)) {
            fprintf(stderr, "shape live: decode/upload step %d failed\n", step + 1);
            goto viewer_done;
        }
        memcpy(latent, next, latent_count * sizeof(float));
        snprintf(title, sizeof(title), "stage2 %s step %d/%d", source, step + 1, options->steps);
        snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn; %lld vertices; t=%.6g -> %.6g",
            loaded.drawn_faces < raw_f ? "sampled" : "all",
            loaded.drawn_faces, (long long) raw_f, (long long) raw_v, t, t_prev);
        printf("shape live: step %d/%d source=%s mesh=%lld vertices %lld faces drawn=%d/%lld%s flow+=%.3fs flow-=%.3fs cfg=%.3fs decode=%.3fs fdg=%.3fs upload=%.3fs\n",
            step + 1, options->steps, source, (long long) raw_v, (long long) raw_f,
            loaded.drawn_faces, (long long) raw_f, loaded.drawn_faces < raw_f ? " sampled" : "",
            flow_t1 - flow_t0, flow_t2 - flow_t1, flow_t3 - flow_t2,
            mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
        if (!hold_frame(&loaded, material, &shader, mesh_style, title, detail, options->hold, &yaw, &pitch, &distance)) {
            goto viewer_done;
        }
    }

    if (options->decode_final && !WindowShouldClose()) {
        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (decode_upload_step(cuda, &decoder, coords, n_coords, latent, slat_mean, slat_std, options->resolution,
                options->decode_max_levels, options->decode_max_input_tokens, options->max_faces,
                options->mesh_chunk_faces, mesh_upload_mode, &loaded, &raw_v, &raw_f, &mesh_timing)) {
            snprintf(title, sizeof(title), "stage2 final_x_t step %d/%d", options->steps, options->steps);
            snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn; %lld vertices; final decode",
                loaded.drawn_faces < raw_f ? "sampled" : "all",
                loaded.drawn_faces, (long long) raw_f, (long long) raw_v);
            printf("shape live: final step %d/%d source=x_t mesh=%lld vertices %lld faces drawn=%d/%lld decode=%.3fs fdg=%.3fs upload=%.3fs\n",
                options->steps, options->steps, (long long) raw_v, (long long) raw_f,
                loaded.drawn_faces, (long long) raw_f,
                mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
            while (!WindowShouldClose()) {
                draw_scene(&loaded, material, &shader, mesh_style, title, detail, &yaw, &pitch, &distance);
            }
        }
    }
    rc = 0;

viewer_done:
    unload_loaded_mesh(&loaded);
    unload_mesh_shader(&shader);
    UnloadMaterial(material);
    if (opened_window) {
        CloseWindow();
    }

cleanup:
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
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    if (owns_weight_stores) {
        trellis_tensor_store_free(&flow_store);
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_cuda) {
        trellis_cuda_free(&owned_cuda);
    }
    return rc;
}

#ifndef TRELLIS_TOOL_LIBRARY
int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * flow_override_path = NULL;
    const char * decoder_override_path = NULL;
    const char * coords_path = NULL;
    const char * cond_path = NULL;
    const char * neg_cond_path = NULL;
    const char * noise_path = NULL;
    const char * source = "pred_x0";
    const char * display = NULL;
    const char * xauthority = NULL;
    int resolution = 512;
    int coords_force_xyz = 0;
    int steps = 12;
    float rescale_t = 3.0f;
    float guidance_strength = 7.5f;
    float guidance_rescale = 0.5f;
    float guidance_min = 0.6f;
    float guidance_max = 1.0f;
    int decode_initial = 1;
    int decode_final = 1;
    int width = 1280;
    int height = 800;
    int max_faces = 0;
    int mesh_chunk_faces = 21000;
    const char * mesh_upload_mode = "gpu_indexed";
    const char * mesh_style = "solid";
    float hold = 0.35f;
    int flow_blocks_override = -1;
    int flow_block_parts_override = -1;
    int flow_no_rope = 0;
    int emulate_bf16_blocks = 0;
    int use_ggml_flash_attn = 0;
    int decode_max_levels = 0;
    int64_t decode_max_input_tokens = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--flow") == 0) {
            flow_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decoder") == 0) {
            decoder_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            resolution = v == NULL ? resolution : atoi(v);
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
        } else if (strcmp(argv[i], "--steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            steps = v == NULL ? steps : atoi(v);
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            const char * v = arg_value(argc, argv, &i);
            rescale_t = v == NULL ? rescale_t : (float) atof(v);
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
        } else if (strcmp(argv[i], "--source") == 0) {
            source = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decode-initial") == 0) {
            decode_initial = 1;
        } else if (strcmp(argv[i], "--no-initial") == 0) {
            decode_initial = 0;
        } else if (strcmp(argv[i], "--no-final") == 0) {
            decode_final = 0;
        } else if (strcmp(argv[i], "--display") == 0) {
            display = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--xauthority") == 0) {
            xauthority = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--width") == 0) {
            const char * v = arg_value(argc, argv, &i);
            width = v == NULL ? width : atoi(v);
        } else if (strcmp(argv[i], "--height") == 0) {
            const char * v = arg_value(argc, argv, &i);
            height = v == NULL ? height : atoi(v);
        } else if (strcmp(argv[i], "--max-faces") == 0) {
            const char * v = arg_value(argc, argv, &i);
            max_faces = v == NULL ? max_faces : atoi(v);
        } else if (strcmp(argv[i], "--mesh-chunk-faces") == 0) {
            const char * v = arg_value(argc, argv, &i);
            mesh_chunk_faces = v == NULL ? mesh_chunk_faces : atoi(v);
        } else if (strcmp(argv[i], "--mesh-upload-mode") == 0) {
            mesh_upload_mode = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--mesh-style") == 0) {
            mesh_style = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--hold") == 0) {
            const char * v = arg_value(argc, argv, &i);
            hold = v == NULL ? hold : (float) atof(v);
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
            decode_max_input_tokens = v == NULL ? decode_max_input_tokens : atoll(v);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (model_dir == NULL || coords_path == NULL || cond_path == NULL || noise_path == NULL ||
        steps <= 0 || resolution <= 0 ||
        (strcmp(source, "pred_x0") != 0 && strcmp(source, "x_t") != 0) ||
        (strcmp(mesh_upload_mode, "gpu_indexed") != 0 && strcmp(mesh_upload_mode, "expanded") != 0 && strcmp(mesh_upload_mode, "indexed") != 0) ||
        (strcmp(mesh_style, "solid") != 0 && strcmp(mesh_style, "wire") != 0 && strcmp(mesh_style, "solid_wire") != 0)) {
        usage(argv[0]);
        return 2;
    }

    configure_display_env(display, xauthority);
    trellis_cuda_context cuda;
    memset(&cuda, 0, sizeof(cuda));
    trellis_status status = trellis_cuda_init(&cuda, 0);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "CUDA init failed: %s\n", trellis_status_string(status));
        return 1;
    }

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
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    float slat_mean[32];
    float slat_std[32];
    trellis_tensor_store flow_store;
    trellis_tensor_store decoder_store;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_dit_flow_weights flow;
    trellis_shape_decoder_weights decoder;
    memset(&flow, 0, sizeof(flow));
    memset(&decoder, 0, sizeof(decoder));

    int64_t n_coords = 0;
    size_t cond_values = 0;
    if (!read_coords_i32_alloc(coords_path, coords_force_xyz, &coords, &n_coords)) goto cleanup;
    if (!read_f32_file_alloc(cond_path, &cond, &cond_values)) goto cleanup;
    if ((cond_values % 1024u) != 0) {
        fprintf(stderr, "shape live: cond count %zu is not divisible by 1024\n", cond_values);
        goto cleanup;
    }
    const int cond_tokens = (int) (cond_values / 1024u);

    char flow_path[4096];
    if (flow_override_path != NULL) {
        snprintf(flow_path, sizeof(flow_path), "%s", flow_override_path);
    } else {
        const char * rel = resolution == 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
        status = trellis_make_model_path(model_dir, rel, flow_path, sizeof(flow_path));
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    char decoder_path[4096];
    if (decoder_override_path != NULL) {
        snprintf(decoder_path, sizeof(decoder_path), "%s", decoder_override_path);
    } else {
        status = trellis_make_model_path(model_dir, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", decoder_path, sizeof(decoder_path));
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    if (!load_shape_flow(&cuda, flow_path, &flow_store, &flow)) goto cleanup;
    if (!load_shape_decoder(&cuda, decoder_path, &decoder_store, &decoder)) goto cleanup;
    if (!load_shape_slat_normalization(model_dir, slat_mean, slat_std)) goto cleanup;
    if (flow.in_channels != 32 || flow.out_channels != 32) {
        fprintf(stderr, "shape live: expected shape flow channels 32->32\n");
        goto cleanup;
    }
    if (flow_blocks_override >= 0) {
        if (flow_blocks_override > flow.n_blocks) {
            fprintf(stderr, "shape live: --flow-blocks %d exceeds checkpoint blocks %d\n", flow_blocks_override, flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = flow_blocks_override;
    }
    if (flow_block_parts_override >= 0) {
        flow.debug_block_parts = flow_block_parts_override;
    }
    if (flow_no_rope) {
        flow.debug_disable_rope = 1;
    }
    if (emulate_bf16_blocks) {
        flow.emulate_bf16_blocks = 1;
    }
    trellis_ggml_set_flash_attn_enabled(use_ggml_flash_attn);

    const size_t latent_count = (size_t) n_coords * (size_t) flow.in_channels;
    const size_t context_count = (size_t) cond_tokens * (size_t) flow.cond_channels;
    const size_t phase_count = (size_t) n_coords * (size_t) (flow.head_dim / 2);
    if (cond_values != context_count) {
        fprintf(stderr, "shape live: unexpected cond size\n");
        goto cleanup;
    }
    neg_cond = (float *) calloc(context_count, sizeof(float));
    latent = (float *) malloc(latent_count * sizeof(float));
    pred_pos = (float *) malloc(latent_count * sizeof(float));
    pred_neg = (float *) malloc(latent_count * sizeof(float));
    pred = (float *) malloc(latent_count * sizeof(float));
    next = (float *) malloc(latent_count * sizeof(float));
    x0 = (float *) malloc(latent_count * sizeof(float));
    pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (neg_cond == NULL || latent == NULL || pred_pos == NULL || pred_neg == NULL ||
        pred == NULL || next == NULL || x0 == NULL || pairs == NULL || cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "shape live: host allocation failed\n");
        goto cleanup;
    }
    if (neg_cond_path != NULL && !read_f32_file_exact(neg_cond_path, neg_cond, context_count)) goto cleanup;
    if (!read_f32_file_exact(noise_path, latent, latent_count)) goto cleanup;
    status = trellis_flow_timestep_pairs_f32(steps, rescale_t, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_sparse_phases_f32(coords, n_coords, flow.head_dim, 1.0f, 10000.0f, cos_phase, sin_phase, phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "shape live: schedule/rope failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "TRELLIS.2 stage2 live mesh denoise viewer");
    SetTargetFPS(60);
    Material material = LoadMaterialDefault();
    material.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    mesh_shader shader;
    memset(&shader, 0, sizeof(shader));
    loaded_mesh loaded;
    memset(&loaded, 0, sizeof(loaded));
    if (strcmp(mesh_upload_mode, "gpu_indexed") == 0 && !load_mesh_shader(&shader)) {
        fprintf(stderr, "shape live: failed to load gpu indexed mesh shader\n");
        goto viewer_done;
    }
    float yaw = 0.75f;
    float pitch = 0.35f;
    float distance = 3.2f;
    char title[256];
    char detail[512];

    snprintf(title, sizeof(title), "shape_512 loading");
    snprintf(detail, sizeof(detail), "tokens=%lld cond_tokens=%d; compiling/loading CUDA graphs", (long long) n_coords, cond_tokens);
    hold_frame(&loaded, material, &shader, mesh_style, title, detail, 0.1f, &yaw, &pitch, &distance);

    if (decode_initial && !WindowShouldClose()) {
        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (decode_upload_step(&cuda, &decoder, coords, n_coords, latent, slat_mean, slat_std, resolution,
                decode_max_levels, decode_max_input_tokens, max_faces, mesh_chunk_faces, mesh_upload_mode,
                &loaded, &raw_v, &raw_f, &mesh_timing)) {
            snprintf(title, sizeof(title), "shape_512 x_t step 0/%d", steps);
            snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn in %d chunks; %lld vertices; initial noise",
                loaded.drawn_faces < raw_f ? "sampled" : "all",
                loaded.drawn_faces, (long long) raw_f, loaded.count, (long long) raw_v);
            printf("shape live: step 0/%d source=x_t mesh=%lld vertices %lld faces drawn=%d/%lld decode=%.3fs fdg=%.3fs upload=%.3fs\n",
                steps, (long long) raw_v, (long long) raw_f,
                loaded.drawn_faces, (long long) raw_f,
                mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
            if (!hold_frame(&loaded, material, &shader, mesh_style, title, detail, hold, &yaw, &pitch, &distance)) goto viewer_done;
        }
    }

    for (int step = 0; step < steps && !WindowShouldClose(); ++step) {
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        snprintf(title, sizeof(title), "shape_512 denoising step %d/%d", step + 1, steps);
        snprintf(detail, sizeof(detail), "running flow t=%.6g -> %.6g; previous mesh remains visible", t, t_prev);
        draw_scene(&loaded, material, &shader, mesh_style, title, detail, &yaw, &pitch, &distance);

        double flow_t0 = GetTime();
        status = run_flow_once_tokens_host(&cuda, &flow, latent, cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_pos);
        double flow_t1 = GetTime();
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: conditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto viewer_done;
        }
        status = run_flow_once_tokens_host(&cuda, &flow, latent, neg_cond, cos_phase, sin_phase, t, n_coords, cond_tokens, pred_neg);
        double flow_t2 = GetTime();
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "shape live: unconditional step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto viewer_done;
        }
        if (t >= guidance_min && t <= guidance_max) {
            cfg_rescale_combine_population_f32(latent, pred_pos, pred_neg, latent_count, 1e-5f, t, guidance_strength, guidance_rescale, pred);
        } else {
            memcpy(pred, pred_pos, latent_count * sizeof(float));
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);
        double flow_t3 = GetTime();
        const float * view_latent = strcmp(source, "x_t") == 0 ? next : x0;

        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (!decode_upload_step(&cuda, &decoder, coords, n_coords, view_latent, slat_mean, slat_std, resolution,
                decode_max_levels, decode_max_input_tokens, max_faces, mesh_chunk_faces, mesh_upload_mode,
                &loaded, &raw_v, &raw_f, &mesh_timing)) {
            fprintf(stderr, "shape live: decode/upload step %d failed\n", step + 1);
            goto viewer_done;
        }
        memcpy(latent, next, latent_count * sizeof(float));
        snprintf(title, sizeof(title), "shape_512 %s step %d/%d", source, step + 1, steps);
        snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn in %d chunks; %lld vertices; t=%.6g -> %.6g",
            loaded.drawn_faces < raw_f ? "sampled" : "all",
            loaded.drawn_faces, (long long) raw_f, loaded.count, (long long) raw_v, t, t_prev);
        printf("shape live: step %d/%d source=%s mesh=%lld vertices %lld faces drawn=%d/%lld%s flow+=%.3fs flow-=%.3fs cfg=%.3fs decode=%.3fs fdg=%.3fs upload=%.3fs\n",
            step + 1, steps, source, (long long) raw_v, (long long) raw_f,
            loaded.drawn_faces, (long long) raw_f, loaded.drawn_faces < raw_f ? " sampled" : "",
            flow_t1 - flow_t0, flow_t2 - flow_t1, flow_t3 - flow_t2,
            mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
        if (!hold_frame(&loaded, material, &shader, mesh_style, title, detail, hold, &yaw, &pitch, &distance)) {
            goto viewer_done;
        }
    }

    if (decode_final && !WindowShouldClose()) {
        int64_t raw_v = 0, raw_f = 0;
        mesh_step_timing mesh_timing;
        if (decode_upload_step(&cuda, &decoder, coords, n_coords, latent, slat_mean, slat_std, resolution,
                decode_max_levels, decode_max_input_tokens, max_faces, mesh_chunk_faces, mesh_upload_mode,
                &loaded, &raw_v, &raw_f, &mesh_timing)) {
            snprintf(title, sizeof(title), "shape_512 final_x_t step %d/%d", steps, steps);
            snprintf(detail, sizeof(detail), "%s: %d/%lld faces drawn in %d chunks; %lld vertices; final decode",
                loaded.drawn_faces < raw_f ? "sampled" : "all",
                loaded.drawn_faces, (long long) raw_f, loaded.count, (long long) raw_v);
            printf("shape live: final step %d/%d source=x_t mesh=%lld vertices %lld faces drawn=%d/%lld decode=%.3fs fdg=%.3fs upload=%.3fs\n",
                steps, steps, (long long) raw_v, (long long) raw_f,
                loaded.drawn_faces, (long long) raw_f,
                mesh_timing.decoder_s, mesh_timing.fdg_s, mesh_timing.upload_s);
            while (!WindowShouldClose()) {
                draw_scene(&loaded, material, &shader, mesh_style, title, detail, &yaw, &pitch, &distance);
            }
        }
    }
    rc = 0;

viewer_done:
    unload_loaded_mesh(&loaded);
    unload_mesh_shader(&shader);
    UnloadMaterial(material);
    CloseWindow();

cleanup:
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
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    trellis_tensor_store_free(&flow_store);
    trellis_tensor_store_free(&decoder_store);
    trellis_cuda_free(&cuda);
    return rc;
}
#endif
