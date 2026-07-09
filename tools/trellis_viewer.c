#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_pipeline_internal.h"
#include "trellis_sparse_backend.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

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
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef _WIN32
typedef CRITICAL_SECTION pthread_mutex_t;
typedef HANDLE pthread_t;

typedef struct viewer_thread_start {
    void * (*fn)(void *);
    void * arg;
} viewer_thread_start;

static DWORD WINAPI viewer_thread_trampoline(LPVOID user_data) {
    viewer_thread_start * start = (viewer_thread_start *) user_data;
    if (start != NULL && start->fn != NULL) {
        start->fn(start->arg);
    }
    free(start);
    return 0;
}

static int pthread_mutex_init(pthread_mutex_t * mutex, void * attr) {
    (void) attr;
    InitializeCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_lock(pthread_mutex_t * mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_unlock(pthread_mutex_t * mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_destroy(pthread_mutex_t * mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

static int pthread_create(pthread_t * thread, void * attr, void * (*fn)(void *), void * arg) {
    (void) attr;
    if (thread == NULL || fn == NULL) {
        return EINVAL;
    }
    viewer_thread_start * start = (viewer_thread_start *) calloc(1, sizeof(*start));
    if (start == NULL) {
        return ENOMEM;
    }
    start->fn = fn;
    start->arg = arg;
    HANDLE handle = CreateThread(NULL, 0, viewer_thread_trampoline, start, 0, NULL);
    if (handle == NULL) {
        free(start);
        return (int) GetLastError();
    }
    *thread = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void ** retval) {
    (void) retval;
    if (thread == NULL) {
        return EINVAL;
    }
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

static int pthread_detach(pthread_t thread) {
    if (thread == NULL) {
        return EINVAL;
    }
    CloseHandle(thread);
    return 0;
}
#endif

extern unsigned char * stbi_load(char const * filename, int * x, int * y, int * comp, int req_comp);
extern unsigned char * stbi_load_from_memory(unsigned char const * buffer, int len, int * x, int * y, int * comp, int req_comp);
extern void stbi_image_free(void * retval_from_stbi_load);
extern int stbi_write_png(char const * filename, int w, int h, int comp, const void * data, int stride_in_bytes);

#ifdef _WIN32
int trellis_win_open_image_file_dialog(char * out, size_t out_size);
#endif

#define VIEWER_MAX_PATH 4096
#define VIEWER_MESH_CHUNK_FACES 120000
#define VIEWER_MAX_PREVIEW_VOXELS 18000
#define VIEWER_PREVIEW_AABB_SIZE 2.15f
#define VIEWER_APP_TITLE "trellis2 local"

typedef enum viewer_job_type {
    VIEWER_JOB_NONE = 0,
    VIEWER_JOB_REMOVE_BG,
    VIEWER_JOB_MESH,
    VIEWER_JOB_TEXTURE,
    VIEWER_JOB_POSTPROCESS,
} viewer_job_type;

typedef enum viewer_picker_kind {
    VIEWER_PICKER_IMAGE = 0,
} viewer_picker_kind;

typedef enum viewer_snapshot_kind {
    VIEWER_SNAPSHOT_NONE = 0,
    VIEWER_SNAPSHOT_VOXELS,
    VIEWER_SNAPSHOT_MESH,
} viewer_snapshot_kind;

typedef struct viewer_options {
    char weights_dir[VIEWER_MAX_PATH];
    char model_dir[VIEWER_MAX_PATH];
    char dino_dir[VIEWER_MAX_PATH];
    char birefnet_path[VIEWER_MAX_PATH];
    char image_path[VIEWER_MAX_PATH];
    char out_dir[VIEWER_MAX_PATH];
    char gltf_path[VIEWER_MAX_PATH];
    char flow_override_path[VIEWER_MAX_PATH];
    char decoder_override_path[VIEWER_MAX_PATH];
    char backend[32];
    char pipeline_type[32];
    int device;
    int sparse_structure_steps;
    int structured_latent_steps;
    int latent_size;
    int resolution;
    int cond_resolution;
    int sparse_resolution;
    uint32_t seed;
    uint32_t noise_seed;
    float rescale_t;
    float guidance_strength;
    float guidance_rescale;
    float guidance_min;
    float guidance_max;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    int emulate_bf16_blocks;
    int use_ggml_flash_attn;
    int no_ggml_flash_attn;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    int texture_size;
    int mesh_postprocess_no_simplify;
    int mesh_postprocess_decimation_target;
    int model_cache;
    int model_cache_budget_mib;
    int width;
    int height;
} viewer_options;

typedef struct viewer_snapshot {
    viewer_snapshot_kind kind;
    int version;
    char label[96];

    int32_t * voxel_xyz;
    int64_t voxel_count;
    int voxel_resolution;

    float * mesh_vertices;
    float * mesh_normals;
    unsigned char * mesh_colors;
    int mesh_vertex_count;
    int mesh_triangle_count;
    int64_t source_vertices;
    int64_t source_faces;
} viewer_snapshot;

typedef struct viewer_artifacts {
    trellis_sparse_structure_result sparse;
    trellis_structured_latent shape_latent;
    trellis_sparse_c2s_guides shape_subs;
    trellis_mesh_host mesh;
    trellis_mesh_host projection_mesh;
    int has_sparse;
    int has_shape;
    int has_mesh;
    int has_postprocess;
    int pipeline_resolution;
} viewer_artifacts;

typedef struct viewer_shared {
    pthread_mutex_t mutex;
    int worker_running;
    int worker_done_token;
    int file_picker_running;
    int selected_image_pending;
    int close_requested;
    int bg_ready;
    int mesh_ready;
    int texture_ready;
    int postprocess_ready;
    char current_image_path[VIEWER_MAX_PATH];
    char processed_image_path[VIEWER_MAX_PATH];
    char selected_image_path[VIEWER_MAX_PATH];
    char stage[128];
    char detail[384];
    char error[512];
    float progress;
    int indeterminate;
    char progress_label[128];
    char progress_detail[384];
    int progress_step;
    int progress_steps;
    int64_t progress_step_us;
    viewer_snapshot snapshot;
    viewer_artifacts artifacts;
    char uv_texture_path[VIEWER_MAX_PATH];
    int uv_version;
    char gltf_output_path[VIEWER_MAX_PATH];
} viewer_shared;

typedef struct viewer_worker_args {
    viewer_shared * shared;
    viewer_options options;
    viewer_job_type job;
    char image_path[VIEWER_MAX_PATH];
} viewer_worker_args;

typedef struct viewer_file_picker_args {
    viewer_shared * shared;
    viewer_picker_kind kind;
} viewer_file_picker_args;

typedef struct display_mesh {
    float * vertices;
    float * normals;
    unsigned char * colors;
    int vertex_count;
    int triangle_count;
    int version;
    int lighting;
    Model * models;
    int model_count;
    int64_t source_vertices;
    int64_t source_faces;
} display_mesh;

typedef struct viewer_env_override {
    const char * name;
    char old_value[256];
    int had_old;
} viewer_env_override;

typedef struct display_voxels {
    int32_t * xyz;
    int64_t count;
    int resolution;
    int version;
} display_voxels;

typedef struct viewer_ui_state {
    float yaw;
    float pitch;
    float distance;
    int wireframe;
    int lighting;
    int grid;
    int settings_open;
    float panel_scroll;
} viewer_ui_state;

static void copy_text(char * dst, size_t dst_size, const char * src) {
    if (dst == NULL || dst_size == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= dst_size) {
        n = dst_size - 1u;
    }
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static int text_is_empty(const char * text) {
    return text == NULL || text[0] == '\0';
}

#ifdef _WIN32
static int viewer_utf8_to_wide(const char * text, WCHAR * out, size_t out_count) {
    if (text_is_empty(text) || out == NULL || out_count == 0 || out_count > INT_MAX) {
        return 0;
    }
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, out, (int) out_count);
    if (n > 0) {
        return 1;
    }
    n = MultiByteToWideChar(CP_ACP, 0, text, -1, out, (int) out_count);
    return n > 0;
}

static DWORD viewer_get_file_attrs(const char * path) {
    WCHAR wide[VIEWER_MAX_PATH];
    if (!viewer_utf8_to_wide(path, wide, sizeof(wide) / sizeof(wide[0]))) {
        return INVALID_FILE_ATTRIBUTES;
    }
    return GetFileAttributesW(wide);
}
#endif

static int viewer_path_exists(const char * path) {
#ifdef _WIN32
    DWORD attrs = viewer_get_file_attrs(path);
    return attrs != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return path != NULL && path[0] != '\0' && stat(path, &st) == 0;
#endif
}

static int viewer_dir_exists(const char * path) {
#ifdef _WIN32
    DWORD attrs = viewer_get_file_attrs(path);
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st;
    return path != NULL && path[0] != '\0' && stat(path, &st) == 0 && (st.st_mode & S_IFDIR) != 0;
#endif
}

static int viewer_file_exists(const char * path) {
#ifdef _WIN32
    DWORD attrs = viewer_get_file_attrs(path);
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
#else
    struct stat st;
    return path != NULL && path[0] != '\0' && stat(path, &st) == 0 && (st.st_mode & S_IFREG) != 0;
#endif
}

static int viewer_join_path(const char * base, const char * child, char * dst, size_t dst_size) {
    if (text_is_empty(base) || text_is_empty(child) || dst == NULL || dst_size == 0) {
        return 0;
    }
    size_t base_len = strlen(base);
    int needs_sep = base_len > 0 && !trellis_path_is_sep(base[base_len - 1u]);
    int n = snprintf(
        dst,
        dst_size,
        "%s%s%s",
        base,
        needs_sep ? (TRELLIS_PATH_SEP == '\\' ? "\\" : "/") : "",
        child);
    return n >= 0 && (size_t) n < dst_size;
}

static int viewer_join_path2(const char * base, const char * mid, const char * child, char * dst, size_t dst_size) {
    char tmp[VIEWER_MAX_PATH];
    return viewer_join_path(base, mid, tmp, sizeof(tmp)) && viewer_join_path(tmp, child, dst, dst_size);
}

static int viewer_parent_path(const char * path, char * dst, size_t dst_size) {
    if (text_is_empty(path) || dst == NULL || dst_size == 0) {
        return 0;
    }
    char tmp[VIEWER_MAX_PATH];
    copy_text(tmp, sizeof(tmp), path);
    size_t len = strlen(tmp);
    while (len > 1 && trellis_path_is_sep(tmp[len - 1u])) {
        tmp[--len] = '\0';
    }
    char * slash = trellis_path_last_sep(tmp);
    if (slash == NULL) {
        return 0;
    }
#ifdef _WIN32
    if (slash == tmp + 2 && tmp[1] == ':') {
        slash[1] = '\0';
    } else
#endif
    if (slash == tmp) {
        slash[1] = '\0';
    } else {
        *slash = '\0';
    }
    copy_text(dst, dst_size, tmp);
    return dst[0] != '\0';
}

typedef struct viewer_weight_scan {
    char root_dir[VIEWER_MAX_PATH];
    char model_dir[VIEWER_MAX_PATH];
    char dino_dir[VIEWER_MAX_PATH];
    char birefnet_path[VIEWER_MAX_PATH];
    int has_model;
    int has_dino;
    int has_birefnet;
    int visited_dirs;
} viewer_weight_scan;

static int viewer_path_basename_equals(const char * path, const char * name) {
    if (path == NULL || name == NULL) {
        return 0;
    }
    const char * slash = trellis_path_last_sep_const(path);
    const char * base = slash != NULL ? slash + 1 : path;
#ifdef _WIN32
    return _stricmp(base, name) == 0;
#else
    return strcmp(base, name) == 0;
#endif
}

static int path_has_ext_ci(const char * path, const char * ext);

static int viewer_read_file_bytes(const char * path, unsigned char ** data_out, int * size_out) {
    if (data_out != NULL) *data_out = NULL;
    if (size_out != NULL) *size_out = 0;
    if (text_is_empty(path) || data_out == NULL || size_out == NULL) {
        return 0;
    }
#ifdef _WIN32
    WCHAR wide[VIEWER_MAX_PATH];
    if (!viewer_utf8_to_wide(path, wide, sizeof(wide) / sizeof(wide[0]))) {
        return 0;
    }
    HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        return 0;
    }
    LARGE_INTEGER size_li;
    if (!GetFileSizeEx(file, &size_li) || size_li.QuadPart <= 0 || size_li.QuadPart > INT_MAX) {
        CloseHandle(file);
        return 0;
    }
    int size = (int) size_li.QuadPart;
    unsigned char * data = (unsigned char *) malloc((size_t) size);
    if (data == NULL) {
        CloseHandle(file);
        return 0;
    }
    int offset = 0;
    while (offset < size) {
        DWORD chunk = (DWORD) (size - offset);
        DWORD got = 0;
        if (!ReadFile(file, data + offset, chunk, &got, NULL) || got == 0) {
            free(data);
            CloseHandle(file);
            return 0;
        }
        offset += (int) got;
    }
    CloseHandle(file);
    *data_out = data;
    *size_out = size;
    return 1;
#else
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long size_long = ftell(f);
    if (size_long <= 0 || size_long > INT_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    int size = (int) size_long;
    unsigned char * data = (unsigned char *) malloc((size_t) size);
    if (data == NULL) {
        fclose(f);
        return 0;
    }
    size_t got = fread(data, 1, (size_t) size, f);
    fclose(f);
    if (got != (size_t) size) {
        free(data);
        return 0;
    }
    *data_out = data;
    *size_out = size;
    return 1;
#endif
}

static int viewer_file_contains_text(const char * path, const char * needle) {
    if (text_is_empty(path) || text_is_empty(needle)) {
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1u, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

static int viewer_trellis_model_dir_valid(const char * dir) {
    char path[VIEWER_MAX_PATH];
    if (!viewer_join_path(dir, "pipeline.json", path, sizeof(path)) ||
        !viewer_file_exists(path) ||
        !viewer_file_contains_text(path, "Trellis2ImageTo3DPipeline") ||
        !viewer_file_contains_text(path, "sparse_structure_flow_model")) {
        return 0;
    }
    if (!viewer_join_path2(dir, "ckpts", "ss_flow_img_dit_1_3B_64_bf16.safetensors", path, sizeof(path)) ||
        !viewer_file_exists(path)) {
        return 0;
    }
    if (!viewer_join_path2(dir, "ckpts", "ss_dec_conv3d_16l8_fp16.safetensors", path, sizeof(path)) ||
        !viewer_file_exists(path)) {
        return 0;
    }
    return 1;
}

static int viewer_dino_dir_valid(const char * dir) {
    char path[VIEWER_MAX_PATH];
    if (!viewer_join_path(dir, "config.json", path, sizeof(path)) ||
        !viewer_file_exists(path) ||
        !viewer_file_contains_text(path, "\"model_type\"") ||
        !viewer_file_contains_text(path, "dinov3_vit")) {
        return 0;
    }
    if (!viewer_join_path(dir, "model.safetensors", path, sizeof(path)) ||
        !viewer_file_exists(path)) {
        return 0;
    }
    return 1;
}

static void viewer_weight_scan_visit_dir(viewer_weight_scan * scan, const char * dir) {
    if (scan == NULL || text_is_empty(dir)) {
        return;
    }
    if (!scan->has_model && viewer_trellis_model_dir_valid(dir)) {
        copy_text(scan->model_dir, sizeof(scan->model_dir), dir);
        scan->has_model = 1;
    }
    if (!scan->has_dino && viewer_dino_dir_valid(dir)) {
        copy_text(scan->dino_dir, sizeof(scan->dino_dir), dir);
        scan->has_dino = 1;
    }
}

static void viewer_weight_scan_visit_file(viewer_weight_scan * scan, const char * path) {
    if (scan == NULL || text_is_empty(path) || scan->has_birefnet) {
        return;
    }
    if (path_has_ext_ci(path, ".gguf")) {
        copy_text(scan->birefnet_path, sizeof(scan->birefnet_path), path);
        scan->has_birefnet = 1;
    }
}

static int viewer_weight_scan_done(const viewer_weight_scan * scan) {
    return scan != NULL && scan->has_model && scan->has_dino && scan->has_birefnet;
}

static void viewer_scan_weights_recursive(viewer_weight_scan * scan, const char * dir, int depth) {
    if (scan == NULL || text_is_empty(dir) || depth > 8 || scan->visited_dirs > 4096) {
        return;
    }
    if (viewer_path_basename_equals(dir, ".cache") || viewer_path_basename_equals(dir, ".git")) {
        return;
    }
    scan->visited_dirs += 1;
    int had_model = scan->has_model;
    int had_dino = scan->has_dino;
    viewer_weight_scan_visit_dir(scan, dir);
    if (viewer_weight_scan_done(scan)) {
        return;
    }
    if ((!had_model && scan->has_model) || (!had_dino && scan->has_dino)) {
        return;
    }

#ifdef _WIN32
    char pattern[VIEWER_MAX_PATH];
    if (!viewer_join_path(dir, "*", pattern, sizeof(pattern))) {
        return;
    }
    WIN32_FIND_DATAA data;
    HANDLE h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0) {
            continue;
        }
        char child[VIEWER_MAX_PATH];
        if (!viewer_join_path(dir, data.cFileName, child, sizeof(child))) {
            continue;
        }
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            viewer_scan_weights_recursive(scan, child, depth + 1);
        } else {
            viewer_weight_scan_visit_file(scan, child);
        }
        if (viewer_weight_scan_done(scan)) {
            break;
        }
    } while (FindNextFileA(h, &data));
    FindClose(h);
#else
    DIR * d = opendir(dir);
    if (d == NULL) {
        return;
    }
    struct dirent * ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        char child[VIEWER_MAX_PATH];
        if (!viewer_join_path(dir, ent->d_name, child, sizeof(child))) {
            continue;
        }
        if (viewer_dir_exists(child)) {
            viewer_scan_weights_recursive(scan, child, depth + 1);
        } else if (viewer_file_exists(child)) {
            viewer_weight_scan_visit_file(scan, child);
        }
        if (viewer_weight_scan_done(scan)) {
            break;
        }
    }
    closedir(d);
#endif
}

static int viewer_scan_weights_pool(const char * selected, viewer_weight_scan * out) {
    if (text_is_empty(selected) || out == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (!viewer_dir_exists(selected)) {
        return 0;
    }

    char current[VIEWER_MAX_PATH];
    copy_text(current, sizeof(current), selected);
    int found_component = 0;
    for (int depth = 0; depth < 6 && !text_is_empty(current); ++depth) {
        if (!viewer_dir_exists(current)) {
            break;
        }
        viewer_weight_scan scan;
        memset(&scan, 0, sizeof(scan));
        copy_text(scan.root_dir, sizeof(scan.root_dir), current);
        viewer_scan_weights_recursive(&scan, current, 0);
        if (scan.has_model && scan.has_dino) {
            *out = scan;
            return 1;
        }
        if (scan.has_model || scan.has_dino || scan.has_birefnet) {
            found_component = 1;
        } else if (!found_component) {
            break;
        }
        char parent[VIEWER_MAX_PATH];
        if (!viewer_parent_path(current, parent, sizeof(parent)) || strcmp(parent, current) == 0) {
            break;
        }
        copy_text(current, sizeof(current), parent);
    }
    return 0;
}

static int viewer_apply_standard_weights_layout(viewer_options * options, const char * dir) {
    if (options == NULL || text_is_empty(dir)) {
        return 0;
    }
    char model_dir[VIEWER_MAX_PATH];
    char dino_dir[VIEWER_MAX_PATH];
    char birefnet_path[VIEWER_MAX_PATH];
    if (!viewer_join_path(dir, "TRELLIS.2-4B", model_dir, sizeof(model_dir)) ||
        !viewer_join_path(dir, "dinov3-vitl16-pretrain-lvd1689m", dino_dir, sizeof(dino_dir)) ||
        !viewer_trellis_model_dir_valid(model_dir) ||
        !viewer_dino_dir_valid(dino_dir)) {
        return 0;
    }
    copy_text(options->weights_dir, sizeof(options->weights_dir), dir);
    copy_text(options->model_dir, sizeof(options->model_dir), model_dir);
    copy_text(options->dino_dir, sizeof(options->dino_dir), dino_dir);
    if (viewer_join_path2(dir, "BiRefNet", "BiRefNet-F16.gguf", birefnet_path, sizeof(birefnet_path)) &&
        viewer_file_exists(birefnet_path)) {
        copy_text(options->birefnet_path, sizeof(options->birefnet_path), birefnet_path);
    } else {
        options->birefnet_path[0] = '\0';
    }
    return 1;
}

static int viewer_find_weights_root(const char * selected, char * root, size_t root_size) {
    if (text_is_empty(selected) || root == NULL || root_size == 0) {
        return 0;
    }
    viewer_weight_scan scan;
    if (viewer_scan_weights_pool(selected, &scan)) {
        copy_text(root, root_size, scan.root_dir);
        return 1;
    }
    return 0;
}

static void viewer_apply_weights_dir(viewer_options * options, const char * dir) {
    if (options == NULL || text_is_empty(dir)) {
        return;
    }
    if (viewer_apply_standard_weights_layout(options, dir)) {
        return;
    }
    viewer_weight_scan scan;
    if (viewer_scan_weights_pool(dir, &scan)) {
        copy_text(options->weights_dir, sizeof(options->weights_dir), scan.root_dir);
        copy_text(options->model_dir, sizeof(options->model_dir), scan.model_dir);
        copy_text(options->dino_dir, sizeof(options->dino_dir), scan.dino_dir);
        copy_text(options->birefnet_path, sizeof(options->birefnet_path), scan.birefnet_path);
        return;
    }

    copy_text(options->weights_dir, sizeof(options->weights_dir), dir);
    options->model_dir[0] = '\0';
    options->dino_dir[0] = '\0';
    options->birefnet_path[0] = '\0';
}

static int viewer_weights_root_looks_valid(const char * dir) {
    viewer_options options;
    memset(&options, 0, sizeof(options));
    if (viewer_apply_standard_weights_layout(&options, dir)) {
        return 1;
    }
    viewer_weight_scan scan;
    return viewer_scan_weights_pool(dir, &scan);
}

static int viewer_weights_ready(const viewer_options * options, int require_birefnet, char * issue, size_t issue_size) {
    char path[VIEWER_MAX_PATH];
    if (issue != NULL && issue_size > 0) {
        issue[0] = '\0';
    }
    if (options == NULL || text_is_empty(options->model_dir) || text_is_empty(options->dino_dir)) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "warning: place TRELLIS.2 near the app or launch with --weights DIR");
        return 0;
    }
    if (!viewer_dir_exists(options->model_dir)) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "missing TRELLIS model folder");
        return 0;
    }
    if (!viewer_join_path(options->model_dir, "pipeline.json", path, sizeof(path)) || !viewer_file_exists(path)) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "missing TRELLIS pipeline manifest");
        return 0;
    }
    if (!viewer_join_path2(options->model_dir, "ckpts", "ss_flow_img_dit_1_3B_64_bf16.safetensors", path, sizeof(path)) ||
        !viewer_file_exists(path)) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "missing TRELLIS sparse-structure weights");
        return 0;
    }
    if (!viewer_join_path(options->dino_dir, "model.safetensors", path, sizeof(path)) || !viewer_file_exists(path)) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "missing DINOv3 image encoder");
        return 0;
    }
    if (require_birefnet && (text_is_empty(options->birefnet_path) || !viewer_file_exists(options->birefnet_path))) {
        if (issue != NULL && issue_size > 0) snprintf(issue, issue_size, "missing BiRefNet GGUF model");
        return 0;
    }
    return 1;
}

static void viewer_autodetect_weights_dir(viewer_options * options) {
    if (options == NULL || !text_is_empty(options->weights_dir)) {
        return;
    }
    char exe_path[VIEWER_MAX_PATH];
    if (trellis_current_executable_path(exe_path, sizeof(exe_path))) {
        char * slash = trellis_path_last_sep(exe_path);
        if (slash != NULL) {
            slash[1] = '\0';
            char packaged[VIEWER_MAX_PATH];
            if (viewer_join_path(exe_path, "TRELLIS.2", packaged, sizeof(packaged)) &&
                viewer_weights_root_looks_valid(packaged)) {
                viewer_apply_weights_dir(options, packaged);
                return;
            }
        }
    }
    const char * env_candidates[] = {
        getenv("TRELLIS2_WEIGHTS_DIR"),
        getenv("TRELLIS_WEIGHTS_DIR"),
        NULL,
    };
    for (int i = 0; env_candidates[i] != NULL; ++i) {
        if (viewer_weights_root_looks_valid(env_candidates[i])) {
            viewer_apply_weights_dir(options, env_candidates[i]);
            return;
        }
    }
    const char * relative_candidates[] = {
        "../TRELLIS.2",
        "../../TRELLIS.2",
        "../../../TRELLIS.2",
        "TRELLIS.2",
        NULL,
    };
    for (int i = 0; relative_candidates[i] != NULL; ++i) {
        if (viewer_weights_root_looks_valid(relative_candidates[i])) {
            viewer_apply_weights_dir(options, relative_candidates[i]);
            return;
        }
    }
    if (!trellis_current_executable_path(exe_path, sizeof(exe_path))) {
        return;
    }
    char * slash = trellis_path_last_sep(exe_path);
    if (slash == NULL) {
        return;
    }
    slash[1] = '\0';
    const char * app_candidates[] = {
        "TRELLIS.2",
        "../TRELLIS.2",
        "../../TRELLIS.2",
        "../../../TRELLIS.2",
        "../../../../TRELLIS.2",
        NULL,
    };
    for (int i = 0; app_candidates[i] != NULL; ++i) {
        char candidate[VIEWER_MAX_PATH];
        if (viewer_join_path(exe_path, app_candidates[i], candidate, sizeof(candidate)) &&
            viewer_weights_root_looks_valid(candidate)) {
            viewer_apply_weights_dir(options, candidate);
            return;
        }
    }
}

static void viewer_autodetect_image_path(viewer_options * options) {
    if (options == NULL || viewer_file_exists(options->image_path)) {
        return;
    }
    const char * relative_candidates[] = {
        "images.jpg",
        "example_image/images.jpg",
        "../images.jpg",
        "../example_image/images.jpg",
        "../../images.jpg",
        "../../example_image/images.jpg",
        "../../../images.jpg",
        "../../../example_image/images.jpg",
        "example_image/T.png",
        "../example_image/T.png",
        "../../example_image/T.png",
        "../../../example_image/T.png",
        "assets/example_image/T.png",
        "../assets/example_image/T.png",
        "../../assets/example_image/T.png",
        "../../../assets/example_image/T.png",
        NULL,
    };
    for (int i = 0; relative_candidates[i] != NULL; ++i) {
        if (viewer_file_exists(relative_candidates[i])) {
            copy_text(options->image_path, sizeof(options->image_path), relative_candidates[i]);
            return;
        }
    }
    char exe_path[VIEWER_MAX_PATH];
    if (!trellis_current_executable_path(exe_path, sizeof(exe_path))) {
        return;
    }
    char * slash = trellis_path_last_sep(exe_path);
    if (slash == NULL) {
        return;
    }
    slash[1] = '\0';
    for (int i = 0; relative_candidates[i] != NULL; ++i) {
        char candidate[VIEWER_MAX_PATH];
        if (viewer_join_path(exe_path, relative_candidates[i], candidate, sizeof(candidate)) &&
            viewer_file_exists(candidate)) {
            copy_text(options->image_path, sizeof(options->image_path), candidate);
            return;
        }
    }
}

static int path_has_ext_ci(const char * path, const char * ext) {
    if (path == NULL || ext == NULL) {
        return 0;
    }
    size_t path_len = strlen(path);
    size_t ext_len = strlen(ext);
    if (path_len < ext_len) {
        return 0;
    }
    const char * p = path + path_len - ext_len;
    for (size_t i = 0; i < ext_len; ++i) {
        char a = p[i];
        char b = ext[i];
        if (a >= 'A' && a <= 'Z') a = (char) (a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char) (b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int is_supported_image_path(const char * path) {
    return path_has_ext_ci(path, ".png") ||
           path_has_ext_ci(path, ".jpg") ||
           path_has_ext_ci(path, ".jpeg") ||
           path_has_ext_ci(path, ".webp") ||
           path_has_ext_ci(path, ".bmp");
}

static int mkdir_p_simple(const char * path) {
    return !text_is_empty(path) && trellis_mkdir_p(path);
}

static int split_dir_stem(const char * path, char * dir, size_t dir_size, char * stem, size_t stem_size) {
    if (text_is_empty(path) || dir == NULL || stem == NULL) {
        return 0;
    }
    const char * slash = strrchr(path, '/');
    const char * base = slash != NULL ? slash + 1 : path;
    if (slash != NULL) {
        size_t n = (size_t) (slash - path);
        if (n == 0) n = 1;
        if (n >= dir_size) return 0;
        memcpy(dir, path, n);
        dir[n] = '\0';
    } else {
        copy_text(dir, dir_size, ".");
    }
    const char * dot = strrchr(base, '.');
    size_t stem_len = dot != NULL && dot > base ? (size_t) (dot - base) : strlen(base);
    if (stem_len == 0 || stem_len >= stem_size) {
        return 0;
    }
    memcpy(stem, base, stem_len);
    stem[stem_len] = '\0';
    return 1;
}

static int make_default_gltf_path(const viewer_options * options, char * out, size_t out_size) {
    if (!text_is_empty(options->gltf_path)) {
        copy_text(out, out_size, options->gltf_path);
        return 1;
    }
    int n = snprintf(out, out_size, "%s/trellis_viewer_output.gltf", options->out_dir);
    return n >= 0 && (size_t) n < out_size;
}

static int make_base_color_path(const char * gltf_path, char * out, size_t out_size) {
    if (path_has_ext_ci(gltf_path, ".glb")) {
        out[0] = '\0';
        return 0;
    }
    char dir[VIEWER_MAX_PATH];
    char stem[512];
    if (!split_dir_stem(gltf_path, dir, sizeof(dir), stem, sizeof(stem))) {
        return 0;
    }
    const char * suffix = "_baseColor.png";
    size_t dir_len = strlen(dir);
    size_t stem_len = strlen(stem);
    size_t suffix_len = strlen(suffix);
    size_t total = dir_len + 1u + stem_len + suffix_len;
    if (total + 1u > out_size) {
        return 0;
    }
    memcpy(out, dir, dir_len);
    out[dir_len] = '/';
    memcpy(out + dir_len + 1u, stem, stem_len);
    memcpy(out + dir_len + 1u + stem_len, suffix, suffix_len);
    out[total] = '\0';
    return 1;
}

static int parse_int_arg(const char * text, int * out) {
    if (text_is_empty(text) || out == NULL) {
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
    if (text_is_empty(text) || out == NULL) {
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
    if (text_is_empty(text) || out == NULL) {
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
    if (text_is_empty(text) || out == NULL) {
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

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int command_available(const char * name);

static void viewer_options_defaults(viewer_options * options) {
    memset(options, 0, sizeof(*options));
    copy_text(options->image_path, sizeof(options->image_path), "images.jpg");
    copy_text(options->out_dir, sizeof(options->out_dir), "viewer_outputs");
    copy_text(options->backend, sizeof(options->backend), TRELLIS_DEFAULT_BACKEND);
    copy_text(options->pipeline_type, sizeof(options->pipeline_type), "512");
    options->device = 0;
    options->sparse_structure_steps = 12;
    options->structured_latent_steps = 12;
    options->latent_size = 16;
    options->resolution = 512;
    options->cond_resolution = 512;
    options->sparse_resolution = 32;
    options->seed = 1u;
    options->noise_seed = 18u;
    options->rescale_t = 3.0f;
    options->guidance_strength = 7.5f;
    options->guidance_rescale = 0.5f;
    options->guidance_min = 0.6f;
    options->guidance_max = 1.0f;
    options->flow_blocks_override = -1;
    options->flow_block_parts_override = -1;
    options->texture_size = 1024;
    options->mesh_postprocess_decimation_target = 1000000;
    options->model_cache = 1;
    options->use_ggml_flash_attn = 1;
    options->width = 1440;
    options->height = 900;
    viewer_autodetect_image_path(options);
}

static void usage(FILE * out, const char * argv0) {
    fprintf(out,
        "Usage:\n"
        "  %s [--weights DIR] [--image FILE] [options]\n"
        "\n"
        "Local raylib image-to-3D GUI. Uses auto-detected weights, or pass --weights DIR. Choose an image in the window.\n"
        "\n"
        "Core options mirror trellis-image-to-gltf:\n"
        "  --weights DIR            Folder containing downloaded TRELLIS, DINO, and optional BiRefNet weights\n"
        "  --model DIR, --dino DIR, --birefnet FILE, --image FILE\n"
        "  --out-dir DIR, --gltf FILE, --backend NAME, --device N\n"
        "  --pipeline 512|1024       GUI stage mode, default 512\n"
        "  --steps N, --sparse-structure-steps N, --structured-latent-steps N\n"
        "  --texture-size N, --seed N, --noise-seed N\n"
        "  --flow PATH, --decoder PATH\n"
        "  --mesh-decimation-target N, --mesh-postprocess-no-simplify\n"
        "  --no-model-cache, --model-cache-budget-mib N\n"
        "  --use-ggml-flash-attn, --no-ggml-flash-attn\n"
        "  --decode-max-levels N, --decode-max-input-tokens N\n"
        "  --width N, --height N, --verbose\n",
        argv0);
}

static int parse_options(int argc, char ** argv, viewer_options * options) {
    viewer_options_defaults(options);
    int explicit_weight_source = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 || strcmp(argv[i], "--weights-dir") == 0) {
            const char * dir = arg_value(argc, argv, &i);
            if (text_is_empty(dir)) return 0;
            viewer_apply_weights_dir(options, dir);
            explicit_weight_source = 1;
        } else if (strcmp(argv[i], "--model") == 0) {
            copy_text(options->model_dir, sizeof(options->model_dir), arg_value(argc, argv, &i));
            explicit_weight_source = 1;
        } else if (strcmp(argv[i], "--dino") == 0) {
            copy_text(options->dino_dir, sizeof(options->dino_dir), arg_value(argc, argv, &i));
            explicit_weight_source = 1;
        } else if (strcmp(argv[i], "--birefnet") == 0) {
            copy_text(options->birefnet_path, sizeof(options->birefnet_path), arg_value(argc, argv, &i));
            explicit_weight_source = 1;
        } else if (strcmp(argv[i], "--image") == 0 || strcmp(argv[i], "--input") == 0) {
            copy_text(options->image_path, sizeof(options->image_path), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--out-dir") == 0) {
            copy_text(options->out_dir, sizeof(options->out_dir), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--gltf") == 0 || strcmp(argv[i], "--glb") == 0 || strcmp(argv[i], "--output") == 0) {
            copy_text(options->gltf_path, sizeof(options->gltf_path), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--backend") == 0) {
            copy_text(options->backend, sizeof(options->backend), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            copy_text(options->pipeline_type, sizeof(options->pipeline_type), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->device)) return 0;
        } else if (strcmp(argv[i], "--steps") == 0) {
            int steps = 0;
            if (!parse_int_arg(arg_value(argc, argv, &i), &steps)) return 0;
            options->sparse_structure_steps = steps;
            options->structured_latent_steps = steps;
        } else if (strcmp(argv[i], "--sparse-structure-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->sparse_structure_steps)) return 0;
        } else if (strcmp(argv[i], "--structured-latent-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->structured_latent_steps)) return 0;
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->texture_size)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options->seed)) return 0;
        } else if (strcmp(argv[i], "--noise-seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options->noise_seed)) return 0;
        } else if (strcmp(argv[i], "--latent-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->latent_size)) return 0;
        } else if (strcmp(argv[i], "--resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->resolution)) return 0;
        } else if (strcmp(argv[i], "--cond-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->cond_resolution)) return 0;
        } else if (strcmp(argv[i], "--sparse-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->sparse_resolution)) return 0;
        } else if (strcmp(argv[i], "--flow") == 0) {
            copy_text(options->flow_override_path, sizeof(options->flow_override_path), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--decoder") == 0) {
            copy_text(options->decoder_override_path, sizeof(options->decoder_override_path), arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options->rescale_t)) return 0;
        } else if (strcmp(argv[i], "--guidance-strength") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options->guidance_strength)) return 0;
        } else if (strcmp(argv[i], "--guidance-rescale") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options->guidance_rescale)) return 0;
        } else if (strcmp(argv[i], "--guidance-min") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options->guidance_min)) return 0;
        } else if (strcmp(argv[i], "--guidance-max") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options->guidance_max)) return 0;
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->flow_blocks_override)) return 0;
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->flow_block_parts_override)) return 0;
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            options->flow_no_rope = 1;
        } else if (strcmp(argv[i], "--emulate-bf16-blocks") == 0) {
            options->emulate_bf16_blocks = 1;
        } else if (strcmp(argv[i], "--use-ggml-flash-attn") == 0) {
            options->use_ggml_flash_attn = 1;
            options->no_ggml_flash_attn = 0;
        } else if (strcmp(argv[i], "--no-ggml-flash-attn") == 0) {
            options->no_ggml_flash_attn = 1;
            options->use_ggml_flash_attn = 0;
        } else if (strcmp(argv[i], "--decode-max-levels") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->decode_max_levels)) return 0;
        } else if (strcmp(argv[i], "--decode-max-input-tokens") == 0) {
            if (!parse_i64_arg(arg_value(argc, argv, &i), &options->decode_max_input_tokens)) return 0;
        } else if (strcmp(argv[i], "--mesh-postprocess-no-simplify") == 0) {
            options->mesh_postprocess_no_simplify = 1;
        } else if (strcmp(argv[i], "--mesh-decimation-target") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->mesh_postprocess_decimation_target)) return 0;
        } else if (strcmp(argv[i], "--model-cache") == 0) {
            options->model_cache = 1;
        } else if (strcmp(argv[i], "--no-model-cache") == 0) {
            options->model_cache = 0;
        } else if (strcmp(argv[i], "--model-cache-budget-mib") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->model_cache_budget_mib)) return 0;
        } else if (strcmp(argv[i], "--width") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->width)) return 0;
        } else if (strcmp(argv[i], "--height") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options->height)) return 0;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 0;
        }
    }
    if (!explicit_weight_source) {
        viewer_autodetect_weights_dir(options);
    }
    if (strcmp(options->pipeline_type, "512") == 0) {
        options->resolution = 512;
        options->cond_resolution = 512;
        options->sparse_resolution = 32;
    } else if (strcmp(options->pipeline_type, "1024") == 0) {
        options->resolution = 1024;
        options->cond_resolution = 1024;
        options->sparse_resolution = 64;
    } else {
        fprintf(stderr, "trellis-viewer: GUI stage mode currently supports --pipeline 512 or 1024\n");
        return 0;
    }
    return !text_is_empty(options->image_path);
}

static void viewer_snapshot_free(viewer_snapshot * snapshot) {
    if (snapshot == NULL) {
        return;
    }
    free(snapshot->voxel_xyz);
    free(snapshot->mesh_vertices);
    free(snapshot->mesh_normals);
    free(snapshot->mesh_colors);
    int version = snapshot->version;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->version = version;
}

static void viewer_artifacts_free(viewer_artifacts * artifacts) {
    if (artifacts == NULL) {
        return;
    }
    trellis_sparse_structure_result_free(&artifacts->sparse);
    trellis_structured_latent_free(&artifacts->shape_latent);
    trellis_sparse_c2s_guides_free(&artifacts->shape_subs);
    trellis_mesh_free(&artifacts->mesh);
    trellis_mesh_free(&artifacts->projection_mesh);
    memset(artifacts, 0, sizeof(*artifacts));
}

static void shared_reset_step_progress_locked(viewer_shared * shared) {
    shared->progress_label[0] = '\0';
    shared->progress_detail[0] = '\0';
    shared->progress_step = 0;
    shared->progress_steps = 0;
    shared->progress_step_us = 0;
}

static void shared_set_stage(
    viewer_shared * shared,
    const char * stage,
    const char * detail,
    float progress,
    int indeterminate) {
    pthread_mutex_lock(&shared->mutex);
    copy_text(shared->stage, sizeof(shared->stage), stage);
    copy_text(shared->detail, sizeof(shared->detail), detail);
    shared->progress = progress;
    shared->indeterminate = indeterminate;
    shared_reset_step_progress_locked(shared);
    pthread_mutex_unlock(&shared->mutex);
}

static void shared_set_step_plan(
    viewer_shared * shared,
    const char * label,
    int steps,
    const char * detail) {
    pthread_mutex_lock(&shared->mutex);
    copy_text(shared->progress_label, sizeof(shared->progress_label), label);
    copy_text(shared->progress_detail, sizeof(shared->progress_detail), detail);
    shared->progress_step = 0;
    shared->progress_steps = steps > 0 ? steps : 0;
    shared->progress_step_us = 0;
    shared->progress = 0.0f;
    shared->indeterminate = 0;
    pthread_mutex_unlock(&shared->mutex);
}

static void shared_set_error(viewer_shared * shared, const char * stage, const char * message) {
    pthread_mutex_lock(&shared->mutex);
    copy_text(shared->stage, sizeof(shared->stage), stage);
    copy_text(shared->detail, sizeof(shared->detail), "failed");
    copy_text(shared->error, sizeof(shared->error), message);
    shared->progress = 0.0f;
    shared->indeterminate = 0;
    shared_reset_step_progress_locked(shared);
    pthread_mutex_unlock(&shared->mutex);
}

static void shared_clear_error(viewer_shared * shared) {
    pthread_mutex_lock(&shared->mutex);
    shared->error[0] = '\0';
    pthread_mutex_unlock(&shared->mutex);
}

static void viewer_env_override_begin(viewer_env_override * env, const char * name, const char * value) {
    if (env == NULL || name == NULL || value == NULL) {
        return;
    }
    memset(env, 0, sizeof(*env));
    env->name = name;
    const char * old_value = getenv(name);
    env->had_old = old_value != NULL;
    if (old_value != NULL) {
        copy_text(env->old_value, sizeof(env->old_value), old_value);
    }
    trellis_setenv(name, value, 1);
}

static void viewer_env_override_end(const viewer_env_override * env) {
    if (env == NULL || env->name == NULL) {
        return;
    }
    if (env->had_old) {
        trellis_setenv(env->name, env->old_value, 1);
    } else {
        trellis_unsetenv(env->name);
    }
}

static void viewer_progress_step_callback(const trellis_progress_step_event * event, void * user_data) {
    viewer_shared * shared = (viewer_shared *) user_data;
    if (shared == NULL || event == NULL || event->steps <= 0) {
        return;
    }
    pthread_mutex_lock(&shared->mutex);
    copy_text(shared->progress_label, sizeof(shared->progress_label), event->label);
    copy_text(shared->progress_detail, sizeof(shared->progress_detail), event->detail);
    shared->progress_step = event->step;
    shared->progress_steps = event->steps;
    shared->progress_step_us = event->step_us;
    shared->progress = (float) event->step / (float) event->steps;
    if (shared->progress < 0.0f) shared->progress = 0.0f;
    if (shared->progress > 1.0f) shared->progress = 1.0f;
    shared->indeterminate = 0;
    pthread_mutex_unlock(&shared->mutex);
}

static int sampled_index_i64(int64_t i, int64_t count, int max_count) {
    if (max_count <= 0 || count <= (int64_t) max_count) {
        return (int) i;
    }
    if (max_count <= 1) {
        return 0;
    }
    return (int) llround((double) i * (double) (count - 1) / (double) (max_count - 1));
}

static void publish_voxel_snapshot(
    viewer_shared * shared,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    int resolution,
    const char * label) {
    if (coords_bxyz == NULL || n_coords <= 0 || resolution <= 0) {
        return;
    }
    int64_t draw_count = n_coords;
    if (draw_count > VIEWER_MAX_PREVIEW_VOXELS) {
        draw_count = VIEWER_MAX_PREVIEW_VOXELS;
    }
    int32_t * xyz = (int32_t *) malloc((size_t) draw_count * 3u * sizeof(int32_t));
    if (xyz == NULL) {
        return;
    }
    for (int64_t i = 0; i < draw_count; ++i) {
        int idx = sampled_index_i64(i, n_coords, VIEWER_MAX_PREVIEW_VOXELS);
        const int32_t * c = coords_bxyz + (size_t) idx * 4u;
        xyz[(size_t) i * 3u + 0u] = c[1];
        xyz[(size_t) i * 3u + 1u] = c[2];
        xyz[(size_t) i * 3u + 2u] = c[3];
    }

    pthread_mutex_lock(&shared->mutex);
    viewer_snapshot_free(&shared->snapshot);
    shared->snapshot.kind = VIEWER_SNAPSHOT_VOXELS;
    shared->snapshot.version += 1;
    copy_text(shared->snapshot.label, sizeof(shared->snapshot.label), label);
    shared->snapshot.voxel_xyz = xyz;
    shared->snapshot.voxel_count = draw_count;
    shared->snapshot.voxel_resolution = resolution;
    pthread_mutex_unlock(&shared->mutex);
}

static Vector3 vec3_from_floats(const float * v) {
    return (Vector3) { v[0], v[1], v[2] };
}

static float vec3_len_sq(Vector3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

static unsigned char clamp_u8(float v);

static uint32_t viewer_hash_coord(int x, int y, int z) {
    uint32_t h = (uint32_t) x * 73856093u ^ (uint32_t) y * 19349663u ^ (uint32_t) z * 83492791u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static size_t viewer_next_pow2_size(size_t v) {
    size_t p = 1u;
    while (p < v && p <= SIZE_MAX / 2u) {
        p <<= 1u;
    }
    return p >= v ? p : 0u;
}

static int32_t * viewer_build_pbr_hash(const trellis_pbr_voxels * voxels, uint32_t * hash_size_out) {
    if (hash_size_out != NULL) *hash_size_out = 0;
    if (voxels == NULL || voxels->coords_bxyz == NULL || voxels->attrs == NULL || voxels->n_coords <= 0) {
        return NULL;
    }
    size_t table_size = viewer_next_pow2_size((size_t) voxels->n_coords * 4u);
    if (table_size == 0 || table_size > UINT32_MAX) {
        return NULL;
    }
    int32_t * entries = (int32_t *) malloc(table_size * 4u * sizeof(int32_t));
    if (entries == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < table_size; ++i) {
        entries[i * 4u + 0u] = 0;
        entries[i * 4u + 1u] = 0;
        entries[i * 4u + 2u] = 0;
        entries[i * 4u + 3u] = -1;
    }
    for (int64_t i = 0; i < voxels->n_coords; ++i) {
        const int32_t * c = voxels->coords_bxyz + (size_t) i * 4u;
        uint32_t slot = viewer_hash_coord(c[1], c[2], c[3]) & ((uint32_t) table_size - 1u);
        while (entries[(size_t) slot * 4u + 3u] >= 0 &&
               (entries[(size_t) slot * 4u + 0u] != c[1] ||
                entries[(size_t) slot * 4u + 1u] != c[2] ||
                entries[(size_t) slot * 4u + 2u] != c[3])) {
            slot = (slot + 1u) & ((uint32_t) table_size - 1u);
        }
        entries[(size_t) slot * 4u + 0u] = c[1];
        entries[(size_t) slot * 4u + 1u] = c[2];
        entries[(size_t) slot * 4u + 2u] = c[3];
        entries[(size_t) slot * 4u + 3u] = (int32_t) i;
    }
    if (hash_size_out != NULL) {
        *hash_size_out = (uint32_t) table_size;
    }
    return entries;
}

static int viewer_lookup_pbr_hash(const int32_t * entries, uint32_t hash_size, int x, int y, int z) {
    if (entries == NULL || hash_size == 0) {
        return -1;
    }
    uint32_t slot = viewer_hash_coord(x, y, z) & (hash_size - 1u);
    for (uint32_t probe = 0; probe < hash_size; ++probe) {
        const int32_t * e = entries + (size_t) slot * 4u;
        if (e[3] < 0) {
            return -1;
        }
        if (e[0] == x && e[1] == y && e[2] == z) {
            return e[3];
        }
        slot = (slot + 1u) & (hash_size - 1u);
    }
    return -1;
}

static void viewer_sample_pbr_color(
    const trellis_pbr_voxels * voxels,
    const int32_t * hash_entries,
    uint32_t hash_size,
    const float p[3],
    unsigned char out[4]) {
    out[0] = 176;
    out[1] = 190;
    out[2] = 206;
    out[3] = 255;
    if (voxels == NULL || voxels->attrs == NULL || voxels->channels < 3 ||
        hash_entries == NULL || hash_size == 0 || voxels->resolution <= 0) {
        return;
    }
    const int res = voxels->resolution;
    float q[3] = {
        (p[0] + 0.5f) * (float) res,
        (p[1] + 0.5f) * (float) res,
        (p[2] + 0.5f) * (float) res,
    };
    int base[3] = {
        (int) floorf(q[0] - 0.5f),
        (int) floorf(q[1] - 0.5f),
        (int) floorf(q[2] - 0.5f),
    };
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    float sum_w = 0.0f;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int c[3] = {base[0] + dx, base[1] + dy, base[2] + dz};
                if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= res || c[1] >= res || c[2] >= res) {
                    continue;
                }
                float wx = 1.0f - fabsf(q[0] - (float) c[0] - 0.5f);
                float wy = 1.0f - fabsf(q[1] - (float) c[1] - 0.5f);
                float wz = 1.0f - fabsf(q[2] - (float) c[2] - 0.5f);
                float w = wx * wy * wz;
                if (w <= 0.0f) continue;
                int id = viewer_lookup_pbr_hash(hash_entries, hash_size, c[0], c[1], c[2]);
                if (id < 0) continue;
                const float * attr = voxels->attrs + (size_t) id * (size_t) voxels->channels;
                rgb[0] += attr[0] * w;
                rgb[1] += attr[1] * w;
                rgb[2] += attr[2] * w;
                sum_w += w;
            }
        }
    }
    if (sum_w <= 1e-6f) {
        int best = -1;
        for (int radius = 0; radius <= 2 && best < 0; ++radius) {
            for (int dz = -radius; dz <= radius && best < 0; ++dz) {
                for (int dy = -radius; dy <= radius && best < 0; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (radius > 0 && abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) continue;
                        int c[3] = {base[0] + dx, base[1] + dy, base[2] + dz};
                        if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= res || c[1] >= res || c[2] >= res) continue;
                        best = viewer_lookup_pbr_hash(hash_entries, hash_size, c[0], c[1], c[2]);
                        if (best >= 0) break;
                    }
                }
            }
        }
        if (best < 0) {
            return;
        }
        const float * attr = voxels->attrs + (size_t) best * (size_t) voxels->channels;
        rgb[0] = attr[0];
        rgb[1] = attr[1];
        rgb[2] = attr[2];
        sum_w = 1.0f;
    }
    out[0] = clamp_u8((rgb[0] / sum_w) * 255.0f);
    out[1] = clamp_u8((rgb[1] / sum_w) * 255.0f);
    out[2] = clamp_u8((rgb[2] / sum_w) * 255.0f);
}

static void publish_mesh_snapshot(
    viewer_shared * shared,
    const trellis_mesh_host * mesh,
    const trellis_pbr_voxels * pbr_voxels,
    const char * label) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return;
    }

    float minv[3] = { mesh->vertices[0], mesh->vertices[1], mesh->vertices[2] };
    float maxv[3] = { mesh->vertices[0], mesh->vertices[1], mesh->vertices[2] };
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        const float * v = mesh->vertices + (size_t) i * 3u;
        for (int axis = 0; axis < 3; ++axis) {
            if (v[axis] < minv[axis]) minv[axis] = v[axis];
            if (v[axis] > maxv[axis]) maxv[axis] = v[axis];
        }
    }
    float center[3] = {
        (minv[0] + maxv[0]) * 0.5f,
        (minv[1] + maxv[1]) * 0.5f,
        (minv[2] + maxv[2]) * 0.5f,
    };
    float extent = fmaxf(fmaxf(maxv[0] - minv[0], maxv[1] - minv[1]), maxv[2] - minv[2]);
    float scale = extent > 1e-6f ? VIEWER_PREVIEW_AABB_SIZE / extent : 1.0f;

    if (mesh->n_faces > (int64_t) INT_MAX / 3) {
        shared_set_error(shared, "Mesh too large", "mesh has too many faces for the current raylib preview buffer");
        return;
    }
    int triangle_count = (int) mesh->n_faces;
    int vertex_count = triangle_count * 3;
    float * vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    float * normals = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    unsigned char * colors = (unsigned char *) malloc((size_t) vertex_count * 4u);
    float * smooth_normals = (float *) calloc((size_t) mesh->n_vertices * 3u, sizeof(float));
    if (vertices == NULL || normals == NULL || colors == NULL || smooth_normals == NULL) {
        free(vertices);
        free(normals);
        free(colors);
        free(smooth_normals);
        return;
    }
    uint32_t pbr_hash_size = 0;
    int32_t * pbr_hash = viewer_build_pbr_hash(pbr_voxels, &pbr_hash_size);

    for (int i = 0; i < triangle_count; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        Vector3 tri[3];
        int valid = 1;
        for (int k = 0; k < 3; ++k) {
            int32_t vi = f[k];
            if (vi < 0 || vi >= mesh->n_vertices) {
                valid = 0;
                break;
            }
            const float * src = mesh->vertices + (size_t) vi * 3u;
            tri[k] = (Vector3) {
                (src[0] - center[0]) * scale,
                (src[2] - center[2]) * scale,
                (src[1] - center[1]) * scale,
            };
        }
        if (!valid) {
            continue;
        }
        Vector3 e0 = Vector3Subtract(tri[1], tri[0]);
        Vector3 e1 = Vector3Subtract(tri[2], tri[0]);
        Vector3 area_n = Vector3CrossProduct(e0, e1);
        if (vec3_len_sq(area_n) <= 1e-14f) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            float * acc = smooth_normals + (size_t) f[k] * 3u;
            acc[0] += area_n.x;
            acc[1] += area_n.y;
            acc[2] += area_n.z;
        }
    }

    for (int i = 0; i < triangle_count; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        Vector3 tri[3];
        unsigned char tri_color[3][4];
        for (int k = 0; k < 3; ++k) {
            int32_t vi = f[k];
            if (vi < 0 || vi >= mesh->n_vertices) {
                tri[k] = (Vector3) {0.0f, 0.0f, 0.0f};
                tri_color[k][0] = 176;
                tri_color[k][1] = 190;
                tri_color[k][2] = 206;
                tri_color[k][3] = 255;
            } else {
                const float * src = mesh->vertices + (size_t) vi * 3u;
                viewer_sample_pbr_color(pbr_voxels, pbr_hash, pbr_hash_size, src, tri_color[k]);
                tri[k] = (Vector3) {
                    (src[0] - center[0]) * scale,
                    (src[2] - center[2]) * scale,
                    (src[1] - center[1]) * scale,
                };
            }
        }
        Vector3 e0 = Vector3Subtract(tri[1], tri[0]);
        Vector3 e1 = Vector3Subtract(tri[2], tri[0]);
        Vector3 face_n = Vector3CrossProduct(e0, e1);
        face_n = vec3_len_sq(face_n) > 1e-14f ? Vector3Normalize(face_n) : (Vector3) {0.0f, 1.0f, 0.0f};
        for (int k = 0; k < 3; ++k) {
            float * dst = vertices + ((size_t) i * 3u + (size_t) k) * 3u;
            float * ndst = normals + ((size_t) i * 3u + (size_t) k) * 3u;
            dst[0] = tri[k].x;
            dst[1] = tri[k].y;
            dst[2] = tri[k].z;
            Vector3 n = face_n;
            int32_t vi = f[k];
            if (vi >= 0 && vi < mesh->n_vertices) {
                Vector3 smooth_n = vec3_from_floats(smooth_normals + (size_t) vi * 3u);
                if (vec3_len_sq(smooth_n) > 1e-14f) {
                    n = Vector3Normalize(smooth_n);
                }
            }
            ndst[0] = n.x;
            ndst[1] = n.y;
            ndst[2] = n.z;
            unsigned char * cdst = colors + ((size_t) i * 3u + (size_t) k) * 4u;
            cdst[0] = tri_color[k][0];
            cdst[1] = tri_color[k][1];
            cdst[2] = tri_color[k][2];
            cdst[3] = tri_color[k][3];
        }
    }
    free(pbr_hash);
    free(smooth_normals);

    pthread_mutex_lock(&shared->mutex);
    viewer_snapshot_free(&shared->snapshot);
    shared->snapshot.kind = VIEWER_SNAPSHOT_MESH;
    shared->snapshot.version += 1;
    copy_text(shared->snapshot.label, sizeof(shared->snapshot.label), label);
    shared->snapshot.mesh_vertices = vertices;
    shared->snapshot.mesh_normals = normals;
    shared->snapshot.mesh_colors = colors;
    shared->snapshot.mesh_vertex_count = vertex_count;
    shared->snapshot.mesh_triangle_count = triangle_count;
    shared->snapshot.source_vertices = mesh->n_vertices;
    shared->snapshot.source_faces = mesh->n_faces;
    pthread_mutex_unlock(&shared->mutex);
}

static size_t model_cache_budget_bytes(const viewer_options * options) {
    if (options->model_cache_budget_mib <= 0) {
        return 0;
    }
    return (size_t) options->model_cache_budget_mib * 1024u * 1024u;
}

static trellis_status sparse_kind_from_backend(trellis_backend_kind graph_kind, trellis_sparse_backend_kind * out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (graph_kind == TRELLIS_BACKEND_CUDA) {
        *out = TRELLIS_SPARSE_BACKEND_CUDA;
        return TRELLIS_STATUS_OK;
    }
    if (graph_kind == TRELLIS_BACKEND_VULKAN) {
        *out = TRELLIS_SPARSE_BACKEND_VULKAN;
        return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_INVALID_ARGUMENT;
}

#ifdef _WIN32
static int append_process_arg(char * cmd, size_t cmd_size, const char * arg) {
    if (cmd == NULL || cmd_size == 0 || arg == NULL) {
        return 0;
    }
    size_t pos = strlen(cmd);
    if (pos + 3u >= cmd_size) {
        return 0;
    }
    if (pos > 0) {
        cmd[pos++] = ' ';
    }
    cmd[pos++] = '"';
    for (const char * p = arg; *p != '\0'; ++p) {
        if (*p == '"') {
            if (pos + 2u >= cmd_size) {
                return 0;
            }
            cmd[pos++] = '\\';
            cmd[pos++] = *p;
        } else {
            if (pos + 1u >= cmd_size) {
                return 0;
            }
            cmd[pos++] = *p;
        }
    }
    if (pos + 2u >= cmd_size) {
        return 0;
    }
    cmd[pos++] = '"';
    cmd[pos] = '\0';
    return 1;
}
#endif

static int run_process(char * const argv[]) {
    if (argv == NULL || argv[0] == NULL) {
        return 0;
    }
#ifdef _WIN32
    char cmd[VIEWER_MAX_PATH * 2u];
    cmd[0] = '\0';
    for (int i = 0; argv[i] != NULL; ++i) {
        if (!append_process_arg(cmd, sizeof(cmd), argv[i])) {
            return 0;
        }
    }
    WCHAR wide_cmd[VIEWER_MAX_PATH * 2u];
    if (!viewer_utf8_to_wide(cmd, wide_cmd, sizeof(wide_cmd) / sizeof(wide_cmd[0]))) {
        return 0;
    }

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = 0;
    if (!CreateProcessW(NULL, wide_cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        return 0;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code == 0;
#else
    return trellis_path_has_sep(argv[0]) ?
        trellis_run_process_exact(argv) :
        trellis_run_process_search_path(argv);
#endif
}

static int convert_webp_to_png(const char * input, const char * output) {
    char * const argv[] = {
        (char *) "ffmpeg",
        (char *) "-y",
        (char *) "-loglevel",
        (char *) "error",
        (char *) "-i",
        (char *) input,
        (char *) output,
        NULL,
    };
    return run_process(argv);
}

static int prepare_image_for_pipeline(
    const viewer_options * options,
    const char * input,
    char * prepared,
    size_t prepared_size) {
    if (!path_has_ext_ci(input, ".webp")) {
        copy_text(prepared, prepared_size, input);
        return 1;
    }
    if (!mkdir_p_simple(options->out_dir)) {
        return 0;
    }
    if (!trellis_make_temp_path(prepared, prepared_size, "trellis_viewer_input", ".png")) {
        return 0;
    }
    return convert_webp_to_png(input, prepared);
}

static trellis_status remove_background_to_png(
    const viewer_options * options,
    const char * input_path,
    char * out_path,
    size_t out_path_size) {
    if (text_is_empty(options->birefnet_path) || !viewer_file_exists(options->birefnet_path)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_p_simple(options->out_dir)) {
        return TRELLIS_STATUS_IO_ERROR;
    }

    char prepared[VIEWER_MAX_PATH];
    if (!prepare_image_for_pipeline(options, input_path, prepared, sizeof(prepared))) {
        return TRELLIS_STATUS_IO_ERROR;
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(prepared, &width, &height, &comp, 4);
    if (rgba == NULL || width <= 0 || height <= 0) {
        stbi_image_free(rgba);
        return TRELLIS_STATUS_IO_ERROR;
    }

    trellis_backend_kind backend_kind = TRELLIS_BACKEND_CPU;
    trellis_status status = trellis_backend_kind_from_name(options->backend, &backend_kind);
    if (status != TRELLIS_STATUS_OK) {
        stbi_image_free(rgba);
        return status;
    }

    trellis_birefnet_model model;
    memset(&model, 0, sizeof(model));
    status = trellis_birefnet_load_gguf_with_backend(&model, options->birefnet_path, backend_kind, options->device);

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
        int n = snprintf(out_path, out_path_size, "%s/trellis_viewer_bg_removed_%ld.png", options->out_dir, trellis_getpid());
        if (n < 0 || (size_t) n >= out_path_size) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        } else if (!stbi_write_png(out_path, width, height, 4, rgba, width * 4)) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }

    free(mask);
    trellis_birefnet_free(&model);
    stbi_image_free(rgba);
    return status;
}

static int write_meshbin(const char * path, const trellis_mesh_host * mesh) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return 0;
    }
    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint64_t n_vertices = (uint64_t) mesh->n_vertices;
    const uint64_t n_faces = (uint64_t) mesh->n_faces;
    const uint32_t flags = 0;
    const uint32_t reserved = 0;
    const size_t vertex_count = (size_t) mesh->n_vertices * 3u;
    const size_t face_count = (size_t) mesh->n_faces * 3u;
    int ok =
        fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        fwrite(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
        fwrite(&n_faces, sizeof(n_faces), 1, f) == 1 &&
        fwrite(&flags, sizeof(flags), 1, f) == 1 &&
        fwrite(&reserved, sizeof(reserved), 1, f) == 1 &&
        fwrite(mesh->vertices, sizeof(float), vertex_count, f) == vertex_count &&
        fwrite(mesh->faces, sizeof(int32_t), face_count, f) == face_count;
    if (fclose(f) != 0) {
        ok = 0;
    }
    return ok;
}

static trellis_status read_meshbin(const char * path, trellis_mesh_host * mesh_out) {
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        n_faces > (uint64_t) SIZE_MAX / (3u * sizeof(int32_t))) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }
    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (status == TRELLIS_STATUS_OK) {
        mesh.n_vertices = (int64_t) n_vertices;
        mesh.n_faces = (int64_t) n_faces;
        mesh.vertices = (float *) malloc((size_t) n_vertices * 3u * sizeof(float));
        mesh.faces = (int32_t *) malloc((size_t) n_faces * 3u * sizeof(int32_t));
        if (mesh.vertices == NULL || mesh.faces == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else if (fread(mesh.vertices, sizeof(float), (size_t) n_vertices * 3u, f) != (size_t) n_vertices * 3u ||
                   fread(mesh.faces, sizeof(int32_t), (size_t) n_faces * 3u, f) != (size_t) n_faces * 3u) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
        if (status == TRELLIS_STATUS_OK && (flags & 1u) != 0) {
            uint64_t uv_bytes = n_vertices * 2u * (uint64_t) sizeof(float);
            if (uv_bytes > (uint64_t) LONG_MAX || fseek(f, (long) uv_bytes, SEEK_CUR) != 0) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static int find_vkmesh_executable(char * out, size_t out_size) {
    char candidate[VIEWER_MAX_PATH];
    if (trellis_current_executable_path(candidate, sizeof(candidate))) {
        char * slash = trellis_path_last_sep(candidate);
        if (slash != NULL) {
            slash[1] = '\0';
            size_t len = strlen(candidate);
            const char * exe_name = "vkmesh" TRELLIS_EXE_SUFFIX;
            size_t exe_len = strlen(exe_name);
            if (len + exe_len + 1u < sizeof(candidate)) {
                memcpy(candidate + len, exe_name, exe_len + 1u);
                if (trellis_access_executable(candidate)) {
                    copy_text(out, out_size, candidate);
                    return 1;
                }
            }
        }
    }
    const char * local_candidates[] = {
        "build-vulkan/vkmesh" TRELLIS_EXE_SUFFIX,
        "build-cuda/vkmesh" TRELLIS_EXE_SUFFIX,
        "build-win/Release/vkmesh" TRELLIS_EXE_SUFFIX,
        "vkmesh" TRELLIS_EXE_SUFFIX,
        NULL,
    };
    for (int i = 0; local_candidates[i] != NULL; ++i) {
        if (trellis_access_executable(local_candidates[i])) {
            copy_text(out, out_size, local_candidates[i]);
            return 1;
        }
    }
    if (command_available("vkmesh")) {
        copy_text(out, out_size, "vkmesh");
        return 1;
    }
    return 0;
}

static int run_vkmesh_external(
    const trellis_mesh_host * mesh,
    trellis_mesh_host * processed_out,
    trellis_mesh_host * projection_out,
    int target,
    int no_simplify) {
    char exe[VIEWER_MAX_PATH];
    if (!find_vkmesh_executable(exe, sizeof(exe))) {
        return -1;
    }
    char input_mesh[VIEWER_MAX_PATH];
    char output_mesh[VIEWER_MAX_PATH];
    char projection_mesh[VIEWER_MAX_PATH];
    if (!trellis_make_temp_path(input_mesh, sizeof(input_mesh), "trellis_viewer_vkmesh_in", ".meshbin") ||
        !trellis_make_temp_path(output_mesh, sizeof(output_mesh), "trellis_viewer_vkmesh_out", ".meshbin") ||
        !trellis_make_temp_path(projection_mesh, sizeof(projection_mesh), "trellis_viewer_vkmesh_projection", ".meshbin")) {
        return 0;
    }
    if (!write_meshbin(input_mesh, mesh)) {
        return 0;
    }
    char target_text[64];
    snprintf(target_text, sizeof(target_text), "%d", target > 0 ? target : 1000000);
    char * argv[16];
    int argc = 0;
    argv[argc++] = exe;
    argv[argc++] = (char *) "--input";
    argv[argc++] = input_mesh;
    argv[argc++] = (char *) "--output";
    argv[argc++] = output_mesh;
    argv[argc++] = (char *) "--projection-mesh-output";
    argv[argc++] = projection_mesh;
    argv[argc++] = (char *) "--postprocess";
    argv[argc++] = (char *) "--decimation-target";
    argv[argc++] = target_text;
    if (no_simplify) {
        argv[argc++] = (char *) "--no-simplify";
    }
    argv[argc++] = (char *) "--no-uv-unwrap";
    argv[argc] = NULL;
    int ok = run_process(argv);
    if (ok) {
        ok = read_meshbin(output_mesh, processed_out) == TRELLIS_STATUS_OK;
        if (ok) {
            trellis_status projection_status = read_meshbin(projection_mesh, projection_out);
            if (projection_status != TRELLIS_STATUS_OK) {
                memset(projection_out, 0, sizeof(*projection_out));
            }
        }
    }
    trellis_unlink(input_mesh);
    trellis_unlink(output_mesh);
    trellis_unlink(projection_mesh);
    return ok ? 1 : 0;
}

static trellis_status run_mesh_postprocess(
    const viewer_options * options,
    trellis_mesh_host * mesh,
    trellis_mesh_host * projection_out) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_mesh_host processed;
    trellis_mesh_host projection;
    memset(&processed, 0, sizeof(processed));
    memset(&projection, 0, sizeof(projection));
#if TRELLIS_HAS_VKMESH_C_API
    trellis_vkmesh_postprocess_options pp;
    memset(&pp, 0, sizeof(pp));
    pp.decimation_target = options->mesh_postprocess_decimation_target > 0 ? options->mesh_postprocess_decimation_target : 1000000;
    pp.no_simplify = options->mesh_postprocess_no_simplify ? 1 : 0;
    trellis_status status = trellis_vkmesh_postprocess(mesh, &processed, &projection, &pp);
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&processed);
        trellis_mesh_free(&projection);
        return status;
    }
#else
    int external_status = run_vkmesh_external(
            mesh,
            &processed,
            &projection,
            options->mesh_postprocess_decimation_target,
            options->mesh_postprocess_no_simplify);
    if (external_status < 0) {
        return TRELLIS_STATUS_NOT_FOUND;
    }
    if (external_status == 0) {
        return TRELLIS_STATUS_ERROR;
    }
#endif
    trellis_mesh_free(mesh);
    *mesh = processed;
    if (projection_out != NULL) {
        trellis_mesh_free(projection_out);
        *projection_out = projection;
    } else {
        trellis_mesh_free(&projection);
    }
    return TRELLIS_STATUS_OK;
}

static void cleanup_backends(
    trellis_backend_context * graph_backend,
    trellis_cuda_context * cuda,
    trellis_sparse_backend * sparse_backend,
    trellis_sparse_backend_kind sparse_kind,
    trellis_pipeline_model_cache * cache,
    int cache_initialized) {
    if (cache_initialized) {
        trellis_pipeline_model_cache_free(cache);
    }
    if (sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        trellis_cuda_free(cuda);
    }
    trellis_sparse_backend_destroy(sparse_backend);
    trellis_backend_free(graph_backend);
}

static trellis_status init_pipeline_backends(
    const viewer_options * options,
    trellis_backend_context * graph_backend,
    trellis_cuda_context * cuda,
    trellis_sparse_backend ** sparse_backend,
    trellis_sparse_backend_kind * sparse_kind,
    trellis_pipeline_model_cache * cache,
    int * cache_initialized) {
    memset(graph_backend, 0, sizeof(*graph_backend));
    memset(cuda, 0, sizeof(*cuda));
    memset(cache, 0, sizeof(*cache));
    *sparse_backend = NULL;
    *cache_initialized = 0;

    trellis_backend_kind graph_kind = TRELLIS_BACKEND_CUDA;
    trellis_status status = trellis_backend_kind_from_name(options->backend, &graph_kind);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = sparse_kind_from_backend(graph_kind, sparse_kind);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = trellis_backend_init(graph_backend, graph_kind, options->device);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (*sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        status = trellis_cuda_init(cuda, options->device);
        if (status != TRELLIS_STATUS_OK) {
            trellis_backend_free(graph_backend);
            return status;
        }
    } else if (*sparse_kind == TRELLIS_SPARSE_BACKEND_VULKAN) {
        status = trellis_sparse_vulkan_backend_create(options->device, sparse_backend);
        if (status != TRELLIS_STATUS_OK) {
            trellis_backend_free(graph_backend);
            return status;
        }
    }
    if (options->model_cache) {
        status = trellis_pipeline_model_cache_init(
            cache,
            graph_backend,
            *sparse_kind != TRELLIS_SPARSE_BACKEND_CUDA,
            model_cache_budget_bytes(options));
        if (status != TRELLIS_STATUS_OK) {
            cleanup_backends(graph_backend, cuda, *sparse_backend, *sparse_kind, cache, 0);
            return status;
        }
        *cache_initialized = 1;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status run_mesh_job(viewer_worker_args * args) {
    viewer_shared * shared = args->shared;
    const viewer_options * options = &args->options;
    char prepared_image[VIEWER_MAX_PATH];
    char issue[256];

    if (!viewer_weights_ready(options, 0, issue, sizeof(issue))) {
        shared_set_error(shared, "Weights incomplete", issue);
        return TRELLIS_STATUS_NOT_FOUND;
    }

    shared_set_stage(shared, "Image prep", "preparing input for pipeline", 0.03f, 1);
    if (!prepare_image_for_pipeline(options, args->image_path, prepared_image, sizeof(prepared_image))) {
        return TRELLIS_STATUS_IO_ERROR;
    }

    trellis_backend_context graph_backend;
    trellis_cuda_context cuda;
    trellis_sparse_backend * sparse_backend = NULL;
    trellis_sparse_backend_kind sparse_kind = TRELLIS_SPARSE_BACKEND_CUDA;
    trellis_pipeline_model_cache cache;
    int cache_initialized = 0;
    trellis_status status = init_pipeline_backends(
        options,
        &graph_backend,
        &cuda,
        &sparse_backend,
        &sparse_kind,
        &cache,
        &cache_initialized);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    viewer_artifacts next;
    memset(&next, 0, sizeof(next));
    next.pipeline_resolution = options->resolution;

    const int sparse_steps = options->sparse_structure_steps > 0 ? options->sparse_structure_steps : 12;
    shared_set_stage(shared, "Sparse structure", "image -> sparse voxels", 0.10f, 1);
    shared_set_step_plan(shared, "sparse structure", sparse_steps, "waiting for first denoise step");
    trellis_sparse_structure_options sparse_options;
    memset(&sparse_options, 0, sizeof(sparse_options));
    sparse_options.model_dir = options->model_dir;
    sparse_options.dino_dir = options->dino_dir;
    sparse_options.image_path = prepared_image;
    sparse_options.latent_size = options->latent_size > 0 ? options->latent_size : 16;
    sparse_options.steps = sparse_steps;
    sparse_options.cond_resolution = options->cond_resolution;
    sparse_options.sparse_resolution = options->sparse_resolution;
    sparse_options.seed = options->seed == 0 ? 1u : options->seed;
    sparse_options.flow_blocks_override = options->flow_blocks_override;
    sparse_options.flow_block_parts_override = options->flow_block_parts_override;
    sparse_options.flow_no_rope = options->flow_no_rope;
    sparse_options.use_ggml_flash_attn = options->no_ggml_flash_attn ? 0 : options->use_ggml_flash_attn;
    sparse_options.backend = &graph_backend;
    sparse_options.cuda = sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    sparse_options.cache = cache_initialized ? &cache : NULL;
    status = trellis_pipeline_run_sparse_structure(&sparse_options, &next.sparse);
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    next.has_sparse = 1;
    trellis_pipeline_model_cache_unpin_all(cache_initialized ? &cache : NULL);
    publish_voxel_snapshot(shared, next.sparse.coords_bxyz, next.sparse.n_coords, next.sparse.resolution, "Sparse voxel intermediate");

    const int shape_steps = options->structured_latent_steps > 0 ? options->structured_latent_steps : 12;
    shared_set_stage(shared, "Shape denoise", "sparse voxels -> shape latent", 0.38f, 1);
    shared_set_step_plan(shared, "structured latent shape", shape_steps, "waiting for first denoise step");
    trellis_structured_latent_options shape_options;
    memset(&shape_options, 0, sizeof(shape_options));
    shape_options.model_dir = options->model_dir;
    shape_options.flow_override_path = text_is_empty(options->flow_override_path) ? NULL : options->flow_override_path;
    shape_options.flow_component = TRELLIS_COMPONENT_SHAPE_SLAT_FLOW;
    shape_options.label = "shape";
    shape_options.normalization_key = "shape_slat_normalization";
    shape_options.coords_bxyz = next.sparse.coords_bxyz;
    shape_options.n_coords = next.sparse.n_coords;
    shape_options.cond = next.sparse.cond;
    shape_options.cond_tokens = next.sparse.cond_tokens;
    shape_options.noise_seed = options->noise_seed == 0 ? 18u : options->noise_seed;
    shape_options.resolution = options->resolution;
    shape_options.steps = shape_steps;
    shape_options.rescale_t = options->rescale_t > 0.0f ? options->rescale_t : 3.0f;
    shape_options.guidance_strength = options->guidance_strength;
    shape_options.guidance_rescale = options->guidance_rescale;
    shape_options.guidance_min = options->guidance_min;
    shape_options.guidance_max = options->guidance_max;
    shape_options.flow_blocks_override = options->flow_blocks_override;
    shape_options.flow_block_parts_override = options->flow_block_parts_override;
    shape_options.flow_no_rope = options->flow_no_rope;
    shape_options.emulate_bf16_blocks = options->emulate_bf16_blocks;
    shape_options.use_ggml_flash_attn = options->no_ggml_flash_attn ? 0 : options->use_ggml_flash_attn;
    shape_options.backend = &graph_backend;
    shape_options.cuda = sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    shape_options.cache = cache_initialized ? &cache : NULL;
    status = trellis_pipeline_run_structured_latent(&shape_options, &next.shape_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    next.has_shape = 1;
    trellis_pipeline_model_cache_unpin_all(cache_initialized ? &cache : NULL);

    shared_set_stage(shared, "Mesh decode", "shape latent -> high-poly mesh", 0.72f, 1);
    trellis_pipeline_mesh_options mesh_options;
    memset(&mesh_options, 0, sizeof(mesh_options));
    mesh_options.model_dir = options->model_dir;
    mesh_options.decoder_override_path = text_is_empty(options->decoder_override_path) ? NULL : options->decoder_override_path;
    mesh_options.latent = &next.shape_latent;
    mesh_options.resolution = next.shape_latent.resolution;
    mesh_options.decode_max_levels = options->decode_max_levels;
    mesh_options.decode_max_input_tokens = options->decode_max_input_tokens;
    mesh_options.cuda = sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    mesh_options.sparse_backend_kind = sparse_kind;
    mesh_options.sparse_device = options->device;
    mesh_options.sparse_backend = sparse_backend;
    mesh_options.cache = cache_initialized ? &cache : NULL;

    viewer_env_override c2s_guides_env;
    memset(&c2s_guides_env, 0, sizeof(c2s_guides_env));
    if (sparse_kind == TRELLIS_SPARSE_BACKEND_VULKAN) {
        /*
         * The viewer persists shape_subs after this worker destroys its Vulkan
         * sparse backend. Keep those subdivision guides host-owned so the later
         * texture worker does not reuse dangling device maps.
         */
        viewer_env_override_begin(&c2s_guides_env, "TRELLIS_VK_DEVICE_C2S_GUIDES", "0");
    }
    status = trellis_pipeline_decode_shape_latent_mesh(&mesh_options, &next.shape_subs, &next.mesh);
    if (sparse_kind == TRELLIS_SPARSE_BACKEND_VULKAN) {
        viewer_env_override_end(&c2s_guides_env);
    }
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    next.has_mesh = 1;
    trellis_pipeline_model_cache_unpin_all(cache_initialized ? &cache : NULL);
    publish_mesh_snapshot(shared, &next.mesh, NULL, "Raw mesh intermediate");

    pthread_mutex_lock(&shared->mutex);
    viewer_artifacts_free(&shared->artifacts);
    shared->artifacts = next;
    memset(&next, 0, sizeof(next));
    shared->mesh_ready = 1;
    shared->texture_ready = 0;
    shared->postprocess_ready = 0;
    shared->uv_texture_path[0] = '\0';
    shared->gltf_output_path[0] = '\0';
    shared->uv_version += 1;
    copy_text(shared->stage, sizeof(shared->stage), "Mesh ready");
    snprintf(shared->detail, sizeof(shared->detail), "%lld vertices, %lld faces", (long long) shared->artifacts.mesh.n_vertices, (long long) shared->artifacts.mesh.n_faces);
    shared->progress = 1.0f;
    shared->indeterminate = 0;
    pthread_mutex_unlock(&shared->mutex);

done:
    viewer_artifacts_free(&next);
    cleanup_backends(&graph_backend, &cuda, sparse_backend, sparse_kind, &cache, cache_initialized);
    return status;
}

static trellis_status run_texture_job(viewer_worker_args * args) {
    viewer_shared * shared = args->shared;
    const viewer_options * options = &args->options;
    char issue[256];

    if (!viewer_weights_ready(options, 0, issue, sizeof(issue))) {
        shared_set_error(shared, "Weights incomplete", issue);
        return TRELLIS_STATUS_NOT_FOUND;
    }

    trellis_backend_context graph_backend;
    trellis_cuda_context cuda;
    trellis_sparse_backend * sparse_backend = NULL;
    trellis_sparse_backend_kind sparse_kind = TRELLIS_SPARSE_BACKEND_CUDA;
    trellis_pipeline_model_cache cache;
    int cache_initialized = 0;
    trellis_status status = init_pipeline_backends(
        options,
        &graph_backend,
        &cuda,
        &sparse_backend,
        &sparse_kind,
        &cache,
        &cache_initialized);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    trellis_structured_latent texture_latent;
    trellis_pbr_voxels pbr_voxels;
    memset(&texture_latent, 0, sizeof(texture_latent));
    memset(&pbr_voxels, 0, sizeof(pbr_voxels));

    const int texture_steps = options->structured_latent_steps > 0 ? options->structured_latent_steps : 12;
    shared_set_stage(shared, "Texture denoise", "image + shape -> texture latent", 0.20f, 1);
    shared_set_step_plan(shared, "structured latent texture", texture_steps, "waiting for first denoise step");
    pthread_mutex_lock(&shared->mutex);
    viewer_artifacts * artifacts = &shared->artifacts;
    int ready = shared->mesh_ready && artifacts->has_mesh && artifacts->has_shape && artifacts->has_sparse;
    pthread_mutex_unlock(&shared->mutex);
    if (!ready) {
        cleanup_backends(&graph_backend, &cuda, sparse_backend, sparse_kind, &cache, cache_initialized);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_structured_latent_options texture_options;
    memset(&texture_options, 0, sizeof(texture_options));
    texture_options.model_dir = options->model_dir;
    texture_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    texture_options.label = "texture";
    texture_options.normalization_key = "tex_slat_normalization";
    texture_options.coords_bxyz = artifacts->shape_latent.coords_bxyz;
    texture_options.n_coords = artifacts->shape_latent.n_coords;
    texture_options.cond = artifacts->sparse.cond;
    texture_options.cond_tokens = artifacts->sparse.cond_tokens;
    texture_options.concat_cond = artifacts->shape_latent.feats;
    texture_options.concat_channels = artifacts->shape_latent.channels;
    texture_options.noise_seed = options->noise_seed == 0 ? 19u : options->noise_seed + 1u;
    texture_options.resolution = artifacts->shape_latent.resolution;
    texture_options.steps = texture_steps;
    texture_options.rescale_t = 3.0f;
    texture_options.guidance_strength = 1.0f;
    texture_options.guidance_rescale = 0.0f;
    texture_options.guidance_min = 0.6f;
    texture_options.guidance_max = 0.9f;
    texture_options.flow_blocks_override = options->flow_blocks_override;
    texture_options.flow_block_parts_override = options->flow_block_parts_override;
    texture_options.flow_no_rope = options->flow_no_rope;
    texture_options.emulate_bf16_blocks = options->emulate_bf16_blocks;
    texture_options.use_ggml_flash_attn = options->no_ggml_flash_attn ? 0 : options->use_ggml_flash_attn;
    texture_options.backend = &graph_backend;
    texture_options.cuda = sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    texture_options.cache = cache_initialized ? &cache : NULL;
    status = trellis_pipeline_run_structured_latent(&texture_options, &texture_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    trellis_pipeline_model_cache_unpin_all(cache_initialized ? &cache : NULL);

    shared_set_stage(shared, "Texture decode", "texture latent -> PBR voxels", 0.58f, 1);
    trellis_pipeline_texture_options decode_options;
    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.model_dir = options->model_dir;
    decode_options.latent = &texture_latent;
    decode_options.guide_subs = &artifacts->shape_subs;
    decode_options.decode_max_levels = options->decode_max_levels;
    decode_options.decode_max_input_tokens = options->decode_max_input_tokens;
    decode_options.cuda = sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    decode_options.sparse_backend_kind = sparse_kind;
    decode_options.sparse_device = options->device;
    decode_options.sparse_backend = sparse_backend;
    decode_options.cache = cache_initialized ? &cache : NULL;
    status = trellis_pipeline_decode_texture_latent_voxels(&decode_options, &pbr_voxels);
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    trellis_pipeline_model_cache_unpin_all(cache_initialized ? &cache : NULL);

    shared_set_stage(shared, "UV bake", "unwrap + bake glTF textures", 0.82f, 1);
    char gltf_path[VIEWER_MAX_PATH];
    char base_color_path[VIEWER_MAX_PATH];
    if (!mkdir_p_simple(options->out_dir) || !make_default_gltf_path(options, gltf_path, sizeof(gltf_path))) {
        status = TRELLIS_STATUS_IO_ERROR;
        goto done;
    }
    const trellis_mesh_host * sample_mesh =
        artifacts->projection_mesh.vertices != NULL && artifacts->projection_mesh.faces != NULL ?
            &artifacts->projection_mesh :
            NULL;
    status = trellis_pipeline_write_gltf(gltf_path, &artifacts->mesh, sample_mesh, &pbr_voxels, options->texture_size);
    if (status != TRELLIS_STATUS_OK) {
        goto done;
    }
    make_base_color_path(gltf_path, base_color_path, sizeof(base_color_path));
    publish_mesh_snapshot(shared, &artifacts->mesh, &pbr_voxels, "Textured mesh preview");
    pthread_mutex_lock(&shared->mutex);
    shared->texture_ready = 1;
    copy_text(shared->gltf_output_path, sizeof(shared->gltf_output_path), gltf_path);
    copy_text(shared->uv_texture_path, sizeof(shared->uv_texture_path), base_color_path);
    shared->uv_version += 1;
    copy_text(shared->stage, sizeof(shared->stage), "Texture ready");
    snprintf(shared->detail, sizeof(shared->detail), "wrote %s", gltf_path);
    shared->progress = 1.0f;
    shared->indeterminate = 0;
    pthread_mutex_unlock(&shared->mutex);

done:
    trellis_pbr_voxels_free(&pbr_voxels);
    trellis_structured_latent_free(&texture_latent);
    cleanup_backends(&graph_backend, &cuda, sparse_backend, sparse_kind, &cache, cache_initialized);
    return status;
}

static trellis_status run_postprocess_job(viewer_worker_args * args) {
    viewer_shared * shared = args->shared;
    const viewer_options * options = &args->options;
    shared_set_stage(shared, "Postprocess", "cleaning topology with vkmesh", 0.22f, 1);

    pthread_mutex_lock(&shared->mutex);
    int ready = shared->mesh_ready && shared->artifacts.has_mesh;
    pthread_mutex_unlock(&shared->mutex);
    if (!ready) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = run_mesh_postprocess(options, &shared->artifacts.mesh, &shared->artifacts.projection_mesh);
    if (status == TRELLIS_STATUS_OK) {
        shared->artifacts.has_postprocess = 1;
        publish_mesh_snapshot(shared, &shared->artifacts.mesh, NULL, "Postprocessed mesh");
        pthread_mutex_lock(&shared->mutex);
        shared->postprocess_ready = 1;
        shared->texture_ready = 0;
        shared->uv_texture_path[0] = '\0';
        shared->gltf_output_path[0] = '\0';
        shared->uv_version += 1;
        copy_text(shared->stage, sizeof(shared->stage), "Postprocess ready");
        snprintf(shared->detail, sizeof(shared->detail), "%lld vertices, %lld faces", (long long) shared->artifacts.mesh.n_vertices, (long long) shared->artifacts.mesh.n_faces);
        shared->progress = 1.0f;
        shared->indeterminate = 0;
        pthread_mutex_unlock(&shared->mutex);
    }
    return status;
}

static void * worker_main(void * user_data) {
    viewer_worker_args * args = (viewer_worker_args *) user_data;
    viewer_shared * shared = args->shared;
    shared_clear_error(shared);
    trellis_set_progress_step_callback(viewer_progress_step_callback, shared);
    trellis_status status = TRELLIS_STATUS_OK;

    if (args->job == VIEWER_JOB_REMOVE_BG) {
        shared_set_stage(shared, "Background removal", "BiRefNet matting network", 0.12f, 1);
        char out_path[VIEWER_MAX_PATH];
        status = remove_background_to_png(&args->options, args->image_path, out_path, sizeof(out_path));
        if (status == TRELLIS_STATUS_OK) {
            pthread_mutex_lock(&shared->mutex);
            copy_text(shared->processed_image_path, sizeof(shared->processed_image_path), out_path);
            copy_text(shared->current_image_path, sizeof(shared->current_image_path), out_path);
            shared->bg_ready = 1;
            shared->mesh_ready = 0;
            shared->texture_ready = 0;
            shared->postprocess_ready = 0;
            viewer_artifacts_free(&shared->artifacts);
            viewer_snapshot_free(&shared->snapshot);
            shared->snapshot.version += 1;
            shared->uv_texture_path[0] = '\0';
            shared->gltf_output_path[0] = '\0';
            shared->uv_version += 1;
            copy_text(shared->stage, sizeof(shared->stage), "Background ready");
            copy_text(shared->detail, sizeof(shared->detail), out_path);
            shared->progress = 1.0f;
            shared->indeterminate = 0;
            pthread_mutex_unlock(&shared->mutex);
        }
    } else if (args->job == VIEWER_JOB_MESH) {
        status = run_mesh_job(args);
    } else if (args->job == VIEWER_JOB_TEXTURE) {
        status = run_texture_job(args);
    } else if (args->job == VIEWER_JOB_POSTPROCESS) {
        status = run_postprocess_job(args);
    }

    if (status != TRELLIS_STATUS_OK) {
        int has_error = 0;
        pthread_mutex_lock(&shared->mutex);
        has_error = shared->error[0] != '\0';
        pthread_mutex_unlock(&shared->mutex);
        if (has_error) {
            goto finish;
        }
        char message[512];
        snprintf(message, sizeof(message), "%s", trellis_status_string(status));
        shared_set_error(shared, "Error", message);
    }

finish:
    trellis_set_progress_step_callback(NULL, NULL);
    pthread_mutex_lock(&shared->mutex);
    shared->worker_running = 0;
    shared->worker_done_token += 1;
    pthread_mutex_unlock(&shared->mutex);
    free(args);
    return NULL;
}

static int start_worker(viewer_shared * shared, const viewer_options * options, viewer_job_type job, const char * image_path, pthread_t * thread_out) {
    pthread_mutex_lock(&shared->mutex);
    if (shared->worker_running) {
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    shared->worker_running = 1;
    shared->progress = 0.0f;
    shared->indeterminate = 1;
    shared->error[0] = '\0';
    shared_reset_step_progress_locked(shared);
    pthread_mutex_unlock(&shared->mutex);

    viewer_worker_args * args = (viewer_worker_args *) calloc(1, sizeof(*args));
    if (args == NULL) {
        pthread_mutex_lock(&shared->mutex);
        shared->worker_running = 0;
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    args->shared = shared;
    args->options = *options;
    args->job = job;
    copy_text(args->image_path, sizeof(args->image_path), image_path);

    if (pthread_create(thread_out, NULL, worker_main, args) != 0) {
        free(args);
        pthread_mutex_lock(&shared->mutex);
        shared->worker_running = 0;
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    return 1;
}

static int command_available(const char * name) {
    return trellis_command_available(name);
}

static int read_first_line_from_command(const char * cmd, char * out, size_t out_size) {
#ifdef _WIN32
    FILE * f = _popen(cmd, "r");
#else
    FILE * f = popen(cmd, "r");
#endif
    if (f == NULL) {
        return 0;
    }
    int ok = 0;
    if (fgets(out, (int) out_size, f) != NULL) {
        size_t n = strlen(out);
        while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r')) {
            out[--n] = '\0';
        }
        ok = out[0] != '\0';
    }
#ifdef _WIN32
    _pclose(f);
#else
    pclose(f);
#endif
    return ok;
}

static int open_image_file_dialog(char * out, size_t out_size, int * dialog_available) {
    if (dialog_available != NULL) {
        *dialog_available = 0;
    }
#ifdef _WIN32
    if (dialog_available != NULL) {
        *dialog_available = 1;
    }
    return trellis_win_open_image_file_dialog(out, out_size);
#else
    if (command_available("zenity")) {
        if (dialog_available != NULL) {
            *dialog_available = 1;
        }
        if (read_first_line_from_command(
                "zenity --file-selection --title='Select input image' --file-filter='Images | *.png *.jpg *.jpeg *.webp *.bmp' 2>/dev/null",
                out,
                out_size)) {
            return 1;
        }
    }
    if (command_available("kdialog")) {
        if (dialog_available != NULL) {
            *dialog_available = 1;
        }
        if (read_first_line_from_command(
                "kdialog --getopenfilename \"$HOME\" 'Images (*.png *.jpg *.jpeg *.webp *.bmp)' 2>/dev/null",
                out,
                out_size)) {
            return 1;
        }
    }
#endif
    return 0;
}

static void * file_picker_main(void * user_data) {
    viewer_file_picker_args * args = (viewer_file_picker_args *) user_data;
    viewer_shared * shared = args == NULL ? NULL : args->shared;
    free(args);
    if (shared == NULL) {
        return NULL;
    }
    char path[VIEWER_MAX_PATH];
    path[0] = '\0';
    int dialog_available = 0;
    int selected = open_image_file_dialog(path, sizeof(path), &dialog_available);
    int unsupported = 0;
    if (selected) {
        unsupported = !viewer_file_exists(path) || !is_supported_image_path(path);
    }

    pthread_mutex_lock(&shared->mutex);
    if (selected && !unsupported) {
        copy_text(shared->selected_image_path, sizeof(shared->selected_image_path), path);
        shared->selected_image_pending = 1;
    } else if (!dialog_available) {
        copy_text(shared->stage, sizeof(shared->stage), "Picker unavailable");
        copy_text(shared->detail, sizeof(shared->detail), "drag an image into the candidate box");
    } else if (unsupported) {
        copy_text(shared->stage, sizeof(shared->stage), "Unsupported image");
        copy_text(shared->detail, sizeof(shared->detail), "use png, jpg, jpeg, webp, or bmp");
    }
    shared->file_picker_running = 0;
    pthread_mutex_unlock(&shared->mutex);
    return NULL;
}

static int start_file_picker(viewer_shared * shared, viewer_picker_kind kind) {
    pthread_mutex_lock(&shared->mutex);
    if (shared->worker_running || shared->file_picker_running) {
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    shared->file_picker_running = 1;
    copy_text(shared->stage, sizeof(shared->stage), "Select image");
    copy_text(shared->detail, sizeof(shared->detail), "opening picker");
    pthread_mutex_unlock(&shared->mutex);

    viewer_file_picker_args * args = (viewer_file_picker_args *) calloc(1, sizeof(*args));
    if (args == NULL) {
        pthread_mutex_lock(&shared->mutex);
        shared->file_picker_running = 0;
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    args->shared = shared;
    args->kind = kind;

    pthread_t thread;
    if (pthread_create(&thread, NULL, file_picker_main, args) != 0) {
        free(args);
        pthread_mutex_lock(&shared->mutex);
        shared->file_picker_running = 0;
        pthread_mutex_unlock(&shared->mutex);
        return 0;
    }
    pthread_detach(thread);
    return 1;
}

static void display_mesh_free(display_mesh * mesh) {
    for (int i = 0; i < mesh->model_count; ++i) {
        UnloadModel(mesh->models[i]);
    }
    free(mesh->models);
    free(mesh->vertices);
    free(mesh->normals);
    free(mesh->colors);
    memset(mesh, 0, sizeof(*mesh));
}

static void display_voxels_free(display_voxels * voxels) {
    free(voxels->xyz);
    memset(voxels, 0, sizeof(*voxels));
}

static void set_candidate_image(
    viewer_options * options,
    viewer_shared * shared,
    const char * path,
    display_mesh * mesh_display,
    display_voxels * voxel_display) {
    if (text_is_empty(path) || !viewer_file_exists(path) || !is_supported_image_path(path)) {
        pthread_mutex_lock(&shared->mutex);
        copy_text(shared->stage, sizeof(shared->stage), "Unsupported image");
        copy_text(shared->detail, sizeof(shared->detail), "use png, jpg, jpeg, webp, or bmp");
        pthread_mutex_unlock(&shared->mutex);
        return;
    }
    copy_text(options->image_path, sizeof(options->image_path), path);
    pthread_mutex_lock(&shared->mutex);
    copy_text(shared->current_image_path, sizeof(shared->current_image_path), path);
    shared->processed_image_path[0] = '\0';
    shared->bg_ready = 0;
    shared->mesh_ready = 0;
    shared->texture_ready = 0;
    shared->postprocess_ready = 0;
    shared->uv_texture_path[0] = '\0';
    shared->gltf_output_path[0] = '\0';
    shared->uv_version += 1;
    viewer_artifacts_free(&shared->artifacts);
    viewer_snapshot_free(&shared->snapshot);
    shared->snapshot.version += 1;
    copy_text(shared->stage, sizeof(shared->stage), "Image loaded");
    copy_text(shared->detail, sizeof(shared->detail), path);
    pthread_mutex_unlock(&shared->mutex);
    display_mesh_free(mesh_display);
    display_voxels_free(voxel_display);
}

static unsigned char clamp_u8(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (unsigned char) lrintf(v);
}

static float viewer_smooth_shade(Vector3 n, int lighting) {
    if (!lighting) {
        return 0.86f;
    }
    if (vec3_len_sq(n) <= 1e-12f) {
        n = (Vector3) {0.0f, 1.0f, 0.0f};
    } else {
        n = Vector3Normalize(n);
    }
    Vector3 key = Vector3Normalize((Vector3) {-0.42f, 0.78f, 0.46f});
    Vector3 fill = Vector3Normalize((Vector3) {0.68f, 0.36f, -0.58f});
    Vector3 rim = Vector3Normalize((Vector3) {0.18f, 0.48f, -0.86f});
    float key_term = fmaxf(0.0f, Vector3DotProduct(n, key));
    float fill_term = fmaxf(0.0f, Vector3DotProduct(n, fill));
    float rim_term = fmaxf(0.0f, Vector3DotProduct(n, rim));
    float sky = fmaxf(0.0f, n.y);
    float ground = fmaxf(0.0f, -n.y);
    float half_lambert = 0.5f + 0.5f * Vector3DotProduct(n, key);
    half_lambert = half_lambert * half_lambert;
    float shade =
        0.36f +
        0.22f * sky +
        0.08f * ground +
        0.34f * half_lambert +
        0.18f * key_term +
        0.10f * fill_term +
        0.08f * rim_term * rim_term;
    if (shade > 1.18f) {
        shade = 1.18f;
    }
    return shade;
}

static void display_mesh_upload(display_mesh * display, int lighting) {
    for (int i = 0; i < display->model_count; ++i) {
        UnloadModel(display->models[i]);
    }
    free(display->models);
    display->models = NULL;
    display->model_count = 0;
    if (display->vertices == NULL || display->normals == NULL || display->colors == NULL || display->vertex_count <= 0) {
        return;
    }

    int chunk_count = (display->triangle_count + VIEWER_MESH_CHUNK_FACES - 1) / VIEWER_MESH_CHUNK_FACES;
    Model * models = (Model *) calloc((size_t) chunk_count, sizeof(Model));
    if (models == NULL) {
        return;
    }

    int uploaded = 0;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        int first_face = chunk * VIEWER_MESH_CHUNK_FACES;
        int faces = display->triangle_count - first_face;
        if (faces > VIEWER_MESH_CHUNK_FACES) faces = VIEWER_MESH_CHUNK_FACES;
        int vertices = faces * 3;
        size_t first_vertex = (size_t) first_face * 3u;

        Mesh mesh = {0};
        mesh.vertexCount = vertices;
        mesh.triangleCount = faces;
        mesh.vertices = (float *) malloc((size_t) vertices * 3u * sizeof(float));
        mesh.normals = (float *) malloc((size_t) vertices * 3u * sizeof(float));
        mesh.colors = (unsigned char *) malloc((size_t) vertices * 4u);
        if (mesh.vertices == NULL || mesh.normals == NULL || mesh.colors == NULL) {
            free(mesh.vertices);
            free(mesh.normals);
            free(mesh.colors);
            break;
        }
        memcpy(mesh.vertices, display->vertices + first_vertex * 3u, (size_t) vertices * 3u * sizeof(float));
        memcpy(mesh.normals, display->normals + first_vertex * 3u, (size_t) vertices * 3u * sizeof(float));
        for (int i = 0; i < vertices; ++i) {
            Vector3 n = vec3_from_floats(mesh.normals + (size_t) i * 3u);
            float shade = viewer_smooth_shade(n, lighting);
            const unsigned char * src_color = display->colors + (first_vertex + (size_t) i) * 4u;
            mesh.colors[(size_t) i * 4u + 0u] = clamp_u8((float) src_color[0] * shade);
            mesh.colors[(size_t) i * 4u + 1u] = clamp_u8((float) src_color[1] * shade);
            mesh.colors[(size_t) i * 4u + 2u] = clamp_u8((float) src_color[2] * shade);
            mesh.colors[(size_t) i * 4u + 3u] = 255;
        }
        UploadMesh(&mesh, false);
        models[uploaded] = LoadModelFromMesh(mesh);
        models[uploaded].materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
        uploaded += 1;
    }
    if (uploaded == 0) {
        free(models);
        return;
    }
    display->models = models;
    display->model_count = uploaded;
    display->lighting = lighting;
}

static void copy_snapshot_to_display(viewer_shared * shared, display_mesh * mesh, display_voxels * voxels, int lighting) {
    viewer_snapshot local;
    memset(&local, 0, sizeof(local));
    pthread_mutex_lock(&shared->mutex);
    int version = shared->snapshot.version;
    viewer_snapshot_kind kind = shared->snapshot.kind;
    if ((kind == VIEWER_SNAPSHOT_MESH && version == mesh->version) ||
        (kind == VIEWER_SNAPSHOT_VOXELS && version == voxels->version) ||
        (kind == VIEWER_SNAPSHOT_NONE && version == mesh->version && version == voxels->version)) {
        pthread_mutex_unlock(&shared->mutex);
        if (kind == VIEWER_SNAPSHOT_MESH && mesh->model_count > 0 && mesh->lighting != lighting) {
            display_mesh_upload(mesh, lighting);
        }
        return;
    }
    local.kind = kind;
    local.version = version;
    local.voxel_count = shared->snapshot.voxel_count;
    local.voxel_resolution = shared->snapshot.voxel_resolution;
    local.mesh_vertex_count = shared->snapshot.mesh_vertex_count;
    local.mesh_triangle_count = shared->snapshot.mesh_triangle_count;
    local.source_vertices = shared->snapshot.source_vertices;
    local.source_faces = shared->snapshot.source_faces;
    if (shared->snapshot.voxel_xyz != NULL && local.voxel_count > 0) {
        local.voxel_xyz = (int32_t *) malloc((size_t) local.voxel_count * 3u * sizeof(int32_t));
        if (local.voxel_xyz != NULL) {
            memcpy(local.voxel_xyz, shared->snapshot.voxel_xyz, (size_t) local.voxel_count * 3u * sizeof(int32_t));
        }
    }
    if (shared->snapshot.mesh_vertices != NULL && local.mesh_vertex_count > 0) {
        local.mesh_vertices = (float *) malloc((size_t) local.mesh_vertex_count * 3u * sizeof(float));
        local.mesh_normals = (float *) malloc((size_t) local.mesh_vertex_count * 3u * sizeof(float));
        local.mesh_colors = (unsigned char *) malloc((size_t) local.mesh_vertex_count * 4u);
        if (local.mesh_vertices != NULL && local.mesh_normals != NULL && local.mesh_colors != NULL) {
            memcpy(local.mesh_vertices, shared->snapshot.mesh_vertices, (size_t) local.mesh_vertex_count * 3u * sizeof(float));
            memcpy(local.mesh_normals, shared->snapshot.mesh_normals, (size_t) local.mesh_vertex_count * 3u * sizeof(float));
            memcpy(local.mesh_colors, shared->snapshot.mesh_colors, (size_t) local.mesh_vertex_count * 4u);
        }
    }
    pthread_mutex_unlock(&shared->mutex);

    if (local.kind == VIEWER_SNAPSHOT_VOXELS && local.voxel_xyz != NULL) {
        display_mesh_free(mesh);
        display_voxels_free(voxels);
        voxels->xyz = local.voxel_xyz;
        voxels->count = local.voxel_count;
        voxels->resolution = local.voxel_resolution;
        voxels->version = local.version;
        local.voxel_xyz = NULL;
    } else if (local.kind == VIEWER_SNAPSHOT_MESH && local.mesh_vertices != NULL && local.mesh_normals != NULL) {
        display_voxels_free(voxels);
        display_mesh_free(mesh);
        mesh->vertices = local.mesh_vertices;
        mesh->normals = local.mesh_normals;
        mesh->colors = local.mesh_colors;
        mesh->vertex_count = local.mesh_vertex_count;
        mesh->triangle_count = local.mesh_triangle_count;
        mesh->source_vertices = local.source_vertices;
        mesh->source_faces = local.source_faces;
        mesh->version = local.version;
        local.mesh_vertices = NULL;
        local.mesh_normals = NULL;
        local.mesh_colors = NULL;
        display_mesh_upload(mesh, lighting);
    } else if (local.kind == VIEWER_SNAPSHOT_NONE) {
        display_mesh_free(mesh);
        display_voxels_free(voxels);
        mesh->version = local.version;
        voxels->version = local.version;
    } else if (mesh->model_count > 0 && mesh->lighting != lighting) {
        display_mesh_upload(mesh, lighting);
    }
    viewer_snapshot_free(&local);
}

static int ui_button(Rectangle r, const char * text, int enabled) {
    Vector2 mouse = GetMousePosition();
    int hot = enabled && CheckCollisionPointRec(mouse, r);
    Color bg = enabled ? (hot ? (Color) {68, 92, 112, 255} : (Color) {44, 56, 68, 255}) : (Color) {35, 39, 45, 255};
    Color border = enabled ? (hot ? (Color) {96, 183, 190, 255} : (Color) {75, 88, 98, 255}) : (Color) {54, 58, 64, 255};
    DrawRectangleRounded(r, 0.14f, 8, bg);
    DrawRectangleRoundedLines(r, 0.14f, 8, border);
    int font_size = 17;
    int tw = MeasureText(text, font_size);
    DrawText(text, (int) (r.x + (r.width - (float) tw) * 0.5f), (int) (r.y + (r.height - (float) font_size) * 0.5f), font_size, enabled ? RAYWHITE : (Color) {118, 126, 134, 255});
    return hot && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static int ui_small_button(Rectangle r, const char * text, int selected, int enabled) {
    Vector2 mouse = GetMousePosition();
    int hot = enabled && CheckCollisionPointRec(mouse, r);
    Color bg = selected ? (Color) {42, 112, 122, 255} : (enabled ? (hot ? (Color) {52, 64, 76, 255} : (Color) {34, 40, 48, 255}) : (Color) {29, 32, 37, 255});
    DrawRectangleRounded(r, 0.18f, 8, bg);
    DrawRectangleRoundedLines(r, 0.18f, 8, selected ? (Color) {116, 221, 211, 255} : (Color) {67, 75, 84, 255});
    int font_size = 14;
    int tw = MeasureText(text, font_size);
    DrawText(text, (int) (r.x + (r.width - (float) tw) * 0.5f), (int) (r.y + (r.height - (float) font_size) * 0.5f), font_size, enabled ? RAYWHITE : (Color) {110, 116, 122, 255});
    return hot && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static int ui_stepper(Rectangle r, const char * label, int * value, int step, int min_value, int max_value, int enabled) {
    DrawText(label, (int) r.x, (int) r.y, 14, (Color) {184, 191, 198, 255});
    Rectangle minus = {r.x, r.y + 20.0f, 34.0f, 28.0f};
    Rectangle value_rect = {r.x + 38.0f, r.y + 20.0f, r.width - 76.0f, 28.0f};
    Rectangle plus = {r.x + r.width - 34.0f, r.y + 20.0f, 34.0f, 28.0f};
    int changed = 0;
    if (ui_small_button(minus, "-", 0, enabled)) {
        *value -= step;
        if (*value < min_value) *value = min_value;
        changed = 1;
    }
    DrawRectangleRounded(value_rect, 0.14f, 8, (Color) {24, 28, 34, 255});
    DrawRectangleRoundedLines(value_rect, 0.14f, 8, (Color) {58, 66, 75, 255});
    char buf[64];
    snprintf(buf, sizeof(buf), "%d", *value);
    int tw = MeasureText(buf, 15);
    DrawText(buf, (int) (value_rect.x + (value_rect.width - (float) tw) * 0.5f), (int) value_rect.y + 7, 15, enabled ? RAYWHITE : (Color) {112, 118, 124, 255});
    if (ui_small_button(plus, "+", 0, enabled)) {
        *value += step;
        if (*value > max_value) *value = max_value;
        changed = 1;
    }
    return changed;
}

static void draw_texture_fit(Texture2D tex, Rectangle dst, Color tint) {
    if (tex.id == 0 || tex.width <= 0 || tex.height <= 0) {
        DrawRectangleRounded(dst, 0.04f, 8, (Color) {22, 25, 30, 255});
        DrawRectangleRoundedLines(dst, 0.04f, 8, (Color) {62, 70, 78, 255});
        return;
    }
    float scale = fminf(dst.width / (float) tex.width, dst.height / (float) tex.height);
    float w = (float) tex.width * scale;
    float h = (float) tex.height * scale;
    Rectangle target = {dst.x + (dst.width - w) * 0.5f, dst.y + (dst.height - h) * 0.5f, w, h};
    DrawTexturePro(tex, (Rectangle) {0, 0, (float) tex.width, (float) tex.height}, target, (Vector2) {0, 0}, 0.0f, tint);
}

static void draw_voxels(const display_voxels * voxels) {
    if (voxels->xyz == NULL || voxels->count <= 0 || voxels->resolution <= 0) {
        return;
    }
    float res = (float) voxels->resolution;
    float cube = 2.0f / res;
    Vector3 size = {cube, cube, cube};
    for (int64_t i = 0; i < voxels->count; ++i) {
        const int32_t * c = voxels->xyz + (size_t) i * 3u;
        Vector3 pos = {
            (((float) c[0] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[2] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[1] + 0.5f) / res - 0.5f) * 2.0f,
        };
        DrawCubeV(pos, size, (Color) {67, 177, 206, 225});
    }
}

static void draw_ground_grid(void) {
    const float floor_y = -VIEWER_PREVIEW_AABB_SIZE * 0.5f;
    rlPushMatrix();
    rlTranslatef(0.0f, floor_y, 0.0f);
    DrawGrid(18, 0.25f);
    rlPopMatrix();
}

static Camera3D make_arcball_camera(const viewer_ui_state * ui) {
    float radius_xz = cosf(ui->pitch) * ui->distance;
    Camera3D camera = {0};
    camera.position = (Vector3) {
        sinf(ui->yaw) * radius_xz,
        sinf(ui->pitch) * ui->distance,
        cosf(ui->yaw) * radius_xz,
    };
    camera.target = (Vector3) {0.0f, 0.0f, 0.0f};
    camera.up = (Vector3) {0.0f, 1.0f, 0.0f};
    camera.fovy = 43.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    return camera;
}

static void update_arcball(viewer_ui_state * ui, Rectangle viewport) {
    Vector2 mouse = GetMousePosition();
    if (CheckCollisionPointRec(mouse, viewport)) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 delta = GetMouseDelta();
            ui->yaw -= delta.x * 0.008f;
            ui->pitch -= delta.y * 0.008f;
            if (ui->pitch < -1.45f) ui->pitch = -1.45f;
            if (ui->pitch > 1.45f) ui->pitch = 1.45f;
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            ui->distance *= powf(0.90f, wheel);
            if (ui->distance < 1.3f) ui->distance = 1.3f;
            if (ui->distance > 8.0f) ui->distance = 8.0f;
        }
    }
}

static void format_step_speed(char * dst, size_t dst_size, int64_t step_us) {
    if (dst == NULL || dst_size == 0) {
        return;
    }
    double seconds = step_us <= 0 ? 0.0 : (double) step_us / 1000000.0;
    if (seconds > 0.0 && seconds < 1.0) {
        snprintf(dst, dst_size, "%.2fit/s", 1.0 / seconds);
    } else if (seconds > 0.0) {
        snprintf(dst, dst_size, "%.2fs/it", seconds);
    } else {
        copy_text(dst, dst_size, "--");
    }
}

static void draw_text_elided(const char * text, int x, int y, int font_size, int max_width, Color color) {
    if (text_is_empty(text) || max_width <= 0) {
        return;
    }
    char buf[512];
    copy_text(buf, sizeof(buf), text);
    if (MeasureText(buf, font_size) <= max_width) {
        DrawText(buf, x, y, font_size, color);
        return;
    }
    const char * ellipsis = "...";
    const int ellipsis_w = MeasureText(ellipsis, font_size);
    size_t n = strlen(buf);
    while (n > 0 && MeasureText(buf, font_size) + ellipsis_w > max_width) {
        buf[--n] = '\0';
    }
    if (n == 0) {
        return;
    }
    strncat(buf, ellipsis, sizeof(buf) - strlen(buf) - 1u);
    DrawText(buf, x, y, font_size, color);
}

static void draw_square_progress_bar(
    Rectangle r,
    float progress,
    int indeterminate,
    int running,
    int step,
    int steps,
    float time_s,
    int error_state) {
    const float gap = 3.0f;
    float block = 10.0f;
    int count = (int) floorf((r.width + gap) / (block + gap));
    if (count < 8) {
        count = 8;
        block = floorf((r.width - gap * (float) (count - 1)) / (float) count);
    }
    if (block < 3.0f) {
        return;
    }
    float total_w = (float) count * block + (float) (count - 1) * gap;
    float x0 = r.x + floorf((r.width - total_w) * 0.5f);
    float y0 = r.y + floorf((r.height - block) * 0.5f);

    int head = -1;
    int filled = 0;
    if (steps > 0) {
        if (step < 0) step = 0;
        if (step > steps) step = steps;
        filled = (int) floorf((float) step * (float) count / (float) steps);
        if (filled < 0) filled = 0;
        if (filled > count) filled = count;
        if (step > 0 && step < steps) {
            head = (int) ceilf((float) step * (float) count / (float) steps) - 1;
            if (head < 0) head = 0;
            if (head >= count) head = count - 1;
        }
    } else if (indeterminate && running) {
        head = (int) floorf(fmodf(time_s * 14.0f, (float) count));
    } else {
        if (progress < 0.0f) progress = 0.0f;
        if (progress > 1.0f) progress = 1.0f;
        filled = (int) floorf(progress * (float) count + 0.5f);
        if (filled < 0) filled = 0;
        if (filled > count) filled = count;
    }

    Color empty = {35, 42, 51, 235};
    Color done = error_state ? (Color) {255, 128, 112, 245} : (Color) {92, 185, 177, 245};
    Color bright = error_state ? (Color) {255, 176, 160, 255} : (Color) {145, 242, 225, 255};
    for (int i = 0; i < count; ++i) {
        Color c = empty;
        if (steps > 0 || !(indeterminate && running)) {
            if (i < filled) {
                c = done;
            }
            if (i == head) {
                c = bright;
            }
        } else {
            int d = head - i;
            if (d < 0) d += count;
            if (i == head) {
                c = bright;
            } else if (d > 0 && d <= 5) {
                unsigned char fade = (unsigned char) (220 - d * 24);
                c = error_state ? (Color) {255, 128, 112, fade} : (Color) {92, 185, 177, fade};
            }
        }
        Rectangle cell = {x0 + (float) i * (block + gap), y0, block, block};
        DrawRectangleRounded(cell, 0.18f, 4, c);
    }
}

static void draw_stage_overlay(Rectangle viewport, viewer_shared * shared, float time_s) {
    char stage[128];
    char detail[384];
    char error[512];
    char progress_label[128];
    char progress_detail[384];
    float progress;
    int indeterminate;
    int running;
    int progress_step;
    int progress_steps;
    int64_t progress_step_us;
    pthread_mutex_lock(&shared->mutex);
    copy_text(stage, sizeof(stage), shared->stage);
    copy_text(detail, sizeof(detail), shared->detail);
    copy_text(error, sizeof(error), shared->error);
    copy_text(progress_label, sizeof(progress_label), shared->progress_label);
    copy_text(progress_detail, sizeof(progress_detail), shared->progress_detail);
    progress = shared->progress;
    indeterminate = shared->indeterminate;
    running = shared->worker_running;
    progress_step = shared->progress_step;
    progress_steps = shared->progress_steps;
    progress_step_us = shared->progress_step_us;
    pthread_mutex_unlock(&shared->mutex);

    const int has_step = running && progress_steps > 0;
    float panel_w = viewport.width < 460.0f ? viewport.width - 36.0f : 430.0f;
    if (panel_w < 320.0f) panel_w = 320.0f;
    Rectangle panel = {viewport.x + viewport.width - panel_w - 20.0f, viewport.y + 18.0f, panel_w, 126.0f};
    DrawRectangleRounded(panel, 0.08f, 8, (Color) {15, 18, 23, 228});
    DrawRectangleRoundedLines(panel, 0.08f, 8, (Color) {72, 86, 96, 220});

    char badge[64];
    if (has_step) {
        snprintf(badge, sizeof(badge), "%02d/%02d", progress_step, progress_steps);
    } else if (running) {
        copy_text(badge, sizeof(badge), "working");
    } else {
        snprintf(badge, sizeof(badge), "%3d%%", (int) lrintf(fmaxf(0.0f, fminf(1.0f, progress)) * 100.0f));
    }
    int badge_w = MeasureText(badge, 18);
    DrawText(badge, (int) (panel.x + panel.width - 16.0f - (float) badge_w), (int) panel.y + 16, 18, (Color) {116, 221, 211, 255});
    draw_text_elided(
        text_is_empty(stage) ? "Ready" : stage,
        (int) panel.x + 16,
        (int) panel.y + 13,
        20,
        (int) panel.width - 44 - badge_w,
        RAYWHITE);

    const char * line = text_is_empty(error) ? (has_step && !text_is_empty(progress_label) ? progress_label : detail) : error;
    draw_text_elided(
        line,
        (int) panel.x + 16,
        (int) panel.y + 43,
        14,
        (int) panel.width - 32,
        text_is_empty(error) ? (Color) {183, 193, 202, 255} : (Color) {255, 128, 112, 255});

    draw_square_progress_bar(
        (Rectangle) {panel.x + 16.0f, panel.y + 68.0f, panel.width - 32.0f, 15.0f},
        progress,
        indeterminate,
        running,
        has_step ? progress_step : 0,
        has_step ? progress_steps : 0,
        time_s,
        !text_is_empty(error));

    char subline[512];
    if (has_step) {
        char speed[64];
        format_step_speed(speed, sizeof(speed), progress_step_us);
        snprintf(
            subline,
            sizeof(subline),
            "%s%s%s",
            speed,
            text_is_empty(progress_detail) ? "" : "  ",
            text_is_empty(progress_detail) ? "" : progress_detail);
    } else {
        copy_text(subline, sizeof(subline), text_is_empty(error) ? detail : error);
    }
    draw_text_elided(subline, (int) panel.x + 16, (int) panel.y + 99, 13, (int) panel.width - 32, (Color) {138, 151, 163, 255});
}

static void draw_view_options(Rectangle viewport, viewer_ui_state * ui) {
    Rectangle base = {viewport.x + 18.0f, viewport.y + 18.0f, 306.0f, 42.0f};
    DrawRectangleRounded(base, 0.10f, 8, (Color) {15, 18, 23, 220});
    float x = base.x + 8.0f;
    if (ui_small_button((Rectangle) {x, base.y + 7.0f, 82.0f, 28.0f}, "Wire", ui->wireframe, 1)) ui->wireframe = !ui->wireframe;
    x += 92.0f;
    if (ui_small_button((Rectangle) {x, base.y + 7.0f, 82.0f, 28.0f}, "Light", ui->lighting, 1)) ui->lighting = !ui->lighting;
    x += 92.0f;
    if (ui_small_button((Rectangle) {x, base.y + 7.0f, 82.0f, 28.0f}, "Grid", ui->grid, 1)) ui->grid = !ui->grid;
}

static void draw_left_panel(
    Rectangle panel,
    viewer_options * options,
    viewer_shared * shared,
    Texture2D image_tex,
    viewer_ui_state * ui,
    pthread_t * worker_thread,
    int * thread_active) {
    DrawRectangleRec(panel, (Color) {17, 20, 25, 255});
    DrawRectangle((int) (panel.x + panel.width - 1.0f), 0, 1, GetScreenHeight(), (Color) {44, 51, 60, 255});
    DrawText(VIEWER_APP_TITLE, (int) panel.x + 18, 18, 24, RAYWHITE);
    DrawText("image-to-3D workspace", (int) panel.x + 20, 48, 13, (Color) {144, 154, 164, 255});

    if (CheckCollisionPointRec(GetMousePosition(), panel)) {
        ui->panel_scroll -= GetMouseWheelMove() * 38.0f;
        if (ui->panel_scroll < 0.0f) ui->panel_scroll = 0.0f;
        if (ui->panel_scroll > 500.0f) ui->panel_scroll = 500.0f;
    }

    int worker_running = 0;
    int file_picker_running = 0;
    int bg_ready = 0;
    int mesh_ready = 0;
    int texture_ready = 0;
    int post_ready = 0;
    char current_image[VIEWER_MAX_PATH];
    pthread_mutex_lock(&shared->mutex);
    worker_running = shared->worker_running;
    file_picker_running = shared->file_picker_running;
    bg_ready = shared->bg_ready;
    mesh_ready = shared->mesh_ready;
    texture_ready = shared->texture_ready;
    post_ready = shared->postprocess_ready;
    copy_text(current_image, sizeof(current_image), shared->current_image_path);
    pthread_mutex_unlock(&shared->mutex);
    worker_running = worker_running || file_picker_running;

    char weights_issue[256];
    int core_weights_ready = viewer_weights_ready(options, 0, weights_issue, sizeof(weights_issue));
    int full_weights_ready = viewer_weights_ready(options, 1, NULL, 0);

    BeginScissorMode((int) panel.x, 66, (int) panel.width, GetScreenHeight() - 102);

    float bw = panel.width - 36.0f;
    float y = panel.y + 76.0f - ui->panel_scroll;
    DrawText("Weights", (int) panel.x + 18, (int) y, 15, (Color) {205, 211, 218, 255});
    y += 23.0f;
    draw_text_elided(
        core_weights_ready ? (text_is_empty(options->weights_dir) ? options->model_dir : options->weights_dir) : weights_issue,
        (int) panel.x + 20,
        (int) y,
        13,
        (int) bw - 4,
        core_weights_ready ? (Color) {139, 210, 190, 255} : (Color) {245, 151, 128, 255});
    y += 25.0f;

    Rectangle image_box = {panel.x + 18.0f, y, panel.width - 36.0f, 168.0f};
    DrawRectangleRounded(image_box, 0.05f, 8, (Color) {24, 28, 34, 255});
    DrawRectangleRoundedLines(image_box, 0.05f, 8, (Color) {70, 80, 90, 255});
    draw_texture_fit(image_tex, (Rectangle) {image_box.x + 8.0f, image_box.y + 8.0f, image_box.width - 16.0f, image_box.height - 16.0f}, WHITE);

    y = image_box.y + image_box.height + 18.0f;
    if (ui_button((Rectangle) {panel.x + 18.0f, y, bw, 42.0f}, "Remove Background", !worker_running && full_weights_ready && !text_is_empty(current_image))) {
        if (start_worker(shared, options, VIEWER_JOB_REMOVE_BG, current_image, worker_thread) && thread_active != NULL) *thread_active = 1;
    }
    y += 50.0f;
    if (ui_button((Rectangle) {panel.x + 18.0f, y, bw, 42.0f}, "Generate Mesh", !worker_running && core_weights_ready && !text_is_empty(current_image))) {
        if (start_worker(shared, options, VIEWER_JOB_MESH, current_image, worker_thread) && thread_active != NULL) *thread_active = 1;
    }
    y += 50.0f;
    if (ui_button((Rectangle) {panel.x + 18.0f, y, bw, 42.0f}, "Generate Texture", !worker_running && core_weights_ready && mesh_ready)) {
        if (start_worker(shared, options, VIEWER_JOB_TEXTURE, current_image, worker_thread) && thread_active != NULL) *thread_active = 1;
    }
    y += 50.0f;
    if (ui_button((Rectangle) {panel.x + 18.0f, y, bw, 42.0f}, "Postprocess Mesh", !worker_running && mesh_ready)) {
        if (start_worker(shared, options, VIEWER_JOB_POSTPROCESS, current_image, worker_thread) && thread_active != NULL) *thread_active = 1;
    }
    y += 58.0f;

    DrawText("Pipeline", (int) panel.x + 18, (int) y, 15, (Color) {205, 211, 218, 255});
    y += 24.0f;
    if (ui_small_button((Rectangle) {panel.x + 18.0f, y, (bw - 8.0f) * 0.5f, 30.0f}, "512", strcmp(options->pipeline_type, "512") == 0, !worker_running)) {
        copy_text(options->pipeline_type, sizeof(options->pipeline_type), "512");
        options->resolution = 512;
        options->cond_resolution = 512;
        options->sparse_resolution = 32;
    }
    if (ui_small_button((Rectangle) {panel.x + 18.0f + (bw + 8.0f) * 0.5f, y, (bw - 8.0f) * 0.5f, 30.0f}, "1024", strcmp(options->pipeline_type, "1024") == 0, !worker_running)) {
        copy_text(options->pipeline_type, sizeof(options->pipeline_type), "1024");
        options->resolution = 1024;
        options->cond_resolution = 1024;
        options->sparse_resolution = 64;
    }
    y += 44.0f;

    int steps_changed = ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Sparse steps", &options->sparse_structure_steps, 1, 1, 50, !worker_running);
    (void) steps_changed;
    y += 58.0f;
    ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Shape/texture steps", &options->structured_latent_steps, 1, 1, 50, !worker_running);
    y += 58.0f;
    ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Texture size", &options->texture_size, 512, 512, 4096, !worker_running);
    y += 58.0f;
    ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Device", &options->device, 1, 0, 16, !worker_running);
    y += 58.0f;
    int seed_i = (int) options->seed;
    if (ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Seed", &seed_i, 1, 0, 999999, !worker_running)) {
        options->seed = (uint32_t) seed_i;
    }
    y += 58.0f;
    int noise_seed_i = (int) options->noise_seed;
    if (ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Noise seed", &noise_seed_i, 1, 0, 999999, !worker_running)) {
        options->noise_seed = (uint32_t) noise_seed_i;
    }
    y += 60.0f;

    if (ui_small_button((Rectangle) {panel.x + 18.0f, y, (bw - 8.0f) * 0.5f, 30.0f}, "Cache", options->model_cache, !worker_running)) {
        options->model_cache = !options->model_cache;
    }
    if (ui_small_button((Rectangle) {panel.x + 18.0f + (bw + 8.0f) * 0.5f, y, (bw - 8.0f) * 0.5f, 30.0f}, "FlashAttn", !options->no_ggml_flash_attn, !worker_running)) {
        options->no_ggml_flash_attn = !options->no_ggml_flash_attn;
        options->use_ggml_flash_attn = !options->no_ggml_flash_attn;
    }
    y += 38.0f;
    if (ui_small_button((Rectangle) {panel.x + 18.0f, y, bw, 30.0f}, "Post no-simplify", options->mesh_postprocess_no_simplify, !worker_running)) {
        options->mesh_postprocess_no_simplify = !options->mesh_postprocess_no_simplify;
    }
    y += 40.0f;
    int target_k = options->mesh_postprocess_decimation_target / 1000;
    if (ui_stepper((Rectangle) {panel.x + 18.0f, y, bw, 48.0f}, "Post target K faces", &target_k, 250, 100, 2000, !worker_running)) {
        options->mesh_postprocess_decimation_target = target_k * 1000;
    }

    EndScissorMode();

    char status[128];
    snprintf(status, sizeof(status), "Weights %s  BG %s  Mesh %s  Tex %s  Post %s",
        core_weights_ready ? "ok" : "-",
        bg_ready ? "ok" : "-",
        mesh_ready ? "ok" : "-",
        texture_ready ? "ok" : "-",
        post_ready ? "ok" : "-");
    if (ui->panel_scroll > 1.0f) {
        DrawRectangleRounded((Rectangle) {panel.x + panel.width - 8.0f, 70.0f, 4.0f, GetScreenHeight() - 110.0f}, 0.5f, 4, (Color) {42, 49, 58, 180});
        float thumb_h = fmaxf(54.0f, (GetScreenHeight() - 110.0f) * 0.62f);
        float thumb_y = 70.0f + (GetScreenHeight() - 110.0f - thumb_h) * (ui->panel_scroll / 500.0f);
        DrawRectangleRounded((Rectangle) {panel.x + panel.width - 8.0f, thumb_y, 4.0f, thumb_h}, 0.5f, 4, (Color) {101, 205, 196, 210});
    }
    DrawText(status, (int) panel.x + 18, GetScreenHeight() - 28, 13, (Color) {139, 149, 158, 255});
}

static int viewer_image_extension(const char * path, char * ext, size_t ext_size) {
    if (text_is_empty(path) || ext == NULL || ext_size < 5u) {
        return 0;
    }
    const char * base = trellis_path_last_sep_const(path);
    base = base != NULL ? base + 1 : path;
    const char * dot = strrchr(base, '.');
    if (dot == NULL || dot[1] == '\0') {
        return 0;
    }
    size_t n = strlen(dot);
    if (n >= ext_size) {
        return 0;
    }
    for (size_t i = 0; i < n; ++i) {
        char c = dot[i];
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
        ext[i] = c;
    }
    ext[n] = '\0';
    return 1;
}

static Texture2D viewer_load_texture_from_path(const char * path, char * error, size_t error_size) {
    Texture2D texture = {0};
    if (error != NULL && error_size > 0) {
        error[0] = '\0';
    }
    if (text_is_empty(path) || !viewer_file_exists(path)) {
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "image file does not exist");
        }
        return texture;
    }

    char load_path[VIEWER_MAX_PATH];
    char ext[16];
    copy_text(load_path, sizeof(load_path), path);
    if (!viewer_image_extension(load_path, ext, sizeof(ext))) {
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "image extension is not supported");
        }
        return texture;
    }

    int delete_temp = 0;
    char temp_path[VIEWER_MAX_PATH];
    if (strcmp(ext, ".webp") == 0) {
        if (!trellis_make_temp_path(temp_path, sizeof(temp_path), "trellis_viewer_preview", ".png") ||
            !convert_webp_to_png(path, temp_path)) {
            if (error != NULL && error_size > 0) {
                snprintf(error, error_size, "webp preview conversion failed");
            }
            return texture;
        }
        copy_text(load_path, sizeof(load_path), temp_path);
        copy_text(ext, sizeof(ext), ".png");
        delete_temp = 1;
    }

    unsigned char * data = NULL;
    int data_size = 0;
    if (!viewer_read_file_bytes(load_path, &data, &data_size)) {
        if (delete_temp) {
            trellis_unlink(load_path);
        }
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "could not read image file");
        }
        return texture;
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load_from_memory(data, data_size, &width, &height, &comp, 4);
    free(data);
    if (delete_temp) {
        trellis_unlink(load_path);
    }
    if (rgba == NULL || width <= 0 || height <= 0) {
        stbi_image_free(rgba);
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "could not decode image preview");
        }
        return texture;
    }

    Image image = {0};
    image.data = rgba;
    image.width = width;
    image.height = height;
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    texture = LoadTextureFromImage(image);
    stbi_image_free(rgba);
    if (texture.id == 0) {
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "could not upload image preview");
        }
        return texture;
    }
    SetTextureFilter(texture, TEXTURE_FILTER_BILINEAR);
    return texture;
}

static Texture2D reload_texture_if_changed(
    Texture2D current,
    char * loaded_path,
    const char * desired_path,
    char * error,
    size_t error_size) {
    if (error != NULL && error_size > 0) {
        error[0] = '\0';
    }
    if (desired_path == NULL || strcmp(loaded_path, desired_path) == 0) {
        return current;
    }
    if (current.id != 0) {
        UnloadTexture(current);
        current.id = 0;
    }
    loaded_path[0] = '\0';
    if (!text_is_empty(desired_path)) {
        Texture2D next = viewer_load_texture_from_path(desired_path, error, error_size);
        copy_text(loaded_path, VIEWER_MAX_PATH, desired_path);
        if (next.id != 0) {
            return next;
        }
    }
    return current;
}

int main(int argc, char ** argv) {
    viewer_options options;
    if (!parse_options(argc, argv, &options)) {
        usage(stderr, argv[0]);
        return 2;
    }
    mkdir_p_simple(options.out_dir);

    viewer_shared shared;
    memset(&shared, 0, sizeof(shared));
    pthread_mutex_init(&shared.mutex, NULL);
    copy_text(shared.current_image_path, sizeof(shared.current_image_path), options.image_path);
    char initial_issue[256];
    if (viewer_weights_ready(&options, 1, initial_issue, sizeof(initial_issue))) {
        copy_text(shared.stage, sizeof(shared.stage), "Ready");
        copy_text(shared.detail, sizeof(shared.detail), text_is_empty(options.weights_dir) ? options.model_dir : options.weights_dir);
    } else {
        copy_text(shared.stage, sizeof(shared.stage), "Weights warning");
        copy_text(shared.detail, sizeof(shared.detail), initial_issue);
    }

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(options.width, options.height, VIEWER_APP_TITLE);
    SetTargetFPS(60);

    viewer_ui_state ui;
    memset(&ui, 0, sizeof(ui));
    ui.distance = 3.4f;
    ui.pitch = 0.28f;
    ui.yaw = 0.55f;
    ui.lighting = 1;
    ui.grid = 1;

    Texture2D image_tex = {0};
    Texture2D uv_tex = {0};
    char loaded_image_path[VIEWER_MAX_PATH] = {0};
    char loaded_uv_path[VIEWER_MAX_PATH] = {0};
    display_mesh mesh_display;
    display_voxels voxel_display;
    memset(&mesh_display, 0, sizeof(mesh_display));
    memset(&voxel_display, 0, sizeof(voxel_display));

    pthread_t worker_thread;
    int have_worker_thread = 0;
    int last_done_token = 0;

    while (!WindowShouldClose()) {
        pthread_mutex_lock(&shared.mutex);
        char desired_image[VIEWER_MAX_PATH];
        char desired_uv[VIEWER_MAX_PATH];
        int done_token = shared.worker_done_token;
        int running = shared.worker_running;
        int picker_running = shared.file_picker_running;
        copy_text(desired_image, sizeof(desired_image), shared.current_image_path);
        copy_text(desired_uv, sizeof(desired_uv), shared.uv_texture_path);
        pthread_mutex_unlock(&shared.mutex);
        int input_busy = running || picker_running;

        if (have_worker_thread && done_token != last_done_token && !running) {
            pthread_join(worker_thread, NULL);
            have_worker_thread = 0;
            last_done_token = done_token;
        }

        int screen_w = GetScreenWidth();
        int screen_h = GetScreenHeight();
        float panel_w = floorf((float) screen_w * 0.25f);
        if (panel_w < 320.0f) panel_w = 320.0f;
        if (panel_w > 430.0f) panel_w = 430.0f;
        Rectangle panel = {0.0f, 0.0f, panel_w, (float) screen_h};
        Rectangle viewport = {panel_w, 0.0f, (float) screen_w - panel_w, (float) screen_h};
        Rectangle image_drop_box = {panel.x + 18.0f, panel.y + 124.0f - ui.panel_scroll, panel.width - 36.0f, 168.0f};
        int image_box_hot = CheckCollisionPointRec(GetMousePosition(), image_drop_box);

        char selected_path[VIEWER_MAX_PATH];
        selected_path[0] = '\0';
        pthread_mutex_lock(&shared.mutex);
        if (shared.selected_image_pending) {
            copy_text(selected_path, sizeof(selected_path), shared.selected_image_path);
            shared.selected_image_path[0] = '\0';
            shared.selected_image_pending = 0;
        }
        pthread_mutex_unlock(&shared.mutex);
        if (!text_is_empty(selected_path) && !running) {
            set_candidate_image(&options, &shared, selected_path, &mesh_display, &voxel_display);
            copy_text(desired_image, sizeof(desired_image), selected_path);
        }

        if (!input_busy && image_box_hot && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            start_file_picker(&shared, VIEWER_PICKER_IMAGE);
        }

        if (IsFileDropped()) {
            FilePathList dropped = LoadDroppedFiles();
            if (!input_busy && dropped.count > 0 && dropped.paths[0] != NULL) {
                if (!viewer_dir_exists(dropped.paths[0]) && image_box_hot) {
                    set_candidate_image(&options, &shared, dropped.paths[0], &mesh_display, &voxel_display);
                    copy_text(desired_image, sizeof(desired_image), dropped.paths[0]);
                }
            }
            UnloadDroppedFiles(dropped);
        }

        char image_load_error[192];
        char uv_load_error[192];
        image_tex = reload_texture_if_changed(image_tex, loaded_image_path, desired_image, image_load_error, sizeof(image_load_error));
        uv_tex = reload_texture_if_changed(uv_tex, loaded_uv_path, desired_uv, uv_load_error, sizeof(uv_load_error));
        if (!text_is_empty(image_load_error)) {
            pthread_mutex_lock(&shared.mutex);
            copy_text(shared.stage, sizeof(shared.stage), "Image preview failed");
            copy_text(shared.detail, sizeof(shared.detail), image_load_error);
            pthread_mutex_unlock(&shared.mutex);
        } else if (!text_is_empty(uv_load_error)) {
            pthread_mutex_lock(&shared.mutex);
            copy_text(shared.stage, sizeof(shared.stage), "Texture preview failed");
            copy_text(shared.detail, sizeof(shared.detail), uv_load_error);
            pthread_mutex_unlock(&shared.mutex);
        }
        copy_snapshot_to_display(&shared, &mesh_display, &voxel_display, ui.lighting);
        update_arcball(&ui, viewport);

        BeginDrawing();
        ClearBackground((Color) {10, 12, 16, 255});

        BeginScissorMode((int) viewport.x, (int) viewport.y, (int) viewport.width, (int) viewport.height);
        DrawRectangleRec(viewport, (Color) {12, 15, 20, 255});
        Camera3D camera = make_arcball_camera(&ui);
        BeginMode3D(camera);
        if (ui.grid) {
            draw_ground_grid();
        }
        DrawCubeWiresV(
            (Vector3) {0, 0, 0},
            (Vector3) {VIEWER_PREVIEW_AABB_SIZE, VIEWER_PREVIEW_AABB_SIZE, VIEWER_PREVIEW_AABB_SIZE},
            (Color) {83, 94, 106, 135});
        if (voxel_display.xyz != NULL) {
            draw_voxels(&voxel_display);
        }
        if (mesh_display.model_count > 0) {
            rlDisableBackfaceCulling();
            for (int i = 0; i < mesh_display.model_count; ++i) {
                DrawModel(mesh_display.models[i], (Vector3) {0, 0, 0}, 1.0f, WHITE);
                if (ui.wireframe) {
                    DrawModelWires(mesh_display.models[i], (Vector3) {0, 0, 0}, 1.002f, (Color) {16, 22, 28, 210});
                }
            }
            rlEnableBackfaceCulling();
        }
        EndMode3D();
        EndScissorMode();

        draw_view_options(viewport, &ui);
        draw_stage_overlay(viewport, &shared, (float) GetTime());

        if (uv_tex.id != 0) {
            Rectangle uv_panel = {viewport.x + viewport.width - 286.0f, viewport.y + viewport.height - 286.0f, 266.0f, 266.0f};
            DrawRectangleRounded(uv_panel, 0.05f, 8, (Color) {15, 18, 23, 228});
            DrawRectangleRoundedLines(uv_panel, 0.05f, 8, (Color) {75, 86, 96, 230});
            DrawText("UV base color", (int) uv_panel.x + 12, (int) uv_panel.y + 10, 15, RAYWHITE);
            draw_texture_fit(uv_tex, (Rectangle) {uv_panel.x + 12.0f, uv_panel.y + 34.0f, uv_panel.width - 24.0f, uv_panel.height - 46.0f}, WHITE);
        }

        draw_left_panel(panel, &options, &shared, image_tex, &ui, &worker_thread, &have_worker_thread);
        pthread_mutex_lock(&shared.mutex);
        have_worker_thread = have_worker_thread || shared.worker_running;
        pthread_mutex_unlock(&shared.mutex);

        if (mesh_display.model_count > 0) {
            char stats[160];
            snprintf(stats, sizeof(stats), "full mesh tris %d / chunks %d", mesh_display.triangle_count, mesh_display.model_count);
            DrawText(stats, (int) viewport.x + 24, screen_h - 30, 14, (Color) {151, 162, 172, 255});
        } else if (voxel_display.xyz != NULL) {
            char stats[160];
            snprintf(stats, sizeof(stats), "preview voxels %lld / res %d", (long long) voxel_display.count, voxel_display.resolution);
            DrawText(stats, (int) viewport.x + 24, screen_h - 30, 14, (Color) {151, 162, 172, 255});
        }

        EndDrawing();
    }

    pthread_mutex_lock(&shared.mutex);
    shared.close_requested = 1;
    int still_running = shared.worker_running;
    pthread_mutex_unlock(&shared.mutex);
    if (have_worker_thread || still_running) {
        pthread_join(worker_thread, NULL);
    }

    display_mesh_free(&mesh_display);
    display_voxels_free(&voxel_display);
    if (image_tex.id != 0) UnloadTexture(image_tex);
    if (uv_tex.id != 0) UnloadTexture(uv_tex);
    viewer_snapshot_free(&shared.snapshot);
    viewer_artifacts_free(&shared.artifacts);
    pthread_mutex_destroy(&shared.mutex);
    CloseWindow();
    return 0;
}
