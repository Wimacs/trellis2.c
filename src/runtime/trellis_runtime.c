#define _POSIX_C_SOURCE 200809L

#include "trellis.h"

#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define TRELLIS_ISATTY _isatty
#define TRELLIS_FILENO _fileno
#else
#include <unistd.h>
#define TRELLIS_ISATTY isatty
#define TRELLIS_FILENO fileno
#endif

const char * trellis_status_string(trellis_status status) {
    switch (status) {
        case TRELLIS_STATUS_OK: return "ok";
        case TRELLIS_STATUS_ERROR: return "error";
        case TRELLIS_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case TRELLIS_STATUS_IO_ERROR: return "io error";
        case TRELLIS_STATUS_PARSE_ERROR: return "parse error";
        case TRELLIS_STATUS_OUT_OF_MEMORY: return "out of memory";
        case TRELLIS_STATUS_CUDA_UNAVAILABLE: return "backend unavailable";
        case TRELLIS_STATUS_NOT_FOUND: return "not found";
        case TRELLIS_STATUS_NOT_IMPLEMENTED: return "not implemented";
        default: return "unknown";
    }
}

const char * trellis_backend_kind_name(trellis_backend_kind kind) {
    switch (kind) {
        case TRELLIS_BACKEND_CPU: return "cpu";
        case TRELLIS_BACKEND_CUDA: return "cuda";
        case TRELLIS_BACKEND_VULKAN: return "vulkan";
        default: return "unknown";
    }
}

trellis_status trellis_backend_kind_from_name(const char * name, trellis_backend_kind * kind_out) {
    if (name == NULL || kind_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (strcmp(name, "cpu") == 0) {
        *kind_out = TRELLIS_BACKEND_CPU;
        return TRELLIS_STATUS_OK;
    }
    if (strcmp(name, "cuda") == 0) {
        *kind_out = TRELLIS_BACKEND_CUDA;
        return TRELLIS_STATUS_OK;
    }
    if (strcmp(name, "vulkan") == 0 || strcmp(name, "vk") == 0) {
        *kind_out = TRELLIS_BACKEND_VULKAN;
        return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_INVALID_ARGUMENT;
}

trellis_status trellis_backend_init(trellis_backend_context * ctx, trellis_backend_kind kind, int device) {
    if (ctx == NULL || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    ctx->backend = NULL;
    ctx->kind = kind;
    ctx->device = device;

    ggml_backend_load_all();

    switch (kind) {
        case TRELLIS_BACKEND_CPU:
            ctx->backend = ggml_backend_cpu_init();
            break;
        case TRELLIS_BACKEND_CUDA:
#ifdef GGML_USE_CUDA
            ctx->backend = ggml_backend_cuda_init(device);
#else
            return TRELLIS_STATUS_CUDA_UNAVAILABLE;
#endif
            break;
        case TRELLIS_BACKEND_VULKAN:
#ifdef GGML_USE_VULKAN
            ctx->backend = ggml_backend_vk_init((size_t) device);
#else
            return TRELLIS_STATUS_CUDA_UNAVAILABLE;
#endif
            break;
        default:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return ctx->backend == NULL ? TRELLIS_STATUS_CUDA_UNAVAILABLE : TRELLIS_STATUS_OK;
}

void trellis_backend_free(trellis_backend_context * ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->backend != NULL) {
        ggml_backend_free(ctx->backend);
    }
    ctx->backend = NULL;
    ctx->kind = TRELLIS_BACKEND_CPU;
    ctx->device = -1;
}

ggml_gallocr_t trellis_backend_new_graph_allocator(const trellis_backend_context * ctx) {
    if (ctx == NULL || ctx->backend == NULL) {
        return NULL;
    }
    return ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
}

trellis_status trellis_backend_compute_graph(const trellis_backend_context * ctx, struct ggml_cgraph * graph) {
    if (ctx == NULL || ctx->backend == NULL || graph == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    enum ggml_status status = ggml_backend_graph_compute(ctx->backend, graph);
    return status == GGML_STATUS_SUCCESS ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

trellis_status trellis_cuda_init(trellis_cuda_context * ctx, int device) {
    return trellis_backend_init(ctx, TRELLIS_BACKEND_CUDA, device);
}

void trellis_cuda_free(trellis_cuda_context * ctx) {
    trellis_backend_free(ctx);
}

ggml_gallocr_t trellis_cuda_new_graph_allocator(const trellis_cuda_context * ctx) {
    return trellis_backend_new_graph_allocator(ctx);
}

trellis_status trellis_cuda_compute_graph(const trellis_cuda_context * ctx, struct ggml_cgraph * graph) {
    return trellis_backend_compute_graph(ctx, graph);
}

trellis_status trellis_make_model_path(
    const char * model_dir,
    const char * relative_path,
    char * dst,
    size_t dst_size) {
    if (model_dir == NULL || relative_path == NULL || dst == NULL || dst_size == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int n = snprintf(dst, dst_size, "%s/%s", model_dir, relative_path);
    if (n < 0 || (size_t) n >= dst_size) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static int g_trellis_verbose = 0;

void trellis_set_verbose(int verbose) {
    g_trellis_verbose = verbose != 0;
}

static const char * log_level_name(trellis_log_level level) {
    switch (level) {
        case TRELLIS_LOG_DEBUG: return "DEBUG";
        case TRELLIS_LOG_INFO: return "INFO ";
        case TRELLIS_LOG_WARN: return "WARN ";
        case TRELLIS_LOG_ERROR: return "ERROR";
        default: return "?????";
    }
}

void trellis_log(trellis_log_level level, const char * fmt, ...) {
    if (level == TRELLIS_LOG_DEBUG && !g_trellis_verbose) {
        return;
    }
    FILE * out = level == TRELLIS_LOG_ERROR ? stderr : stdout;
    fprintf(out, "[%s] ", log_level_name(level));
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    size_t n = fmt == NULL ? 0 : strlen(fmt);
    if (n == 0 || fmt[n - 1] != '\n') {
        fputc('\n', out);
    }
    fflush(out);
}

static void build_progress_bar(char * dst, size_t dst_size, int step, int steps, char progress_char, int show_head) {
    if (dst == NULL || dst_size == 0) {
        return;
    }
    const int width = 50;
    int current = 0;
    if (steps > 0) {
        current = (int) ((double) step * (double) width / (double) steps);
        if (current < 0) current = 0;
        if (current >= width) current = width - 1;
    }

    size_t pos = 0;
    if (pos < dst_size) dst[pos++] = ' ';
    if (pos < dst_size) dst[pos++] = ' ';
    if (pos < dst_size) dst[pos++] = '|';
    for (int i = 0; i < width && pos + 1 < dst_size; ++i) {
        if (i > current) {
            dst[pos++] = ' ';
        } else if (show_head && i == current && i != width - 1) {
            dst[pos++] = '>';
        } else {
            dst[pos++] = progress_char;
        }
    }
    if (pos < dst_size) dst[pos++] = '|';
    if (pos < dst_size) {
        dst[pos] = '\0';
    } else {
        dst[dst_size - 1] = '\0';
    }
}

static double bytes_to_mib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static trellis_progress_step_callback g_progress_step_callback = NULL;
static void * g_progress_step_user_data = NULL;

void trellis_set_progress_step_callback(
    trellis_progress_step_callback callback,
    void * user_data) {
    g_progress_step_callback = callback;
    g_progress_step_user_data = user_data;
}

static void print_progress_line(
    const char * label,
    int step,
    int steps,
    const char * speed_text,
    const char * detail,
    char progress_char,
    int show_head) {
    if (step <= 0 || steps <= 0) {
        return;
    }
    char bar[64];
    build_progress_bar(bar, sizeof(bar), step, steps, progress_char, show_head);
    const int done = step >= steps;
    const int interactive = TRELLIS_ISATTY(TRELLIS_FILENO(stdout));
    if (interactive) {
        printf("\r%s %d/%d - %s", bar, step, steps, speed_text == NULL ? "" : speed_text);
        if (label != NULL && label[0] != '\0') {
            printf(" - %s", label);
        }
        if (detail != NULL && detail[0] != '\0') {
            printf(" - %s", detail);
        }
        printf("\033[K%s", done ? "\n" : "");
    } else if (done || step == 1 || (steps >= 10 && (step % (steps / 10)) == 0)) {
        printf("%s %d/%d - %s", label == NULL ? "progress" : label, step, steps, speed_text == NULL ? "" : speed_text);
        if (detail != NULL && detail[0] != '\0') {
            printf(" - %s", detail);
        }
        printf("\n");
    }
    fflush(stdout);
}

void trellis_progress_bytes(
    const char * label,
    int step,
    int steps,
    uint64_t bytes_processed,
    uint64_t bytes_total,
    int64_t elapsed_us) {
    char speed_text[96];
    double seconds = elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0;
    double mib_s = seconds <= 0.0 ? 0.0 : bytes_to_mib(bytes_processed) / seconds;
    if (mib_s >= 1024.0) {
        snprintf(speed_text, sizeof(speed_text), "%.2fGB/s", mib_s / 1024.0);
    } else {
        snprintf(speed_text, sizeof(speed_text), "%.2fMB/s", mib_s);
    }
    char detail[96];
    if (bytes_total > 0) {
        snprintf(detail, sizeof(detail), "%.1f/%.1f MiB", bytes_to_mib(bytes_processed), bytes_to_mib(bytes_total));
    } else {
        snprintf(detail, sizeof(detail), "%.1f MiB", bytes_to_mib(bytes_processed));
    }
    print_progress_line(label, step, steps, speed_text, detail, '#', 0);
}

void trellis_progress_steps(
    const char * label,
    int step,
    int steps,
    int64_t step_us,
    const char * detail) {
    if (g_progress_step_callback != NULL) {
        trellis_progress_step_event event;
        event.label = label;
        event.step = step;
        event.steps = steps;
        event.step_us = step_us;
        event.detail = detail;
        g_progress_step_callback(&event, g_progress_step_user_data);
    }

    char speed_text[96];
    double seconds = step_us <= 0 ? 0.0 : (double) step_us / 1000000.0;
    if (seconds > 0.0 && seconds < 1.0) {
        snprintf(speed_text, sizeof(speed_text), "%.2fit/s", 1.0 / seconds);
    } else {
        snprintf(speed_text, sizeof(speed_text), "%.2fs/it", seconds);
    }
    print_progress_line(label, step, steps, speed_text, detail, '=', 1);
}

typedef struct model_load_progress_context {
    const char * label;
    int64_t start_us;
} model_load_progress_context;

static uint64_t safetensors_total_bytes(const trellis_safetensors * st) {
    if (st == NULL) {
        return 0;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < st->n_tensors; ++i) {
        if (st->tensors[i].data_end >= st->tensors[i].data_begin) {
            total += st->tensors[i].data_end - st->tensors[i].data_begin;
        }
    }
    return total;
}

static void model_load_progress(
    const trellis_tensor_store_load_progress * progress,
    void * user_data) {
    model_load_progress_context * ctx = (model_load_progress_context *) user_data;
    if (progress == NULL || ctx == NULL || progress->tensor_count == 0 || progress->tensor_index == 0) {
        return;
    }
    const int64_t now_us = ggml_time_us();
    trellis_progress_bytes(
        ctx->label,
        (int) progress->tensor_index,
        (int) progress->tensor_count,
        progress->bytes_loaded,
        progress->total_bytes,
        now_us - ctx->start_us);
}

static int trellis_load_tensor_store_impl(
    const trellis_backend_context * backend,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    bool preserve_16bit_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_model_load_result * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (backend == NULL || path == NULL || store == NULL) {
        TRELLIS_ERROR("%s: invalid load request", label == NULL ? "model" : label);
        return 0;
    }
    memset(store, 0, sizeof(*store));

    TRELLIS_INFO("%s: loading '%s'", label == NULL ? "model" : label, path);
    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("%s: open failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        return 0;
    }

    const size_t graph_tensors = st.n_tensors + tensor_slack;
    const uint64_t total_bytes = safetensors_total_bytes(&st);
    TRELLIS_INFO(
        "%s: checkpoint tensors=%zu data=%.1f MiB",
        label == NULL ? "model" : label,
        st.n_tensors,
        (double) total_bytes / (1024.0 * 1024.0));
    trellis_safetensors_close(&st);

    status = trellis_tensor_store_init(store, graph_tensors, 0);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("%s: tensor store init failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        return 0;
    }

    size_t loaded = 0;
    model_load_progress_context progress;
    progress.label = label == NULL ? "model" : label;
    progress.start_us = ggml_time_us();
    if (preserve_16bit_weights) {
        status = trellis_tensor_store_load_safetensors_ex(
            store,
            backend,
            path,
            transpose_linear_weights,
            &loaded,
            model_load_progress,
            &progress);
    } else {
        status = trellis_tensor_store_load_safetensors_f32_ex(
            store,
            backend,
            path,
            transpose_linear_weights,
            &loaded,
            model_load_progress,
            &progress);
    }
    const int64_t elapsed_us = ggml_time_us() - progress.start_us;
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("%s: load failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        trellis_tensor_store_free(store);
        return 0;
    }

    if (result != NULL) {
        result->tensors = loaded;
        result->bytes = total_bytes;
        result->seconds = elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0;
    }
    TRELLIS_INFO(
        "%s: loaded %zu tensors to %s in %.2fs%s",
        label == NULL ? "model" : label,
        loaded,
        trellis_backend_kind_name(backend->kind),
        elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0,
        preserve_16bit_weights ? " (native 16-bit where available)" : "");
    return 1;
}

int trellis_load_tensor_store_f32(
    const trellis_backend_context * backend,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_model_load_result * result) {
    return trellis_load_tensor_store_impl(
        backend,
        label,
        path,
        transpose_linear_weights,
        false,
        tensor_slack,
        store,
        result);
}

int trellis_load_tensor_store(
    const trellis_backend_context * backend,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_model_load_result * result) {
    return trellis_load_tensor_store_impl(
        backend,
        label,
        path,
        transpose_linear_weights,
        true,
        tensor_slack,
        store,
        result);
}
