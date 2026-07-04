#include "trellis_tool_model.h"

#include "ggml.h"
#include "trellis_tool_cli.h"

#include <string.h>

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
    trellis_tool_progress_bytes(
        ctx->label,
        (int) progress->tensor_index,
        (int) progress->tensor_count,
        progress->bytes_loaded,
        progress->total_bytes,
        now_us - ctx->start_us);
}

int trellis_tool_load_tensor_store_f32(
    const trellis_cuda_context * cuda,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_tool_model_load_result * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (cuda == NULL || path == NULL || store == NULL) {
        TRELLIS_TOOL_ERROR("%s: invalid load request", label == NULL ? "model" : label);
        return 0;
    }
    memset(store, 0, sizeof(*store));

    TRELLIS_TOOL_INFO("%s: loading '%s'", label == NULL ? "model" : label, path);
    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("%s: open failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        return 0;
    }

    const size_t graph_tensors = st.n_tensors + tensor_slack;
    const uint64_t total_bytes = safetensors_total_bytes(&st);
    TRELLIS_TOOL_INFO(
        "%s: checkpoint tensors=%zu data=%.1f MiB",
        label == NULL ? "model" : label,
        st.n_tensors,
        (double) total_bytes / (1024.0 * 1024.0));
    trellis_safetensors_close(&st);

    status = trellis_tensor_store_init(store, graph_tensors, 0);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("%s: tensor store init failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        return 0;
    }

    size_t loaded = 0;
    model_load_progress_context progress;
    progress.label = label == NULL ? "model" : label;
    progress.start_us = ggml_time_us();
    status = trellis_tensor_store_load_safetensors_f32_ex(
        store,
        cuda,
        path,
        transpose_linear_weights,
        &loaded,
        model_load_progress,
        &progress);
    const int64_t elapsed_us = ggml_time_us() - progress.start_us;
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_TOOL_ERROR("%s: load failed: %s", label == NULL ? "model" : label, trellis_status_string(status));
        trellis_tensor_store_free(store);
        return 0;
    }

    if (result != NULL) {
        result->tensors = loaded;
        result->bytes = total_bytes;
        result->seconds = elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0;
    }
    TRELLIS_TOOL_INFO(
        "%s: loaded %zu tensors to CUDA in %.2fs",
        label == NULL ? "model" : label,
        loaded,
        elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0);
    return 1;
}
