#include "trellis.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char * trellis_strdup(const char * s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char * out = (char *) malloc(n + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n + 1);
    return out;
}

static trellis_status store_append(trellis_tensor_store * store, const char * name, struct ggml_tensor * tensor) {
    if (store == NULL || name == NULL || tensor == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (store->n_entries == store->capacity) {
        size_t next_capacity = store->capacity == 0 ? 128 : store->capacity * 2;
        trellis_tensor_store_entry * next = (trellis_tensor_store_entry *) realloc(
            store->entries,
            next_capacity * sizeof(trellis_tensor_store_entry));
        if (next == NULL) {
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        store->entries = next;
        store->capacity = next_capacity;
    }
    char * owned_name = trellis_strdup(name);
    if (owned_name == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    store->entries[store->n_entries].name = owned_name;
    store->entries[store->n_entries].tensor = tensor;
    store->n_entries += 1;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tensor_store_init(
    trellis_tensor_store * store,
    size_t graph_tensors,
    size_t tensor_data_bytes) {
    if (store == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(store, 0, sizeof(*store));
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_tensors + tensor_data_bytes + 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    store->ctx = ggml_init(params);
    if (store->ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return TRELLIS_STATUS_OK;
}

void trellis_tensor_store_free(trellis_tensor_store * store) {
    if (store == NULL) {
        return;
    }
    if (store->buffer != NULL) {
        ggml_backend_buffer_free(store->buffer);
    }
    for (size_t i = 0; i < store->n_entries; ++i) {
        free(store->entries[i].name);
    }
    free(store->entries);
    if (store->ctx != NULL) {
        ggml_free(store->ctx);
    }
    memset(store, 0, sizeof(*store));
}

const struct ggml_tensor * trellis_tensor_store_get_const(
    const trellis_tensor_store * store,
    const char * name) {
    if (store == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < store->n_entries; ++i) {
        if (strcmp(store->entries[i].name, name) == 0) {
            return store->entries[i].tensor;
        }
    }
    return NULL;
}

struct ggml_tensor * trellis_tensor_store_get(
    trellis_tensor_store * store,
    const char * name) {
    return (struct ggml_tensor *) trellis_tensor_store_get_const(store, name);
}

static int is_linear_weight_name(const char * name) {
    if (name == NULL) {
        return 0;
    }
    size_t n = strlen(name);
    return n >= 7 && strcmp(name + n - 7, ".weight") == 0;
}

static trellis_status make_tensor_for_meta(
    trellis_tensor_store * store,
    const trellis_safetensor_meta * meta,
    bool transpose_linear_weights,
    struct ggml_tensor ** out) {
    if (store == NULL || meta == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;

    int64_t ne[TRELLIS_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    if (transpose_linear_weights && meta->n_dims == 2 && is_linear_weight_name(meta->name)) {
        ne[0] = meta->shape[1];
        ne[1] = meta->shape[0];
    } else {
        for (int i = 0; i < meta->n_dims; ++i) {
            ne[i] = meta->shape[meta->n_dims - 1 - i];
        }
    }

    struct ggml_tensor * tensor = NULL;
    switch (meta->n_dims) {
        case 1:
            tensor = ggml_new_tensor_1d(store->ctx, GGML_TYPE_F32, ne[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(store->ctx, GGML_TYPE_F32, ne[0], ne[1]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(store->ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
            break;
        case 4:
            tensor = ggml_new_tensor_4d(store->ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
            break;
        case 5:
            tensor = ggml_new_tensor_4d(store->ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3] * ne[4]);
            break;
        default:
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    if (tensor == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_set_name(tensor, meta->name);
    *out = tensor;
    return TRELLIS_STATUS_OK;
}

static trellis_status set_tensor_data_from_meta(
    const trellis_safetensors * st,
    const trellis_safetensor_meta * meta,
    struct ggml_tensor * tensor,
    bool transpose_linear_weights) {
    if (st == NULL || meta == NULL || tensor == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint64_t n64 = trellis_safetensor_nelements(meta);
    if (n64 > (uint64_t) SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    size_t n = (size_t) n64;
    float * tmp = (float *) malloc(n * sizeof(float));
    if (tmp == NULL && n != 0) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = trellis_safetensors_read_f32(st, meta, tmp, n);
    if (status != TRELLIS_STATUS_OK) {
        free(tmp);
        return status;
    }

    (void) transpose_linear_weights;
    ggml_backend_tensor_set(tensor, tmp, 0, ggml_nbytes(tensor));
    free(tmp);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tensor_store_load_safetensors_f32(
    trellis_tensor_store * store,
    const trellis_cuda_context * cuda,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors) {
    if (store == NULL || store->ctx == NULL || cuda == NULL || cuda->backend == NULL || safetensors_path == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (store->n_entries != 0 || store->buffer != NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < st.n_tensors; ++i) {
        struct ggml_tensor * tensor = NULL;
        status = make_tensor_for_meta(store, &st.tensors[i], transpose_linear_weights, &tensor);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
        status = store_append(store, st.tensors[i].name, tensor);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
    }

    store->buffer = ggml_backend_alloc_ctx_tensors(store->ctx, cuda->backend);
    if (store->buffer == NULL) {
        trellis_safetensors_close(&st);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < st.n_tensors; ++i) {
        struct ggml_tensor * tensor = store->entries[i].tensor;
        status = set_tensor_data_from_meta(&st, &st.tensors[i], tensor, transpose_linear_weights);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
    }

    if (loaded_tensors != NULL) {
        *loaded_tensors = st.n_tensors;
    }
    trellis_safetensors_close(&st);
    return TRELLIS_STATUS_OK;
}
