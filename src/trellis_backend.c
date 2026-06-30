#include "trellis.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdlib.h>

const char * trellis_status_string(trellis_status status) {
    switch (status) {
        case TRELLIS_STATUS_OK: return "ok";
        case TRELLIS_STATUS_ERROR: return "error";
        case TRELLIS_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case TRELLIS_STATUS_IO_ERROR: return "io error";
        case TRELLIS_STATUS_PARSE_ERROR: return "parse error";
        case TRELLIS_STATUS_OUT_OF_MEMORY: return "out of memory";
        case TRELLIS_STATUS_CUDA_UNAVAILABLE: return "cuda unavailable";
        case TRELLIS_STATUS_NOT_FOUND: return "not found";
        case TRELLIS_STATUS_NOT_IMPLEMENTED: return "not implemented";
        default: return "unknown";
    }
}

trellis_status trellis_cuda_init(trellis_cuda_context * ctx, int device) {
    if (ctx == NULL || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    ctx->backend = NULL;
    ctx->device = device;

#ifdef GGML_USE_CUDA
    ggml_backend_load_all();
    ctx->backend = ggml_backend_cuda_init(device);
    if (ctx->backend == NULL) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    return TRELLIS_STATUS_OK;
#else
    (void) device;
    return TRELLIS_STATUS_CUDA_UNAVAILABLE;
#endif
}

void trellis_cuda_free(trellis_cuda_context * ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->backend != NULL) {
        ggml_backend_free(ctx->backend);
    }
    ctx->backend = NULL;
    ctx->device = -1;
}

ggml_gallocr_t trellis_cuda_new_graph_allocator(const trellis_cuda_context * ctx) {
    if (ctx == NULL || ctx->backend == NULL) {
        return NULL;
    }
    return ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
}

trellis_status trellis_cuda_compute_graph(const trellis_cuda_context * ctx, struct ggml_cgraph * graph) {
    if (ctx == NULL || ctx->backend == NULL || graph == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    enum ggml_status status = ggml_backend_graph_compute(ctx->backend, graph);
    return status == GGML_STATUS_SUCCESS ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}
