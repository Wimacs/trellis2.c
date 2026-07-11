#include "trellis_sparse_backend.h"

trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out) {
    (void) device;
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    return TRELLIS_STATUS_NOT_IMPLEMENTED;
}
