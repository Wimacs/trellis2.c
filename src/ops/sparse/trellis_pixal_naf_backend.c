#include "trellis_sparse_backend.h"
#include "projected_attention.h"

trellis_status trellis_pixal_naf_attention_project_sparse_vulkan_f32(
    struct trellis_sparse_backend * backend,
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count) {
    if (backend == NULL || backend->ops == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (backend->ops->pixal_naf_attention_project_sparse == NULL) {
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    return backend->ops->pixal_naf_attention_project_sparse(
        backend,
        queries,
        keys,
        values,
        coords_bxyz,
        n_coords,
        cameras,
        desc,
        projected_out,
        projected_count);
}
