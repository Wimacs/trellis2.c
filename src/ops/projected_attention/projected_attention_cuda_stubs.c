#include "projected_attention.h"

trellis_status trellis_pixal_naf_attention_project_sparse_cuda_f32(
    const float * queries_dev,
    const float * keys_dev,
    const float * values_dev,
    const int32_t * coords_bxyz_dev,
    int64_t n_coords,
    const trellis_pixal_camera * cameras_dev,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out_dev,
    size_t projected_count) {
    (void) queries_dev;
    (void) keys_dev;
    (void) values_dev;
    (void) coords_bxyz_dev;
    (void) n_coords;
    (void) cameras_dev;
    (void) desc;
    (void) projected_out_dev;
    (void) projected_count;
    return TRELLIS_STATUS_CUDA_UNAVAILABLE;
}

trellis_status trellis_pixal_naf_attention_project_sparse_cuda_host_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    int device,
    float * projected_out,
    size_t projected_count) {
    (void) queries;
    (void) keys;
    (void) values;
    (void) coords_bxyz;
    (void) n_coords;
    (void) cameras;
    (void) desc;
    (void) device;
    (void) projected_out;
    (void) projected_count;
    return TRELLIS_STATUS_CUDA_UNAVAILABLE;
}
