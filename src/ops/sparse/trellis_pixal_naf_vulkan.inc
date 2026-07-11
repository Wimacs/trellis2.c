#include "projected_attention.h"

typedef struct pixal_naf_vk_push {
    uint32_t n_coords;
    uint32_t batch;
    uint32_t query_h;
    uint32_t query_w;
    uint32_t feature_h;
    uint32_t feature_w;
    uint32_t image_resolution;
    uint32_t grid_resolution;
} pixal_naf_vk_push;

_Static_assert(
    sizeof(pixal_naf_vk_push) == sizeof(sparse_vk_push),
    "Pixal3D NAF push constants must match the shared Vulkan pipeline layout");

static int pixal_naf_checked_mul_size(size_t a, size_t b, size_t * out) {
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int pixal_naf_tensor_count4(
    int a,
    int b,
    int c,
    int d,
    size_t * count) {
    size_t value = 0;
    return a > 0 && b > 0 && c > 0 && d > 0 &&
        pixal_naf_checked_mul_size((size_t) a, (size_t) b, &value) &&
        pixal_naf_checked_mul_size(value, (size_t) c, &value) &&
        pixal_naf_checked_mul_size(value, (size_t) d, count);
}

static int pixal_naf_validate_host_args(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count,
    size_t * query_count,
    size_t * key_count,
    size_t * value_count,
    size_t * coord_count) {
    if (queries == NULL || keys == NULL || values == NULL || coords_bxyz == NULL ||
        cameras == NULL || desc == NULL || projected_out == NULL ||
        n_coords < 0 || (uint64_t) n_coords > UINT32_MAX / 4u ||
        desc->batch <= 0 || desc->query_h <= 0 || desc->query_w <= 0 ||
        desc->feature_h < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->feature_w < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->image_resolution <= 0 || desc->grid_resolution <= 0 ||
        desc->query_h % desc->feature_h != 0 ||
        desc->query_w % desc->feature_w != 0 ||
        desc->query_h / desc->feature_h != desc->query_w / desc->feature_w) {
        return 0;
    }
    size_t expected_output = 0;
    size_t camera_count = 0;
    if (!pixal_naf_tensor_count4(
            desc->batch,
            desc->query_h,
            desc->query_w,
            TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
            query_count) ||
        !pixal_naf_tensor_count4(
            desc->batch,
            desc->feature_h,
            desc->feature_w,
            TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
            key_count) ||
        !pixal_naf_tensor_count4(
            desc->batch,
            desc->feature_h,
            desc->feature_w,
            TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
            value_count) ||
        !pixal_naf_checked_mul_size((size_t) n_coords, 4u, coord_count) ||
        !pixal_naf_checked_mul_size(
            (size_t) n_coords,
            TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
            &expected_output) ||
        !pixal_naf_checked_mul_size((size_t) desc->batch, 3u, &camera_count) ||
        *query_count > SIZE_MAX / sizeof(float) ||
        *key_count > SIZE_MAX / sizeof(float) ||
        *value_count > SIZE_MAX / sizeof(float) ||
        expected_output > SIZE_MAX / sizeof(float) ||
        *coord_count > SIZE_MAX / sizeof(int32_t) ||
        camera_count > SIZE_MAX / sizeof(float) || projected_count < expected_output) {
        return 0;
    }
    for (int b = 0; b < desc->batch; ++b) {
        if (!isfinite(cameras[b].camera_angle_x) ||
            cameras[b].camera_angle_x <= 0.0f ||
            cameras[b].camera_angle_x >= 3.14159265358979323846f ||
            !isfinite(cameras[b].distance) ||
            !isfinite(cameras[b].mesh_scale) || cameras[b].mesh_scale == 0.0f) {
            return 0;
        }
    }
    for (int64_t i = 0; i < n_coords; ++i) {
        const int32_t b = coords_bxyz[(size_t) i * 4u + 0u];
        const int32_t x = coords_bxyz[(size_t) i * 4u + 1u];
        const int32_t y = coords_bxyz[(size_t) i * 4u + 2u];
        const int32_t z = coords_bxyz[(size_t) i * 4u + 3u];
        if (b < 0 || b >= desc->batch || x < 0 || y < 0 || z < 0 ||
            x >= desc->grid_resolution || y >= desc->grid_resolution ||
            z >= desc->grid_resolution) {
            return 0;
        }
    }
    return 1;
}

static trellis_status pixal_naf_dispatch_dims(
    const trellis_sparse_vk_backend * vk,
    uint64_t total_groups,
    uint32_t * groups_x,
    uint32_t * groups_y,
    uint32_t * groups_z) {
    if (vk == NULL || total_groups == 0 || groups_x == NULL || groups_y == NULL || groups_z == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t max_x = vk->max_workgroup_count[0] == 0 ? 65535u : vk->max_workgroup_count[0];
    const uint64_t max_y = vk->max_workgroup_count[1] == 0 ? 65535u : vk->max_workgroup_count[1];
    const uint64_t max_z = vk->max_workgroup_count[2] == 0 ? 65535u : vk->max_workgroup_count[2];
    const uint64_t gx = total_groups < max_x ? total_groups : max_x;
    const uint64_t yz = (total_groups + gx - 1u) / gx;
    const uint64_t gy = yz < max_y ? yz : max_y;
    const uint64_t gz = (yz + gy - 1u) / gy;
    if (gx == 0 || gy == 0 || gz == 0 || gz > max_z ||
        gx > UINT32_MAX / gy || gx * gy > UINT32_MAX / gz) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *groups_x = (uint32_t) gx;
    *groups_y = (uint32_t) gy;
    *groups_z = (uint32_t) gz;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_pixal_naf_attention_project_sparse(
    trellis_sparse_backend * backend,
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const struct trellis_pixal_camera * cameras,
    const struct trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    size_t query_count = 0;
    size_t key_count = 0;
    size_t value_count = 0;
    size_t coord_count = 0;
    if (vk == NULL || vk->pipelines[SPARSE_VK_PIPE_PIXAL_NAF_ATTENTION_PROJECT] == VK_NULL_HANDLE ||
        !pixal_naf_validate_host_args(
            queries,
            keys,
            values,
            coords_bxyz,
            n_coords,
            cameras,
            desc,
            projected_out,
            projected_count,
            &query_count,
            &key_count,
            &value_count,
            &coord_count)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (n_coords == 0) {
        return TRELLIS_STATUS_OK;
    }
    const size_t device_output_count =
        (size_t) n_coords * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
    const size_t camera_count = (size_t) desc->batch * 3u;

    float * camera_data = (float *) malloc(camera_count * sizeof(float));
    if (camera_data == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int b = 0; b < desc->batch; ++b) {
        camera_data[(size_t) b * 3u + 0u] = cameras[b].camera_angle_x;
        camera_data[(size_t) b * 3u + 1u] = cameras[b].distance;
        camera_data[(size_t) b * 3u + 2u] = cameras[b].mesh_scale;
    }

    trellis_sparse_buffer * query_buffer = NULL;
    trellis_sparse_buffer * key_buffer = NULL;
    trellis_sparse_buffer * value_buffer = NULL;
    trellis_sparse_buffer * coord_buffer = NULL;
    trellis_sparse_buffer * camera_buffer = NULL;
    trellis_sparse_buffer * output_buffer = NULL;
    trellis_status status = vk_upload_f32(backend, queries, query_count, &query_buffer);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_f32(backend, keys, key_count, &key_buffer);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_f32(backend, values, value_count, &value_buffer);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_i32(backend, coords_bxyz, coord_count, &coord_buffer);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_f32(backend, camera_data, camera_count, &camera_buffer);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_f32(backend, device_output_count, &output_buffer);
    }

    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    if (status == TRELLIS_STATUS_OK) {
        status = pixal_naf_dispatch_dims(
            vk,
            (uint64_t) n_coords * TRELLIS_PIXAL_NAF_ATTENTION_HEADS,
            &groups_x,
            &groups_y,
            &groups_z);
    }
    if (status == TRELLIS_STATUS_OK) {
        const pixal_naf_vk_push push = {
            (uint32_t) n_coords,
            (uint32_t) desc->batch,
            (uint32_t) desc->query_h,
            (uint32_t) desc->query_w,
            (uint32_t) desc->feature_h,
            (uint32_t) desc->feature_w,
            (uint32_t) desc->image_resolution,
            (uint32_t) desc->grid_resolution,
        };
        sparse_vk_push pipeline_push;
        memcpy(&pipeline_push, &push, sizeof(pipeline_push));
        status = vk_dispatch_pipeline(
            vk,
            vk->pipelines[SPARSE_VK_PIPE_PIXAL_NAF_ATTENTION_PROJECT],
            &pipeline_push,
            query_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            coord_buffer,
            camera_buffer,
            NULL,
            NULL,
            NULL,
            NULL,
            groups_x,
            groups_y,
            groups_z);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_f32(backend, output_buffer, projected_out, device_output_count);
    }

    vk_free_buffer(backend, output_buffer);
    vk_free_buffer(backend, camera_buffer);
    vk_free_buffer(backend, coord_buffer);
    vk_free_buffer(backend, value_buffer);
    vk_free_buffer(backend, key_buffer);
    vk_free_buffer(backend, query_buffer);
    free(camera_data);
    return status;
}
