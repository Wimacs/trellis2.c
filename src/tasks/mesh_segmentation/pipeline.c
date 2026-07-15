#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_model_package.h"
#include "trellis_platform.h"
#include "trellis_sparse_backend.h"
#include "image_to_3d_internal.h"
#include "adapter.h"
#include "gltf_io.h"
#include "condition_render.h"
#include "labels.h"
#include "material_features.h"
#include "partition.h"
#include "parts_glb.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define SEGMENTATION_NORMALIZED_EXTENT 0.99999
#define SEGMENTATION_AABB_LIMIT 0.5001f

static int segmentation_copy_path(
    char * output,
    size_t output_size,
    const char * path);

typedef struct segmentation_named_path {
    const char * label;
    const char * path;
} segmentation_named_path;

static int segmentation_path_is_set(const char * path) {
    return path != NULL && path[0] != '\0';
}

static int segmentation_has_glb_extension(const char * path) {
    if (path == NULL) return 0;
    const size_t length = strlen(path);
    if (length < 4u) return 0;
    const char * extension = path + length - 4u;
    return extension[0] == '.' &&
        (extension[1] == 'g' || extension[1] == 'G') &&
        (extension[2] == 'l' || extension[2] == 'L') &&
        (extension[3] == 'b' || extension[3] == 'B');
}

static int segmentation_canonical_path(
    const char * path,
    char output[PATH_MAX]) {
    if (!segmentation_path_is_set(path) || output == NULL) return 0;
#ifdef _WIN32
    return _fullpath(output, path, PATH_MAX) != NULL;
#else
    if (realpath(path, output) != NULL) return 1;

    char absolute[PATH_MAX];
    if (path[0] == '/') {
        const int count = snprintf(absolute, sizeof(absolute), "%s", path);
        if (count < 0 || (size_t) count >= sizeof(absolute)) return 0;
    } else {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == NULL) return 0;
        const int count = snprintf(absolute, sizeof(absolute), "%s/%s", cwd, path);
        if (count < 0 || (size_t) count >= sizeof(absolute)) return 0;
    }

    /* Lexically normalize a non-existent path. Existing paths took the
     * realpath branch above, so symlink aliases are still caught. */
    size_t component_offsets[PATH_MAX / 2u];
    size_t depth = 0;
    size_t output_length = 1u;
    output[0] = '/';
    output[1] = '\0';
    char * cursor = absolute;
    while (*cursor != '\0') {
        while (*cursor == '/') ++cursor;
        if (*cursor == '\0') break;
        char * component = cursor;
        while (*cursor != '\0' && *cursor != '/') ++cursor;
        const size_t component_length = (size_t) (cursor - component);
        if (component_length == 1u && component[0] == '.') continue;
        if (component_length == 2u && component[0] == '.' && component[1] == '.') {
            if (depth > 0u) output_length = component_offsets[--depth];
            output[output_length] = '\0';
            continue;
        }
        if (depth >= sizeof(component_offsets) / sizeof(component_offsets[0])) return 0;
        const size_t prior_length = output_length;
        if (output_length > 1u) {
            if (output_length + 1u >= PATH_MAX) return 0;
            output[output_length++] = '/';
        }
        if (component_length >= PATH_MAX - output_length) return 0;
        component_offsets[depth++] = prior_length;
        memcpy(output + output_length, component, component_length);
        output_length += component_length;
        output[output_length] = '\0';
    }
    return 1;
#endif
}

static int segmentation_paths_equal(const char * left, const char * right) {
    char canonical_left[PATH_MAX];
    char canonical_right[PATH_MAX];
    if (!segmentation_canonical_path(left, canonical_left) ||
        !segmentation_canonical_path(right, canonical_right)) {
        return 0;
    }
#ifdef _WIN32
    return _stricmp(canonical_left, canonical_right) == 0;
#else
    return strcmp(canonical_left, canonical_right) == 0;
#endif
}

static int segmentation_stat_is_directory(const char * path) {
#ifdef _WIN32
    struct _stat64 info;
    return _stat64(path, &info) == 0 && (info.st_mode & _S_IFMT) == _S_IFDIR;
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

static int segmentation_stat_is_regular_file(const char * path) {
#ifdef _WIN32
    struct _stat64 info;
    return _stat64(path, &info) == 0 && (info.st_mode & _S_IFMT) == _S_IFREG;
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
#endif
}

static int segmentation_parent_is_writable(const char * path) {
#ifdef _WIN32
    return _access(path, 2) == 0;
#else
    return access(path, W_OK) == 0;
#endif
}

static int segmentation_output_parent(
    const char * canonical_path,
    char parent[PATH_MAX]) {
    if (!segmentation_copy_path(parent, PATH_MAX, canonical_path)) return 0;
    char * separator = trellis_path_last_sep(parent);
    if (separator == NULL) return segmentation_copy_path(parent, PATH_MAX, ".");
    if (separator == parent) {
        parent[1] = '\0';
        return 1;
    }
#ifdef _WIN32
    if (separator == parent + 2 && parent[1] == ':') {
        parent[3] = '\0';
        return 1;
    }
#endif
    *separator = '\0';
    return 1;
}

static trellis_status segmentation_prepare_output_path(
    const segmentation_named_path * output,
    int require_glb) {
    if (output == NULL || !segmentation_path_is_set(output->path)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (require_glb && !segmentation_has_glb_extension(output->path)) {
        TRELLIS_ERROR(
            "mesh segmentation: %s must use the .glb extension: %s",
            output->label,
            output->path);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char canonical[PATH_MAX];
    if (!segmentation_canonical_path(output->path, canonical) ||
        !trellis_mkdir_parent(canonical)) {
        TRELLIS_ERROR(
            "mesh segmentation: cannot create parent directory for %s '%s'",
            output->label,
            output->path);
        return TRELLIS_STATUS_NOT_FOUND;
    }
    char parent[PATH_MAX];
    if (!segmentation_output_parent(canonical, parent) ||
        !segmentation_stat_is_directory(parent) ||
        !segmentation_parent_is_writable(parent)) {
        TRELLIS_ERROR(
            "mesh segmentation: parent directory for %s is unavailable or not writable: %s",
            output->label,
            output->path);
        return TRELLIS_STATUS_NOT_FOUND;
    }
    if (trellis_access_exists(canonical) &&
        segmentation_stat_is_directory(canonical)) {
        TRELLIS_ERROR(
            "mesh segmentation: %s names a directory, not a file: %s",
            output->label,
            output->path);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status segmentation_validate_path_contract(
    const trellis_mesh_segmentation_options * options) {
    const segmentation_named_path reads[] = {
        { "input mesh", options->input_path },
        { "condition image", options->condition_image_path },
        { "shape latent input", options->shape_latent_path },
        { "source-texture latent input", options->texture_latent_path },
        { "segmentation latent input", options->segmentation_latent_path },
    };
    const segmentation_named_path writes[] = {
        { "output mesh", options->output_path },
        { "rendered condition output", options->rendered_condition_output_path },
        { "shape latent output", options->shape_latent_output_path },
        { "source-texture latent output", options->texture_latent_output_path },
        { "segmentation latent output", options->segmentation_latent_output_path },
    };
    for (size_t i = 0; i < sizeof(writes) / sizeof(writes[0]); ++i) {
        if (!segmentation_path_is_set(writes[i].path)) continue;
        const trellis_status status = segmentation_prepare_output_path(
            &writes[i],
            i == 0u);
        if (status != TRELLIS_STATUS_OK) return status;
        for (size_t j = 0; j < sizeof(reads) / sizeof(reads[0]); ++j) {
            if (segmentation_path_is_set(reads[j].path) &&
                segmentation_paths_equal(writes[i].path, reads[j].path)) {
                TRELLIS_ERROR(
                    "mesh segmentation: path conflict: %s and %s both name '%s'",
                    writes[i].label,
                    reads[j].label,
                    writes[i].path);
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
        for (size_t j = 0; j < i; ++j) {
            if (segmentation_path_is_set(writes[j].path) &&
                segmentation_paths_equal(writes[i].path, writes[j].path)) {
                TRELLIS_ERROR(
                    "mesh segmentation: path conflict: %s and %s both name '%s'",
                    writes[i].label,
                    writes[j].label,
                    writes[i].path);
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status segmentation_require_readable_file(
    const char * label,
    const char * path) {
    if (!segmentation_path_is_set(path) ||
        !segmentation_stat_is_regular_file(path) ||
        !trellis_access_read(path)) {
        TRELLIS_ERROR(
            "mesh segmentation: required %s is not a readable regular file: %s",
            label != NULL ? label : "file",
            path != NULL ? path : "(null)");
        return TRELLIS_STATUS_NOT_FOUND;
    }
    return TRELLIS_STATUS_OK;
}

static const char * segmentation_sparse_backend_name(
    trellis_sparse_backend_kind kind) {
    switch (kind) {
        case TRELLIS_SPARSE_BACKEND_CUDA: return "cuda";
        case TRELLIS_SPARSE_BACKEND_VULKAN: return "vulkan";
        case TRELLIS_SPARSE_BACKEND_CPU: return "cpu";
        default: return "unknown";
    }
}

static trellis_status segmentation_sparse_backend_kind(
    trellis_backend_kind graph_kind,
    trellis_sparse_backend_kind * sparse_kind_out) {
    if (sparse_kind_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    if (graph_kind == TRELLIS_BACKEND_CUDA) {
        *sparse_kind_out = TRELLIS_SPARSE_BACKEND_CUDA;
        return TRELLIS_STATUS_OK;
    }
    if (graph_kind == TRELLIS_BACKEND_VULKAN) {
        *sparse_kind_out = TRELLIS_SPARSE_BACKEND_VULKAN;
        return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_INVALID_ARGUMENT;
}

static int segmentation_copy_path(
    char * output,
    size_t output_size,
    const char * path) {
    if (output == NULL || output_size == 0 || path == NULL || path[0] == '\0') {
        return 0;
    }
    const int count = snprintf(output, output_size, "%s", path);
    return count >= 0 && (size_t) count < output_size;
}

static int segmentation_component_path(
    const trellis_model_package * package,
    const char * role,
    const char * override_path,
    char output[4096]) {
    if (override_path != NULL && override_path[0] != '\0') {
        return segmentation_copy_path(output, 4096, override_path);
    }
    const trellis_status status = trellis_model_package_resolve_component_path(
        package,
        role,
        output,
        4096);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: component role '%s' is unavailable: %s",
            role,
            trellis_status_string(status));
        return 0;
    }
    return 1;
}

static int segmentation_component_is(
    const trellis_model_package * package,
    const char * role,
    const char * architecture) {
    const trellis_model_component_instance * component =
        trellis_model_package_find_component(package, role);
    return component != NULL && component->architecture != NULL &&
        strcmp(component->architecture, architecture) == 0;
}

static int segmentation_flow_policy_is_exact(
    const trellis_model_component_instance * component) {
    return component != NULL && component->architecture != NULL &&
        strcmp(component->architecture, "trellis_dit_flow") == 0 &&
        component->execution.compute_dtype == TRELLIS_DTYPE_BF16 &&
        component->execution.attention == TRELLIS_ATTENTION_FLASH &&
        component->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16 &&
        component->execution.emulate_bf16_blocks == 1;
}

static int segmentation_cache_anchors_match(
    const trellis_shape_latent_cache_info * left,
    const trellis_shape_latent_cache_info * right) {
    if (left == NULL || right == NULL) return 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(left->anchor_aabb_min[axis] - right->anchor_aabb_min[axis]) > 1e-5f ||
            fabsf(left->anchor_aabb_max[axis] - right->anchor_aabb_max[axis]) > 1e-5f) {
            return 0;
        }
    }
    return 1;
}

static int segmentation_latent_coords_match(
    const trellis_structured_latent * left,
    const trellis_structured_latent * right) {
    if (left == NULL || right == NULL || left->coords_bxyz == NULL ||
        right->coords_bxyz == NULL || left->n_coords != right->n_coords ||
        left->n_coords <= 0 || left->resolution != right->resolution ||
        left->channels != 32 || right->channels != 32 ||
        (uint64_t) left->n_coords > SIZE_MAX / (4u * sizeof(int32_t))) {
        return 0;
    }
    return memcmp(
        left->coords_bxyz,
        right->coords_bxyz,
        (size_t) left->n_coords * 4u * sizeof(int32_t)) == 0;
}

static trellis_status segmentation_read_latent_cache(
    const char * label,
    const char * path,
    int resolution,
    trellis_structured_latent * latent_out,
    trellis_shape_latent_cache_info * info_out) {
    const trellis_status status = trellis_shape_latent_cache_read(
        path,
        latent_out,
        info_out);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: %s latent cache '%s' is invalid: %s",
            label,
            path,
            trellis_status_string(status));
        return status;
    }
    if (latent_out->channels != 32 || info_out->resolution != resolution) {
        TRELLIS_ERROR(
            "mesh segmentation: %s latent cache contract mismatch "
            "resolution=%d channels=%d",
            label,
            info_out->resolution,
            latent_out->channels);
        trellis_structured_latent_free(latent_out);
        memset(info_out, 0, sizeof(*info_out));
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

/* SegviGen's released data pipeline normalizes flattened glTF world
 * coordinates directly. It does not apply the inverse TRELLIS export-axis
 * rotation used by the image-to-3D texturing task. */
static trellis_status segmentation_normalize_asset_mesh(
    const trellis_mesh_rigging_asset * asset,
    const trellis_shape_latent_cache_info * cache_anchor,
    trellis_mesh_host * mesh_out) {
    if (asset == NULL || mesh_out == NULL || asset->positions == NULL ||
        asset->triangles == NULL || asset->vertex_count == 0 ||
        asset->triangle_count == 0 || asset->vertex_count > (size_t) INT64_MAX ||
        asset->triangle_count > (size_t) INT64_MAX ||
        asset->vertex_count > (size_t) INT32_MAX ||
        asset->vertex_count > SIZE_MAX / (3u * sizeof(float)) ||
        asset->triangle_count > SIZE_MAX / (3u * sizeof(int32_t))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));

    double raw_min[3] = {0.0, 0.0, 0.0};
    double raw_max[3] = {0.0, 0.0, 0.0};
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        const float * position = asset->positions + vertex * 3u;
        for (int axis = 0; axis < 3; ++axis) {
            if (!isfinite(position[axis])) return TRELLIS_STATUS_PARSE_ERROR;
            if (vertex == 0 || position[axis] < raw_min[axis]) raw_min[axis] = position[axis];
            if (vertex == 0 || position[axis] > raw_max[axis]) raw_max[axis] = position[axis];
        }
    }
    double raw_center[3];
    double raw_extent[3];
    double raw_max_extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        raw_center[axis] = (raw_min[axis] + raw_max[axis]) * 0.5;
        raw_extent[axis] = raw_max[axis] - raw_min[axis];
        if (raw_extent[axis] > raw_max_extent) raw_max_extent = raw_extent[axis];
    }
    if (!(raw_max_extent > 0.0) || !isfinite(raw_max_extent)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    double target_center[3] = {0.0, 0.0, 0.0};
    double target_max_extent = SEGMENTATION_NORMALIZED_EXTENT;
    if (cache_anchor != NULL) {
        double anchor_extent[3];
        double anchor_max_extent = 0.0;
        double ratio_drift = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            target_center[axis] =
                ((double) cache_anchor->anchor_aabb_min[axis] +
                 (double) cache_anchor->anchor_aabb_max[axis]) * 0.5;
            anchor_extent[axis] =
                (double) cache_anchor->anchor_aabb_max[axis] -
                (double) cache_anchor->anchor_aabb_min[axis];
            if (anchor_extent[axis] > anchor_max_extent) {
                anchor_max_extent = anchor_extent[axis];
            }
        }
        if (!(anchor_max_extent > 0.0) || !isfinite(anchor_max_extent)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const double drift = fabs(
                raw_extent[axis] / raw_max_extent -
                anchor_extent[axis] / anchor_max_extent);
            if (drift > ratio_drift) ratio_drift = drift;
        }
        if (ratio_drift > 0.05) {
            TRELLIS_ERROR(
                "mesh segmentation: latent cache belongs to incompatible geometry "
                "(normalized extent drift %.2f%% > 5%%)",
                ratio_drift * 100.0);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        target_max_extent = anchor_max_extent;
    }

    float * vertices = (float *) malloc(asset->vertex_count * 3u * sizeof(float));
    int32_t * faces = (int32_t *) malloc(asset->triangle_count * 3u * sizeof(int32_t));
    if (vertices == NULL || faces == NULL) {
        free(vertices);
        free(faces);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const double scale = target_max_extent / raw_max_extent;
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = (float) (
                ((double) asset->positions[vertex * 3u + (size_t) axis] -
                 raw_center[axis]) * scale + target_center[axis]);
            if (!isfinite(value) || value < -SEGMENTATION_AABB_LIMIT ||
                value > SEGMENTATION_AABB_LIMIT) {
                free(vertices);
                free(faces);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            vertices[vertex * 3u + (size_t) axis] = value;
        }
    }
    for (size_t index = 0; index < asset->triangle_count * 3u; ++index) {
        if (asset->triangles[index] >= asset->vertex_count ||
            asset->triangles[index] > (uint32_t) INT32_MAX) {
            free(vertices);
            free(faces);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        faces[index] = (int32_t) asset->triangles[index];
    }
    mesh_out->vertices = vertices;
    mesh_out->faces = faces;
    mesh_out->n_vertices = (int64_t) asset->vertex_count;
    mesh_out->n_faces = (int64_t) asset->triangle_count;
    TRELLIS_INFO(
        "mesh segmentation: normalized world geometry vertices=%lld triangles=%lld scale=%.9g%s",
        (long long) mesh_out->n_vertices,
        (long long) mesh_out->n_faces,
        scale,
        cache_anchor != NULL ? " (latent anchor aligned)" : "");
    return TRELLIS_STATUS_OK;
}

static trellis_status segmentation_make_shape_features(
    const trellis_flexible_dual_grid * grid,
    const trellis_flexible_dual_grid_options * grid_options,
    float ** features_out) {
    if (grid == NULL || grid_options == NULL || features_out == NULL ||
        grid->n <= 0 || grid->coords == NULL || grid->dual_vertices == NULL ||
        grid->intersected == NULL ||
        (uint64_t) grid->n > SIZE_MAX / (6u * sizeof(float))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *features_out = NULL;
    float voxel_size[3];
    for (int axis = 0; axis < 3; ++axis) {
        voxel_size[axis] =
            (grid_options->aabb_max[axis] - grid_options->aabb_min[axis]) /
            (float) grid_options->grid_size[axis];
        if (!(voxel_size[axis] > 0.0f) || !isfinite(voxel_size[axis])) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    float * features = (float *) malloc((size_t) grid->n * 6u * sizeof(float));
    if (features == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (int64_t row = 0; row < grid->n; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            const float local =
                grid->dual_vertices[(size_t) row * 3u + (size_t) axis] /
                    voxel_size[axis] -
                (float) grid->coords[(size_t) row * 4u + 1u + (size_t) axis];
            if (!isfinite(local) || local < -1e-4f || local > 1.0001f) {
                free(features);
                return TRELLIS_STATUS_ERROR;
            }
            features[(size_t) row * 6u + (size_t) axis] = local - 0.5f;
            features[(size_t) row * 6u + 3u + (size_t) axis] =
                grid->intersected[(size_t) row * 3u + (size_t) axis] ? 0.5f : -0.5f;
        }
    }
    *features_out = features;
    return TRELLIS_STATUS_OK;
}

static trellis_status segmentation_load_encoder(
    const trellis_backend_context * weight_backend,
    const char * label,
    const char * path,
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_encoder_weights * encoder) {
    if (!trellis_load_tensor_store_f32(
            weight_backend,
            label,
            path,
            true,
            64,
            store,
            NULL)) {
        return TRELLIS_STATUS_ERROR;
    }
    char issue[512];
    issue[0] = '\0';
    const trellis_status status = trellis_sparse_unet_vae_encoder_bind_weights(
        store,
        encoder,
        issue,
        sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: %s bind failed: %s%s%s",
            label,
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
    }
    return status;
}

static trellis_status segmentation_validate_latent_values(
    const char * label,
    const trellis_structured_latent * latent) {
    if (latent == NULL || latent->coords_bxyz == NULL || latent->feats == NULL ||
        latent->n_coords <= 0 || latent->channels != 32 ||
        latent->n_coords > INT64_MAX / latent->channels) {
        return TRELLIS_STATUS_ERROR;
    }
    const int64_t count = latent->n_coords * (int64_t) latent->channels;
    float minimum = latent->feats[0];
    float maximum = latent->feats[0];
    double sum = 0.0;
    double square_sum = 0.0;
    for (int64_t index = 0; index < count; ++index) {
        const float value = latent->feats[index];
        if (!isfinite(value)) {
            TRELLIS_ERROR(
                "mesh segmentation: %s encoder produced non-finite value at %lld",
                label,
                (long long) index);
            return TRELLIS_STATUS_ERROR;
        }
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        sum += value;
        square_sum += (double) value * (double) value;
    }
    const double mean = sum / (double) count;
    double variance = square_sum / (double) count - mean * mean;
    if (variance < 0.0) variance = 0.0;
    TRELLIS_INFO(
        "mesh segmentation: %s latent tokens=%lld min=%.6g max=%.6g mean=%.6g std=%.6g",
        label,
        (long long) latent->n_coords,
        minimum,
        maximum,
        mean,
        sqrt(variance));
    return TRELLIS_STATUS_OK;
}

static trellis_status segmentation_run_encoder(
    const trellis_backend_context * weight_backend,
    trellis_sparse_backend_kind sparse_kind,
    int device,
    trellis_sparse_backend * sparse_backend,
    const char * label,
    const char * weights_path,
    const trellis_flexible_dual_grid * grid,
    const float * input_features,
    int resolution,
    trellis_structured_latent * latent_out) {
    if (weight_backend == NULL || label == NULL || weights_path == NULL ||
        grid == NULL || grid->coords == NULL || grid->n <= 0 ||
        input_features == NULL || latent_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_tensor_store store;
    trellis_sparse_unet_vae_encoder_weights encoder;
    memset(&store, 0, sizeof(store));
    memset(&encoder, 0, sizeof(encoder));
    memset(latent_out, 0, sizeof(*latent_out));
    trellis_status status = segmentation_load_encoder(
        weight_backend,
        label,
        weights_path,
        &store,
        &encoder);
    if (status != TRELLIS_STATUS_OK) {
        trellis_tensor_store_free(&store);
        return status;
    }
    trellis_sparse_unet_vae_encoder_forward_options encoder_options;
    memset(&encoder_options, 0, sizeof(encoder_options));
    encoder_options.backend_kind = sparse_kind;
    encoder_options.device = device;
    encoder_options.sparse_backend = sparse_backend;
    status = trellis_sparse_unet_vae_encoder_forward_backend_f32_host(
        &encoder,
        grid->coords,
        input_features,
        grid->n,
        &encoder_options,
        &latent_out->coords_bxyz,
        &latent_out->feats,
        &latent_out->n_coords,
        &latent_out->channels);
    trellis_tensor_store_free(&store);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: %s forward failed: %s",
            label,
            trellis_status_string(status));
        trellis_structured_latent_free(latent_out);
        return status;
    }
    latent_out->resolution = resolution;
    status = segmentation_validate_latent_values(label, latent_out);
    if (status != TRELLIS_STATUS_OK) {
        trellis_structured_latent_free(latent_out);
    }
    return status;
}

trellis_status trellis_pipeline_trellis2_segment_mesh(
    const trellis_mesh_segmentation_options * options) {
    trellis_runtime_init();

    const trellis_mesh_segmentation_small_part_mode small_part_mode =
        options != NULL &&
        options->struct_size >= TRELLIS_MESH_SEGMENTATION_OPTIONS_V2_SIZE ?
            options->small_part_mode :
            TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP;

    if (options == NULL ||
        options->struct_size < TRELLIS_MESH_SEGMENTATION_OPTIONS_V1_SIZE ||
        options->model_dir == NULL || options->model_dir[0] == '\0' ||
        options->segmentation_model_dir == NULL ||
        options->segmentation_model_dir[0] == '\0' ||
        ((!segmentation_path_is_set(options->segmentation_latent_path)) &&
         !segmentation_path_is_set(options->dino_dir)) ||
        options->input_path == NULL || options->input_path[0] == '\0' ||
        options->output_path == NULL || options->output_path[0] == '\0' ||
        options->device < 0 || options->resolution != 512 ||
        options->steps <= 0 || options->steps > 1000 ||
        options->min_component_faces < 0 || options->min_palette_voxels <= 0 ||
        !isfinite(options->palette_merge_distance) ||
        options->palette_merge_distance < 0.0f ||
        options->palette_merge_distance > 1.732051f ||
        (small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP &&
         small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE &&
         small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD) ||
        options->flow_blocks_override < -1 ||
        options->flow_block_parts_override < -1 ||
        options->flow_block_parts_override > 3 ||
        (options->condition_image_prepared != 0 &&
         options->condition_image_prepared != 1) ||
        (options->flow_no_rope != 0 && options->flow_no_rope != 1) ||
        (options->emulate_bf16_blocks != 0 &&
         options->emulate_bf16_blocks != 1) ||
        (options->no_flash_attn != 0 && options->no_flash_attn != 1)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    trellis_model_package base_package = TRELLIS_MODEL_PACKAGE_INIT;
    trellis_model_package segmentation_package = TRELLIS_MODEL_PACKAGE_INIT;
    trellis_backend_context graph_backend;
    trellis_backend_context cpu_weight_backend;
    trellis_cuda_context cuda;
    trellis_sparse_backend * sparse_backend = NULL;
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    trellis_mesh_host normalized_mesh;
    trellis_mesh_host cache_projection_mesh;
    trellis_prepared_condition_image prepared_image;
    trellis_flexible_dual_grid grid;
    trellis_structured_latent shape_latent;
    trellis_structured_latent source_texture_latent;
    trellis_structured_latent segmentation_latent;
    trellis_shape_latent_cache_info shape_cache_info;
    trellis_shape_latent_cache_info texture_cache_info;
    trellis_shape_latent_cache_info segmentation_cache_info;
    trellis_sparse_c2s_guides guides;
    trellis_image_condition_result condition;
    trellis_pbr_voxels decoded_labels;
    float * shape_encoder_input = NULL;
    float * texture_encoder_input = NULL;
    uint32_t * semantic_labels = NULL;
    uint32_t * face_part_ids = NULL;
    size_t semantic_count = 0;
    size_t part_count = 0;
    trellis_mesh_segmentation_small_part_stats small_part_stats;
    int graph_backend_initialized = 0;
    int cpu_weight_backend_initialized = 0;
    int cuda_initialized = 0;
    int shape_cache_hit = 0;
    int texture_cache_hit = 0;
    int segmentation_cache_hit = 0;
    memset(&graph_backend, 0, sizeof(graph_backend));
    memset(&cpu_weight_backend, 0, sizeof(cpu_weight_backend));
    memset(&cuda, 0, sizeof(cuda));
    memset(&normalized_mesh, 0, sizeof(normalized_mesh));
    memset(&cache_projection_mesh, 0, sizeof(cache_projection_mesh));
    memset(&prepared_image, 0, sizeof(prepared_image));
    memset(&grid, 0, sizeof(grid));
    memset(&shape_latent, 0, sizeof(shape_latent));
    memset(&source_texture_latent, 0, sizeof(source_texture_latent));
    memset(&segmentation_latent, 0, sizeof(segmentation_latent));
    memset(&shape_cache_info, 0, sizeof(shape_cache_info));
    memset(&texture_cache_info, 0, sizeof(texture_cache_info));
    memset(&segmentation_cache_info, 0, sizeof(segmentation_cache_info));
    memset(&small_part_stats, 0, sizeof(small_part_stats));
    memset(&guides, 0, sizeof(guides));
    memset(&condition, 0, sizeof(condition));
    memset(&decoded_labels, 0, sizeof(decoded_labels));

    const char * backend_name =
        options->backend != NULL && options->backend[0] != '\0' ?
            options->backend : TRELLIS_DEFAULT_BACKEND;
    trellis_backend_kind graph_kind = TRELLIS_BACKEND_CUDA;
    status = trellis_backend_kind_from_name(backend_name, &graph_kind);
    if (status != TRELLIS_STATUS_OK ||
        strcmp(trellis_backend_kind_name(graph_kind), TRELLIS_DEFAULT_BACKEND) != 0) {
        TRELLIS_ERROR(
            "mesh segmentation: this binary was compiled for %s, requested %s",
            TRELLIS_DEFAULT_BACKEND,
            backend_name);
        status = TRELLIS_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }
    trellis_sparse_backend_kind sparse_kind = TRELLIS_SPARSE_BACKEND_CUDA;
    status = segmentation_sparse_backend_kind(graph_kind, &sparse_kind);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    /* Package/task rejection intentionally precedes cache, mesh, image, or
     * GPU work. The base package owns codecs/normalization; the second package
     * owns only the SegviGen Full paired flow. */
    status = trellis_model_package_load(options->model_dir, &base_package);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: failed to load base model package '%s': %s",
            options->model_dir,
            trellis_status_string(status));
        goto cleanup;
    }
    status = trellis_model_package_load(
        options->segmentation_model_dir,
        &segmentation_package);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: failed to load SegviGen package '%s': %s",
            options->segmentation_model_dir,
            trellis_status_string(status));
        goto cleanup;
    }
    if (base_package.family == NULL ||
        strcmp(base_package.family, "trellis2") != 0 ||
        segmentation_package.family == NULL ||
        strcmp(segmentation_package.family, "trellis2") != 0 ||
        segmentation_package.task == NULL ||
        strcmp(segmentation_package.task, "mesh_segmentation") != 0 ||
        segmentation_package.profile == NULL ||
        strcmp(segmentation_package.profile, "512_full") != 0 ||
        !segmentation_component_is(
            &segmentation_package,
            "segmentation_flow",
            "trellis_dit_flow")) {
        TRELLIS_ERROR(
            "mesh segmentation: incompatible package contract "
            "base_family=%s segmentation_family=%s task=%s profile=%s",
            base_package.family != NULL ? base_package.family : "unknown",
            segmentation_package.family != NULL ? segmentation_package.family : "unknown",
            segmentation_package.task != NULL ? segmentation_package.task : "unknown",
            segmentation_package.profile != NULL ? segmentation_package.profile : "unknown");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    const trellis_model_component_instance * segmentation_flow_component =
        trellis_model_package_find_component(
            &segmentation_package,
            "segmentation_flow");
    if (!segmentation_flow_policy_is_exact(segmentation_flow_component)) {
        TRELLIS_ERROR(
            "mesh segmentation: SegviGen segmentation_flow must declare the exact "
            "official policy compute_dtype=bf16 attention=flash "
            "flash_kv_dtype=bf16 emulate_bf16_blocks=true; got "
            "compute_dtype=%s attention=%s flash_kv_dtype=%s "
            "emulate_bf16_blocks=%d",
            segmentation_flow_component != NULL ?
                trellis_dtype_name(
                    segmentation_flow_component->execution.compute_dtype) : "missing",
            segmentation_flow_component != NULL ?
                trellis_attention_mode_name(
                    segmentation_flow_component->execution.attention) : "missing",
            segmentation_flow_component != NULL ?
                trellis_dtype_name(
                    segmentation_flow_component->execution.flash_kv_dtype) : "missing",
            segmentation_flow_component != NULL ?
                segmentation_flow_component->execution.emulate_bf16_blocks : -1);
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    const int shape_cache_requested =
        segmentation_path_is_set(options->shape_latent_path);
    const int texture_cache_requested =
        segmentation_path_is_set(options->texture_latent_path);
    const int segmentation_cache_requested =
        segmentation_path_is_set(options->segmentation_latent_path);
    if (!segmentation_component_is(
            &base_package,
            "shape_decoder",
            "sparse_unet_vae_decoder") ||
        (!shape_cache_requested && !segmentation_component_is(
            &base_package,
            "shape_encoder",
            "sparse_unet_vae_encoder")) ||
        !segmentation_component_is(
            &base_package,
            "texture_decoder",
            "sparse_unet_vae_decoder") ||
        (!segmentation_cache_requested && !texture_cache_requested &&
         !segmentation_component_is(
            &base_package,
            "texture_encoder",
            "sparse_unet_vae_encoder"))) {
        TRELLIS_ERROR(
            "mesh segmentation: base package is missing a required codec "
            "architecture; shape_decoder is always required and uncached "
            "latents also require their encoders");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    status = segmentation_validate_path_contract(options);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = segmentation_require_readable_file("input mesh", options->input_path);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    if (!segmentation_path_is_set(options->segmentation_latent_path) &&
        segmentation_path_is_set(options->condition_image_path)) {
        status = segmentation_require_readable_file(
            "condition image",
            options->condition_image_path);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        if (!options->condition_image_prepared &&
            segmentation_path_is_set(options->birefnet_path)) {
            status = segmentation_require_readable_file(
                "BiRefNet override",
                options->birefnet_path);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
        }
    }

    if (options->shape_latent_path != NULL &&
        options->shape_latent_path[0] != '\0') {
        status = segmentation_read_latent_cache(
            "shape",
            options->shape_latent_path,
            options->resolution,
            &shape_latent,
            &shape_cache_info);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        shape_cache_hit = 1;
    }
    if (options->texture_latent_path != NULL &&
        options->texture_latent_path[0] != '\0') {
        status = segmentation_read_latent_cache(
            "source texture",
            options->texture_latent_path,
            options->resolution,
            &source_texture_latent,
            &texture_cache_info);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        texture_cache_hit = 1;
    }
    if (options->segmentation_latent_path != NULL &&
        options->segmentation_latent_path[0] != '\0') {
        status = segmentation_read_latent_cache(
            "generated segmentation",
            options->segmentation_latent_path,
            options->resolution,
            &segmentation_latent,
            &segmentation_cache_info);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        segmentation_cache_hit = 1;
    }
    if (shape_cache_hit && texture_cache_hit &&
        (!segmentation_cache_anchors_match(&shape_cache_info, &texture_cache_info) ||
         !segmentation_latent_coords_match(&shape_latent, &source_texture_latent))) {
        TRELLIS_ERROR(
            "mesh segmentation: shape and source-texture latent caches do not "
            "share the exact sparse coordinate/anchor contract");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }
    if (segmentation_cache_hit && shape_cache_hit &&
        (!segmentation_cache_anchors_match(
             &segmentation_cache_info,
             &shape_cache_info) ||
         !segmentation_latent_coords_match(
             &segmentation_latent,
             &shape_latent))) {
        TRELLIS_ERROR(
            "mesh segmentation: generated segmentation and shape caches do not "
            "share the exact sparse coordinate/anchor contract");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    char shape_encoder_path[4096] = {0};
    char texture_encoder_path[4096] = {0};
    char shape_decoder_path[4096] = {0};
    char texture_decoder_path[4096] = {0};
    char segmentation_flow_path[4096] = {0};
    if ((!segmentation_cache_hit && !segmentation_component_path(
             &segmentation_package,
             "segmentation_flow",
             options->segmentation_flow_path,
             segmentation_flow_path)) ||
        !segmentation_component_path(
            &base_package,
            "texture_decoder",
            options->texture_decoder_path,
            texture_decoder_path)) {
        status = TRELLIS_STATUS_NOT_FOUND;
        goto cleanup;
    }
    if (!segmentation_component_path(
            &base_package,
            "shape_decoder",
            options->shape_decoder_path,
            shape_decoder_path)) {
        status = TRELLIS_STATUS_NOT_FOUND;
        goto cleanup;
    }
    if (!shape_cache_hit && !segmentation_component_path(
                   &base_package,
                   "shape_encoder",
                   options->shape_encoder_path,
                   shape_encoder_path)) {
        status = TRELLIS_STATUS_NOT_FOUND;
        goto cleanup;
    }
    if (!segmentation_cache_hit && !texture_cache_hit && !segmentation_component_path(
            &base_package,
            "texture_encoder",
            options->texture_encoder_path,
            texture_encoder_path)) {
        status = TRELLIS_STATUS_NOT_FOUND;
        goto cleanup;
    }

    if (!segmentation_cache_hit) {
        status = segmentation_require_readable_file(
            "segmentation flow weight",
            segmentation_flow_path);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    status = segmentation_require_readable_file(
        "texture decoder weight",
        texture_decoder_path);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = segmentation_require_readable_file(
        "shape decoder weight",
        shape_decoder_path);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    if (!shape_cache_hit) {
        status = segmentation_require_readable_file(
            "shape encoder weight",
            shape_encoder_path);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    if (!segmentation_cache_hit && !texture_cache_hit) {
        status = segmentation_require_readable_file(
            "texture encoder weight",
            texture_encoder_path);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    if (!segmentation_cache_hit) {
        char dino_weights_path[4096];
        status = trellis_make_model_path(
            options->dino_dir,
            "model.safetensors",
            dino_weights_path,
            sizeof(dino_weights_path));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "mesh segmentation: DINO weight path is invalid under '%s'",
                options->dino_dir);
            goto cleanup;
        }
        status = segmentation_require_readable_file(
            "DINO model weight",
            dino_weights_path);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }

    char gltf_error[512];
    gltf_error[0] = '\0';
    status = trellis_mesh_rigging_gltf_load(
        options->input_path,
        &asset,
        gltf_error,
        sizeof(gltf_error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: failed to load mesh '%s': %s%s%s",
            options->input_path,
            trellis_status_string(status),
            gltf_error[0] == '\0' ? "" : " ",
            gltf_error);
        goto cleanup;
    }
    const trellis_shape_latent_cache_info * cache_anchor = shape_cache_hit ?
        &shape_cache_info : (segmentation_cache_hit ?
            &segmentation_cache_info : (texture_cache_hit ? &texture_cache_info : NULL));
    status = segmentation_normalize_asset_mesh(
        &asset,
        cache_anchor,
        &normalized_mesh);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    int64_t stage_start_us = ggml_time_us();
    if (!shape_cache_hit || (!segmentation_cache_hit && !texture_cache_hit)) {
        trellis_flexible_dual_grid_options grid_options;
        trellis_flexible_dual_grid_options_default(&grid_options);
        for (int axis = 0; axis < 3; ++axis) {
            grid_options.grid_size[axis] = options->resolution;
        }
        status = trellis_mesh_to_flexible_dual_grid_host(
            normalized_mesh.vertices,
            normalized_mesh.n_vertices,
            normalized_mesh.faces,
            normalized_mesh.n_faces,
            &grid_options,
            &grid);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "mesh segmentation: mesh -> Flexible Dual Grid failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
        TRELLIS_INFO(
            "perf_stage name=segmentation_flexible_dual_grid ms=%.3f tokens=%lld",
            (double) (ggml_time_us() - stage_start_us) / 1000.0,
            (long long) grid.n);
        if (!shape_cache_hit) {
            status = segmentation_make_shape_features(
                &grid,
                &grid_options,
                &shape_encoder_input);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
        }
        if (!segmentation_cache_hit && !texture_cache_hit) {
            status = trellis_mesh_segmentation_material_features(
                options->input_path,
                &asset,
                &normalized_mesh,
                &grid,
                &texture_encoder_input);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR(
                    "mesh segmentation: PBR surface feature sampling failed: %s",
                    trellis_status_string(status));
                goto cleanup;
            }
        }
    }

    if (!segmentation_cache_hit) {
        if (options->condition_image_path != NULL &&
            options->condition_image_path[0] != '\0') {
            if (options->condition_image_prepared) {
                status = trellis_pipeline_adopt_prepared_condition_image(
                    options->condition_image_path,
                    &prepared_image);
            } else {
                status = trellis_pipeline_prepare_condition_image(
                    options->model_dir,
                    options->dino_dir,
                    options->condition_image_path,
                    options->birefnet_path,
                    graph_kind,
                    options->device,
                    1,
                    &prepared_image);
            }
        } else {
            status = trellis_mesh_segmentation_render_condition(
                &asset,
                512,
                &prepared_image,
                options->rendered_condition_output_path);
        }
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "mesh segmentation: condition image preparation failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
    }

    status = trellis_backend_init(&graph_backend, graph_kind, options->device);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: graph backend init failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    graph_backend_initialized = 1;
    if (sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        status = trellis_cuda_init(&cuda, options->device);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        cuda_initialized = 1;
    } else {
        status = trellis_sparse_vulkan_backend_create(options->device, &sparse_backend);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        if (!shape_cache_hit || (!segmentation_cache_hit && !texture_cache_hit)) {
            status = trellis_backend_init(&cpu_weight_backend, TRELLIS_BACKEND_CPU, 0);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            cpu_weight_backend_initialized = 1;
        }
    }
    TRELLIS_INFO(
        "mesh segmentation: graph=%s sparse=%s device=%d profile=512_full",
        trellis_backend_kind_name(graph_kind),
        segmentation_sparse_backend_name(sparse_kind),
        options->device);

    const trellis_backend_context * encoder_weight_backend =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : &cpu_weight_backend;
    if (!shape_cache_hit) {
        stage_start_us = ggml_time_us();
        status = segmentation_run_encoder(
            encoder_weight_backend,
            sparse_kind,
            options->device,
            sparse_backend,
            "shape encoder",
            shape_encoder_path,
            &grid,
            shape_encoder_input,
            options->resolution,
            &shape_latent);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        TRELLIS_INFO(
            "perf_stage name=segmentation_shape_encode ms=%.3f tokens=%lld",
            (double) (ggml_time_us() - stage_start_us) / 1000.0,
            (long long) shape_latent.n_coords);
        if (options->shape_latent_output_path != NULL &&
            options->shape_latent_output_path[0] != '\0') {
            status = trellis_shape_latent_cache_write(
                options->shape_latent_output_path,
                &shape_latent,
                &normalized_mesh,
                NULL);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
        }
    }

    /* SegviGen conditions its label decoder on the shape decoder's predicted
     * subdivisions. Encoder return_subs are reversed downsampling maps for
     * reconstruction and are not the decoder's learned to_subdiv decisions. */
    trellis_pipeline_mesh_options decoder_options;
    memset(&decoder_options, 0, sizeof(decoder_options));
    decoder_options.model_dir = options->model_dir;
    decoder_options.decoder_override_path = shape_decoder_path;
    decoder_options.latent = &shape_latent;
    decoder_options.resolution = options->resolution;
    decoder_options.cuda =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    decoder_options.sparse_backend_kind = sparse_kind;
    decoder_options.sparse_device = options->device;
    decoder_options.sparse_backend = sparse_backend;
    stage_start_us = ggml_time_us();
    status = trellis_pipeline_decode_shape_latent_mesh(
        &decoder_options,
        &guides,
        &cache_projection_mesh);
    if (status != TRELLIS_STATUS_OK || guides.n_levels != 4) {
        TRELLIS_ERROR(
            "mesh segmentation: shape guide decode failed: %s guides=%d",
            trellis_status_string(status),
            guides.n_levels);
        if (status == TRELLIS_STATUS_OK) status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=segmentation_shape_decode_guides ms=%.3f tokens=%lld",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) shape_latent.n_coords);
    /* Only predicted subdivision guides are needed by the label decoder. */
    trellis_mesh_free(&cache_projection_mesh);

    if (!segmentation_cache_hit && !texture_cache_hit) {
        stage_start_us = ggml_time_us();
        status = segmentation_run_encoder(
            encoder_weight_backend,
            sparse_kind,
            options->device,
            sparse_backend,
            "texture encoder",
            texture_encoder_path,
            &grid,
            texture_encoder_input,
            options->resolution,
            &source_texture_latent);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        TRELLIS_INFO(
            "perf_stage name=segmentation_texture_encode ms=%.3f tokens=%lld",
            (double) (ggml_time_us() - stage_start_us) / 1000.0,
            (long long) source_texture_latent.n_coords);
        if (options->texture_latent_output_path != NULL &&
            options->texture_latent_output_path[0] != '\0') {
            status = trellis_shape_latent_cache_write(
                options->texture_latent_output_path,
                &source_texture_latent,
                &normalized_mesh,
                NULL);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
        }
    }
    const trellis_structured_latent * paired_latent = segmentation_cache_hit ?
        &segmentation_latent : &source_texture_latent;
    if (!segmentation_latent_coords_match(&shape_latent, paired_latent)) {
        TRELLIS_ERROR(
            "mesh segmentation: shape and %s latent use different sparse "
            "coordinate orderings",
            segmentation_cache_hit ? "generated segmentation" : "source-texture");
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    free(shape_encoder_input);
    shape_encoder_input = NULL;
    free(texture_encoder_input);
    texture_encoder_input = NULL;
    trellis_flexible_dual_grid_free(&grid);
    if (cpu_weight_backend_initialized) {
        trellis_backend_free(&cpu_weight_backend);
        cpu_weight_backend_initialized = 0;
    }

    if (!segmentation_cache_hit) {
        trellis_image_condition_options condition_options;
        memset(&condition_options, 0, sizeof(condition_options));
        condition_options.model_dir = options->model_dir;
        condition_options.dino_dir = options->dino_dir;
        condition_options.image_path = prepared_image.source_path;
        condition_options.cond_resolution = options->resolution;
        condition_options.foreground_alpha_threshold = 204;
        condition_options.foreground_crop_scale = 1.0f;
        condition_options.backend = &graph_backend;
        condition_options.cuda =
            sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        stage_start_us = ggml_time_us();
        status = trellis_pipeline_run_image_condition(&condition_options, &condition);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        TRELLIS_INFO(
            "perf_stage name=segmentation_image_condition ms=%.3f tokens=%d",
            (double) (ggml_time_us() - stage_start_us) / 1000.0,
            condition.cond_tokens);

        const trellis_model_component_instance * flow_component =
            trellis_model_package_find_component(
                &segmentation_package,
                "segmentation_flow");
        trellis_ggml_attention_policy attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
        int package_emulates_bf16 = 0;
        status = trellis_image_to_3d_component_execution_policy(
            flow_component,
            &attention,
            &package_emulates_bf16);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        if (options->no_flash_attn) {
            attention.mode = TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
        }

        trellis_structured_latent_options flow_options;
        memset(&flow_options, 0, sizeof(flow_options));
        flow_options.model_dir = options->model_dir;
        flow_options.flow_override_path = segmentation_flow_path;
        flow_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
        flow_options.label = "segmentation";
        flow_options.normalization_key = "tex_slat_normalization";
        flow_options.coords_bxyz = shape_latent.coords_bxyz;
        flow_options.n_coords = shape_latent.n_coords;
        flow_options.cond = condition.cond;
        flow_options.cond_tokens = condition.cond_tokens;
        flow_options.concat_cond = shape_latent.feats;
        flow_options.concat_channels = shape_latent.channels;
        flow_options.paired_fixed_state = source_texture_latent.feats;
        flow_options.noise_seed = options->seed == 0 ? 1u : options->seed;
        flow_options.resolution = options->resolution;
        flow_options.steps = options->steps;
        flow_options.rescale_t = 3.0f;
        flow_options.guidance_strength = 1.0f;
        flow_options.guidance_rescale = 0.0f;
        flow_options.guidance_min = 0.6f;
        flow_options.guidance_max = 0.9f;
        flow_options.flow_blocks_override = options->flow_blocks_override;
        flow_options.flow_block_parts_override =
            options->flow_block_parts_override;
        flow_options.flow_no_rope = options->flow_no_rope;
        flow_options.emulate_bf16_blocks =
            options->emulate_bf16_blocks || package_emulates_bf16;
        flow_options.attention_policy = attention;
        flow_options.use_ggml_flash_attn =
            attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
        flow_options.backend = &graph_backend;
        flow_options.cuda =
            sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        stage_start_us = ggml_time_us();
        status = trellis_pipeline_run_structured_latent(
            &flow_options,
            &segmentation_latent);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        TRELLIS_INFO(
            "perf_stage name=segvigen_full_paired_flow ms=%.3f state_tokens=%lld transformer_tokens=%lld",
            (double) (ggml_time_us() - stage_start_us) / 1000.0,
            (long long) segmentation_latent.n_coords,
            (long long) segmentation_latent.n_coords * 2ll);
        trellis_image_condition_result_free(&condition);
        trellis_structured_latent_free(&source_texture_latent);
    } else {
        TRELLIS_INFO(
            "mesh segmentation: using generated segmentation latent cache; "
            "condition encoder and paired flow skipped");
    }
    if (options->segmentation_latent_output_path != NULL &&
        options->segmentation_latent_output_path[0] != '\0') {
        status = trellis_shape_latent_cache_write(
            options->segmentation_latent_output_path,
            &segmentation_latent,
            &normalized_mesh,
            NULL);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "mesh segmentation: generated segmentation latent export failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
    }
    trellis_pipeline_texture_options texture_decoder_options;
    memset(&texture_decoder_options, 0, sizeof(texture_decoder_options));
    texture_decoder_options.model_dir = options->model_dir;
    texture_decoder_options.decoder_override_path = texture_decoder_path;
    texture_decoder_options.latent = &segmentation_latent;
    texture_decoder_options.guide_subs = &guides;
    texture_decoder_options.cuda =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    texture_decoder_options.sparse_backend_kind = sparse_kind;
    texture_decoder_options.sparse_device = options->device;
    texture_decoder_options.sparse_backend = sparse_backend;
    stage_start_us = ggml_time_us();
    status = trellis_pipeline_decode_texture_latent_voxels(
        &texture_decoder_options,
        &decoded_labels);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    if (decoded_labels.n_coords <= 0 || decoded_labels.coords_bxyz == NULL ||
        decoded_labels.attrs == NULL || decoded_labels.channels < 3) {
        TRELLIS_ERROR(
            "mesh segmentation: label decoder returned an invalid tensor "
            "tokens=%lld channels=%d",
            (long long) decoded_labels.n_coords,
            decoded_labels.channels);
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=segmentation_label_decode ms=%.3f voxels=%lld channels=%d",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) decoded_labels.n_coords,
        decoded_labels.channels);
    trellis_structured_latent_free(&segmentation_latent);

    status = trellis_mesh_segmentation_labels_from_voxels(
        normalized_mesh.vertices,
        asset.vertex_count,
        asset.triangles,
        asset.triangle_count,
        decoded_labels.coords_bxyz,
        decoded_labels.attrs,
        decoded_labels.n_coords,
        decoded_labels.channels,
        options->resolution,
        (size_t) options->min_palette_voxels,
        options->palette_merge_distance,
        &semantic_labels,
        &semantic_count);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: voxel-to-face label assignment failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    status = trellis_mesh_segmentation_partition_faces_geometric_ex(
        normalized_mesh.vertices,
        asset.triangles,
        asset.triangle_count,
        asset.vertex_count,
        semantic_labels,
        (size_t) options->min_component_faces,
        1e-5f,
        small_part_mode,
        &face_part_ids,
        &part_count,
        &small_part_stats);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: geometric face partitioning failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    const char * small_part_mode_name =
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE ?
            "merge" :
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD ?
            "discard" : "keep";
    TRELLIS_INFO(
        "mesh segmentation: small_part_mode=%s scan=%s geometric_shells=%zu candidates=%zu "
        "candidate_parts=%zu affected_faces=%zu input_parts=%zu output_parts=%zu",
        small_part_mode_name,
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP ?
            "disabled" : "enabled",
        small_part_stats.geometric_shell_count,
        small_part_stats.candidate_shell_count,
        small_part_stats.candidate_part_count,
        small_part_stats.affected_face_count,
        small_part_stats.input_part_count,
        small_part_stats.output_part_count);
    gltf_error[0] = '\0';
    status = trellis_mesh_segmentation_write_parts_glb(
        options->output_path,
        &asset,
        face_part_ids,
        asset.triangle_count,
        part_count,
        NULL,
        gltf_error,
        sizeof(gltf_error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh segmentation: multi-part GLB export failed: %s%s%s",
            trellis_status_string(status),
            gltf_error[0] == '\0' ? "" : " ",
            gltf_error);
        goto cleanup;
    }
    TRELLIS_INFO(
        "mesh segmentation: wrote %s semantic_labels=%zu physical_parts=%zu "
        "source_faces=%zu output_faces=%zu discarded_faces=%zu",
        options->output_path,
        semantic_count,
        part_count,
        asset.triangle_count,
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD ?
            asset.triangle_count - small_part_stats.affected_face_count :
            asset.triangle_count,
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD ?
            small_part_stats.affected_face_count : 0u);

cleanup:
    free(face_part_ids);
    free(semantic_labels);
    free(texture_encoder_input);
    free(shape_encoder_input);
    trellis_pbr_voxels_free(&decoded_labels);
    trellis_structured_latent_free(&segmentation_latent);
    trellis_image_condition_result_free(&condition);
    trellis_structured_latent_free(&source_texture_latent);
    trellis_structured_latent_free(&shape_latent);
    trellis_sparse_c2s_guides_free(&guides);
    trellis_flexible_dual_grid_free(&grid);
    trellis_mesh_free(&cache_projection_mesh);
    trellis_mesh_free(&normalized_mesh);
    trellis_mesh_rigging_asset_free(&asset);
    if (cpu_weight_backend_initialized) trellis_backend_free(&cpu_weight_backend);
    if (cuda_initialized) trellis_cuda_free(&cuda);
    trellis_sparse_backend_destroy(sparse_backend);
    if (graph_backend_initialized) trellis_backend_free(&graph_backend);
    trellis_pipeline_prepared_condition_image_free(&prepared_image);
    trellis_model_package_free(&segmentation_package);
    trellis_model_package_free(&base_package);
    return status;
}
