#define CGLTF_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "trellis.h"
#include "trellis_platform.h"
#include "image_to_3d_internal.h"

#include "cgltf_write.h"
#include "stb_image_write.h"
#include "xatlas_c.h"

#ifndef TRELLIS_HAS_GLTF_VULKAN_BAKE
#define TRELLIS_HAS_GLTF_VULKAN_BAKE 0
#endif

#if TRELLIS_HAS_GLTF_VULKAN_BAKE
#include <vulkan/vulkan.h>

#include "trellis_gltf_bake_vert_spv.h"
#include "trellis_gltf_bake_frag_spv.h"
#include "trellis_gltf_downsample_spv.h"
#include "trellis_gltf_dilate_spv.h"
#include "trellis_gltf_fill_empty_spv.h"
#endif

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct trellis_gltf_mesh {
    float * positions;
    float * normals;
    float * uvs;
    uint32_t * indices;
    uint32_t * chart_map;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t chart_map_size;
} trellis_gltf_mesh;

#define TRELLIS_GLTF_UV_OUTPUT_PADDING 4u
#define TRELLIS_GLTF_UV_PACK_MAX_PASSES 8u
/* A 2x edge raster contains four coverage samples per output texel. */
#define TRELLIS_GLTF_BAKE_SUPERSAMPLE_SCALE 2u

typedef struct trellis_gltf_chart_record {
    uint32_t key;
    uint32_t face;
} trellis_gltf_chart_record;

typedef struct trellis_gltf_edge_record {
    uint64_t key;
    uint32_t face;
    uint32_t local_edge;
} trellis_gltf_edge_record;

typedef struct trellis_gltf_chart_input {
    float * positions;
    float * normals;
    uint32_t * indices;
    uint32_t * vertex_map;
    uint32_t vertex_count;
    uint32_t face_count;
} trellis_gltf_chart_input;

static int mkdir_p_export(const char * path) {
    return trellis_mkdir_p(path);
}

static int split_output_paths(
    const char * gltf_path,
    char * dir,
    size_t dir_size,
    char * stem,
    size_t stem_size) {
    if (gltf_path == NULL || dir == NULL || stem == NULL || dir_size == 0 || stem_size == 0) {
        return 0;
    }
    const char * slash = trellis_path_last_sep_const(gltf_path);
    const char * base = slash != NULL ? slash + 1 : gltf_path;
    if (slash != NULL) {
        size_t n = (size_t) (slash - gltf_path);
        if (n + 1u > dir_size) {
            return 0;
        }
        memcpy(dir, gltf_path, n);
        dir[n] = '\0';
    } else {
        if (dir_size < 2) {
            return 0;
        }
        dir[0] = '.';
        dir[1] = '\0';
    }
    const char * dot = strrchr(base, '.');
    size_t n = dot != NULL ? (size_t) (dot - base) : strlen(base);
    if (n == 0 || n + 1u > stem_size) {
        return 0;
    }
    memcpy(stem, base, n);
    stem[n] = '\0';
    return 1;
}

static int make_joined_path(char * dst, size_t dst_size, const char * dir, const char * name) {
    int n = snprintf(dst, dst_size, "%s/%s", dir, name);
    return n >= 0 && (size_t) n < dst_size;
}

static int make_named_file(
    char * uri,
    size_t uri_size,
    char * abs,
    size_t abs_size,
    const char * dir,
    const char * stem,
    const char * suffix) {
    int n = snprintf(uri, uri_size, "%s%s", stem, suffix);
    if (n < 0 || (size_t) n >= uri_size) {
        return 0;
    }
    return make_joined_path(abs, abs_size, dir, uri);
}

static int path_has_extension_ci(const char * path, const char * ext) {
    if (path == NULL || ext == NULL) return 0;
    size_t path_len = strlen(path);
    size_t ext_len = strlen(ext);
    if (path_len < ext_len) return 0;
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

static float clamp01_export(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static int next_pow2_export(size_t n, size_t * out) {
    size_t v = 1;
    while (v < n) {
        if (v > SIZE_MAX / 2u) {
            return 0;
        }
        v *= 2u;
    }
    *out = v;
    return 1;
}

static void mesh_compute_normals(const trellis_mesh_host * mesh, float * normals) {
    memset(normals, 0, (size_t) mesh->n_vertices * 3u * sizeof(float));
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        const float * a = mesh->vertices + (size_t) f[0] * 3u;
        const float * b = mesh->vertices + (size_t) f[1] * 3u;
        const float * c = mesh->vertices + (size_t) f[2] * 3u;
        float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        float n[3] = {
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        };
        for (int k = 0; k < 3; ++k) {
            float * dst = normals + (size_t) f[k] * 3u;
            dst[0] += n[0]; dst[1] += n[1]; dst[2] += n[2];
        }
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        float * n = normals + (size_t) i * 3u;
        float len = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 1e-20f) {
            n[0] /= len; n[1] /= len; n[2] /= len;
        } else {
            n[0] = 0.0f; n[1] = 1.0f; n[2] = 0.0f;
        }
    }
}

static void gltf_mesh_free(trellis_gltf_mesh * mesh) {
    if (mesh == NULL) {
        return;
    }
    free(mesh->positions);
    free(mesh->normals);
    free(mesh->uvs);
    free(mesh->indices);
    free(mesh->chart_map);
    memset(mesh, 0, sizeof(*mesh));
}

static void gltf_chart_input_free(trellis_gltf_chart_input * chart) {
    if (chart == NULL) return;
    free(chart->positions);
    free(chart->normals);
    free(chart->indices);
    free(chart->vertex_map);
    memset(chart, 0, sizeof(*chart));
}

static uint32_t gltf_uv_output_padding_for_chart_count(
    uint32_t chart_count,
    int texture_size) {
    const uint64_t output_texels =
        (uint64_t) (uint32_t) texture_size * (uint32_t) texture_size;
    uint32_t padding = TRELLIS_GLTF_UV_OUTPUT_PADDING;
    while (padding > 0u) {
        const uint64_t footprint_side = (uint64_t) padding * 2u + 3u;
        const uint64_t minimum_guard_texels =
            (uint64_t) chart_count * footprint_side * footprint_side;
        if (minimum_guard_texels <= output_texels / 2u) break;
        --padding;
    }
    return padding;
}

static trellis_status gltf_xatlas_pack_for_output(
    xatlasAtlas * atlas,
    int texture_size,
    uint32_t minimum_chart_count,
    uint32_t * atlas_padding_out) {
    if (atlas == NULL || texture_size <= 0 || atlas_padding_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    int64_t stage_start_us = trellis_now_us();
    xatlasComputeCharts(atlas, NULL);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_compute_charts ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);

    /* Probe without padding so fragmented meshes do not spend most texels on guards. */
    xatlasPackOptions probe_options;
    xatlasPackOptionsInit(&probe_options);
    probe_options.resolution = (uint32_t) texture_size;
    probe_options.padding = 0u;
    /* Keep the owner image only when the chart count proves that padding will
     * be zero. This avoids a duplicate pack for heavily fragmented atlases
     * without making normal 2K/4K probes build an image that a later padded
     * pack would immediately discard. */
    const uint32_t known_chart_count =
        atlas->chartCount > 0u ? atlas->chartCount : minimum_chart_count;
    const uint32_t guaranteed_padding = gltf_uv_output_padding_for_chart_count(
        known_chart_count,
        texture_size);
    probe_options.createImage =
        TRELLIS_HAS_GLTF_VULKAN_BAKE != 0 && guaranteed_padding == 0u;
    stage_start_us = trellis_now_us();
    xatlasPackCharts(atlas, &probe_options);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_pack_probe ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);
    if (atlas->atlasCount != 1u || atlas->width == 0u || atlas->height == 0u ||
        atlas->chartCount == 0u || atlas->meshes == NULL) {
        TRELLIS_ERROR(
            "glTF export: xatlas probe pack failed (pages=%u atlas=%ux%u charts=%u)",
            atlas->atlasCount,
            atlas->width,
            atlas->height,
            atlas->chartCount);
        return TRELLIS_STATUS_ERROR;
    }
    const uint32_t probe_max_dimension =
        atlas->width > atlas->height ? atlas->width : atlas->height;

    /* Keep at least half of the atlas budget for chart interiors and packing loss. */
    uint32_t target_output_padding = gltf_uv_output_padding_for_chart_count(
        atlas->chartCount,
        texture_size);
    if (target_output_padding < TRELLIS_GLTF_UV_OUTPUT_PADDING) {
        TRELLIS_INFO(
            "glTF export: UV islands=%u exceed %upx guard capacity at %d; using target=%upx",
            atlas->chartCount,
            TRELLIS_GLTF_UV_OUTPUT_PADDING,
            texture_size,
            target_output_padding);
    }

    uint32_t best_atlas_padding = 0u;
    uint32_t best_max_dimension = probe_max_dimension;
    uint32_t pack_passes = 1u;
    int reached_target = target_output_padding == 0u;
    if (reached_target &&
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
        atlas->image != NULL
#else
        1
#endif
    ) {
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
        /* The probe owner map is the final chart map. */
#endif
        *atlas_padding_out = 0u;
        TRELLIS_INFO(
            "glTF export: xatlas padding atlas=0 output=0.00x0.00 px target=0 passes=1");
        return TRELLIS_STATUS_OK;
    }
    const uint64_t initial_padding_numerator =
        (uint64_t) target_output_padding * probe_max_dimension +
        (uint32_t) texture_size - 1u;
    const uint64_t initial_padding =
        initial_padding_numerator / (uint32_t) texture_size;
    uint32_t atlas_padding = target_output_padding;
    if (initial_padding >= target_output_padding && initial_padding <= 256u) {
        atlas_padding = (uint32_t) initial_padding;
    }
    for (uint32_t pass = 0;
         target_output_padding > 0u && pass < TRELLIS_GLTF_UV_PACK_MAX_PASSES;
         ++pass) {
        xatlasPackOptions pack_options;
        xatlasPackOptionsInit(&pack_options);
        pack_options.resolution = (uint32_t) texture_size;
        pack_options.padding = atlas_padding;
        pack_options.createImage = TRELLIS_HAS_GLTF_VULKAN_BAKE != 0;
        stage_start_us = trellis_now_us();
        xatlasPackCharts(atlas, &pack_options);
        TRELLIS_INFO(
            "perf_stage name=gltf_uv_pack_candidate ms=%.3f pass=%u padding=%u",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            pass + 1u,
            atlas_padding);
        ++pack_passes;
        if (atlas->atlasCount != 1u || atlas->width == 0u || atlas->height == 0u ||
            atlas->meshes == NULL) {
            TRELLIS_ERROR(
                "glTF export: xatlas candidate pack failed (pages=%u atlas=%ux%u)",
                atlas->atlasCount,
                atlas->width,
                atlas->height);
            return TRELLIS_STATUS_ERROR;
        }

        const uint32_t max_dimension = atlas->width > atlas->height ? atlas->width : atlas->height;
        if ((uint64_t) max_dimension > (uint64_t) probe_max_dimension * 2u) {
            if (atlas_padding > target_output_padding) {
                atlas_padding = target_output_padding;
                continue;
            }
            break;
        }
        if ((uint64_t) atlas_padding * best_max_dimension >
            (uint64_t) best_atlas_padding * max_dimension) {
            best_atlas_padding = atlas_padding;
            best_max_dimension = max_dimension;
        }

        /* xatlas resolution is a density target, so convert atlas pixels to output pixels. */
        if ((uint64_t) atlas_padding * (uint32_t) texture_size >=
            (uint64_t) target_output_padding * max_dimension) {
            reached_target = 1;
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
            if (atlas->image == NULL) break;
#endif
            const float output_padding_x =
                (float) atlas_padding * (float) texture_size / (float) atlas->width;
            const float output_padding_y =
                (float) atlas_padding * (float) texture_size / (float) atlas->height;
            *atlas_padding_out = atlas_padding;
            TRELLIS_INFO(
                "glTF export: xatlas padding atlas=%u output=%.2fx%.2f px target=%u passes=%u",
                atlas_padding,
                output_padding_x,
                output_padding_y,
                target_output_padding,
                pack_passes);
            return TRELLIS_STATUS_OK;
        }
        const uint64_t required_numerator =
            (uint64_t) target_output_padding * max_dimension +
            (uint32_t) texture_size - 1u;
        uint64_t required = required_numerator / (uint32_t) texture_size;
        if (required <= atlas_padding) required = (uint64_t) atlas_padding + 1u;
        if (required > 256u) break;
        atlas_padding = (uint32_t) required;
    }

    /* Fall back to the best candidate if no pass reached the requested output guard. */
    xatlasPackOptions final_options;
    xatlasPackOptionsInit(&final_options);
    final_options.resolution = (uint32_t) texture_size;
    final_options.padding = best_atlas_padding;
    final_options.createImage = TRELLIS_HAS_GLTF_VULKAN_BAKE != 0;
    stage_start_us = trellis_now_us();
    xatlasPackCharts(atlas, &final_options);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_pack_final ms=%.3f padding=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        best_atlas_padding);
    ++pack_passes;
    if (atlas->atlasCount != 1u || atlas->width == 0u || atlas->height == 0u ||
        atlas->meshes == NULL ||
        (TRELLIS_HAS_GLTF_VULKAN_BAKE && atlas->image == NULL)) {
        TRELLIS_ERROR(
            "glTF export: xatlas final pack failed (pages=%u atlas=%ux%u)",
            atlas->atlasCount,
            atlas->width,
            atlas->height);
        return TRELLIS_STATUS_ERROR;
    }

    const float output_padding_x =
        (float) best_atlas_padding * (float) texture_size / (float) atlas->width;
    const float output_padding_y =
        (float) best_atlas_padding * (float) texture_size / (float) atlas->height;
    if (!reached_target) {
        TRELLIS_INFO(
            "glTF export: UV padding best-effort target=%u actual=%.2fx%.2f px",
            target_output_padding,
            output_padding_x,
            output_padding_y);
    }
    *atlas_padding_out = best_atlas_padding;
    TRELLIS_INFO(
        "glTF export: xatlas padding atlas=%u output=%.2fx%.2f px target=%u passes=%u",
        best_atlas_padding,
        output_padding_x,
        output_padding_y,
        target_output_padding,
        pack_passes);
    return TRELLIS_STATUS_OK;
}

static trellis_status gltf_copy_xatlas_chart_map(
    const xatlasAtlas * atlas,
    int texture_size,
    trellis_gltf_mesh * out) {
    if (atlas == NULL || out == NULL || texture_size <= 0 || atlas->atlasCount != 1u ||
        atlas->width == 0u || atlas->height == 0u || atlas->image == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t map_size = (size_t) texture_size * (size_t) texture_size;
    if (map_size > SIZE_MAX / sizeof(uint32_t)) return TRELLIS_STATUS_OUT_OF_MEMORY;
    uint32_t * chart_map = (uint32_t *) malloc(map_size * sizeof(uint32_t));
    if (chart_map == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;

    /* Sample at output texel centers to match the normalized UV raster transform. */
    for (uint32_t y = 0; y < (uint32_t) texture_size; ++y) {
        uint32_t sy = (uint32_t) (
            ((uint64_t) y * 2u + 1u) * (uint64_t) atlas->height /
            ((uint64_t) texture_size * 2u));
        if (sy >= atlas->height) sy = atlas->height - 1u;
        for (uint32_t x = 0; x < (uint32_t) texture_size; ++x) {
            uint32_t sx = (uint32_t) (
                ((uint64_t) x * 2u + 1u) * (uint64_t) atlas->width /
                ((uint64_t) texture_size * 2u));
            if (sx >= atlas->width) sx = atlas->width - 1u;
            const uint32_t encoded = atlas->image[(size_t) sy * atlas->width + sx];
            chart_map[(size_t) y * (uint32_t) texture_size + x] =
                (encoded & xatlasImageHasChartIndexBit) != 0u
                    ? (encoded & xatlasImageChartIndexMask) + 1u
                    : 0u;
        }
    }
    out->chart_map = chart_map;
    out->chart_map_size = (uint32_t) texture_size;
    return TRELLIS_STATUS_OK;
}

static trellis_status unwrap_mesh_xatlas_direct(
    const trellis_mesh_host * mesh,
    int texture_size,
    trellis_gltf_mesh * out) {
    if (mesh == NULL || out == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX / 3) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    float * input_normals = (float *) malloc((size_t) mesh->n_vertices * 3u * sizeof(float));
    uint32_t * input_indices = (uint32_t *) malloc((size_t) mesh->n_faces * 3u * sizeof(uint32_t));
    if (input_normals == NULL || input_indices == NULL) {
        free(input_normals);
        free(input_indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    mesh_compute_normals(mesh, input_normals);
    for (int64_t i = 0; i < mesh->n_faces * 3; ++i) {
        input_indices[i] = (uint32_t) mesh->faces[i];
    }

    /* Small meshes use a single direct CPU xatlas input. */
    xatlasSetPrint(NULL, false);
    xatlasAtlas * atlas = xatlasCreate();
    if (atlas == NULL) {
        free(input_normals);
        free(input_indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    xatlasMeshDecl decl;
    xatlasMeshDeclInit(&decl);
    decl.vertexCount = (uint32_t) mesh->n_vertices;
    decl.vertexPositionData = mesh->vertices;
    decl.vertexPositionStride = 3u * sizeof(float);
    decl.vertexNormalData = input_normals;
    decl.vertexNormalStride = 3u * sizeof(float);
    decl.indexCount = (uint32_t) mesh->n_faces * 3u;
    decl.indexData = input_indices;
    decl.indexFormat = XATLAS_INDEX_FORMAT_UINT32;
    xatlasAddMeshError add_error = xatlasAddMesh(atlas, &decl, 1);
    if (add_error != XATLAS_ADD_MESH_ERROR_SUCCESS) {
        TRELLIS_ERROR("glTF export: xatlas AddMesh failed: %s", xatlasAddMeshErrorString(add_error));
        xatlasDestroy(atlas);
        free(input_normals);
        free(input_indices);
        return TRELLIS_STATUS_ERROR;
    }
    uint32_t atlas_padding = 0;
    trellis_status pack_status = gltf_xatlas_pack_for_output(
        atlas,
        texture_size,
        1u,
        &atlas_padding);
    if (pack_status != TRELLIS_STATUS_OK) {
        xatlasDestroy(atlas);
        free(input_normals);
        free(input_indices);
        return pack_status;
    }
    if (atlas->meshCount != 1 || atlas->meshes == NULL || atlas->meshes[0].vertexCount == 0) {
        xatlasDestroy(atlas);
        free(input_normals);
        free(input_indices);
        return TRELLIS_STATUS_ERROR;
    }
    const xatlasMesh * xm = &atlas->meshes[0];
    out->vertex_count = xm->vertexCount;
    out->index_count = xm->indexCount;
    out->positions = (float *) malloc((size_t) out->vertex_count * 3u * sizeof(float));
    out->normals = (float *) malloc((size_t) out->vertex_count * 3u * sizeof(float));
    out->uvs = (float *) malloc((size_t) out->vertex_count * 2u * sizeof(float));
    out->indices = (uint32_t *) malloc((size_t) out->index_count * sizeof(uint32_t));
    if (out->positions == NULL || out->normals == NULL || out->uvs == NULL || out->indices == NULL) {
        gltf_mesh_free(out);
        xatlasDestroy(atlas);
        free(input_normals);
        free(input_indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const float atlas_w = atlas->width > 0 ? (float) atlas->width : (float) texture_size;
    const float atlas_h = atlas->height > 0 ? (float) atlas->height : (float) texture_size;
    for (uint32_t i = 0; i < out->vertex_count; ++i) {
        const xatlasVertex * v = &xm->vertexArray[i];
        const uint32_t src = v->xref;
        memcpy(out->positions + (size_t) i * 3u, mesh->vertices + (size_t) src * 3u, 3u * sizeof(float));
        memcpy(out->normals + (size_t) i * 3u, input_normals + (size_t) src * 3u, 3u * sizeof(float));
        out->uvs[(size_t) i * 2u + 0u] = clamp01_export(v->uv[0] / atlas_w);
        out->uvs[(size_t) i * 2u + 1u] = clamp01_export(v->uv[1] / atlas_h);
    }
    memcpy(out->indices, xm->indexArray, (size_t) out->index_count * sizeof(uint32_t));
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
    trellis_status chart_map_status = gltf_copy_xatlas_chart_map(atlas, texture_size, out);
    if (chart_map_status != TRELLIS_STATUS_OK) {
        gltf_mesh_free(out);
        xatlasDestroy(atlas);
        free(input_normals);
        free(input_indices);
        return chart_map_status;
    }
#endif
    TRELLIS_INFO(
        "glTF export: xatlas unwrap vertices=%u faces=%u atlas=%ux%u charts=%u padding=%u",
        out->vertex_count,
        out->index_count / 3u,
        atlas->width,
        atlas->height,
        atlas->chartCount,
        atlas_padding);
    xatlasDestroy(atlas);
    free(input_normals);
    free(input_indices);
    return TRELLIS_STATUS_OK;
}

#if TRELLIS_HAS_GLTF_VULKAN_BAKE
typedef struct trellis_gltf_vk_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t bytes;
} trellis_gltf_vk_buffer;

typedef struct trellis_gltf_vk_image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
} trellis_gltf_vk_image;

typedef struct trellis_gltf_vk {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkRenderPass render_pass;
    VkPipeline graphics_pipeline;
    VkPipeline compute_pipelines[3];
    VkDescriptorPool descriptor_pool;
} trellis_gltf_vk;

typedef enum trellis_gltf_vk_compute_pipeline {
    TRELLIS_GLTF_COMPUTE_DOWNSAMPLE = 0,
    TRELLIS_GLTF_COMPUTE_DILATE = 1,
    TRELLIS_GLTF_COMPUTE_FILL_EMPTY = 2,
    TRELLIS_GLTF_COMPUTE_COUNT = 3,
} trellis_gltf_vk_compute_pipeline;

typedef struct trellis_gltf_vk_shader {
    const unsigned char * data;
    unsigned int len;
    const char * name;
} trellis_gltf_vk_shader;

#define TRELLIS_GLTF_SHADER_DESC(data_, name_) { data_, (unsigned int) sizeof(data_), name_ }

static const trellis_gltf_vk_shader g_gltf_bake_vert_shader = {
    trellis_gltf_bake_vert_spv,
    (unsigned int) sizeof(trellis_gltf_bake_vert_spv),
    "gltf_bake_vert",
};

static const trellis_gltf_vk_shader g_gltf_bake_frag_shader = {
    trellis_gltf_bake_frag_spv,
    (unsigned int) sizeof(trellis_gltf_bake_frag_spv),
    "gltf_bake_frag",
};

static const trellis_gltf_vk_shader g_gltf_compute_shaders[TRELLIS_GLTF_COMPUTE_COUNT] = {
    TRELLIS_GLTF_SHADER_DESC(trellis_gltf_downsample_spv, "gltf_downsample"),
    TRELLIS_GLTF_SHADER_DESC(trellis_gltf_dilate_spv, "gltf_dilate"),
    TRELLIS_GLTF_SHADER_DESC(trellis_gltf_fill_empty_spv, "gltf_fill_empty"),
};

#undef TRELLIS_GLTF_SHADER_DESC

typedef struct trellis_gltf_vk_push {
    uint32_t texture_size;
    uint32_t triangle_count;
    uint32_t voxel_count;
    uint32_t channels;
    uint32_t hash_size;
    uint32_t resolution;
    uint32_t pad0;
    uint32_t pad1;
    float bounds_min_x;
    float bounds_min_y;
    float bounds_min_z;
    float bounds_inv_extent;
    uint32_t chart_grid;
    uint32_t chart_target_faces;
    uint32_t chart_normal_bins;
    uint32_t chart_flags;
    uint32_t project_vertex_count;
    uint32_t project_face_count;
    uint32_t project_node_count;
    uint32_t project_flags;
} trellis_gltf_vk_push;

typedef struct trellis_gltf_bvh_tri {
    float bmin[3];
    float bmax[3];
    float centroid[3];
    uint32_t face;
} trellis_gltf_bvh_tri;

typedef struct trellis_gltf_bvh_node {
    float bmin[3];
    float bmax[3];
    uint32_t left;
    uint32_t meta;
} trellis_gltf_bvh_node;

static uint32_t u32_from_float_export(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = value;
    return v.u;
}

static void gltf_bvh_bounds_init(float bmin[3], float bmax[3]) {
    bmin[0] = bmin[1] = bmin[2] = FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;
}

static void gltf_bvh_bounds_add(float bmin[3], float bmax[3], const float pmin[3], const float pmax[3]) {
    for (int i = 0; i < 3; ++i) {
        if (pmin[i] < bmin[i]) bmin[i] = pmin[i];
        if (pmax[i] > bmax[i]) bmax[i] = pmax[i];
    }
}

static uint32_t gltf_bvh_build_recursive(
    trellis_gltf_bvh_tri * tris,
    int start,
    int count,
    trellis_gltf_bvh_node * nodes,
    uint32_t * node_count) {
    uint32_t node_id = (*node_count)++;
    trellis_gltf_bvh_node * node = &nodes[node_id];
    gltf_bvh_bounds_init(node->bmin, node->bmax);
    for (int i = start; i < start + count; ++i) {
        gltf_bvh_bounds_add(node->bmin, node->bmax, tris[i].bmin, tris[i].bmax);
    }
    if (count <= 4) {
        node->left = (uint32_t) start;
        node->meta = 0x80000000u | (uint32_t) count;
        return node_id;
    }

    float cmin[3];
    float cmax[3];
    gltf_bvh_bounds_init(cmin, cmax);
    for (int i = start; i < start + count; ++i) {
        gltf_bvh_bounds_add(cmin, cmax, tris[i].centroid, tris[i].centroid);
    }
    float ex[3] = { cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2] };
    int axis = 0;
    if (ex[1] > ex[axis]) axis = 1;
    if (ex[2] > ex[axis]) axis = 2;
    float split = 0.5f * (cmin[axis] + cmax[axis]);
    int mid = start;
    for (int i = start; i < start + count; ++i) {
        if (tris[i].centroid[axis] < split) {
            trellis_gltf_bvh_tri tmp = tris[mid];
            tris[mid] = tris[i];
            tris[i] = tmp;
            ++mid;
        }
    }
    int left_count = mid - start;
    if (left_count <= 0 || left_count >= count) left_count = count / 2;
    uint32_t left = gltf_bvh_build_recursive(tris, start, left_count, nodes, node_count);
    uint32_t right = gltf_bvh_build_recursive(tris, start + left_count, count - left_count, nodes, node_count);
    node->left = left;
    node->meta = right;
    return node_id;
}

static trellis_status build_projection_bvh_buffer(
    const trellis_mesh_host * mesh,
    uint32_t ** words_out,
    size_t * word_count_out,
    uint32_t * vertex_count_out,
    uint32_t * face_count_out,
    uint32_t * node_count_out) {
    *words_out = NULL;
    *word_count_out = 0;
    *vertex_count_out = 0;
    *face_count_out = 0;
    *node_count_out = 0;
    if (mesh == NULL) {
        return TRELLIS_STATUS_OK;
    }
    if (mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_gltf_bvh_tri * tris =
        (trellis_gltf_bvh_tri *) malloc((size_t) mesh->n_faces * sizeof(*tris));
    trellis_gltf_bvh_node * nodes =
        (trellis_gltf_bvh_node *) malloc((size_t) mesh->n_faces * 2u * sizeof(*nodes));
    uint32_t * tri_indices =
        (uint32_t *) malloc((size_t) mesh->n_faces * sizeof(*tri_indices));
    if (tris == NULL || nodes == NULL || tri_indices == NULL) {
        free(tris);
        free(nodes);
        free(tri_indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * face = mesh->faces + (size_t) i * 3u;
        gltf_bvh_bounds_init(tris[i].bmin, tris[i].bmax);
        float sum[3] = { 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < 3; ++k) {
            if (face[k] < 0 || (int64_t) face[k] >= mesh->n_vertices) {
                free(tris);
                free(nodes);
                free(tri_indices);
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
            const float * v = mesh->vertices + (size_t) face[k] * 3u;
            gltf_bvh_bounds_add(tris[i].bmin, tris[i].bmax, v, v);
            sum[0] += v[0];
            sum[1] += v[1];
            sum[2] += v[2];
        }
        tris[i].centroid[0] = sum[0] / 3.0f;
        tris[i].centroid[1] = sum[1] / 3.0f;
        tris[i].centroid[2] = sum[2] / 3.0f;
        tris[i].face = (uint32_t) i;
    }

    uint32_t node_count = 0;
    (void) gltf_bvh_build_recursive(tris, 0, (int) mesh->n_faces, nodes, &node_count);
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        tri_indices[i] = tris[i].face;
    }
    free(tris);

    const size_t vertex_words = (size_t) mesh->n_vertices * 3u;
    const size_t face_words = (size_t) mesh->n_faces * 3u;
    const size_t node_words = (size_t) node_count * 8u;
    const size_t tri_words = (size_t) mesh->n_faces;
    const size_t total_words = vertex_words + face_words + node_words + tri_words;
    uint32_t * words = (uint32_t *) malloc(total_words * sizeof(uint32_t));
    if (words == NULL) {
        free(nodes);
        free(tri_indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < mesh->n_vertices * 3; ++i) {
        words[i] = u32_from_float_export(mesh->vertices[i]);
    }
    size_t face_base = vertex_words;
    for (int64_t i = 0; i < mesh->n_faces * 3; ++i) {
        words[face_base + (size_t) i] = (uint32_t) mesh->faces[i];
    }
    size_t node_base = face_base + face_words;
    for (uint32_t i = 0; i < node_count; ++i) {
        size_t base = node_base + (size_t) i * 8u;
        words[base + 0u] = u32_from_float_export(nodes[i].bmin[0]);
        words[base + 1u] = u32_from_float_export(nodes[i].bmin[1]);
        words[base + 2u] = u32_from_float_export(nodes[i].bmin[2]);
        words[base + 3u] = nodes[i].left;
        words[base + 4u] = u32_from_float_export(nodes[i].bmax[0]);
        words[base + 5u] = u32_from_float_export(nodes[i].bmax[1]);
        words[base + 6u] = u32_from_float_export(nodes[i].bmax[2]);
        words[base + 7u] = nodes[i].meta;
    }
    memcpy(words + node_base + node_words, tri_indices, tri_words * sizeof(uint32_t));
    free(nodes);
    free(tri_indices);

    *words_out = words;
    *word_count_out = total_words;
    *vertex_count_out = (uint32_t) mesh->n_vertices;
    *face_count_out = (uint32_t) mesh->n_faces;
    *node_count_out = node_count;
    return TRELLIS_STATUS_OK;
}

static uint32_t hash_coord_export32(int32_t x, int32_t y, int32_t z) {
    uint32_t h = (uint32_t) x * 73856093u ^ (uint32_t) y * 19349663u ^ (uint32_t) z * 83492791u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static void gltf_vk_buffer_destroy(trellis_gltf_vk * vk, trellis_gltf_vk_buffer * b) {
    if (vk == NULL || b == NULL) return;
    if (b->mapped != NULL) vkUnmapMemory(vk->device, b->memory);
    if (b->buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk->device, b->buffer, NULL);
    if (b->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static void gltf_vk_image_destroy(trellis_gltf_vk * vk, trellis_gltf_vk_image * image) {
    if (vk == NULL || image == NULL) return;
    if (image->view != VK_NULL_HANDLE) vkDestroyImageView(vk->device, image->view, NULL);
    if (image->image != VK_NULL_HANDLE) vkDestroyImage(vk->device, image->image, NULL);
    if (image->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, image->memory, NULL);
    memset(image, 0, sizeof(*image));
}

static void gltf_vk_destroy(trellis_gltf_vk * vk) {
    if (vk == NULL) return;
    if (vk->device != VK_NULL_HANDLE) vkDeviceWaitIdle(vk->device);
    if (vk->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    if (vk->graphics_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->graphics_pipeline, NULL);
    for (uint32_t i = 0; i < TRELLIS_GLTF_COMPUTE_COUNT; ++i) {
        if (vk->compute_pipelines[i] != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->compute_pipelines[i], NULL);
    }
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->render_pass != VK_NULL_HANDLE) vkDestroyRenderPass(vk->device, vk->render_pass, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

static uint32_t gltf_vk_find_memory_type(VkPhysicalDevice physical, uint32_t type_bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int gltf_vk_buffer_memory_try(
    trellis_gltf_vk * vk,
    trellis_gltf_vk_buffer * out,
    const VkMemoryRequirements * req,
    uint32_t memory_type) {
    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req->size;
    alloc_info.memoryTypeIndex = memory_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    void * mapped = NULL;
    if (vkAllocateMemory(vk->device, &alloc_info, NULL, &memory) != VK_SUCCESS) return 0;
    if (vkMapMemory(vk->device, memory, 0, req->size, 0, &mapped) != VK_SUCCESS) {
        vkFreeMemory(vk->device, memory, NULL);
        return 0;
    }
    if (vkBindBufferMemory(vk->device, out->buffer, memory, 0) != VK_SUCCESS) {
        vkUnmapMemory(vk->device, memory);
        vkFreeMemory(vk->device, memory, NULL);
        return 0;
    }
    out->memory = memory;
    out->mapped = mapped;
    return 1;
}

static int gltf_vk_buffer_create_impl(
    trellis_gltf_vk * vk,
    size_t bytes,
    int host_readback,
    trellis_gltf_vk_buffer * out) {
    memset(out, 0, sizeof(*out));
    if (bytes == 0) bytes = 4;
    out->bytes = bytes;
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = (VkDeviceSize) bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk->device, &buffer_info, NULL, &out->buffer) != VK_SUCCESS) return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, out->buffer, &req);
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &props);
    const VkMemoryPropertyFlags host_flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint32_t pass_count = host_readback ? 3u : 2u;
    for (uint32_t pass = 0; pass < pass_count; ++pass) {
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
            const int device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            const int host_cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
            int preferred = 0;
            if (host_readback) {
                preferred =
                    (pass == 0u && !device_local && host_cached) ||
                    (pass == 1u && !device_local && !host_cached) ||
                    (pass == 2u && device_local);
            } else {
                preferred = (pass == 0u && device_local) || (pass == 1u && !device_local);
            }
            if ((req.memoryTypeBits & (1u << i)) == 0 ||
                (flags & host_flags) != host_flags || !preferred) {
                continue;
            }
            if (gltf_vk_buffer_memory_try(vk, out, &req, i)) {
                /* Every bake buffer is fully uploaded or overwritten before its first read. */
                return 1;
            }
        }
    }
    gltf_vk_buffer_destroy(vk, out);
    return 0;
}

static int gltf_vk_buffer_create(
    trellis_gltf_vk * vk,
    size_t bytes,
    trellis_gltf_vk_buffer * out) {
    return gltf_vk_buffer_create_impl(vk, bytes, 0, out);
}

static int gltf_vk_readback_buffer_create(
    trellis_gltf_vk * vk,
    size_t bytes,
    trellis_gltf_vk_buffer * out) {
    return gltf_vk_buffer_create_impl(vk, bytes, 1, out);
}

static int gltf_vk_image_create(
    trellis_gltf_vk * vk,
    uint32_t width,
    uint32_t height,
    trellis_gltf_vk_image * out) {
    memset(out, 0, sizeof(*out));
    VkImageCreateInfo image_info;
    memset(&image_info, 0, sizeof(image_info));
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(vk->device, &image_info, NULL, &out->image) != VK_SUCCESS) return 0;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk->device, out->image, &req);
    uint32_t memory_type = gltf_vk_find_memory_type(
        vk->physical_device,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == UINT32_MAX) {
        gltf_vk_image_destroy(vk, out);
        return 0;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(vk->device, &alloc_info, NULL, &out->memory) != VK_SUCCESS ||
        vkBindImageMemory(vk->device, out->image, out->memory, 0) != VK_SUCCESS) {
        gltf_vk_image_destroy(vk, out);
        return 0;
    }

    VkImageViewCreateInfo view_info;
    memset(&view_info, 0, sizeof(view_info));
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = out->image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(vk->device, &view_info, NULL, &out->view) != VK_SUCCESS) {
        gltf_vk_image_destroy(vk, out);
        return 0;
    }
    return 1;
}

static int gltf_vk_create_compute_pipeline(
    trellis_gltf_vk * vk,
    const trellis_gltf_vk_shader * shader,
    VkPipeline * pipeline) {
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shader->len;
    shader_info.pCode = (const uint32_t *) shader->data;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk->device, &shader_info, NULL, &module) != VK_SUCCESS) {
        TRELLIS_ERROR("glTF export: failed to create Vulkan shader module %s", shader->name);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = vk->pipeline_layout;
    VkResult result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
    vkDestroyShaderModule(vk->device, module, NULL);
    return result == VK_SUCCESS;
}

static int gltf_vk_create_shader_module(
    trellis_gltf_vk * vk,
    const trellis_gltf_vk_shader * shader,
    VkShaderModule * module) {
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shader->len;
    shader_info.pCode = (const uint32_t *) shader->data;
    if (vkCreateShaderModule(vk->device, &shader_info, NULL, module) != VK_SUCCESS) {
        TRELLIS_ERROR("glTF export: failed to create Vulkan shader module %s", shader->name);
        return 0;
    }
    return 1;
}

static int gltf_vk_create_render_pass(trellis_gltf_vk * vk) {
    VkAttachmentDescription attachments[2];
    memset(attachments, 0, sizeof(attachments));
    for (uint32_t i = 0; i < 2; ++i) {
        attachments[i].format = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[i].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[i].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }

    VkAttachmentReference color_refs[2];
    memset(color_refs, 0, sizeof(color_refs));
    color_refs[0].attachment = 0;
    color_refs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_refs[1].attachment = 1;
    color_refs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass;
    memset(&subpass, 0, sizeof(subpass));
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 2;
    subpass.pColorAttachments = color_refs;

    VkSubpassDependency deps[2];
    memset(deps, 0, sizeof(deps));
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo render_pass_info;
    memset(&render_pass_info, 0, sizeof(render_pass_info));
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 2;
    render_pass_info.pAttachments = attachments;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 2;
    render_pass_info.pDependencies = deps;
    return vkCreateRenderPass(vk->device, &render_pass_info, NULL, &vk->render_pass) == VK_SUCCESS;
}

static int gltf_vk_create_graphics_pipeline(trellis_gltf_vk * vk) {
    VkShaderModule vert_module = VK_NULL_HANDLE;
    VkShaderModule frag_module = VK_NULL_HANDLE;
    if (!gltf_vk_create_shader_module(vk, &g_gltf_bake_vert_shader, &vert_module) ||
        !gltf_vk_create_shader_module(vk, &g_gltf_bake_frag_shader, &frag_module)) {
        if (vert_module != VK_NULL_HANDLE) vkDestroyShaderModule(vk->device, vert_module, NULL);
        if (frag_module != VK_NULL_HANDLE) vkDestroyShaderModule(vk->device, frag_module, NULL);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stages[2];
    memset(stages, 0, sizeof(stages));
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_bindings[2];
    memset(vertex_bindings, 0, sizeof(vertex_bindings));
    vertex_bindings[0].binding = 0;
    vertex_bindings[0].stride = 3u * sizeof(float);
    vertex_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vertex_bindings[1].binding = 1;
    vertex_bindings[1].stride = 2u * sizeof(float);
    vertex_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attrs[2];
    memset(vertex_attrs, 0, sizeof(vertex_attrs));
    vertex_attrs[0].location = 0;
    vertex_attrs[0].binding = 0;
    vertex_attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vertex_attrs[0].offset = 0;
    vertex_attrs[1].location = 1;
    vertex_attrs[1].binding = 1;
    vertex_attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
    vertex_attrs[1].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input;
    memset(&vertex_input, 0, sizeof(vertex_input));
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 2;
    vertex_input.pVertexBindingDescriptions = vertex_bindings;
    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions = vertex_attrs;

    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    memset(&input_assembly, 0, sizeof(input_assembly));
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state;
    memset(&viewport_state, 0, sizeof(viewport_state));
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer;
    memset(&rasterizer, 0, sizeof(rasterizer));
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample;
    memset(&multisample, 0, sizeof(multisample));
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    memset(&depth_stencil, 0, sizeof(depth_stencil));
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendAttachmentState blend_attachments[2];
    memset(blend_attachments, 0, sizeof(blend_attachments));
    for (uint32_t i = 0; i < 2; ++i) {
        blend_attachments[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }

    VkPipelineColorBlendStateCreateInfo blend_state;
    memset(&blend_state, 0, sizeof(blend_state));
    blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_state.attachmentCount = 2;
    blend_state.pAttachments = blend_attachments;

    VkDynamicState dynamic_states[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state;
    memset(&dynamic_state, 0, sizeof(dynamic_state));
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &blend_state;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = vk->pipeline_layout;
    pipeline_info.renderPass = vk->render_pass;
    pipeline_info.subpass = 0;

    VkResult result = vkCreateGraphicsPipelines(
        vk->device,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        NULL,
        &vk->graphics_pipeline);
    vkDestroyShaderModule(vk->device, frag_module, NULL);
    vkDestroyShaderModule(vk->device, vert_module, NULL);
    return result == VK_SUCCESS;
}

static int gltf_vk_init(trellis_gltf_vk * vk, int device_index) {
    memset(vk, 0, sizeof(*vk));
    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "trellis2-gltf-bake";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    if (vkCreateInstance(&instance_info, NULL, &vk->instance) != VK_SUCCESS) return 0;

    uint32_t physical_count = 0;
    if (vkEnumeratePhysicalDevices(vk->instance, &physical_count, NULL) != VK_SUCCESS || physical_count == 0) {
        gltf_vk_destroy(vk);
        return 0;
    }
    VkPhysicalDevice * physicals = (VkPhysicalDevice *) malloc((size_t) physical_count * sizeof(VkPhysicalDevice));
    if (physicals == NULL) {
        gltf_vk_destroy(vk);
        return 0;
    }
    if (vkEnumeratePhysicalDevices(vk->instance, &physical_count, physicals) != VK_SUCCESS) {
        free(physicals);
        gltf_vk_destroy(vk);
        return 0;
    }
    if (device_index < 0 || (uint32_t) device_index >= physical_count) {
        TRELLIS_ERROR(
            "glTF export: Vulkan device index %d is out of range (available devices: %u)",
            device_index,
            physical_count);
        free(physicals);
        gltf_vk_destroy(vk);
        return 0;
    }
    int found = 0;
    uint32_t p = (uint32_t) device_index;
    {
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[p], &queue_count, NULL);
        VkQueueFamilyProperties * queues =
            (VkQueueFamilyProperties *) malloc((size_t) queue_count * sizeof(VkQueueFamilyProperties));
        if (queues == NULL) {
            free(physicals);
            gltf_vk_destroy(vk);
            return 0;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[p], &queue_count, queues);
        for (uint32_t q = 0; q < queue_count; ++q) {
            if ((queues[q].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
                continue;
            }
            float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info;
            memset(&queue_info, 0, sizeof(queue_info));
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = q;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            VkDeviceCreateInfo device_info;
            memset(&device_info, 0, sizeof(device_info));
            device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue_info;
            if (vkCreateDevice(physicals[p], &device_info, NULL, &vk->device) == VK_SUCCESS) {
                vk->physical_device = physicals[p];
                vk->queue_family = q;
                vkGetDeviceQueue(vk->device, q, 0, &vk->queue);
                found = 1;
                break;
            }
        }
        free(queues);
    }
    if (found) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk->physical_device, &props);
        TRELLIS_INFO("glTF export: using Vulkan device %d: %s", device_index, props.deviceName);
    }
    free(physicals);
    if (!found) {
        gltf_vk_destroy(vk);
        return 0;
    }

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = vk->queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool) != VK_SUCCESS) {
        gltf_vk_destroy(vk);
        return 0;
    }

    VkDescriptorSetLayoutBinding bindings[10];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 10; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 10;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->descriptor_set_layout) != VK_SUCCESS) {
        gltf_vk_destroy(vk);
        return 0;
    }

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(trellis_gltf_vk_push);
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    if (vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipeline_layout) != VK_SUCCESS) {
        gltf_vk_destroy(vk);
        return 0;
    }

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 10u * 64u;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 64;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool) != VK_SUCCESS) {
        gltf_vk_destroy(vk);
        return 0;
    }

    if (!gltf_vk_create_render_pass(vk)) {
        gltf_vk_destroy(vk);
        return 0;
    }
    if (!gltf_vk_create_graphics_pipeline(vk)) {
        gltf_vk_destroy(vk);
        return 0;
    }
    for (uint32_t i = 0; i < TRELLIS_GLTF_COMPUTE_COUNT; ++i) {
        if (!gltf_vk_create_compute_pipeline(vk, &g_gltf_compute_shaders[i], &vk->compute_pipelines[i])) {
            gltf_vk_destroy(vk);
            return 0;
        }
    }
    return 1;
}

static int gltf_vk_allocate_descriptor_set(
    trellis_gltf_vk * vk,
    trellis_gltf_vk_buffer * buffers[10],
    VkDescriptorSet * set_out) {
    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk->descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vk->descriptor_set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(vk->device, &alloc_info, &set) != VK_SUCCESS) return 0;

    VkDescriptorBufferInfo buffer_infos[10];
    VkWriteDescriptorSet writes[10];
    memset(buffer_infos, 0, sizeof(buffer_infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 10; ++i) {
        buffer_infos[i].buffer = buffers[i]->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = (VkDeviceSize) buffers[i]->bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(vk->device, 10, writes, 0, NULL);
    *set_out = set;
    return 1;
}

static int gltf_vk_texture_postprocess(
    trellis_gltf_vk * vk,
    trellis_gltf_vk_buffer * buffers[10],
    trellis_gltf_vk_buffer * chart_map,
    const trellis_gltf_vk_push * initial_push,
    uint32_t supersample_scale,
    trellis_gltf_vk_buffer ** base_out,
    trellis_gltf_vk_buffer ** mr_out) {
    if (vk == NULL || buffers == NULL || chart_map == NULL || initial_push == NULL ||
        base_out == NULL || mr_out == NULL) return 0;
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vk->device, &cmd_alloc, &cmd) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    int ok = 0;
    if (vkBeginCommandBuffer(cmd, &begin_info) == VK_SUCCESS) {
        trellis_gltf_vk_push push = *initial_push;
        const uint32_t image_groups = (push.texture_size + 15u) / 16u;
        const VkDeviceSize target_image_bytes =
            (VkDeviceSize) push.texture_size * push.texture_size * sizeof(uint32_t);
        int recorded = 1;

        /* Resolve the supersampled raster before any padding is generated. */
        push.pad0 = supersample_scale;
        VkDescriptorSet downsample_set = VK_NULL_HANDLE;
        if (!gltf_vk_allocate_descriptor_set(vk, buffers, &downsample_set)) {
            recorded = 0;
        } else {
            vkCmdBindPipeline(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                vk->compute_pipelines[TRELLIS_GLTF_COMPUTE_DOWNSAMPLE]);
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                vk->pipeline_layout,
                0,
                1,
                &downsample_set,
                0,
                NULL);
            vkCmdPushConstants(
                cmd,
                vk->pipeline_layout,
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(push),
                &push);
            vkCmdDispatch(cmd, image_groups, image_groups, 1);

            VkBufferMemoryBarrier barriers[2];
            memset(barriers, 0, sizeof(barriers));
            for (uint32_t i = 0; i < 2; ++i) {
                barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].buffer = buffers[8u + i]->buffer;
                barriers[i].offset = 0;
                barriers[i].size = target_image_bytes;
            }
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                NULL,
                2,
                barriers,
                0,
                NULL);
        }

        trellis_gltf_vk_buffer * cur_base = buffers[8];
        trellis_gltf_vk_buffer * cur_mr = buffers[9];
        trellis_gltf_vk_buffer * next_base = buffers[6];
        trellis_gltf_vk_buffer * next_mr = buffers[7];
        push.pad0 = 0u;
        /* Two extra passes cover xatlas' conservative/bilinear fringe; the chart map caps growth. */
        const uint32_t dilation_passes = TRELLIS_GLTF_UV_OUTPUT_PADDING + 2u;
        for (uint32_t pass = 0; pass < dilation_passes && recorded; ++pass) {
            trellis_gltf_vk_buffer * desc[10];
            for (int i = 0; i < 10; ++i) desc[i] = buffers[i];
            desc[5] = chart_map;
            desc[6] = cur_base;
            desc[7] = cur_mr;
            desc[8] = next_base;
            desc[9] = next_mr;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (!gltf_vk_allocate_descriptor_set(vk, desc, &set)) {
                recorded = 0;
                break;
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->compute_pipelines[TRELLIS_GLTF_COMPUTE_DILATE]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &set, 0, NULL);
            vkCmdPushConstants(
                cmd, vk->pipeline_layout,
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(push), &push);
            vkCmdDispatch(cmd, image_groups, image_groups, 1);

            VkBufferMemoryBarrier barriers[2];
            memset(barriers, 0, sizeof(barriers));
            trellis_gltf_vk_buffer * written[2] = { next_base, next_mr };
            for (uint32_t i = 0; i < 2; ++i) {
                barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].buffer = written[i]->buffer;
                barriers[i].offset = 0;
                barriers[i].size = target_image_bytes;
            }
            vkCmdPipelineBarrier(
                cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 2, barriers, 0, NULL);
            trellis_gltf_vk_buffer * tmp = cur_base;
            cur_base = next_base;
            next_base = tmp;
            tmp = cur_mr;
            cur_mr = next_mr;
            next_mr = tmp;
        }

        /* Unowned atlas space stays empty; global jump-fill can carry color across UV islands. */
        if (recorded) {
            VkBufferMemoryBarrier copy_barriers[2];
            memset(copy_barriers, 0, sizeof(copy_barriers));
            trellis_gltf_vk_buffer * written[2] = { cur_base, cur_mr };
            for (uint32_t i = 0; i < 2; ++i) {
                copy_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                copy_barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                copy_barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                copy_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                copy_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                copy_barriers[i].buffer = written[i]->buffer;
                copy_barriers[i].offset = 0;
                copy_barriers[i].size = target_image_bytes;
            }
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                NULL,
                2,
                copy_barriers,
                0,
                NULL);
        }

        if (recorded && vkEndCommandBuffer(cmd) == VK_SUCCESS) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            ok = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                vkQueueWaitIdle(vk->queue) == VK_SUCCESS;
        }
        if (ok) {
            *base_out = cur_base;
            *mr_out = cur_mr;
        }
    }
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
    return ok;
}

static int gltf_vk_copy_texture_readback(
    trellis_gltf_vk * vk,
    const trellis_gltf_vk_buffer * base_src,
    const trellis_gltf_vk_buffer * mr_src,
    trellis_gltf_vk_buffer * base_dst,
    trellis_gltf_vk_buffer * mr_dst,
    size_t bytes) {
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vk->device, &cmd_alloc, &cmd) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    int ok = vkBeginCommandBuffer(cmd, &begin_info) == VK_SUCCESS;
    if (ok) {
        VkBufferCopy copy;
        memset(&copy, 0, sizeof(copy));
        copy.size = (VkDeviceSize) bytes;
        vkCmdCopyBuffer(cmd, base_src->buffer, base_dst->buffer, 1, &copy);
        vkCmdCopyBuffer(cmd, mr_src->buffer, mr_dst->buffer, 1, &copy);

        VkBufferMemoryBarrier barriers[2];
        memset(barriers, 0, sizeof(barriers));
        trellis_gltf_vk_buffer * written[2] = { base_dst, mr_dst };
        for (uint32_t i = 0; i < 2u; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].buffer = written[i]->buffer;
            barriers[i].offset = 0;
            barriers[i].size = (VkDeviceSize) bytes;
        }
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            NULL,
            2,
            barriers,
            0,
            NULL);
        ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        ok = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
            vkQueueWaitIdle(vk->queue) == VK_SUCCESS;
    }
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
    return ok;
}

static uint32_t gltf_uv_chart_target_faces(void) {
    uint32_t value = 8192u;
    const char * env = getenv("TRELLIS_GLTF_UV_CHART_FACES");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed >= 512ul && parsed <= 65536ul) {
            value = (uint32_t) parsed;
        }
    }
    return value;
}

static int gltf_edge_record_compare(const void * a, const void * b) {
    const trellis_gltf_edge_record * ea = (const trellis_gltf_edge_record *) a;
    const trellis_gltf_edge_record * eb = (const trellis_gltf_edge_record *) b;
    if (ea->key != eb->key) return ea->key < eb->key ? -1 : 1;
    if (ea->face != eb->face) return ea->face < eb->face ? -1 : 1;
    if (ea->local_edge != eb->local_edge) return ea->local_edge < eb->local_edge ? -1 : 1;
    return 0;
}

static int gltf_group_chart_records(
    const trellis_gltf_chart_record * records,
    uint32_t face_count,
    uint32_t chart_count,
    trellis_gltf_chart_record ** grouped_out) {
    *grouped_out = NULL;
    if (records == NULL || face_count == 0u || chart_count == 0u) return 0;

    trellis_gltf_chart_record * grouped =
        (trellis_gltf_chart_record *) malloc((size_t) face_count * sizeof(*grouped));
    uint32_t * offsets =
        (uint32_t *) calloc((size_t) chart_count + 1u, sizeof(*offsets));
    uint32_t * cursors =
        (uint32_t *) malloc((size_t) chart_count * sizeof(*cursors));
    if (grouped == NULL || offsets == NULL || cursors == NULL) {
        free(grouped);
        free(offsets);
        free(cursors);
        return 0;
    }

    for (uint32_t face = 0; face < face_count; ++face) {
        uint32_t key = records[face].key;
        if (key >= chart_count) {
            free(grouped);
            free(offsets);
            free(cursors);
            return 0;
        }
        ++offsets[key + 1u];
    }
    for (uint32_t chart = 1u; chart <= chart_count; ++chart) {
        offsets[chart] += offsets[chart - 1u];
    }
    memcpy(cursors, offsets, (size_t) chart_count * sizeof(*cursors));
    for (uint32_t face = 0; face < face_count; ++face) {
        const trellis_gltf_chart_record record = records[face];
        grouped[cursors[record.key]++] = record;
    }

    free(offsets);
    free(cursors);
    *grouped_out = grouped;
    return 1;
}

static uint64_t gltf_edge_key_u32(uint32_t a, uint32_t b) {
    uint32_t lo = a < b ? a : b;
    uint32_t hi = a < b ? b : a;
    return ((uint64_t) lo << 32) | (uint64_t) hi;
}

static float gltf_vec3_dot(const float * a, const float * b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void gltf_vec3_normalize(float * v) {
    float len2 = gltf_vec3_dot(v, v);
    if (len2 > 1e-20f) {
        float inv = 1.0f / sqrtf(len2);
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    } else {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 1.0f;
    }
}

static int gltf_face_normal_area(
    const trellis_mesh_host * mesh,
    uint32_t face_id,
    float normal[3],
    float * area_out) {
    const int32_t * face = mesh->faces + (size_t) face_id * 3u;
    for (uint32_t i = 0; i < 3u; ++i) {
        if (face[i] < 0 || (uint32_t) face[i] >= (uint32_t) mesh->n_vertices) {
            return 0;
        }
    }
    const float * p0 = mesh->vertices + (size_t) (uint32_t) face[0] * 3u;
    const float * p1 = mesh->vertices + (size_t) (uint32_t) face[1] * 3u;
    const float * p2 = mesh->vertices + (size_t) (uint32_t) face[2] * 3u;
    float ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
    float vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
    float cx = uy * vz - uz * vy;
    float cy = uz * vx - ux * vz;
    float cz = ux * vy - uy * vx;
    float len = sqrtf(cx * cx + cy * cy + cz * cz);
    if (len > 1e-20f) {
        float inv = 1.0f / len;
        normal[0] = cx * inv;
        normal[1] = cy * inv;
        normal[2] = cz * inv;
    } else {
        normal[0] = 0.0f;
        normal[1] = 0.0f;
        normal[2] = 1.0f;
    }
    *area_out = 0.5f * len;
    return 1;
}

static float gltf_uv_chart_cone_cos(void) {
    float degrees = 90.0f;
    const char * env = getenv("TRELLIS_GLTF_UV_CONE_DEGREES");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        float parsed = strtof(env, &end);
        if (end != env && parsed >= 5.0f && parsed <= 180.0f) {
            degrees = parsed;
        }
    }
    return cosf(degrees * 0.017453292519943295769f);
}

static int gltf_compute_chart_records_connected(
    const trellis_mesh_host * mesh,
    uint32_t target_faces,
    trellis_gltf_chart_record ** records_out,
    uint32_t * chart_count_out,
    uint32_t * max_chart_faces_out) {
    *records_out = NULL;
    *chart_count_out = 0;
    *max_chart_faces_out = 0;
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    uint32_t face_count = (uint32_t) mesh->n_faces;
    uint64_t edge_count64 = (uint64_t) face_count * 3u;
    if (edge_count64 > SIZE_MAX / sizeof(trellis_gltf_edge_record)) {
        return 0;
    }
    trellis_gltf_chart_record * records =
        (trellis_gltf_chart_record *) malloc((size_t) face_count * sizeof(trellis_gltf_chart_record));
    trellis_gltf_edge_record * edges =
        (trellis_gltf_edge_record *) malloc((size_t) edge_count64 * sizeof(trellis_gltf_edge_record));
    int32_t * neighbors = (int32_t *) malloc((size_t) face_count * 3u * sizeof(int32_t));
    float * face_normals = (float *) malloc((size_t) face_count * 3u * sizeof(float));
    float * face_areas = (float *) malloc((size_t) face_count * sizeof(float));
    int32_t * chart_ids = (int32_t *) malloc((size_t) face_count * sizeof(int32_t));
    uint32_t * queue = (uint32_t *) malloc((size_t) face_count * sizeof(uint32_t));
    if (records == NULL || edges == NULL || neighbors == NULL ||
        face_normals == NULL || face_areas == NULL || chart_ids == NULL || queue == NULL) {
        free(records);
        free(edges);
        free(neighbors);
        free(face_normals);
        free(face_areas);
        free(chart_ids);
        free(queue);
        return 0;
    }

    int64_t stage_start_us = trellis_now_us();
    for (uint32_t i = 0; i < face_count * 3u; ++i) {
        neighbors[i] = -1;
    }
    for (uint32_t i = 0; i < face_count; ++i) {
        chart_ids[i] = -1;
        float n[3];
        float area = 0.0f;
        if (!gltf_face_normal_area(mesh, i, n, &area)) {
            free(records);
            free(edges);
            free(neighbors);
            free(face_normals);
            free(face_areas);
            free(chart_ids);
            free(queue);
            return 0;
        }
        face_normals[(size_t) i * 3u + 0u] = n[0];
        face_normals[(size_t) i * 3u + 1u] = n[1];
        face_normals[(size_t) i * 3u + 2u] = n[2];
        face_areas[i] = area > 1e-20f ? area : 1.0f;

        const int32_t * f = mesh->faces + (size_t) i * 3u;
        uint32_t v0 = (uint32_t) f[0], v1 = (uint32_t) f[1], v2 = (uint32_t) f[2];
        edges[(size_t) i * 3u + 0u] = (trellis_gltf_edge_record) { gltf_edge_key_u32(v0, v1), i, 0u };
        edges[(size_t) i * 3u + 1u] = (trellis_gltf_edge_record) { gltf_edge_key_u32(v1, v2), i, 1u };
        edges[(size_t) i * 3u + 2u] = (trellis_gltf_edge_record) { gltf_edge_key_u32(v2, v0), i, 2u };
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_adjacency_prepare ms=%.3f faces=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        face_count);

    stage_start_us = trellis_now_us();
    qsort(edges, (size_t) edge_count64, sizeof(*edges), gltf_edge_record_compare);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_edge_sort ms=%.3f edges=%" PRIu64,
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        edge_count64);
    stage_start_us = trellis_now_us();
    size_t e = 0;
    while (e < (size_t) edge_count64) {
        size_t end = e + 1u;
        while (end < (size_t) edge_count64 && edges[end].key == edges[e].key) ++end;
        if (end - e == 2u) {
            const trellis_gltf_edge_record * a = &edges[e];
            const trellis_gltf_edge_record * b = &edges[e + 1u];
            neighbors[(size_t) a->face * 3u + a->local_edge] = (int32_t) b->face;
            neighbors[(size_t) b->face * 3u + b->local_edge] = (int32_t) a->face;
        }
        e = end;
    }
    free(edges);
    edges = NULL;
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_adjacency_emit ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);

    stage_start_us = trellis_now_us();
    const float cone_cos = gltf_uv_chart_cone_cos();
    uint32_t chunk_count = 0;
    uint32_t max_chart_faces = 0;
    for (uint32_t seed = 0; seed < face_count; ++seed) {
        if (chart_ids[seed] >= 0) continue;
        if (chunk_count == INT32_MAX) {
            free(records);
            free(neighbors);
            free(face_normals);
            free(face_areas);
            free(chart_ids);
            free(queue);
            return 0;
        }
        uint32_t chunk = chunk_count++;
        uint32_t head = 0;
        uint32_t tail = 0;
        uint32_t chart_faces = 0;
        float axis_sum[3] = {
            face_normals[(size_t) seed * 3u + 0u] * face_areas[seed],
            face_normals[(size_t) seed * 3u + 1u] * face_areas[seed],
            face_normals[(size_t) seed * 3u + 2u] * face_areas[seed],
        };
        float axis[3] = { axis_sum[0], axis_sum[1], axis_sum[2] };
        gltf_vec3_normalize(axis);
        chart_ids[seed] = (int32_t) chunk;
        queue[tail++] = seed;
        chart_faces = 1u;

        while (head < tail) {
            uint32_t f = queue[head++];
            for (uint32_t le = 0; le < 3u; ++le) {
                if (chart_faces >= target_faces) break;
                int32_t nb_i32 = neighbors[(size_t) f * 3u + le];
                if (nb_i32 < 0) continue;
                uint32_t nb = (uint32_t) nb_i32;
                if (chart_ids[nb] >= 0) continue;
                const float * nn = face_normals + (size_t) nb * 3u;
                float score = gltf_vec3_dot(axis, nn);
                if (score < cone_cos) continue;
                chart_ids[nb] = (int32_t) chunk;
                queue[tail++] = nb;
                ++chart_faces;
                float area = face_areas[nb];
                axis_sum[0] += nn[0] * area;
                axis_sum[1] += nn[1] * area;
                axis_sum[2] += nn[2] * area;
                axis[0] = axis_sum[0];
                axis[1] = axis_sum[1];
                axis[2] = axis_sum[2];
                gltf_vec3_normalize(axis);
            }
        }
        for (uint32_t i = 0; i < tail; ++i) {
            uint32_t f = queue[i];
            records[f].key = chunk;
            records[f].face = f;
        }
        if (chart_faces > max_chart_faces) max_chart_faces = chart_faces;
    }

    *records_out = records;
    *chart_count_out = chunk_count;
    *max_chart_faces_out = max_chart_faces;
    records = NULL;
    free(records);
    free(neighbors);
    free(face_normals);
    free(face_areas);
    free(chart_ids);
    free(queue);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_connected_split ms=%.3f chunks=%u max_chunk_faces=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        chunk_count,
        max_chart_faces);
    return 1;
}

static trellis_status unwrap_mesh_xatlas_charted(
    const trellis_mesh_host * mesh,
    int texture_size,
    trellis_gltf_mesh * out) {
    if (mesh == NULL || out == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX / 3) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    uint32_t target_faces = gltf_uv_chart_target_faces();
    if ((uint64_t) mesh->n_faces < (uint64_t) target_faces * 2u) {
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }

    float * input_normals = (float *) malloc((size_t) mesh->n_vertices * 3u * sizeof(float));
    if (input_normals == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    mesh_compute_normals(mesh, input_normals);

    trellis_gltf_chart_record * records = NULL;
    uint32_t clustered_chart_count = 0;
    uint32_t max_chart_faces = 0;
    if (!gltf_compute_chart_records_connected(mesh, target_faces, &records, &clustered_chart_count, &max_chart_faces)) {
        free(input_normals);
        return TRELLIS_STATUS_ERROR;
    }
    uint32_t chart_count = clustered_chart_count;
    if (chart_count <= 1u) {
        free(records);
        free(input_normals);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    int64_t stage_start_us = trellis_now_us();
    trellis_gltf_chart_record * grouped_records = NULL;
    if (!gltf_group_chart_records(
            records,
            (uint32_t) mesh->n_faces,
            chart_count,
            &grouped_records)) {
        free(records);
        free(input_normals);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    free(records);
    records = grouped_records;
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_group_faces ms=%.3f chunks=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        chart_count);

    trellis_gltf_chart_input * charts =
        (trellis_gltf_chart_input *) calloc((size_t) chart_count, sizeof(trellis_gltf_chart_input));
    int32_t * global_to_local = (int32_t *) malloc((size_t) mesh->n_vertices * sizeof(int32_t));
    if (charts == NULL || global_to_local == NULL) {
        free(charts);
        free(global_to_local);
        free(records);
        free(input_normals);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) global_to_local[i] = -1;

    xatlasSetPrint(NULL, false);
    xatlasAtlas * atlas = xatlasCreate();
    if (atlas == NULL) {
        free(global_to_local);
        free(charts);
        free(records);
        free(input_normals);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    stage_start_us = trellis_now_us();
    uint32_t chart_index = 0;
    int64_t start = 0;
    while (start < mesh->n_faces) {
        int64_t end = start + 1;
        while (end < mesh->n_faces && records[end].key == records[start].key) ++end;
        uint32_t face_count = (uint32_t) (end - start);
        uint32_t max_vertices = face_count * 3u;
        trellis_gltf_chart_input * chart = &charts[chart_index];
        chart->positions = (float *) malloc((size_t) max_vertices * 3u * sizeof(float));
        chart->normals = (float *) malloc((size_t) max_vertices * 3u * sizeof(float));
        chart->indices = (uint32_t *) malloc((size_t) face_count * 3u * sizeof(uint32_t));
        chart->vertex_map = (uint32_t *) malloc((size_t) max_vertices * sizeof(uint32_t));
        uint32_t * touched = (uint32_t *) malloc((size_t) max_vertices * sizeof(uint32_t));
        if (chart->positions == NULL || chart->normals == NULL ||
            chart->indices == NULL || chart->vertex_map == NULL || touched == NULL) {
            free(touched);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            break;
        }

        uint32_t local_vertices = 0;
        uint32_t touched_count = 0;
        for (uint32_t f = 0; f < face_count; ++f) {
            uint32_t face_id = records[start + f].face;
            const int32_t * face = mesh->faces + (size_t) face_id * 3u;
            for (uint32_t k = 0; k < 3u; ++k) {
                int32_t gv_i32 = face[k];
                if (gv_i32 < 0 || (uint32_t) gv_i32 >= (uint32_t) mesh->n_vertices) {
                    status = TRELLIS_STATUS_INVALID_ARGUMENT;
                    break;
                }
                uint32_t gv = (uint32_t) gv_i32;
                int32_t local = global_to_local[gv];
                if (local < 0) {
                    local = (int32_t) local_vertices;
                    global_to_local[gv] = local;
                    touched[touched_count++] = gv;
                    memcpy(chart->positions + (size_t) local_vertices * 3u,
                        mesh->vertices + (size_t) gv * 3u,
                        3u * sizeof(float));
                    memcpy(chart->normals + (size_t) local_vertices * 3u,
                        input_normals + (size_t) gv * 3u,
                        3u * sizeof(float));
                    chart->vertex_map[local_vertices] = gv;
                    ++local_vertices;
                }
                chart->indices[(size_t) f * 3u + k] = (uint32_t) local;
            }
            if (status != TRELLIS_STATUS_OK) break;
        }
        for (uint32_t i = 0; i < touched_count; ++i) global_to_local[touched[i]] = -1;
        free(touched);
        if (status != TRELLIS_STATUS_OK) break;

        chart->vertex_count = local_vertices;
        chart->face_count = face_count;
        xatlasMeshDecl decl;
        xatlasMeshDeclInit(&decl);
        decl.vertexCount = chart->vertex_count;
        decl.vertexPositionData = chart->positions;
        decl.vertexPositionStride = 3u * sizeof(float);
        decl.vertexNormalData = chart->normals;
        decl.vertexNormalStride = 3u * sizeof(float);
        decl.indexCount = chart->face_count * 3u;
        decl.indexData = chart->indices;
        decl.indexFormat = XATLAS_INDEX_FORMAT_UINT32;
        xatlasAddMeshError add_error = xatlasAddMesh(atlas, &decl, chart_count);
        if (add_error != XATLAS_ADD_MESH_ERROR_SUCCESS) {
            TRELLIS_ERROR("glTF export: xatlas AddMesh chart failed: %s", xatlasAddMeshErrorString(add_error));
            status = TRELLIS_STATUS_ERROR;
            break;
        }
        /* xatlas copies MeshDecl data before AddMesh returns. Keep only the
         * local-to-global vertex map needed to assemble the final mesh. */
        free(chart->positions);
        free(chart->normals);
        free(chart->indices);
        chart->positions = NULL;
        chart->normals = NULL;
        chart->indices = NULL;
        ++chart_index;
        start = end;
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_add_mesh ms=%.3f chunks=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        chart_index);

    uint32_t atlas_padding = 0;
    if (status == TRELLIS_STATUS_OK) {
        status = gltf_xatlas_pack_for_output(
            atlas,
            texture_size,
            chart_count,
            &atlas_padding);
        if (status == TRELLIS_STATUS_OK &&
            (atlas->meshCount != chart_count || atlas->meshes == NULL)) {
            status = TRELLIS_STATUS_ERROR;
        }
    }

    stage_start_us = trellis_now_us();
    uint64_t total_vertices = 0;
    uint64_t total_indices = 0;
    if (status == TRELLIS_STATUS_OK) {
        for (uint32_t c = 0; c < chart_count; ++c) {
            total_vertices += atlas->meshes[c].vertexCount;
            total_indices += atlas->meshes[c].indexCount;
        }
        if (total_vertices == 0 || total_vertices > UINT32_MAX || total_indices > UINT32_MAX) {
            status = TRELLIS_STATUS_ERROR;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        out->vertex_count = (uint32_t) total_vertices;
        out->index_count = (uint32_t) total_indices;
        out->positions = (float *) malloc((size_t) out->vertex_count * 3u * sizeof(float));
        out->normals = (float *) malloc((size_t) out->vertex_count * 3u * sizeof(float));
        out->uvs = (float *) malloc((size_t) out->vertex_count * 2u * sizeof(float));
        out->indices = (uint32_t *) malloc((size_t) out->index_count * sizeof(uint32_t));
        if (out->positions == NULL || out->normals == NULL || out->uvs == NULL || out->indices == NULL) {
            gltf_mesh_free(out);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        const float atlas_w = atlas->width > 0 ? (float) atlas->width : (float) texture_size;
        const float atlas_h = atlas->height > 0 ? (float) atlas->height : (float) texture_size;
        uint32_t vertex_base = 0;
        uint32_t index_base = 0;
        for (uint32_t c = 0; c < chart_count; ++c) {
            const xatlasMesh * xm = &atlas->meshes[c];
            const trellis_gltf_chart_input * chart = &charts[c];
            for (uint32_t i = 0; i < xm->vertexCount; ++i) {
                const xatlasVertex * v = &xm->vertexArray[i];
                uint32_t src = v->xref;
                if (src >= chart->vertex_count) {
                    status = TRELLIS_STATUS_ERROR;
                    break;
                }
                uint32_t global = chart->vertex_map[src];
                memcpy(out->positions + (size_t) (vertex_base + i) * 3u,
                    mesh->vertices + (size_t) global * 3u,
                    3u * sizeof(float));
                memcpy(out->normals + (size_t) (vertex_base + i) * 3u,
                    input_normals + (size_t) global * 3u,
                    3u * sizeof(float));
                out->uvs[(size_t) (vertex_base + i) * 2u + 0u] = clamp01_export(v->uv[0] / atlas_w);
                out->uvs[(size_t) (vertex_base + i) * 2u + 1u] = clamp01_export(v->uv[1] / atlas_h);
            }
            if (status != TRELLIS_STATUS_OK) break;
            for (uint32_t i = 0; i < xm->indexCount; ++i) {
                if (xm->indexArray[i] >= xm->vertexCount) {
                    status = TRELLIS_STATUS_ERROR;
                    break;
                }
                out->indices[index_base + i] = vertex_base + xm->indexArray[i];
            }
            if (status != TRELLIS_STATUS_OK) break;
            vertex_base += xm->vertexCount;
            index_base += xm->indexCount;
        }
        if (status == TRELLIS_STATUS_OK) {
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
            status = gltf_copy_xatlas_chart_map(atlas, texture_size, out);
#endif
        }
        if (status == TRELLIS_STATUS_OK) {
            TRELLIS_INFO(
                "glTF export: connected xatlas unwrap faces=%" PRId64 " target=%u clusters=%u max_cluster_faces=%u xatlas_charts=%u vertices=%u atlas=%ux%u padding=%u",
                mesh->n_faces,
                target_faces,
                chart_count,
                max_chart_faces,
                atlas->chartCount,
                out->vertex_count,
                atlas->width,
                atlas->height,
                atlas_padding);
            TRELLIS_INFO(
                "perf_stage name=gltf_uv_output ms=%.3f vertices=%u faces=%u",
                (double) (trellis_now_us() - stage_start_us) / 1000.0,
                out->vertex_count,
                out->index_count / 3u);
        }
    }

    xatlasDestroy(atlas);
    for (uint32_t i = 0; i < chart_count; ++i) gltf_chart_input_free(&charts[i]);
    free(global_to_local);
    free(charts);
    free(records);
    free(input_normals);
    if (status != TRELLIS_STATUS_OK) gltf_mesh_free(out);
    return status;
}

static int gltf_vk_render_bake(
    trellis_gltf_vk * vk,
    trellis_gltf_vk_buffer * buffers[10],
    const trellis_gltf_vk_push * push,
    uint32_t raster_size,
    trellis_gltf_vk_image * base_image,
    trellis_gltf_vk_image * mr_image) {
    if (raster_size == 0u) return 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!gltf_vk_allocate_descriptor_set(vk, buffers, &set)) return 0;

    VkImageView attachments[2] = { base_image->view, mr_image->view };
    VkFramebufferCreateInfo framebuffer_info;
    memset(&framebuffer_info, 0, sizeof(framebuffer_info));
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = vk->render_pass;
    framebuffer_info.attachmentCount = 2;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = raster_size;
    framebuffer_info.height = raster_size;
    framebuffer_info.layers = 1;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(vk->device, &framebuffer_info, NULL, &framebuffer) != VK_SUCCESS) {
        return 0;
    }

    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vk->device, &cmd_alloc, &cmd) != VK_SUCCESS) {
        vkDestroyFramebuffer(vk->device, framebuffer, NULL);
        return 0;
    }

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    int ok = 0;
    if (vkBeginCommandBuffer(cmd, &begin_info) == VK_SUCCESS) {
        VkClearValue clears[2];
        memset(clears, 0, sizeof(clears));
        VkRenderPassBeginInfo render_begin;
        memset(&render_begin, 0, sizeof(render_begin));
        render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_begin.renderPass = vk->render_pass;
        render_begin.framebuffer = framebuffer;
        render_begin.renderArea.offset.x = 0;
        render_begin.renderArea.offset.y = 0;
        render_begin.renderArea.extent.width = raster_size;
        render_begin.renderArea.extent.height = raster_size;
        render_begin.clearValueCount = 2;
        render_begin.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &render_begin, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
        memset(&viewport, 0, sizeof(viewport));
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) raster_size;
        viewport.height = (float) raster_size;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor;
        memset(&scissor, 0, sizeof(scissor));
        scissor.extent.width = raster_size;
        scissor.extent.height = raster_size;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->graphics_pipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        VkBuffer vertex_buffers[2] = { buffers[0]->buffer, buffers[1]->buffer };
        VkDeviceSize vertex_offsets[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, vertex_buffers, vertex_offsets);
        vkCmdBindIndexBuffer(cmd, buffers[2]->buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->pipeline_layout, 0, 1, &set, 0, NULL);
        vkCmdPushConstants(
            cmd, vk->pipeline_layout,
            VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(*push), push);
        vkCmdDrawIndexed(cmd, push->triangle_count * 3u, 1, 0, 0, 0);
        vkCmdEndRenderPass(cmd);

        VkImageSubresourceLayers subresource;
        memset(&subresource, 0, sizeof(subresource));
        subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource.mipLevel = 0;
        subresource.baseArrayLayer = 0;
        subresource.layerCount = 1;
        VkBufferImageCopy copy_region;
        memset(&copy_region, 0, sizeof(copy_region));
        copy_region.bufferOffset = 0;
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource = subresource;
        copy_region.imageOffset.x = 0;
        copy_region.imageOffset.y = 0;
        copy_region.imageOffset.z = 0;
        copy_region.imageExtent.width = raster_size;
        copy_region.imageExtent.height = raster_size;
        copy_region.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(
            cmd,
            base_image->image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            buffers[6]->buffer,
            1,
            &copy_region);
        vkCmdCopyImageToBuffer(
            cmd,
            mr_image->image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            buffers[7]->buffer,
            1,
            &copy_region);

        VkBufferMemoryBarrier barriers[2];
        memset(barriers, 0, sizeof(barriers));
        for (uint32_t i = 0; i < 2; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].buffer = buffers[6u + i]->buffer;
            barriers[i].offset = 0;
            barriers[i].size = (VkDeviceSize) buffers[6u + i]->bytes;
        }
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            NULL,
            2,
            barriers,
            0,
            NULL);

        if (vkEndCommandBuffer(cmd) == VK_SUCCESS) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            ok = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                vkQueueWaitIdle(vk->queue) == VK_SUCCESS;
        }
    }
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
    vkDestroyFramebuffer(vk->device, framebuffer, NULL);
    return ok;
}

static trellis_status build_pbr_hash_entries(
    const trellis_pbr_voxels * voxels,
    int32_t ** entries_out,
    uint32_t * hash_size_out) {
    *entries_out = NULL;
    *hash_size_out = 0;
    size_t table_size = 0;
    if (!next_pow2_export((size_t) voxels->n_coords * 2u, &table_size) || table_size > UINT32_MAX) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    int32_t * entries = (int32_t *) malloc(table_size * 4u * sizeof(int32_t));
    if (entries == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (size_t i = 0; i < table_size; ++i) {
        entries[i * 4u + 3u] = -1;
    }
    for (int64_t i = 0; i < voxels->n_coords; ++i) {
        const int32_t * c = voxels->coords_bxyz + (size_t) i * 4u;
        uint32_t slot = hash_coord_export32(c[1], c[2], c[3]) & ((uint32_t) table_size - 1u);
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
    *entries_out = entries;
    *hash_size_out = (uint32_t) table_size;
    return TRELLIS_STATUS_OK;
}

static trellis_status bake_textures_vulkan(
    const trellis_gltf_mesh * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    int device,
    uint8_t ** base_out,
    uint8_t ** mr_out) {
    if (mesh == NULL || voxels == NULL || base_out == NULL || mr_out == NULL ||
        mesh->positions == NULL || mesh->uvs == NULL || mesh->indices == NULL ||
        mesh->chart_map == NULL || mesh->chart_map_size != (uint32_t) texture_size ||
        voxels->attrs == NULL || voxels->coords_bxyz == NULL || voxels->channels < 3 ||
        mesh->index_count / 3u > UINT32_MAX || voxels->n_coords > UINT32_MAX ||
        voxels->channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *base_out = NULL;
    *mr_out = NULL;
    const size_t pixel_count = (size_t) texture_size * (size_t) texture_size;
    const size_t image_bytes = pixel_count * sizeof(uint32_t);
    if ((uint32_t) texture_size > UINT32_MAX / TRELLIS_GLTF_BAKE_SUPERSAMPLE_SCALE) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint32_t supersample_scale = TRELLIS_GLTF_BAKE_SUPERSAMPLE_SCALE;
    uint32_t raster_size = (uint32_t) texture_size * supersample_scale;
    size_t raster_pixel_count = (size_t) raster_size * (size_t) raster_size;
    if (raster_pixel_count > SIZE_MAX / sizeof(uint32_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    size_t raster_image_bytes = raster_pixel_count * sizeof(uint32_t);
    uint8_t * base = (uint8_t *) malloc(image_bytes);
    uint8_t * mr = (uint8_t *) malloc(image_bytes);
    if (base == NULL || mr == NULL) {
        free(base);
        free(mr);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    int32_t * hash_entries = NULL;
    uint32_t hash_size = 0;
    int64_t stage_start_us = trellis_now_us();
    trellis_status status = build_pbr_hash_entries(voxels, &hash_entries, &hash_size);
    if (status != TRELLIS_STATUS_OK) {
        free(base);
        free(mr);
        return status;
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_pbr_hash ms=%.3f voxels=%" PRId64 " slots=%u",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        voxels->n_coords,
        hash_size);

    uint32_t * projection_words = NULL;
    size_t projection_word_count = 0;
    uint32_t projection_vertex_count = 0;
    uint32_t projection_face_count = 0;
    uint32_t projection_node_count = 0;
    const int projection_enabled = sample_mesh != NULL;
    if (projection_enabled) {
        stage_start_us = trellis_now_us();
        status = build_projection_bvh_buffer(
            sample_mesh,
            &projection_words,
            &projection_word_count,
            &projection_vertex_count,
            &projection_face_count,
            &projection_node_count);
        if (status != TRELLIS_STATUS_OK) {
            free(hash_entries);
            free(base);
            free(mr);
            return status;
        }
        TRELLIS_INFO(
            "glTF export: source projection BVH vertices=%u faces=%u nodes=%u",
            projection_vertex_count,
            projection_face_count,
            projection_node_count);
        TRELLIS_INFO(
            "perf_stage name=gltf_bake_projection_bvh ms=%.3f faces=%u nodes=%u",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            projection_face_count,
            projection_node_count);
    }

    trellis_gltf_vk vk;
    trellis_gltf_vk_buffer buffers[10];
    trellis_gltf_vk_buffer readback[2];
    trellis_gltf_vk_buffer chart_map_buffer;
    trellis_gltf_vk_image base_image;
    trellis_gltf_vk_image mr_image;
    memset(&vk, 0, sizeof(vk));
    memset(buffers, 0, sizeof(buffers));
    memset(readback, 0, sizeof(readback));
    memset(&chart_map_buffer, 0, sizeof(chart_map_buffer));
    memset(&base_image, 0, sizeof(base_image));
    memset(&mr_image, 0, sizeof(mr_image));
    stage_start_us = trellis_now_us();
    if (!gltf_vk_init(&vk, device)) {
        free(projection_words);
        free(hash_entries);
        free(base);
        free(mr);
        return TRELLIS_STATUS_ERROR;
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_vk_init ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);

    VkPhysicalDeviceProperties physical_properties;
    vkGetPhysicalDeviceProperties(vk.physical_device, &physical_properties);
    if (raster_size > physical_properties.limits.maxImageDimension2D ||
        raster_size > physical_properties.limits.maxFramebufferWidth ||
        raster_size > physical_properties.limits.maxFramebufferHeight ||
        raster_size > physical_properties.limits.maxViewportDimensions[0] ||
        raster_size > physical_properties.limits.maxViewportDimensions[1] ||
        raster_image_bytes > physical_properties.limits.maxStorageBufferRange) {
        TRELLIS_INFO(
            "glTF export: 4x texture supersampling at %ux%u exceeds Vulkan limits; using 1 sample",
            raster_size,
            raster_size);
        supersample_scale = 1u;
        raster_size = (uint32_t) texture_size;
        raster_pixel_count = (size_t) raster_size * (size_t) raster_size;
        raster_image_bytes = raster_pixel_count * sizeof(uint32_t);
    }
    if (raster_size > physical_properties.limits.maxImageDimension2D ||
        raster_size > physical_properties.limits.maxFramebufferWidth ||
        raster_size > physical_properties.limits.maxFramebufferHeight ||
        raster_size > physical_properties.limits.maxViewportDimensions[0] ||
        raster_size > physical_properties.limits.maxViewportDimensions[1] ||
        raster_image_bytes > physical_properties.limits.maxStorageBufferRange) {
        TRELLIS_ERROR("glTF export: texture=%ux%u exceeds Vulkan device limits", raster_size, raster_size);
        status = TRELLIS_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    const size_t positions_bytes = (size_t) mesh->vertex_count * 3u * sizeof(float);
    const size_t uvs_bytes = (size_t) mesh->vertex_count * 2u * sizeof(float);
    const size_t indices_bytes = (size_t) mesh->index_count * sizeof(uint32_t);
    const size_t hash_bytes = (size_t) hash_size * 4u * sizeof(int32_t);
    const size_t attrs_bytes = (size_t) voxels->n_coords * (size_t) voxels->channels * sizeof(float);
    const size_t projection_bytes = projection_word_count * sizeof(uint32_t);
    stage_start_us = trellis_now_us();
    /* Reserve the required device-local attachments before host-visible buffers.
     * If the shared device heap is tight, later buffers can fall back to host memory. */
    if (!gltf_vk_image_create(&vk, raster_size, raster_size, &base_image) ||
        !gltf_vk_image_create(&vk, raster_size, raster_size, &mr_image)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    int ok =
        gltf_vk_buffer_create(&vk, positions_bytes, &buffers[0]) &&
        gltf_vk_buffer_create(&vk, uvs_bytes, &buffers[1]) &&
        gltf_vk_buffer_create(&vk, indices_bytes, &buffers[2]) &&
        gltf_vk_buffer_create(&vk, hash_bytes, &buffers[3]) &&
        gltf_vk_buffer_create(&vk, attrs_bytes, &buffers[4]) &&
        gltf_vk_buffer_create(&vk, projection_bytes > 0 ? projection_bytes : 4u, &buffers[5]) &&
        gltf_vk_buffer_create(&vk, raster_image_bytes, &buffers[6]) &&
        gltf_vk_buffer_create(&vk, raster_image_bytes, &buffers[7]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[8]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[9]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &chart_map_buffer) &&
        gltf_vk_readback_buffer_create(&vk, image_bytes, &readback[0]) &&
        gltf_vk_readback_buffer_create(&vk, image_bytes, &readback[1]);
    if (!ok) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    memcpy(buffers[0].mapped, mesh->positions, positions_bytes);
    memcpy(buffers[1].mapped, mesh->uvs, uvs_bytes);
    memcpy(buffers[2].mapped, mesh->indices, indices_bytes);
    memcpy(buffers[3].mapped, hash_entries, hash_bytes);
    memcpy(buffers[4].mapped, voxels->attrs, attrs_bytes);
    if (projection_bytes > 0) {
        memcpy(buffers[5].mapped, projection_words, projection_bytes);
    }
    memcpy(chart_map_buffer.mapped, mesh->chart_map, image_bytes);
    free(hash_entries);
    hash_entries = NULL;
    free(projection_words);
    projection_words = NULL;
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_alloc_upload ms=%.3f hash_mib=%.1f projection_mib=%.1f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        (double) hash_bytes / (1024.0 * 1024.0),
        (double) projection_bytes / (1024.0 * 1024.0));

    trellis_gltf_vk_push push;
    memset(&push, 0, sizeof(push));
    push.texture_size = (uint32_t) texture_size;
    push.triangle_count = mesh->index_count / 3u;
    push.voxel_count = (uint32_t) voxels->n_coords;
    push.channels = (uint32_t) voxels->channels;
    push.hash_size = hash_size;
    push.resolution = (uint32_t) (voxels->resolution > 0 ? voxels->resolution : 512);
    push.project_vertex_count = projection_vertex_count;
    push.project_face_count = projection_face_count;
    push.project_node_count = projection_node_count;
    push.project_flags = projection_enabled ? 1u : 0u;

    TRELLIS_INFO(
        "glTF export: Vulkan raster texture bake tris=%u voxels=%" PRId64 " texture=%d raster=%ux%u samples=%u hash=%u projection_nodes=%u",
        push.triangle_count,
        voxels->n_coords,
        texture_size,
        raster_size,
        raster_size,
        supersample_scale * supersample_scale,
        hash_size,
        push.project_node_count);

    trellis_gltf_vk_buffer * desc[10];
    for (int i = 0; i < 10; ++i) desc[i] = &buffers[i];
    stage_start_us = trellis_now_us();
    if (!gltf_vk_render_bake(&vk, desc, &push, raster_size, &base_image, &mr_image)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_raster ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);

    trellis_gltf_vk_buffer * cur_base = &buffers[8];
    trellis_gltf_vk_buffer * cur_mr = &buffers[9];
    stage_start_us = trellis_now_us();
    if (!gltf_vk_texture_postprocess(
            &vk, desc, &chart_map_buffer, &push, supersample_scale, &cur_base, &cur_mr)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_postprocess ms=%.3f",
        (double) (trellis_now_us() - stage_start_us) / 1000.0);

    stage_start_us = trellis_now_us();
    if (!gltf_vk_copy_texture_readback(
            &vk, cur_base, cur_mr, &readback[0], &readback[1], image_bytes)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    memcpy(base, readback[0].mapped, image_bytes);
    memcpy(mr, readback[1].mapped, image_bytes);
    TRELLIS_INFO(
        "perf_stage name=gltf_bake_readback ms=%.3f bytes=%zu",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        image_bytes * 2u);

    const char * coverage_stats = getenv("TRELLIS_GLTF_BAKE_STATS");
    if (coverage_stats != NULL && coverage_stats[0] != '\0' && strcmp(coverage_stats, "0") != 0) {
        size_t valid_texels = 0;
        size_t owned_texels = 0;
        size_t owned_empty_texels = 0;
        size_t valid_outside_chart = 0;
        const uint32_t * baked_mr = (const uint32_t *) mr;
        for (size_t i = 0; i < pixel_count; ++i) {
            const int valid = (baked_mr[i] >> 24u) != 0u;
            const int owned = mesh->chart_map[i] != 0u;
            valid_texels += valid ? 1u : 0u;
            owned_texels += owned ? 1u : 0u;
            owned_empty_texels += owned && !valid ? 1u : 0u;
            valid_outside_chart += valid && !owned ? 1u : 0u;
        }
        TRELLIS_INFO(
            "glTF export: texture coverage valid=%zu owned=%zu owned_empty=%zu valid_outside=%zu",
            valid_texels,
            owned_texels,
            owned_empty_texels,
            valid_outside_chart);
    }

    *base_out = base;
    *mr_out = mr;
    base = NULL;
    mr = NULL;
    status = TRELLIS_STATUS_OK;

cleanup:
    gltf_vk_image_destroy(&vk, &mr_image);
    gltf_vk_image_destroy(&vk, &base_image);
    gltf_vk_buffer_destroy(&vk, &readback[1]);
    gltf_vk_buffer_destroy(&vk, &readback[0]);
    gltf_vk_buffer_destroy(&vk, &chart_map_buffer);
    for (int i = 0; i < 10; ++i) gltf_vk_buffer_destroy(&vk, &buffers[i]);
    gltf_vk_destroy(&vk);
    free(projection_words);
    free(hash_entries);
    free(base);
    free(mr);
    return status;
}
#endif

static trellis_status unwrap_mesh_xatlas(
    const trellis_mesh_host * mesh,
    int texture_size,
    trellis_gltf_mesh * out) {
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
    if (mesh != NULL && mesh->n_faces >= (int64_t) gltf_uv_chart_target_faces() * 2) {
        trellis_status status = unwrap_mesh_xatlas_charted(mesh, texture_size, out);
        if (status == TRELLIS_STATUS_OK) {
            return status;
        }
        if (status != TRELLIS_STATUS_NOT_IMPLEMENTED) return status;
        TRELLIS_INFO("glTF export: connected chart split unavailable, using direct xatlas");
    }
#endif
    return unwrap_mesh_xatlas_direct(mesh, texture_size, out);
}

static int gltf_coordinate_transform_is_valid(
    trellis_pipeline_gltf_coordinate_transform transform) {
    return transform == TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS ||
        transform == TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D;
}

trellis_status trellis_pipeline_transform_gltf_vectors(
    trellis_pipeline_gltf_coordinate_transform transform,
    float * positions_xyz,
    float * normals_xyz,
    size_t vector_count) {
    if (!gltf_coordinate_transform_is_valid(transform) ||
        (vector_count != 0 && (positions_xyz == NULL || normals_xyz == NULL))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < vector_count; ++i) {
        float * p = positions_xyz + i * 3u;
        float * n = normals_xyz + i * 3u;
        const float px = p[0];
        const float py = p[1];
        const float pz = p[2];
        const float nx = n[0];
        const float ny = n[1];
        const float nz = n[2];
        if (transform == TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D) {
            /* Pixal3D applies B after o-voxel's A.  The composition B*A is a
             * 180-degree Y rotation in raw decoder coordinates. */
            p[0] = -px;
            p[1] = py;
            p[2] = -pz;
            n[0] = -nx;
            n[1] = ny;
            n[2] = -nz;
        } else {
            /* Generic o-voxel/TRELLIS conversion A. */
            p[0] = px;
            p[1] = pz;
            p[2] = -py;
            n[0] = nx;
            n[1] = nz;
            n[2] = -ny;
        }
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status transform_mesh_to_export_axes(
    trellis_gltf_mesh * mesh,
    trellis_pipeline_gltf_coordinate_transform transform) {
    if (mesh == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return trellis_pipeline_transform_gltf_vectors(
        transform,
        mesh->positions,
        mesh->normals,
        mesh->vertex_count);
}

static size_t align4(size_t n) {
    return (n + 3u) & ~((size_t) 3u);
}

static void compute_mesh_buffer_layout(const trellis_gltf_mesh * mesh, size_t offsets[4], size_t sizes[4], size_t * total_size_out) {
    sizes[0] = (size_t) mesh->index_count * sizeof(uint32_t);
    sizes[1] = (size_t) mesh->vertex_count * 3u * sizeof(float);
    sizes[2] = (size_t) mesh->vertex_count * 3u * sizeof(float);
    offsets[0] = 0;
    offsets[1] = align4(offsets[0] + sizes[0]);
    offsets[2] = align4(offsets[1] + sizes[1]);
    if (mesh->uvs != NULL) {
        sizes[3] = (size_t) mesh->vertex_count * 2u * sizeof(float);
        offsets[3] = align4(offsets[2] + sizes[2]);
        *total_size_out = align4(offsets[3] + sizes[3]);
    } else {
        offsets[3] = 0;
        sizes[3] = 0;
        *total_size_out = align4(offsets[2] + sizes[2]);
    }
}

static trellis_status build_mesh_binary_buffer(
    const trellis_gltf_mesh * mesh,
    size_t offsets[4],
    size_t sizes[4],
    size_t * total_size_out,
    uint8_t ** data_out) {
    *data_out = NULL;
    compute_mesh_buffer_layout(mesh, offsets, sizes, total_size_out);
    uint8_t * data = (uint8_t *) calloc(*total_size_out, 1);
    if (data == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(data + offsets[0], mesh->indices, sizes[0]);
    memcpy(data + offsets[1], mesh->positions, sizes[1]);
    memcpy(data + offsets[2], mesh->normals, sizes[2]);
    if (mesh->uvs != NULL) {
        memcpy(data + offsets[3], mesh->uvs, sizes[3]);
    }
    *data_out = data;
    return TRELLIS_STATUS_OK;
}

static trellis_status write_binary_buffer(
    const char * path,
    const trellis_gltf_mesh * mesh,
    size_t offsets[4],
    size_t sizes[4],
    size_t * total_size_out) {
    uint8_t * data = NULL;
    trellis_status status = build_mesh_binary_buffer(mesh, offsets, sizes, total_size_out, &data);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        free(data);
        return TRELLIS_STATUS_IO_ERROR;
    }
    if (fwrite(data, 1, *total_size_out, f) != *total_size_out) {
        free(data);
        fclose(f);
        return TRELLIS_STATUS_IO_ERROR;
    }
    free(data);
    if (fclose(f) != 0) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status encode_png_rgba_memory(
    const uint8_t * rgba,
    int texture_size,
    uint8_t ** png_out,
    size_t * png_size_out) {
    *png_out = NULL;
    *png_size_out = 0;
    int png_len = 0;
    unsigned char * png = stbi_write_png_to_mem(
        rgba,
        texture_size * 4,
        texture_size,
        texture_size,
        4,
        &png_len);
    if (png == NULL || png_len <= 0) {
        if (png != NULL) STBIW_FREE(png);
        return TRELLIS_STATUS_ERROR;
    }
    *png_out = png;
    *png_size_out = (size_t) png_len;
    return TRELLIS_STATUS_OK;
}

static trellis_status build_glb_binary_buffer(
    const trellis_gltf_mesh * mesh,
    const uint8_t * base_png,
    size_t base_png_size,
    const uint8_t * mr_png,
    size_t mr_png_size,
    size_t offsets[6],
    size_t sizes[6],
    size_t * total_size_out,
    uint8_t ** data_out) {
    *data_out = NULL;
    size_t mesh_offsets[4];
    size_t mesh_sizes[4];
    size_t mesh_size = 0;
    compute_mesh_buffer_layout(mesh, mesh_offsets, mesh_sizes, &mesh_size);
    for (int i = 0; i < 4; ++i) {
        offsets[i] = mesh_offsets[i];
        sizes[i] = mesh_sizes[i];
    }
    offsets[4] = align4(mesh_size);
    sizes[4] = base_png_size;
    offsets[5] = align4(offsets[4] + sizes[4]);
    sizes[5] = mr_png_size;
    *total_size_out = align4(offsets[5] + sizes[5]);
    uint8_t * data = (uint8_t *) calloc(*total_size_out, 1);
    if (data == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(data + offsets[0], mesh->indices, sizes[0]);
    memcpy(data + offsets[1], mesh->positions, sizes[1]);
    memcpy(data + offsets[2], mesh->normals, sizes[2]);
    memcpy(data + offsets[3], mesh->uvs, sizes[3]);
    memcpy(data + offsets[4], base_png, sizes[4]);
    memcpy(data + offsets[5], mr_png, sizes[5]);
    *data_out = data;
    return TRELLIS_STATUS_OK;
}

static trellis_status write_gltf_json(
    const char * gltf_path,
    const char * bin_uri,
    const char * base_uri,
    const char * mr_uri,
    const trellis_gltf_mesh * mesh,
    const size_t offsets[6],
    const size_t sizes[6],
    size_t buffer_size,
    int include_textures,
    int write_glb,
    const void * bin_data) {
    if (mesh == NULL || mesh->positions == NULL || mesh->normals == NULL ||
        mesh->indices == NULL || mesh->vertex_count == 0 || mesh->index_count == 0 ||
        (write_glb && (bin_data == NULL || buffer_size == 0)) ||
        (include_textures && mesh->uvs == NULL)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cgltf_data data;
    memset(&data, 0, sizeof(data));
    data.file_type = write_glb ? cgltf_file_type_glb : cgltf_file_type_gltf;
    data.asset.version = (char *) "2.0";
    data.asset.generator = (char *) "trellis2.c";
    if (write_glb) {
        data.bin = bin_data;
        data.bin_size = buffer_size;
    }

    cgltf_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.uri = write_glb ? NULL : (char *) bin_uri;
    buffer.size = buffer_size;
    data.buffers = &buffer;
    data.buffers_count = 1;

    cgltf_buffer_view views[6];
    memset(views, 0, sizeof(views));
    const int has_uvs = mesh->uvs != NULL;
    int view_count = 3 + has_uvs + (write_glb && include_textures ? 2 : 0);
    for (int i = 0; i < view_count; ++i) {
        views[i].buffer = &buffer;
        views[i].offset = offsets[i];
        views[i].size = sizes[i];
    }
    views[0].type = cgltf_buffer_view_type_indices;
    views[1].type = views[2].type = cgltf_buffer_view_type_vertices;
    views[1].stride = 3u * sizeof(float);
    views[2].stride = 3u * sizeof(float);
    if (has_uvs) {
        views[3].type = cgltf_buffer_view_type_vertices;
        views[3].stride = 2u * sizeof(float);
    }
    data.buffer_views = views;
    data.buffer_views_count = (cgltf_size) view_count;

    cgltf_accessor accessors[4];
    memset(accessors, 0, sizeof(accessors));
    accessors[0].buffer_view = &views[0];
    accessors[0].component_type = cgltf_component_type_r_32u;
    accessors[0].type = cgltf_type_scalar;
    accessors[0].count = mesh->index_count;
    accessors[1].buffer_view = &views[1];
    accessors[1].component_type = cgltf_component_type_r_32f;
    accessors[1].type = cgltf_type_vec3;
    accessors[1].count = mesh->vertex_count;
    accessors[1].has_min = accessors[1].has_max = 1;
    for (int c = 0; c < 3; ++c) {
        accessors[1].min[c] = mesh->positions[c];
        accessors[1].max[c] = mesh->positions[c];
    }
    for (uint32_t i = 1; i < mesh->vertex_count; ++i) {
        const float * p = mesh->positions + (size_t) i * 3u;
        for (int c = 0; c < 3; ++c) {
            if (p[c] < accessors[1].min[c]) accessors[1].min[c] = p[c];
            if (p[c] > accessors[1].max[c]) accessors[1].max[c] = p[c];
        }
    }
    accessors[2].buffer_view = &views[2];
    accessors[2].component_type = cgltf_component_type_r_32f;
    accessors[2].type = cgltf_type_vec3;
    accessors[2].count = mesh->vertex_count;
    if (has_uvs) {
        accessors[3].buffer_view = &views[3];
        accessors[3].component_type = cgltf_component_type_r_32f;
        accessors[3].type = cgltf_type_vec2;
        accessors[3].count = mesh->vertex_count;
    }
    data.accessors = accessors;
    data.accessors_count = (cgltf_size) (3 + has_uvs);

    cgltf_image images[2];
    memset(images, 0, sizeof(images));
    if (include_textures) {
        if (write_glb) {
            images[0].buffer_view = &views[4];
            images[0].mime_type = (char *) "image/png";
            images[1].buffer_view = &views[5];
            images[1].mime_type = (char *) "image/png";
        } else {
            images[0].uri = (char *) base_uri;
            images[1].uri = (char *) mr_uri;
        }
        data.images = images;
        data.images_count = 2;
    }

    cgltf_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    sampler.mag_filter = cgltf_filter_type_linear;
    sampler.min_filter = cgltf_filter_type_linear;
    sampler.wrap_s = cgltf_wrap_mode_clamp_to_edge;
    sampler.wrap_t = cgltf_wrap_mode_clamp_to_edge;
    if (include_textures) {
        data.samplers = &sampler;
        data.samplers_count = 1;
    }

    cgltf_texture textures[2];
    memset(textures, 0, sizeof(textures));
    textures[0].image = &images[0];
    textures[0].sampler = &sampler;
    textures[1].image = &images[1];
    textures[1].sampler = &sampler;
    if (include_textures) {
        data.textures = textures;
        data.textures_count = 2;
    }

    cgltf_material material;
    memset(&material, 0, sizeof(material));
    material.name = (char *) (include_textures ? "trellis_pbr" : "trellis_shape");
    material.has_pbr_metallic_roughness = 1;
    if (include_textures) {
        material.pbr_metallic_roughness.base_color_texture.texture = &textures[0];
        material.pbr_metallic_roughness.metallic_roughness_texture.texture = &textures[1];
    }
    material.pbr_metallic_roughness.base_color_factor[0] = include_textures ? 1.0f : 0.72f;
    material.pbr_metallic_roughness.base_color_factor[1] = include_textures ? 1.0f : 0.74f;
    material.pbr_metallic_roughness.base_color_factor[2] = include_textures ? 1.0f : 0.78f;
    material.pbr_metallic_roughness.base_color_factor[3] = 1.0f;
    material.pbr_metallic_roughness.metallic_factor = include_textures ? 1.0f : 0.0f;
    material.pbr_metallic_roughness.roughness_factor = include_textures ? 1.0f : 0.8f;
    material.alpha_mode = cgltf_alpha_mode_opaque;
    material.double_sided = 1;
    data.materials = &material;
    data.materials_count = 1;

    cgltf_attribute attributes[3];
    memset(attributes, 0, sizeof(attributes));
    attributes[0].name = (char *) "POSITION";
    attributes[0].type = cgltf_attribute_type_position;
    attributes[0].data = &accessors[1];
    attributes[1].name = (char *) "NORMAL";
    attributes[1].type = cgltf_attribute_type_normal;
    attributes[1].data = &accessors[2];
    if (has_uvs) {
        attributes[2].name = (char *) "TEXCOORD_0";
        attributes[2].type = cgltf_attribute_type_texcoord;
        attributes[2].index = 0;
        attributes[2].data = &accessors[3];
    }

    cgltf_primitive primitive;
    memset(&primitive, 0, sizeof(primitive));
    primitive.type = cgltf_primitive_type_triangles;
    primitive.indices = &accessors[0];
    primitive.material = &material;
    primitive.attributes = attributes;
    primitive.attributes_count = (cgltf_size) (2 + has_uvs);

    cgltf_mesh cgmesh;
    memset(&cgmesh, 0, sizeof(cgmesh));
    cgmesh.name = (char *) "trellis_mesh";
    cgmesh.primitives = &primitive;
    cgmesh.primitives_count = 1;
    data.meshes = &cgmesh;
    data.meshes_count = 1;

    cgltf_node node;
    memset(&node, 0, sizeof(node));
    node.name = (char *) "trellis";
    node.mesh = &cgmesh;
    data.nodes = &node;
    data.nodes_count = 1;

    cgltf_node * scene_nodes[1] = { &node };
    cgltf_scene scene;
    memset(&scene, 0, sizeof(scene));
    scene.name = (char *) "Scene";
    scene.nodes = scene_nodes;
    scene.nodes_count = 1;
    data.scenes = &scene;
    data.scenes_count = 1;
    data.scene = &scene;

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    options.type = write_glb ? cgltf_file_type_glb : cgltf_file_type_gltf;
    cgltf_result result = cgltf_write_file(&options, gltf_path, &data);
    return result == cgltf_result_success ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static trellis_status shape_mesh_from_host(
    const trellis_mesh_host * mesh,
    trellis_gltf_mesh * out) {
    if (mesh == NULL || out == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        (uint64_t) mesh->n_vertices > UINT32_MAX ||
        (uint64_t) mesh->n_faces > UINT32_MAX / 3u) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    out->vertex_count = (uint32_t) mesh->n_vertices;
    out->index_count = (uint32_t) mesh->n_faces * 3u;
    out->positions = (float *) malloc(
        (size_t) out->vertex_count * 3u * sizeof(float));
    out->normals = (float *) malloc(
        (size_t) out->vertex_count * 3u * sizeof(float));
    out->indices = (uint32_t *) malloc(
        (size_t) out->index_count * sizeof(uint32_t));
    if (out->positions == NULL || out->normals == NULL || out->indices == NULL) {
        gltf_mesh_free(out);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(
        out->positions,
        mesh->vertices,
        (size_t) out->vertex_count * 3u * sizeof(float));
    for (uint32_t i = 0; i < out->index_count; ++i) {
        const int32_t index = mesh->faces[i];
        if (index < 0 || (uint32_t) index >= out->vertex_count) {
            gltf_mesh_free(out);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        out->indices[i] = (uint32_t) index;
    }
    mesh_compute_normals(mesh, out->normals);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_write_shape_gltf_ex(
    const char * path,
    const trellis_mesh_host * mesh,
    trellis_pipeline_gltf_coordinate_transform coordinate_transform) {
    if (path == NULL || mesh == NULL ||
        !gltf_coordinate_transform_is_valid(coordinate_transform)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char dir[4096];
    char stem[512];
    if (!split_output_paths(path, dir, sizeof(dir), stem, sizeof(stem)) ||
        !mkdir_p_export(dir)) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    const int write_glb = path_has_extension_ci(path, ".glb");
    char bin_uri[1024];
    char bin_path[4096];
    memset(bin_uri, 0, sizeof(bin_uri));
    memset(bin_path, 0, sizeof(bin_path));
    if (!write_glb &&
        !make_named_file(
            bin_uri,
            sizeof(bin_uri),
            bin_path,
            sizeof(bin_path),
            dir,
            stem,
            ".bin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_gltf_mesh gltf_mesh;
    memset(&gltf_mesh, 0, sizeof(gltf_mesh));
    uint8_t * binary = NULL;
    size_t offsets[6] = {0, 0, 0, 0, 0, 0};
    size_t sizes[6] = {0, 0, 0, 0, 0, 0};
    size_t binary_size = 0;
    trellis_status status = shape_mesh_from_host(mesh, &gltf_mesh);
    if (status == TRELLIS_STATUS_OK) {
        status = transform_mesh_to_export_axes(
            &gltf_mesh,
            coordinate_transform);
    }
    if (status == TRELLIS_STATUS_OK && write_glb) {
        status = build_mesh_binary_buffer(
            &gltf_mesh, offsets, sizes, &binary_size, &binary);
    } else if (status == TRELLIS_STATUS_OK) {
        status = write_binary_buffer(
            bin_path, &gltf_mesh, offsets, sizes, &binary_size);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = write_gltf_json(
            path,
            write_glb ? NULL : bin_uri,
            NULL,
            NULL,
            &gltf_mesh,
            offsets,
            sizes,
            binary_size,
            0,
            write_glb,
            write_glb ? binary : NULL);
    }
    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO(
            "glTF export: wrote shape-only %s %s vertices=%u faces=%u",
            write_glb ? "GLB" : "glTF",
            path,
            gltf_mesh.vertex_count,
            gltf_mesh.index_count / 3u);
    }
    free(binary);
    gltf_mesh_free(&gltf_mesh);
    return status;
}

trellis_status trellis_pipeline_write_gltf_ex(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    int device,
    trellis_pipeline_gltf_coordinate_transform coordinate_transform) {
    if (path == NULL || mesh == NULL || voxels == NULL || mesh->vertices == NULL ||
        mesh->faces == NULL || mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        voxels->attrs == NULL || voxels->coords_bxyz == NULL || voxels->n_coords <= 0 || device < 0 ||
        !gltf_coordinate_transform_is_valid(coordinate_transform)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (texture_size <= 0) {
        texture_size = 1024;
    }
    if (texture_size < 64 || texture_size > 8192) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t export_start_us = trellis_now_us();
    char dir[4096];
    char stem[512];
    if (!split_output_paths(path, dir, sizeof(dir), stem, sizeof(stem)) || !mkdir_p_export(dir)) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    int write_glb = path_has_extension_ci(path, ".glb");
    char bin_uri[1024], base_uri[1024], mr_uri[1024];
    char bin_path[4096], base_path[4096], mr_path[4096];
    memset(bin_uri, 0, sizeof(bin_uri));
    memset(base_uri, 0, sizeof(base_uri));
    memset(mr_uri, 0, sizeof(mr_uri));
    memset(bin_path, 0, sizeof(bin_path));
    memset(base_path, 0, sizeof(base_path));
    memset(mr_path, 0, sizeof(mr_path));
    if (!write_glb &&
        (!make_named_file(bin_uri, sizeof(bin_uri), bin_path, sizeof(bin_path), dir, stem, ".bin") ||
         !make_named_file(base_uri, sizeof(base_uri), base_path, sizeof(base_path), dir, stem, "_baseColor.png") ||
         !make_named_file(mr_uri, sizeof(mr_uri), mr_path, sizeof(mr_path), dir, stem, "_metallicRoughness.png"))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_gltf_mesh gltf_mesh;
    uint8_t * base = NULL;
    uint8_t * mr = NULL;
    uint8_t * base_png = NULL;
    uint8_t * mr_png = NULL;
    uint8_t * glb_bin = NULL;
    size_t base_png_size = 0;
    size_t mr_png_size = 0;
    memset(&gltf_mesh, 0, sizeof(gltf_mesh));
    int64_t stage_start_us = trellis_now_us();
    trellis_status status = unwrap_mesh_xatlas(mesh, texture_size, &gltf_mesh);
    TRELLIS_INFO(
        "perf_stage name=gltf_uv_unwrap ms=%.3f faces=%" PRId64 " status=%s",
        (double) (trellis_now_us() - stage_start_us) / 1000.0,
        mesh->n_faces,
        trellis_status_string(status));
    if (status == TRELLIS_STATUS_OK) {
#if TRELLIS_HAS_GLTF_VULKAN_BAKE
        stage_start_us = trellis_now_us();
        status = bake_textures_vulkan(&gltf_mesh, sample_mesh, voxels, texture_size, device, &base, &mr);
        TRELLIS_INFO(
            "perf_stage name=gltf_texture_bake ms=%.3f status=%s",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            trellis_status_string(status));
#else
        (void) sample_mesh;
        TRELLIS_ERROR("glTF export: texture bake requires a Vulkan build");
        status = TRELLIS_STATUS_NOT_IMPLEMENTED;
#endif
    }
    if (status == TRELLIS_STATUS_OK) {
        stage_start_us = trellis_now_us();
        if (write_glb) {
            status = encode_png_rgba_memory(base, texture_size, &base_png, &base_png_size);
            if (status == TRELLIS_STATUS_OK) {
                status = encode_png_rgba_memory(mr, texture_size, &mr_png, &mr_png_size);
            }
        } else {
            if (!stbi_write_png(base_path, texture_size, texture_size, 4, base, texture_size * 4) ||
                !stbi_write_png(mr_path, texture_size, texture_size, 4, mr, texture_size * 4)) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
        TRELLIS_INFO(
            "perf_stage name=gltf_png_encode ms=%.3f bytes=%zu status=%s",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            write_glb ? base_png_size + mr_png_size : 0u,
            trellis_status_string(status));
    }
    size_t offsets[6] = {0, 0, 0, 0, 0, 0};
    size_t sizes[6] = {0, 0, 0, 0, 0, 0};
    size_t buffer_size = 0;
    if (status == TRELLIS_STATUS_OK) {
        stage_start_us = trellis_now_us();
        status = transform_mesh_to_export_axes(&gltf_mesh, coordinate_transform);
        if (status == TRELLIS_STATUS_OK && write_glb) {
            status = build_glb_binary_buffer(
                &gltf_mesh,
                base_png,
                base_png_size,
                mr_png,
                mr_png_size,
                offsets,
                sizes,
                &buffer_size,
                &glb_bin);
        } else if (status == TRELLIS_STATUS_OK) {
            status = write_binary_buffer(bin_path, &gltf_mesh, offsets, sizes, &buffer_size);
        }
        TRELLIS_INFO(
            "perf_stage name=gltf_build_binary ms=%.3f bytes=%zu status=%s",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            buffer_size,
            trellis_status_string(status));
    }
    if (status == TRELLIS_STATUS_OK) {
        stage_start_us = trellis_now_us();
        status = write_gltf_json(
            path,
            write_glb ? NULL : bin_uri,
            write_glb ? NULL : base_uri,
            write_glb ? NULL : mr_uri,
            &gltf_mesh,
            offsets,
            sizes,
            buffer_size,
            1,
            write_glb,
            glb_bin);
        TRELLIS_INFO(
            "perf_stage name=gltf_write_output ms=%.3f status=%s",
            (double) (trellis_now_us() - stage_start_us) / 1000.0,
            trellis_status_string(status));
    }
    if (status == TRELLIS_STATUS_OK) {
        if (write_glb) {
            TRELLIS_INFO("glTF export: wrote packed GLB %s", path);
        } else {
            TRELLIS_INFO(
                "glTF export: wrote %s with %s, %s, %s",
                path,
                bin_uri,
                base_uri,
                mr_uri);
        }
    }
    if (base_png != NULL) STBIW_FREE(base_png);
    if (mr_png != NULL) STBIW_FREE(mr_png);
    free(glb_bin);
    free(base);
    free(mr);
    gltf_mesh_free(&gltf_mesh);
    TRELLIS_INFO(
        "perf_stage name=gltf_export_total ms=%.3f status=%s",
        (double) (trellis_now_us() - export_start_us) / 1000.0,
        trellis_status_string(status));
    return status;
}

trellis_status trellis_pipeline_write_gltf(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    int device) {
    return trellis_pipeline_write_gltf_ex(
        path,
        mesh,
        sample_mesh,
        voxels,
        texture_size,
        device,
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS);
}
