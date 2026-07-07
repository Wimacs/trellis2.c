#define CGLTF_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "trellis.h"
#include "../../src/pipeline/trellis_pipeline_internal.h"

#include "cgltf_write.h"
#include "stb_image_write.h"
#include "xatlas_c.h"

#ifdef GGML_USE_VULKAN
#include <vulkan/vulkan.h>

#include "trellis_gltf_bake_vert_spv.h"
#include "trellis_gltf_bake_frag_spv.h"
#include "trellis_gltf_dilate_spv.h"
#include "trellis_gltf_fill_empty_spv.h"
#include "trellis_gltf_chart_keys_spv.h"
#endif

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct trellis_gltf_mesh {
    float * positions;
    float * normals;
    float * uvs;
    uint32_t * indices;
    uint32_t vertex_count;
    uint32_t index_count;
} trellis_gltf_mesh;

typedef struct trellis_gltf_chart_record {
    uint32_t key;
    uint32_t face;
} trellis_gltf_chart_record;

typedef struct trellis_gltf_chart_input {
    float * positions;
    float * normals;
    uint32_t * indices;
    uint32_t * vertex_map;
    uint32_t vertex_count;
    uint32_t face_count;
} trellis_gltf_chart_input;

typedef struct trellis_pbr_hash {
    uint64_t * keys;
    int32_t * values;
    size_t size;
} trellis_pbr_hash;

static int mkdir_p_export(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 1;
    }
    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(tmp)) {
        return 0;
    }
    for (char * p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return 0;
            }
            *p = '/';
        }
    }
    return mkdir(tmp, 0775) == 0 || errno == EEXIST;
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
    const char * slash = strrchr(gltf_path, '/');
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

static float clamp01_export(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static uint64_t coord_key_export(int32_t x, int32_t y, int32_t z) {
    uint64_t key = (((uint64_t) (uint32_t) x) << 42) ^
        (((uint64_t) (uint32_t) y) << 21) ^
        ((uint64_t) (uint32_t) z);
    return key == 0 ? 1 : key;
}

static uint64_t hash_u64_export(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
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

static void pbr_hash_free(trellis_pbr_hash * hash) {
    if (hash == NULL) {
        return;
    }
    free(hash->keys);
    free(hash->values);
    memset(hash, 0, sizeof(*hash));
}

static trellis_status pbr_hash_build(const trellis_pbr_voxels * voxels, trellis_pbr_hash * hash) {
    if (voxels == NULL || hash == NULL || voxels->coords_bxyz == NULL || voxels->n_coords <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(hash, 0, sizeof(*hash));
    size_t table_size = 0;
    if (!next_pow2_export((size_t) voxels->n_coords * 4u, &table_size)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    hash->keys = (uint64_t *) calloc(table_size, sizeof(uint64_t));
    hash->values = (int32_t *) malloc(table_size * sizeof(int32_t));
    if (hash->keys == NULL || hash->values == NULL) {
        pbr_hash_free(hash);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    hash->size = table_size;
    for (size_t i = 0; i < table_size; ++i) {
        hash->values[i] = -1;
    }
    for (int64_t i = 0; i < voxels->n_coords; ++i) {
        const int32_t * c = voxels->coords_bxyz + (size_t) i * 4u;
        const uint64_t key = coord_key_export(c[1], c[2], c[3]);
        size_t slot = (size_t) hash_u64_export(key) & (table_size - 1u);
        while (hash->values[slot] >= 0 && hash->keys[slot] != key) {
            slot = (slot + 1u) & (table_size - 1u);
        }
        hash->keys[slot] = key;
        hash->values[slot] = (int32_t) i;
    }
    return TRELLIS_STATUS_OK;
}

static int pbr_hash_find(const trellis_pbr_hash * hash, int32_t x, int32_t y, int32_t z) {
    const uint64_t key = coord_key_export(x, y, z);
    size_t slot = (size_t) hash_u64_export(key) & (hash->size - 1u);
    while (hash->values[slot] >= 0) {
        if (hash->keys[slot] == key) {
            return hash->values[slot];
        }
        slot = (slot + 1u) & (hash->size - 1u);
    }
    return -1;
}

static void sample_pbr(
    const trellis_pbr_voxels * voxels,
    const trellis_pbr_hash * hash,
    const float p[3],
    float out[6]) {
    const int resolution = voxels->resolution > 0 ? voxels->resolution : 512;
    int32_t gx = (int32_t) floorf((p[0] + 0.5f) * (float) resolution + 0.5f);
    int32_t gy = (int32_t) floorf((p[1] + 0.5f) * (float) resolution + 0.5f);
    int32_t gz = (int32_t) floorf((p[2] + 0.5f) * (float) resolution + 0.5f);
    if (gx < 0) gx = 0; if (gy < 0) gy = 0; if (gz < 0) gz = 0;
    if (gx >= resolution) gx = resolution - 1;
    if (gy >= resolution) gy = resolution - 1;
    if (gz >= resolution) gz = resolution - 1;
    int best = -1;
    for (int radius = 0; best < 0 && radius <= 2; ++radius) {
        for (int dz = -radius; best < 0 && dz <= radius; ++dz) {
            for (int dy = -radius; best < 0 && dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (radius > 0 && abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) {
                        continue;
                    }
                    int32_t x = gx + dx, y = gy + dy, z = gz + dz;
                    if (x < 0 || y < 0 || z < 0 || x >= resolution || y >= resolution || z >= resolution) {
                        continue;
                    }
                    best = pbr_hash_find(hash, x, y, z);
                    if (best >= 0) {
                        break;
                    }
                }
            }
        }
    }
    out[0] = out[1] = out[2] = 0.8f;
    out[3] = 0.0f;
    out[4] = 0.8f;
    out[5] = 1.0f;
    if (best >= 0) {
        const float * src = voxels->attrs + (size_t) best * (size_t) voxels->channels;
        for (int c = 0; c < voxels->channels && c < 6; ++c) {
            out[c] = clamp01_export(src[c]);
        }
        if (voxels->channels < 6) {
            out[5] = 1.0f;
        }
    }
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

    /* CuMesh accelerates chart clustering on GPU, then hands CPU chart meshes to xatlas. */
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
    xatlasPackOptions pack_options;
    xatlasPackOptionsInit(&pack_options);
    pack_options.resolution = (uint32_t) texture_size;
    pack_options.padding = 4;
    pack_options.createImage = false;
    xatlasGenerate(atlas, NULL, &pack_options);
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
        out->uvs[(size_t) i * 2u + 1u] = clamp01_export(1.0f - v->uv[1] / atlas_h);
    }
    memcpy(out->indices, xm->indexArray, (size_t) out->index_count * sizeof(uint32_t));
    TRELLIS_INFO(
        "glTF export: xatlas unwrap vertices=%u faces=%u atlas=%ux%u charts=%u",
        out->vertex_count,
        out->index_count / 3u,
        atlas->width,
        atlas->height,
        atlas->chartCount);
    xatlasDestroy(atlas);
    free(input_normals);
    free(input_indices);
    return TRELLIS_STATUS_OK;
}

static float edge2(float ax, float ay, float bx, float by, float px, float py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static void write_texel(uint8_t * base, uint8_t * mr, int offset, const float pbr[6]) {
    base[offset + 0] = (uint8_t) lrintf(clamp01_export(pbr[0]) * 255.0f);
    base[offset + 1] = (uint8_t) lrintf(clamp01_export(pbr[1]) * 255.0f);
    base[offset + 2] = (uint8_t) lrintf(clamp01_export(pbr[2]) * 255.0f);
    base[offset + 3] = (uint8_t) lrintf(clamp01_export(pbr[5]) * 255.0f);
    mr[offset + 0] = 0;
    mr[offset + 1] = (uint8_t) lrintf(clamp01_export(pbr[4]) * 255.0f);
    mr[offset + 2] = (uint8_t) lrintf(clamp01_export(pbr[3]) * 255.0f);
    mr[offset + 3] = 255;
}

static void dilate_textures(uint8_t * base, uint8_t * mr, int size, int iterations) {
    for (int it = 0; it < iterations; ++it) {
        int changed = 0;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int off = (y * size + x) * 4;
                if (base[off + 3] != 0) {
                    continue;
                }
                int found = -1;
                for (int dy = -1; found < 0 && dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= size || ny >= size) {
                            continue;
                        }
                        int noff = (ny * size + nx) * 4;
                        if (base[noff + 3] != 0) {
                            found = noff;
                            break;
                        }
                    }
                }
                if (found >= 0) {
                    memcpy(base + off, base + found, 4);
                    memcpy(mr + off, mr + found, 4);
                    changed = 1;
                }
            }
        }
        if (!changed) {
            break;
        }
    }
}

static trellis_status bake_textures(
    const trellis_gltf_mesh * mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size,
    uint8_t ** base_out,
    uint8_t ** mr_out) {
    if (mesh == NULL || voxels == NULL || base_out == NULL || mr_out == NULL ||
        mesh->positions == NULL || mesh->uvs == NULL || mesh->indices == NULL ||
        voxels->attrs == NULL || voxels->coords_bxyz == NULL || voxels->channels < 3) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *base_out = NULL;
    *mr_out = NULL;
    const size_t image_bytes = (size_t) texture_size * (size_t) texture_size * 4u;
    uint8_t * base = (uint8_t *) calloc(image_bytes, 1);
    uint8_t * mr = (uint8_t *) calloc(image_bytes, 1);
    if (base == NULL || mr == NULL) {
        free(base);
        free(mr);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_pbr_hash hash;
    trellis_status status = pbr_hash_build(voxels, &hash);
    if (status != TRELLIS_STATUS_OK) {
        free(base);
        free(mr);
        return status;
    }
    for (uint32_t tri = 0; tri < mesh->index_count / 3u; ++tri) {
        uint32_t i0 = mesh->indices[tri * 3u + 0u];
        uint32_t i1 = mesh->indices[tri * 3u + 1u];
        uint32_t i2 = mesh->indices[tri * 3u + 2u];
        const float * uv0 = mesh->uvs + (size_t) i0 * 2u;
        const float * uv1 = mesh->uvs + (size_t) i1 * 2u;
        const float * uv2 = mesh->uvs + (size_t) i2 * 2u;
        float x0 = uv0[0] * (float) (texture_size - 1);
        float y0 = (1.0f - uv0[1]) * (float) (texture_size - 1);
        float x1 = uv1[0] * (float) (texture_size - 1);
        float y1 = (1.0f - uv1[1]) * (float) (texture_size - 1);
        float x2 = uv2[0] * (float) (texture_size - 1);
        float y2 = (1.0f - uv2[1]) * (float) (texture_size - 1);
        float area = edge2(x0, y0, x1, y1, x2, y2);
        if (fabsf(area) < 1e-8f) {
            continue;
        }
        int min_x = (int) floorf(fminf(x0, fminf(x1, x2))) - 1;
        int max_x = (int) ceilf(fmaxf(x0, fmaxf(x1, x2))) + 1;
        int min_y = (int) floorf(fminf(y0, fminf(y1, y2))) - 1;
        int max_y = (int) ceilf(fmaxf(y0, fmaxf(y1, y2))) + 1;
        if (min_x < 0) min_x = 0; if (min_y < 0) min_y = 0;
        if (max_x >= texture_size) max_x = texture_size - 1;
        if (max_y >= texture_size) max_y = texture_size - 1;
        const float * p0 = mesh->positions + (size_t) i0 * 3u;
        const float * p1 = mesh->positions + (size_t) i1 * 3u;
        const float * p2 = mesh->positions + (size_t) i2 * 3u;
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                float px = (float) x + 0.5f;
                float py = (float) y + 0.5f;
                float w0 = edge2(x1, y1, x2, y2, px, py) / area;
                float w1 = edge2(x2, y2, x0, y0, px, py) / area;
                float w2 = edge2(x0, y0, x1, y1, px, py) / area;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) {
                    continue;
                }
                float p[3] = {
                    w0 * p0[0] + w1 * p1[0] + w2 * p2[0],
                    w0 * p0[1] + w1 * p1[1] + w2 * p2[1],
                    w0 * p0[2] + w1 * p1[2] + w2 * p2[2],
                };
                float pbr[6];
                sample_pbr(voxels, &hash, p, pbr);
                write_texel(base, mr, (y * texture_size + x) * 4, pbr);
            }
        }
    }
    dilate_textures(base, mr, texture_size, 32);
    for (int y = 0; y < texture_size; ++y) {
        for (int x = 0; x < texture_size; ++x) {
            int off = (y * texture_size + x) * 4;
            if (base[off + 3] == 0) {
                base[off + 0] = 204; base[off + 1] = 204; base[off + 2] = 204; base[off + 3] = 255;
                mr[off + 0] = 0; mr[off + 1] = 204; mr[off + 2] = 0; mr[off + 3] = 255;
            }
        }
    }
    pbr_hash_free(&hash);
    *base_out = base;
    *mr_out = mr;
    return TRELLIS_STATUS_OK;
}

#ifdef GGML_USE_VULKAN
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
    TRELLIS_GLTF_COMPUTE_DILATE = 0,
    TRELLIS_GLTF_COMPUTE_FILL_EMPTY = 1,
    TRELLIS_GLTF_COMPUTE_CHART_KEYS = 2,
    TRELLIS_GLTF_COMPUTE_COUNT = 3,
} trellis_gltf_vk_compute_pipeline;

typedef struct trellis_gltf_vk_shader {
    const unsigned char * data;
    unsigned int len;
    const char * name;
} trellis_gltf_vk_shader;

static const trellis_gltf_vk_shader g_gltf_bake_vert_shader = {
    trellis_gltf_bake_vert_spv,
    trellis_gltf_bake_vert_spv_len,
    "gltf_bake_vert",
};

static const trellis_gltf_vk_shader g_gltf_bake_frag_shader = {
    trellis_gltf_bake_frag_spv,
    trellis_gltf_bake_frag_spv_len,
    "gltf_bake_frag",
};

static const trellis_gltf_vk_shader g_gltf_compute_shaders[TRELLIS_GLTF_COMPUTE_COUNT] = {
    { trellis_gltf_dilate_spv, trellis_gltf_dilate_spv_len, "gltf_dilate" },
    { trellis_gltf_fill_empty_spv, trellis_gltf_fill_empty_spv_len, "gltf_fill_empty" },
    { trellis_gltf_chart_keys_spv, trellis_gltf_chart_keys_spv_len, "gltf_chart_keys" },
};

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
} trellis_gltf_vk_push;

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

static int gltf_vk_buffer_create(trellis_gltf_vk * vk, size_t bytes, trellis_gltf_vk_buffer * out) {
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
    uint32_t memory_type = gltf_vk_find_memory_type(
        vk->physical_device,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memory_type == UINT32_MAX) {
        gltf_vk_buffer_destroy(vk, out);
        return 0;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(vk->device, &alloc_info, NULL, &out->memory) != VK_SUCCESS ||
        vkBindBufferMemory(vk->device, out->buffer, out->memory, 0) != VK_SUCCESS ||
        vkMapMemory(vk->device, out->memory, 0, req.size, 0, &out->mapped) != VK_SUCCESS) {
        gltf_vk_buffer_destroy(vk, out);
        return 0;
    }
    memset(out->mapped, 0, bytes);
    return 1;
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

static int gltf_vk_init(trellis_gltf_vk * vk) {
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
    int found = 0;
    for (uint32_t p = 0; p < physical_count && !found; ++p) {
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[p], &queue_count, NULL);
        VkQueueFamilyProperties * queues =
            (VkQueueFamilyProperties *) malloc((size_t) queue_count * sizeof(VkQueueFamilyProperties));
        if (queues == NULL) continue;
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

static int gltf_vk_dispatch(
    trellis_gltf_vk * vk,
    trellis_gltf_vk_compute_pipeline pipeline,
    trellis_gltf_vk_buffer * buffers[10],
    const trellis_gltf_vk_push * push,
    uint32_t groups_x,
    uint32_t groups_y) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!gltf_vk_allocate_descriptor_set(vk, buffers, &set)) return 0;

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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->compute_pipelines[pipeline]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &set, 0, NULL);
        vkCmdPushConstants(cmd, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
        vkCmdDispatch(cmd, groups_x, groups_y, 1);
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

static uint32_t gltf_uv_chart_grid(uint32_t face_count, uint32_t target_faces) {
    uint32_t grid = 1u;
    uint64_t desired = ((uint64_t) face_count + (uint64_t) target_faces - 1u) / (uint64_t) target_faces;
    while (grid < 255u && (uint64_t) grid * (uint64_t) grid * (uint64_t) grid < desired) {
        ++grid;
    }
    const char * env = getenv("TRELLIS_GLTF_UV_CHART_GRID");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed >= 1ul && parsed <= 255ul) {
            grid = (uint32_t) parsed;
        }
    }
    return grid;
}

static int gltf_chart_record_compare(const void * a, const void * b) {
    const trellis_gltf_chart_record * ra = (const trellis_gltf_chart_record *) a;
    const trellis_gltf_chart_record * rb = (const trellis_gltf_chart_record *) b;
    if (ra->key != rb->key) return ra->key < rb->key ? -1 : 1;
    if (ra->face != rb->face) return ra->face < rb->face ? -1 : 1;
    return 0;
}

static int gltf_compute_chart_records_vulkan(
    const trellis_mesh_host * mesh,
    uint32_t target_faces,
    trellis_gltf_chart_record ** records_out,
    uint32_t * grid_out) {
    *records_out = NULL;
    *grid_out = 1u;
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    float min_x = mesh->vertices[0], min_y = mesh->vertices[1], min_z = mesh->vertices[2];
    float max_x = min_x, max_y = min_y, max_z = min_z;
    for (int64_t i = 1; i < mesh->n_vertices; ++i) {
        const float * p = mesh->vertices + (size_t) i * 3u;
        if (p[0] < min_x) min_x = p[0];
        if (p[1] < min_y) min_y = p[1];
        if (p[2] < min_z) min_z = p[2];
        if (p[0] > max_x) max_x = p[0];
        if (p[1] > max_y) max_y = p[1];
        if (p[2] > max_z) max_z = p[2];
    }
    float extent_x = max_x - min_x;
    float extent_y = max_y - min_y;
    float extent_z = max_z - min_z;
    float extent = fmaxf(extent_x, fmaxf(extent_y, extent_z));
    if (extent <= 1e-20f) extent = 1.0f;

    uint32_t face_count = (uint32_t) mesh->n_faces;
    uint32_t grid = gltf_uv_chart_grid(face_count, target_faces);
    trellis_gltf_chart_record * records =
        (trellis_gltf_chart_record *) malloc((size_t) face_count * sizeof(trellis_gltf_chart_record));
    if (records == NULL) return 0;

    trellis_gltf_vk vk;
    trellis_gltf_vk_buffer buffers[10];
    memset(&vk, 0, sizeof(vk));
    memset(buffers, 0, sizeof(buffers));
    int ok = 0;
    if (!gltf_vk_init(&vk)) goto cleanup;

    size_t positions_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    size_t records_bytes = (size_t) face_count * sizeof(trellis_gltf_chart_record);
    if (!gltf_vk_buffer_create(&vk, positions_bytes, &buffers[0]) ||
        !gltf_vk_buffer_create(&vk, faces_bytes, &buffers[1]) ||
        !gltf_vk_buffer_create(&vk, records_bytes, &buffers[2])) {
        goto cleanup;
    }
    for (uint32_t i = 3; i < 10; ++i) {
        if (!gltf_vk_buffer_create(&vk, 4u, &buffers[i])) goto cleanup;
    }

    memcpy(buffers[0].mapped, mesh->vertices, positions_bytes);
    memcpy(buffers[1].mapped, mesh->faces, faces_bytes);
    memset(buffers[2].mapped, 0, records_bytes);

    trellis_gltf_vk_push push;
    memset(&push, 0, sizeof(push));
    push.triangle_count = face_count;
    push.bounds_min_x = min_x;
    push.bounds_min_y = min_y;
    push.bounds_min_z = min_z;
    push.bounds_inv_extent = 1.0f / extent;
    push.chart_grid = grid;
    push.chart_target_faces = target_faces;
    push.chart_normal_bins = 6u;

    trellis_gltf_vk_buffer * desc[10];
    for (int i = 0; i < 10; ++i) desc[i] = &buffers[i];
    uint32_t groups = (face_count + 127u) / 128u;
    if (!gltf_vk_dispatch(&vk, TRELLIS_GLTF_COMPUTE_CHART_KEYS, desc, &push, groups, 1u)) {
        goto cleanup;
    }
    memcpy(records, buffers[2].mapped, records_bytes);
    *records_out = records;
    *grid_out = grid;
    records = NULL;
    ok = 1;

cleanup:
    for (int i = 0; i < 10; ++i) gltf_vk_buffer_destroy(&vk, &buffers[i]);
    gltf_vk_destroy(&vk);
    free(records);
    return ok;
}

static trellis_status unwrap_mesh_xatlas_charted_vulkan(
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
    uint32_t grid = 1u;
    if (!gltf_compute_chart_records_vulkan(mesh, target_faces, &records, &grid)) {
        free(input_normals);
        return TRELLIS_STATUS_ERROR;
    }
    qsort(records, (size_t) mesh->n_faces, sizeof(*records), gltf_chart_record_compare);

    uint32_t chart_count = 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if (i == 0 || records[i].key != records[i - 1].key) ++chart_count;
    }
    if (chart_count <= 1u) {
        free(records);
        free(input_normals);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }

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
        ++chart_index;
        start = end;
    }

    if (status == TRELLIS_STATUS_OK) {
        xatlasPackOptions pack_options;
        xatlasPackOptionsInit(&pack_options);
        pack_options.resolution = (uint32_t) texture_size;
        pack_options.padding = 4;
        pack_options.createImage = false;
        xatlasGenerate(atlas, NULL, &pack_options);
        if (atlas->meshCount != chart_count || atlas->meshes == NULL) {
            status = TRELLIS_STATUS_ERROR;
        }
    }

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
                out->uvs[(size_t) (vertex_base + i) * 2u + 1u] = clamp01_export(1.0f - v->uv[1] / atlas_h);
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
            TRELLIS_INFO(
                "glTF export: Vulkan charted xatlas unwrap faces=%" PRId64 " grid=%u charts=%u xatlas_charts=%u vertices=%u atlas=%ux%u",
                mesh->n_faces,
                grid,
                chart_count,
                atlas->chartCount,
                out->vertex_count,
                atlas->width,
                atlas->height);
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
    trellis_gltf_vk_image * base_image,
    trellis_gltf_vk_image * mr_image) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!gltf_vk_allocate_descriptor_set(vk, buffers, &set)) return 0;

    VkImageView attachments[2] = { base_image->view, mr_image->view };
    VkFramebufferCreateInfo framebuffer_info;
    memset(&framebuffer_info, 0, sizeof(framebuffer_info));
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = vk->render_pass;
    framebuffer_info.attachmentCount = 2;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = push->texture_size;
    framebuffer_info.height = push->texture_size;
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
        render_begin.renderArea.extent.width = push->texture_size;
        render_begin.renderArea.extent.height = push->texture_size;
        render_begin.clearValueCount = 2;
        render_begin.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &render_begin, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
        memset(&viewport, 0, sizeof(viewport));
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) push->texture_size;
        viewport.height = (float) push->texture_size;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor;
        memset(&scissor, 0, sizeof(scissor));
        scissor.extent.width = push->texture_size;
        scissor.extent.height = push->texture_size;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->graphics_pipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        VkBuffer vertex_buffers[2] = { buffers[0]->buffer, buffers[1]->buffer };
        VkDeviceSize vertex_offsets[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, vertex_buffers, vertex_offsets);
        vkCmdBindIndexBuffer(cmd, buffers[2]->buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->pipeline_layout, 0, 1, &set, 0, NULL);
        vkCmdPushConstants(cmd, vk->pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(*push), push);
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
        copy_region.imageExtent.width = push->texture_size;
        copy_region.imageExtent.height = push->texture_size;
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
    if (!next_pow2_export((size_t) voxels->n_coords * 4u, &table_size) || table_size > UINT32_MAX) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    int32_t * entries = (int32_t *) malloc(table_size * 4u * sizeof(int32_t));
    if (entries == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (size_t i = 0; i < table_size; ++i) {
        entries[i * 4u + 0u] = 0;
        entries[i * 4u + 1u] = 0;
        entries[i * 4u + 2u] = 0;
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
    const trellis_pbr_voxels * voxels,
    int texture_size,
    uint8_t ** base_out,
    uint8_t ** mr_out) {
    if (mesh == NULL || voxels == NULL || base_out == NULL || mr_out == NULL ||
        mesh->positions == NULL || mesh->uvs == NULL || mesh->indices == NULL ||
        voxels->attrs == NULL || voxels->coords_bxyz == NULL || voxels->channels < 3 ||
        mesh->index_count / 3u > UINT32_MAX || voxels->n_coords > UINT32_MAX ||
        voxels->channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *base_out = NULL;
    *mr_out = NULL;
    const size_t pixel_count = (size_t) texture_size * (size_t) texture_size;
    const size_t image_bytes = pixel_count * sizeof(uint32_t);
    uint8_t * base = (uint8_t *) malloc(image_bytes);
    uint8_t * mr = (uint8_t *) malloc(image_bytes);
    if (base == NULL || mr == NULL) {
        free(base);
        free(mr);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    int32_t * hash_entries = NULL;
    uint32_t hash_size = 0;
    trellis_status status = build_pbr_hash_entries(voxels, &hash_entries, &hash_size);
    if (status != TRELLIS_STATUS_OK) {
        free(base);
        free(mr);
        return status;
    }

    trellis_gltf_vk vk;
    trellis_gltf_vk_buffer buffers[10];
    trellis_gltf_vk_image base_image;
    trellis_gltf_vk_image mr_image;
    memset(&vk, 0, sizeof(vk));
    memset(buffers, 0, sizeof(buffers));
    memset(&base_image, 0, sizeof(base_image));
    memset(&mr_image, 0, sizeof(mr_image));
    if (!gltf_vk_init(&vk)) {
        free(hash_entries);
        free(base);
        free(mr);
        return TRELLIS_STATUS_ERROR;
    }

    const size_t positions_bytes = (size_t) mesh->vertex_count * 3u * sizeof(float);
    const size_t uvs_bytes = (size_t) mesh->vertex_count * 2u * sizeof(float);
    const size_t indices_bytes = (size_t) mesh->index_count * sizeof(uint32_t);
    const size_t hash_bytes = (size_t) hash_size * 4u * sizeof(int32_t);
    const size_t attrs_bytes = (size_t) voxels->n_coords * (size_t) voxels->channels * sizeof(float);
    int ok =
        gltf_vk_buffer_create(&vk, positions_bytes, &buffers[0]) &&
        gltf_vk_buffer_create(&vk, uvs_bytes, &buffers[1]) &&
        gltf_vk_buffer_create(&vk, indices_bytes, &buffers[2]) &&
        gltf_vk_buffer_create(&vk, hash_bytes, &buffers[3]) &&
        gltf_vk_buffer_create(&vk, attrs_bytes, &buffers[4]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[5]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[6]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[7]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[8]) &&
        gltf_vk_buffer_create(&vk, image_bytes, &buffers[9]);
    if (!ok) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (!gltf_vk_image_create(&vk, (uint32_t) texture_size, (uint32_t) texture_size, &base_image) ||
        !gltf_vk_image_create(&vk, (uint32_t) texture_size, (uint32_t) texture_size, &mr_image)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    memcpy(buffers[0].mapped, mesh->positions, positions_bytes);
    memcpy(buffers[1].mapped, mesh->uvs, uvs_bytes);
    memcpy(buffers[2].mapped, mesh->indices, indices_bytes);
    memcpy(buffers[3].mapped, hash_entries, hash_bytes);
    memcpy(buffers[4].mapped, voxels->attrs, attrs_bytes);
    memset(buffers[5].mapped, 0, image_bytes);
    memset(buffers[6].mapped, 0, image_bytes);
    memset(buffers[7].mapped, 0, image_bytes);
    memset(buffers[8].mapped, 0, image_bytes);
    memset(buffers[9].mapped, 0, image_bytes);

    trellis_gltf_vk_push push;
    memset(&push, 0, sizeof(push));
    push.texture_size = (uint32_t) texture_size;
    push.triangle_count = mesh->index_count / 3u;
    push.voxel_count = (uint32_t) voxels->n_coords;
    push.channels = (uint32_t) voxels->channels;
    push.hash_size = hash_size;
    push.resolution = (uint32_t) (voxels->resolution > 0 ? voxels->resolution : 512);

    TRELLIS_INFO(
        "glTF export: Vulkan raster texture bake tris=%u voxels=%" PRId64 " texture=%d hash=%u",
        push.triangle_count,
        voxels->n_coords,
        texture_size,
        hash_size);

    trellis_gltf_vk_buffer * desc[10];
    for (int i = 0; i < 10; ++i) desc[i] = &buffers[i];
    if (!gltf_vk_render_bake(&vk, desc, &push, &base_image, &mr_image)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    trellis_gltf_vk_buffer * cur_base = &buffers[6];
    trellis_gltf_vk_buffer * cur_mr = &buffers[7];
    trellis_gltf_vk_buffer * next_base = &buffers[8];
    trellis_gltf_vk_buffer * next_mr = &buffers[9];
    const uint32_t image_groups = ((uint32_t) texture_size + 15u) / 16u;
    for (int it = 0; it < 32; ++it) {
        desc[6] = cur_base;
        desc[7] = cur_mr;
        desc[8] = next_base;
        desc[9] = next_mr;
        if (!gltf_vk_dispatch(&vk, TRELLIS_GLTF_COMPUTE_DILATE, desc, &push, image_groups, image_groups)) {
            status = TRELLIS_STATUS_ERROR;
            goto cleanup;
        }
        trellis_gltf_vk_buffer * tmp = cur_base;
        cur_base = next_base;
        next_base = tmp;
        tmp = cur_mr;
        cur_mr = next_mr;
        next_mr = tmp;
    }

    desc[6] = cur_base;
    desc[7] = cur_mr;
    desc[8] = next_base;
    desc[9] = next_mr;
    if (!gltf_vk_dispatch(&vk, TRELLIS_GLTF_COMPUTE_FILL_EMPTY, desc, &push, image_groups, image_groups)) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    memcpy(base, cur_base->mapped, image_bytes);
    memcpy(mr, cur_mr->mapped, image_bytes);
    *base_out = base;
    *mr_out = mr;
    base = NULL;
    mr = NULL;
    status = TRELLIS_STATUS_OK;

cleanup:
    gltf_vk_image_destroy(&vk, &mr_image);
    gltf_vk_image_destroy(&vk, &base_image);
    for (int i = 0; i < 10; ++i) gltf_vk_buffer_destroy(&vk, &buffers[i]);
    gltf_vk_destroy(&vk);
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
#ifdef GGML_USE_VULKAN
    if (mesh != NULL && mesh->n_faces >= (int64_t) gltf_uv_chart_target_faces() * 2) {
        trellis_status status = unwrap_mesh_xatlas_charted_vulkan(mesh, texture_size, out);
        if (status == TRELLIS_STATUS_OK) {
            return status;
        }
        if (status != TRELLIS_STATUS_NOT_IMPLEMENTED) {
            TRELLIS_INFO("glTF export: Vulkan charted UV unwrap failed, falling back to direct xatlas");
        }
    }
#endif
    return unwrap_mesh_xatlas_direct(mesh, texture_size, out);
}

static void transform_mesh_to_viewer_axes(trellis_gltf_mesh * mesh) {
    for (uint32_t i = 0; i < mesh->vertex_count; ++i) {
        float * p = mesh->positions + (size_t) i * 3u;
        float y = p[1];
        p[1] = p[2];
        p[2] = -y;
        float * n = mesh->normals + (size_t) i * 3u;
        y = n[1];
        n[1] = n[2];
        n[2] = -y;
    }
}

static size_t align4(size_t n) {
    return (n + 3u) & ~((size_t) 3u);
}

static trellis_status write_binary_buffer(
    const char * path,
    const trellis_gltf_mesh * mesh,
    size_t offsets[4],
    size_t sizes[4],
    size_t * total_size_out) {
    sizes[0] = (size_t) mesh->index_count * sizeof(uint32_t);
    sizes[1] = (size_t) mesh->vertex_count * 3u * sizeof(float);
    sizes[2] = (size_t) mesh->vertex_count * 3u * sizeof(float);
    sizes[3] = (size_t) mesh->vertex_count * 2u * sizeof(float);
    offsets[0] = 0;
    offsets[1] = align4(offsets[0] + sizes[0]);
    offsets[2] = align4(offsets[1] + sizes[1]);
    offsets[3] = align4(offsets[2] + sizes[2]);
    size_t total = align4(offsets[3] + sizes[3]);
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    const uint8_t zero[4] = {0, 0, 0, 0};
#define WRITE_CHUNK(data, size, next_offset) do { \
    if (fwrite((data), 1, (size), f) != (size)) { fclose(f); return TRELLIS_STATUS_IO_ERROR; } \
    size_t pos = (size_t) ftell(f); \
    while (pos < (next_offset)) { \
        size_t pad = (next_offset) - pos; \
        if (pad > 4u) pad = 4u; \
        if (fwrite(zero, 1, pad, f) != pad) { fclose(f); return TRELLIS_STATUS_IO_ERROR; } \
        pos += pad; \
    } \
} while (0)
    WRITE_CHUNK(mesh->indices, sizes[0], offsets[1]);
    WRITE_CHUNK(mesh->positions, sizes[1], offsets[2]);
    WRITE_CHUNK(mesh->normals, sizes[2], offsets[3]);
    WRITE_CHUNK(mesh->uvs, sizes[3], total);
#undef WRITE_CHUNK
    if (fclose(f) != 0) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    *total_size_out = total;
    return TRELLIS_STATUS_OK;
}

static trellis_status write_gltf_json(
    const char * gltf_path,
    const char * bin_uri,
    const char * base_uri,
    const char * mr_uri,
    const trellis_gltf_mesh * mesh,
    const size_t offsets[4],
    const size_t sizes[4],
    size_t buffer_size) {
    cgltf_data data;
    memset(&data, 0, sizeof(data));
    data.file_type = cgltf_file_type_gltf;
    data.asset.version = (char *) "2.0";
    data.asset.generator = (char *) "trellis2.c";

    cgltf_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.uri = (char *) bin_uri;
    buffer.size = buffer_size;
    data.buffers = &buffer;
    data.buffers_count = 1;

    cgltf_buffer_view views[4];
    memset(views, 0, sizeof(views));
    for (int i = 0; i < 4; ++i) {
        views[i].buffer = &buffer;
        views[i].offset = offsets[i];
        views[i].size = sizes[i];
    }
    views[0].type = cgltf_buffer_view_type_indices;
    views[1].type = views[2].type = views[3].type = cgltf_buffer_view_type_vertices;
    views[1].stride = 3u * sizeof(float);
    views[2].stride = 3u * sizeof(float);
    views[3].stride = 2u * sizeof(float);
    data.buffer_views = views;
    data.buffer_views_count = 4;

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
    accessors[3].buffer_view = &views[3];
    accessors[3].component_type = cgltf_component_type_r_32f;
    accessors[3].type = cgltf_type_vec2;
    accessors[3].count = mesh->vertex_count;
    data.accessors = accessors;
    data.accessors_count = 4;

    cgltf_image images[2];
    memset(images, 0, sizeof(images));
    images[0].uri = (char *) base_uri;
    images[1].uri = (char *) mr_uri;
    data.images = images;
    data.images_count = 2;

    cgltf_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    sampler.mag_filter = cgltf_filter_type_linear;
    sampler.min_filter = cgltf_filter_type_linear;
    sampler.wrap_s = cgltf_wrap_mode_clamp_to_edge;
    sampler.wrap_t = cgltf_wrap_mode_clamp_to_edge;
    data.samplers = &sampler;
    data.samplers_count = 1;

    cgltf_texture textures[2];
    memset(textures, 0, sizeof(textures));
    textures[0].image = &images[0];
    textures[0].sampler = &sampler;
    textures[1].image = &images[1];
    textures[1].sampler = &sampler;
    data.textures = textures;
    data.textures_count = 2;

    cgltf_material material;
    memset(&material, 0, sizeof(material));
    material.name = (char *) "trellis_pbr";
    material.has_pbr_metallic_roughness = 1;
    material.pbr_metallic_roughness.base_color_texture.texture = &textures[0];
    material.pbr_metallic_roughness.metallic_roughness_texture.texture = &textures[1];
    material.pbr_metallic_roughness.base_color_factor[0] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[1] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[2] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[3] = 1.0f;
    material.pbr_metallic_roughness.metallic_factor = 1.0f;
    material.pbr_metallic_roughness.roughness_factor = 1.0f;
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
    attributes[2].name = (char *) "TEXCOORD_0";
    attributes[2].type = cgltf_attribute_type_texcoord;
    attributes[2].index = 0;
    attributes[2].data = &accessors[3];

    cgltf_primitive primitive;
    memset(&primitive, 0, sizeof(primitive));
    primitive.type = cgltf_primitive_type_triangles;
    primitive.indices = &accessors[0];
    primitive.material = &material;
    primitive.attributes = attributes;
    primitive.attributes_count = 3;

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
    options.type = cgltf_file_type_gltf;
    cgltf_result result = cgltf_write_file(&options, gltf_path, &data);
    return result == cgltf_result_success ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

trellis_status trellis_pipeline_write_gltf(
    const char * path,
    const trellis_mesh_host * mesh,
    const trellis_pbr_voxels * voxels,
    int texture_size) {
    if (path == NULL || mesh == NULL || voxels == NULL || mesh->vertices == NULL ||
        mesh->faces == NULL || mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        voxels->attrs == NULL || voxels->coords_bxyz == NULL || voxels->n_coords <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (texture_size <= 0) {
        texture_size = 1024;
    }
    if (texture_size < 64 || texture_size > 8192) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char dir[4096];
    char stem[512];
    if (!split_output_paths(path, dir, sizeof(dir), stem, sizeof(stem)) || !mkdir_p_export(dir)) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    char bin_uri[1024], base_uri[1024], mr_uri[1024];
    char bin_path[4096], base_path[4096], mr_path[4096];
    if (!make_named_file(bin_uri, sizeof(bin_uri), bin_path, sizeof(bin_path), dir, stem, ".bin") ||
        !make_named_file(base_uri, sizeof(base_uri), base_path, sizeof(base_path), dir, stem, "_baseColor.png") ||
        !make_named_file(mr_uri, sizeof(mr_uri), mr_path, sizeof(mr_path), dir, stem, "_metallicRoughness.png")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_gltf_mesh gltf_mesh;
    uint8_t * base = NULL;
    uint8_t * mr = NULL;
    memset(&gltf_mesh, 0, sizeof(gltf_mesh));
    trellis_status status = unwrap_mesh_xatlas(mesh, texture_size, &gltf_mesh);
    if (status == TRELLIS_STATUS_OK) {
#ifdef GGML_USE_VULKAN
        status = bake_textures_vulkan(&gltf_mesh, voxels, texture_size, &base, &mr);
#else
        status = bake_textures(&gltf_mesh, voxels, texture_size, &base, &mr);
#endif
    }
    if (status == TRELLIS_STATUS_OK) {
        if (!stbi_write_png(base_path, texture_size, texture_size, 4, base, texture_size * 4) ||
            !stbi_write_png(mr_path, texture_size, texture_size, 4, mr, texture_size * 4)) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }
    size_t offsets[4] = {0, 0, 0, 0};
    size_t sizes[4] = {0, 0, 0, 0};
    size_t buffer_size = 0;
    if (status == TRELLIS_STATUS_OK) {
        transform_mesh_to_viewer_axes(&gltf_mesh);
        status = write_binary_buffer(bin_path, &gltf_mesh, offsets, sizes, &buffer_size);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = write_gltf_json(path, bin_uri, base_uri, mr_uri, &gltf_mesh, offsets, sizes, buffer_size);
    }
    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO(
            "glTF export: wrote %s with %s, %s, %s",
            path,
            bin_uri,
            base_uri,
            mr_uri);
    }
    free(base);
    free(mr);
    gltf_mesh_free(&gltf_mesh);
    return status;
}
