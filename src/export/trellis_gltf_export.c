#define CGLTF_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "trellis.h"
#include "../pipeline/trellis_pipeline_internal.h"

#include "cgltf_write.h"
#include "stb_image_write.h"
#include "xatlas_c.h"

#ifdef GGML_USE_VULKAN
#include <vulkan/vulkan.h>

#include "trellis_gltf_bake_spv.h"
#include "trellis_gltf_dilate_spv.h"
#include "trellis_gltf_fill_empty_spv.h"
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

static trellis_status unwrap_mesh_xatlas(
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
        status = bake_textures(&gltf_mesh, voxels, texture_size, &base, &mr);
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
