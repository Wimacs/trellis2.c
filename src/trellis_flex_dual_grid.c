#include "trellis.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct fdg_hash {
    int32_t * keys;
    int32_t * vals;
    uint8_t * used;
    size_t capacity;
} fdg_hash;

static size_t fdg_next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

static uint64_t fdg_hash3(int32_t x, int32_t y, int32_t z) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint32_t) x; h *= 1099511628211ull;
    h ^= (uint32_t) y; h *= 1099511628211ull;
    h ^= (uint32_t) z; h *= 1099511628211ull;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

static int fdg_hash_init(fdg_hash * h, int64_t n) {
    if (h == NULL || n < 0) {
        return 0;
    }
    memset(h, 0, sizeof(*h));
    size_t need = (size_t) n * 4u + 16u;
    h->capacity = fdg_next_pow2(need);
    h->keys = (int32_t *) malloc(h->capacity * 3u * sizeof(int32_t));
    h->vals = (int32_t *) malloc(h->capacity * sizeof(int32_t));
    h->used = (uint8_t *) calloc(h->capacity, 1u);
    if (h->keys == NULL || h->vals == NULL || h->used == NULL) {
        free(h->keys);
        free(h->vals);
        free(h->used);
        memset(h, 0, sizeof(*h));
        return 0;
    }
    return 1;
}

static void fdg_hash_free(fdg_hash * h) {
    if (h == NULL) {
        return;
    }
    free(h->keys);
    free(h->vals);
    free(h->used);
    memset(h, 0, sizeof(*h));
}

static int fdg_hash_insert(fdg_hash * h, int32_t x, int32_t y, int32_t z, int32_t value) {
    if (h == NULL || h->capacity == 0) {
        return 0;
    }
    size_t mask = h->capacity - 1u;
    size_t slot = (size_t) fdg_hash3(x, y, z) & mask;
    for (size_t probe = 0; probe < h->capacity; ++probe) {
        size_t k = slot * 3u;
        if (!h->used[slot]) {
            h->used[slot] = 1u;
            h->keys[k + 0u] = x;
            h->keys[k + 1u] = y;
            h->keys[k + 2u] = z;
            h->vals[slot] = value;
            return 1;
        }
        if (h->keys[k + 0u] == x && h->keys[k + 1u] == y && h->keys[k + 2u] == z) {
            h->vals[slot] = value;
            return 1;
        }
        slot = (slot + 1u) & mask;
    }
    return 0;
}

static int32_t fdg_hash_lookup(const fdg_hash * h, int32_t x, int32_t y, int32_t z) {
    if (h == NULL || h->capacity == 0) {
        return -1;
    }
    size_t mask = h->capacity - 1u;
    size_t slot = (size_t) fdg_hash3(x, y, z) & mask;
    for (size_t probe = 0; probe < h->capacity; ++probe) {
        size_t k = slot * 3u;
        if (!h->used[slot]) {
            return -1;
        }
        if (h->keys[k + 0u] == x && h->keys[k + 1u] == y && h->keys[k + 2u] == z) {
            return h->vals[slot];
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

static float fdg_sigmoid(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

static float fdg_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

static int fdg_lookup_quad(
    const fdg_hash * h,
    const int32_t base[3],
    int axis,
    int32_t out[4]) {
    static const int32_t offsets[3][4][3] = {
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
        {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
        {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
    };
    for (int i = 0; i < 4; ++i) {
        out[i] = fdg_hash_lookup(
            h,
            base[0] + offsets[axis][i][0],
            base[1] + offsets[axis][i][1],
            base[2] + offsets[axis][i][2]);
        if (out[i] < 0) {
            return 0;
        }
    }
    return 1;
}

static void fdg_write_face(int32_t * faces, int64_t face, int32_t a, int32_t b, int32_t c) {
    faces[face * 3 + 0] = a;
    faces[face * 3 + 1] = b;
    faces[face * 3 + 2] = c;
}

void trellis_mesh_free(trellis_mesh_host * mesh) {
    if (mesh == NULL) {
        return;
    }
    free(mesh->vertices);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

trellis_status trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    trellis_mesh_host * mesh_out) {
    if (coords == NULL || feats == NULL || mesh_out == NULL || n < 0 || channels < 7 || resolution <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (n > (int64_t) INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    if (n == 0) {
        return TRELLIS_STATUS_OK;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    fdg_hash h;
    if (!fdg_hash_init(&h, n)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    float * vertices = (float *) malloc((size_t) n * 3u * sizeof(float));
    float * weights = (float *) malloc((size_t) n * sizeof(float));
    if (vertices == NULL || weights == NULL) {
        free(vertices);
        free(weights);
        fdg_hash_free(&h);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const float inv_res = 1.0f / (float) resolution;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t x = coords[i * 4 + 1];
        const int32_t y = coords[i * 4 + 2];
        const int32_t z = coords[i * 4 + 3];
        if (!fdg_hash_insert(&h, x, y, z, (int32_t) i)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        const float * f = feats + (size_t) i * (size_t) channels;
        for (int axis = 0; axis < 3; ++axis) {
            float local = 2.0f * fdg_sigmoid(f[axis]) - 0.5f;
            vertices[i * 3 + axis] = ((float) coords[i * 4 + 1 + axis] + local) * inv_res - 0.5f;
        }
        weights[i] = fdg_softplus(f[6]);
    }

    int64_t quads = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float * f = feats + (size_t) i * (size_t) channels;
        int32_t base[3] = {coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]};
        int32_t q[4];
        for (int axis = 0; axis < 3; ++axis) {
            if (f[3 + axis] > 0.0f && fdg_lookup_quad(&h, base, axis, q)) {
                ++quads;
            }
        }
    }

    int64_t face_count = quads * 2;
    int32_t * faces = NULL;
    if (face_count > 0) {
        faces = (int32_t *) malloc((size_t) face_count * 3u * sizeof(int32_t));
        if (faces == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
    }

    int64_t face = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float * f = feats + (size_t) i * (size_t) channels;
        int32_t base[3] = {coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]};
        int32_t q[4];
        for (int axis = 0; axis < 3; ++axis) {
            if (f[3 + axis] <= 0.0f || !fdg_lookup_quad(&h, base, axis, q)) {
                continue;
            }
            float w02 = weights[q[0]] * weights[q[2]];
            float w13 = weights[q[1]] * weights[q[3]];
            if (w02 > w13) {
                fdg_write_face(faces, face++, q[0], q[1], q[2]);
                fdg_write_face(faces, face++, q[0], q[2], q[3]);
            } else {
                fdg_write_face(faces, face++, q[0], q[1], q[3]);
                fdg_write_face(faces, face++, q[3], q[1], q[2]);
            }
        }
    }

    mesh_out->vertices = vertices;
    mesh_out->faces = faces;
    mesh_out->n_vertices = n;
    mesh_out->n_faces = face;
    vertices = NULL;
    faces = NULL;

cleanup:
    free(vertices);
    free(faces);
    free(weights);
    fdg_hash_free(&h);
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(mesh_out);
    }
    return status;
}
