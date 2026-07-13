#include "image_to_3d_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TSLAT_VERSION 1u
#define TSLAT_HEADER_BYTES 80u
#define TSLAT_COORD_FRAME_TRELLIS_DECODER_V1 1u
#define TSLAT_DECODER_AABB_LIMIT 0.5001f

static const unsigned char tslat_magic[8] = {'T', 'S', 'L', 'A', 'T', '0', '1', '\0'};

static int write_exact(FILE * file, const void * data, size_t bytes) {
    return bytes == 0 || fwrite(data, 1, bytes, file) == bytes;
}

static int read_exact(FILE * file, void * data, size_t bytes) {
    return bytes == 0 || fread(data, 1, bytes, file) == bytes;
}

static int write_u32_le(FILE * file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char) value,
        (unsigned char) (value >> 8u),
        (unsigned char) (value >> 16u),
        (unsigned char) (value >> 24u),
    };
    return write_exact(file, bytes, sizeof(bytes));
}

static int write_u64_le(FILE * file, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = (unsigned char) (value >> (unsigned) (i * 8));
    return write_exact(file, bytes, sizeof(bytes));
}

static int write_f32_le(FILE * file, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32_le(file, bits);
}

static int read_u32_le(FILE * file, uint32_t * value_out) {
    unsigned char bytes[4];
    if (!read_exact(file, bytes, sizeof(bytes))) return 0;
    *value_out = (uint32_t) bytes[0] |
        (uint32_t) bytes[1] << 8u |
        (uint32_t) bytes[2] << 16u |
        (uint32_t) bytes[3] << 24u;
    return 1;
}

static int read_u64_le(FILE * file, uint64_t * value_out) {
    unsigned char bytes[8];
    if (!read_exact(file, bytes, sizeof(bytes))) return 0;
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= (uint64_t) bytes[i] << (unsigned) (i * 8);
    *value_out = value;
    return 1;
}

static int read_f32_le(FILE * file, float * value_out) {
    uint32_t bits = 0;
    if (!read_u32_le(file, &bits)) return 0;
    memcpy(value_out, &bits, sizeof(bits));
    return 1;
}

static int safe_payload_sizes(
    int64_t n_coords,
    int channels,
    uint64_t * coords_bytes_out,
    uint64_t * feats_bytes_out) {
    if (n_coords <= 0 || channels <= 0 ||
        n_coords > INT64_MAX / 4 || n_coords > INT64_MAX / channels) return 0;
    const uint64_t n = (uint64_t) n_coords;
    if (n > UINT64_MAX / (4u * sizeof(int32_t)) ||
        n > UINT64_MAX / ((uint64_t) channels * sizeof(float))) return 0;
    *coords_bytes_out = n * 4u * sizeof(int32_t);
    *feats_bytes_out = n * (uint64_t) channels * sizeof(float);
    return *coords_bytes_out <= SIZE_MAX && *feats_bytes_out <= SIZE_MAX &&
        *coords_bytes_out <= UINT64_MAX - TSLAT_HEADER_BYTES &&
        *feats_bytes_out <= UINT64_MAX - TSLAT_HEADER_BYTES - *coords_bytes_out;
}

static int mesh_anchor_aabb(
    const trellis_mesh_host * mesh,
    float minimum[3],
    float maximum[3]) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->n_vertices <= 0) return 0;
    for (int axis = 0; axis < 3; ++axis) {
        minimum[axis] = mesh->vertices[axis];
        maximum[axis] = mesh->vertices[axis];
    }
    for (int64_t vertex = 0; vertex < mesh->n_vertices; ++vertex) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = mesh->vertices[(size_t) vertex * 3u + (size_t) axis];
            if (!isfinite(value)) return 0;
            if (value < minimum[axis]) minimum[axis] = value;
            if (value > maximum[axis]) maximum[axis] = value;
        }
    }
    float max_extent = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float extent = maximum[axis] - minimum[axis];
        if (extent > max_extent) max_extent = extent;
    }
    return max_extent > 0.0f && isfinite(max_extent);
}

static int cache_info_valid(const trellis_shape_latent_cache_info * info) {
    if (info == NULL || info->version != TSLAT_VERSION ||
        (info->resolution != 512 && info->resolution != 1024) ||
        info->channels != 32 || info->n_coords <= 0) return 0;
    float max_extent = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(info->anchor_aabb_min[axis]) || !isfinite(info->anchor_aabb_max[axis]) ||
            info->anchor_aabb_max[axis] < info->anchor_aabb_min[axis] ||
            info->anchor_aabb_min[axis] < -TSLAT_DECODER_AABB_LIMIT ||
            info->anchor_aabb_max[axis] > TSLAT_DECODER_AABB_LIMIT) return 0;
        const float extent = info->anchor_aabb_max[axis] - info->anchor_aabb_min[axis];
        if (extent > max_extent) max_extent = extent;
    }
    return max_extent > 0.0f;
}

trellis_status trellis_shape_latent_cache_write(
    const char * path,
    const trellis_structured_latent * latent,
    const trellis_mesh_host * anchor_mesh,
    trellis_shape_latent_cache_info * info_out) {
    if (path == NULL || path[0] == '\0' || latent == NULL ||
        latent->coords_bxyz == NULL || latent->feats == NULL ||
        latent->channels != 32 ||
        (latent->resolution != 512 && latent->resolution != 1024)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_shape_latent_cache_info info;
    memset(&info, 0, sizeof(info));
    info.version = TSLAT_VERSION;
    info.resolution = latent->resolution;
    info.channels = latent->channels;
    info.n_coords = latent->n_coords;
    if (!mesh_anchor_aabb(anchor_mesh, info.anchor_aabb_min, info.anchor_aabb_max)) {
        TRELLIS_ERROR("shape latent cache: anchor mesh has no finite non-empty AABB");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!cache_info_valid(&info)) {
        TRELLIS_ERROR(
            "shape latent cache: decoder-frame anchor is out of range "
            "min=(%.9g,%.9g,%.9g) max=(%.9g,%.9g,%.9g)",
            (double) info.anchor_aabb_min[0],
            (double) info.anchor_aabb_min[1],
            (double) info.anchor_aabb_min[2],
            (double) info.anchor_aabb_max[0],
            (double) info.anchor_aabb_max[1],
            (double) info.anchor_aabb_max[2]);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    uint64_t coords_bytes = 0;
    uint64_t feats_bytes = 0;
    if (!safe_payload_sizes(latent->n_coords, latent->channels, &coords_bytes, &feats_bytes)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t row = 0; row < latent->n_coords; ++row) {
        const int32_t * coord = latent->coords_bxyz + (size_t) row * 4u;
        if (coord[0] != 0) return TRELLIS_STATUS_PARSE_ERROR;
        for (int axis = 1; axis < 4; ++axis) {
            if (coord[axis] < 0 || coord[axis] >= latent->resolution) return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    const int64_t values = latent->n_coords * (int64_t) latent->channels;
    for (int64_t index = 0; index < values; ++index) {
        if (!isfinite(latent->feats[index])) return TRELLIS_STATUS_PARSE_ERROR;
    }

    const size_t path_length = strlen(path);
    if (path_length > SIZE_MAX - 5u) return TRELLIS_STATUS_INVALID_ARGUMENT;
    char * temporary = (char *) malloc(path_length + 5u);
    if (temporary == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    snprintf(temporary, path_length + 5u, "%s.tmp", path);
    FILE * file = fopen(temporary, "wb");
    if (file == NULL) {
        free(temporary);
        return TRELLIS_STATUS_IO_ERROR;
    }

    int ok = write_exact(file, tslat_magic, sizeof(tslat_magic)) &&
        write_u32_le(file, TSLAT_VERSION) &&
        write_u32_le(file, TSLAT_HEADER_BYTES) &&
        write_u32_le(file, (uint32_t) latent->resolution) &&
        write_u32_le(file, (uint32_t) latent->channels) &&
        write_u64_le(file, (uint64_t) latent->n_coords) &&
        write_u32_le(file, TSLAT_COORD_FRAME_TRELLIS_DECODER_V1) &&
        write_u32_le(file, 0u);
    for (int axis = 0; ok && axis < 3; ++axis) ok = write_f32_le(file, info.anchor_aabb_min[axis]);
    for (int axis = 0; ok && axis < 3; ++axis) ok = write_f32_le(file, info.anchor_aabb_max[axis]);
    ok = ok && write_u64_le(file, coords_bytes) && write_u64_le(file, feats_bytes);
    for (int64_t index = 0; ok && index < latent->n_coords * 4; ++index) {
        ok = write_u32_le(file, (uint32_t) latent->coords_bxyz[index]);
    }
    for (int64_t index = 0; ok && index < values; ++index) {
        ok = write_f32_le(file, latent->feats[index]);
    }
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    if (ok && rename(temporary, path) != 0) ok = 0;
    if (!ok) remove(temporary);
    free(temporary);
    if (!ok) return TRELLIS_STATUS_IO_ERROR;
    if (info_out != NULL) *info_out = info;
    TRELLIS_INFO(
        "shape latent cache: wrote %s tokens=%lld resolution=%d bytes=%llu",
        path,
        (long long) latent->n_coords,
        latent->resolution,
        (unsigned long long) (TSLAT_HEADER_BYTES + coords_bytes + feats_bytes));
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_shape_latent_cache_read(
    const char * path,
    trellis_structured_latent * latent_out,
    trellis_shape_latent_cache_info * info_out) {
    if (path == NULL || path[0] == '\0' || latent_out == NULL || info_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(latent_out, 0, sizeof(*latent_out));
    memset(info_out, 0, sizeof(*info_out));
    FILE * file = fopen(path, "rb");
    if (file == NULL) return TRELLIS_STATUS_NOT_FOUND;

    unsigned char magic[8];
    uint32_t version = 0, header_bytes = 0, resolution = 0, channels = 0;
    uint32_t coord_frame = 0, reserved = 0;
    uint64_t n_coords = 0, coords_bytes = 0, feats_bytes = 0;
    int ok = read_exact(file, magic, sizeof(magic)) &&
        read_u32_le(file, &version) && read_u32_le(file, &header_bytes) &&
        read_u32_le(file, &resolution) && read_u32_le(file, &channels) &&
        read_u64_le(file, &n_coords) && read_u32_le(file, &coord_frame) &&
        read_u32_le(file, &reserved);
    trellis_shape_latent_cache_info info;
    memset(&info, 0, sizeof(info));
    info.version = version;
    info.resolution = (int) resolution;
    info.channels = (int) channels;
    if (n_coords <= (uint64_t) INT64_MAX) info.n_coords = (int64_t) n_coords;
    for (int axis = 0; ok && axis < 3; ++axis) ok = read_f32_le(file, &info.anchor_aabb_min[axis]);
    for (int axis = 0; ok && axis < 3; ++axis) ok = read_f32_le(file, &info.anchor_aabb_max[axis]);
    ok = ok && read_u64_le(file, &coords_bytes) && read_u64_le(file, &feats_bytes);
    uint64_t expected_coords = 0, expected_feats = 0;
    if (!ok || memcmp(magic, tslat_magic, sizeof(magic)) != 0 ||
        header_bytes != TSLAT_HEADER_BYTES || coord_frame != TSLAT_COORD_FRAME_TRELLIS_DECODER_V1 ||
        reserved != 0 || n_coords > (uint64_t) INT64_MAX || !cache_info_valid(&info) ||
        !safe_payload_sizes(info.n_coords, info.channels, &expected_coords, &expected_feats) ||
        coords_bytes != expected_coords || feats_bytes != expected_feats) {
        fclose(file);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const long file_size = ftell(file);
    const uint64_t expected_size = TSLAT_HEADER_BYTES + coords_bytes + feats_bytes;
    if (file_size < 0 || (uint64_t) file_size != expected_size ||
        fseek(file, TSLAT_HEADER_BYTES, SEEK_SET) != 0) {
        fclose(file);
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    int32_t * coords = (int32_t *) malloc((size_t) coords_bytes);
    float * feats = (float *) malloc((size_t) feats_bytes);
    if (coords == NULL || feats == NULL) {
        free(coords);
        free(feats);
        fclose(file);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t index = 0; ok && index < info.n_coords * 4; ++index) {
        uint32_t value = 0;
        ok = read_u32_le(file, &value);
        coords[index] = (int32_t) value;
    }
    const int64_t values = info.n_coords * (int64_t) info.channels;
    for (int64_t index = 0; ok && index < values; ++index) ok = read_f32_le(file, &feats[index]);
    if (fclose(file) != 0) ok = 0;
    for (int64_t row = 0; ok && row < info.n_coords; ++row) {
        const int32_t * coord = coords + (size_t) row * 4u;
        ok = coord[0] == 0;
        for (int axis = 1; ok && axis < 4; ++axis) ok = coord[axis] >= 0 && coord[axis] < info.resolution;
    }
    for (int64_t index = 0; ok && index < values; ++index) ok = isfinite(feats[index]);
    if (!ok) {
        free(coords);
        free(feats);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    latent_out->coords_bxyz = coords;
    latent_out->n_coords = info.n_coords;
    latent_out->resolution = info.resolution;
    latent_out->feats = feats;
    latent_out->channels = info.channels;
    *info_out = info;
    TRELLIS_INFO(
        "shape latent cache: loaded %s tokens=%lld resolution=%d",
        path,
        (long long) info.n_coords,
        info.resolution);
    return TRELLIS_STATUS_OK;
}
