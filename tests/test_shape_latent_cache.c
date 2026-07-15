#include "trellis_platform.h"
#include "image_to_3d_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            return 1; \
        } \
    } while (0)

static int make_cache_path(char path[PATH_MAX]) {
    FILE * file = trellis_open_temp_file(
        path,
        PATH_MAX,
        "trellis_shape_latent",
        ".tslat");
    if (file == NULL) return 0;
    const int closed = fclose(file) == 0;
    const int removed = trellis_unlink(path) == 0;
    return closed && removed;
}

static int rewrite_file_prefix(const char * path, size_t prefix_bytes) {
    if (path == NULL || prefix_bytes == 0) return 0;
    unsigned char * prefix = (unsigned char *) malloc(prefix_bytes);
    if (prefix == NULL) return 0;
    FILE * file = fopen(path, "rb");
    int ok = file != NULL && fread(prefix, 1, prefix_bytes, file) == prefix_bytes;
    if (file != NULL && fclose(file) != 0) ok = 0;
    file = ok ? fopen(path, "wb") : NULL;
    if (file == NULL) ok = 0;
    if (ok && fwrite(prefix, 1, prefix_bytes, file) != prefix_bytes) ok = 0;
    if (file != NULL && fclose(file) != 0) ok = 0;
    free(prefix);
    return ok;
}

static void fill_fixture(
    trellis_structured_latent * latent,
    trellis_mesh_host * mesh,
    int32_t coords[12],
    float feats[96],
    float vertices[9],
    int32_t faces[3]) {
    const int32_t fixture_coords[12] = {
        0, 101, 202, 303,
        0, 102, 202, 303,
        0, 101, 203, 303,
    };
    memcpy(coords, fixture_coords, sizeof(fixture_coords));
    for (int i = 0; i < 96; ++i) feats[i] = (float) i / 17.0f - 2.0f;
    const float fixture_vertices[9] = {
        -0.4f, -0.3f, -0.2f,
         0.4f, -0.3f,  0.1f,
         0.0f,  0.3f,  0.2f,
    };
    memcpy(vertices, fixture_vertices, sizeof(fixture_vertices));
    faces[0] = 0;
    faces[1] = 1;
    faces[2] = 2;

    memset(latent, 0, sizeof(*latent));
    latent->coords_bxyz = coords;
    latent->n_coords = 3;
    latent->resolution = 512;
    latent->feats = feats;
    latent->channels = 32;
    memset(mesh, 0, sizeof(*mesh));
    mesh->vertices = vertices;
    mesh->faces = faces;
    mesh->n_vertices = 3;
    mesh->n_faces = 1;
}

int main(void) {
    char path[PATH_MAX];
    CHECK(make_cache_path(path));

    int32_t coords[12];
    float feats[96];
    float vertices[9];
    int32_t faces[3];
    trellis_structured_latent source;
    trellis_mesh_host mesh;
    fill_fixture(&source, &mesh, coords, feats, vertices, faces);

    trellis_shape_latent_cache_info written;
    memset(&written, 0, sizeof(written));
    CHECK(trellis_shape_latent_cache_write(path, &source, &mesh, &written) ==
        TRELLIS_STATUS_OK);
    CHECK(written.version == 1);
    CHECK(written.resolution == 512);
    CHECK(written.channels == 32);
    CHECK(written.n_coords == 3);
    CHECK(fabsf(written.anchor_aabb_min[0] + 0.4f) < 1e-7f);
    CHECK(fabsf(written.anchor_aabb_max[2] - 0.2f) < 1e-7f);

    trellis_structured_latent loaded;
    trellis_shape_latent_cache_info read_info;
    memset(&loaded, 0, sizeof(loaded));
    memset(&read_info, 0, sizeof(read_info));
    CHECK(trellis_shape_latent_cache_read(path, &loaded, &read_info) ==
        TRELLIS_STATUS_OK);
    CHECK(memcmp(coords, loaded.coords_bxyz, sizeof(coords)) == 0);
    CHECK(memcmp(feats, loaded.feats, sizeof(feats)) == 0);
    CHECK(read_info.version == written.version);
    CHECK(read_info.resolution == written.resolution);
    CHECK(read_info.channels == written.channels);
    CHECK(read_info.n_coords == written.n_coords);
    CHECK(memcmp(
        read_info.anchor_aabb_min,
        written.anchor_aabb_min,
        sizeof(written.anchor_aabb_min)) == 0);
    CHECK(memcmp(
        read_info.anchor_aabb_max,
        written.anchor_aabb_max,
        sizeof(written.anchor_aabb_max)) == 0);
    trellis_structured_latent_free(&loaded);

    /* The decoder can place a boundary-voxel vertex half a voxel outside the
     * nominal cube. This is valid FlexDualGrid output and must remain
     * cacheable at every supported resolution. */
    const float decoder_limit =
        trellis_shape_latent_decoder_aabb_limit(source.resolution);
    vertices[0] = -decoder_limit + 1e-7f;
    vertices[3] = decoder_limit - 1e-7f;
    CHECK(trellis_shape_latent_cache_write(path, &source, &mesh, NULL) ==
        TRELLIS_STATUS_OK);
    vertices[0] = -decoder_limit - 1e-4f;
    CHECK(trellis_shape_latent_cache_write(path, &source, &mesh, NULL) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    fill_fixture(&source, &mesh, coords, feats, vertices, faces);

    /* Reusing a diagnostic output path must atomically replace the prior
     * cache on Windows as well as POSIX. */
    feats[0] = 123.25f;
    CHECK(trellis_shape_latent_cache_write(path, &source, &mesh, NULL) ==
        TRELLIS_STATUS_OK);
    CHECK(trellis_shape_latent_cache_read(path, &loaded, &read_info) ==
        TRELLIS_STATUS_OK);
    CHECK(loaded.feats[0] == feats[0]);
    trellis_structured_latent_free(&loaded);
    char sibling[PATH_MAX + 80];
    for (int attempt = 0; attempt < 100; ++attempt) {
        CHECK(snprintf(
            sibling,
            sizeof(sibling),
            "%s.trellis-tmp-%ld-%d.tslat",
            path,
            trellis_getpid(),
            attempt) > 0);
        CHECK(!trellis_access_exists(sibling));
    }

    FILE * file = fopen(path, "r+b");
    CHECK(file != NULL);
    CHECK(fputc('X', file) != EOF);
    CHECK(fclose(file) == 0);
    CHECK(trellis_shape_latent_cache_read(path, &loaded, &read_info) ==
        TRELLIS_STATUS_PARSE_ERROR);
    CHECK(loaded.coords_bxyz == NULL);
    CHECK(loaded.feats == NULL);

    CHECK(trellis_unlink(path) == 0);
    CHECK(trellis_shape_latent_cache_write(path, &source, &mesh, NULL) ==
        TRELLIS_STATUS_OK);
    CHECK(rewrite_file_prefix(path, 81));
    CHECK(trellis_shape_latent_cache_read(path, &loaded, &read_info) ==
        TRELLIS_STATUS_PARSE_ERROR);
    CHECK(loaded.coords_bxyz == NULL);
    CHECK(loaded.feats == NULL);

    CHECK(trellis_unlink(path) == 0);
    puts("shape latent cache tests passed");
    return 0;
}
