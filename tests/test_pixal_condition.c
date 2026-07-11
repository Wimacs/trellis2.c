#include "pixal_projection.h"
#include "trellis_platform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void check_close(float actual, float expected, float tolerance, int line) {
    const float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    if (fabsf(actual - expected) > tolerance * scale) {
        fprintf(stderr, "CHECK_CLOSE failed at %s:%d: actual=%g expected=%g\n",
            __FILE__, line, actual, expected);
        ++failures;
    }
}

#define CHECK_CLOSE(actual, expected, tolerance) \
    check_close((actual), (expected), (tolerance), __LINE__)

static void test_camera_distance_resolution(void) {
    float distance = -1.0f;
    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.8575560450553894f, 1.0f, 0.0f, &distance) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(distance, 1.0937500142266299f, 1e-6f);

    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.2f, 1.0f, 0.0f, &distance) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(distance, 4.983322211629619f, 1e-6f);

    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.2f, 2.0f, 0.0f, &distance) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(distance, 2.4916611058148096f, 1e-6f);

    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.2f, 2.0f, 3.25f, &distance) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(distance, 3.25f, 0.0f);

    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.2f, 1.0f, -1.0f, &distance) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(distance, 4.983322211629619f, 1e-6f);
    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.0f, 1.0f, 0.0f, &distance) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(trellis_pixal_resolve_camera_distance(
        0.2f, 0.0f, 0.0f, &distance) == TRELLIS_STATUS_INVALID_ARGUMENT);
}

static void test_dense_projection_matches_grid_sample_golden(void) {
    /* BHWC/last-channel layout: [[1,2],[3,4]], with a second scaled channel. */
    const float patch_features[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
    };
    const float expected[] = {
        2.79375005f,
        2.92708349f,
        1.99375010f,
        1.59374988f,
        3.19375014f,
        3.59375000f,
        2.39374995f,
        2.26041651f,
    };
    trellis_pixal_camera camera = {
        .camera_angle_x = 1.57079632679489661923f,
        .distance = 2.0f,
        .mesh_scale = 1.0f,
    };
    float projected[8 * 2] = {0};
    CHECK_TRUE(trellis_pixal_project_patch_features_dense_f32(
        patch_features, 2, 2, 2, 32, 2, &camera, projected, 16) == TRELLIS_STATUS_OK);
    for (int i = 0; i < 8; ++i) {
        CHECK_CLOSE(projected[2 * i + 0], expected[i], 1e-6f);
        CHECK_CLOSE(projected[2 * i + 1], 10.0f * expected[i], 1e-6f);
    }
}

static void test_sparse_projection_preserves_coord_order(void) {
    const float patch_features[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const int32_t coords[] = {
        0, 1, 1, 1,
        0, 0, 0, 0,
        0, 1, 0, 1,
    };
    const float expected[] = {2.26041651f, 2.79375005f, 3.59375000f};
    trellis_pixal_camera camera = {
        .camera_angle_x = 1.57079632679489661923f,
        .distance = 2.0f,
        .mesh_scale = 1.0f,
    };
    float projected[3] = {0};
    CHECK_TRUE(trellis_pixal_project_patch_features_sparse_f32(
        patch_features, 2, 2, 1, 32, 2, &camera,
        coords, 3, projected, 3) == TRELLIS_STATUS_OK);
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(projected[i], expected[i], 1e-6f);
    }

    const int32_t invalid_batch[] = {1, 0, 0, 0};
    CHECK_TRUE(trellis_pixal_project_patch_features_sparse_f32(
        patch_features, 2, 2, 1, 32, 2, &camera,
        invalid_batch, 1, projected, 1) == TRELLIS_STATUS_INVALID_ARGUMENT);
}

static void test_pixal_pipeline_options_versioning(void) {
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    CHECK_TRUE(TRELLIS_PIXAL3D_OPTIONS_V1_SIZE <= sizeof(pixal_options));
    CHECK_TRUE(
        trellis_pipeline_image_to_gltf_ex(NULL, NULL) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    pixal_options.struct_size = TRELLIS_PIXAL3D_OPTIONS_V1_SIZE - 1u;
    CHECK_TRUE(
        trellis_pipeline_image_to_gltf_ex(NULL, &pixal_options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(
        trellis_pipeline_image_to_gltf(NULL) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
}

static int write_pixal_marker_checkpoint(const char * path) {
    const char * header =
        "{\"blocks.0.cross_attn.proj_linear.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[0,4]}}";
    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    const uint64_t header_size = (uint64_t) strlen(header);
    unsigned char header_size_le[8];
    for (int i = 0; i < 8; ++i) {
        header_size_le[i] = (unsigned char) ((header_size >> (8 * i)) & 0xffu);
    }
    const float value = 0.0f;
    int ok =
        fwrite(header_size_le, 1, sizeof(header_size_le), file) == sizeof(header_size_le) &&
        fwrite(header, 1, (size_t) header_size, file) == (size_t) header_size &&
        fwrite(&value, 1, sizeof(value), file) == sizeof(value);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int write_opaque_ppm(const char * path) {
    static const unsigned char pixels[6] = {255, 0, 0, 0, 255, 0};
    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    int ok =
        fputs("P6\n2 1\n255\n", file) >= 0 &&
        fwrite(pixels, 1, sizeof(pixels), file) == sizeof(pixels);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void test_pixal_pipeline_rejects_opaque_input_without_birefnet(void) {
    char model_dir[PATH_MAX];
    char ckpts_dir[PATH_MAX];
    char checkpoint_path[PATH_MAX];
    char image_path[PATH_MAX];
    if (!trellis_make_temp_path(
            model_dir, sizeof(model_dir), "trellis_pixal_opaque_model", NULL) ||
        !trellis_make_temp_path(
            image_path, sizeof(image_path), "trellis_pixal_opaque", ".ppm") ||
        snprintf(ckpts_dir, sizeof(ckpts_dir), "%s/ckpts", model_dir) < 0 ||
        snprintf(
            checkpoint_path,
            sizeof(checkpoint_path),
            "%s/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            ckpts_dir) < 0 ||
        !trellis_mkdir_p(ckpts_dir) ||
        !write_pixal_marker_checkpoint(checkpoint_path) ||
        !write_opaque_ppm(image_path)) {
        CHECK_TRUE(0);
        trellis_unlink(image_path);
        trellis_unlink(checkpoint_path);
        remove(ckpts_dir);
        remove(model_dir);
        return;
    }

    trellis_image_to_gltf_options options = {0};
    options.model_dir = model_dir;
    options.dino_dir = model_dir;
    options.image_path = image_path;
    options.device = 0;
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;

    const char * ambient_birefnet = getenv("TRELLIS_BIREFNET_PATH");
    char * saved_birefnet = NULL;
    if (ambient_birefnet != NULL) {
        saved_birefnet = (char *) malloc(strlen(ambient_birefnet) + 1u);
        if (saved_birefnet != NULL) {
            strcpy(saved_birefnet, ambient_birefnet);
        }
    }
    CHECK_TRUE(ambient_birefnet == NULL || saved_birefnet != NULL);
    CHECK_TRUE(trellis_unsetenv("TRELLIS_BIREFNET_PATH") == 0);
    CHECK_TRUE(
        trellis_pipeline_image_to_gltf_ex(&options, &pixal_options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    if (saved_birefnet != NULL) {
        CHECK_TRUE(trellis_setenv(
            "TRELLIS_BIREFNET_PATH", saved_birefnet, 1) == 0);
    } else {
        CHECK_TRUE(trellis_unsetenv("TRELLIS_BIREFNET_PATH") == 0);
    }
    free(saved_birefnet);

    trellis_unlink(image_path);
    trellis_unlink(checkpoint_path);
    remove(ckpts_dir);
    remove(model_dir);
}

int main(void) {
    test_camera_distance_resolution();
    test_dense_projection_matches_grid_sample_golden();
    test_sparse_projection_preserves_coord_order();
    test_pixal_pipeline_options_versioning();
    test_pixal_pipeline_rejects_opaque_input_without_birefnet();
    if (failures != 0) {
        fprintf(stderr, "%d Pixal3D condition test failures\n", failures);
        return 1;
    }
    printf("Pixal3D condition tests passed\n");
    return 0;
}
