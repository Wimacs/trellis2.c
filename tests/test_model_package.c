#include "trellis_model_package.h"
#include "trellis_platform.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define REMOVE_DIR _rmdir
#else
#include <unistd.h>
#define REMOVE_DIR rmdir
#endif

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

static int make_temp_directory(char path[PATH_MAX], const char * prefix) {
    if (!trellis_make_temp_path(path, PATH_MAX, prefix, NULL)) return 0;
    trellis_unlink(path);
    return trellis_mkdir_one(path);
}

static int write_text(const char * path, const char * text) {
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    const size_t length = strlen(text);
    const int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok;
}

static int write_marker_checkpoint(const char * root, int pixal) {
    char directory[PATH_MAX];
    char path[PATH_MAX];
    if (snprintf(directory, sizeof(directory), "%s/ckpts", root) < 0 ||
        !trellis_mkdir_one(directory) ||
        snprintf(path, sizeof(path), "%s/ss_flow_img_dit_1_3B_64_bf16.safetensors", directory) < 0) {
        return 0;
    }
    const char * header = pixal ?
        "{\"blocks.0.cross_attn.proj_linear.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}" :
        "{\"legacy.trellis.marker\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    const uint64_t header_size = (uint64_t) strlen(header);
    unsigned char encoded[8];
    for (int i = 0; i < 8; ++i) encoded[i] = (unsigned char) ((header_size >> (8 * i)) & 0xffu);
    const float zero = 0.0f;
    const int ok = fwrite(encoded, 1, sizeof(encoded), file) == sizeof(encoded) &&
        fwrite(header, 1, (size_t) header_size, file) == (size_t) header_size &&
        fwrite(&zero, 1, sizeof(zero), file) == sizeof(zero) && fclose(file) == 0;
    return ok;
}

static void remove_package_tree(const char * root, int has_manifest, int has_checkpoint) {
    char path[PATH_MAX];
    if (has_manifest) {
        snprintf(path, sizeof(path), "%s/model.json", root);
        trellis_unlink(path);
    }
    if (has_checkpoint) {
        snprintf(path, sizeof(path), "%s/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", root);
        trellis_unlink(path);
        snprintf(path, sizeof(path), "%s/ckpts", root);
        REMOVE_DIR(path);
    }
    REMOVE_DIR(root);
}

static const char * valid_test_manifest(void) {
    return
        "{"
        "\"schema_version\":1,"
        "\"id\":\"fixture\","
        "\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\","
        "\"profile\":\"single_tensor\","
        "\"components\":[{"
        "\"role\":\"linear\","
        "\"architecture\":\"test_linear\","
        "\"weights\":\"weights/linear.safetensors\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"none\"}"
        "}]}";
}

static void test_manifest_load(void) {
    char root[PATH_MAX];
    char path[PATH_MAX];
    CHECK_TRUE(make_temp_directory(root, "trellis_model_manifest"));
    CHECK_TRUE(snprintf(path, sizeof(path), "%s/model.json", root) > 0);
    CHECK_TRUE(write_text(path, valid_test_manifest()));

    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(package.source == TRELLIS_MODEL_PACKAGE_SOURCE_MANIFEST);
    CHECK_TRUE(package.schema_version == 1);
    CHECK_TRUE(strcmp(package.family, "test_fixture") == 0);
    CHECK_TRUE(strcmp(package.task, "test_weight_binding") == 0);
    CHECK_TRUE(package.legacy_pixal_marker_probes == 0);
    const trellis_model_component_instance * component =
        trellis_model_package_find_component(&package, "linear");
    CHECK_TRUE(component != NULL);
    CHECK_TRUE(strcmp(component->architecture, "test_linear") == 0);
    CHECK_TRUE(component->execution.compute_dtype == TRELLIS_DTYPE_F32);
    CHECK_TRUE(component->execution.attention == TRELLIS_ATTENTION_NONE);
    trellis_model_package_free(&package);
    remove_package_tree(root, 1, 0);
}

static void test_repository_templates(void) {
    char root[PATH_MAX];
    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(snprintf(root, sizeof(root), "%s/models/trellis2", TRELLIS_TEST_SOURCE_DIR) > 0);
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(package.family, "trellis2") == 0);
    CHECK_TRUE(strcmp(package.profile, "512") == 0);
    const trellis_model_component_instance * shape_encoder =
        trellis_model_package_find_component(&package, "shape_encoder");
    CHECK_TRUE(shape_encoder != NULL);
    CHECK_TRUE(strcmp(shape_encoder->architecture, "sparse_unet_vae_encoder") == 0);
    CHECK_TRUE(shape_encoder->execution.compute_dtype == TRELLIS_DTYPE_F32);
    const trellis_model_component_instance * texture_encoder =
        trellis_model_package_find_component(&package, "texture_encoder");
    CHECK_TRUE(texture_encoder != NULL);
    CHECK_TRUE(strcmp(texture_encoder->architecture, "sparse_unet_vae_encoder") == 0);
    const trellis_model_component_instance * flow =
        trellis_model_package_find_component(&package, "shape_flow_1024");
    CHECK_TRUE(flow != NULL && flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(flow->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.emulate_bf16_blocks == 0);
    char resolved[PATH_MAX];
    CHECK_TRUE(trellis_model_package_resolve_component_path(
        &package, "shape_flow_1024", resolved, sizeof(resolved)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strstr(resolved, "/models/trellis2/ckpts/") != NULL);
    trellis_model_package_free(&package);

    CHECK_TRUE(snprintf(root, sizeof(root), "%s/models/segvigen", TRELLIS_TEST_SOURCE_DIR) > 0);
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(package.family, "trellis2") == 0);
    CHECK_TRUE(strcmp(package.task, "mesh_segmentation") == 0);
    CHECK_TRUE(strcmp(package.profile, "512_full") == 0);
    const trellis_model_component_instance * segmentation_flow =
        trellis_model_package_find_component(&package, "segmentation_flow");
    CHECK_TRUE(segmentation_flow != NULL);
    CHECK_TRUE(strcmp(segmentation_flow->architecture, "trellis_dit_flow") == 0);
    CHECK_TRUE(segmentation_flow->execution.compute_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(segmentation_flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(segmentation_flow->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(segmentation_flow->execution.emulate_bf16_blocks == 1);
    trellis_model_package_free(&package);

    CHECK_TRUE(snprintf(root, sizeof(root), "%s/models/pixal3d", TRELLIS_TEST_SOURCE_DIR) > 0);
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(package.family, "pixal3d") == 0);
    CHECK_TRUE(strcmp(package.profile, "1024_cascade") == 0);
    flow = trellis_model_package_find_component(&package, "shape_flow_1024");
    CHECK_TRUE(flow != NULL && flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(flow->execution.compute_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.emulate_bf16_blocks == 1);
    CHECK_TRUE(trellis_model_package_find_component(&package, "naf_encoder") != NULL);
    trellis_model_package_free(&package);

    CHECK_TRUE(snprintf(root, sizeof(root), "%s/models/tokenskin", TRELLIS_TEST_SOURCE_DIR) > 0);
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(package.family, "tokenskin") == 0);
    CHECK_TRUE(strcmp(package.task, "mesh_rigging") == 0);
    CHECK_TRUE(strcmp(package.profile, "qwen3-0.6b-fsq") == 0);
    const trellis_model_component_instance * mesh_encoder =
        trellis_model_package_find_component(&package, "mesh_encoder");
    CHECK_TRUE(mesh_encoder != NULL);
    CHECK_TRUE(strcmp(mesh_encoder->architecture, "tokenskin_michelangelo_encoder") == 0);
    CHECK_TRUE(mesh_encoder->execution.compute_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(mesh_encoder->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(mesh_encoder->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(trellis_model_package_find_component(
        &package, "rig_language_model") != NULL);
    CHECK_TRUE(trellis_model_package_find_component(
        &package, "skin_decoder") != NULL);
    trellis_model_package_free(&package);
}

static void test_legacy_fallback(int pixal) {
    char root[PATH_MAX];
    CHECK_TRUE(make_temp_directory(root, pixal ? "trellis_model_pixal" : "trellis_model_legacy"));
    CHECK_TRUE(write_marker_checkpoint(root, pixal));

    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    CHECK_TRUE(package.source == TRELLIS_MODEL_PACKAGE_SOURCE_LEGACY);
    CHECK_TRUE(package.legacy_pixal_marker_probes == 1);
    CHECK_TRUE(strcmp(package.family, pixal ? "pixal3d" : "trellis2") == 0);
    CHECK_TRUE(strcmp(package.profile, pixal ? "1024_cascade" : "512") == 0);
    CHECK_TRUE((trellis_model_package_find_component(
        &package, "shape_encoder") != NULL) == !pixal);
    const trellis_model_component_instance * flow =
        trellis_model_package_find_component(&package, "sparse_structure_flow");
    CHECK_TRUE(flow != NULL);
    CHECK_TRUE(strcmp(flow->architecture, pixal ? "pixal_dit_flow" : "trellis_dit_flow") == 0);
    CHECK_TRUE(flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(flow->execution.flash_kv_dtype == (pixal ? TRELLIS_DTYPE_BF16 : TRELLIS_DTYPE_F16));
    trellis_model_package_free(&package);
    remove_package_tree(root, 0, 1);
}

static void expect_invalid_manifest(const char * manifest) {
    char root[PATH_MAX];
    char path[PATH_MAX];
    CHECK_TRUE(make_temp_directory(root, "trellis_model_invalid"));
    CHECK_TRUE(write_marker_checkpoint(root, 1));
    CHECK_TRUE(snprintf(path, sizeof(path), "%s/model.json", root) > 0);
    CHECK_TRUE(write_text(path, manifest));
    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(package.component_count == 0);
    CHECK_TRUE(package.legacy_pixal_marker_probes == 0);
    trellis_model_package_free(&package);
    remove_package_tree(root, 1, 1);
}

static void test_manifest_validation(void) {
    expect_invalid_manifest(
        "{\"schema_version\":2,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":[]}");
    expect_invalid_manifest(
        "{\"schema_version\":1,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":["
        "{\"role\":\"same\",\"architecture\":\"test_linear\",\"weights\":\"a.bin\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"none\"}},"
        "{\"role\":\"same\",\"architecture\":\"test_linear\",\"weights\":\"b.bin\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"none\"}}]}");
    expect_invalid_manifest(
        "{\"schema_version\":1,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":["
        "{\"role\":\"x\",\"architecture\":\"test_linear\",\"weights\":\"../escape.bin\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"none\"}}]}");
    expect_invalid_manifest(
        "{\"schema_version\":1,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":["
        "{\"role\":\"x\",\"architecture\":\"test_linear\",\"weights\":\"x.bin\","
        "\"execution\":{\"compute_dtype\":\"fp8\",\"attention\":\"none\"}}]}");
    expect_invalid_manifest(
        "{\"schema_version\":1,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":["
        "{\"role\":\"x\",\"architecture\":\"test_linear\",\"weights\":\"x.bin\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"magic\"}}]}");
    expect_invalid_manifest(
        "{\"schema_version\":1,\"id\":\"x\",\"family\":\"test_fixture\","
        "\"task\":\"test_weight_binding\",\"profile\":\"x\",\"components\":["
        "{\"role\":\"x\",\"architecture\":\"test_linear\",\"weights\":\"x.bin\","
        "\"execution\":{\"compute_dtype\":\"f32\",\"attention\":\"none\","
        "\"emulate_bf16_blocks\":\"yes\"}}]}");
}

static void test_safe_weight_paths(void) {
    CHECK_TRUE(trellis_model_weights_path_is_safe("ckpts/model.safetensors"));
    CHECK_TRUE(!trellis_model_weights_path_is_safe("/tmp/model.safetensors"));
    CHECK_TRUE(!trellis_model_weights_path_is_safe("../model.safetensors"));
    CHECK_TRUE(!trellis_model_weights_path_is_safe("ckpts/../model.safetensors"));
    CHECK_TRUE(!trellis_model_weights_path_is_safe("ckpts//model.safetensors"));
    CHECK_TRUE(!trellis_model_weights_path_is_safe("C:\\model.safetensors"));
}

int main(void) {
    test_manifest_load();
    test_repository_templates();
    test_legacy_fallback(0);
    test_legacy_fallback(1);
    test_manifest_validation();
    test_safe_weight_paths();
    if (g_failures != 0) {
        fprintf(stderr, "%d model package test(s) failed\n", g_failures);
        return 1;
    }
    printf("model package tests passed\n");
    return 0;
}
