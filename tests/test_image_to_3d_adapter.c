#include "adapter.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef TRELLIS_TEST_SOURCE_DIR
#define TRELLIS_TEST_SOURCE_DIR "."
#endif

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

#define CHECK_CLOSE(actual, expected, tolerance) \
    CHECK_TRUE(fabsf((actual) - (expected)) <= (tolerance))

static void test_descriptor_registry(void) {
    CHECK_TRUE(trellis_image_to_3d_adapter_count() == 2);
    CHECK_TRUE(trellis_image_to_3d_adapter_at(2) == NULL);
    CHECK_TRUE(trellis_image_to_3d_adapter_find(NULL) == NULL);
    CHECK_TRUE(trellis_image_to_3d_adapter_find("unknown") == NULL);

    const trellis_image_to_3d_adapter_descriptor * trellis =
        trellis_image_to_3d_adapter_find("trellis2");
    CHECK_TRUE(trellis != NULL);
    CHECK_TRUE(trellis->abi_version == TRELLIS_IMAGE_TO_3D_ADAPTER_ABI_VERSION);
    CHECK_TRUE(trellis->struct_size >= sizeof(*trellis));
    CHECK_TRUE(strcmp(trellis->id, "trellis2.image_to_3d") == 0);
    CHECK_TRUE(!trellis->uses_projected_conditioning);
    CHECK_TRUE(!trellis->requires_transparent_foreground);
    CHECK_TRUE(trellis->cascade_quantization ==
        TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_FLOOR);
    CHECK_TRUE(trellis->gltf_coordinate_transform ==
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS);

    const trellis_image_to_3d_adapter_descriptor * pixal =
        trellis_image_to_3d_adapter_find("pixal3d");
    CHECK_TRUE(pixal != NULL);
    CHECK_TRUE(strcmp(pixal->id, "pixal3d.image_to_3d") == 0);
    CHECK_TRUE(pixal->uses_projected_conditioning);
    CHECK_TRUE(pixal->requires_transparent_foreground);
    CHECK_TRUE(pixal->cascade_quantization ==
        TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_ROUND);
    CHECK_CLOSE(pixal->default_camera_angle_x, 0.8575560450553894f, 1e-7f);
    CHECK_CLOSE(pixal->default_camera_distance, 1.0937500142266299f, 1e-7f);
    CHECK_CLOSE(pixal->default_mesh_scale, 1.0f, 1e-7f);
    CHECK_TRUE(pixal->gltf_coordinate_transform ==
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D);
}

static void test_trellis_repository_package(void) {
    char root[4096];
    CHECK_TRUE(snprintf(
        root,
        sizeof(root),
        "%s/models/trellis2",
        TRELLIS_TEST_SOURCE_DIR) > 0);

    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    const trellis_image_to_3d_adapter_descriptor * adapter =
        trellis_image_to_3d_adapter_find_for_package(&package);
    CHECK_TRUE(adapter != NULL);
    CHECK_TRUE(strcmp(adapter->family, "trellis2") == 0);
    CHECK_TRUE(trellis_image_to_3d_adapter_validate_package(
        adapter,
        &package) == TRELLIS_STATUS_OK);

    const trellis_image_to_3d_adapter_descriptor * resolved = NULL;
    CHECK_TRUE(trellis_image_to_3d_adapter_resolve_package(
        &package,
        &resolved) == TRELLIS_STATUS_OK);
    CHECK_TRUE(resolved == adapter);

    const trellis_model_component_instance * flow =
        trellis_model_package_find_component(&package, "shape_flow_1024");
    CHECK_TRUE(flow != NULL);
    CHECK_TRUE(flow->execution.compute_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(flow->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.emulate_bf16_blocks == 0);
    trellis_ggml_attention_policy attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    int emulate_bf16_blocks = -1;
    CHECK_TRUE(trellis_image_to_3d_component_execution_policy(
        flow,
        &attention,
        &emulate_bf16_blocks) == TRELLIS_STATUS_OK);
    CHECK_TRUE(attention.mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16);
    CHECK_TRUE(emulate_bf16_blocks == 0);

    const trellis_model_component_instance * low_resolution_flow =
        trellis_model_package_find_component(&package, "shape_flow_512");
    CHECK_TRUE(low_resolution_flow != NULL);
    CHECK_TRUE(low_resolution_flow->execution.flash_kv_dtype == TRELLIS_DTYPE_F16);
    trellis_model_component_instance * mutable_flow =
        (trellis_model_component_instance *) flow;
    mutable_flow->execution.flash_kv_dtype = TRELLIS_DTYPE_F16;
    const trellis_status mismatched_policy_status =
        trellis_image_to_3d_adapter_validate_package(adapter, &package);
    mutable_flow->execution.flash_kv_dtype = TRELLIS_DTYPE_BF16;
    CHECK_TRUE(mismatched_policy_status == TRELLIS_STATUS_PARSE_ERROR);

    /* The family adapter must reject a package missing a base task role. */
    const size_t saved_count = package.component_count;
    package.component_count = saved_count - 1;
    CHECK_TRUE(trellis_image_to_3d_adapter_validate_package(
        adapter,
        &package) == TRELLIS_STATUS_PARSE_ERROR);
    package.component_count = saved_count;

    /* A known family carrying a non-3D task is found by family, then rejected
     * by the strongly typed task validator. */
    char * saved_task = package.task;
    package.task = (char *) "ai_weight_binding";
    CHECK_TRUE(trellis_image_to_3d_adapter_find_for_package(&package) == adapter);
    resolved = adapter;
    CHECK_TRUE(trellis_image_to_3d_adapter_resolve_package(
        &package,
        &resolved) == TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(resolved == NULL);
    package.task = saved_task;

    trellis_model_package_free(&package);
}

static void test_pixal_repository_package(void) {
    char root[4096];
    CHECK_TRUE(snprintf(
        root,
        sizeof(root),
        "%s/models/pixal3d",
        TRELLIS_TEST_SOURCE_DIR) > 0);

    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    CHECK_TRUE(trellis_model_package_load(root, &package) == TRELLIS_STATUS_OK);
    const trellis_image_to_3d_adapter_descriptor * adapter = NULL;
    CHECK_TRUE(trellis_image_to_3d_adapter_resolve_package(
        &package,
        &adapter) == TRELLIS_STATUS_OK);
    CHECK_TRUE(adapter != NULL && strcmp(adapter->family, "pixal3d") == 0);

    const trellis_model_component_instance * flow =
        trellis_model_package_find_component(&package, "shape_flow_1024");
    CHECK_TRUE(flow != NULL);
    CHECK_TRUE(strcmp(flow->architecture, "pixal_dit_flow") == 0);
    CHECK_TRUE(flow->execution.compute_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.attention == TRELLIS_ATTENTION_FLASH);
    CHECK_TRUE(flow->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16);
    CHECK_TRUE(flow->execution.emulate_bf16_blocks == 1);
    trellis_ggml_attention_policy attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    int emulate_bf16_blocks = 0;
    CHECK_TRUE(trellis_image_to_3d_component_execution_policy(
        flow,
        &attention,
        &emulate_bf16_blocks) == TRELLIS_STATUS_OK);
    CHECK_TRUE(attention.mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16);
    CHECK_TRUE(emulate_bf16_blocks == 1);

    const trellis_model_component_instance * naf =
        trellis_model_package_find_component(&package, "naf_encoder");
    CHECK_TRUE(naf != NULL && strcmp(naf->architecture, "pixal_naf") == 0);

    /* naf_encoder is last in the repository manifest: truncating it must make
     * the Pixal adapter invalid. */
    const size_t saved_count = package.component_count;
    package.component_count = saved_count - 1;
    CHECK_TRUE(trellis_image_to_3d_adapter_validate_package(
        adapter,
        &package) == TRELLIS_STATUS_PARSE_ERROR);
    package.component_count = saved_count;

    /* Projected conditioning cannot be paired with the Trellis flow graph. */
    trellis_model_component_instance * sparse_flow =
        (trellis_model_component_instance *)
            trellis_model_package_find_component(&package, "sparse_structure_flow");
    CHECK_TRUE(sparse_flow != NULL);
    char * saved_architecture = sparse_flow->architecture;
    sparse_flow->architecture = (char *) "trellis_dit_flow";
    CHECK_TRUE(trellis_image_to_3d_adapter_validate_package(
        adapter,
        &package) == TRELLIS_STATUS_PARSE_ERROR);
    sparse_flow->architecture = saved_architecture;

    trellis_model_package_free(&package);
}

static void test_fake_non_3d_package(void) {
    trellis_model_component_instance component = {
        (char *) "linear",
        (char *) "test_linear",
        (char *) "weights/linear.safetensors",
        TRELLIS_EXECUTION_POLICY_INIT,
    };
    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    package.id = (char *) "binding-fixture";
    package.family = (char *) "test_fixture";
    package.task = (char *) "test_weight_binding";
    package.profile = (char *) "single_tensor";
    package.components = &component;
    package.component_count = 1;

    CHECK_TRUE(trellis_image_to_3d_adapter_find_for_package(&package) == NULL);
    const trellis_image_to_3d_adapter_descriptor * adapter = NULL;
    CHECK_TRUE(trellis_image_to_3d_adapter_resolve_package(
        &package,
        &adapter) == TRELLIS_STATUS_NOT_FOUND);
    CHECK_TRUE(adapter == NULL);
}

static void test_model_pinned_entrypoints_reject_wrong_family(void) {
    char trellis_root[4096];
    char pixal_root[4096];
    CHECK_TRUE(snprintf(
        trellis_root,
        sizeof(trellis_root),
        "%s/models/trellis2",
        TRELLIS_TEST_SOURCE_DIR) > 0);
    CHECK_TRUE(snprintf(
        pixal_root,
        sizeof(pixal_root),
        "%s/models/pixal3d",
        TRELLIS_TEST_SOURCE_DIR) > 0);

    trellis_image_to_gltf_options options = {0};
    options.dino_dir = "unused-dino";
    options.image_path = "unused.png";
    options.device = 0;

    options.model_dir = pixal_root;
    CHECK_TRUE(
        trellis_pipeline_trellis2_image_to_gltf(&options) ==
        TRELLIS_STATUS_PARSE_ERROR);

    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    options.model_dir = trellis_root;
    CHECK_TRUE(
        trellis_pipeline_pixal3d_image_to_gltf(&options, &pixal_options) ==
        TRELLIS_STATUS_PARSE_ERROR);
}

int main(void) {
    test_descriptor_registry();
    test_trellis_repository_package();
    test_pixal_repository_package();
    test_fake_non_3d_package();
    test_model_pinned_entrypoints_reject_wrong_family();
    if (g_failures != 0) {
        fprintf(stderr, "%d image_to_3d adapter test(s) failed\n", g_failures);
        return 1;
    }
    printf("image_to_3d adapter tests passed\n");
    return 0;
}
