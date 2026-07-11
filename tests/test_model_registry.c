#include "trellis_model_package.h"
#include "trellis_registry.h"

#include <stdio.h>
#include <string.h>

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

static void test_descriptor_headers_and_lookup(void) {
    CHECK_TRUE(trellis_registry_family_count() >= 3);
    for (size_t i = 0; i < trellis_registry_family_count(); ++i) {
        const trellis_model_family_descriptor * descriptor = trellis_registry_family_at(i);
        CHECK_TRUE(descriptor != NULL);
        CHECK_TRUE(descriptor->abi_version == TRELLIS_REGISTRY_ABI_VERSION);
        CHECK_TRUE(descriptor->struct_size == sizeof(*descriptor));
        CHECK_TRUE(trellis_registry_find_family(descriptor->id) == descriptor);
    }
    CHECK_TRUE(trellis_registry_architecture_count() >= 8);
    for (size_t i = 0; i < trellis_registry_architecture_count(); ++i) {
        const trellis_architecture_descriptor * descriptor = trellis_registry_architecture_at(i);
        CHECK_TRUE(descriptor != NULL);
        CHECK_TRUE(descriptor->abi_version == TRELLIS_REGISTRY_ABI_VERSION);
        CHECK_TRUE(descriptor->struct_size == sizeof(*descriptor));
        CHECK_TRUE(trellis_registry_find_architecture(descriptor->id) == descriptor);
    }
    CHECK_TRUE(trellis_registry_task_count() >= 2);
    for (size_t i = 0; i < trellis_registry_task_count(); ++i) {
        const trellis_task_descriptor * descriptor = trellis_registry_task_at(i);
        CHECK_TRUE(descriptor != NULL);
        CHECK_TRUE(descriptor->abi_version == TRELLIS_REGISTRY_ABI_VERSION);
        CHECK_TRUE(descriptor->struct_size == sizeof(*descriptor));
        CHECK_TRUE(trellis_registry_find_task(descriptor->id) == descriptor);
    }
    CHECK_TRUE(trellis_registry_family_at(trellis_registry_family_count()) == NULL);
    CHECK_TRUE(trellis_registry_architecture_at(trellis_registry_architecture_count()) == NULL);
    CHECK_TRUE(trellis_registry_task_at(trellis_registry_task_count()) == NULL);
    CHECK_TRUE(trellis_registry_find_family("missing") == NULL);
    CHECK_TRUE(strcmp(
        trellis_registry_find_family("trellis2")->default_profile,
        "512") == 0);
    CHECK_TRUE(strcmp(
        trellis_registry_find_family("pixal3d")->default_profile,
        "1024_cascade") == 0);
}

static void test_non_3d_task_fixture(void) {
    const trellis_task_descriptor * task = trellis_registry_find_task("test_weight_binding");
    CHECK_TRUE(task != NULL);
    CHECK_TRUE(strcmp(task->input_kind, "tensor") == 0);
    CHECK_TRUE(strcmp(task->output_kind, "binding_report") == 0);
    CHECK_TRUE((task->flags & TRELLIS_TASK_FLAG_TEST_FIXTURE) != 0);

    trellis_model_component_instance component = {
        "linear",
        "test_linear",
        "weights/linear.safetensors",
        TRELLIS_EXECUTION_POLICY_INIT,
    };
    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    package.schema_version = 1;
    package.id = "fixture";
    package.family = "test_fixture";
    package.task = "test_weight_binding";
    package.profile = "single_tensor";
    package.components = &component;
    package.component_count = 1;
    CHECK_TRUE(trellis_registry_validate_package(&package) == TRELLIS_STATUS_OK);

    component.architecture = "trellis_dit_flow";
    CHECK_TRUE(trellis_registry_validate_package(&package) == TRELLIS_STATUS_PARSE_ERROR);
    component.architecture = "unknown_architecture";
    CHECK_TRUE(trellis_registry_validate_package(&package) == TRELLIS_STATUS_PARSE_ERROR);
}

int main(void) {
    test_descriptor_headers_and_lookup();
    test_non_3d_task_fixture();
    if (g_failures != 0) {
        fprintf(stderr, "%d model registry test(s) failed\n", g_failures);
        return 1;
    }
    printf("model registry tests passed\n");
    return 0;
}
