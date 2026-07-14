#include "image_to_3d_internal.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void test_unpaired_layout(void) {
    const float dynamic[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float concat[] = {10.0f, 20.0f};
    float output[6] = {0};
    const float expected[] = {1.0f, 2.0f, 10.0f, 3.0f, 4.0f, 20.0f};
    CHECK_TRUE(
        trellis_structured_latent_pack_flow_input(
            dynamic, NULL, concat, 2, 2, 1, output, 6) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
}

static void test_segvigen_dynamic_then_fixed_layout(void) {
    const float dynamic[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float fixed[] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float shape[] = {10.0f, 20.0f};
    float output[12] = {0};
    const float expected[] = {
        1.0f, 2.0f, 10.0f,
        3.0f, 4.0f, 20.0f,
        5.0f, 6.0f, 10.0f,
        7.0f, 8.0f, 20.0f,
    };
    CHECK_TRUE(
        trellis_structured_latent_pack_flow_input(
            dynamic, fixed, shape, 2, 2, 1, output, 12) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
}

static void test_contract_errors(void) {
    const float values[] = {1.0f};
    float output[1] = {0};
    CHECK_TRUE(
        trellis_structured_latent_pack_flow_input(
            NULL, NULL, NULL, 1, 1, 0, output, 1) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(
        trellis_structured_latent_pack_flow_input(
            values, NULL, NULL, 1, 1, 1, output, 1) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(
        trellis_structured_latent_pack_flow_input(
            values, values, NULL, 1, 1, 0, output, 1) ==
        TRELLIS_STATUS_OUT_OF_MEMORY);
}

int main(void) {
    test_unpaired_layout();
    test_segvigen_dynamic_then_fixed_layout();
    test_contract_errors();
    if (failures != 0) {
        fprintf(stderr, "%d structured-latent pairing checks failed\n", failures);
        return 1;
    }
    printf("structured-latent paired-token layout passed\n");
    return 0;
}
