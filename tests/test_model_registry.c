#include "trellis_model_package.h"
#include "trellis_registry.h"

#include <stdio.h>
#include <stddef.h>
#include <string.h>

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

/* Keep an independent copy of the original public layout.  The production
 * struct intentionally has no size/version header, so an exact layout check
 * is the regression guard for consumers built against the old C ABI. */
typedef struct legacy_image_to_gltf_options_v1 {
    const char * model_dir;
    const char * dino_dir;
    const char * birefnet_path;
    const char * image_path;
    const char * gltf_path;
    const char * flow_override_path;
    const char * decoder_override_path;
    const char * backend;
    const char * pipeline_type;
    int device;
    int sparse_structure_steps;
    int structured_latent_steps;
    int latent_size;
    int resolution;
    int cond_resolution;
    int sparse_resolution;
    uint32_t seed;
    uint32_t noise_seed;
    float rescale_t;
    float guidance_strength;
    float guidance_rescale;
    float guidance_min;
    float guidance_max;
    int flow_blocks_override;
    int flow_block_parts_override;
    int flow_no_rope;
    int emulate_bf16_blocks;
    int use_ggml_flash_attn;
    int no_ggml_flash_attn;
    int decode_max_levels;
    int64_t decode_max_input_tokens;
    int max_num_tokens;
    int texture_size;
    int mesh_postprocess;
    int mesh_postprocess_no_simplify;
    int mesh_postprocess_decimation_target;
    int mesh_remesh;
    int mesh_remesh_resolution;
    float mesh_remesh_band;
    float mesh_remesh_project;
    int model_cache;
    int model_cache_budget_mib;
    const char * vkmesh_path;
    int vkmesh_gpu_workspace_budget_mib;
} legacy_image_to_gltf_options_v1;

#define CHECK_IMAGE_TO_GLTF_ABI_FIELD(field) \
    CHECK_TRUE(offsetof(trellis_image_to_gltf_options, field) == \
        offsetof(legacy_image_to_gltf_options_v1, field))

static void test_image_to_gltf_public_options_contract(void) {
    CHECK_TRUE(sizeof(trellis_image_to_gltf_options) ==
        sizeof(legacy_image_to_gltf_options_v1));
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(model_dir);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(pipeline_type);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(device);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(seed);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(rescale_t);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(decode_max_input_tokens);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(mesh_postprocess);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(vkmesh_path);
    CHECK_IMAGE_TO_GLTF_ABI_FIELD(vkmesh_gpu_workspace_budget_mib);

    trellis_image_to_gltf_feature_options feature_options =
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_INIT;
    CHECK_TRUE(feature_options.struct_size == sizeof(feature_options));
    CHECK_TRUE(feature_options.version ==
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION);
    CHECK_TRUE(feature_options.struct_size >=
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_V1_SIZE);
    CHECK_TRUE(feature_options.shape_only == 0);
    CHECK_TRUE(feature_options.prepared_image_output_path == NULL);
    CHECK_TRUE(feature_options.shape_latent_output_path == NULL);

    trellis_image_to_gltf_options options = {0};
    options.model_dir = "must-not-be-opened";
    options.dino_dir = "must-not-be-opened";
    options.image_path = "must-not-be-opened.png";
    options.device = 0;

    feature_options.struct_size =
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_V1_SIZE - 1u;
    CHECK_TRUE(trellis_pipeline_trellis2_image_to_gltf_ex(
        &options, &feature_options) == TRELLIS_STATUS_INVALID_ARGUMENT);

    feature_options.struct_size =
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_V1_SIZE;
    feature_options.version =
        TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION + 1u;
    CHECK_TRUE(trellis_pipeline_trellis2_image_to_gltf_ex(
        &options, &feature_options) == TRELLIS_STATUS_INVALID_ARGUMENT);
}

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
    CHECK_TRUE(strcmp(
        trellis_registry_find_family("tokenskin")->default_profile,
        "qwen3-0.6b-fsq") == 0);
    const trellis_task_descriptor * rigging =
        trellis_registry_find_task("mesh_rigging");
    CHECK_TRUE(rigging != NULL);
    CHECK_TRUE(strcmp(rigging->input_kind, "mesh") == 0);
    CHECK_TRUE(strcmp(rigging->output_kind, "rigged_mesh") == 0);
    const trellis_task_descriptor * texturing =
        trellis_registry_find_task("mesh_texturing");
    CHECK_TRUE(texturing != NULL);
    CHECK_TRUE(strcmp(texturing->input_kind, "mesh+image") == 0);
    CHECK_TRUE(strcmp(texturing->output_kind, "textured_mesh") == 0);
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

static void test_tokenskin_public_options_contract(void) {
    trellis_tokenskin_rig_options options = TRELLIS_TOKENSKIN_RIG_OPTIONS_INIT;
    CHECK_TRUE(options.struct_size == sizeof(options));
    CHECK_TRUE(options.sample_count == 54000);
    CHECK_TRUE(options.max_length == 2048);
    CHECK_TRUE(options.num_beams == 10);
    options.model_dir = "must-not-be-opened";
    options.input_path = "must-not-be-opened.glb";
    options.output_path = "must-not-be-written.glb";
    options.struct_size = TRELLIS_TOKENSKIN_RIG_OPTIONS_V1_SIZE - 1u;
    CHECK_TRUE(trellis_pipeline_tokenskin_rig(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.struct_size = TRELLIS_TOKENSKIN_RIG_OPTIONS_V1_SIZE;
    options.num_beams = TRELLIS_TOKENSKIN_MAX_BEAMS + 1;
    CHECK_TRUE(trellis_pipeline_tokenskin_rig(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.num_beams = 1;
    options.sample_count = 2047;
    CHECK_TRUE(trellis_pipeline_tokenskin_rig(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
}

static void test_mesh_texturing_public_options_contract(void) {
    trellis_mesh_texturing_options options = TRELLIS_MESH_TEXTURING_OPTIONS_INIT;
    CHECK_TRUE(options.struct_size == sizeof(options));
    CHECK_TRUE(options.resolution == 512);
    CHECK_TRUE(options.texture_size == 1024);
    CHECK_TRUE(options.steps == 12);
    CHECK_TRUE(options.seed == 42u);
    CHECK_TRUE(options.image_prepared == 0);
    CHECK_TRUE(TRELLIS_MESH_TEXTURING_OPTIONS_V2_SIZE >
        TRELLIS_MESH_TEXTURING_OPTIONS_V1_SIZE);
    CHECK_TRUE(options.struct_size >= TRELLIS_MESH_TEXTURING_OPTIONS_V2_SIZE);
    options.model_dir = "must-not-be-opened";
    options.dino_dir = "must-not-be-opened";
    options.input_path = "must-not-be-opened.glb";
    options.image_path = "must-not-be-opened.png";
    options.output_path = "must-not-be-written.glb";
    options.struct_size = TRELLIS_MESH_TEXTURING_OPTIONS_V1_SIZE - 1u;
    CHECK_TRUE(trellis_pipeline_trellis2_texture_mesh(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.struct_size = TRELLIS_MESH_TEXTURING_OPTIONS_V1_SIZE;
    options.resolution = 768;
    CHECK_TRUE(trellis_pipeline_trellis2_texture_mesh(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.resolution = 512;
    options.texture_size = 63;
    CHECK_TRUE(trellis_pipeline_trellis2_texture_mesh(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.texture_size = 8193;
    CHECK_TRUE(trellis_pipeline_trellis2_texture_mesh(&options) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    test_image_to_gltf_public_options_contract();
    test_descriptor_headers_and_lookup();
    test_non_3d_task_fixture();
    test_tokenskin_public_options_contract();
    test_mesh_texturing_public_options_contract();
    if (g_failures != 0) {
        fprintf(stderr, "%d model registry test(s) failed\n", g_failures);
        return 1;
    }
    printf("model registry tests passed\n");
    return 0;
}
