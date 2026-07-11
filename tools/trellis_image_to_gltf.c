#include "trellis.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_banner(void) {
    fputs(
        "  _______ ____  _____ _     _     ___ ____  ____    ____ \n"
        " |__   __|  _ \\| ____| |   | |   |_ _/ ___||___ \\  / ___|\n"
        "    | |  | |_) |  _| | |   | |    | |\\___ \\  __) || |    \n"
        "    | |  |  _ <| |___| |___| |___ | | ___) |/ __/ | |___ \n"
        "    |_|  |_| \\_\\_____|_____|_____|___|____/|_____(_)____|\n"
        "\n"
        "                 trellis2.c image-to-3D pipeline\n"
        "\n",
        stdout);
    fflush(stdout);
}

static void usage(FILE * out, const char * argv0) {
    fprintf(out,
        "Usage:\n"
        "  %s --model DIR --dino DIR --image FILE [--gltf FILE | --glb FILE | --output FILE] [options]\n"
        "\n"
        "Runs image -> sparse structure -> shape SLat -> mesh/subs -> texture SLat -> GLB/glTF textures.\n"
        "\n"
        "Options:\n"
        "  --model DIR             TRELLIS.2 model directory containing ckpts/\n"
        "  --dino DIR              DINOv3 image encoder directory containing model.safetensors\n"
        "  --naf FILE              Pixal3D NAF weights converted to safetensors; not used by TRELLIS.2\n"
        "  --birefnet FILE         BiRefNet GGUF; required for opaque Pixal3D input, optional for TRELLIS.2\n"
        "  --image FILE            Input image. PNG/JPEG load directly; WebP is converted with ffmpeg first.\n"
        "  --gltf FILE             Output glTF 2.0 path; use .glb to embed PBR textures, default output.glb\n"
        "  --glb FILE              Alias of --gltf\n"
        "  --output FILE           Alias of --gltf\n"
        "  --texture-size N        glTF texture size, default 1024\n"
        "  --pipeline NAME         512, 1024, 1024_cascade, or Pixal3D 1536_cascade; default 1024_cascade\n"
        "  --mesh-postprocess      Run vkmesh TRELLIS topology cleanup before GLB/glTF export, default on\n"
        "  --no-mesh-postprocess   Disable topology cleanup for raw/debug exports\n"
        "  --mesh-postprocess-simplify Run vkmesh simplify after remesh/cleanup, default off\n"
        "  --mesh-postprocess-no-simplify Skip vkmesh simplify, default on\n"
        "  --mesh-decimation-target N Postprocess final face target, default 1000000\n"
        "  --mesh-remesh           Run narrow-band remesh during postprocess, default on\n"
        "  --no-mesh-remesh        Disable remesh and use cleanup/simplify only\n"
        "  --mesh-remesh-resolution N Override remesh grid resolution\n"
        "  --mesh-remesh-band X    Remesh narrow-band size in voxels, default 1\n"
        "  --mesh-remesh-project X Project remesh vertices back to source, default 0\n"
        "  --vkmesh-gpu-workspace-budget-mib N vkmesh GPU workspace cap; default auto, max 2048 MiB\n"
        "  --vkmesh FILE           vkmesh executable path; default searches sibling binary then PATH\n"
        "  --no-model-cache        Disable persistent model weight cache\n"
        "  --model-cache-budget-mib N GPU-resident weight cache budget; 0/unset means unlimited\n"
        "  --backend NAME          Full pipeline backend: " TRELLIS_DEFAULT_BACKEND " for this build\n"
        "  --device N              Backend device, default 0\n"
        "  --steps N               Sparse-structure and structured-latent Euler steps, default 12\n"
        "  --sparse-structure-steps N Override sparse-structure steps\n"
        "  --structured-latent-steps N Override shape and texture SLat steps\n"
        "  --seed N                Sparse-structure latent seed, default 1\n"
        "  --noise-seed N          Structured-latent noise seed, default 18\n"
        "  --latent-size N         Sparse-structure latent grid edge, default 16\n"
        "  --resolution N          Legacy shape resolution override, 512 or 1024; --pipeline controls normal runs\n"
        "  --cond-resolution N     DINO input square edge, default 512\n"
        "  --sparse-resolution N   Sparse-structure output edge, default 32\n"
        "  --fov X                 Pixal3D horizontal camera FOV in radians, default 0.857556; updates auto distance\n"
        "  --camera-distance X     Pixal3D projection distance; 0/default derives 1/(2*mesh_scale*tan(fov/2))\n"
        "  --mesh-scale X          Pixal3D projection-space mesh scale, default 1; updates auto distance\n"
        "  --flow PATH             Override shape SLat flow safetensors path\n"
        "  --decoder PATH          Override FlexiDualGridVaeDecoder safetensors path\n"
        "  --rescale-t X           Shape SLat timestep rescale factor, default 3.0\n"
        "  --guidance-strength X   Shape SLat CFG strength, default 7.5\n"
        "  --guidance-rescale X    Shape SLat CFG rescale, default 0.5\n"
        "  --guidance-min X        Shape SLat CFG interval min, default 0.6\n"
        "  --guidance-max X        Shape SLat CFG interval max, default 1.0\n"
        "  --flow-blocks N         Debug: run only first N transformer blocks in both flows\n"
        "  --flow-block-parts N    Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope          Debug: disable sparse RoPE\n"
        "  --emulate-bf16-blocks   Debug: round structured-latent block activations like reference bf16 flow\n"
        "  --use-ggml-flash-attn   Force ggml flash attention (TRELLIS.2 default; Pixal3D debug override)\n"
        "  --no-ggml-flash-attn    Force explicit SDPA\n"
        "  --decode-max-levels N   Debug: run only first N shape decoder levels, default full\n"
        "  --decode-max-input-tokens N Debug: truncate shape decoder input tokens\n"
        "  --max-num-tokens N      Cascade token budget hint, default 49152\n"
        "  --verbose               Print debug logs\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int parse_int_arg(const char * text, int * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    char * end = NULL;
    long v = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return 0;
    }
    errno = 0;
    *out = (int) v;
    return 1;
}

static int parse_i64_arg(const char * text, int64_t * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    char * end = NULL;
    long long v = strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        return 0;
    }
    errno = 0;
    *out = (int64_t) v;
    return 1;
}

static int parse_u32_arg(const char * text, uint32_t * out) {
    if (text == NULL || out == NULL || text[0] == '\0' || text[0] == '-') {
        return 0;
    }
    errno = 0;
    char * end = NULL;
    unsigned long v = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || v > UINT32_MAX) {
        return 0;
    }
    errno = 0;
    *out = (uint32_t) v;
    return 1;
}

static int parse_float_arg(const char * text, float * out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    char * end = NULL;
    float v = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(v)) {
        return 0;
    }
    *out = v;
    return 1;
}

int main(int argc, char ** argv) {
    print_banner();

    trellis_image_to_gltf_options options;
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    memset(&options, 0, sizeof(options));
    options.device = 0;
    options.sparse_structure_steps = 12;
    options.structured_latent_steps = 12;
    options.latent_size = 16;
    options.pipeline_type = "1024_cascade";
    options.resolution = 1024;
    options.cond_resolution = 512;
    options.sparse_resolution = 32;
    options.seed = 1u;
    options.noise_seed = 18u;
    options.texture_size = 1024;
    options.rescale_t = 3.0f;
    options.guidance_strength = 7.5f;
    options.guidance_rescale = 0.5f;
    options.guidance_min = 0.6f;
    options.guidance_max = 1.0f;
    options.flow_blocks_override = -1;
    options.flow_block_parts_override = -1;
    options.mesh_postprocess = 1;
    options.mesh_postprocess_no_simplify = 1;
    options.mesh_postprocess_decimation_target = 1000000;
    options.mesh_remesh = 1;
    options.mesh_remesh_resolution = 0;
    options.mesh_remesh_band = 1.0f;
    options.mesh_remesh_project = 0.0f;
    options.max_num_tokens = 49152;
    options.model_cache = 1;
    options.model_cache_budget_mib = 0;
    /* Zero means automatic: TRELLIS.2 uses FlashAttention while Pixal3D uses
       explicit SDPA until the shared flash path has BF16-safe K/V support. */
    options.use_ggml_flash_attn = 0;
    pixal_options.camera_angle_x = 0.8575560450553894f;
    pixal_options.camera_distance = 0.0f;
    pixal_options.mesh_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            options.dino_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--naf") == 0) {
            pixal_options.naf_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--birefnet") == 0) {
            options.birefnet_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--image") == 0) {
            options.image_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--gltf") == 0 ||
                   strcmp(argv[i], "--glb") == 0 ||
                   strcmp(argv[i], "--output") == 0) {
            options.gltf_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--mesh-postprocess") == 0) {
            options.mesh_postprocess = 1;
        } else if (strcmp(argv[i], "--no-mesh-postprocess") == 0) {
            options.mesh_postprocess = 0;
            options.mesh_postprocess_no_simplify = 0;
        } else if (strcmp(argv[i], "--mesh-postprocess-no-simplify") == 0) {
            options.mesh_postprocess = 1;
            options.mesh_postprocess_no_simplify = 1;
        } else if (strcmp(argv[i], "--mesh-postprocess-simplify") == 0) {
            options.mesh_postprocess = 1;
            options.mesh_postprocess_no_simplify = 0;
        } else if (strcmp(argv[i], "--mesh-decimation-target") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.mesh_postprocess_decimation_target)) goto bad_args;
            options.mesh_postprocess_no_simplify = 0;
        } else if (strcmp(argv[i], "--mesh-remesh") == 0) {
            options.mesh_postprocess = 1;
            options.mesh_remesh = 1;
        } else if (strcmp(argv[i], "--no-mesh-remesh") == 0) {
            options.mesh_remesh = 0;
        } else if (strcmp(argv[i], "--mesh-remesh-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.mesh_remesh_resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--mesh-remesh-band") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.mesh_remesh_band)) goto bad_args;
        } else if (strcmp(argv[i], "--mesh-remesh-project") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.mesh_remesh_project)) goto bad_args;
        } else if (strcmp(argv[i], "--vkmesh-gpu-workspace-budget-mib") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.vkmesh_gpu_workspace_budget_mib)) goto bad_args;
        } else if (strcmp(argv[i], "--vkmesh") == 0) {
            options.vkmesh_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--model-cache") == 0) {
            options.model_cache = 1;
        } else if (strcmp(argv[i], "--no-model-cache") == 0) {
            options.model_cache = 0;
        } else if (strcmp(argv[i], "--model-cache-budget-mib") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.model_cache_budget_mib)) goto bad_args;
        } else if (strcmp(argv[i], "--backend") == 0) {
            options.backend = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            options.pipeline_type = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--ggml-backend") == 0 || strcmp(argv[i], "--sparse-backend") == 0 ||
                   strcmp(argv[i], "--ggml-device") == 0) {
            fprintf(stderr, "%s is no longer supported; use --backend with a binary built for cuda or vulkan\n", argv[i]);
            return 2;
        } else if (strcmp(argv[i], "--flow") == 0) {
            options.flow_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--decoder") == 0) {
            options.decoder_override_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.device)) goto bad_args;
        } else if (strcmp(argv[i], "--steps") == 0) {
            int steps = 0;
            if (!parse_int_arg(arg_value(argc, argv, &i), &steps)) goto bad_args;
            options.sparse_structure_steps = steps;
            options.structured_latent_steps = steps;
        } else if (strcmp(argv[i], "--sparse-structure-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.sparse_structure_steps)) goto bad_args;
        } else if (strcmp(argv[i], "--structured-latent-steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.structured_latent_steps)) goto bad_args;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options.seed)) goto bad_args;
        } else if (strcmp(argv[i], "--noise-seed") == 0) {
            if (!parse_u32_arg(arg_value(argc, argv, &i), &options.noise_seed)) goto bad_args;
        } else if (strcmp(argv[i], "--latent-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.latent_size)) goto bad_args;
        } else if (strcmp(argv[i], "--resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--cond-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.cond_resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--sparse-resolution") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.sparse_resolution)) goto bad_args;
        } else if (strcmp(argv[i], "--fov") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &pixal_options.camera_angle_x)) goto bad_args;
        } else if (strcmp(argv[i], "--camera-distance") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &pixal_options.camera_distance)) goto bad_args;
        } else if (strcmp(argv[i], "--mesh-scale") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &pixal_options.mesh_scale)) goto bad_args;
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.texture_size)) goto bad_args;
        } else if (strcmp(argv[i], "--rescale-t") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.rescale_t)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-strength") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_strength)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-rescale") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_rescale)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-min") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_min)) goto bad_args;
        } else if (strcmp(argv[i], "--guidance-max") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &options.guidance_max)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.flow_blocks_override)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.flow_block_parts_override)) goto bad_args;
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            options.flow_no_rope = 1;
        } else if (strcmp(argv[i], "--emulate-bf16-blocks") == 0) {
            options.emulate_bf16_blocks = 1;
        } else if (strcmp(argv[i], "--use-ggml-flash-attn") == 0) {
            options.use_ggml_flash_attn = 1;
            options.no_ggml_flash_attn = 0;
        } else if (strcmp(argv[i], "--no-ggml-flash-attn") == 0) {
            options.use_ggml_flash_attn = 0;
            options.no_ggml_flash_attn = 1;
        } else if (strcmp(argv[i], "--decode-max-levels") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.decode_max_levels)) goto bad_args;
        } else if (strcmp(argv[i], "--decode-max-input-tokens") == 0) {
            if (!parse_i64_arg(arg_value(argc, argv, &i), &options.decode_max_input_tokens)) goto bad_args;
        } else if (strcmp(argv[i], "--max-num-tokens") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.max_num_tokens)) goto bad_args;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else {
            TRELLIS_ERROR("unknown option: %s", argv[i]);
            goto bad_args;
        }
    }

    if (options.gltf_path == NULL || options.gltf_path[0] == '\0') {
        options.gltf_path = "output.glb";
    }

    if (pixal_options.camera_distance < 0.0f) {
        TRELLIS_ERROR(
            "--camera-distance must be 0 for automatic FOV/mesh-scale fitting or greater than 0 for an explicit distance");
        goto bad_args;
    }

    if (options.model_dir == NULL || options.dino_dir == NULL ||
        options.image_path == NULL ||
        options.sparse_structure_steps <= 0 || options.structured_latent_steps <= 0 ||
        options.latent_size <= 0 || options.cond_resolution <= 0 ||
        options.sparse_resolution <= 0 || options.texture_size <= 0 ||
        options.max_num_tokens <= 0 ||
        options.mesh_postprocess_decimation_target <= 0 ||
        options.mesh_remesh_resolution < 0 ||
        options.mesh_remesh_band <= 0.0f ||
        options.mesh_remesh_project < 0.0f ||
        pixal_options.camera_angle_x <= 0.0f ||
        pixal_options.camera_angle_x >= 3.14159265358979323846f ||
        pixal_options.mesh_scale <= 0.0f ||
        options.vkmesh_gpu_workspace_budget_mib < 0 ||
        options.model_cache_budget_mib < 0 ||
        (options.resolution != 512 && options.resolution != 1024)) {
        goto bad_args;
    }

    trellis_status status = trellis_pipeline_image_to_gltf_ex(&options, &pixal_options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("trellis-image-to-gltf failed: %s", trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(stderr, argv[0]);
    return 2;
}
