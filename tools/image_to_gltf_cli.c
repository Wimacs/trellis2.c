#include "image_to_gltf_cli.h"

#include "trellis.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_pixal3d_cli(trellis_image_to_gltf_cli_model model) {
    return model == TRELLIS_IMAGE_TO_GLTF_CLI_PIXAL3D;
}

static const char * cli_model_name(trellis_image_to_gltf_cli_model model) {
    return is_pixal3d_cli(model) ? "Pixal3D" : "TRELLIS.2";
}

static const char * cli_executable_name(trellis_image_to_gltf_cli_model model) {
    return is_pixal3d_cli(model) ?
        "pixal3d-image-to-gltf" : "trellis2-image-to-gltf";
}

static void print_banner(trellis_image_to_gltf_cli_model model) {
    fputs(
        "  _______ ____  _____ _     _     ___ ____  ____    ____ \n"
        " |__   __|  _ \\| ____| |   | |   |_ _/ ___||___ \\  / ___|\n"
        "    | |  | |_) |  _| | |   | |    | |\\___ \\  __) || |    \n"
        "    | |  |  _ <| |___| |___| |___ | | ___) |/ __/ | |___ \n"
        "    |_|  |_| \\_\\_____|_____|_____|___|____/|_____(_)____|\n"
        "\n"
        "                 trellis2.c model CLI\n"
        "\n",
        stdout);
    fprintf(stdout, "                 %s image-to-3D\n\n", cli_model_name(model));
    fflush(stdout);
}

static void usage(
    FILE * out,
    const char * argv0,
    trellis_image_to_gltf_cli_model model) {
    const int pixal3d = is_pixal3d_cli(model);
    fprintf(out,
        "Usage:\n"
        "  %s --model DIR --dino DIR --image FILE [--gltf FILE | --glb FILE | --output FILE] [options]\n"
        "\n"
        "Runs the %s image-to-3D model. This executable rejects model packages from other families.\n"
        "%s\n"
        "\n"
        "Options:\n"
        "  --model DIR             %s model directory containing model.json and ckpts/\n"
        "  --dino DIR              DINOv3 image encoder directory containing model.safetensors\n"
        "  --birefnet FILE         Override auto-discovered BiRefNet GGUF for opaque input\n"
        "  --image FILE            Input image. PNG/JPEG load directly; WebP is converted with ffmpeg first.\n"
        "  --gltf FILE             Output glTF 2.0 path; use .glb to embed PBR textures, default output.glb\n"
        "  --glb FILE              Alias of --gltf\n"
        "  --output FILE           Alias of --gltf\n"
        "  --texture-size N        glTF texture size, default 1024\n"
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
        "  --use-ggml-flash-attn   Force ggml Flash Attention\n"
        "  --no-ggml-flash-attn    Force explicit SDPA\n"
        "  --decode-max-levels N   Debug: run only first N shape decoder levels, default full\n"
        "  --decode-max-input-tokens N Debug: truncate shape decoder input tokens\n"
        "  --verbose               Print debug logs\n",
        argv0,
        cli_model_name(model),
        pixal3d ?
            "Profile: 1024_cascade by default; --pipeline also accepts 1536_cascade." :
            "Profile: 512 by default; --pipeline also accepts 1024.",
        cli_model_name(model));
    if (pixal3d) {
        fputs(
            "\nPixal3D options:\n"
            "  --naf FILE              NAF weights converted to safetensors; default auto-discovery\n"
            "  --pipeline NAME         1024_cascade (default) or 1536_cascade\n"
            "  --fov X                 Horizontal camera FOV in radians, default 0.857556\n"
            "  --camera-distance X     Projection distance; 0 derives it from FOV and mesh scale\n"
            "  --mesh-scale X          Projection-space mesh scale, default 1\n"
            "  --max-num-tokens N      Cascade token budget hint, default 49152\n",
            out);
    } else {
        fputs(
            "\nTRELLIS.2 options:\n"
            "  --pipeline NAME         512 (default) or 1024\n",
            out);
    }
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

int trellis_image_to_gltf_cli_main(
    trellis_image_to_gltf_cli_model model,
    int argc,
    char ** argv) {
    if (model != TRELLIS_IMAGE_TO_GLTF_CLI_TRELLIS2 &&
        model != TRELLIS_IMAGE_TO_GLTF_CLI_PIXAL3D) {
        fprintf(stderr, "invalid image-to-3D CLI model\n");
        return 2;
    }
    const int pixal3d = is_pixal3d_cli(model);
    print_banner(model);

    trellis_image_to_gltf_options options;
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    memset(&options, 0, sizeof(options));
    options.device = 0;
    options.sparse_structure_steps = 12;
    options.structured_latent_steps = 12;
    options.latent_size = 16;
    options.pipeline_type = pixal3d ? "1024_cascade" : "512";
    options.resolution = pixal3d ? 1024 : 512;
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
    /* Zero means the model package/adapter selects its FlashAttention policy. */
    options.use_ggml_flash_attn = 0;
    pixal_options.camera_angle_x = 0.8575560450553894f;
    pixal_options.camera_distance = 0.0f;
    pixal_options.mesh_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            options.dino_dir = arg_value(argc, argv, &i);
        } else if (pixal3d && strcmp(argv[i], "--naf") == 0) {
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
            if (options.pipeline_type == NULL) goto bad_args;
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
        } else if (pixal3d && strcmp(argv[i], "--fov") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &pixal_options.camera_angle_x)) goto bad_args;
        } else if (pixal3d && strcmp(argv[i], "--camera-distance") == 0) {
            if (!parse_float_arg(arg_value(argc, argv, &i), &pixal_options.camera_distance)) goto bad_args;
        } else if (pixal3d && strcmp(argv[i], "--mesh-scale") == 0) {
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
        } else if (pixal3d && strcmp(argv[i], "--max-num-tokens") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.max_num_tokens)) goto bad_args;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_set_verbose(1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0], model);
            return 0;
        } else {
            TRELLIS_ERROR("unknown option: %s", argv[i]);
            goto bad_args;
        }
    }

    if (options.gltf_path == NULL || options.gltf_path[0] == '\0') {
        options.gltf_path = "output.glb";
    }

    if (pixal3d &&
        strcmp(options.pipeline_type, "1024_cascade") != 0 &&
        strcmp(options.pipeline_type, "1536_cascade") != 0) {
        TRELLIS_ERROR(
            "Pixal3D --pipeline must be 1024_cascade or 1536_cascade, got '%s'",
            options.pipeline_type);
        goto bad_args;
    }
    if (!pixal3d &&
        strcmp(options.pipeline_type, "512") != 0 &&
        strcmp(options.pipeline_type, "1024") != 0) {
        TRELLIS_ERROR(
            "TRELLIS.2 --pipeline must be 512 or 1024, got '%s'",
            options.pipeline_type);
        goto bad_args;
    }

    if (pixal3d && pixal_options.camera_distance < 0.0f) {
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
        (pixal3d &&
         (pixal_options.camera_angle_x <= 0.0f ||
          pixal_options.camera_angle_x >= 3.14159265358979323846f ||
          pixal_options.mesh_scale <= 0.0f)) ||
        options.vkmesh_gpu_workspace_budget_mib < 0 ||
        options.model_cache_budget_mib < 0 ||
        (options.resolution != 512 && options.resolution != 1024)) {
        goto bad_args;
    }

    trellis_status status = pixal3d ?
        trellis_pipeline_pixal3d_image_to_gltf(&options, &pixal_options) :
        trellis_pipeline_trellis2_image_to_gltf(&options);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "%s failed: %s",
            cli_executable_name(model),
            trellis_status_string(status));
        return 1;
    }
    return 0;

bad_args:
    usage(stderr, argv[0], model);
    return 2;
}
