#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_tool_live.h"
#include "raylib.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>
#include <unistd.h>
#endif

typedef struct live_view_state {
    int max_voxels;
    float hold;
    float yaw;
    float pitch;
    float distance;
} live_view_state;

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s [--image input.png] [--model TRELLIS.2-4B]\n"
        "\n"
        "Default C+ggml+CUDA live pipeline: image -> stage1 voxel denoise/decode -> stage2 mesh denoise/decode.\n"
        "\n"
        "Options:\n"
        "  --image FILE             Input image, default assets/example_image/T.png\n"
        "  --model DIR              TRELLIS.2 model directory, default TRELLIS.2-4B\n"
        "  --dino DIR               DINOv3 model directory, default dinov3-vitl16-pretrain-lvd1689m\n"
        "  --device N               CUDA device, default 0\n"
        "  --steps N                Stage1 and stage2 Euler steps, default 12\n"
        "  --stage1-steps N         Override stage1 steps\n"
        "  --stage2-steps N         Override stage2 steps\n"
        "  --cond-resolution N      DINO input square edge, default 512\n"
        "  --stage1-latent-size N   Stage1 latent edge, default 16\n"
        "  --sparse-resolution N    Stage1 output sparse edge, default 32\n"
        "  --resolution N           Stage2 shape resolution/checkpoint, default 512\n"
        "  --seed N                 Deterministic stage1/stage2 noise seed, default 1\n"
        "  --display DISPLAY        X11 display, e.g. :1. Auto-detected when unset\n"
        "  --xauthority PATH        Optional Xauthority file\n"
        "  --width N                Window width, default 1280\n"
        "  --height N               Window height, default 800\n"
        "  --hold SECONDS           Minimum seconds per decoded frame, default 0.6\n"
        "  --max-voxels N           Draw at most N stage1 voxels, default 12000\n"
        "  --max-faces N            Draw at most N stage2 faces; 0 draws all, default 0\n"
        "  --mesh-upload-mode M     gpu_indexed, expanded, or indexed; default gpu_indexed\n"
        "  --mesh-style STYLE       solid, wire, or solid_wire; default solid\n"
        "  --source NAME            Stage2 pred_x0 or x_t, default pred_x0\n"
        "  --stage1-out DIR         Optional stage1 voxel snapshot output directory\n"
        "  --dump-intermediates DIR Optional stage1 tensor dump directory\n"
        "  --flow-blocks N          Debug: run only first N blocks in both flows\n"
        "  --flow-block-parts N     Debug: per-block parts 1=self, 2=self+cross, 3=full\n"
        "  --flow-no-rope           Debug: disable sparse RoPE\n"
        "  --decode-max-levels N    Debug: stage2 decoder max levels\n"
        "  --decode-max-input-tokens N Debug: truncate stage2 decode input tokens\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int path_exists(const char * path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static int env_is_empty(const char * name) {
    const char * value = getenv(name);
    return value == NULL || value[0] == '\0';
}

static void detect_local_x11_display(char * dst, size_t dst_size) {
#ifdef _WIN32
    if (dst != NULL && dst_size > 0) {
        dst[0] = '\0';
    }
#else
    int best = -1;
    DIR * dir = opendir("/tmp/.X11-unix");
    if (dir != NULL) {
        struct dirent * ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] != 'X') {
                continue;
            }
            char * end = NULL;
            long display = strtol(ent->d_name + 1, &end, 10);
            if (end != NULL && *end == '\0' && display >= 0 && display <= 9999) {
                if (best < 0 || display < best) {
                    best = (int) display;
                }
            }
        }
        closedir(dir);
    }
    snprintf(dst, dst_size, ":%d", best >= 0 ? best : 0);
#endif
}

static int detect_xauthority(char * dst, size_t dst_size) {
#ifdef _WIN32
    (void) dst;
    (void) dst_size;
    return 0;
#else
    const char * home = getenv("HOME");
    if (home != NULL && home[0] != '\0') {
        char home_auth[PATH_MAX];
        int n = snprintf(home_auth, sizeof(home_auth), "%s/.Xauthority", home);
        if (n >= 0 && (size_t) n < sizeof(home_auth) && path_exists(home_auth)) {
            snprintf(dst, dst_size, "%s", home_auth);
            return 1;
        }
    }
    char gdm_auth[PATH_MAX];
    int n = snprintf(gdm_auth, sizeof(gdm_auth), "/run/user/%ld/gdm/Xauthority", (long) getuid());
    if (n >= 0 && (size_t) n < sizeof(gdm_auth) && path_exists(gdm_auth)) {
        snprintf(dst, dst_size, "%s", gdm_auth);
        return 1;
    }
    return 0;
#endif
}

static void configure_display_env(const char * display, const char * xauthority) {
#ifdef _WIN32
    (void) display;
    (void) xauthority;
#else
    if (display != NULL && display[0] != '\0') {
        trellis_setenv("DISPLAY", display, 1);
    } else if (env_is_empty("DISPLAY")) {
        char detected[32];
        detect_local_x11_display(detected, sizeof(detected));
        trellis_setenv("DISPLAY", detected, 1);
    }
    if (xauthority != NULL && xauthority[0] != '\0') {
        trellis_setenv("XAUTHORITY", xauthority, 1);
    } else if (env_is_empty("XAUTHORITY")) {
        char detected[PATH_MAX];
        if (detect_xauthority(detected, sizeof(detected))) {
            trellis_setenv("XAUTHORITY", detected, 1);
        }
    }
#endif
}

static int sampled_index(int i, int count, int max_count) {
    if (max_count <= 0 || count <= max_count) {
        return i;
    }
    if (max_count <= 1) {
        return 0;
    }
    return (int) llround((double) i * (double) (count - 1) / (double) (max_count - 1));
}

static void update_camera(float * yaw, float * pitch, float * distance) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.008f;
        *pitch -= delta.y * 0.008f;
        if (*pitch < -1.35f) *pitch = -1.35f;
        if (*pitch > 1.35f) *pitch = 1.35f;
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        *distance *= powf(0.9f, wheel);
        if (*distance < 1.2f) *distance = 1.2f;
        if (*distance > 8.0f) *distance = 8.0f;
    }
}

static void draw_voxels_xyz(
    const int32_t * coords,
    int64_t n_coords,
    int resolution,
    int max_voxels) {
    if (coords == NULL || n_coords <= 0 || resolution <= 0) {
        return;
    }
    int draw_count = n_coords > INT_MAX ? INT_MAX : (int) n_coords;
    if (max_voxels > 0 && draw_count > max_voxels) {
        draw_count = max_voxels;
    }
    const float res = (float) resolution;
    const float cube = 2.0f / res;
    Vector3 size = {cube, cube, cube};
    Color color = {73, 169, 255, 225};
    Color wire = {18, 42, 62, 95};
    for (int i = 0; i < draw_count; ++i) {
        int idx = sampled_index(i, (int) n_coords, max_voxels);
        const int32_t * c = coords + (size_t) idx * 3u;
        Vector3 pos = {
            (((float) c[0] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[2] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[1] + 0.5f) / res - 0.5f) * 2.0f,
        };
        DrawCubeV(pos, size, color);
        DrawCubeWiresV(pos, size, wire);
    }
}

static void draw_voxel_volume_guide(void) {
    DrawCubeWiresV((Vector3) {0.0f, 0.0f, 0.0f}, (Vector3) {2.0f, 2.0f, 2.0f}, (Color) {120, 132, 148, 145});
    DrawLine3D((Vector3) {-1.05f, -1.05f, -1.05f}, (Vector3) {1.05f, -1.05f, -1.05f}, (Color) {220, 88, 88, 180});
    DrawLine3D((Vector3) {-1.05f, -1.05f, -1.05f}, (Vector3) {-1.05f, 1.05f, -1.05f}, (Color) {94, 196, 112, 180});
    DrawLine3D((Vector3) {-1.05f, -1.05f, -1.05f}, (Vector3) {-1.05f, -1.05f, 1.05f}, (Color) {84, 148, 238, 180});
}

static void draw_voxel_scene(
    live_view_state * view,
    const trellis_tool_stage1_frame * frame,
    const char * detail) {
    update_camera(&view->yaw, &view->pitch, &view->distance);
    float radius_xz = cosf(view->pitch) * view->distance;
    Camera3D camera = {
        .position = {
            sinf(view->yaw) * radius_xz,
            sinf(view->pitch) * view->distance,
            cosf(view->yaw) * radius_xz,
        },
        .target = {0.0f, 0.0f, 0.0f},
        .up = {0.0f, 1.0f, 0.0f},
        .fovy = 45.0f,
        .projection = CAMERA_PERSPECTIVE,
    };

    BeginDrawing();
    ClearBackground((Color) {18, 20, 24, 255});
    BeginMode3D(camera);
    draw_voxel_volume_guide();
    if (frame != NULL) {
        draw_voxels_xyz(frame->coords_xyz, frame->n_coords, frame->resolution, view->max_voxels);
    }
    EndMode3D();
    DrawRectangle(0, 0, GetScreenWidth(), 92, (Color) {0, 0, 0, 175});
    DrawText("stage1 sparse structure", 18, 16, 22, RAYWHITE);
    DrawText(detail == NULL ? "" : detail, 18, 50, 16, (Color) {195, 204, 216, 255});
    DrawText("drag rotate; wheel zoom; Esc close", 18, GetScreenHeight() - 30, 16, (Color) {180, 188, 200, 255});
    DrawFPS(GetScreenWidth() - 96, 14);
    EndDrawing();
}

static int hold_voxel_frame(
    live_view_state * view,
    const trellis_tool_stage1_frame * frame,
    const char * detail,
    float hold) {
    double until = GetTime() + (double) hold;
    do {
        if (WindowShouldClose()) {
            return 0;
        }
        draw_voxel_scene(view, frame, detail);
    } while (GetTime() < until);
    return 1;
}

static int stage1_frame_callback(const trellis_tool_stage1_frame * frame, void * user_data) {
    live_view_state * view = (live_view_state *) user_data;
    char detail[256];
    snprintf(detail, sizeof(detail), "step %d/%d: %lld active voxels at %d^3; t=%.6g -> %.6g",
        frame->step,
        frame->steps,
        (long long) frame->n_coords,
        frame->resolution,
        frame->t,
        frame->t_prev);
    printf("live: stage1 step %d/%d voxels=%lld resolution=%d\n",
        frame->step,
        frame->steps,
        (long long) frame->n_coords,
        frame->resolution);
    return hold_voxel_frame(view, frame, detail, view->hold);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char * image_path = "assets/example_image/T.png";
    const char * model_dir = "TRELLIS.2-4B";
    const char * dino_dir = "dinov3-vitl16-pretrain-lvd1689m";
    const char * display = NULL;
    const char * xauthority = NULL;
    const char * stage1_out_dir = NULL;
    const char * dump_dir = NULL;
    const char * source = "pred_x0";
    const char * mesh_upload_mode = "gpu_indexed";
    const char * mesh_style = "solid";
    int device = 0;
    int width = 1280;
    int height = 800;
    int steps = 12;
    int stage1_steps = 12;
    int stage2_steps = 12;
    int cond_resolution = 512;
    int stage1_latent_size = 16;
    int sparse_resolution = 32;
    int stage2_resolution = 512;
    int max_voxels = 12000;
    int max_faces = 0;
    int flow_blocks_override = -1;
    int flow_block_parts_override = -1;
    int flow_no_rope = 0;
    int decode_max_levels = 0;
    int64_t decode_max_input_tokens = 0;
    uint32_t seed = 1u;
    float hold = 0.6f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--image") == 0) {
            image_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--model") == 0) {
            model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            dino_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            const char * v = arg_value(argc, argv, &i);
            device = v == NULL ? device : atoi(v);
        } else if (strcmp(argv[i], "--steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            steps = v == NULL ? steps : atoi(v);
            stage1_steps = steps;
            stage2_steps = steps;
        } else if (strcmp(argv[i], "--stage1-steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            stage1_steps = v == NULL ? stage1_steps : atoi(v);
        } else if (strcmp(argv[i], "--stage2-steps") == 0) {
            const char * v = arg_value(argc, argv, &i);
            stage2_steps = v == NULL ? stage2_steps : atoi(v);
        } else if (strcmp(argv[i], "--cond-resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            cond_resolution = v == NULL ? cond_resolution : atoi(v);
        } else if (strcmp(argv[i], "--stage1-latent-size") == 0) {
            const char * v = arg_value(argc, argv, &i);
            stage1_latent_size = v == NULL ? stage1_latent_size : atoi(v);
        } else if (strcmp(argv[i], "--sparse-resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            sparse_resolution = v == NULL ? sparse_resolution : atoi(v);
        } else if (strcmp(argv[i], "--resolution") == 0) {
            const char * v = arg_value(argc, argv, &i);
            stage2_resolution = v == NULL ? stage2_resolution : atoi(v);
        } else if (strcmp(argv[i], "--seed") == 0) {
            const char * v = arg_value(argc, argv, &i);
            seed = v == NULL ? seed : (uint32_t) strtoul(v, NULL, 10);
        } else if (strcmp(argv[i], "--display") == 0) {
            display = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--xauthority") == 0) {
            xauthority = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--width") == 0) {
            const char * v = arg_value(argc, argv, &i);
            width = v == NULL ? width : atoi(v);
        } else if (strcmp(argv[i], "--height") == 0) {
            const char * v = arg_value(argc, argv, &i);
            height = v == NULL ? height : atoi(v);
        } else if (strcmp(argv[i], "--hold") == 0) {
            const char * v = arg_value(argc, argv, &i);
            hold = v == NULL ? hold : (float) atof(v);
        } else if (strcmp(argv[i], "--max-voxels") == 0) {
            const char * v = arg_value(argc, argv, &i);
            max_voxels = v == NULL ? max_voxels : atoi(v);
        } else if (strcmp(argv[i], "--max-faces") == 0) {
            const char * v = arg_value(argc, argv, &i);
            max_faces = v == NULL ? max_faces : atoi(v);
        } else if (strcmp(argv[i], "--mesh-upload-mode") == 0) {
            mesh_upload_mode = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--mesh-style") == 0) {
            mesh_style = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--source") == 0) {
            source = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--stage1-out") == 0) {
            stage1_out_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dump-intermediates") == 0) {
            dump_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--flow-blocks") == 0) {
            const char * v = arg_value(argc, argv, &i);
            flow_blocks_override = v == NULL ? flow_blocks_override : atoi(v);
        } else if (strcmp(argv[i], "--flow-block-parts") == 0) {
            const char * v = arg_value(argc, argv, &i);
            flow_block_parts_override = v == NULL ? flow_block_parts_override : atoi(v);
        } else if (strcmp(argv[i], "--flow-no-rope") == 0) {
            flow_no_rope = 1;
        } else if (strcmp(argv[i], "--decode-max-levels") == 0) {
            const char * v = arg_value(argc, argv, &i);
            decode_max_levels = v == NULL ? decode_max_levels : atoi(v);
        } else if (strcmp(argv[i], "--decode-max-input-tokens") == 0) {
            const char * v = arg_value(argc, argv, &i);
            decode_max_input_tokens = v == NULL ? decode_max_input_tokens : atoll(v);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (image_path == NULL || model_dir == NULL || dino_dir == NULL ||
        stage1_steps <= 0 || stage2_steps <= 0 || width <= 0 || height <= 0 ||
        (strcmp(source, "pred_x0") != 0 && strcmp(source, "x_t") != 0)) {
        usage(argv[0]);
        return 2;
    }

    trellis_cuda_context cuda;
    memset(&cuda, 0, sizeof(cuda));
    fprintf(stderr, "live: initializing CUDA and preloading all safetensors before opening raylib\n");
    trellis_status status = trellis_cuda_init(&cuda, device);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "cuda: %s\n", trellis_status_string(status));
        return 1;
    }

    trellis_tool_stage1_weights stage1_weights;
    trellis_tool_stage2_weights stage2_weights;
    memset(&stage1_weights, 0, sizeof(stage1_weights));
    memset(&stage2_weights, 0, sizeof(stage2_weights));
    if (!trellis_tool_stage1_weights_load(&cuda, model_dir, dino_dir, &stage1_weights) ||
        !trellis_tool_stage2_weights_load(&cuda, model_dir, stage2_resolution, NULL, NULL, &stage2_weights)) {
        trellis_tool_stage2_weights_free(&stage2_weights);
        trellis_tool_stage1_weights_free(&stage1_weights);
        trellis_cuda_free(&cuda);
        return 1;
    }
    fprintf(stderr, "live: weights resident on CUDA; opening raylib window\n");

    configure_display_env(display, xauthority);
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "TRELLIS.2 live image-to-3D");
    SetTargetFPS(60);

    live_view_state view;
    memset(&view, 0, sizeof(view));
    view.max_voxels = max_voxels;
    view.hold = hold;
    view.yaw = 0.75f;
    view.pitch = 0.35f;
    view.distance = 3.2f;
    draw_voxel_scene(&view, NULL, "weights resident; preparing image condition");

    trellis_tool_stage1_image_options stage1;
    memset(&stage1, 0, sizeof(stage1));
    stage1.model_dir = model_dir;
    stage1.dino_dir = dino_dir;
    stage1.image_path = image_path;
    stage1.out_dir = stage1_out_dir;
    stage1.dump_dir = dump_dir;
    stage1.latent_size = stage1_latent_size;
    stage1.steps = stage1_steps;
    stage1.cond_resolution = cond_resolution;
    stage1.sparse_resolution = sparse_resolution;
    stage1.seed = seed;
    stage1.flow_blocks_override = flow_blocks_override;
    stage1.flow_block_parts_override = flow_block_parts_override;
    stage1.flow_no_rope = flow_no_rope;
    stage1.voxel_threshold = 0.0f;
    stage1.weights = &stage1_weights;

    trellis_tool_stage1_result stage1_result;
    memset(&stage1_result, 0, sizeof(stage1_result));
    int rc = trellis_tool_run_stage1_image(&cuda, &stage1, stage1_frame_callback, &view, &stage1_result);
    if (rc != 0 || WindowShouldClose()) {
        trellis_tool_stage1_result_free(&stage1_result);
        CloseWindow();
        trellis_tool_stage2_weights_free(&stage2_weights);
        trellis_tool_stage1_weights_free(&stage1_weights);
        trellis_cuda_free(&cuda);
        return rc == 0 ? 0 : rc;
    }
    if (stage1_result.n_coords <= 0) {
        fprintf(stderr, "live: stage1 produced no final voxels; cannot continue to stage2\n");
        trellis_tool_stage1_result_free(&stage1_result);
        CloseWindow();
        trellis_tool_stage2_weights_free(&stage2_weights);
        trellis_tool_stage1_weights_free(&stage1_weights);
        trellis_cuda_free(&cuda);
        return 1;
    }

    char detail[256];
    snprintf(detail, sizeof(detail), "stage1 done: %lld coords; loading stage2 flow and mesh decoder",
        (long long) stage1_result.n_coords);
    draw_voxel_scene(&view, NULL, detail);

    trellis_tool_stage2_live_options stage2;
    memset(&stage2, 0, sizeof(stage2));
    stage2.model_dir = model_dir;
    stage2.coords_bxyz = stage1_result.coords_bxyz;
    stage2.n_coords = stage1_result.n_coords;
    stage2.cond = stage1_result.cond;
    stage2.cond_tokens = stage1_result.cond_tokens;
    stage2.noise_seed = seed + 17u;
    stage2.source = source;
    stage2.device = device;
    stage2.resolution = stage2_resolution;
    stage2.steps = stage2_steps;
    stage2.rescale_t = 3.0f;
    stage2.guidance_strength = 7.5f;
    stage2.guidance_rescale = 0.5f;
    stage2.guidance_min = 0.6f;
    stage2.guidance_max = 1.0f;
    stage2.decode_initial = 1;
    stage2.decode_final = 1;
    stage2.width = width;
    stage2.height = height;
    stage2.max_faces = max_faces;
    stage2.mesh_chunk_faces = 21000;
    stage2.mesh_upload_mode = mesh_upload_mode;
    stage2.mesh_style = mesh_style;
    stage2.hold = hold;
    stage2.flow_blocks_override = flow_blocks_override;
    stage2.flow_block_parts_override = flow_block_parts_override;
    stage2.flow_no_rope = flow_no_rope;
    stage2.decode_max_levels = decode_max_levels;
    stage2.decode_max_input_tokens = decode_max_input_tokens;
    stage2.use_existing_window = 1;
    stage2.cuda = &cuda;
    stage2.weights = &stage2_weights;

    rc = trellis_tool_run_stage2_mesh_live(&stage2);
    trellis_tool_stage1_result_free(&stage1_result);
    if (!WindowShouldClose()) {
        CloseWindow();
    }
    trellis_tool_stage2_weights_free(&stage2_weights);
    trellis_tool_stage1_weights_free(&stage1_weights);
    trellis_cuda_free(&cuda);
    return rc;
}
