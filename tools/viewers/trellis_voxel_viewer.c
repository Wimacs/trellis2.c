#define _POSIX_C_SOURCE 200809L

#include "trellis_platform.h"
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

typedef struct voxel_frame {
    int step;
    int resolution;
    int count;
    char source[32];
    char file[512];
    int32_t * coords;
} voxel_frame;

typedef struct frame_list {
    voxel_frame * frames;
    int count;
    int capacity;
} frame_list;

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s --snapshot-dir DIR [--source x_t|pred_x0|all]\n"
        "Options:\n"
        "  --snapshot-dir DIR   Directory containing frames.tsv and coords_*.i32\n"
        "  --source NAME        Frame source to display, default x_t\n"
        "  --display DISPLAY    X11 display, e.g. :1. Auto-detected when unset\n"
        "  --xauthority PATH    Optional Xauthority file for the desktop session\n"
        "  --width N            Window width, default 1280\n"
        "  --height N           Window height, default 800\n"
        "  --max-voxels N       Draw at most N voxels per frame, default 12000\n"
        "  --hold SECONDS       Seconds per frame while playing, default 0.35\n"
        "  --dry-run            Load frames and print a summary without opening a window\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int append_frame(frame_list * list, const voxel_frame * frame) {
    if (list->count == list->capacity) {
        int next_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        voxel_frame * next = (voxel_frame *) realloc(list->frames, (size_t) next_capacity * sizeof(voxel_frame));
        if (next == NULL) {
            return 0;
        }
        list->frames = next;
        list->capacity = next_capacity;
    }
    list->frames[list->count++] = *frame;
    return 1;
}

static void free_frames(frame_list * list) {
    if (list == NULL) {
        return;
    }
    for (int i = 0; i < list->count; ++i) {
        free(list->frames[i].coords);
    }
    free(list->frames);
    list->frames = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int join_path(char * dst, size_t dst_size, const char * dir, const char * file) {
    int n = snprintf(dst, dst_size, "%s/%s", dir, file);
    return n >= 0 && (size_t) n < dst_size;
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

static int load_coords(const char * path, int count, int32_t ** out_coords) {
    *out_coords = NULL;
    if (count < 0) {
        return 0;
    }
    size_t n_values = (size_t) count * 3u;
    int32_t * coords = (int32_t *) malloc(n_values * sizeof(int32_t));
    if (coords == NULL && n_values != 0) {
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        free(coords);
        return 0;
    }
    size_t got = fread(coords, sizeof(int32_t), n_values, f);
    fclose(f);
    if (got != n_values) {
        free(coords);
        return 0;
    }
    *out_coords = coords;
    return 1;
}

static int parse_frames_tsv(const char * snapshot_dir, const char * source_filter, frame_list * out) {
    char path[1024];
    if (!join_path(path, sizeof(path), snapshot_dir, "frames.tsv")) {
        return 0;
    }
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[2048];
    int line_no = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        if (line_no == 1 && strncmp(line, "step\t", 5) == 0) {
            continue;
        }
        char source[32] = {0};
        char file[512] = {0};
        int step = 0;
        int resolution = 0;
        int count = 0;
        if (sscanf(line, "%d\t%31[^\t]\t%d\t%d\t%511[^\n]", &step, source, &resolution, &count, file) != 5) {
            continue;
        }
        if (strcmp(source_filter, "all") != 0 && strcmp(source_filter, source) != 0) {
            continue;
        }

        voxel_frame frame;
        memset(&frame, 0, sizeof(frame));
        frame.step = step;
        frame.resolution = resolution;
        frame.count = count;
        snprintf(frame.source, sizeof(frame.source), "%s", source);
        snprintf(frame.file, sizeof(frame.file), "%s", file);

        char coords_path[1024];
        if (!join_path(coords_path, sizeof(coords_path), snapshot_dir, file) ||
            !load_coords(coords_path, count, &frame.coords) ||
            !append_frame(out, &frame)) {
            free(frame.coords);
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return out->count > 0;
}

static int sampled_index(int i, int count, int max_voxels) {
    if (max_voxels <= 0 || count <= max_voxels) {
        return i;
    }
    if (max_voxels <= 1) {
        return 0;
    }
    return (int) llround((double) i * (double) (count - 1) / (double) (max_voxels - 1));
}

static void draw_voxels(const voxel_frame * frame, int max_voxels) {
    if (frame == NULL || frame->coords == NULL || frame->resolution <= 0) {
        return;
    }
    int draw_count = frame->count;
    if (max_voxels > 0 && draw_count > max_voxels) {
        draw_count = max_voxels;
    }
    float res = (float) frame->resolution;
    float cube = 2.0f / res;
    Vector3 size = {cube, cube, cube};
    Color color = {73, 169, 255, 225};
    Color wire = {18, 42, 62, 95};
    for (int i = 0; i < draw_count; ++i) {
        int idx = sampled_index(i, frame->count, max_voxels);
        const int32_t * c = &frame->coords[idx * 3];
        Vector3 pos = {
            (((float) c[0] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[2] + 0.5f) / res - 0.5f) * 2.0f,
            (((float) c[1] + 0.5f) / res - 0.5f) * 2.0f,
        };
        DrawCubeV(pos, size, color);
        DrawCubeWiresV(pos, size, wire);
    }
}

int main(int argc, char ** argv) {
    const char * snapshot_dir = NULL;
    const char * source = "x_t";
    const char * display = NULL;
    const char * xauthority = NULL;
    int width = 1280;
    int height = 800;
    int max_voxels = 12000;
    float hold = 0.35f;
    int dry_run = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--snapshot-dir") == 0) {
            snapshot_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--source") == 0) {
            source = arg_value(argc, argv, &i);
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
        } else if (strcmp(argv[i], "--max-voxels") == 0) {
            const char * v = arg_value(argc, argv, &i);
            max_voxels = v == NULL ? max_voxels : atoi(v);
        } else if (strcmp(argv[i], "--hold") == 0) {
            const char * v = arg_value(argc, argv, &i);
            hold = v == NULL ? hold : (float) atof(v);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (snapshot_dir == NULL || source == NULL) {
        usage(argv[0]);
        return 2;
    }

    frame_list frames;
    memset(&frames, 0, sizeof(frames));
    if (!parse_frames_tsv(snapshot_dir, source, &frames)) {
        fprintf(stderr, "no frames loaded from %s\n", snapshot_dir);
        free_frames(&frames);
        return 1;
    }

    if (dry_run) {
        int64_t total_voxels = 0;
        int max_count = 0;
        for (int i = 0; i < frames.count; ++i) {
            total_voxels += frames.frames[i].count;
            if (frames.frames[i].count > max_count) {
                max_count = frames.frames[i].count;
            }
        }
        printf("loaded %d frames from %s (source=%s, total_voxels=%lld, max_frame_voxels=%d)\n",
            frames.count, snapshot_dir, source, (long long) total_voxels, max_count);
        free_frames(&frames);
        return 0;
    }

    configure_display_env(display, xauthority);
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "TRELLIS.2 stage1 voxel viewer");
    SetTargetFPS(60);

    int current = 0;
    int paused = 0;
    double next_time = GetTime() + hold;
    float yaw = 0.75f;
    float pitch = 0.35f;
    float distance = 3.2f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            current = (current + 1) % frames.count;
            next_time = GetTime() + hold;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            current = (current + frames.count - 1) % frames.count;
            next_time = GetTime() + hold;
        }
        if (IsKeyPressed(KEY_R)) {
            current = 0;
            next_time = GetTime() + hold;
        }
        if (!paused && GetTime() >= next_time) {
            current = (current + 1) % frames.count;
            next_time = GetTime() + hold;
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 delta = GetMouseDelta();
            yaw -= delta.x * 0.008f;
            pitch -= delta.y * 0.008f;
            if (pitch < -1.35f) pitch = -1.35f;
            if (pitch > 1.35f) pitch = 1.35f;
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            distance *= powf(0.9f, wheel);
            if (distance < 1.2f) distance = 1.2f;
            if (distance > 8.0f) distance = 8.0f;
        }

        float radius_xz = cosf(pitch) * distance;
        Camera3D camera = {
            .position = {
                sinf(yaw) * radius_xz,
                sinf(pitch) * distance,
                cosf(yaw) * radius_xz,
            },
            .target = {0.0f, 0.0f, 0.0f},
            .up = {0.0f, 1.0f, 0.0f},
            .fovy = 45.0f,
            .projection = CAMERA_PERSPECTIVE,
        };

        voxel_frame * frame = &frames.frames[current];
        BeginDrawing();
        ClearBackground((Color) {18, 20, 24, 255});
        BeginMode3D(camera);
        draw_voxels(frame, max_voxels);
        EndMode3D();

        DrawRectangle(0, 0, GetScreenWidth(), 82, (Color) {0, 0, 0, 170});
        char title[256];
        snprintf(title, sizeof(title), "sparse structure %s step %d", frame->source, frame->step);
        DrawText(title, 18, 16, 22, RAYWHITE);
        char detail[256];
        snprintf(detail, sizeof(detail), "%d active voxels at %d^3; frame %d/%d%s",
            frame->count, frame->resolution, current + 1, frames.count, paused ? "; paused" : "");
        DrawText(detail, 18, 48, 16, (Color) {195, 204, 216, 255});
        DrawText("drag rotate; wheel zoom; space pause; left/right step; R reset", 18, GetScreenHeight() - 30, 16, (Color) {180, 188, 200, 255});
        DrawFPS(GetScreenWidth() - 96, 14);
        EndDrawing();
    }

    CloseWindow();
    free_frames(&frames);
    return 0;
}
