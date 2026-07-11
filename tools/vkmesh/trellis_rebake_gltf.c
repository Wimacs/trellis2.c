#include "trellis.h"
#include "../../src/tasks/image_to_3d/image_to_3d_internal.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE * out, const char * argv0) {
    fprintf(out,
        "Usage:\n"
        "  %s --mesh processed.meshbin --voxels pbr_voxels.bin --gltf out.glb [options]\n"
        "\n"
        "Re-bakes a GLB from material inputs dumped by TRELLIS_MATERIAL_DUMP_DIR.\n"
        "\n"
        "Options:\n"
        "  --sample projection_source.meshbin Optional source mesh for CuMesh-style texture projection\n"
        "  --texture-size N                Texture size, default 1024\n"
        "  --device N                      Vulkan physical-device index, default 0\n",
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
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    char * end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return 0;
    }
    *out = (int) v;
    return 1;
}

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static trellis_status load_meshbin(const char * path, trellis_mesh_host * mesh_out) {
    if (path == NULL || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "trellis-rebake-gltf: failed to open meshbin %s: %s\n", path, strerror(errno));
        return TRELLIS_STATUS_IO_ERROR;
    }

    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    size_t vertex_count = 0;
    size_t face_count = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) INT64_MAX ||
        n_faces > (uint64_t) INT64_MAX ||
        n_vertices > (uint64_t) SIZE_MAX ||
        n_faces > (uint64_t) SIZE_MAX ||
        !checked_mul_size((size_t) n_vertices, 3u, &vertex_count) ||
        !checked_mul_size((size_t) n_faces, 3u, &face_count) ||
        vertex_count > SIZE_MAX / sizeof(float) ||
        face_count > SIZE_MAX / sizeof(int32_t)) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }
    (void) reserved;

    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (status == TRELLIS_STATUS_OK) {
        mesh.vertices = (float *) malloc(vertex_count * sizeof(float));
        mesh.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
        mesh.n_vertices = (int64_t) n_vertices;
        mesh.n_faces = (int64_t) n_faces;
        if (mesh.vertices == NULL || mesh.faces == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else if (fread(mesh.vertices, sizeof(float), vertex_count, f) != vertex_count ||
                   fread(mesh.faces, sizeof(int32_t), face_count, f) != face_count) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        for (size_t i = 0; i < face_count; ++i) {
            if (mesh.faces[i] < 0 || (uint64_t) mesh.faces[i] >= n_vertices) {
                status = TRELLIS_STATUS_PARSE_ERROR;
                break;
            }
        }
    }
    if (status == TRELLIS_STATUS_OK && (flags & 1u) != 0) {
        size_t uv_count = 0;
        if (!checked_mul_size((size_t) n_vertices, 2u, &uv_count) ||
            uv_count > SIZE_MAX / sizeof(float) ||
            uv_count > (size_t) LONG_MAX / sizeof(float) ||
            fseek(f, (long) (uv_count * sizeof(float)), SEEK_CUR) != 0) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        fprintf(stderr, "trellis-rebake-gltf: failed to read meshbin %s: %s\n", path, trellis_status_string(status));
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static trellis_status read_pbr_voxels(const char * path, trellis_pbr_voxels * voxels_out) {
    if (path == NULL || voxels_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(voxels_out, 0, sizeof(*voxels_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "trellis-rebake-gltf: failed to open PBR voxels %s: %s\n", path, strerror(errno));
        return TRELLIS_STATUS_IO_ERROR;
    }

    char magic[8];
    int64_t n_coords = 0;
    int channels = 0;
    int resolution = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_coords, sizeof(n_coords), 1, f) != 1 ||
        fread(&channels, sizeof(channels), 1, f) != 1 ||
        fread(&resolution, sizeof(resolution), 1, f) != 1 ||
        memcmp(magic, "TRLPBR1", 7) != 0 ||
        n_coords <= 0 || channels <= 0 || channels > 64) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }

    trellis_pbr_voxels voxels;
    memset(&voxels, 0, sizeof(voxels));
    if (status == TRELLIS_STATUS_OK) {
        size_t coords_count = 0;
        size_t attrs_count = 0;
        if (!checked_mul_size((size_t) n_coords, 4u, &coords_count) ||
            !checked_mul_size((size_t) n_coords, (size_t) channels, &attrs_count)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            voxels.coords_bxyz = (int32_t *) malloc(coords_count * sizeof(int32_t));
            voxels.attrs = (float *) malloc(attrs_count * sizeof(float));
            voxels.n_coords = n_coords;
            voxels.channels = channels;
            voxels.resolution = resolution;
            if (voxels.coords_bxyz == NULL || voxels.attrs == NULL) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
            } else if (fread(voxels.coords_bxyz, sizeof(int32_t), coords_count, f) != coords_count ||
                       fread(voxels.attrs, sizeof(float), attrs_count, f) != attrs_count) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
    }
    fclose(f);
    if (status != TRELLIS_STATUS_OK) {
        trellis_pbr_voxels_free(&voxels);
        fprintf(stderr, "trellis-rebake-gltf: failed to read PBR voxels %s: %s\n", path, trellis_status_string(status));
        return status;
    }
    *voxels_out = voxels;
    return TRELLIS_STATUS_OK;
}

int main(int argc, char ** argv) {
    const char * mesh_path = NULL;
    const char * sample_path = NULL;
    const char * voxels_path = NULL;
    const char * gltf_path = NULL;
    int texture_size = 1024;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mesh") == 0) {
            mesh_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--sample") == 0) {
            sample_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--voxels") == 0) {
            voxels_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--gltf") == 0 || strcmp(argv[i], "--glb") == 0) {
            gltf_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            const char * value = arg_value(argc, argv, &i);
            if (!parse_int_arg(value, &texture_size)) {
                fprintf(stderr, "trellis-rebake-gltf: invalid --texture-size\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--device") == 0) {
            const char * value = arg_value(argc, argv, &i);
            if (!parse_int_arg(value, &device) || device < 0) {
                fprintf(stderr, "trellis-rebake-gltf: invalid --device\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "trellis-rebake-gltf: unknown or incomplete argument: %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (mesh_path == NULL || voxels_path == NULL || gltf_path == NULL) {
        usage(stderr, argv[0]);
        return 2;
    }

    trellis_mesh_host mesh;
    trellis_mesh_host sample_mesh;
    trellis_pbr_voxels voxels;
    memset(&mesh, 0, sizeof(mesh));
    memset(&sample_mesh, 0, sizeof(sample_mesh));
    memset(&voxels, 0, sizeof(voxels));

    trellis_status status = load_meshbin(mesh_path, &mesh);
    if (status == TRELLIS_STATUS_OK && sample_path != NULL) {
        status = load_meshbin(sample_path, &sample_mesh);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = read_pbr_voxels(voxels_path, &voxels);
    }
    if (status == TRELLIS_STATUS_OK) {
        const trellis_mesh_host * sample =
            sample_mesh.vertices != NULL && sample_mesh.faces != NULL ? &sample_mesh : NULL;
        status = trellis_pipeline_write_gltf(gltf_path, &mesh, sample, &voxels, texture_size, device);
    }

    trellis_mesh_free(&sample_mesh);
    trellis_mesh_free(&mesh);
    trellis_pbr_voxels_free(&voxels);

    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "trellis-rebake-gltf: failed: %s\n", trellis_status_string(status));
        return 1;
    }
    fprintf(stderr, "trellis-rebake-gltf: wrote %s\n", gltf_path);
    return 0;
}
