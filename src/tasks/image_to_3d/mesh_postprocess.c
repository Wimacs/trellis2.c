#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "trellis.h"
#include "trellis_platform.h"
#include "image_to_3d_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unlink_if_set(const char * path) {
    if (path != NULL && path[0] != '\0') {
        trellis_unlink(path);
    }
}

static int mkdir_p(const char * path) {
    return trellis_mkdir_p(path);
}

static int mkdir_parent(const char * path) {
    return trellis_mkdir_parent(path);
}

static trellis_status trellis_pipeline_write_meshbin(const char * path, const trellis_mesh_host * mesh) {
    if (path == NULL || mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_parent(path)) {
        TRELLIS_ERROR("pipeline mesh: failed to create output directory for %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline mesh: failed to open %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint64_t n_vertices = (uint64_t) mesh->n_vertices;
    const uint64_t n_faces = (uint64_t) mesh->n_faces;
    const uint32_t flags = 0;
    const uint32_t reserved = 0;
    const size_t vertex_count = (size_t) mesh->n_vertices * 3u;
    const size_t face_count = (size_t) mesh->n_faces * 3u;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fwrite(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fwrite(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fwrite(&flags, sizeof(flags), 1, f) != 1 ||
        fwrite(&reserved, sizeof(reserved), 1, f) != 1 ||
        fwrite(mesh->vertices, sizeof(float), vertex_count, f) != vertex_count ||
        fwrite(mesh->faces, sizeof(int32_t), face_count, f) != face_count) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    return status;
}

static trellis_status trellis_pipeline_load_meshbin(const char * path, trellis_mesh_host * mesh_out) {
    if (path == NULL || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline mesh: failed to open meshbin %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }

    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) INT64_MAX ||
        n_faces > (uint64_t) INT64_MAX ||
        n_vertices > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        n_faces > (uint64_t) SIZE_MAX / (3u * sizeof(int32_t))) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }

    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (status == TRELLIS_STATUS_OK) {
        const size_t vertex_count = (size_t) n_vertices * 3u;
        const size_t face_count = (size_t) n_faces * 3u;
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
        if (status == TRELLIS_STATUS_OK && (flags & 1u) != 0) {
            const uint64_t uv_bytes = n_vertices * 2u * (uint64_t) sizeof(float);
            if (uv_bytes > (uint64_t) LONG_MAX || fseek(f, (long) uv_bytes, SEEK_CUR) != 0) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static trellis_status trellis_pipeline_write_pbr_voxels_debug(
    const char * path,
    const trellis_pbr_voxels * voxels) {
    if (path == NULL || voxels == NULL || voxels->coords_bxyz == NULL || voxels->attrs == NULL ||
        voxels->n_coords <= 0 || voxels->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_parent(path)) {
        TRELLIS_ERROR("pipeline: failed to create PBR voxel dump directory for %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline: failed to open PBR voxel dump %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const char magic[8] = { 'T', 'R', 'L', 'P', 'B', 'R', '1', '\0' };
    const size_t coords_count = (size_t) voxels->n_coords * 4u;
    const size_t attrs_count = (size_t) voxels->n_coords * (size_t) voxels->channels;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fwrite(&voxels->n_coords, sizeof(voxels->n_coords), 1, f) != 1 ||
        fwrite(&voxels->channels, sizeof(voxels->channels), 1, f) != 1 ||
        fwrite(&voxels->resolution, sizeof(voxels->resolution), 1, f) != 1 ||
        fwrite(voxels->coords_bxyz, sizeof(int32_t), coords_count, f) != coords_count ||
        fwrite(voxels->attrs, sizeof(float), attrs_count, f) != attrs_count) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline: failed to write PBR voxel dump %s", path);
    }
    return status;
}

static int make_material_dump_path(char * out, size_t out_size, const char * dir, const char * name) {
    int n = snprintf(out, out_size, "%s/%s", dir, name);
    return n >= 0 && (size_t) n < out_size;
}

trellis_status trellis_pipeline_dump_raw_mesh_if_requested(
    const char * dump_dir,
    const trellis_mesh_host * mesh) {
    if (dump_dir == NULL || dump_dir[0] == '\0') {
        return TRELLIS_STATUS_OK;
    }
    char raw_mesh_path[4096];
    if (!make_material_dump_path(
            raw_mesh_path,
            sizeof(raw_mesh_path),
            dump_dir,
            "raw.meshbin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pipeline_write_meshbin(raw_mesh_path, mesh);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    TRELLIS_INFO("pipeline: dumped raw shape mesh to %s", raw_mesh_path);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_dump_material_inputs_if_requested(
    const char * dump_dir,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels) {
    if (dump_dir == NULL || dump_dir[0] == '\0') {
        return TRELLIS_STATUS_OK;
    }
    if (!mkdir_p(dump_dir)) {
        TRELLIS_ERROR("pipeline: failed to create material dump directory %s", dump_dir);
        return TRELLIS_STATUS_IO_ERROR;
    }

    char path[4096];
    if (!make_material_dump_path(path, sizeof(path), dump_dir, "processed.meshbin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pipeline_write_meshbin(path, mesh);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    if (sample_mesh != NULL && sample_mesh->vertices != NULL && sample_mesh->faces != NULL &&
        sample_mesh->n_vertices > 0 && sample_mesh->n_faces > 0) {
        if (!make_material_dump_path(path, sizeof(path), dump_dir, "projection_source.meshbin")) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        status = trellis_pipeline_write_meshbin(path, sample_mesh);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
    }

    if (!make_material_dump_path(path, sizeof(path), dump_dir, "pbr_voxels.bin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    status = trellis_pipeline_write_pbr_voxels_debug(path, voxels);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    if (!make_material_dump_path(path, sizeof(path), dump_dir, "manifest.txt")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    fprintf(f, "processed_vertices=%lld\n", (long long) mesh->n_vertices);
    fprintf(f, "processed_faces=%lld\n", (long long) mesh->n_faces);
    if (sample_mesh != NULL && sample_mesh->vertices != NULL && sample_mesh->faces != NULL) {
        fprintf(f, "projection_vertices=%lld\n", (long long) sample_mesh->n_vertices);
        fprintf(f, "projection_faces=%lld\n", (long long) sample_mesh->n_faces);
    } else {
        fprintf(f, "projection_vertices=0\n");
        fprintf(f, "projection_faces=0\n");
    }
    fprintf(f, "pbr_voxels=%lld\n", (long long) voxels->n_coords);
    fprintf(f, "pbr_channels=%d\n", voxels->channels);
    fprintf(f, "pbr_resolution=%d\n", voxels->resolution);
    if (fclose(f) != 0) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    TRELLIS_INFO("pipeline: dumped material bake inputs to %s", dump_dir);
    return TRELLIS_STATUS_OK;
}

static int run_vkmesh_postprocess_command(
    const char * vkmesh_path,
    const char * input_mesh,
    const char * output_mesh,
    const char * projection_mesh,
    int decimation_target,
    int no_simplify,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project,
    int device,
    int gpu_workspace_budget_mib) {
    char target[64];
    char remesh_resolution_buf[64];
    char remesh_band_buf[64];
    char remesh_project_buf[64];
    char device_buf[64];
    char workspace_budget_buf[64];
    snprintf(target, sizeof(target), "%d", decimation_target > 0 ? decimation_target : 1000000);
    snprintf(remesh_resolution_buf, sizeof(remesh_resolution_buf), "%d", remesh_resolution > 0 ? remesh_resolution : 1024);
    snprintf(remesh_band_buf, sizeof(remesh_band_buf), "%.9g", remesh_band > 0.0f ? remesh_band : 1.0f);
    snprintf(remesh_project_buf, sizeof(remesh_project_buf), "%.9g", remesh_project > 0.0f ? remesh_project : 0.0f);
    snprintf(device_buf, sizeof(device_buf), "%d", device >= 0 ? device : 0);
    snprintf(workspace_budget_buf, sizeof(workspace_budget_buf), "%d", gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0);
    char sibling_vkmesh[PATH_MAX];
    const char * exe = vkmesh_path != NULL && vkmesh_path[0] != '\0' ? vkmesh_path : NULL;
    if (exe == NULL) {
        if (trellis_current_executable_path(sibling_vkmesh, sizeof(sibling_vkmesh))) {
            char * slash = trellis_path_last_sep(sibling_vkmesh);
            if (slash != NULL) {
                slash[1] = '\0';
                size_t dir_len = strlen(sibling_vkmesh);
                const char * vkmesh_name = "vkmesh" TRELLIS_EXE_SUFFIX;
                size_t name_len = strlen(vkmesh_name);
                if (dir_len + name_len + 1u < sizeof(sibling_vkmesh)) {
                    memcpy(sibling_vkmesh + dir_len, vkmesh_name, name_len + 1u);
                    if (trellis_access_executable(sibling_vkmesh)) {
                        exe = sibling_vkmesh;
                    }
                }
            }
        }
    }
    if (exe == NULL) {
        exe = "vkmesh";
    }
    TRELLIS_INFO("mesh postprocess: using vkmesh executable %s", exe);
    char * argv[32];
    int argc = 0;
    argv[argc++] = (char *) exe;
    argv[argc++] = (char *) "--input";
    argv[argc++] = (char *) input_mesh;
    argv[argc++] = (char *) "--output";
    argv[argc++] = (char *) output_mesh;
    if (projection_mesh != NULL && projection_mesh[0] != '\0') {
        argv[argc++] = (char *) "--projection-mesh-output";
        argv[argc++] = (char *) projection_mesh;
    }
    argv[argc++] = (char *) "--postprocess";
    argv[argc++] = (char *) "--device";
    argv[argc++] = device_buf;
    argv[argc++] = (char *) "--gpu-workspace-budget-mib";
    argv[argc++] = workspace_budget_buf;
    if (remesh) {
        argv[argc++] = (char *) "--remesh";
        argv[argc++] = (char *) "--remesh-resolution";
        argv[argc++] = remesh_resolution_buf;
        argv[argc++] = (char *) "--remesh-band";
        argv[argc++] = remesh_band_buf;
        argv[argc++] = (char *) "--remesh-project";
        argv[argc++] = remesh_project_buf;
    } else {
        argv[argc++] = (char *) "--no-remesh";
    }
    argv[argc++] = (char *) "--decimation-target";
    argv[argc++] = target;
    if (no_simplify) {
        argv[argc++] = (char *) "--no-simplify";
    }
    argv[argc++] = (char *) "--no-uv-unwrap";
    argv[argc] = NULL;
    return trellis_path_has_sep(exe) ?
        trellis_run_process_exact(argv) :
        trellis_run_process_search_path(argv);
}

trellis_status trellis_pipeline_postprocess_mesh_with_vkmesh(
    trellis_mesh_host * mesh,
    trellis_mesh_host * projection_mesh_out,
    const char * vkmesh_path,
    int decimation_target,
    int no_simplify,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project,
    int device,
    int gpu_workspace_budget_mib) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (projection_mesh_out != NULL) {
        memset(projection_mesh_out, 0, sizeof(*projection_mesh_out));
    }

#if TRELLIS_HAS_VKMESH_C_API
    (void) vkmesh_path;
    trellis_vkmesh_postprocess_options vkmesh_options;
    memset(&vkmesh_options, 0, sizeof(vkmesh_options));
    vkmesh_options.decimation_target = decimation_target > 0 ? decimation_target : 1000000;
    vkmesh_options.no_simplify = no_simplify ? 1 : 0;
    vkmesh_options.remesh = remesh ? 1 : 0;
    vkmesh_options.remesh_resolution = remesh_resolution > 0 ? remesh_resolution : 1024;
    vkmesh_options.remesh_band = remesh_band > 0.0f ? remesh_band : 1.0f;
    vkmesh_options.remesh_project = remesh_project > 0.0f ? remesh_project : 0.0f;
    vkmesh_options.device = device >= 0 ? device : 0;
    vkmesh_options.gpu_workspace_budget_mib = gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0;

    trellis_mesh_host processed;
    trellis_mesh_host projection;
    memset(&processed, 0, sizeof(processed));
    memset(&projection, 0, sizeof(projection));

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh C API device=%d workspace_budget_mib=%d target=%d no_simplify=%d remesh=%d resolution=%d band=%.9g project=%.9g input_faces=%lld",
        vkmesh_options.device,
        vkmesh_options.gpu_workspace_budget_mib,
        vkmesh_options.decimation_target,
        no_simplify ? 1 : 0,
        vkmesh_options.remesh,
        vkmesh_options.remesh_resolution,
        vkmesh_options.remesh_band,
        vkmesh_options.remesh_project,
        (long long) mesh->n_faces);
    trellis_status status = trellis_vkmesh_postprocess(
        mesh,
        &processed,
        projection_mesh_out != NULL ? &projection : NULL,
        &vkmesh_options);
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&processed);
        trellis_mesh_free(&projection);
        TRELLIS_ERROR("mesh postprocess: vkmesh C API failed: %s", trellis_status_string(status));
        return status;
    }
    if (projection_mesh_out != NULL) {
        *projection_mesh_out = projection;
        memset(&projection, 0, sizeof(projection));
        TRELLIS_INFO(
            "mesh postprocess: loaded projection source vertices=%lld faces=%lld",
            (long long) projection_mesh_out->n_vertices,
            (long long) projection_mesh_out->n_faces);
    }

    int64_t old_vertices = mesh->n_vertices;
    int64_t old_faces = mesh->n_faces;
    trellis_mesh_free(mesh);
    *mesh = processed;
    TRELLIS_INFO(
        "mesh postprocess: done vertices=%lld->%lld faces=%lld->%lld",
        (long long) old_vertices,
        (long long) mesh->n_vertices,
        (long long) old_faces,
        (long long) mesh->n_faces);
    trellis_mesh_free(&projection);
    return TRELLIS_STATUS_OK;
#else
    char input_mesh[4096];
    char output_mesh[4096];
    char projection_mesh[4096];
    if (!trellis_make_temp_path(input_mesh, sizeof(input_mesh), "trellis2_vkmesh_in", ".meshbin") ||
        !trellis_make_temp_path(output_mesh, sizeof(output_mesh), "trellis2_vkmesh_out", ".meshbin") ||
        !trellis_make_temp_path(projection_mesh, sizeof(projection_mesh), "trellis2_vkmesh_projection", ".meshbin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pipeline_write_meshbin(input_mesh, mesh);
    if (status != TRELLIS_STATUS_OK) {
        unlink_if_set(input_mesh);
        unlink_if_set(output_mesh);
        unlink_if_set(projection_mesh);
        return status;
    }

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh device=%d workspace_budget_mib=%d target=%d no_simplify=%d remesh=%d resolution=%d band=%.9g project=%.9g input_faces=%lld",
        device >= 0 ? device : 0,
        gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0,
        decimation_target > 0 ? decimation_target : 1000000,
        no_simplify ? 1 : 0,
        remesh ? 1 : 0,
        remesh_resolution > 0 ? remesh_resolution : 1024,
        remesh_band > 0.0f ? remesh_band : 1.0f,
        remesh_project > 0.0f ? remesh_project : 0.0f,
        (long long) mesh->n_faces);
    if (!run_vkmesh_postprocess_command(
            vkmesh_path,
            input_mesh,
            output_mesh,
            projection_mesh_out != NULL ? projection_mesh : NULL,
            decimation_target,
            no_simplify,
            remesh,
            remesh_resolution,
            remesh_band,
            remesh_project,
            device,
            gpu_workspace_budget_mib)) {
        TRELLIS_ERROR("mesh postprocess: vkmesh failed; pass --vkmesh /path/to/vkmesh if it is not in PATH");
        unlink_if_set(input_mesh);
        unlink_if_set(output_mesh);
        unlink_if_set(projection_mesh);
        return TRELLIS_STATUS_ERROR;
    }

    trellis_mesh_host processed;
    status = trellis_pipeline_load_meshbin(output_mesh, &processed);
    if (status == TRELLIS_STATUS_OK && projection_mesh_out != NULL) {
        status = trellis_pipeline_load_meshbin(projection_mesh, projection_mesh_out);
        if (status == TRELLIS_STATUS_OK) {
            TRELLIS_INFO(
                "mesh postprocess: loaded projection source vertices=%lld faces=%lld",
                (long long) projection_mesh_out->n_vertices,
                (long long) projection_mesh_out->n_faces);
        }
    }
    unlink_if_set(input_mesh);
    unlink_if_set(output_mesh);
    unlink_if_set(projection_mesh);
    if (status != TRELLIS_STATUS_OK) {
        if (projection_mesh_out != NULL) {
            trellis_mesh_free(projection_mesh_out);
        }
        return status;
    }

    int64_t old_vertices = mesh->n_vertices;
    int64_t old_faces = mesh->n_faces;
    trellis_mesh_free(mesh);
    *mesh = processed;
    TRELLIS_INFO(
        "mesh postprocess: done vertices=%lld->%lld faces=%lld->%lld",
        (long long) old_vertices,
        (long long) mesh->n_vertices,
        (long long) old_faces,
        (long long) mesh->n_faces);
    return TRELLIS_STATUS_OK;
#endif
}
