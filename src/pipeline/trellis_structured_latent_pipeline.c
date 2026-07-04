#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_dit_flow_executor.h"
#include "trellis_pipeline_internal.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int file_size_bytes(const char * path, size_t * size_out) {
    if (path == NULL || size_out == NULL) {
        return 0;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long n = ftell(f);
    fclose(f);
    if (n < 0) {
        return 0;
    }
    *size_out = (size_t) n;
    return 1;
}

static char * read_text_file_null_terminated(const char * path) {
    size_t size = 0;
    if (!file_size_bytes(path, &size)) {
        return NULL;
    }
    char * data = (char *) malloc(size + 1u);
    if (data == NULL) {
        return NULL;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        free(data);
        return NULL;
    }
    const size_t got = fread(data, 1, size, f);
    fclose(f);
    if (got != size) {
        free(data);
        return NULL;
    }
    data[size] = '\0';
    return data;
}

static const char * skip_json_ws(const char * p) {
    while (p != NULL && *p != '\0' && isspace((unsigned char) *p)) {
        ++p;
    }
    return p;
}

static int parse_json_float_array_32(const char * start, const char * key, float out[32]) {
    if (start == NULL || key == NULL || out == NULL) {
        return 0;
    }
    const char * p = strstr(start, key);
    if (p == NULL) {
        return 0;
    }
    p = strchr(p, '[');
    if (p == NULL) {
        return 0;
    }
    ++p;
    for (int i = 0; i < 32; ++i) {
        p = skip_json_ws(p);
        if (p == NULL || *p == '\0') {
            return 0;
        }
        char * end = NULL;
        out[i] = strtof(p, &end);
        if (end == p) {
            return 0;
        }
        p = skip_json_ws(end);
        if (i < 31) {
            if (p == NULL || *p != ',') {
                return 0;
            }
            ++p;
        }
    }
    p = skip_json_ws(p);
    return p != NULL && *p == ']';
}

static int load_slat_normalization(
    const char * model_dir,
    const char * key,
    const char * label,
    float mean[32],
    float std[32]) {
    char path[4096];
    trellis_status status = trellis_make_model_path(model_dir, "pipeline.json", path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("structured latent %s: normalization path failed: %s", label, trellis_status_string(status));
        return 0;
    }
    char * json = read_text_file_null_terminated(path);
    if (json == NULL) {
        TRELLIS_ERROR("structured latent %s: failed to read %s", label, path);
        return 0;
    }
    char section_key[128];
    int n = snprintf(section_key, sizeof(section_key), "\"%s\"", key);
    if (n < 0 || (size_t) n >= sizeof(section_key)) {
        free(json);
        return 0;
    }
    const char * section = strstr(json, section_key);
    int ok = 0;
    if (section != NULL &&
        parse_json_float_array_32(section, "\"mean\"", mean) &&
        parse_json_float_array_32(section, "\"std\"", std)) {
        ok = 1;
    }
    if (!ok) {
        TRELLIS_ERROR("structured latent %s: failed to parse %s from %s", label, key, path);
    }
    free(json);
    return ok;
}

static void denormalize_slat_f32(
    const float * norm,
    int64_t tokens,
    int channels,
    const float * mean,
    const float * std,
    float * out) {
    for (int64_t i = 0; i < tokens; ++i) {
        for (int c = 0; c < channels; ++c) {
            out[(size_t) i * (size_t) channels + (size_t) c] =
                norm[(size_t) i * (size_t) channels + (size_t) c] * std[c] + mean[c];
        }
    }
}

static void normalize_slat_f32(
    const float * value,
    int64_t tokens,
    int channels,
    const float * mean,
    const float * std,
    float * out) {
    for (int64_t i = 0; i < tokens; ++i) {
        for (int c = 0; c < channels; ++c) {
            float s = std[c];
            if (s == 0.0f) {
                s = 1.0f;
            }
            out[(size_t) i * (size_t) channels + (size_t) c] =
                (value[(size_t) i * (size_t) channels + (size_t) c] - mean[c]) / s;
        }
    }
}

static float lcg_uniform(uint32_t * state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float) ((*state >> 8) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

static void fill_gaussian_latent(float * dst, size_t count, uint32_t seed) {
    uint32_t state = seed == 0 ? 1u : seed;
    const float two_pi = 6.2831853071795864769f;
    for (size_t i = 0; i < count; i += 2) {
        float u1 = lcg_uniform(&state);
        float u2 = lcg_uniform(&state);
        if (u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        const float r = sqrtf(-2.0f * logf(u1));
        dst[i] = r * cosf(two_pi * u2);
        if (i + 1 < count) {
            dst[i + 1] = r * sinf(two_pi * u2);
        }
    }
}

static int make_slat_flow_path(
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    char * out,
    size_t out_size) {
    if (override_path != NULL && override_path[0] != '\0') {
        int n = snprintf(out, out_size, "%s", override_path);
        return n >= 0 && (size_t) n < out_size;
    }
    const char * rel = NULL;
    if (component == TRELLIS_COMPONENT_TEX_SLAT_FLOW) {
        rel = resolution == 1024 ?
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors";
    } else {
        rel = resolution == 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
    }
    return trellis_make_model_path(model_dir, rel, out, out_size) == TRELLIS_STATUS_OK;
}

static int load_slat_flow(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_model_component component,
    const char * label,
    trellis_tensor_store * store,
    trellis_dit_flow_weights * flow) {
    char load_label[128];
    snprintf(load_label, sizeof(load_label), "structured-latent %s flow", label);
    if (!trellis_load_tensor_store_f32(cuda, load_label, path, true, 64, store, NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = component == TRELLIS_COMPONENT_TEX_SLAT_FLOW ?
        trellis_tex_slat_flow_bind_weights(store, flow, issue, sizeof(issue)) :
        trellis_shape_slat_flow_bind_weights(store, flow, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("structured-latent %s flow: bind failed: %s%s%s",
            label,
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        return 0;
    }
    TRELLIS_INFO(
        "structured-latent %s flow: ready blocks=%d in=%d out=%d cond=%d heads=%d head_dim=%d",
        label,
        flow->n_blocks,
        flow->in_channels,
        flow->out_channels,
        flow->cond_channels,
        flow->heads,
        flow->head_dim);
    return 1;
}

static void cfg_rescale_combine_population_f32(
    const float * x_t,
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float sigma_min,
    float t,
    float guidance_strength,
    float guidance_rescale,
    float * pred) {
    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    const float one_minus_sigma_min = 1.0f - sigma_min;
    for (size_t i = 0; i < n; ++i) {
        pred[i] = guidance_strength * pred_pos[i] + (1.0f - guidance_strength) * pred_neg[i];
    }
    if (guidance_rescale <= 0.0f) {
        return;
    }

    double mean_pos = 0.0;
    double mean_cfg = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float x0_pos = one_minus_sigma_min * x_t[i] - sigma_t * pred_pos[i];
        const float x0_cfg = one_minus_sigma_min * x_t[i] - sigma_t * pred[i];
        mean_pos += x0_pos;
        mean_cfg += x0_cfg;
    }
    mean_pos /= (double) n;
    mean_cfg /= (double) n;

    double var_pos = 0.0;
    double var_cfg = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x0_pos = (double) (one_minus_sigma_min * x_t[i] - sigma_t * pred_pos[i]) - mean_pos;
        const double x0_cfg = (double) (one_minus_sigma_min * x_t[i] - sigma_t * pred[i]) - mean_cfg;
        var_pos += x0_pos * x0_pos;
        var_cfg += x0_cfg * x0_cfg;
    }
    const double std_pos = sqrt(var_pos / (double) n);
    const double std_cfg = sqrt(var_cfg / (double) n);
    if (std_cfg <= 0.0) {
        return;
    }
    const float ratio = (float) (std_pos / std_cfg);
    for (size_t i = 0; i < n; ++i) {
        const float x0_cfg = one_minus_sigma_min * x_t[i] - sigma_t * pred[i];
        const float x0_rescaled = x0_cfg * ratio;
        const float x0 = guidance_rescale * x0_rescaled + (1.0f - guidance_rescale) * x0_cfg;
        pred[i] = (one_minus_sigma_min * x_t[i] - x0) / sigma_t;
    }
}

void trellis_structured_latent_free(trellis_structured_latent * latent) {
    if (latent == NULL) {
        return;
    }
    free(latent->coords_bxyz);
    free(latent->feats);
    memset(latent, 0, sizeof(*latent));
}

void trellis_shape_latent_free(trellis_shape_latent * latent) {
    trellis_structured_latent_free(latent);
}

trellis_status trellis_pipeline_run_structured_latent(
    const trellis_structured_latent_options * options,
    trellis_structured_latent * latent_out) {
    if (latent_out != NULL) {
        memset(latent_out, 0, sizeof(*latent_out));
    }
    if (options == NULL || latent_out == NULL || options->model_dir == NULL ||
        options->coords_bxyz == NULL || options->n_coords <= 0 ||
        options->cond == NULL || options->cond_tokens <= 0 || options->cuda == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    int resolution = options->resolution > 0 ? options->resolution : 512;
    int steps = options->steps > 0 ? options->steps : 12;
    float rescale_t = options->rescale_t > 0.0f ? options->rescale_t : 3.0f;
    const trellis_model_component component =
        options->flow_component == TRELLIS_COMPONENT_TEX_SLAT_FLOW ?
            TRELLIS_COMPONENT_TEX_SLAT_FLOW :
            TRELLIS_COMPONENT_SHAPE_SLAT_FLOW;
    const char * label = options->label != NULL && options->label[0] != '\0' ?
        options->label :
        (component == TRELLIS_COMPONENT_TEX_SLAT_FLOW ? "texture" : "shape");
    const char * normalization_key = options->normalization_key != NULL && options->normalization_key[0] != '\0' ?
        options->normalization_key :
        (component == TRELLIS_COMPONENT_TEX_SLAT_FLOW ? "tex_slat_normalization" : "shape_slat_normalization");

    char flow_path[4096];
    trellis_tensor_store flow_store;
    trellis_dit_flow_weights flow;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&flow, 0, sizeof(flow));
    trellis_dit_flow_executor flow_executor_cond;
    trellis_dit_flow_executor flow_executor_uncond;
    memset(&flow_executor_cond, 0, sizeof(flow_executor_cond));
    memset(&flow_executor_uncond, 0, sizeof(flow_executor_uncond));

    float * neg_cond = NULL;
    float * latent = NULL;
    float * pred_pos = NULL;
    float * pred_neg = NULL;
    float * pred = NULL;
    float * next = NULL;
    float * x0 = NULL;
    float * denorm = NULL;
    float * run_input = NULL;
    float * concat_norm = NULL;
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    float slat_mean[32];
    float slat_std[32];
    float concat_mean_local[32];
    float concat_std_local[32];
    const float * concat_mean = options->concat_mean;
    const float * concat_std = options->concat_std;

    if (!make_slat_flow_path(options->model_dir, options->flow_override_path, component, resolution, flow_path, sizeof(flow_path))) {
        TRELLIS_ERROR("structured latent %s: failed to build flow path", label);
        goto cleanup;
    }

    TRELLIS_INFO(
        "structured latent %s: pipeline inference resolution=%d steps=%d tokens=%lld noise_seed=%u",
        label,
        resolution,
        steps,
        (long long) options->n_coords,
        (unsigned) options->noise_seed);

    if (!load_slat_flow(options->cuda, flow_path, component, label, &flow_store, &flow)) {
        goto cleanup;
    }
    const int state_channels = flow.out_channels;
    if (state_channels != 32 || flow.cond_channels != 1024) {
        TRELLIS_ERROR(
            "structured latent %s: unexpected flow channels in=%d out=%d cond=%d",
            label,
            flow.in_channels,
            flow.out_channels,
            flow.cond_channels);
        goto cleanup;
    }
    if (component == TRELLIS_COMPONENT_TEX_SLAT_FLOW) {
        if (options->concat_cond == NULL || options->concat_channels <= 0 ||
            state_channels + options->concat_channels != flow.in_channels) {
            TRELLIS_ERROR(
                "structured latent texture: concat conditioning mismatch state=%d concat=%d flow_in=%d",
                state_channels,
                options->concat_channels,
                flow.in_channels);
            goto cleanup;
        }
        if (concat_mean == NULL || concat_std == NULL) {
            if (!load_slat_normalization(
                    options->model_dir,
                    "shape_slat_normalization",
                    "texture concat shape",
                    concat_mean_local,
                    concat_std_local)) {
                goto cleanup;
            }
            concat_mean = concat_mean_local;
            concat_std = concat_std_local;
        }
    } else if (flow.in_channels != state_channels) {
        TRELLIS_ERROR(
            "structured latent %s: flow input/output mismatch in=%d out=%d",
            label,
            flow.in_channels,
            state_channels);
        goto cleanup;
    }
    if (!load_slat_normalization(options->model_dir, normalization_key, label, slat_mean, slat_std)) {
        goto cleanup;
    }
    if (options->flow_blocks_override >= 0) {
        if (options->flow_blocks_override > flow.n_blocks) {
            TRELLIS_ERROR(
                "structured latent %s: --flow-blocks %d exceeds checkpoint blocks %d",
                label,
                options->flow_blocks_override,
                flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = options->flow_blocks_override;
        TRELLIS_INFO("structured latent %s: debug flow blocks override=%d", label, flow.n_blocks);
    }
    if (options->flow_block_parts_override >= 0) {
        if (options->flow_block_parts_override > 3) {
            TRELLIS_ERROR("structured latent %s: --flow-block-parts must be in [0,3]", label);
            goto cleanup;
        }
        flow.debug_block_parts = options->flow_block_parts_override;
        TRELLIS_INFO("structured latent %s: debug flow block parts override=%d", label, flow.debug_block_parts);
    }
    if (options->flow_no_rope) {
        flow.debug_disable_rope = 1;
        TRELLIS_INFO("structured latent %s: debug sparse RoPE disabled", label);
    }
    if (options->emulate_bf16_blocks) {
        flow.emulate_bf16_blocks = 1;
        TRELLIS_INFO("structured latent %s: bf16 block activation round-trip enabled", label);
    }
    trellis_ggml_set_flash_attn_enabled(options->use_ggml_flash_attn);

    const size_t state_count = (size_t) options->n_coords * (size_t) state_channels;
    const size_t input_count = (size_t) options->n_coords * (size_t) flow.in_channels;
    const size_t context_count = (size_t) options->cond_tokens * (size_t) flow.cond_channels;
    const size_t phase_count = (size_t) options->n_coords * (size_t) (flow.head_dim / 2);
    const size_t concat_count = component == TRELLIS_COMPONENT_TEX_SLAT_FLOW ?
        (size_t) options->n_coords * (size_t) options->concat_channels :
        0u;

    if (options->neg_cond == NULL) {
        neg_cond = (float *) calloc(context_count, sizeof(float));
    }
    latent = (float *) malloc(state_count * sizeof(float));
    pred_pos = (float *) malloc(state_count * sizeof(float));
    pred_neg = (float *) malloc(state_count * sizeof(float));
    pred = (float *) malloc(state_count * sizeof(float));
    next = (float *) malloc(state_count * sizeof(float));
    x0 = (float *) malloc(state_count * sizeof(float));
    denorm = (float *) malloc(state_count * sizeof(float));
    run_input = (float *) malloc(input_count * sizeof(float));
    if (concat_count > 0) {
        concat_norm = (float *) malloc(concat_count * sizeof(float));
    }
    pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    latent_out->coords_bxyz = (int32_t *) malloc((size_t) options->n_coords * 4u * sizeof(int32_t));
    if ((options->neg_cond == NULL && neg_cond == NULL) || latent == NULL || pred_pos == NULL ||
        pred_neg == NULL || pred == NULL || next == NULL || x0 == NULL || denorm == NULL ||
        run_input == NULL || (concat_count > 0 && concat_norm == NULL) ||
        pairs == NULL || cos_phase == NULL || sin_phase == NULL || latent_out->coords_bxyz == NULL) {
        TRELLIS_ERROR("structured latent %s: host allocation failed", label);
        goto cleanup;
    }
    if (options->noise != NULL) {
        memcpy(latent, options->noise, state_count * sizeof(float));
        TRELLIS_INFO("structured latent %s: using caller-provided initial noise", label);
    } else {
        const uint32_t noise_seed = options->noise_seed == 0 ? 1u : options->noise_seed;
        fill_gaussian_latent(latent, state_count, noise_seed);
        TRELLIS_INFO("structured latent %s: generated initial noise seed=%u values=%zu", label, (unsigned) noise_seed, state_count);
    }
    if (concat_count > 0) {
        normalize_slat_f32(
            options->concat_cond,
            options->n_coords,
            options->concat_channels,
            concat_mean,
            concat_std,
            concat_norm);
    }

    status = trellis_flow_timestep_pairs_f32(steps, rescale_t, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_sparse_phases_f32(
            options->coords_bxyz,
            options->n_coords,
            flow.head_dim,
            1.0f,
            10000.0f,
            cos_phase,
            sin_phase,
            phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("structured latent %s: schedule/rope failed: %s", label, trellis_status_string(status));
        goto cleanup;
    }

    const float * neg_context = options->neg_cond != NULL ? options->neg_cond : neg_cond;
    status = trellis_dit_flow_executor_init_single(
        &flow_executor_cond,
        options->cuda,
        &flow,
        options->n_coords,
        options->cond_tokens,
        options->cond,
        cos_phase,
        sin_phase);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_dit_flow_executor_init_single(
            &flow_executor_uncond,
            options->cuda,
            &flow,
            options->n_coords,
            options->cond_tokens,
            neg_context,
            cos_phase,
            sin_phase);
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("structured latent %s: flow executor init failed: %s", label, trellis_status_string(status));
        goto cleanup;
    }
    for (int step = 0; step < steps; ++step) {
        const int64_t step_start_us = ggml_time_us();
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        if (concat_count > 0) {
            for (int64_t i = 0; i < options->n_coords; ++i) {
                float * dst = run_input + (size_t) i * (size_t) flow.in_channels;
                memcpy(dst, latent + (size_t) i * (size_t) state_channels, (size_t) state_channels * sizeof(float));
                memcpy(
                    dst + state_channels,
                    concat_norm + (size_t) i * (size_t) options->concat_channels,
                    (size_t) options->concat_channels * sizeof(float));
            }
        } else {
            memcpy(run_input, latent, state_count * sizeof(float));
        }
        status = trellis_dit_flow_executor_run_single(
            &flow_executor_cond,
            run_input,
            t,
            pred_pos);
        if (status == TRELLIS_STATUS_OK) {
            status = trellis_dit_flow_executor_run_single(
                &flow_executor_uncond,
                run_input,
                t,
                pred_neg);
        }
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("structured latent %s: flow step %d failed: %s", label, step + 1, trellis_status_string(status));
            goto cleanup;
        }

        if (t >= options->guidance_min && t <= options->guidance_max) {
            cfg_rescale_combine_population_f32(
                latent,
                pred_pos,
                pred_neg,
                state_count,
                1e-5f,
                t,
                options->guidance_strength,
                options->guidance_rescale,
                pred);
        } else {
            memcpy(pred, pred_pos, state_count * sizeof(float));
        }
        trellis_flow_euler_step_f32(latent, pred, state_count, 1e-5f, t, t_prev, next, x0);
        memcpy(latent, next, state_count * sizeof(float));

        char detail[128];
        snprintf(detail, sizeof(detail), "t=%.6g->%.6g tokens=%lld", t, t_prev, (long long) options->n_coords);
        char progress_label[96];
        snprintf(progress_label, sizeof(progress_label), "structured latent %s", label);
        trellis_progress_steps(progress_label, step + 1, steps, ggml_time_us() - step_start_us, detail);
    }

    denormalize_slat_f32(latent, options->n_coords, state_channels, slat_mean, slat_std, denorm);
    latent_out->feats = denorm;
    denorm = NULL;
    memcpy(latent_out->coords_bxyz, options->coords_bxyz, (size_t) options->n_coords * 4u * sizeof(int32_t));
    latent_out->n_coords = options->n_coords;
    latent_out->resolution = resolution;
    latent_out->channels = state_channels;
    status = TRELLIS_STATUS_OK;

cleanup:
    trellis_ggml_set_flash_attn_enabled(0);
    if (status != TRELLIS_STATUS_OK) {
        trellis_structured_latent_free(latent_out);
    }
    free(neg_cond);
    free(latent);
    free(pred_pos);
    free(pred_neg);
    free(pred);
    free(next);
    free(x0);
    free(denorm);
    free(run_input);
    free(concat_norm);
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    trellis_dit_flow_executor_free(&flow_executor_cond);
    trellis_dit_flow_executor_free(&flow_executor_uncond);
    trellis_tensor_store_free(&flow_store);
    return status;
}
