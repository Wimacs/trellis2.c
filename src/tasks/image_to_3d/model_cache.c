#include "image_to_3d_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double bytes_to_mib(size_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static void cache_entry_init(
    trellis_pipeline_cache_entry * entry,
    const char * name,
    trellis_pipeline_cache_entry_kind kind) {
    memset(entry, 0, sizeof(*entry));
    entry->name = name;
    entry->kind = kind;
}

static size_t cache_entry_count(void) {
    return 10;
}

static trellis_pipeline_cache_entry * cache_entry_at(trellis_pipeline_model_cache * cache, size_t index) {
    if (cache == NULL) {
        return NULL;
    }
    switch (index) {
        case 0: return &cache->dino;
        case 1: return &cache->sparse_structure_flow;
        case 2: return &cache->sparse_structure_decoder;
        case 3: return &cache->shape_flow_512;
        case 4: return &cache->shape_flow_1024;
        case 5: return &cache->texture_flow_512;
        case 6: return &cache->texture_flow_1024;
        case 7: return &cache->shape_decoder;
        case 8: return &cache->texture_decoder;
        case 9: return &cache->pixal_naf;
        default: return NULL;
    }
}

static size_t cache_store_budget_bytes(const trellis_tensor_store * store) {
    if (store == NULL || store->buffer == NULL) {
        return 0;
    }
    if (ggml_backend_buffer_is_host(store->buffer)) {
        return 0;
    }
    return ggml_backend_buffer_get_size(store->buffer);
}

static size_t estimate_safetensors_data_bytes(const char * path) {
    trellis_safetensors st;
    memset(&st, 0, sizeof(st));
    if (trellis_safetensors_open(path, &st) != TRELLIS_STATUS_OK) {
        return 0;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < st.n_tensors; ++i) {
        if (st.tensors[i].data_end >= st.tensors[i].data_begin) {
            total += st.tensors[i].data_end - st.tensors[i].data_begin;
        }
    }
    trellis_safetensors_close(&st);
    return total > (uint64_t) SIZE_MAX ? SIZE_MAX : (size_t) total;
}

static void cache_entry_release(trellis_pipeline_model_cache * cache, trellis_pipeline_cache_entry * entry) {
    if (cache == NULL || entry == NULL || !entry->loaded) {
        return;
    }
    TRELLIS_INFO("model cache: evict %s gpu=%.1f MiB", entry->name, bytes_to_mib(entry->budget_bytes));
    if (cache->used_budget_bytes >= entry->budget_bytes) {
        cache->used_budget_bytes -= entry->budget_bytes;
    } else {
        cache->used_budget_bytes = 0;
    }
    trellis_tensor_store_free(&entry->store);
    memset(&entry->weights, 0, sizeof(entry->weights));
    entry->loaded = 0;
    entry->pinned = 0;
    entry->last_used = 0;
    entry->budget_bytes = 0;
    entry->path[0] = '\0';
}

static void cache_touch(trellis_pipeline_model_cache * cache, trellis_pipeline_cache_entry * entry) {
    if (cache == NULL || entry == NULL) {
        return;
    }
    entry->last_used = ++cache->clock;
    entry->pinned = 1;
}

static void cache_evict_to_fit(
    trellis_pipeline_model_cache * cache,
    size_t incoming_bytes,
    trellis_pipeline_cache_entry * keep_entry) {
    if (cache == NULL || cache->budget_bytes == 0) {
        return;
    }
    if (incoming_bytes > cache->budget_bytes) {
        TRELLIS_WARN(
            "model cache: single entry estimate %.1f MiB exceeds budget %.1f MiB; evicting what can be evicted",
            bytes_to_mib(incoming_bytes),
            bytes_to_mib(cache->budget_bytes));
    }
    const size_t count = cache_entry_count();
    while (cache->used_budget_bytes + incoming_bytes > cache->budget_bytes) {
        trellis_pipeline_cache_entry * victim = NULL;
        for (size_t i = 0; i < count; ++i) {
            trellis_pipeline_cache_entry * entry = cache_entry_at(cache, i);
            if (!entry->loaded || entry->pinned || entry == keep_entry) {
                continue;
            }
            if (victim == NULL || entry->last_used < victim->last_used) {
                victim = entry;
            }
        }
        if (victim == NULL) {
            TRELLIS_WARN(
                "model cache: budget %.1f MiB cannot fit active set used=%.1f MiB incoming=%.1f MiB",
                bytes_to_mib(cache->budget_bytes),
                bytes_to_mib(cache->used_budget_bytes),
                bytes_to_mib(incoming_bytes));
            break;
        }
        cache_entry_release(cache, victim);
    }
}

static trellis_status copy_path(char * dst, size_t dst_size, const char * src) {
    if (dst == NULL || dst_size == 0 || src == NULL || src[0] == '\0') {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int n = snprintf(dst, dst_size, "%s", src);
    return n >= 0 && (size_t) n < dst_size ? TRELLIS_STATUS_OK : TRELLIS_STATUS_INVALID_ARGUMENT;
}

static trellis_status make_join_path(const char * dir, const char * file, char * dst, size_t dst_size) {
    if (dir == NULL || file == NULL || dst == NULL || dst_size == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int n = snprintf(dst, dst_size, "%s/%s", dir, file);
    return n >= 0 && (size_t) n < dst_size ? TRELLIS_STATUS_OK : TRELLIS_STATUS_INVALID_ARGUMENT;
}

static trellis_status make_slat_flow_path(
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    char * out,
    size_t out_size) {
    if (override_path != NULL && override_path[0] != '\0') {
        return copy_path(out, out_size, override_path);
    }
    const char * rel = NULL;
    if (component == TRELLIS_COMPONENT_TEX_SLAT_FLOW) {
        rel = resolution >= 1024 ?
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors";
    } else {
        rel = resolution >= 1024 ?
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" :
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
    }
    return trellis_make_model_path(model_dir, rel, out, out_size);
}

static trellis_status cache_prepare_load(
    trellis_pipeline_model_cache * cache,
    trellis_pipeline_cache_entry * entry,
    const char * path) {
    if (cache == NULL || entry == NULL || path == NULL || path[0] == '\0') {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (entry->loaded && strcmp(entry->path, path) == 0) {
        cache_touch(cache, entry);
        TRELLIS_INFO(
            "model cache: hit %s gpu=%.1f MiB used=%.1f/%.1f MiB",
            entry->name,
            bytes_to_mib(entry->budget_bytes),
            bytes_to_mib(cache->used_budget_bytes),
            cache->budget_bytes == 0 ? 0.0 : bytes_to_mib(cache->budget_bytes));
        return TRELLIS_STATUS_OK;
    }
    if (entry->loaded) {
        cache_entry_release(cache, entry);
    }
    const size_t estimate = cache->budget_bytes > 0 ?
        estimate_safetensors_data_bytes(path) :
        0;
    cache_evict_to_fit(cache, estimate, entry);
    trellis_status status = copy_path(entry->path, sizeof(entry->path), path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (cache->budget_bytes == 0) {
        TRELLIS_INFO("model cache: load %s budget=unlimited", entry->name);
    } else {
        TRELLIS_INFO(
            "model cache: load %s estimate=%.1f MiB budget=limited",
            entry->name,
            bytes_to_mib(estimate));
    }
    return TRELLIS_STATUS_OK;
}

static void cache_finish_load(trellis_pipeline_model_cache * cache, trellis_pipeline_cache_entry * entry) {
    entry->budget_bytes = cache_store_budget_bytes(&entry->store);
    cache->used_budget_bytes += entry->budget_bytes;
    entry->loaded = 1;
    cache_touch(cache, entry);
    TRELLIS_INFO(
        "model cache: resident %s gpu=%.1f MiB used=%.1f/%.1f MiB",
        entry->name,
        bytes_to_mib(entry->budget_bytes),
        bytes_to_mib(cache->used_budget_bytes),
        cache->budget_bytes == 0 ? 0.0 : bytes_to_mib(cache->budget_bytes));
    cache_evict_to_fit(cache, 0, entry);
}

trellis_status trellis_pipeline_model_cache_init(
    trellis_pipeline_model_cache * cache,
    const trellis_backend_context * backend,
    int use_cpu_decoder_weight_backend,
    size_t budget_bytes) {
    if (cache == NULL || backend == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(cache, 0, sizeof(*cache));
    cache->backend = backend;
    cache->decoder_weight_backend = backend;
    cache->budget_bytes = budget_bytes;

    cache_entry_init(&cache->dino, "dino", TRELLIS_PIPELINE_CACHE_ENTRY_DINO);
    cache_entry_init(&cache->sparse_structure_flow, "sparse-structure flow", TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW);
    cache_entry_init(&cache->sparse_structure_decoder, "sparse-structure decoder", TRELLIS_PIPELINE_CACHE_ENTRY_SS_DECODER);
    cache_entry_init(&cache->shape_flow_512, "shape flow 512", TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW);
    cache_entry_init(&cache->shape_flow_1024, "shape flow 1024", TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW);
    cache_entry_init(&cache->texture_flow_512, "texture flow 512", TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW);
    cache_entry_init(&cache->texture_flow_1024, "texture flow 1024", TRELLIS_PIPELINE_CACHE_ENTRY_DIT_FLOW);
    cache_entry_init(&cache->shape_decoder, "shape decoder", TRELLIS_PIPELINE_CACHE_ENTRY_SPARSE_UNET_DECODER);
    cache_entry_init(&cache->texture_decoder, "texture decoder", TRELLIS_PIPELINE_CACHE_ENTRY_SPARSE_UNET_DECODER);
    cache_entry_init(&cache->pixal_naf, "Pixal3D NAF image encoder", TRELLIS_PIPELINE_CACHE_ENTRY_PIXAL_NAF);

    if (use_cpu_decoder_weight_backend) {
        trellis_status status = trellis_backend_init(&cache->cpu_decoder_weight_backend, TRELLIS_BACKEND_CPU, 0);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("model cache: CPU decoder weight backend init failed: %s", trellis_status_string(status));
            return status;
        }
        cache->decoder_weight_backend = &cache->cpu_decoder_weight_backend;
        cache->owns_cpu_decoder_weight_backend = 1;
    }
    TRELLIS_INFO(
        "model cache: initialized budget=%s decoder_weights=%s",
        budget_bytes == 0 ? "unlimited" : "limited",
        trellis_backend_kind_name(cache->decoder_weight_backend->kind));
    return TRELLIS_STATUS_OK;
}

void trellis_pipeline_model_cache_free(trellis_pipeline_model_cache * cache) {
    if (cache == NULL) {
        return;
    }
    const size_t count = cache_entry_count();
    for (size_t i = 0; i < count; ++i) {
        trellis_pipeline_cache_entry * entry = cache_entry_at(cache, i);
        cache_entry_release(cache, entry);
    }
    if (cache->owns_cpu_decoder_weight_backend) {
        trellis_backend_free(&cache->cpu_decoder_weight_backend);
    }
    memset(cache, 0, sizeof(*cache));
}

void trellis_pipeline_model_cache_unpin_all(trellis_pipeline_model_cache * cache) {
    if (cache == NULL) {
        return;
    }
    const size_t count = cache_entry_count();
    for (size_t i = 0; i < count; ++i) {
        trellis_pipeline_cache_entry * entry = cache_entry_at(cache, i);
        if (entry != NULL) {
            entry->pinned = 0;
        }
    }
}

void trellis_pipeline_model_cache_evict_unpinned(trellis_pipeline_model_cache * cache) {
    if (cache == NULL) {
        return;
    }
    const size_t count = cache_entry_count();
    for (size_t i = 0; i < count; ++i) {
        trellis_pipeline_cache_entry * entry = cache_entry_at(cache, i);
        if (entry != NULL && entry->loaded && !entry->pinned) {
            cache_entry_release(cache, entry);
        }
    }
}

trellis_status trellis_pipeline_model_cache_get_dino(
    trellis_pipeline_model_cache * cache,
    const char * dino_dir,
    const trellis_dino_vit_weights ** weights_out) {
    if (cache == NULL || dino_dir == NULL || dino_dir[0] == '\0' ||
        weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    char path[4096];
    trellis_status status = make_join_path(dino_dir, "model.safetensors", path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    trellis_pipeline_cache_entry * entry = &cache->dino;
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store(cache->backend, "sparse-structure dino image encoder", path, true, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = trellis_dino_vit_bind_weights(&entry->store, &entry->weights.dino, issue, sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("sparse-structure dino image encoder: bind failed: %s%s%s",
                trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO("sparse-structure dino image encoder: ready");
        cache_finish_load(cache, entry);
    }
    *weights_out = &entry->weights.dino;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_sparse_structure_flow_model(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_dit_flow_model ** model_out) {
    if (cache == NULL || model_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *model_out = NULL;
    char path[4096];
    trellis_status status = override_path != NULL && override_path[0] != '\0' ?
        copy_path(path, sizeof(path), override_path) :
        trellis_make_model_path(
            model_dir,
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            path,
            sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    trellis_pipeline_cache_entry * entry = &cache->sparse_structure_flow;
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store(cache->backend, "sparse-structure flow", path, true, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = trellis_ss_flow_model_bind_weights(
            &entry->store,
            &entry->weights.flow_model,
            issue,
            sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("sparse-structure flow: bind failed: %s%s%s",
                trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO(
            "sparse-structure flow: ready blocks=%d channels=%d",
            entry->weights.flow_model.base.n_blocks,
            entry->weights.flow_model.base.in_channels);
        if (entry->weights.flow_model.projection.enabled) {
            TRELLIS_INFO(
                "sparse-structure flow: Pixal3D projection channels=%d",
                entry->weights.flow_model.projection.proj_channels);
        }
        cache_finish_load(cache, entry);
    }
    *model_out = &entry->weights.flow_model;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_sparse_structure_flow(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const trellis_dit_flow_weights ** weights_out) {
    if (cache == NULL || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    const trellis_dit_flow_model * model = NULL;
    trellis_status status = trellis_pipeline_model_cache_get_sparse_structure_flow_model(
        cache, model_dir, NULL, &model);
    if (status == TRELLIS_STATUS_OK) {
        *weights_out = &model->base;
    }
    return status;
}

trellis_status trellis_pipeline_model_cache_get_sparse_structure_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_ss_decoder_weights ** weights_out) {
    if (cache == NULL || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    char path[4096];
    trellis_status status = override_path != NULL && override_path[0] != '\0' ?
        copy_path(path, sizeof(path), override_path) :
        trellis_make_model_path(
            model_dir,
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            path,
            sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    trellis_pipeline_cache_entry * entry = &cache->sparse_structure_decoder;
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store(cache->backend, "sparse-structure decoder", path, false, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = trellis_ss_decoder_bind_weights(&entry->store, &entry->weights.ss_decoder, issue, sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("sparse-structure decoder: bind failed: %s%s%s",
                trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO("sparse-structure decoder: ready");
        cache_finish_load(cache, entry);
    }
    *weights_out = &entry->weights.ss_decoder;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_slat_flow_model(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    const char * label,
    const trellis_dit_flow_model ** model_out) {
    if (cache == NULL || model_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *model_out = NULL;
    const int is_tex = component == TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    trellis_pipeline_cache_entry * entry = is_tex ?
        (resolution >= 1024 ? &cache->texture_flow_1024 : &cache->texture_flow_512) :
        (resolution >= 1024 ? &cache->shape_flow_1024 : &cache->shape_flow_512);
    char path[4096];
    trellis_status status = make_slat_flow_path(model_dir, override_path, component, resolution, path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        char load_label[128];
        snprintf(load_label, sizeof(load_label), "structured-latent %s flow", label != NULL ? label : "slat");
        if (!trellis_load_tensor_store(cache->backend, load_label, path, true, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = is_tex ?
            trellis_tex_slat_flow_model_bind_weights(
                &entry->store, &entry->weights.flow_model, issue, sizeof(issue)) :
            trellis_shape_slat_flow_model_bind_weights(
                &entry->store, &entry->weights.flow_model, issue, sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("structured-latent %s flow: bind failed: %s%s%s",
                label != NULL ? label : "slat",
                trellis_status_string(status),
                issue[0] == '\0' ? "" : " ",
                issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO(
            "structured-latent %s flow: ready blocks=%d in=%d out=%d cond=%d heads=%d head_dim=%d",
            label != NULL ? label : "slat",
            entry->weights.flow_model.base.n_blocks,
            entry->weights.flow_model.base.in_channels,
            entry->weights.flow_model.base.out_channels,
            entry->weights.flow_model.base.cond_channels,
            entry->weights.flow_model.base.heads,
            entry->weights.flow_model.base.head_dim);
        if (entry->weights.flow_model.projection.enabled) {
            TRELLIS_INFO(
                "structured-latent %s flow: Pixal3D projection channels=%d",
                label != NULL ? label : "slat",
                entry->weights.flow_model.projection.proj_channels);
        }
        cache_finish_load(cache, entry);
    }
    *model_out = &entry->weights.flow_model;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_slat_flow(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    trellis_model_component component,
    int resolution,
    const char * label,
    const trellis_dit_flow_weights ** weights_out) {
    if (cache == NULL || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    const trellis_dit_flow_model * model = NULL;
    trellis_status status = trellis_pipeline_model_cache_get_slat_flow_model(
        cache,
        model_dir,
        override_path,
        component,
        resolution,
        label,
        &model);
    if (status == TRELLIS_STATUS_OK) {
        *weights_out = &model->base;
    }
    return status;
}

trellis_status trellis_pipeline_model_cache_get_shape_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_sparse_unet_vae_decoder_weights ** weights_out) {
    if (cache == NULL || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    char path[4096];
    trellis_status status = override_path != NULL && override_path[0] != '\0' ?
        copy_path(path, sizeof(path), override_path) :
        trellis_make_model_path(model_dir, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    trellis_pipeline_cache_entry * entry = &cache->shape_decoder;
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store_f32(cache->decoder_weight_backend, "FlexiDualGridVaeDecoder", path, true, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = trellis_flexi_dual_grid_vae_decoder_bind_weights(
            &entry->store,
            &entry->weights.sparse_unet_decoder,
            issue,
            sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("FlexiDualGridVaeDecoder: bind failed: %s%s%s",
                trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO("FlexiDualGridVaeDecoder: ready");
        cache_finish_load(cache, entry);
    }
    *weights_out = &entry->weights.sparse_unet_decoder;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_texture_decoder(
    trellis_pipeline_model_cache * cache,
    const char * model_dir,
    const char * override_path,
    const trellis_sparse_unet_vae_decoder_weights ** weights_out) {
    if (cache == NULL || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    char path[4096];
    trellis_status status = override_path != NULL && override_path[0] != '\0' ?
        copy_path(path, sizeof(path), override_path) :
        trellis_make_model_path(
            model_dir,
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            path,
            sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    trellis_pipeline_cache_entry * entry = &cache->texture_decoder;
    status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store_f32(cache->decoder_weight_backend, "Texture SparseUnetVaeDecoder", path, true, 64, &entry->store, NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256];
        status = trellis_tex_slat_decoder_bind_weights(
            &entry->store,
            &entry->weights.sparse_unet_decoder,
            issue,
            sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("Texture SparseUnetVaeDecoder: bind failed: %s%s%s",
                trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO("Texture SparseUnetVaeDecoder: ready");
        cache_finish_load(cache, entry);
    }
    *weights_out = &entry->weights.sparse_unet_decoder;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_model_cache_get_pixal_naf(
    trellis_pipeline_model_cache * cache,
    const char * path,
    const trellis_pixal_naf_ggml_weights ** weights_out) {
    if (cache == NULL || path == NULL || path[0] == '\0' || weights_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *weights_out = NULL;
    trellis_pipeline_cache_entry * entry = &cache->pixal_naf;
    trellis_status status = cache_prepare_load(cache, entry, path);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (!entry->loaded) {
        if (!trellis_load_tensor_store_f32(
                cache->backend,
                "Pixal3D NAF image encoder",
                path,
                false,
                16,
                &entry->store,
                NULL)) {
            return TRELLIS_STATUS_ERROR;
        }
        char issue[256] = {0};
        status = trellis_pixal_naf_bind_ggml_weights(
            &entry->store,
            &entry->weights.pixal_naf,
            issue,
            sizeof(issue));
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "Pixal3D NAF: weight bind failed: %s%s%s",
                trellis_status_string(status),
                issue[0] == '\0' ? "" : " ",
                issue);
            trellis_tensor_store_free(&entry->store);
            return status;
        }
        TRELLIS_INFO("Pixal3D NAF image encoder: ready");
        cache_finish_load(cache, entry);
    }
    *weights_out = &entry->weights.pixal_naf;
    return TRELLIS_STATUS_OK;
}
