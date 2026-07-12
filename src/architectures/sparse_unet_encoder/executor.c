#include "trellis.h"
#include "s2c_host.h"
#include "sparse/trellis_sparse_backend.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static trellis_status encoder_sparse_backend_create(
    trellis_sparse_backend_kind kind,
    int device,
    trellis_sparse_backend ** backend_out) {
    if (backend_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *backend_out = NULL;
    switch (kind) {
        case TRELLIS_SPARSE_BACKEND_CPU:
            return trellis_sparse_cpu_backend_create(backend_out);
        case TRELLIS_SPARSE_BACKEND_VULKAN:
            return trellis_sparse_vulkan_backend_create(device, backend_out);
        default:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
}

static trellis_status encoder_convnext_exec(
    trellis_sparse_backend * backend,
    const trellis_sparse_unet_vae_encoder_convnext_block_weights * block,
    const trellis_sparse_rulebook * rulebook,
    trellis_sparse_buffer * input,
    int64_t n,
    trellis_sparse_buffer ** output) {
    if (backend == NULL || backend->ops == NULL || block == NULL || rulebook == NULL ||
        input == NULL || output == NULL || n <= 0 || block->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *output = NULL;
    const trellis_sparse_backend_ops * ops = backend->ops;
    const int channels = block->channels;
    trellis_sparse_buffer * conv = NULL;
    trellis_sparse_buffer * norm = NULL;
    trellis_sparse_buffer * mlp0 = NULL;
    trellis_sparse_buffer * mlp2 = NULL;
    trellis_sparse_buffer * out = NULL;

    trellis_status status = ops->alloc_f32(backend, (size_t) n * (size_t) channels, &conv);
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) n * (size_t) channels, &norm);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) n * (size_t) channels * 4u, &mlp0);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) n * (size_t) channels, &mlp2);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) n * (size_t) channels, &out);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(
            backend,
            rulebook,
            input,
            block->conv_w,
            block->conv_b,
            conv,
            n,
            channels,
            channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(
            backend,
            conv,
            block->norm_gamma,
            block->norm_beta,
            norm,
            n,
            channels,
            1e-6f);
    }
    if (status == TRELLIS_STATUS_OK && ops->linear_silu != NULL) {
        status = ops->linear_silu(
            backend,
            norm,
            block->mlp0_w,
            block->mlp0_b,
            mlp0,
            n,
            channels,
            4 * channels);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(
            backend,
            norm,
            block->mlp0_w,
            block->mlp0_b,
            mlp0,
            n,
            channels,
            4 * channels);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(
                backend, mlp0, (size_t) n * (size_t) channels * 4u);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(
            backend,
            mlp0,
            block->mlp2_w,
            block->mlp2_b,
            mlp2,
            n,
            4 * channels,
            channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->add(
            backend, input, mlp2, out, (size_t) n * (size_t) channels);
    }

    ops->free_buffer(backend, conv);
    ops->free_buffer(backend, norm);
    ops->free_buffer(backend, mlp0);
    ops->free_buffer(backend, mlp2);
    if (status == TRELLIS_STATUS_OK) {
        *output = out;
    } else {
        ops->free_buffer(backend, out);
    }
    return status;
}

static trellis_status encoder_copy_guide(
    const int32_t * fine_coords,
    int64_t fine_n,
    int32_t ** parent,
    int32_t ** subidx,
    trellis_sparse_c2s_guide_level * guide) {
    if (fine_coords == NULL || parent == NULL || subidx == NULL || *parent == NULL ||
        *subidx == NULL || guide == NULL || fine_n <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(guide, 0, sizeof(*guide));
    guide->coords_bxyz = (int32_t *) malloc((size_t) fine_n * 4u * sizeof(int32_t));
    if (guide->coords_bxyz == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(guide->coords_bxyz, fine_coords, (size_t) fine_n * 4u * sizeof(int32_t));
    guide->parent = *parent;
    guide->subidx = *subidx;
    guide->n_coords = fine_n;
    *parent = NULL;
    *subidx = NULL;
    return TRELLIS_STATUS_OK;
}

static trellis_status encoder_s2c_exec(
    trellis_sparse_backend * backend,
    const trellis_sparse_unet_vae_encoder_s2c_block_weights * block,
    const int32_t * fine_coords,
    const trellis_sparse_rulebook * fine_rulebook,
    trellis_sparse_buffer * input,
    int64_t fine_n,
    int32_t ** coarse_coords_out,
    trellis_sparse_rulebook ** coarse_rulebook_out,
    trellis_sparse_buffer ** output,
    int64_t * coarse_n_out,
    trellis_sparse_c2s_guide_level * guide) {
    if (backend == NULL || backend->ops == NULL || block == NULL || fine_coords == NULL ||
        fine_rulebook == NULL || input == NULL || fine_n <= 0 || coarse_coords_out == NULL ||
        coarse_rulebook_out == NULL || output == NULL || coarse_n_out == NULL || guide == NULL ||
        block->in_channels <= 0 || block->out_channels <= 0 || block->out_channels % 8 != 0 ||
        (8 * block->in_channels) % block->out_channels != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coarse_coords_out = NULL;
    *coarse_rulebook_out = NULL;
    *output = NULL;
    *coarse_n_out = 0;
    memset(guide, 0, sizeof(*guide));

    const trellis_sparse_backend_ops * ops = backend->ops;
    const int ci = block->in_channels;
    const int co = block->out_channels;
    const int packed_channels = co / 8;
    const int skip_group = 8 * ci / co;
    int32_t * coarse_coords = NULL;
    int32_t * parent = NULL;
    int32_t * subidx = NULL;
    int64_t coarse_n = 0;
    float * conv1_host = NULL;
    float * input_host = NULL;
    float * packed_host = NULL;
    float * skip_host = NULL;
    trellis_sparse_buffer * norm1 = NULL;
    trellis_sparse_buffer * conv1 = NULL;
    trellis_sparse_buffer * packed = NULL;
    trellis_sparse_buffer * skip = NULL;
    trellis_sparse_buffer * norm2 = NULL;
    trellis_sparse_buffer * conv2 = NULL;
    trellis_sparse_buffer * out = NULL;
    trellis_sparse_rulebook * coarse_rulebook = NULL;

    trellis_status status = trellis_sparse_s2c_host_build(
        fine_coords,
        fine_n,
        &coarse_coords,
        &coarse_n,
        &parent,
        &subidx);
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) fine_n * (size_t) ci, &norm1);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(
            backend, (size_t) fine_n * (size_t) packed_channels, &conv1);
    }
    if (status == TRELLIS_STATUS_OK && ops->row_norm_silu != NULL) {
        status = ops->row_norm_silu(
            backend,
            input,
            block->norm1_gamma,
            block->norm1_beta,
            norm1,
            fine_n,
            ci,
            1e-6f);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(
            backend,
            input,
            block->norm1_gamma,
            block->norm1_beta,
            norm1,
            fine_n,
            ci,
            1e-6f);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(backend, norm1, (size_t) fine_n * (size_t) ci);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(
            backend,
            fine_rulebook,
            norm1,
            block->conv1_w,
            block->conv1_b,
            conv1,
            fine_n,
            ci,
            packed_channels);
    }

    if (status == TRELLIS_STATUS_OK) {
        conv1_host = (float *) malloc(
            (size_t) fine_n * (size_t) packed_channels * sizeof(float));
        input_host = (float *) malloc((size_t) fine_n * (size_t) ci * sizeof(float));
        packed_host = (float *) calloc((size_t) coarse_n * (size_t) co, sizeof(float));
        skip_host = (float *) calloc((size_t) coarse_n * (size_t) co, sizeof(float));
        if (conv1_host == NULL || input_host == NULL || packed_host == NULL || skip_host == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->download_f32(
            backend,
            conv1,
            conv1_host,
            (size_t) fine_n * (size_t) packed_channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->download_f32(
            backend, input, input_host, (size_t) fine_n * (size_t) ci);
    }
    if (status == TRELLIS_STATUS_OK) {
        for (int64_t fine_row = 0; fine_row < fine_n; ++fine_row) {
            const int32_t coarse_row = parent[fine_row];
            const int32_t child = subidx[fine_row];
            memcpy(
                packed_host + (int64_t) coarse_row * co + child * packed_channels,
                conv1_host + fine_row * packed_channels,
                (size_t) packed_channels * sizeof(float));
            for (int channel = 0; channel < ci; ++channel) {
                const int flattened_channel = child * ci + channel;
                const int out_channel = flattened_channel / skip_group;
                skip_host[(int64_t) coarse_row * co + out_channel] +=
                    input_host[fine_row * ci + channel] / (float) skip_group;
            }
        }
        status = ops->upload_f32(
            backend, packed_host, (size_t) coarse_n * (size_t) co, &packed);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->upload_f32(
            backend, skip_host, (size_t) coarse_n * (size_t) co, &skip);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->build_rulebook(backend, coarse_coords, coarse_n, &coarse_rulebook);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) coarse_n * (size_t) co, &norm2);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) coarse_n * (size_t) co, &conv2);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) coarse_n * (size_t) co, &out);
    }
    if (status == TRELLIS_STATUS_OK && ops->row_norm_silu != NULL) {
        status = ops->row_norm_silu(
            backend, packed, NULL, NULL, norm2, coarse_n, co, 1e-6f);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(
            backend, packed, NULL, NULL, norm2, coarse_n, co, 1e-6f);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(
                backend, norm2, (size_t) coarse_n * (size_t) co);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(
            backend,
            coarse_rulebook,
            norm2,
            block->conv2_w,
            block->conv2_b,
            conv2,
            coarse_n,
            co,
            co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->add(
            backend, conv2, skip, out, (size_t) coarse_n * (size_t) co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = encoder_copy_guide(fine_coords, fine_n, &parent, &subidx, guide);
    }

    ops->free_buffer(backend, norm1);
    ops->free_buffer(backend, conv1);
    ops->free_buffer(backend, packed);
    ops->free_buffer(backend, skip);
    ops->free_buffer(backend, norm2);
    ops->free_buffer(backend, conv2);
    free(conv1_host);
    free(input_host);
    free(packed_host);
    free(skip_host);
    free(parent);
    free(subidx);

    if (status == TRELLIS_STATUS_OK) {
        *coarse_coords_out = coarse_coords;
        *coarse_rulebook_out = coarse_rulebook;
        *output = out;
        *coarse_n_out = coarse_n;
    } else {
        free(coarse_coords);
        if (coarse_rulebook != NULL) {
            ops->free_rulebook(backend, coarse_rulebook);
        }
        ops->free_buffer(backend, out);
        trellis_sparse_c2s_guides failed_guide;
        memset(&failed_guide, 0, sizeof(failed_guide));
        failed_guide.levels[0] = *guide;
        trellis_sparse_c2s_guides_free(&failed_guide);
        memset(guide, 0, sizeof(*guide));
    }
    return status;
}

static void encoder_reverse_guides(
    trellis_sparse_c2s_guides * down_order,
    int n_guides,
    trellis_sparse_c2s_guides * reverse_order) {
    if (down_order == NULL || reverse_order == NULL || n_guides < 0) {
        return;
    }
    memset(reverse_order, 0, sizeof(*reverse_order));
    for (int i = 0; i < n_guides; ++i) {
        reverse_order->levels[i] = down_order->levels[n_guides - 1 - i];
        memset(&down_order->levels[n_guides - 1 - i], 0, sizeof(down_order->levels[0]));
    }
    reverse_order->n_levels = n_guides;
}

trellis_status trellis_sparse_unet_vae_encoder_forward_backend_f32_host(
    const trellis_sparse_unet_vae_encoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    const trellis_sparse_unet_vae_encoder_forward_options * options,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out) {
    if (weights == NULL || coords == NULL || feats == NULL || options == NULL ||
        coords_out == NULL || feats_out == NULL || n_out == NULL || channels_out == NULL ||
        n <= 0 || weights->in_channels <= 0 || weights->latent_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *feats_out = NULL;
    *n_out = 0;
    *channels_out = 0;
    if (options->return_subs != NULL) {
        memset(options->return_subs, 0, sizeof(*options->return_subs));
    }
    if (options->backend_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
#if TRELLIS_HAS_CUDA_DECODER
        return trellis_sparse_unet_vae_encoder_forward_f32_host(
            weights,
            coords,
            feats,
            n,
            options->device,
            options->max_levels,
            options->return_subs,
            coords_out,
            feats_out,
            n_out,
            channels_out);
#else
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
#endif
    }

    const int levels = weights->levels <= 0 ? TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS : weights->levels;
    const int levels_to_run =
        options->max_levels <= 0 || options->max_levels > levels ? levels : options->max_levels;
    if (levels <= 0 || levels > TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS ||
        levels_to_run <= 0 || levels_to_run > levels) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_sparse_backend * backend = (trellis_sparse_backend *) options->sparse_backend;
    const int owns_backend = backend == NULL;
    trellis_sparse_buffer * input = NULL;
    trellis_sparse_buffer * cur_h = NULL;
    trellis_sparse_rulebook * cur_rulebook = NULL;
    int32_t * cur_coords = NULL;
    int64_t cur_n = n;
    int cur_channels = weights->channels[0];
    int32_t * host_coords = NULL;
    float * host_feats = NULL;
    trellis_sparse_c2s_guides down_guides;
    int n_guides = 0;
    memset(&down_guides, 0, sizeof(down_guides));

    trellis_status status = TRELLIS_STATUS_OK;
    if (backend == NULL) {
        status = encoder_sparse_backend_create(options->backend_kind, options->device, &backend);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
    } else if (backend->ops == NULL || backend->kind != options->backend_kind) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_sparse_backend_ops * ops = backend->ops;

    cur_coords = (int32_t *) malloc((size_t) n * 4u * sizeof(int32_t));
    if (cur_coords == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(cur_coords, coords, (size_t) n * 4u * sizeof(int32_t));
    status = ops->upload_f32(
        backend, feats, (size_t) n * (size_t) weights->in_channels, &input);
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(
            backend, (size_t) n * (size_t) weights->channels[0], &cur_h);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(
            backend,
            input,
            weights->input_w,
            weights->input_b,
            cur_h,
            n,
            weights->in_channels,
            weights->channels[0]);
    }
    ops->free_buffer(backend, input);
    input = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = ops->build_rulebook(backend, cur_coords, cur_n, &cur_rulebook);
    }

    for (int level = 0; status == TRELLIS_STATUS_OK && level < levels_to_run; ++level) {
        cur_channels = weights->channels[level];
        for (int block = 0;
             status == TRELLIS_STATUS_OK && block < weights->blocks_per_level[level];
             ++block) {
            trellis_sparse_buffer * next_h = NULL;
            status = encoder_convnext_exec(
                backend,
                &weights->blocks[level][block],
                cur_rulebook,
                cur_h,
                cur_n,
                &next_h);
            if (status == TRELLIS_STATUS_OK) {
                ops->free_buffer(backend, cur_h);
                cur_h = next_h;
            }
        }
        if (status != TRELLIS_STATUS_OK || level >= levels_to_run - 1 || level >= levels - 1) {
            break;
        }

        int32_t * coarse_coords = NULL;
        trellis_sparse_rulebook * coarse_rulebook = NULL;
        trellis_sparse_buffer * coarse_h = NULL;
        int64_t coarse_n = 0;
        status = encoder_s2c_exec(
            backend,
            &weights->down_blocks[level],
            cur_coords,
            cur_rulebook,
            cur_h,
            cur_n,
            &coarse_coords,
            &coarse_rulebook,
            &coarse_h,
            &coarse_n,
            &down_guides.levels[n_guides]);
        if (status == TRELLIS_STATUS_OK) {
            ++n_guides;
            free(cur_coords);
            ops->free_rulebook(backend, cur_rulebook);
            ops->free_buffer(backend, cur_h);
            cur_coords = coarse_coords;
            cur_rulebook = coarse_rulebook;
            cur_h = coarse_h;
            cur_n = coarse_n;
            cur_channels = weights->channels[level + 1];
        }
    }

    if (status == TRELLIS_STATUS_OK && levels_to_run == levels) {
        trellis_sparse_buffer * norm = NULL;
        trellis_sparse_buffer * posterior = NULL;
        float * posterior_host = NULL;
        const int posterior_channels = 2 * weights->latent_channels;
        status = ops->alloc_f32(
            backend, (size_t) cur_n * (size_t) cur_channels, &norm);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->alloc_f32(
                backend, (size_t) cur_n * (size_t) posterior_channels, &posterior);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->row_norm(
                backend, cur_h, NULL, NULL, norm, cur_n, cur_channels, 1e-5f);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->linear(
                backend,
                norm,
                weights->to_latent_w,
                weights->to_latent_b,
                posterior,
                cur_n,
                cur_channels,
                posterior_channels);
        }
        if (status == TRELLIS_STATUS_OK) {
            posterior_host = (float *) malloc(
                (size_t) cur_n * (size_t) posterior_channels * sizeof(float));
            host_feats = (float *) malloc(
                (size_t) cur_n * (size_t) weights->latent_channels * sizeof(float));
            if (posterior_host == NULL || host_feats == NULL) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
            }
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->download_f32(
                backend,
                posterior,
                posterior_host,
                (size_t) cur_n * (size_t) posterior_channels);
        }
        if (status == TRELLIS_STATUS_OK) {
            for (int64_t row = 0; row < cur_n; ++row) {
                memcpy(
                    host_feats + row * weights->latent_channels,
                    posterior_host + row * posterior_channels,
                    (size_t) weights->latent_channels * sizeof(float));
            }
            cur_channels = weights->latent_channels;
        }
        free(posterior_host);
        ops->free_buffer(backend, norm);
        ops->free_buffer(backend, posterior);
    } else if (status == TRELLIS_STATUS_OK) {
        host_feats = (float *) malloc(
            (size_t) cur_n * (size_t) cur_channels * sizeof(float));
        if (host_feats == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            status = ops->download_f32(
                backend, cur_h, host_feats, (size_t) cur_n * (size_t) cur_channels);
        }
    }

    if (status == TRELLIS_STATUS_OK) {
        host_coords = (int32_t *) malloc((size_t) cur_n * 4u * sizeof(int32_t));
        if (host_coords == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            memcpy(host_coords, cur_coords, (size_t) cur_n * 4u * sizeof(int32_t));
        }
    }
    if (status == TRELLIS_STATUS_OK && options->return_subs != NULL) {
        encoder_reverse_guides(&down_guides, n_guides, options->return_subs);
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_out = host_coords;
        *feats_out = host_feats;
        *n_out = cur_n;
        *channels_out = cur_channels;
        host_coords = NULL;
        host_feats = NULL;
    }

cleanup:
    free(host_coords);
    free(host_feats);
    trellis_sparse_c2s_guides_free(&down_guides);
    if (cur_rulebook != NULL && backend != NULL && backend->ops != NULL) {
        backend->ops->free_rulebook(backend, cur_rulebook);
    }
    if (cur_h != NULL && backend != NULL && backend->ops != NULL) {
        backend->ops->free_buffer(backend, cur_h);
    }
    free(cur_coords);
    if (owns_backend) {
        trellis_sparse_backend_destroy(backend);
    }
    if (status != TRELLIS_STATUS_OK && options->return_subs != NULL) {
        trellis_sparse_c2s_guides_free(options->return_subs);
    }
    return status;
}
