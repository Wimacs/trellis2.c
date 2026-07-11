#include "generation.h"

#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
    TOKENSKIN_START_TOKENS = 2,
    TOKENSKIN_QWEN_PREFIX_TOKENS =
        TOKENSKIN_MESH_TOKENS + TOKENSKIN_START_TOKENS,
};

static trellis_status read_embedding_row(
    const struct ggml_tensor * embedding,
    int32_t token,
    float output[TOKENSKIN_QWEN_HIDDEN]) {
    if (embedding == NULL || token < 0 || token >= TOKENSKIN_QWEN_VOCAB ||
        embedding->ne[0] != TOKENSKIN_QWEN_HIDDEN ||
        embedding->ne[1] != TOKENSKIN_QWEN_VOCAB) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t offset = (size_t) token * embedding->nb[1];
    if (embedding->type == GGML_TYPE_BF16) {
        ggml_bf16_t temporary[TOKENSKIN_QWEN_HIDDEN];
        ggml_backend_tensor_get(
            embedding, temporary, offset, sizeof(temporary));
        ggml_bf16_to_fp32_row(
            temporary, output, TOKENSKIN_QWEN_HIDDEN);
        return TRELLIS_STATUS_OK;
    }
    if (embedding->type == GGML_TYPE_F16) {
        ggml_fp16_t temporary[TOKENSKIN_QWEN_HIDDEN];
        ggml_backend_tensor_get(
            embedding, temporary, offset, sizeof(temporary));
        for (int i = 0; i < TOKENSKIN_QWEN_HIDDEN; ++i) {
            output[i] = ggml_fp16_to_fp32(temporary[i]);
        }
        return TRELLIS_STATUS_OK;
    }
    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(
            embedding,
            output,
            offset,
            TOKENSKIN_QWEN_HIDDEN * sizeof(float));
        return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_NOT_IMPLEMENTED;
}

static trellis_status build_prefix(
    const tokenskin_runtime_model * model,
    const float * mesh_embeddings,
    float ** prefix_out) {
    if (model == NULL || mesh_embeddings == NULL || prefix_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t count =
        (size_t) TOKENSKIN_QWEN_PREFIX_TOKENS * TOKENSKIN_QWEN_HIDDEN;
    float * prefix = (float *) malloc(count * sizeof(float));
    if (prefix == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    memcpy(
        prefix,
        mesh_embeddings,
        (size_t) TOKENSKIN_MESH_TOKENS * TOKENSKIN_QWEN_HIDDEN * sizeof(float));
    trellis_status status = read_embedding_row(
        model->qwen.token_embedding,
        TRELLIS_TOKENSKIN_TOKEN_BOS,
        prefix + (size_t) TOKENSKIN_MESH_TOKENS * TOKENSKIN_QWEN_HIDDEN);
    if (status == TRELLIS_STATUS_OK) {
        status = read_embedding_row(
            model->qwen.token_embedding,
            TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION,
            prefix + (size_t) (TOKENSKIN_MESH_TOKENS + 1) *
                TOKENSKIN_QWEN_HIDDEN);
    }
    if (status != TRELLIS_STATUS_OK) {
        free(prefix);
        return status;
    }
    *prefix_out = prefix;
    return TRELLIS_STATUS_OK;
}

static trellis_status generate_single_sequence(
    const tokenskin_runtime_model * model,
    const float * mesh_embeddings_512x896,
    const trellis_tokenskin_rig_options * options,
    int32_t ** sequence_out,
    size_t * sequence_count_out) {
    if (model == NULL || model->backend.backend == NULL ||
        mesh_embeddings_512x896 == NULL || options == NULL ||
        options->max_length <= TOKENSKIN_QWEN_PREFIX_TOKENS ||
        options->max_length > TOKENSKIN_QWEN_MAX_POSITIONS ||
        options->top_k < 0 || options->top_p <= 0.0f || options->top_p > 1.0f ||
        options->temperature < 0.0f || options->repetition_penalty <= 0.0f ||
        sequence_out == NULL || sequence_count_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *sequence_out = NULL;
    *sequence_count_out = 0;
    trellis_status status = TRELLIS_STATUS_ERROR;
    float * prefix = NULL;
    float * logits = NULL;
    uint8_t * grammar_mask = NULL;
    int32_t * sequence = NULL;
    tokenskin_qwen_executor * executor = NULL;
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));

    status = build_prefix(model, mesh_embeddings_512x896, &prefix);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    logits = (float *) malloc(TOKENSKIN_QWEN_VOCAB * sizeof(float));
    grammar_mask = (uint8_t *) malloc(TOKENSKIN_QWEN_VOCAB * sizeof(uint8_t));
    sequence = (int32_t *) malloc((size_t) options->max_length * sizeof(int32_t));
    if (logits == NULL || grammar_mask == NULL || sequence == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    sequence[0] = TRELLIS_TOKENSKIN_TOKEN_BOS;
    sequence[1] = TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION;
    size_t sequence_count = TOKENSKIN_START_TOKENS;

    status = tokenskin_qwen_executor_create(
        &model->backend,
        &model->qwen,
        &model->attention_policy,
        options->max_length,
        &executor);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = trellis_tokenskin_sampler_init(
        &sampler, TOKENSKIN_QWEN_VOCAB, options->seed);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = tokenskin_qwen_prefill(
        executor, prefix, TOKENSKIN_QWEN_PREFIX_TOKENS, logits);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    trellis_tokenskin_sampler_options sampling =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    sampling.greedy = options->temperature == 0.0f;
    sampling.temperature = options->temperature;
    sampling.repetition_penalty = options->repetition_penalty;
    sampling.top_k = (size_t) options->top_k;
    sampling.top_p = options->top_p;
    const trellis_tokenskin_eos_mode eos_mode = options->official_eos_compat ?
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT :
        TRELLIS_TOKENSKIN_EOS_CORRECTED;

    while (tokenskin_qwen_executor_length(executor) < options->max_length) {
        trellis_tokenskin_mask_info info;
        status = trellis_tokenskin_tokenizer_next_mask(
            sequence,
            sequence_count,
            eos_mode,
            grammar_mask,
            TOKENSKIN_QWEN_VOCAB,
            &info);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        int32_t token = -1;
        status = trellis_tokenskin_sampler_sample(
            &sampler,
            logits,
            grammar_mask,
            TOKENSKIN_QWEN_VOCAB,
            sequence + TOKENSKIN_START_TOKENS,
            sequence_count - TOKENSKIN_START_TOKENS,
            &sampling,
            &token);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        sequence[sequence_count++] = token;
        if ((sequence_count - TOKENSKIN_START_TOKENS) % 32u == 0u ||
            token == TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS ||
            token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            TRELLIS_INFO(
                "TokenSkin generation: tokens=%zu bones=%zu state=%d token=%d",
                sequence_count - TOKENSKIN_START_TOKENS,
                info.bone_count,
                (int) info.state,
                token);
        }
        if (token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            *sequence_out = sequence;
            *sequence_count_out = sequence_count;
            sequence = NULL;
            status = TRELLIS_STATUS_OK;
            goto cleanup;
        }
        status = tokenskin_qwen_decode(executor, token, logits);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    TRELLIS_ERROR(
        "TokenSkin generation reached max_length=%d before global EOS",
        options->max_length);
    status = TRELLIS_STATUS_PARSE_ERROR;

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
    tokenskin_qwen_executor_free(executor);
    free(sequence);
    free(grammar_mask);
    free(logits);
    free(prefix);
    return status;
}

enum {
    TOKENSKIN_MAX_BEAMS = TRELLIS_TOKENSKIN_MAX_BEAMS,
};

typedef struct tokenskin_beam_candidate {
    size_t parent;
    int32_t token;
    float score;
} tokenskin_beam_candidate;

static uint64_t beam_rng_next(uint64_t * state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static double beam_rng_unit(uint64_t * state) {
    return (double) (beam_rng_next(state) >> 11u) *
        (1.0 / 9007199254740992.0);
}

static int candidate_compare_descending(const void * left, const void * right) {
    const tokenskin_beam_candidate * a =
        (const tokenskin_beam_candidate *) left;
    const tokenskin_beam_candidate * b =
        (const tokenskin_beam_candidate *) right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->parent < b->parent) return -1;
    if (a->parent > b->parent) return 1;
    if (a->token < b->token) return -1;
    if (a->token > b->token) return 1;
    return 0;
}

static size_t sample_flat_candidates_without_replacement(
    float * scores,
    size_t score_count,
    size_t vocab_size,
    size_t requested,
    uint64_t * rng_state,
    tokenskin_beam_candidate * candidates) {
    size_t selected_count = 0;
    while (selected_count < requested) {
        size_t positive_infinities = 0;
        float maximum = -INFINITY;
        size_t finite_count = 0;
        for (size_t i = 0; i < score_count; ++i) {
            if (isinf(scores[i]) && scores[i] > 0.0f) {
                ++positive_infinities;
            } else if (isfinite(scores[i])) {
                ++finite_count;
                if (scores[i] > maximum) maximum = scores[i];
            }
        }
        if (positive_infinities == 0 && finite_count == 0) break;

        size_t selected = SIZE_MAX;
        if (positive_infinities != 0) {
            size_t rank = (size_t) (beam_rng_unit(rng_state) *
                (double) positive_infinities);
            if (rank >= positive_infinities) rank = positive_infinities - 1u;
            for (size_t i = 0; i < score_count; ++i) {
                if (!(isinf(scores[i]) && scores[i] > 0.0f)) continue;
                if (rank-- == 0u) {
                    selected = i;
                    break;
                }
            }
        } else {
            double sum = 0.0;
            for (size_t i = 0; i < score_count; ++i) {
                if (isfinite(scores[i])) {
                    sum += exp((double) scores[i] - (double) maximum);
                }
            }
            if (!(sum > 0.0) || !isfinite(sum)) break;
            const double target = beam_rng_unit(rng_state) * sum;
            double cumulative = 0.0;
            size_t last = SIZE_MAX;
            for (size_t i = 0; i < score_count; ++i) {
                if (!isfinite(scores[i])) continue;
                cumulative += exp((double) scores[i] - (double) maximum);
                last = i;
                if (target < cumulative) {
                    selected = i;
                    break;
                }
            }
            if (selected == SIZE_MAX) selected = last;
        }
        if (selected == SIZE_MAX) break;
        candidates[selected_count].parent = selected / vocab_size;
        candidates[selected_count].token = (int32_t) (selected % vocab_size);
        candidates[selected_count].score = scores[selected];
        scores[selected] = -INFINITY;
        ++selected_count;
    }
    return selected_count;
}

static trellis_status log_softmax_f32(
    const float * logits,
    size_t count,
    float * log_probabilities) {
    float maximum = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        if (isfinite(logits[i]) && logits[i] > maximum) maximum = logits[i];
    }
    if (!isfinite(maximum)) return TRELLIS_STATUS_PARSE_ERROR;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (isfinite(logits[i])) {
            sum += exp((double) logits[i] - (double) maximum);
        }
    }
    if (!(sum > 0.0) || !isfinite(sum)) return TRELLIS_STATUS_PARSE_ERROR;
    const float normalization = maximum + (float) log(sum);
    for (size_t i = 0; i < count; ++i) {
        log_probabilities[i] = isfinite(logits[i]) ?
            logits[i] - normalization : -INFINITY;
    }
    return TRELLIS_STATUS_OK;
}

static void insert_finished_beam(
    int32_t * finished_sequences,
    size_t * finished_counts,
    float * finished_scores,
    size_t * finished_count,
    size_t beam_capacity,
    size_t sequence_capacity,
    const int32_t * parent_sequence,
    size_t parent_count,
    int32_t eos_token,
    float normalized_score) {
    size_t position = 0;
    while (position < *finished_count &&
           finished_scores[position] >= normalized_score) {
        ++position;
    }
    if (position >= beam_capacity) return;
    size_t new_count = *finished_count < beam_capacity ?
        *finished_count + 1u : *finished_count;
    for (size_t i = new_count; i > position + 1u; --i) {
        finished_scores[i - 1u] = finished_scores[i - 2u];
        finished_counts[i - 1u] = finished_counts[i - 2u];
        memcpy(
            finished_sequences + (i - 1u) * sequence_capacity,
            finished_sequences + (i - 2u) * sequence_capacity,
            finished_counts[i - 2u] * sizeof(int32_t));
    }
    memcpy(
        finished_sequences + position * sequence_capacity,
        parent_sequence,
        parent_count * sizeof(int32_t));
    finished_sequences[position * sequence_capacity + parent_count] = eos_token;
    finished_counts[position] = parent_count + 1u;
    finished_scores[position] = normalized_score;
    *finished_count = new_count;
}

static trellis_status generate_beam_sample_sequence(
    const tokenskin_runtime_model * model,
    const float * mesh_embeddings,
    const trellis_tokenskin_rig_options * options,
    int32_t ** sequence_out,
    size_t * sequence_count_out) {
    const size_t beams = (size_t) options->num_beams;
    const size_t vocab = TOKENSKIN_QWEN_VOCAB;
    const size_t sequence_capacity = (size_t) options->max_length;
    if (beams < 2u || beams > TOKENSKIN_MAX_BEAMS ||
        beams > SIZE_MAX / 2u || beams * 2u > SIZE_MAX / vocab ||
        beams * 2u * vocab > SIZE_MAX / sizeof(float) ||
        beams * 2u > SIZE_MAX / sequence_capacity ||
        beams * 2u * sequence_capacity > SIZE_MAX / sizeof(int32_t)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    float * prefix = NULL;
    tokenskin_qwen_executor ** executors = (tokenskin_qwen_executor **) calloc(
        beams * 2u, sizeof(*executors));
    float * logits = (float *) malloc(beams * 2u * vocab * sizeof(float));
    int32_t * sequences = (int32_t *) malloc(
        beams * 2u * sequence_capacity * sizeof(int32_t));
    size_t * counts = (size_t *) calloc(beams * 2u, sizeof(size_t));
    float * beam_scores = (float *) malloc(beams * 2u * sizeof(float));
    float * processed = (float *) malloc(beams * vocab * sizeof(float));
    float * log_probabilities = (float *) malloc(vocab * sizeof(float));
    uint8_t * grammar_mask = (uint8_t *) malloc(vocab * sizeof(uint8_t));
    tokenskin_beam_candidate * candidates =
        (tokenskin_beam_candidate *) malloc(beams * 2u * sizeof(*candidates));
    int32_t * finished_sequences = (int32_t *) malloc(
        beams * sequence_capacity * sizeof(int32_t));
    size_t * finished_counts = (size_t *) calloc(beams, sizeof(size_t));
    float * finished_scores = (float *) malloc(beams * sizeof(float));
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    if (executors == NULL || logits == NULL || sequences == NULL || counts == NULL ||
        beam_scores == NULL || processed == NULL || log_probabilities == NULL ||
        grammar_mask == NULL || candidates == NULL || finished_sequences == NULL ||
        finished_counts == NULL || finished_scores == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t i = 0; i < beams * 2u; ++i) beam_scores[i] = -INFINITY;
    for (size_t i = 0; i < beams; ++i) finished_scores[i] = -INFINITY;
    status = build_prefix(model, mesh_embeddings, &prefix);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    for (size_t i = 0; i < beams * 2u; ++i) {
        status = tokenskin_qwen_executor_create(
            &model->backend,
            &model->qwen,
            &model->attention_policy,
            options->max_length,
            &executors[i]);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }
    status = trellis_tokenskin_sampler_init(&sampler, vocab, options->seed);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    sequences[0] = TRELLIS_TOKENSKIN_TOKEN_BOS;
    sequences[1] = TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION;
    counts[0] = TOKENSKIN_START_TOKENS;
    beam_scores[0] = 0.0f;
    status = tokenskin_qwen_prefill(
        executors[0], prefix, TOKENSKIN_QWEN_PREFIX_TOKENS, logits);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    trellis_tokenskin_sampler_options sampling =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    sampling.temperature = options->temperature;
    sampling.repetition_penalty = options->repetition_penalty;
    sampling.top_k = (size_t) options->top_k;
    sampling.top_p = options->top_p;
    sampling.min_tokens_to_keep = 2u;
    const trellis_tokenskin_eos_mode eos_mode = options->official_eos_compat ?
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT :
        TRELLIS_TOKENSKIN_EOS_CORRECTED;
    uint64_t rng_state = options->seed;
    size_t active_count = 1u;
    size_t finished_count = 0u;
    int current_bank = 0;

    while (active_count != 0u &&
           tokenskin_qwen_executor_length(
               executors[(size_t) current_bank * beams]) < options->max_length) {
        for (size_t i = 0; i < beams * vocab; ++i) processed[i] = -INFINITY;
        for (size_t parent = 0; parent < active_count; ++parent) {
            const size_t slot = (size_t) current_bank * beams + parent;
            const int32_t * parent_sequence =
                sequences + slot * sequence_capacity;
            status = log_softmax_f32(
                logits + slot * vocab, vocab, log_probabilities);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            status = trellis_tokenskin_tokenizer_next_mask(
                parent_sequence,
                counts[slot],
                eos_mode,
                grammar_mask,
                vocab,
                NULL);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            status = trellis_tokenskin_sampler_process_logits(
                &sampler,
                log_probabilities,
                grammar_mask,
                vocab,
                parent_sequence + TOKENSKIN_START_TOKENS,
                counts[slot] - TOKENSKIN_START_TOKENS,
                &sampling,
                processed + parent * vocab,
                vocab);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            for (size_t token = 0; token < vocab; ++token) {
                float * score = &processed[parent * vocab + token];
                if (isfinite(*score)) *score += beam_scores[slot];
            }
        }

        const size_t candidate_count = sample_flat_candidates_without_replacement(
            processed,
            active_count * vocab,
            vocab,
            beams * 2u,
            &rng_state,
            candidates);
        if (candidate_count == 0u) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const size_t finished_eligible = candidate_count < beams ?
            candidate_count : beams;
        for (size_t i = 0; i < finished_eligible; ++i) {
            if (candidates[i].token != TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) continue;
            const size_t parent_slot =
                (size_t) current_bank * beams + candidates[i].parent;
            const size_t generated_length =
                counts[parent_slot] - TOKENSKIN_START_TOKENS + 1u;
            insert_finished_beam(
                finished_sequences,
                finished_counts,
                finished_scores,
                &finished_count,
                beams,
                sequence_capacity,
                sequences + parent_slot * sequence_capacity,
                counts[parent_slot],
                candidates[i].token,
                candidates[i].score / (float) generated_length);
        }
        /* torch.multinomial preserves draw order for the top-num-beams EOS
         * eligibility mask. Running beams are then selected by score. */
        qsort(
            candidates,
            candidate_count,
            sizeof(candidates[0]),
            candidate_compare_descending);

        const int next_bank = 1 - current_bank;
        size_t next_active = 0u;
        for (size_t i = 0; i < candidate_count && next_active < beams; ++i) {
            if (candidates[i].token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) continue;
            const size_t source_slot =
                (size_t) current_bank * beams + candidates[i].parent;
            const size_t destination_slot =
                (size_t) next_bank * beams + next_active;
            if (counts[source_slot] + 1u > sequence_capacity) continue;
            status = tokenskin_qwen_executor_copy_state(
                executors[destination_slot], executors[source_slot]);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            memcpy(
                sequences + destination_slot * sequence_capacity,
                sequences + source_slot * sequence_capacity,
                counts[source_slot] * sizeof(int32_t));
            sequences[destination_slot * sequence_capacity + counts[source_slot]] =
                candidates[i].token;
            counts[destination_slot] = counts[source_slot] + 1u;
            beam_scores[destination_slot] = candidates[i].score;
            status = tokenskin_qwen_decode(
                executors[destination_slot],
                candidates[i].token,
                logits + destination_slot * vocab);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
            ++next_active;
        }
        if (next_active == 0u) {
            if (finished_count != 0u) break;
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        active_count = next_active;
        current_bank = next_bank;
        const size_t best_slot = (size_t) current_bank * beams;
        const size_t generated =
            counts[best_slot] - TOKENSKIN_START_TOKENS;
        if (finished_count >= beams) {
            /* Transformers' current early_stopping=False heuristic compares
             * the worst retained finished score with the best still-running
             * beam at the current generated length (length_penalty=1). */
            const float highest_attainable_score =
                beam_scores[best_slot] / (float) generated;
            if (finished_scores[finished_count - 1u] >=
                highest_attainable_score) {
                break;
            }
        }
        if (generated % 32u == 0u) {
            size_t bones = 0;
            (void) trellis_tokenskin_tokenizer_bones_in_sequence(
                sequences + best_slot * sequence_capacity,
                counts[best_slot],
                &bones,
                NULL,
                NULL,
                NULL);
            TRELLIS_INFO(
                "TokenSkin beam generation: tokens=%zu active=%zu finished=%zu bones=%zu score=%.5f",
                generated,
                active_count,
                finished_count,
                bones,
                beam_scores[best_slot]);
        }
    }
    if (finished_count == 0u) {
        TRELLIS_ERROR(
            "TokenSkin beam generation reached max_length=%d without global EOS",
            options->max_length);
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }
    *sequence_out = (int32_t *) malloc(finished_counts[0] * sizeof(int32_t));
    if (*sequence_out == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(
        *sequence_out,
        finished_sequences,
        finished_counts[0] * sizeof(int32_t));
    *sequence_count_out = finished_counts[0];
    TRELLIS_INFO(
        "TokenSkin beam generation selected score=%.6f tokens=%zu",
        finished_scores[0],
        finished_counts[0] - TOKENSKIN_START_TOKENS);
    status = TRELLIS_STATUS_OK;

cleanup:
    if (status != TRELLIS_STATUS_OK) {
        free(*sequence_out);
        *sequence_out = NULL;
        *sequence_count_out = 0;
    }
    trellis_tokenskin_sampler_free(&sampler);
    if (executors != NULL) {
        for (size_t i = 0; i < beams * 2u; ++i) {
            tokenskin_qwen_executor_free(executors[i]);
        }
    }
    free(prefix);
    free(finished_scores);
    free(finished_counts);
    free(finished_sequences);
    free(candidates);
    free(grammar_mask);
    free(log_probabilities);
    free(processed);
    free(beam_scores);
    free(counts);
    free(sequences);
    free(logits);
    free(executors);
    return status;
}

trellis_status tokenskin_generate_rig_tokens(
    const tokenskin_runtime_model * model,
    const float * mesh_embeddings_512x896,
    const trellis_tokenskin_rig_options * options,
    int32_t ** sequence_out,
    size_t * sequence_count_out) {
    if (model == NULL || model->backend.backend == NULL ||
        mesh_embeddings_512x896 == NULL || options == NULL ||
        options->num_beams <= 0 || options->num_beams > TOKENSKIN_MAX_BEAMS ||
        sequence_out == NULL || sequence_count_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *sequence_out = NULL;
    *sequence_count_out = 0;
    if (options->num_beams == 1 || options->temperature <= 0.0f) {
        return generate_single_sequence(
            model,
            mesh_embeddings_512x896,
            options,
            sequence_out,
            sequence_count_out);
    }
    return generate_beam_sample_sequence(
        model,
        mesh_embeddings_512x896,
        options,
        sequence_out,
        sequence_count_out);
}
