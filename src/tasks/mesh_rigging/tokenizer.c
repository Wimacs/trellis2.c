#include "tokenizer.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

typedef struct skeleton_scan {
    trellis_tokenskin_grammar_state state;
    size_t bones;
    size_t eos_index;
    bool has_eos;
    bool branch_parent_triple;
} skeleton_scan;

static bool eos_mode_is_valid(trellis_tokenskin_eos_mode mode) {
    return mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ||
        mode == TRELLIS_TOKENSKIN_EOS_CORRECTED;
}

static bool token_is_coordinate(int32_t token) {
    return token >= 0 && token < TRELLIS_TOKENSKIN_NUM_DISCRETE;
}

static bool token_is_part(int32_t token) {
    return token == TRELLIS_TOKENSKIN_TOKEN_SPRING ||
        token == TRELLIS_TOKENSKIN_TOKEN_BODY ||
        token == TRELLIS_TOKENSKIN_TOKEN_HAND;
}

static bool token_is_class(int32_t token) {
    return token == TRELLIS_TOKENSKIN_TOKEN_CLASS_NONE ||
        token == TRELLIS_TOKENSKIN_TOKEN_CLASS_RIGNET ||
        token == TRELLIS_TOKENSKIN_TOKEN_CLASS_VROID ||
        token == TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION;
}

static trellis_status skeleton_consume(
    skeleton_scan * scan,
    int32_t token,
    size_t index) {
    if (scan == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;

    const trellis_tokenskin_grammar_state previous = scan->state;
    switch (previous) {
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BOS:
            if (token != TRELLIS_TOKENSKIN_TOKEN_BOS) return TRELLIS_STATUS_PARSE_ERROR;
            scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_CLASS_PART_OR_JOINT;
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_CLASS_PART_OR_JOINT:
            if (token_is_coordinate(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2;
            } else if (token_is_class(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT;
            } else if (token_is_part(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT;
            } else {
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT:
            if (token_is_coordinate(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2;
            } else if (token_is_part(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT;
            } else if (token == TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_SKELETON_COMPLETE;
                scan->has_eos = true;
                scan->eos_index = index;
            } else {
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2:
            if (!token_is_coordinate(token)) return TRELLIS_STATUS_PARSE_ERROR;
            scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_3;
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_3:
            if (!token_is_coordinate(token)) return TRELLIS_STATUS_PARSE_ERROR;
            if (scan->branch_parent_triple) {
                scan->branch_parent_triple = false;
            } else {
                ++scan->bones;
            }
            scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BRANCH_PART_OR_JOINT;
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BRANCH_PART_OR_JOINT:
            if (token == TRELLIS_TOKENSKIN_TOKEN_BRANCH) {
                scan->branch_parent_triple = true;
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT;
            } else if (token_is_coordinate(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2;
            } else if (token_is_part(token)) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT;
            } else if (token == TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS) {
                scan->state = TRELLIS_TOKENSKIN_GRAMMAR_SKELETON_COMPLETE;
                scan->has_eos = true;
                scan->eos_index = index;
            } else {
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT:
            if (!token_is_coordinate(token)) return TRELLIS_STATUS_PARSE_ERROR;
            scan->state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2;
            break;

        case TRELLIS_TOKENSKIN_GRAMMAR_SKELETON_COMPLETE:
        case TRELLIS_TOKENSKIN_GRAMMAR_SEQUENCE_COMPLETE:
        case TRELLIS_TOKENSKIN_GRAMMAR_INVALID:
        default:
            return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status scan_skeleton(
    const int32_t * ids,
    size_t count,
    skeleton_scan * scan_out) {
    if ((ids == NULL && count != 0u) || scan_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    skeleton_scan scan;
    memset(&scan, 0, sizeof(scan));
    scan.state = TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BOS;
    scan.eos_index = SIZE_MAX;

    for (size_t i = 0; i < count; ++i) {
        const trellis_status status = skeleton_consume(&scan, ids[i], i);
        if (status != TRELLIS_STATUS_OK) return status;
        if (scan.has_eos) break;
    }
    *scan_out = scan;
    return TRELLIS_STATUS_OK;
}

static void mask_coordinates(uint8_t * mask) {
    memset(mask, 1, TRELLIS_TOKENSKIN_NUM_DISCRETE * sizeof(mask[0]));
}

static void mask_parts(uint8_t * mask) {
    mask[TRELLIS_TOKENSKIN_TOKEN_SPRING] = 1;
    mask[TRELLIS_TOKENSKIN_TOKEN_BODY] = 1;
    mask[TRELLIS_TOKENSKIN_TOKEN_HAND] = 1;
}

static void mask_classes(uint8_t * mask) {
    mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_NONE] = 1;
    mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_RIGNET] = 1;
    mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_VROID] = 1;
    mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION] = 1;
}

static trellis_status mask_skeleton_state(
    trellis_tokenskin_grammar_state state,
    uint8_t * mask) {
    switch (state) {
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BOS:
            mask[TRELLIS_TOKENSKIN_TOKEN_BOS] = 1;
            return TRELLIS_STATUS_OK;
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_CLASS_PART_OR_JOINT:
            mask_coordinates(mask);
            mask_classes(mask);
            mask_parts(mask);
            return TRELLIS_STATUS_OK;
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT:
            mask_coordinates(mask);
            mask_parts(mask);
            mask[TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS] = 1;
            return TRELLIS_STATUS_OK;
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2:
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_3:
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT:
            mask_coordinates(mask);
            return TRELLIS_STATUS_OK;
        case TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BRANCH_PART_OR_JOINT:
            mask_coordinates(mask);
            mask_parts(mask);
            mask[TRELLIS_TOKENSKIN_TOKEN_BRANCH] = 1;
            mask[TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS] = 1;
            return TRELLIS_STATUS_OK;
        default:
            return TRELLIS_STATUS_PARSE_ERROR;
    }
}

static void fill_mask_info(
    trellis_tokenskin_mask_info * info,
    const skeleton_scan * scan) {
    if (info == NULL || scan == NULL) return;
    memset(info, 0, sizeof(*info));
    info->state = scan->state;
    info->bone_count = scan->bones;
    info->skeleton_eos_index = scan->eos_index;
    info->has_skeleton_eos = scan->has_eos;
}

trellis_status trellis_tokenskin_tokenizer_next_mask(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_eos_mode eos_mode,
    uint8_t * mask,
    size_t mask_count,
    trellis_tokenskin_mask_info * info_out) {
    if ((ids == NULL && count != 0u) || mask == NULL ||
        mask_count < TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE ||
        !eos_mode_is_valid(eos_mode)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mask, 0, mask_count * sizeof(mask[0]));

    skeleton_scan scan;
    trellis_status status = scan_skeleton(ids, count, &scan);
    if (status != TRELLIS_STATUS_OK) return status;
    fill_mask_info(info_out, &scan);

    if (!scan.has_eos) {
        return mask_skeleton_state(scan.state, mask);
    }
    if (scan.bones == 0u) return TRELLIS_STATUS_PARSE_ERROR;

    const size_t skin_begin = scan.eos_index + 1u;
    bool has_global_eos = false;
    size_t skin_count = 0u;
    for (size_t i = skin_begin; i < count; ++i) {
        const int32_t token = ids[i];
        if (token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            if (i + 1u != count) return TRELLIS_STATUS_PARSE_ERROR;
            has_global_eos = true;
            break;
        }
        const int32_t skin_token_begin =
            eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
            TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS :
            TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE;
        if (token < skin_token_begin ||
            token >= TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        ++skin_count;
    }

    const size_t true_skin_count = scan.bones * TRELLIS_TOKENSKIN_TOKENS_PER_SKIN;
    const size_t eos_after = eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
        true_skin_count - 1u : true_skin_count;
    if (info_out != NULL) {
        info_out->skin_token_count = skin_count;
        info_out->has_global_eos = has_global_eos;
    }

    if (has_global_eos) {
        if (skin_count != eos_after) return TRELLIS_STATUS_PARSE_ERROR;
        if (info_out != NULL) info_out->state = TRELLIS_TOKENSKIN_GRAMMAR_SEQUENCE_COMPLETE;
        return TRELLIS_STATUS_OK;
    }

    if (skin_count == eos_after) {
        mask[TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS] = 1;
        if (info_out != NULL) info_out->next_is_global_eos = true;
    } else {
        const size_t skin_token_begin =
            eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
            TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS :
            TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE;
        memset(
            mask + skin_token_begin,
            1,
            (size_t) (TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS -
                skin_token_begin));
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tokenskin_tokenizer_bones_in_sequence(
    const int32_t * ids,
    size_t count,
    size_t * bones_out,
    bool * has_skeleton_eos_out,
    size_t * skeleton_eos_index_out,
    trellis_tokenskin_grammar_state * state_out) {
    if (bones_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    skeleton_scan scan;
    const trellis_status status = scan_skeleton(ids, count, &scan);
    if (status != TRELLIS_STATUS_OK) return status;
    *bones_out = scan.bones;
    if (has_skeleton_eos_out != NULL) *has_skeleton_eos_out = scan.has_eos;
    if (skeleton_eos_index_out != NULL) *skeleton_eos_index_out = scan.eos_index;
    if (state_out != NULL) *state_out = scan.state;
    return TRELLIS_STATUS_OK;
}

float trellis_tokenskin_tokenizer_undiscretize(int32_t token) {
    if (!token_is_coordinate(token)) return 0.0f;
    return ((float) token + 0.5f) /
        (float) TRELLIS_TOKENSKIN_NUM_DISCRETE * 2.0f - 1.0f;
}

static bool read_point(
    const int32_t * ids,
    size_t begin,
    size_t limit,
    float point[3]) {
    if (begin > limit || limit - begin < 3u) return false;
    for (size_t axis = 0; axis < 3u; ++axis) {
        if (!token_is_coordinate(ids[begin + axis])) return false;
        point[axis] = trellis_tokenskin_tokenizer_undiscretize(ids[begin + axis]);
    }
    return true;
}

void trellis_tokenskin_skeleton_free(trellis_tokenskin_skeleton * skeleton) {
    if (skeleton == NULL) return;
    free(skeleton->joints_xyz);
    free(skeleton->parent_joints_xyz);
    free(skeleton->parents);
    memset(skeleton, 0, sizeof(*skeleton));
}

trellis_status trellis_tokenskin_tokenizer_decode_skeleton(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_skeleton * skeleton_out) {
    if (ids == NULL || count == 0u || skeleton_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    skeleton_scan scan;
    trellis_status status = scan_skeleton(ids, count, &scan);
    if (status != TRELLIS_STATUS_OK) return status;
    if (!scan.has_eos || scan.bones == 0u || scan.bones > SIZE_MAX / (3u * sizeof(float))) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    trellis_tokenskin_skeleton decoded;
    memset(&decoded, 0, sizeof(decoded));
    decoded.class_token_id = TRELLIS_TOKENSKIN_TOKEN_CLASS_NONE;
    decoded.joint_count = scan.bones;
    decoded.joints_xyz = (float *) malloc(scan.bones * 3u * sizeof(float));
    decoded.parent_joints_xyz = (float *) malloc(scan.bones * 3u * sizeof(float));
    decoded.parents = (int32_t *) malloc(scan.bones * sizeof(int32_t));
    if (decoded.joints_xyz == NULL || decoded.parent_joints_xyz == NULL ||
        decoded.parents == NULL) {
        trellis_tokenskin_skeleton_free(&decoded);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    size_t cursor = 1u;
    size_t joint = 0u;
    bool branch = false;
    bool have_last_joint = false;
    float last_joint[3] = {0.0f, 0.0f, 0.0f};
    while (cursor < scan.eos_index) {
        const int32_t token = ids[cursor];
        if (token_is_coordinate(token)) {
            float parent_point[3];
            float current_point[3];
            if (branch) {
                if (!read_point(ids, cursor, scan.eos_index, parent_point) ||
                    !read_point(ids, cursor + 3u, scan.eos_index, current_point)) {
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
                cursor += 6u;
            } else {
                if (!read_point(ids, cursor, scan.eos_index, current_point)) {
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
                if (have_last_joint) {
                    memcpy(parent_point, last_joint, sizeof(parent_point));
                } else {
                    memcpy(parent_point, current_point, sizeof(parent_point));
                }
                cursor += 3u;
            }
            if (joint >= scan.bones) {
                status = TRELLIS_STATUS_PARSE_ERROR;
                goto cleanup;
            }
            memcpy(decoded.joints_xyz + joint * 3u, current_point, sizeof(current_point));
            memcpy(
                decoded.parent_joints_xyz + joint * 3u,
                parent_point,
                sizeof(parent_point));
            memcpy(last_joint, current_point, sizeof(last_joint));
            have_last_joint = true;
            branch = false;
            ++joint;
        } else if (token == TRELLIS_TOKENSKIN_TOKEN_BRANCH) {
            branch = true;
            have_last_joint = false;
            ++cursor;
        } else if (token_is_part(token)) {
            ++cursor;
        } else if (token_is_class(token)) {
            decoded.class_token_id = token;
            ++cursor;
        } else {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
    }
    if (joint != scan.bones || branch) {
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    for (size_t i = 0; i < decoded.joint_count; ++i) {
        float * resolved_parent = decoded.parent_joints_xyz + i * 3u;
        if (i == 0u) {
            decoded.parents[i] = -1;
            memcpy(resolved_parent, decoded.joints_xyz, 3u * sizeof(float));
            continue;
        }
        const float requested_parent[3] = {
            resolved_parent[0], resolved_parent[1], resolved_parent[2]
        };
        float best_distance = FLT_MAX;
        size_t best_parent = SIZE_MAX;
        for (size_t j = i; j-- > 0u;) {
            const float * candidate = decoded.joints_xyz + j * 3u;
            const float dx = candidate[0] - requested_parent[0];
            const float dy = candidate[1] - requested_parent[1];
            const float dz = candidate[2] - requested_parent[2];
            const float distance = dx * dx + dy * dy + dz * dz;
            if (distance < best_distance) {
                best_distance = distance;
                best_parent = j;
            }
        }
        if (best_parent == SIZE_MAX || best_parent > INT32_MAX) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        decoded.parents[i] = (int32_t) best_parent;
        memcpy(
            resolved_parent,
            decoded.joints_xyz + best_parent * 3u,
            3u * sizeof(float));
    }

    *skeleton_out = decoded;
    return TRELLIS_STATUS_OK;

cleanup:
    trellis_tokenskin_skeleton_free(&decoded);
    return status;
}

trellis_status trellis_tokenskin_tokenizer_extract_skin_indices(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_eos_mode eos_mode,
    int32_t * indices_out,
    size_t indices_capacity,
    size_t * indices_count_out) {
    if (ids == NULL || indices_out == NULL || indices_count_out == NULL ||
        !eos_mode_is_valid(eos_mode)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    skeleton_scan scan;
    trellis_status status = scan_skeleton(ids, count, &scan);
    if (status != TRELLIS_STATUS_OK) return status;
    if (!scan.has_eos || scan.bones == 0u ||
        scan.bones > SIZE_MAX / TRELLIS_TOKENSKIN_TOKENS_PER_SKIN) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const size_t required = scan.bones * TRELLIS_TOKENSKIN_TOKENS_PER_SKIN;
    if (indices_capacity < required) return TRELLIS_STATUS_INVALID_ARGUMENT;
    const size_t skin_begin = scan.eos_index + 1u;

    if (eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT) {
        if (skin_begin > count || count - skin_begin != required ||
            ids[count - 1u] != TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    } else {
        if (skin_begin > count || count - skin_begin != required + 1u ||
            ids[skin_begin + required] != TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }

    for (size_t i = 0; i < required; ++i) {
        const int32_t token = ids[skin_begin + i];
        const bool official_alias =
            eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT &&
            i + 1u == required &&
            token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS;
        const int32_t skin_token_begin =
            eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
            TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS :
            TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE;
        if (!official_alias &&
            (token < skin_token_begin ||
             token >= TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        indices_out[i] = token - TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE;
    }
    *indices_count_out = required;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tokenskin_tokenizer_fsq_code(
    int32_t fsq_index,
    trellis_tokenskin_eos_mode eos_mode,
    float code_out[TRELLIS_TOKENSKIN_FSQ_CODE_DIM]) {
    if (code_out == NULL || !eos_mode_is_valid(eos_mode)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int32_t minimum = eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
        TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS -
            TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE :
        0;
    const int32_t maximum = eos_mode == TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT ?
        TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE :
        TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE - 1;
    if (fsq_index < minimum || fsq_index > maximum) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    int32_t basis = 1;
    for (size_t i = 0; i < TRELLIS_TOKENSKIN_FSQ_CODE_DIM; ++i) {
        int32_t quotient = fsq_index / basis;
        if (fsq_index % basis < 0) --quotient;
        int32_t digit = quotient % 8;
        if (digit < 0) digit += 8;
        code_out[i] = ((float) digit - 4.0f) / 4.0f;
        basis *= 8;
    }
    return TRELLIS_STATUS_OK;
}
