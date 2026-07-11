#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        goto cleanup; \
    } \
} while (0)

static size_t count_mask(const uint8_t * mask, size_t count) {
    size_t result = 0u;
    for (size_t i = 0; i < count; ++i) result += mask[i] != 0u;
    return result;
}

static int nearly_equal(float a, float b) {
    return fabsf(a - b) <= 1e-7f;
}

static void test_grammar_masks(void) {
    uint8_t * mask = (uint8_t *) malloc(TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE);
    trellis_tokenskin_mask_info info;
    const int32_t bos[] = {TRELLIS_TOKENSKIN_TOKEN_BOS};
    const int32_t head[] = {
        TRELLIS_TOKENSKIN_TOKEN_BOS,
        TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION,
    };
    CHECK_TRUE(mask != NULL);

    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        NULL, 0u, TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.state == TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BOS);
    CHECK_TRUE(count_mask(mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE) == 1u);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_BOS] == 1u);

    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        bos, sizeof(bos) / sizeof(bos[0]), TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.state == TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_CLASS_PART_OR_JOINT);
    CHECK_TRUE(count_mask(mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE) == 263u);
    CHECK_TRUE(mask[0] && mask[255]);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_SPRING]);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_HAND]);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_NONE]);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION]);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_BRANCH]);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS]);

    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        head, sizeof(head) / sizeof(head[0]), TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.state == TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT);
    CHECK_TRUE(count_mask(mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE) == 260u);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS]);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_BRANCH]);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION]);

cleanup:
    free(mask);
}

static void test_branch_bones_and_decode(void) {
    const int32_t ids[] = {
        TRELLIS_TOKENSKIN_TOKEN_BOS,
        TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION,
        128, 128, 128,                    /* root */
        160, 128, 128,                    /* chained child of root */
        TRELLIS_TOKENSKIN_TOKEN_BRANCH,
        128, 128, 128,                    /* explicit parent: root */
        128, 160, 128,                    /* branch child */
        TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS,
    };
    size_t bones = 0u;
    size_t eos_index = 0u;
    bool has_eos = false;
    trellis_tokenskin_grammar_state state = TRELLIS_TOKENSKIN_GRAMMAR_INVALID;
    trellis_tokenskin_skeleton skeleton;
    memset(&skeleton, 0, sizeof(skeleton));

    CHECK_TRUE(trellis_tokenskin_tokenizer_bones_in_sequence(
        ids, sizeof(ids) / sizeof(ids[0]), &bones, &has_eos, &eos_index, &state) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(bones == 3u);
    CHECK_TRUE(has_eos);
    CHECK_TRUE(eos_index == sizeof(ids) / sizeof(ids[0]) - 1u);
    CHECK_TRUE(state == TRELLIS_TOKENSKIN_GRAMMAR_SKELETON_COMPLETE);

    CHECK_TRUE(trellis_tokenskin_tokenizer_decode_skeleton(
        ids, sizeof(ids) / sizeof(ids[0]), &skeleton) == TRELLIS_STATUS_OK);
    CHECK_TRUE(skeleton.joint_count == 3u);
    CHECK_TRUE(skeleton.class_token_id == TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION);
    CHECK_TRUE(skeleton.parents[0] == -1);
    CHECK_TRUE(skeleton.parents[1] == 0);
    CHECK_TRUE(skeleton.parents[2] == 0);
    CHECK_TRUE(nearly_equal(skeleton.joints_xyz[0], 0.00390625f));
    CHECK_TRUE(nearly_equal(skeleton.joints_xyz[3], 0.25390625f));
    CHECK_TRUE(nearly_equal(skeleton.joints_xyz[7], 0.25390625f));
    CHECK_TRUE(nearly_equal(skeleton.parent_joints_xyz[3], skeleton.joints_xyz[0]));
    CHECK_TRUE(nearly_equal(skeleton.parent_joints_xyz[6], skeleton.joints_xyz[0]));
    CHECK_TRUE(nearly_equal(skeleton.parent_joints_xyz[7], skeleton.joints_xyz[1]));
    CHECK_TRUE(nearly_equal(skeleton.parent_joints_xyz[8], skeleton.joints_xyz[2]));

cleanup:
    trellis_tokenskin_skeleton_free(&skeleton);
}

static void test_eos_modes_and_alias(void) {
    const int32_t prefix[] = {
        TRELLIS_TOKENSKIN_TOKEN_BOS,
        TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION,
        128, 128, 128,
        TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS,
    };
    int32_t official_pending[sizeof(prefix) / sizeof(prefix[0]) + 3u];
    int32_t official_complete[sizeof(prefix) / sizeof(prefix[0]) + 4u];
    int32_t corrected_pending[sizeof(prefix) / sizeof(prefix[0]) + 3u];
    int32_t corrected_ready[sizeof(prefix) / sizeof(prefix[0]) + 4u];
    int32_t corrected_complete[sizeof(prefix) / sizeof(prefix[0]) + 5u];
    uint8_t * mask = (uint8_t *) malloc(TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE);
    trellis_tokenskin_mask_info info;
    int32_t indices[4] = {0};
    size_t indices_count = 0u;
    float alias_code[5];
    float zero_code[5];
    CHECK_TRUE(mask != NULL);

    memcpy(official_pending, prefix, sizeof(prefix));
    official_pending[6] = 267;
    official_pending[7] = 268;
    official_pending[8] = 269;
    memcpy(corrected_pending, official_pending, sizeof(official_pending));

    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        official_pending, sizeof(official_pending) / sizeof(official_pending[0]),
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.bone_count == 1u);
    CHECK_TRUE(info.skin_token_count == 3u);
    CHECK_TRUE(info.next_is_global_eos);
    CHECK_TRUE(count_mask(mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE) == 1u);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS]);

    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        corrected_pending, sizeof(corrected_pending) / sizeof(corrected_pending[0]),
        TRELLIS_TOKENSKIN_EOS_CORRECTED,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(!info.next_is_global_eos);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS]);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE]);
    CHECK_TRUE(!mask[TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS]);

    memcpy(official_complete, official_pending, sizeof(official_pending));
    official_complete[9] = TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS;
    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        official_complete, sizeof(official_complete) / sizeof(official_complete[0]),
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.has_global_eos);
    CHECK_TRUE(info.state == TRELLIS_TOKENSKIN_GRAMMAR_SEQUENCE_COMPLETE);
    CHECK_TRUE(count_mask(mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE) == 0u);
    CHECK_TRUE(trellis_tokenskin_tokenizer_extract_skin_indices(
        official_complete, sizeof(official_complete) / sizeof(official_complete[0]),
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        indices, sizeof(indices) / sizeof(indices[0]), &indices_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(indices_count == 4u);
    CHECK_TRUE(indices[0] == 0 && indices[1] == 1 && indices[2] == 2);
    CHECK_TRUE(indices[3] == TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE);

    CHECK_TRUE(trellis_tokenskin_tokenizer_fsq_code(
        indices[3], TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT, alias_code) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_tokenskin_tokenizer_fsq_code(
        0, TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT, zero_code) == TRELLIS_STATUS_OK);
    for (size_t i = 0; i < 5u; ++i) {
        CHECK_TRUE(nearly_equal(alias_code[i], -1.0f));
        CHECK_TRUE(nearly_equal(alias_code[i], zero_code[i]));
    }
    CHECK_TRUE(trellis_tokenskin_tokenizer_fsq_code(
        TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE,
        TRELLIS_TOKENSKIN_EOS_CORRECTED,
        alias_code) == TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(trellis_tokenskin_tokenizer_fsq_code(
        -1, TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT, alias_code) ==
        TRELLIS_STATUS_OK);
    for (size_t i = 0; i < 5u; ++i) {
        CHECK_TRUE(nearly_equal(alias_code[i], 0.75f));
    }
    CHECK_TRUE(trellis_tokenskin_tokenizer_fsq_code(
        -1, TRELLIS_TOKENSKIN_EOS_CORRECTED, alias_code) ==
        TRELLIS_STATUS_PARSE_ERROR);

    official_complete[6] = TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS;
    official_complete[7] = TRELLIS_TOKENSKIN_TOKEN_PAD;
    official_complete[8] = TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION;
    CHECK_TRUE(trellis_tokenskin_tokenizer_extract_skin_indices(
        official_complete, sizeof(official_complete) / sizeof(official_complete[0]),
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT,
        indices, sizeof(indices) / sizeof(indices[0]), &indices_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(indices[0] == -9 && indices[1] == -8 && indices[2] == -1 &&
        indices[3] == TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE);

    memcpy(corrected_ready, corrected_pending, sizeof(corrected_pending));
    corrected_ready[9] = 270;
    CHECK_TRUE(trellis_tokenskin_tokenizer_next_mask(
        corrected_ready, sizeof(corrected_ready) / sizeof(corrected_ready[0]),
        TRELLIS_TOKENSKIN_EOS_CORRECTED,
        mask, TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE, &info) == TRELLIS_STATUS_OK);
    CHECK_TRUE(info.next_is_global_eos);
    CHECK_TRUE(mask[TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS]);

    memcpy(corrected_complete, corrected_ready, sizeof(corrected_ready));
    corrected_complete[10] = TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS;
    CHECK_TRUE(trellis_tokenskin_tokenizer_extract_skin_indices(
        corrected_complete, sizeof(corrected_complete) / sizeof(corrected_complete[0]),
        TRELLIS_TOKENSKIN_EOS_CORRECTED,
        indices, sizeof(indices) / sizeof(indices[0]), &indices_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(indices_count == 4u);
    CHECK_TRUE(indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3);

cleanup:
    free(mask);
}

static void test_invalid_branch_decode(void) {
    const int32_t incomplete_branch[] = {
        TRELLIS_TOKENSKIN_TOKEN_BOS,
        TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION,
        128, 128, 128,
        TRELLIS_TOKENSKIN_TOKEN_BRANCH,
        128, 128, 128,
        TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS,
    };
    trellis_tokenskin_skeleton skeleton;
    memset(&skeleton, 0, sizeof(skeleton));
    CHECK_TRUE(trellis_tokenskin_tokenizer_decode_skeleton(
        incomplete_branch,
        sizeof(incomplete_branch) / sizeof(incomplete_branch[0]),
        &skeleton) == TRELLIS_STATUS_PARSE_ERROR);

cleanup:
    trellis_tokenskin_skeleton_free(&skeleton);
}

int main(void) {
    test_grammar_masks();
    test_branch_bones_and_decode();
    test_eos_modes_and_alias();
    test_invalid_branch_decode();
    if (g_failures != 0) {
        fprintf(stderr, "%d TokenSkin tokenizer test(s) failed\n", g_failures);
        return 1;
    }
    puts("TokenSkin tokenizer tests passed");
    return 0;
}
