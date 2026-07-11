#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_shader_subgroup_basic : require

layout(push_constant) uniform Push {
    uint reserved;
    uint n;
    uint in_channels;
    uint out_channels;
    uint total;
    uint table_mask;
    uint flags;
    float eps;
} p;

layout(binding = 0) readonly buffer F0 { float f0[]; };
layout(binding = 1) readonly buffer F1 { float f1[]; };
layout(binding = 2) readonly buffer F2 { float f2[]; };
layout(binding = 3) buffer F3 { float f3[]; };
layout(binding = 4) readonly buffer I0 { int i0[]; };
layout(binding = 5) readonly buffer I1 { int i1[]; };
layout(binding = 6) readonly buffer U0 { uint u0[]; };

const uint BLOCK_M = 32u;
const uint BLOCK_N = 32u;
const uint FRAG = 16u;
const uint TILE_K = 16u;
const uint KERNEL_VOLUME = 27u;

shared float16_t a_tile[BLOCK_M * TILE_K];
shared float16_t b_tile[TILE_K * BLOCK_N];
shared float c_tile[BLOCK_M * BLOCK_N];

uint row_tile_id() {
    return gl_WorkGroupID.z * gl_NumWorkGroups.y + gl_WorkGroupID.y;
}

void load_a_from_src_rows(uint pair_base, uint pair_count, uint pair_tile, uint c0) {
    const uint lane = gl_LocalInvocationID.x;
    for (uint load = lane; load < BLOCK_M * TILE_K; load += gl_WorkGroupSize.x) {
        uint r = load / TILE_K;
        uint k = load - r * TILE_K;
        uint pair = pair_tile * BLOCK_M + r;
        uint c = c0 + k;
        float v = 0.0;
        if (pair < pair_count && c < p.in_channels) {
            uint src = uint(i0[pair_base + pair]);
            v = f0[src * p.in_channels + c];
        }
        a_tile[load] = float16_t(v);
    }
}

void load_b_from_weight(uint offset, uint c0) {
    const uint lane = gl_LocalInvocationID.x;
    const uint oc_base = gl_WorkGroupID.x * BLOCK_N;
    for (uint load = lane; load < TILE_K * BLOCK_N; load += gl_WorkGroupSize.x) {
        uint k = load / BLOCK_N;
        uint n = load - k * BLOCK_N;
        uint c = c0 + k;
        uint oc = oc_base + n;
        float v = (oc < p.out_channels && c < p.in_channels) ?
            f1[(oc * KERNEL_VOLUME + offset) * p.in_channels + c] : 0.0;
        b_tile[load] = float16_t(v);
    }
}

void load_a_from_rows(uint row_tile, uint c0) {
    const uint lane = gl_LocalInvocationID.x;
    for (uint load = lane; load < BLOCK_M * TILE_K; load += gl_WorkGroupSize.x) {
        uint r = load / TILE_K;
        uint k = load - r * TILE_K;
        uint row = row_tile * BLOCK_M + r;
        uint c = c0 + k;
        float v = (row < p.n && c < p.in_channels) ? f0[row * p.in_channels + c] : 0.0;
        a_tile[load] = float16_t(v);
    }
}

void load_b_from_linear_weight(uint c0) {
    const uint lane = gl_LocalInvocationID.x;
    const uint oc_base = gl_WorkGroupID.x * BLOCK_N;
    for (uint load = lane; load < TILE_K * BLOCK_N; load += gl_WorkGroupSize.x) {
        uint k = load / BLOCK_N;
        uint n = load - k * BLOCK_N;
        uint c = c0 + k;
        uint oc = oc_base + n;
        float v = (oc < p.out_channels && c < p.in_channels) ?
            f1[oc * p.in_channels + c] : 0.0;
        b_tile[load] = float16_t(v);
    }
}
