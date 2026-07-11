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
layout(binding = 5) buffer I1 { int i1[]; };
layout(binding = 6) buffer U0 { uint u0[]; };
layout(binding = 7) buffer I2 { int i2[]; };
layout(binding = 8) buffer I3 { int i3[]; };
layout(binding = 9) buffer I4 { int i4[]; };

const uint RULEBOOK_HASH_BASE = 32u;
const uint SCAN_BLOCK_SIZE = 128u;
const uint KERNEL_VOLUME = 27u;

uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint pack_coord_key(int b, int x, int y, int z) {
    uint raw =
        ((uint(b) & 3u) << 30) |
        ((uint(x) & 1023u) << 20) |
        ((uint(y) & 1023u) << 10) |
        (uint(z) & 1023u);
    return raw + 1u;
}

int find_coord_key(int b, int x, int y, int z) {
    uint key = pack_coord_key(b, x, y, z);
    uint slot = hash_u32(key) & p.table_mask;
    for (uint probe = 0; probe <= p.table_mask; ++probe) {
        uint found = u0[RULEBOOK_HASH_BASE + slot];
        if (found == key) {
            return i1[slot];
        }
        if (found == 0u) {
            return -1;
        }
        slot = (slot + 1u) & p.table_mask;
    }
    return -1;
}

uint gray_to_binary(uint gray) {
    uint binary = gray;
    for (uint v = 1u; v < KERNEL_VOLUME; ++v) {
        binary ^= gray >> v;
    }
    return binary;
}

bool sort_pair_should_swap(uint key_a, int idx_a, uint key_b, int idx_b, bool ascending) {
    bool greater = key_a > key_b || (key_a == key_b && uint(idx_a) > uint(idx_b));
    bool less = key_a < key_b || (key_a == key_b && uint(idx_a) < uint(idx_b));
    return ascending ? greater : less;
}
