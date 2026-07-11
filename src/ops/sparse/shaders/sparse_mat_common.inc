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
layout(binding = 7) readonly buffer I2 { int i2[]; };

const uint TILE_K = 64u;

shared float x_tile[16][64];
shared float w_tile[16][64];
