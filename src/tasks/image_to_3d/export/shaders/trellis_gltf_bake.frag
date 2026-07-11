#version 450

layout(location = 0) in vec3 v_position;

layout(location = 0) out vec4 out_base;
layout(location = 1) out vec4 out_mr;

layout(std430, binding = 3) readonly buffer HashEntries { int hash_entries[]; };
layout(std430, binding = 4) readonly buffer VoxelAttrs { float voxel_attrs[]; };
layout(std430, binding = 5) readonly buffer ProjectionAux { uint project_aux[]; };

layout(push_constant) uniform Push {
    uint texture_size;
    uint triangle_count;
    uint voxel_count;
    uint channels;
    uint hash_size;
    uint resolution;
    uint pad0;
    uint pad1;
    float bounds_min_x;
    float bounds_min_y;
    float bounds_min_z;
    float bounds_inv_extent;
    uint chart_grid;
    uint chart_target_faces;
    uint chart_normal_bins;
    uint chart_flags;
    uint project_vertex_count;
    uint project_face_count;
    uint project_node_count;
    uint project_flags;
} pc;

uint hash_coord(ivec3 c) {
    uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

int lookup_voxel(ivec3 c) {
    if (pc.hash_size == 0u) return -1;
    uint slot = hash_coord(c) & (pc.hash_size - 1u);
    for (uint probe = 0u; probe < pc.hash_size; ++probe) {
        uint off = slot * 4u;
        int value = hash_entries[off + 3u];
        if (value < 0) return -1;
        if (hash_entries[off + 0u] == c.x &&
            hash_entries[off + 1u] == c.y &&
            hash_entries[off + 2u] == c.z) {
            return value;
        }
        slot = (slot + 1u) & (pc.hash_size - 1u);
    }
    return -1;
}

vec4 fetch_base(int voxel_id) {
    vec4 pbr = vec4(0.8, 0.8, 0.8, 1.0);
    if (voxel_id < 0 || pc.channels == 0u) return pbr;
    uint base = uint(voxel_id) * pc.channels;
    if (pc.channels > 0u) pbr.x = clamp(voxel_attrs[base + 0u], 0.0, 1.0);
    if (pc.channels > 1u) pbr.y = clamp(voxel_attrs[base + 1u], 0.0, 1.0);
    if (pc.channels > 2u) pbr.z = clamp(voxel_attrs[base + 2u], 0.0, 1.0);
    return pbr;
}

float fetch_metallic(int voxel_id) {
    if (voxel_id < 0 || pc.channels <= 3u) return 0.0;
    return clamp(voxel_attrs[uint(voxel_id) * pc.channels + 3u], 0.0, 1.0);
}

float fetch_roughness(int voxel_id) {
    if (voxel_id < 0 || pc.channels <= 4u) return 0.8;
    return clamp(voxel_attrs[uint(voxel_id) * pc.channels + 4u], 0.0, 1.0);
}

float fetch_alpha(int voxel_id) {
    if (voxel_id < 0 || pc.channels <= 5u) return 1.0;
    return clamp(voxel_attrs[uint(voxel_id) * pc.channels + 5u], 0.0, 1.0);
}

int nearest_voxel(ivec3 g) {
    int best = -1;
    int res = int(pc.resolution);
    for (int radius = 0; radius <= 2 && best < 0; ++radius) {
        for (int dz = -radius; dz <= radius && best < 0; ++dz) {
            for (int dy = -radius; dy <= radius && best < 0; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (radius > 0 && abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) continue;
                    ivec3 c = g + ivec3(dx, dy, dz);
                    if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= res || c.y >= res || c.z >= res) continue;
                    best = lookup_voxel(c);
                    if (best >= 0) break;
                }
            }
        }
    }
    return best;
}

void sample_pbr(vec3 p, out vec4 base, out float metallic, out float roughness) {
    int res = max(int(pc.resolution), 1);
    vec3 q = (p + vec3(0.5)) * float(res);
    ivec3 base_coord = ivec3(floor(q - vec3(0.5)));

    vec4 accum_base = vec4(0.0);
    float accum_metallic = 0.0;
    float accum_roughness = 0.0;
    float sum_w = 0.0;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                ivec3 c = base_coord + ivec3(dx, dy, dz);
                if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= res || c.y >= res || c.z >= res) continue;
                float wx = 1.0 - abs(q.x - float(c.x) - 0.5);
                float wy = 1.0 - abs(q.y - float(c.y) - 0.5);
                float wz = 1.0 - abs(q.z - float(c.z) - 0.5);
                float w = wx * wy * wz;
                if (w <= 0.0) continue;
                int id = lookup_voxel(c);
                if (id < 0) continue;
                accum_base += fetch_base(id) * w;
                accum_base.a += (fetch_alpha(id) - 1.0) * w;
                accum_metallic += fetch_metallic(id) * w;
                accum_roughness += fetch_roughness(id) * w;
                sum_w += w;
            }
        }
    }
    if (sum_w > 1e-6) {
        base = accum_base / sum_w;
        metallic = accum_metallic / sum_w;
        roughness = accum_roughness / sum_w;
        return;
    }

    base = vec4(0.0);
    metallic = 0.0;
    roughness = 0.0;
}

float aabb_distance_sq(vec3 pnt, vec3 bmin, vec3 bmax) {
    vec3 d = max(max(bmin - pnt, pnt - bmax), vec3(0.0));
    return dot(d, d);
}

vec3 project_vertex(uint vertex_id) {
    uint base = vertex_id * 3u;
    return vec3(
        uintBitsToFloat(project_aux[base + 0u]),
        uintBitsToFloat(project_aux[base + 1u]),
        uintBitsToFloat(project_aux[base + 2u]));
}

vec3 closest_point_triangle(vec3 pnt, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = pnt - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;

    vec3 bp = pnt - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    vec3 cp = pnt - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0 / max(va + vb + vc, 1e-30);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

vec3 project_face_closest(vec3 pnt, uint face_id, uint face_offset, out float dist_sq) {
    uint fbase = face_offset + face_id * 3u;
    uint ia = project_aux[fbase + 0u];
    uint ib = project_aux[fbase + 1u];
    uint ic = project_aux[fbase + 2u];
    vec3 a = project_vertex(ia);
    vec3 b = project_vertex(ib);
    vec3 c = project_vertex(ic);
    vec3 q = closest_point_triangle(pnt, a, b, c);
    vec3 d = pnt - q;
    dist_sq = dot(d, d);
    return q;
}

vec3 project_to_source_mesh(vec3 pnt) {
    if (pc.project_node_count == 0u || pc.project_face_count == 0u || pc.project_vertex_count == 0u) {
        return pnt;
    }

    uint face_offset = pc.project_vertex_count * 3u;
    uint node_offset = face_offset + pc.project_face_count * 3u;
    uint tri_offset = node_offset + pc.project_node_count * 8u;
    float best = 3.402823466e+38;
    vec3 best_point = pnt;

    uint stack[64];
    uint sp = 0u;
    stack[sp++] = 0u;
    while (sp > 0u) {
        uint node_id = stack[--sp];
        if (node_id >= pc.project_node_count) continue;
        uint nbase = node_offset + node_id * 8u;
        vec3 bmin = vec3(
            uintBitsToFloat(project_aux[nbase + 0u]),
            uintBitsToFloat(project_aux[nbase + 1u]),
            uintBitsToFloat(project_aux[nbase + 2u]));
        uint left = project_aux[nbase + 3u];
        vec3 bmax = vec3(
            uintBitsToFloat(project_aux[nbase + 4u]),
            uintBitsToFloat(project_aux[nbase + 5u]),
            uintBitsToFloat(project_aux[nbase + 6u]));
        uint meta = project_aux[nbase + 7u];
        if (aabb_distance_sq(pnt, bmin, bmax) > best) continue;
        if ((meta & 0x80000000u) != 0u) {
            uint count = meta & 0x7fffffffu;
            for (uint i = 0u; i < count; ++i) {
                uint face_id = project_aux[tri_offset + left + i];
                if (face_id >= pc.project_face_count) continue;
                float d;
                vec3 q = project_face_closest(pnt, face_id, face_offset, d);
                if (d < best) {
                    best = d;
                    best_point = q;
                }
            }
        } else {
            uint right = meta;
            if (left >= pc.project_node_count || right >= pc.project_node_count) continue;
            uint lbase = node_offset + left * 8u;
            uint rbase = node_offset + right * 8u;
            vec3 lmin = vec3(
                uintBitsToFloat(project_aux[lbase + 0u]),
                uintBitsToFloat(project_aux[lbase + 1u]),
                uintBitsToFloat(project_aux[lbase + 2u]));
            vec3 lmax = vec3(
                uintBitsToFloat(project_aux[lbase + 4u]),
                uintBitsToFloat(project_aux[lbase + 5u]),
                uintBitsToFloat(project_aux[lbase + 6u]));
            vec3 rmin = vec3(
                uintBitsToFloat(project_aux[rbase + 0u]),
                uintBitsToFloat(project_aux[rbase + 1u]),
                uintBitsToFloat(project_aux[rbase + 2u]));
            vec3 rmax = vec3(
                uintBitsToFloat(project_aux[rbase + 4u]),
                uintBitsToFloat(project_aux[rbase + 5u]),
                uintBitsToFloat(project_aux[rbase + 6u]));
            float ld = aabb_distance_sq(pnt, lmin, lmax);
            float rd = aabb_distance_sq(pnt, rmin, rmax);
            if (ld < rd) {
                if (rd <= best && sp < 64u) stack[sp++] = right;
                if (ld <= best && sp < 64u) stack[sp++] = left;
            } else {
                if (ld <= best && sp < 64u) stack[sp++] = left;
                if (rd <= best && sp < 64u) stack[sp++] = right;
            }
        }
    }
    return best_point;
}

void main() {
    vec4 base;
    float metallic;
    float roughness;
    vec3 sample_position = project_to_source_mesh(v_position);
    sample_pbr(sample_position, base, metallic, roughness);
    out_base = base;
    out_mr = vec4(0.0, roughness, metallic, 1.0);
}
