/*
 * CPU mesh -> Flexible Dual Grid conversion.
 *
 * Ported from O-Voxel's MIT-licensed flexible_dual_grid.cpp:
 * Copyright (C) 2025 Jianfeng Xiang <belljig@outlook.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "trellis.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Int3 {
    int32_t x;
    int32_t y;
    int32_t z;

    int32_t & operator[](int axis) { return (&x)[axis]; }
    int32_t operator[](int axis) const { return (&x)[axis]; }
};

struct Bool3 {
    uint8_t x;
    uint8_t y;
    uint8_t z;

    uint8_t & operator[](int axis) { return (&x)[axis]; }
    uint8_t operator[](int axis) const { return (&x)[axis]; }
};

struct VoxelCoord {
    int32_t x;
    int32_t y;
    int32_t z;

    int32_t & operator[](int axis) { return (&x)[axis]; }
    int32_t operator[](int axis) const { return (&x)[axis]; }

    bool operator==(const VoxelCoord & other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelCoordHash {
    size_t operator()(const VoxelCoord & v) const {
        const size_t p1 = 73856093u;
        const size_t p2 = 19349663u;
        const size_t p3 = 83492791u;
        return static_cast<size_t>(v.x) * p1 ^
               static_cast<size_t>(v.y) * p2 ^
               static_cast<size_t>(v.z) * p3;
    }
};

using VoxelTable = std::unordered_map<VoxelCoord, size_t, VoxelCoordHash>;

template <typename T>
T clamp_value(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

template <typename T, typename U>
U lerp_value(const T & a, const T & b, const T & t, const U & value_a, const U & value_b) {
    if (a == b) {
        return value_a;
    }
    const T alpha = (t - a) / (b - a);
    return (T(1) - alpha) * value_a + alpha * value_b;
}

bool finite3(const float * value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

void intersect_qef(
    const Eigen::Vector3f & voxel_size,
    const Eigen::Vector3i & grid_min,
    const Eigen::Vector3i & grid_max,
    const std::vector<Eigen::Vector3f> & triangles,
    VoxelTable & table,
    std::vector<Int3> & voxels,
    std::vector<Eigen::Vector3f> & means,
    std::vector<float> & counts,
    std::vector<Bool3> & intersected,
    std::vector<Eigen::Matrix4f> & qefs) {
    const size_t triangle_count = triangles.size() / 3u;

    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const Eigen::Vector3f & v0 = triangles[triangle * 3u + 0u];
        const Eigen::Vector3f & v1 = triangles[triangle * 3u + 1u];
        const Eigen::Vector3f & v2 = triangles[triangle * 3u + 2u];

        const Eigen::Vector3f e0 = v1 - v0;
        const Eigen::Vector3f e1 = v2 - v1;
        const Eigen::Vector3f cross = e0.cross(e1);
        if (!(cross.squaredNorm() > 0.0f)) {
            continue;
        }
        const Eigen::Vector3f normal = cross.normalized();
        Eigen::Vector4f plane;
        plane << normal.x(), normal.y(), normal.z(), -normal.dot(v0);
        const Eigen::Matrix4f qef = plane * plane.transpose();

        auto scan_line_fill = [&](int axis2) {
            const int axis0 = (axis2 + 1) % 3;
            const int axis1 = (axis2 + 2) % 3;
            std::array<Eigen::Vector3d, 3> points = {
                Eigen::Vector3d(v0[axis0], v0[axis1], v0[axis2]),
                Eigen::Vector3d(v1[axis0], v1[axis1], v1[axis2]),
                Eigen::Vector3d(v2[axis0], v2[axis1], v2[axis2]),
            };
            std::sort(points.begin(), points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b) {
                return a.y() < b.y();
            });

            const int start = clamp_value(
                static_cast<int>(points[0].y() / voxel_size[axis1]), grid_min[axis1], grid_max[axis1] - 1);
            const int middle = clamp_value(
                static_cast<int>(points[1].y() / voxel_size[axis1]), grid_min[axis1], grid_max[axis1] - 1);
            const int end = clamp_value(
                static_cast<int>(points[2].y() / voxel_size[axis1]), grid_min[axis1], grid_max[axis1] - 1);

            auto scan_half = [&](int row_start,
                                 int row_end,
                                 const Eigen::Vector3d & t0,
                                 const Eigen::Vector3d & t1,
                                 const Eigen::Vector3d & t2) {
                for (int y_index = row_start; y_index < row_end; ++y_index) {
                    const double y = static_cast<double>(y_index + 1) * voxel_size[axis1];
                    Eigen::Vector2d t3 = lerp_value(
                        t0.y(), t1.y(), y,
                        Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
                    Eigen::Vector2d t4 = lerp_value(
                        t0.y(), t2.y(), y,
                        Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t2.x(), t2.z()));
                    if (t3.x() > t4.x()) {
                        std::swap(t3, t4);
                    }
                    const int line_start = clamp_value(
                        static_cast<int>(t3.x() / voxel_size[axis0]), grid_min[axis0], grid_max[axis0] - 1);
                    const int line_end = clamp_value(
                        static_cast<int>(t4.x() / voxel_size[axis0]), grid_min[axis0], grid_max[axis0] - 1);
                    for (int x_index = line_start; x_index < line_end; ++x_index) {
                        const double x = static_cast<double>(x_index + 1) * voxel_size[axis0];
                        const double z = lerp_value(t3.x(), t4.x(), x, t3.y(), t4.y());
                        const int z_index = static_cast<int>(z / voxel_size[axis2]);
                        if (z_index < grid_min[axis2] || z_index >= grid_max[axis2]) {
                            continue;
                        }

                        for (int dx = 0; dx < 2; ++dx) {
                            for (int dy = 0; dy < 2; ++dy) {
                                VoxelCoord coord{};
                                coord[axis0] = x_index + dx;
                                coord[axis1] = y_index + dy;
                                coord[axis2] = z_index;
                                Eigen::Vector3d hit;
                                hit[axis0] = x;
                                hit[axis1] = y;
                                hit[axis2] = z;

                                const auto found = table.find(coord);
                                if (found == table.end()) {
                                    const size_t index = voxels.size();
                                    table.emplace(coord, index);
                                    voxels.push_back({coord.x, coord.y, coord.z});
                                    means.push_back(hit.cast<float>());
                                    counts.push_back(1.0f);
                                    intersected.push_back({0u, 0u, 0u});
                                    qefs.push_back(qef);
                                    if (dx == 0 && dy == 0) {
                                        intersected.back()[axis2] = 1u;
                                    }
                                } else {
                                    const size_t index = found->second;
                                    means[index] += hit.cast<float>();
                                    counts[index] += 1.0f;
                                    if (dx == 0 && dy == 0) {
                                        intersected[index][axis2] = 1u;
                                    }
                                    qefs[index] += qef;
                                }
                            }
                        }
                    }
                }
            };

            scan_half(start, middle, points[0], points[1], points[2]);
            scan_half(middle, end, points[2], points[1], points[0]);
        };

        scan_line_fill(0);
        scan_line_fill(1);
        scan_line_fill(2);
    }
}

void face_qef(
    const Eigen::Vector3f & voxel_size,
    const Eigen::Vector3i & grid_min,
    const Eigen::Vector3i & grid_max,
    const std::vector<Eigen::Vector3f> & triangles,
    float face_weight,
    const VoxelTable & table,
    std::vector<Eigen::Matrix4f> & qefs) {
    const size_t triangle_count = triangles.size() / 3u;

    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const Eigen::Vector3f & v0 = triangles[triangle * 3u + 0u];
        const Eigen::Vector3f & v1 = triangles[triangle * 3u + 1u];
        const Eigen::Vector3f & v2 = triangles[triangle * 3u + 2u];
        const Eigen::Vector3f e0 = v1 - v0;
        const Eigen::Vector3f e1 = v2 - v1;
        const Eigen::Vector3f e2 = v0 - v2;
        const Eigen::Vector3f cross = e0.cross(e1);
        if (!(cross.squaredNorm() > 0.0f)) {
            continue;
        }
        const Eigen::Vector3f normal = cross.normalized();
        Eigen::Vector4f plane;
        plane << normal.x(), normal.y(), normal.z(), -normal.dot(v0);
        const Eigen::Matrix4f qef = plane * plane.transpose();

        const Eigen::Vector3f bbox_min_f = v0.cwiseMin(v1).cwiseMin(v2).cwiseQuotient(voxel_size);
        const Eigen::Vector3f bbox_max_f = v0.cwiseMax(v1).cwiseMax(v2).cwiseQuotient(voxel_size);
        const Eigen::Vector3i bbox_min(
            std::max(static_cast<int>(bbox_min_f.x()), grid_min.x()),
            std::max(static_cast<int>(bbox_min_f.y()), grid_min.y()),
            std::max(static_cast<int>(bbox_min_f.z()), grid_min.z()));
        const Eigen::Vector3i bbox_max(
            std::min(static_cast<int>(bbox_max_f.x() + 1.0f), grid_max.x()),
            std::min(static_cast<int>(bbox_max_f.y() + 1.0f), grid_max.y()),
            std::min(static_cast<int>(bbox_max_f.z() + 1.0f), grid_max.z()));

        const Eigen::Vector3f corner(
            normal.x() > 0.0f ? voxel_size.x() : 0.0f,
            normal.y() > 0.0f ? voxel_size.y() : 0.0f,
            normal.z() > 0.0f ? voxel_size.z() : 0.0f);
        const float plane_d1 = normal.dot(corner - v0);
        const float plane_d2 = normal.dot(voxel_size - corner - v0);

        const int mul_xy = normal.z() < 0.0f ? -1 : 1;
        const Eigen::Vector2f n_xy_e0(-mul_xy * e0.y(), mul_xy * e0.x());
        const Eigen::Vector2f n_xy_e1(-mul_xy * e1.y(), mul_xy * e1.x());
        const Eigen::Vector2f n_xy_e2(-mul_xy * e2.y(), mul_xy * e2.x());
        const float d_xy_e0 = -n_xy_e0.dot(v0.head<2>()) + n_xy_e0.cwiseMax(0.0f).dot(voxel_size.head<2>());
        const float d_xy_e1 = -n_xy_e1.dot(v1.head<2>()) + n_xy_e1.cwiseMax(0.0f).dot(voxel_size.head<2>());
        const float d_xy_e2 = -n_xy_e2.dot(v2.head<2>()) + n_xy_e2.cwiseMax(0.0f).dot(voxel_size.head<2>());

        const int mul_yz = normal.x() < 0.0f ? -1 : 1;
        const Eigen::Vector2f n_yz_e0(-mul_yz * e0.z(), mul_yz * e0.y());
        const Eigen::Vector2f n_yz_e1(-mul_yz * e1.z(), mul_yz * e1.y());
        const Eigen::Vector2f n_yz_e2(-mul_yz * e2.z(), mul_yz * e2.y());
        const Eigen::Vector2f voxel_yz(voxel_size.y(), voxel_size.z());
        const float d_yz_e0 = -n_yz_e0.dot(Eigen::Vector2f(v0.y(), v0.z())) + n_yz_e0.cwiseMax(0.0f).dot(voxel_yz);
        const float d_yz_e1 = -n_yz_e1.dot(Eigen::Vector2f(v1.y(), v1.z())) + n_yz_e1.cwiseMax(0.0f).dot(voxel_yz);
        const float d_yz_e2 = -n_yz_e2.dot(Eigen::Vector2f(v2.y(), v2.z())) + n_yz_e2.cwiseMax(0.0f).dot(voxel_yz);

        const int mul_zx = normal.y() < 0.0f ? -1 : 1;
        const Eigen::Vector2f n_zx_e0(-mul_zx * e0.x(), mul_zx * e0.z());
        const Eigen::Vector2f n_zx_e1(-mul_zx * e1.x(), mul_zx * e1.z());
        const Eigen::Vector2f n_zx_e2(-mul_zx * e2.x(), mul_zx * e2.z());
        const Eigen::Vector2f voxel_zx(voxel_size.z(), voxel_size.x());
        const float d_zx_e0 = -n_zx_e0.dot(Eigen::Vector2f(v0.z(), v0.x())) + n_zx_e0.cwiseMax(0.0f).dot(voxel_zx);
        const float d_zx_e1 = -n_zx_e1.dot(Eigen::Vector2f(v1.z(), v1.x())) + n_zx_e1.cwiseMax(0.0f).dot(voxel_zx);
        const float d_zx_e2 = -n_zx_e2.dot(Eigen::Vector2f(v2.z(), v2.x())) + n_zx_e2.cwiseMax(0.0f).dot(voxel_zx);

        for (int z = bbox_min.z(); z < bbox_max.z(); ++z) {
            for (int y = bbox_min.y(); y < bbox_max.y(); ++y) {
                for (int x = bbox_min.x(); x < bbox_max.x(); ++x) {
                    const Eigen::Vector3f p = voxel_size.cwiseProduct(Eigen::Vector3f(x, y, z));
                    const float normal_dot_p = normal.dot(p);
                    if ((normal_dot_p + plane_d1) * (normal_dot_p + plane_d2) > 0.0f) {
                        continue;
                    }
                    const Eigen::Vector2f p_xy(p.x(), p.y());
                    if (n_xy_e0.dot(p_xy) + d_xy_e0 < 0.0f ||
                        n_xy_e1.dot(p_xy) + d_xy_e1 < 0.0f ||
                        n_xy_e2.dot(p_xy) + d_xy_e2 < 0.0f) {
                        continue;
                    }
                    const Eigen::Vector2f p_yz(p.y(), p.z());
                    if (n_yz_e0.dot(p_yz) + d_yz_e0 < 0.0f ||
                        n_yz_e1.dot(p_yz) + d_yz_e1 < 0.0f ||
                        n_yz_e2.dot(p_yz) + d_yz_e2 < 0.0f) {
                        continue;
                    }
                    const Eigen::Vector2f p_zx(p.z(), p.x());
                    if (n_zx_e0.dot(p_zx) + d_zx_e0 < 0.0f ||
                        n_zx_e1.dot(p_zx) + d_zx_e1 < 0.0f ||
                        n_zx_e2.dot(p_zx) + d_zx_e2 < 0.0f) {
                        continue;
                    }

                    const auto found = table.find(VoxelCoord{x, y, z});
                    if (found != table.end()) {
                        qefs[found->second] += face_weight * qef;
                    }
                }
            }
        }
    }
}

void boundary_qef(
    const Eigen::Vector3f & voxel_size,
    const Eigen::Vector3i & grid_min,
    const Eigen::Vector3i & grid_max,
    const std::vector<Eigen::Vector3f> & boundaries,
    float boundary_weight,
    const VoxelTable & table,
    std::vector<Eigen::Matrix4f> & qefs) {
    for (size_t edge = 0; edge < boundaries.size() / 2u; ++edge) {
        const Eigen::Vector3f & v0 = boundaries[edge * 2u + 0u];
        const Eigen::Vector3f & v1 = boundaries[edge * 2u + 1u];
        Eigen::Vector3d direction = (v1 - v0).cast<double>();
        const double length = direction.norm();
        if (length < 1e-6) {
            continue;
        }
        direction.normalize();

        const Eigen::Matrix3f a = Eigen::Matrix3f::Identity() -
                                  (direction * direction.transpose()).cast<float>();
        const Eigen::Vector3f b = -a * v0;
        const float c = v0.transpose() * a * v0;
        Eigen::Matrix4f qef = Eigen::Matrix4f::Zero();
        qef.block<3, 3>(0, 0) = a;
        qef.block<3, 1>(0, 3) = b;
        qef.block<1, 3>(3, 0) = b.transpose();
        qef(3, 3) = c;

        const Eigen::Vector3i start_voxel = v0.cwiseQuotient(voxel_size).array().floor().cast<int>();
        const Eigen::Vector3i end_voxel = v1.cwiseQuotient(voxel_size).array().floor().cast<int>();
        (void) end_voxel;
        const Eigen::Vector3i step = (direction.array() > 0.0).select(
            Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(-1, -1, -1));

        Eigen::Vector3d t_max;
        Eigen::Vector3d t_delta;
        for (int axis = 0; axis < 3; ++axis) {
            if (direction[axis] == 0.0) {
                t_max[axis] = std::numeric_limits<double>::infinity();
                t_delta[axis] = std::numeric_limits<double>::infinity();
            } else {
                const float border = voxel_size[axis] *
                                     (start_voxel[axis] + (step[axis] > 0 ? 1 : 0));
                t_max[axis] = (border - v0[axis]) / direction[axis];
                t_delta[axis] = voxel_size[axis] / std::abs(direction[axis]);
            }
        }

        Eigen::Vector3i current = start_voxel;
        std::vector<VoxelCoord> visited;
        visited.push_back({current.x(), current.y(), current.z()});
        while (true) {
            int axis;
            if (t_max.x() < t_max.y()) {
                axis = t_max.x() < t_max.z() ? 0 : 2;
            } else {
                axis = t_max.y() < t_max.z() ? 1 : 2;
            }
            if (t_max[axis] > length) {
                break;
            }
            current[axis] += step[axis];
            t_max[axis] += t_delta[axis];
            visited.push_back({current.x(), current.y(), current.z()});
        }

        for (const VoxelCoord & coord : visited) {
            if (coord.x < grid_min.x() || coord.x >= grid_max.x() ||
                coord.y < grid_min.y() || coord.y >= grid_max.y() ||
                coord.z < grid_min.z() || coord.z >= grid_max.z()) {
                continue;
            }
            const auto found = table.find(coord);
            if (found != table.end()) {
                qefs[found->second] += boundary_weight * qef;
            }
        }
    }
}

Eigen::Vector3f solve_qef_in_voxel(
    const Eigen::Matrix4f & original_qef,
    const Eigen::Vector3f & mean,
    float count,
    float regularization_weight,
    const Int3 & coord,
    const Eigen::Vector3f & voxel_size) {
    Eigen::Matrix4f qef = original_qef;
    const float min_corner[3] = {
        coord.x * voxel_size.x(), coord.y * voxel_size.y(), coord.z * voxel_size.z(),
    };
    const float max_corner[3] = {
        (coord.x + 1) * voxel_size.x(),
        (coord.y + 1) * voxel_size.y(),
        (coord.z + 1) * voxel_size.z(),
    };

    if (regularization_weight > 0.0f) {
        const Eigen::Vector3f point = mean / count;
        Eigen::Matrix4f regularization = Eigen::Matrix4f::Zero();
        regularization.topLeftCorner<3, 3>() = Eigen::Matrix3f::Identity();
        regularization.block<3, 1>(0, 3) = -point;
        regularization.block<1, 3>(3, 0) = -point.transpose();
        regularization(3, 3) = point.dot(point);
        qef += regularization_weight * count * regularization;
    }

    const Eigen::Matrix3f a = qef.topLeftCorner<3, 3>();
    const Eigen::Vector3f b = -qef.block<3, 1>(0, 3);
    Eigen::Vector3f result = a.colPivHouseholderQr().solve(b);
    const bool inside =
        result.x() >= min_corner[0] && result.x() <= max_corner[0] &&
        result.y() >= min_corner[1] && result.y() <= max_corner[1] &&
        result.z() >= min_corner[2] && result.z() <= max_corner[2];
    if (inside) {
        return result;
    }

    float best = std::numeric_limits<float>::infinity();
    auto consider = [&](const Eigen::Vector4f & point) {
        const float error = point.transpose() * qef * point;
        if (error < best) {
            best = error;
            result = point.head<3>();
        }
    };

    for (int fixed_axis = 0; fixed_axis < 3; ++fixed_axis) {
        const int axis1 = (fixed_axis + 1) % 3;
        const int axis2 = (fixed_axis + 2) % 3;
        Eigen::Matrix2f sub_a;
        Eigen::Matrix2f sub_b;
        sub_a << qef(axis1, axis1), qef(axis1, axis2),
                 qef(axis2, axis1), qef(axis2, axis2);
        sub_b << qef(axis1, fixed_axis), qef(axis1, 3),
                 qef(axis2, fixed_axis), qef(axis2, 3);
        const auto solver = sub_a.colPivHouseholderQr();
        for (int side = 0; side < 2; ++side) {
            const float fixed = side == 0 ? min_corner[fixed_axis] : max_corner[fixed_axis];
            const Eigen::Vector2f rhs = -sub_b * Eigen::Vector2f(fixed, 1.0f);
            const Eigen::Vector2f value = solver.solve(rhs);
            if (value.x() < min_corner[axis1] || value.x() > max_corner[axis1] ||
                value.y() < min_corner[axis2] || value.y() > max_corner[axis2]) {
                continue;
            }
            Eigen::Vector4f point;
            point[fixed_axis] = fixed;
            point[axis1] = value.x();
            point[axis2] = value.y();
            point[3] = 1.0f;
            consider(point);
        }
    }

    for (int free_axis = 0; free_axis < 3; ++free_axis) {
        const int axis1 = (free_axis + 1) % 3;
        const int axis2 = (free_axis + 2) % 3;
        const float coefficient = qef(free_axis, free_axis);
        const Eigen::Vector3f row(qef(free_axis, axis1), qef(free_axis, axis2), qef(free_axis, 3));
        for (int side1 = 0; side1 < 2; ++side1) {
            for (int side2 = 0; side2 < 2; ++side2) {
                const float fixed1 = side1 == 0 ? min_corner[axis1] : max_corner[axis1];
                const float fixed2 = side2 == 0 ? min_corner[axis2] : max_corner[axis2];
                const float value = -row.dot(Eigen::Vector3f(fixed1, fixed2, 1.0f)) / coefficient;
                if (value < min_corner[free_axis] || value > max_corner[free_axis]) {
                    continue;
                }
                Eigen::Vector4f point;
                point[free_axis] = value;
                point[axis1] = fixed1;
                point[axis2] = fixed2;
                point[3] = 1.0f;
                consider(point);
            }
        }
    }

    for (int x_side = 0; x_side < 2; ++x_side) {
        for (int y_side = 0; y_side < 2; ++y_side) {
            for (int z_side = 0; z_side < 2; ++z_side) {
                Eigen::Vector4f point;
                point << (x_side ? min_corner[0] : max_corner[0]),
                         (y_side ? min_corner[1] : max_corner[1]),
                         (z_side ? min_corner[2] : max_corner[2]),
                         1.0f;
                consider(point);
            }
        }
    }
    return result;
}

trellis_status convert_impl(
    const float * vertices,
    const int32_t * faces,
    int64_t n_faces,
    const trellis_flexible_dual_grid_options & options,
    trellis_flexible_dual_grid * output) {
    const Eigen::Vector3f aabb_min(options.aabb_min[0], options.aabb_min[1], options.aabb_min[2]);
    const Eigen::Vector3f aabb_max(options.aabb_max[0], options.aabb_max[1], options.aabb_max[2]);
    const Eigen::Vector3f voxel_size(
        (aabb_max.x() - aabb_min.x()) / options.grid_size[0],
        (aabb_max.y() - aabb_min.y()) / options.grid_size[1],
        (aabb_max.z() - aabb_min.z()) / options.grid_size[2]);
    const Eigen::Vector3i grid_min(0, 0, 0);
    const Eigen::Vector3i grid_max(options.grid_size[0], options.grid_size[1], options.grid_size[2]);

    std::vector<Eigen::Vector3f> triangles;
    triangles.reserve(static_cast<size_t>(n_faces) * 3u);
    for (int64_t face = 0; face < n_faces; ++face) {
        for (int corner = 0; corner < 3; ++corner) {
            const int32_t vertex_index = faces[face * 3 + corner];
            const float * vertex = vertices + static_cast<size_t>(vertex_index) * 3u;
            triangles.emplace_back(
                vertex[0] - aabb_min.x(),
                vertex[1] - aabb_min.y(),
                vertex[2] - aabb_min.z());
        }
    }

    VoxelTable table;
    table.reserve(static_cast<size_t>(n_faces));
    std::vector<Int3> voxels;
    std::vector<Eigen::Vector3f> means;
    std::vector<float> counts;
    std::vector<Bool3> intersected;
    std::vector<Eigen::Matrix4f> qefs;
    intersect_qef(voxel_size, grid_min, grid_max, triangles, table, voxels, means, counts, intersected, qefs);

    if (options.face_weight > 0.0f) {
        face_qef(voxel_size, grid_min, grid_max, triangles, options.face_weight, table, qefs);
    }

    if (options.boundary_weight > 0.0f) {
        std::map<std::pair<int32_t, int32_t>, int> edge_counts;
        for (int64_t face = 0; face < n_faces; ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                int32_t index0 = faces[face * 3 + corner];
                int32_t index1 = faces[face * 3 + (corner + 1) % 3];
                if (index0 > index1) {
                    std::swap(index0, index1);
                }
                ++edge_counts[std::make_pair(index0, index1)];
            }
        }
        std::vector<Eigen::Vector3f> boundaries;
        for (const auto & edge : edge_counts) {
            if (edge.second != 1) {
                continue;
            }
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
                const int32_t vertex_index = endpoint == 0 ? edge.first.first : edge.first.second;
                const float * vertex = vertices + static_cast<size_t>(vertex_index) * 3u;
                boundaries.emplace_back(
                    vertex[0] - aabb_min.x(),
                    vertex[1] - aabb_min.y(),
                    vertex[2] - aabb_min.z());
            }
        }
        boundary_qef(
            voxel_size, grid_min, grid_max, boundaries,
            options.boundary_weight, table, qefs);
    }

    const size_t count = voxels.size();
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        count > std::numeric_limits<size_t>::max() / (4u * sizeof(int32_t)) ||
        count > std::numeric_limits<size_t>::max() / (3u * sizeof(float))) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (count == 0u) {
        return TRELLIS_STATUS_OK;
    }

    int32_t * out_coords = static_cast<int32_t *>(std::malloc(count * 4u * sizeof(int32_t)));
    float * out_dual = static_cast<float *>(std::malloc(count * 3u * sizeof(float)));
    uint8_t * out_intersected = static_cast<uint8_t *>(std::malloc(count * 3u * sizeof(uint8_t)));
    if (out_coords == nullptr || out_dual == nullptr || out_intersected == nullptr) {
        std::free(out_coords);
        std::free(out_dual);
        std::free(out_intersected);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < count; ++index) {
        const Eigen::Vector3f dual = solve_qef_in_voxel(
            qefs[index], means[index], counts[index], options.regularization_weight,
            voxels[index], voxel_size);
        out_coords[index * 4u + 0u] = 0;
        out_coords[index * 4u + 1u] = voxels[index].x;
        out_coords[index * 4u + 2u] = voxels[index].y;
        out_coords[index * 4u + 3u] = voxels[index].z;
        out_dual[index * 3u + 0u] = dual.x();
        out_dual[index * 3u + 1u] = dual.y();
        out_dual[index * 3u + 2u] = dual.z();
        out_intersected[index * 3u + 0u] = intersected[index].x;
        out_intersected[index * 3u + 1u] = intersected[index].y;
        out_intersected[index * 3u + 2u] = intersected[index].z;
    }

    output->coords = out_coords;
    output->dual_vertices = out_dual;
    output->intersected = out_intersected;
    output->n = static_cast<int64_t>(count);
    return TRELLIS_STATUS_OK;
}

}  // namespace

extern "C" void trellis_flexible_dual_grid_options_default(
    trellis_flexible_dual_grid_options * options) {
    if (options == nullptr) {
        return;
    }
    std::memset(options, 0, sizeof(*options));
    options->grid_size[0] = 512;
    options->grid_size[1] = 512;
    options->grid_size[2] = 512;
    for (int axis = 0; axis < 3; ++axis) {
        options->aabb_min[axis] = -0.5f;
        options->aabb_max[axis] = 0.5f;
    }
    options->face_weight = 1.0f;
    options->boundary_weight = 0.2f;
    options->regularization_weight = 0.01f;
}

extern "C" void trellis_flexible_dual_grid_free(trellis_flexible_dual_grid * grid) {
    if (grid == nullptr) {
        return;
    }
    std::free(grid->coords);
    std::free(grid->dual_vertices);
    std::free(grid->intersected);
    std::memset(grid, 0, sizeof(*grid));
}

extern "C" trellis_status trellis_mesh_to_flexible_dual_grid_host(
    const float * vertices,
    int64_t n_vertices,
    const int32_t * faces,
    int64_t n_faces,
    const trellis_flexible_dual_grid_options * options,
    trellis_flexible_dual_grid * grid_out) {
    if (grid_out == nullptr || options == nullptr || n_vertices < 0 || n_faces < 0 ||
        (n_vertices > 0 && vertices == nullptr) || (n_faces > 0 && faces == nullptr) ||
        n_vertices > std::numeric_limits<int32_t>::max()) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    std::memset(grid_out, 0, sizeof(*grid_out));

    for (int axis = 0; axis < 3; ++axis) {
        if (options->grid_size[axis] <= 0 ||
            !std::isfinite(options->aabb_min[axis]) ||
            !std::isfinite(options->aabb_max[axis]) ||
            !(options->aabb_max[axis] > options->aabb_min[axis])) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!std::isfinite(options->face_weight) || options->face_weight < 0.0f ||
        !std::isfinite(options->boundary_weight) || options->boundary_weight < 0.0f ||
        !std::isfinite(options->regularization_weight) || options->regularization_weight < 0.0f) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (static_cast<uint64_t>(n_faces) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / (3u * sizeof(Eigen::Vector3f)))) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (!finite3(vertices + static_cast<size_t>(vertex) * 3u)) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    for (int64_t face = 0; face < n_faces; ++face) {
        for (int corner = 0; corner < 3; ++corner) {
            const int32_t index = faces[face * 3 + corner];
            if (index < 0 || index >= n_vertices) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    try {
        return convert_impl(vertices, faces, n_faces, *options, grid_out);
    } catch (const std::bad_alloc &) {
        trellis_flexible_dual_grid_free(grid_out);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        trellis_flexible_dual_grid_free(grid_out);
        return TRELLIS_STATUS_ERROR;
    }
}
