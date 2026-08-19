/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "Common.h"
#include "LODSelect.h"

namespace gsplat
{
at::Tensor lod_build_binary_children(const at::Tensor &parents)
{
    DEVICE_GUARD(parents);
    CHECK_INPUT(parents);
    TORCH_CHECK(parents.dim() == 1, "lod_build_binary_children: parents must be [M]");
    TORCH_CHECK(
        parents.scalar_type() == at::kLong,
        "lod_build_binary_children: parents must be int64"
    );

    auto children = at::full(
        {parents.size(0), 2},
        -1,
        parents.options().dtype(at::kInt)
    );
    launch_lod_build_binary_children_kernel(parents, children);
    return children;
}

std::tuple<at::Tensor, at::Tensor> lod_select_topdown(
    const at::Tensor &centers,
    const at::Tensor &sizes,
    const at::Tensor &radii,
    const at::Tensor &children,
    const at::Tensor &is_leaf,
    const at::Tensor &root_ids,
    const at::Tensor &cam_pos,
    const at::Tensor &w2c,
    const at::Tensor &K,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double error_threshold_px,
    int64_t max_depth
)
{
    DEVICE_GUARD(centers);
    CHECK_INPUT(centers);
    CHECK_INPUT(sizes);
    CHECK_INPUT(radii);
    CHECK_INPUT(children);
    CHECK_INPUT(is_leaf);
    CHECK_INPUT(root_ids);
    CHECK_INPUT(cam_pos);
    CHECK_INPUT(w2c);
    CHECK_INPUT(K);

    const int64_t M = centers.size(0);
    TORCH_CHECK(
        centers.dim() == 2 && centers.size(1) == 3,
        "lod_select_topdown: centers must be [M, 3]"
    );
    TORCH_CHECK(sizes.sizes() == at::IntArrayRef({M}), "lod_select_topdown: sizes must be [M]");
    TORCH_CHECK(radii.sizes() == at::IntArrayRef({M}), "lod_select_topdown: radii must be [M]");
    TORCH_CHECK(
        children.sizes() == at::IntArrayRef({M, 2}),
        "lod_select_topdown: children must be [M, 2]"
    );
    TORCH_CHECK(
        is_leaf.sizes() == at::IntArrayRef({M}),
        "lod_select_topdown: is_leaf must be [M]"
    );
    TORCH_CHECK(
        root_ids.dim() == 1 && root_ids.numel() > 0,
        "lod_select_topdown: root_ids must be non-empty [R]"
    );
    TORCH_CHECK(cam_pos.numel() == 3, "lod_select_topdown: cam_pos must contain 3 values");
    TORCH_CHECK(w2c.sizes() == at::IntArrayRef({4, 4}), "lod_select_topdown: w2c must be [4, 4]");
    TORCH_CHECK(K.sizes() == at::IntArrayRef({3, 3}), "lod_select_topdown: K must be [3, 3]");
    TORCH_CHECK(centers.scalar_type() == at::kFloat, "lod_select_topdown: centers must be float32");
    TORCH_CHECK(sizes.scalar_type() == at::kFloat, "lod_select_topdown: sizes must be float32");
    TORCH_CHECK(radii.scalar_type() == at::kFloat, "lod_select_topdown: radii must be float32");
    TORCH_CHECK(children.scalar_type() == at::kInt, "lod_select_topdown: children must be int32");
    TORCH_CHECK(is_leaf.scalar_type() == at::kBool, "lod_select_topdown: is_leaf must be bool");
    TORCH_CHECK(root_ids.scalar_type() == at::kInt, "lod_select_topdown: root_ids must be int32");
    TORCH_CHECK(cam_pos.scalar_type() == at::kFloat, "lod_select_topdown: cam_pos must be float32");
    TORCH_CHECK(w2c.scalar_type() == at::kFloat, "lod_select_topdown: w2c must be float32");
    TORCH_CHECK(K.scalar_type() == at::kFloat, "lod_select_topdown: K must be float32");
    TORCH_CHECK(M > 0, "lod_select_topdown: hierarchy must not be empty");
    TORCH_CHECK(image_width > 0 && image_height > 0, "lod_select_topdown: image dimensions must be positive");
    TORCH_CHECK(max_depth >= 0 && max_depth < 31, "lod_select_topdown: max_depth must be in [0, 30]");

    auto int_options = centers.options().dtype(at::kInt);
    auto proxy_ids = at::empty({M}, int_options);
    auto leaf_ids = at::empty({M}, int_options);
    auto counts = at::zeros({4}, int_options);

    launch_lod_select_topdown_kernels(
        centers,
        sizes,
        radii,
        children,
        is_leaf,
        root_ids,
        cam_pos,
        w2c,
        K,
        static_cast<int>(image_width),
        static_cast<int>(image_height),
        static_cast<float>(near_plane),
        static_cast<float>(error_threshold_px),
        static_cast<int>(max_depth),
        proxy_ids,
        leaf_ids,
        counts
    );

    // One synchronization replaces the Python selector's multiple dynamic-size
    // synchronizations. Compact outputs avoid retaining the full temporary
    // buffers after this call returns.
    auto counts_cpu = counts.to(at::kCPU);
    const auto *count_data = counts_cpu.const_data_ptr<int32_t>();
    const int64_t proxy_count = count_data[2];
    const int64_t leaf_count = count_data[3];
    const int next_frontier_slot = (static_cast<int>(max_depth) & 1) ^ 1;
    TORCH_CHECK(
        count_data[next_frontier_slot] == 0,
        "lod_select_topdown: max_depth is shallower than the BVH; ",
        count_data[next_frontier_slot],
        " internal nodes remain"
    );
    TORCH_CHECK(proxy_count <= M && leaf_count <= M, "lod_select_topdown: output overflow");

    return {
        proxy_ids.narrow(0, 0, proxy_count).clone(),
        leaf_ids.narrow(0, 0, leaf_count).clone()
    };
}

std::tuple<at::Tensor, int64_t, int64_t> lod_select_active_topdown(
    const at::Tensor &centers,
    const at::Tensor &sizes,
    const at::Tensor &radii,
    const at::Tensor &children,
    const at::Tensor &is_leaf,
    const at::Tensor &leaf_starts,
    const at::Tensor &leaf_lengths,
    const at::Tensor &root_ids,
    const at::Tensor &cam_pos,
    const at::Tensor &w2c,
    const at::Tensor &K,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double error_threshold_px,
    int64_t max_depth,
    int64_t exact_capacity,
    int64_t proxy_pool_offset
)
{
    DEVICE_GUARD(centers);
    CHECK_INPUT(centers);
    CHECK_INPUT(sizes);
    CHECK_INPUT(radii);
    CHECK_INPUT(children);
    CHECK_INPUT(is_leaf);
    CHECK_INPUT(leaf_starts);
    CHECK_INPUT(leaf_lengths);
    CHECK_INPUT(root_ids);
    CHECK_INPUT(cam_pos);
    CHECK_INPUT(w2c);
    CHECK_INPUT(K);
    const int64_t M = centers.size(0);
    TORCH_CHECK(
        centers.dim() == 2 && centers.size(1) == 3,
        "lod_select_active_topdown: centers must be [M, 3]"
    );
    TORCH_CHECK(
        sizes.sizes() == at::IntArrayRef({M})
            && radii.sizes() == at::IntArrayRef({M}),
        "lod_select_active_topdown: sizes and radii must be [M]"
    );
    TORCH_CHECK(
        children.sizes() == at::IntArrayRef({M, 2}),
        "lod_select_active_topdown: children must be [M, 2]"
    );
    TORCH_CHECK(
        is_leaf.sizes() == at::IntArrayRef({M}),
        "lod_select_active_topdown: is_leaf must be [M]"
    );
    TORCH_CHECK(
        leaf_starts.scalar_type() == at::kLong
            && leaf_lengths.scalar_type() == at::kLong,
        "lod_select_active_topdown: leaf ranges must be int64"
    );
    TORCH_CHECK(
        leaf_starts.sizes() == at::IntArrayRef({centers.size(0)})
            && leaf_lengths.sizes() == at::IntArrayRef({centers.size(0)}),
        "lod_select_active_topdown: leaf ranges must be [M]"
    );
    TORCH_CHECK(
        root_ids.dim() == 1 && root_ids.numel() > 0,
        "lod_select_active_topdown: root_ids must be non-empty [R]"
    );
    TORCH_CHECK(
        cam_pos.numel() == 3,
        "lod_select_active_topdown: cam_pos must contain 3 values"
    );
    TORCH_CHECK(
        w2c.sizes() == at::IntArrayRef({4, 4})
            && K.sizes() == at::IntArrayRef({3, 3}),
        "lod_select_active_topdown: invalid camera matrix shape"
    );
    TORCH_CHECK(
        centers.scalar_type() == at::kFloat
            && sizes.scalar_type() == at::kFloat
            && radii.scalar_type() == at::kFloat
            && cam_pos.scalar_type() == at::kFloat
            && w2c.scalar_type() == at::kFloat
            && K.scalar_type() == at::kFloat,
        "lod_select_active_topdown: geometry and camera tensors must be float32"
    );
    TORCH_CHECK(
        children.scalar_type() == at::kInt
            && root_ids.scalar_type() == at::kInt,
        "lod_select_active_topdown: children and root_ids must be int32"
    );
    TORCH_CHECK(
        is_leaf.scalar_type() == at::kBool,
        "lod_select_active_topdown: is_leaf must be bool"
    );
    TORCH_CHECK(M > 0, "lod_select_active_topdown: hierarchy must not be empty");
    TORCH_CHECK(
        image_width > 0 && image_height > 0,
        "lod_select_active_topdown: image dimensions must be positive"
    );
    TORCH_CHECK(
        max_depth >= 0 && max_depth < 31,
        "lod_select_active_topdown: max_depth must be in [0, 30]"
    );
    TORCH_CHECK(
        exact_capacity >= 0 && proxy_pool_offset >= exact_capacity,
        "lod_select_active_topdown: invalid pool offsets"
    );
    TORCH_CHECK(
        exact_capacity + M <= std::numeric_limits<int32_t>::max()
            && proxy_pool_offset + M <= std::numeric_limits<int32_t>::max(),
        "lod_select_active_topdown: pool exceeds int32 index range"
    );

    auto options = centers.options().dtype(at::kInt);
    auto active_ids = at::empty(
        {exact_capacity + centers.size(0)},
        options
    );
    auto counts = at::zeros({6}, options);
    launch_lod_select_active_topdown_kernels(
        centers,
        sizes,
        radii,
        children,
        is_leaf,
        leaf_starts,
        leaf_lengths,
        root_ids,
        cam_pos,
        w2c,
        K,
        static_cast<int>(image_width),
        static_cast<int>(image_height),
        static_cast<float>(near_plane),
        static_cast<float>(error_threshold_px),
        static_cast<int>(max_depth),
        static_cast<int>(exact_capacity),
        static_cast<int>(proxy_pool_offset),
        active_ids,
        counts
    );
    auto counts_cpu = counts.to(at::kCPU);
    const auto *data = counts_cpu.const_data_ptr<int32_t>();
    const int64_t active_count = data[2];
    const int64_t exact_leaf_count = data[4];
    const int64_t proxy_count = data[5];
    const int next_slot = (static_cast<int>(max_depth) & 1) ^ 1;
    TORCH_CHECK(
        data[next_slot] == 0,
        "lod_select_active_topdown: max_depth is shallower than the BVH"
    );
    TORCH_CHECK(
        data[3] == 0,
        "lod_select_active_topdown: active output overflow"
    );
    TORCH_CHECK(
        active_count <= active_ids.size(0),
        "lod_select_active_topdown: output overflow"
    );
    return {
        active_ids.narrow(0, 0, active_count).clone(),
        exact_leaf_count,
        proxy_count
    };
}

void register_lod_select_cuda_impl(torch::Library &m)
{
    m.impl("lod_build_binary_children", &lod_build_binary_children);
    m.impl("lod_select_topdown", &lod_select_topdown);
    m.impl("lod_select_active_topdown", &lod_select_active_topdown);
}
} // namespace gsplat
