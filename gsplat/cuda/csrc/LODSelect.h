/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace at
{
class Tensor;
}

namespace gsplat
{
void launch_lod_build_binary_children_kernel(
    const at::Tensor &parents,
    at::Tensor &children
);

void launch_lod_select_topdown_kernels(
    const at::Tensor &centers,
    const at::Tensor &sizes,
    const at::Tensor &radii,
    const at::Tensor &children,
    const at::Tensor &is_leaf,
    const at::Tensor &cam_pos,
    const at::Tensor &w2c,
    const at::Tensor &K,
    int image_width,
    int image_height,
    float near_plane,
    float error_threshold_px,
    int max_depth,
    at::Tensor &proxy_ids,
    at::Tensor &leaf_ids,
    at::Tensor &counts
);
} // namespace gsplat
