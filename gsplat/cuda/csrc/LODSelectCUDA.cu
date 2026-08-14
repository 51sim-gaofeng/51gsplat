/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

#include "LODSelect.h"

namespace gsplat
{
namespace
{
constexpr int kThreads = 256;

__global__ void build_binary_children_kernel(
    int64_t num_nodes,
    const int64_t *__restrict__ parents,
    int32_t *__restrict__ children
)
{
    const int64_t node = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if(node >= num_nodes)
    {
        return;
    }

    const int64_t parent = parents[node];
    if(parent < 0 || parent >= num_nodes)
    {
        return;
    }

    // The flat LOD cache is preorder: the first child immediately follows its
    // parent; the other child follows the complete first-child subtree.
    const int slot = node == parent + 1 ? 0 : 1;
    children[parent * 2 + slot] = static_cast<int32_t>(node);
}

__global__ void initialize_frontier_kernel(
    int32_t *__restrict__ frontier,
    int32_t *__restrict__ counts,
    const float *__restrict__ K,
    int width,
    int height,
    float near_plane,
    float *__restrict__ frustum_planes
)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        frontier[0] = 0;
        counts[0] = 1;
        counts[1] = 0;
        counts[2] = 0;
        counts[3] = 0;

        const float fx = K[0];
        const float fy = K[4];
        const float cx = K[2];
        const float cy = K[5];
        const float right_z = static_cast<float>(width) - cx;
        const float bottom_z = static_cast<float>(height) - cy;
        const float left_norm = rsqrtf(fx * fx + cx * cx);
        const float right_norm = rsqrtf(fx * fx + right_z * right_z);
        const float top_norm = rsqrtf(fy * fy + cy * cy);
        const float bottom_norm = rsqrtf(fy * fy + bottom_z * bottom_z);

        frustum_planes[0] = fx * left_norm;
        frustum_planes[1] = 0.0f;
        frustum_planes[2] = cx * left_norm;
        frustum_planes[3] = 0.0f;
        frustum_planes[4] = -fx * right_norm;
        frustum_planes[5] = 0.0f;
        frustum_planes[6] = right_z * right_norm;
        frustum_planes[7] = 0.0f;
        frustum_planes[8] = 0.0f;
        frustum_planes[9] = fy * top_norm;
        frustum_planes[10] = cy * top_norm;
        frustum_planes[11] = 0.0f;
        frustum_planes[12] = 0.0f;
        frustum_planes[13] = -fy * bottom_norm;
        frustum_planes[14] = bottom_z * bottom_norm;
        frustum_planes[15] = 0.0f;
        frustum_planes[16] = 0.0f;
        frustum_planes[17] = 0.0f;
        frustum_planes[18] = 1.0f;
        frustum_planes[19] = -near_plane;
    }
}

__device__ __forceinline__ bool sphere_visible(
    const float x,
    const float y,
    const float z,
    const float radius,
    const float *w2c,
    const float *frustum_planes
)
{
    const float px = w2c[0] * x + w2c[1] * y + w2c[2] * z + w2c[3];
    const float py = w2c[4] * x + w2c[5] * y + w2c[6] * z + w2c[7];
    const float pz = w2c[8] * x + w2c[9] * y + w2c[10] * z + w2c[11];

#pragma unroll
    for(int plane_index = 0; plane_index < 5; ++plane_index)
    {
        const float *plane = frustum_planes + plane_index * 4;
        if(plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < -radius)
        {
            return false;
        }
    }
    return true;
}

__global__ void traverse_level_kernel(
    int32_t launch_capacity,
    const int32_t *__restrict__ current_count,
    const int32_t *__restrict__ current_frontier,
    int32_t *__restrict__ next_count,
    int32_t *__restrict__ next_frontier,
    const float *__restrict__ centers,
    const float *__restrict__ sizes,
    const float *__restrict__ radii,
    const int32_t *__restrict__ children,
    const bool *__restrict__ is_leaf,
    const float *__restrict__ cam_pos,
    const float *__restrict__ w2c,
    const float *__restrict__ K,
    const float *__restrict__ frustum_planes,
    float error_threshold_px,
    int32_t *__restrict__ proxy_count,
    int32_t *__restrict__ proxy_ids,
    int32_t *__restrict__ leaf_count,
    int32_t *__restrict__ leaf_ids
)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= launch_capacity || index >= *current_count)
    {
        return;
    }

    const int32_t node = current_frontier[index];
    const float x = centers[node * 3];
    const float y = centers[node * 3 + 1];
    const float z = centers[node * 3 + 2];
    if(!sphere_visible(x, y, z, radii[node], w2c, frustum_planes))
    {
        return;
    }

    const float dx = x - cam_pos[0];
    const float dy = y - cam_pos[1];
    const float dz = z - cam_pos[2];
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1e-6f);
    const float projected = 2.0f * sizes[node] * K[0] / distance;

    if(projected <= error_threshold_px)
    {
        proxy_ids[atomicAdd(proxy_count, 1)] = node;
        return;
    }

    if(is_leaf[node])
    {
        leaf_ids[atomicAdd(leaf_count, 1)] = node;
        return;
    }

    const int32_t left = children[node * 2];
    const int32_t right = children[node * 2 + 1];
    if(left >= 0 && right >= 0)
    {
        const int32_t output = atomicAdd(next_count, 2);
        next_frontier[output] = left;
        next_frontier[output + 1] = right;
    }
}
} // namespace

void launch_lod_build_binary_children_kernel(
    const at::Tensor &parents,
    at::Tensor &children
)
{
    const int64_t num_nodes = parents.size(0);
    if(num_nodes <= 1)
    {
        return;
    }

    const int blocks = static_cast<int>((num_nodes - 1 + kThreads - 1) / kThreads);
    build_binary_children_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_nodes,
        parents.const_data_ptr<int64_t>(),
        children.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
)
{
    const int64_t num_nodes = centers.size(0);
    const int64_t frontier_capacity = (num_nodes + 1) / 2;
    auto frontier_options = centers.options().dtype(at::kInt);
    auto frontier_a = at::empty({frontier_capacity}, frontier_options);
    auto frontier_b = at::empty({frontier_capacity}, frontier_options);
    auto frustum_planes = at::empty({5, 4}, centers.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    initialize_frontier_kernel<<<1, 1, 0, stream>>>(
        frontier_a.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        K.const_data_ptr<float>(),
        image_width,
        image_height,
        near_plane,
        frustum_planes.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for(int level = 0; level <= max_depth; ++level)
    {
        const int current_slot = level & 1;
        const int next_slot = current_slot ^ 1;
        auto &current_frontier = current_slot == 0 ? frontier_a : frontier_b;
        auto &next_frontier = next_slot == 0 ? frontier_a : frontier_b;

        C10_CUDA_CHECK(cudaMemsetAsync(
            counts.data_ptr<int32_t>() + next_slot,
            0,
            sizeof(int32_t),
            stream
        ));

        const int64_t theoretical_width = int64_t{1} << level;
        const int32_t launch_capacity = static_cast<int32_t>(
            std::min(theoretical_width, frontier_capacity)
        );
        const int blocks = (launch_capacity + kThreads - 1) / kThreads;
        traverse_level_kernel<<<blocks, kThreads, 0, stream>>>(
            launch_capacity,
            counts.const_data_ptr<int32_t>() + current_slot,
            current_frontier.const_data_ptr<int32_t>(),
            counts.data_ptr<int32_t>() + next_slot,
            next_frontier.data_ptr<int32_t>(),
            centers.const_data_ptr<float>(),
            sizes.const_data_ptr<float>(),
            radii.const_data_ptr<float>(),
            children.const_data_ptr<int32_t>(),
            is_leaf.const_data_ptr<bool>(),
            cam_pos.const_data_ptr<float>(),
            w2c.const_data_ptr<float>(),
            K.const_data_ptr<float>(),
            frustum_planes.const_data_ptr<float>(),
            error_threshold_px,
            counts.data_ptr<int32_t>() + 2,
            proxy_ids.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>() + 3,
            leaf_ids.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
} // namespace gsplat
