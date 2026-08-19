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
    const int32_t *__restrict__ root_ids,
    int32_t root_count,
    const float *__restrict__ K,
    int width,
    int height,
    float near_plane,
    float *__restrict__ frustum_planes
)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < root_count)
    {
        frontier[index] = root_ids[index];
    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        counts[0] = root_count;
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

// Node visibility + on-screen size that matches the render camera model.
// PINHOLE keeps the frustum-plane test; FISHEYE (camera_model==2) uses OpenCV
// 4-coeff equidistant projection (r = f*theta_d) so wide-FoV edges are not
// clipped by a pinhole frustum and proxy sizing matches the fisheye render.
__device__ __forceinline__ void node_visibility_and_size(
    const float x,
    const float y,
    const float z,
    const float radius,
    const float size,
    const float *cam_pos,
    const float *w2c,
    const float *K,
    const float *frustum_planes,
    const int camera_model,
    const float k1,
    const float k2,
    const float k3,
    const float k4,
    const int image_width,
    const int image_height,
    const float near_full_dist,
    bool &visible,
    float &projected
)
{
    const float dx = x - cam_pos[0];
    const float dy = y - cam_pos[1];
    const float dz = z - cam_pos[2];
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1e-6f);

    if(camera_model == 2) // FISHEYE
    {
        const float px = w2c[0] * x + w2c[1] * y + w2c[2] * z + w2c[3];
        const float py = w2c[4] * x + w2c[5] * y + w2c[6] * z + w2c[7];
        const float pz = w2c[8] * x + w2c[9] * y + w2c[10] * z + w2c[11];
        const float fx = K[0];
        const float fy = K[4];
        const float cx = K[2];
        const float cy = K[5];
        const float rho = sqrtf(px * px + py * py);
        const float theta = atan2f(rho, pz);
        // Angular-cone visibility: the fisheye field is a cone, not a pixel box.
        // A pixel-box test drops near-camera nodes whose center projects just
        // outside the frame (steep ground under a narrow vertical FoV) even
        // though the node still covers on-screen pixels -> holes. Bound by the
        // image-corner angle (equidistant approx) and widen by the node's
        // angular radius so boundary-straddling nodes are kept.
        const float corner_x = fmaxf(cx, static_cast<float>(image_width) - cx);
        const float corner_y = fmaxf(cy, static_cast<float>(image_height) - cy);
        const float r_corner = sqrtf(corner_x * corner_x + corner_y * corner_y);
        const float theta_max = r_corner / fmaxf(fminf(fx, fy), 1e-6f);
        const float ang_r = radius / distance;
        visible = (theta - ang_r) <= theta_max;
        const float t2 = theta * theta;
        const float t4 = t2 * t2;
        const float t6 = t4 * t2;
        const float t8 = t4 * t4;
        const float s_prime = 1.0f + 3.0f * k1 * t2 + 5.0f * k2 * t4
            + 7.0f * k3 * t6 + 9.0f * k4 * t8;
        projected = 2.0f * size / distance * fx * fmaxf(s_prime, 0.05f);
    }
    else
    {
        visible = sphere_visible(x, y, z, radius, w2c, frustum_planes);
        projected = 2.0f * size * K[0] / distance;
    }
    // Near nodes: force refinement to exact leaves so close-up detail is kept.
    if(near_full_dist > 0.0f && distance < near_full_dist)
    {
        projected = 1e30f;
    }
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
    int camera_model,
    float k1,
    float k2,
    float k3,
    float k4,
    int image_width,
    int image_height,
    float error_threshold_px,
    float near_full_dist,
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
    bool visible;
    float projected;
    node_visibility_and_size(
        x, y, z, radii[node], sizes[node],
        cam_pos, w2c, K, frustum_planes,
        camera_model, k1, k2, k3, k4, image_width, image_height,
        near_full_dist,
        visible, projected);
    if(!visible)
    {
        return;
    }

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

__global__ void traverse_level_active_kernel(
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
    const int64_t *__restrict__ leaf_starts,
    const int64_t *__restrict__ leaf_lengths,
    const float *__restrict__ cam_pos,
    const float *__restrict__ w2c,
    const float *__restrict__ K,
    const float *__restrict__ frustum_planes,
    int camera_model,
    float k1,
    float k2,
    float k3,
    float k4,
    int image_width,
    int image_height,
    float error_threshold_px,
    float near_full_dist,
    int32_t exact_capacity,
    int32_t proxy_pool_offset,
    int32_t active_capacity,
    int32_t *__restrict__ active_count,
    int32_t *__restrict__ overflow,
    int32_t *__restrict__ exact_leaf_count,
    int32_t *__restrict__ proxy_count,
    int32_t *__restrict__ active_ids
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
    bool visible;
    float projected;
    node_visibility_and_size(
        x, y, z, radii[node], sizes[node],
        cam_pos, w2c, K, frustum_planes,
        camera_model, k1, k2, k3, k4, image_width, image_height,
        near_full_dist,
        visible, projected);
    if(!visible)
    {
        return;
    }

    if(projected <= error_threshold_px)
    {
        atomicAdd(proxy_count, 1);
        const int32_t output = atomicAdd(active_count, 1);
        if(output < active_capacity)
        {
            active_ids[output] = proxy_pool_offset + node;
        }
        else
        {
            atomicExch(overflow, 1);
        }
        return;
    }

    if(is_leaf[node])
    {
        const int64_t start64 = leaf_starts[node];
        const int64_t length64 = leaf_lengths[node];
        if(start64 < 0 || length64 < 0 || start64 + length64 > exact_capacity)
        {
            atomicExch(overflow, 1);
            return;
        }
        const int32_t length = static_cast<int32_t>(length64);
        atomicAdd(exact_leaf_count, 1);
        const int32_t output = atomicAdd(active_count, length);
        const int32_t start = static_cast<int32_t>(start64);
        if(output <= active_capacity - length)
        {
            for(int32_t offset = 0; offset < length; ++offset)
            {
                active_ids[output + offset] = start + offset;
            }
        }
        else
        {
            atomicExch(overflow, 1);
        }
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
    const at::Tensor &root_ids,
    const at::Tensor &cam_pos,
    const at::Tensor &w2c,
    const at::Tensor &K,
    int camera_model,
    float k1,
    float k2,
    float k3,
    float k4,
    int image_width,
    int image_height,
    float near_plane,
    float error_threshold_px,
    float near_full_dist,
    int max_depth,
    at::Tensor &proxy_ids,
    at::Tensor &leaf_ids,
    at::Tensor &counts
)
{
    const int64_t num_nodes = centers.size(0);
    const int64_t root_count = root_ids.size(0);
    const int64_t frontier_capacity = num_nodes;
    auto frontier_options = centers.options().dtype(at::kInt);
    auto frontier_a = at::empty({frontier_capacity}, frontier_options);
    auto frontier_b = at::empty({frontier_capacity}, frontier_options);
    auto frustum_planes = at::empty({5, 4}, centers.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int initialize_blocks = static_cast<int>(
        (root_count + kThreads - 1) / kThreads
    );
    initialize_frontier_kernel<<<initialize_blocks, kThreads, 0, stream>>>(
        frontier_a.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        root_ids.const_data_ptr<int32_t>(),
        static_cast<int32_t>(root_count),
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

        const int64_t theoretical_width = root_count * (int64_t{1} << level);
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
            camera_model,
            k1, k2, k3, k4,
            image_width,
            image_height,
            error_threshold_px,
            near_full_dist,
            counts.data_ptr<int32_t>() + 2,
            proxy_ids.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>() + 3,
            leaf_ids.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void launch_lod_select_active_topdown_kernels(
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
    int camera_model,
    float k1,
    float k2,
    float k3,
    float k4,
    int image_width,
    int image_height,
    float near_plane,
    float error_threshold_px,
    float near_full_dist,
    int max_depth,
    int exact_capacity,
    int proxy_pool_offset,
    at::Tensor &active_ids,
    at::Tensor &counts
)
{
    const int64_t num_nodes = centers.size(0);
    const int64_t root_count = root_ids.size(0);
    auto options = centers.options().dtype(at::kInt);
    auto frontier_a = at::empty({num_nodes}, options);
    auto frontier_b = at::empty({num_nodes}, options);
    auto frustum_planes = at::empty({5, 4}, centers.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int initialize_blocks
        = static_cast<int>((root_count + kThreads - 1) / kThreads);
    initialize_frontier_kernel<<<initialize_blocks, kThreads, 0, stream>>>(
        frontier_a.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        root_ids.const_data_ptr<int32_t>(),
        static_cast<int32_t>(root_count),
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
        auto &current = current_slot == 0 ? frontier_a : frontier_b;
        auto &next = next_slot == 0 ? frontier_a : frontier_b;
        C10_CUDA_CHECK(cudaMemsetAsync(
            counts.data_ptr<int32_t>() + next_slot,
            0,
            sizeof(int32_t),
            stream
        ));
        const int64_t theoretical_width = root_count * (int64_t{1} << level);
        const int32_t capacity = static_cast<int32_t>(
            std::min(theoretical_width, num_nodes)
        );
        const int blocks = (capacity + kThreads - 1) / kThreads;
        traverse_level_active_kernel<<<blocks, kThreads, 0, stream>>>(
            capacity,
            counts.const_data_ptr<int32_t>() + current_slot,
            current.const_data_ptr<int32_t>(),
            counts.data_ptr<int32_t>() + next_slot,
            next.data_ptr<int32_t>(),
            centers.const_data_ptr<float>(),
            sizes.const_data_ptr<float>(),
            radii.const_data_ptr<float>(),
            children.const_data_ptr<int32_t>(),
            is_leaf.const_data_ptr<bool>(),
            leaf_starts.const_data_ptr<int64_t>(),
            leaf_lengths.const_data_ptr<int64_t>(),
            cam_pos.const_data_ptr<float>(),
            w2c.const_data_ptr<float>(),
            K.const_data_ptr<float>(),
            frustum_planes.const_data_ptr<float>(),
            camera_model,
            k1, k2, k3, k4,
            image_width,
            image_height,
            error_threshold_px,
            near_full_dist,
            exact_capacity,
            proxy_pool_offset,
            static_cast<int32_t>(active_ids.size(0)),
            counts.data_ptr<int32_t>() + 2,
            counts.data_ptr<int32_t>() + 3,
            counts.data_ptr<int32_t>() + 4,
            counts.data_ptr<int32_t>() + 5,
            active_ids.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
} // namespace gsplat
