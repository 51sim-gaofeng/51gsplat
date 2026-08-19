/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include "Common.h"
#include "Constants.h"
#include "SHCommon.h"
#include "SHCompression.h"
#include "Utils.cuh"
#include "Projection.h"
#include "Utils.h"

namespace higs
{
static constexpr int CTA_SIZE         = 256;
static constexpr int PROJ_MIN_BLOCKS  = 6;
static constexpr int FUSED_MIN_BLOCKS = 4;

using gsplat::CameraModelType;
using gsplat::mat2;
using gsplat::mat3;
using gsplat::vec2;
using gsplat::vec3;
using gsplat::vec4;

// OpenCV fisheye (4-coefficient) distortion applied on top of equidistant fisheye_proj.
// theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8), t = theta.
// Covariance uses chain-rule scaling of the fisheye Jacobian: direct-theta terms scale
// by s = theta_d/theta; d(theta)/d(x,y,z) terms scale by phi' = d(theta_d)/d(theta).
// With k1..k4 == 0 this reduces exactly to gsplat::fisheye_proj.
inline __device__ void fisheye_proj_distorted(
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t /*width*/,
    const uint32_t /*height*/,
    const float k1,
    const float k2,
    const float k3,
    const float k4,
    mat2 &cov2d,
    vec2 &mean2d)
{
    const float x = mean3d[0], y = mean3d[1], z = mean3d[2];
    const float eps = 1e-7f;
    const float xy_len = glm::length(glm::vec2({x, y})) + eps;
    const float theta  = glm::atan(xy_len, z + eps);

    const float t2 = theta * theta;
    const float t4 = t2 * t2;
    const float t6 = t4 * t2;
    const float t8 = t4 * t4;
    const float s        = 1.f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
    const float s_prime  = 2.f * k1 * theta + 4.f * k2 * theta * t2
                         + 6.f * k3 * theta * t4 + 8.f * k4 * theta * t6;
    const float phi        = theta * s;
    const float phi_prime  = s + theta * s_prime;

    mean2d = vec2({x * fx * phi / xy_len + cx, y * fy * phi / xy_len + cy});

    const float x2         = x * x + eps;
    const float y2         = y * y;
    const float xy         = x * y;
    const float x2y2       = x2 + y2;
    const float x2y2z2_inv = 1.f / (x2y2 + z * z);

    const float a = z * x2y2z2_inv / x2y2 * phi_prime;
    const float b = theta / xy_len / x2y2 * s;
    glm::mat3x2 J = glm::mat3x2(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv * phi_prime,
        -fy * y * x2y2z2_inv * phi_prime
    );
    cov2d = J * cov3d * glm::transpose(J);
}

template<bool FUSE_SH, SHInputMode SH_MODE = SHInputMode::RAW_HALF>
__global__ void __launch_bounds__(CTA_SIZE, FUSE_SH ? FUSED_MIN_BLOCKS : PROJ_MIN_BLOCKS) projection_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t source_N,
    const float *__restrict__ means,      // [B, 3, N]
    const float *__restrict__ covars,     // [B, N, 6] optional
    const __half *__restrict__ inference, // [B, N, 8] half — packed {quat(4), scale(3), opacity(1)}
    const void *__restrict__ active_indices,
    const bool active_indices_int64,
    const __half *__restrict__ source_colors,
    const float *__restrict__ viewmats,   // [B, C, 4, 4]
    const float *__restrict__ Ks,         // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    const float k1,
    const float k2,
    const float k3,
    const float k4,
    // outputs
    uint32_t *__restrict__ visible,    // [(B*C*N+31)/32] packed bitfield
    float *__restrict__ means2d,       // [B, C, N, 2]
    float *__restrict__ depths,        // [B, C, N]
    __half *__restrict__ conics,       // [B, C, N, 4] half {l0,l1,l2,opacity}
    float *__restrict__ compensations, // [B, C, N] optional
    // SH parameters (used only when FUSE_SH=true; dead-code-eliminated otherwise)
    const int32_t degrees_to_use,
    const void *__restrict__ sh_input,
    const float sh_bias,
    const float sh_min_value,
    const SHDecodeParams sh_decode_params,
    __half *__restrict__ colors // [N, 4] half {R,G,B,0}
)
{
    // parallelize over B * C * N.
    // OOB threads must not early-return — they participate in __ballot_sync.
    const int32_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t lane = threadIdx.x & 31u;

    auto processGaussian = [&]() -> bool
    {
        if(idx >= B * C * N)
        {
            return false;
        }
        const int32_t bid = idx / (C * N); // batch id
        const int32_t cid = (idx / N) % C; // camera id
        const int32_t gid = idx % N;       // gaussian id
        const int64_t source_idx
            = active_indices == nullptr
                ? gid
                : (active_indices_int64 ? reinterpret_cast<const int64_t *>(active_indices)[gid]
                                        : reinterpret_cast<const int32_t *>(active_indices)[gid]);
        if(source_idx < 0 || source_idx >= source_N)
        {
            return false;
        }
        const int32_t source_gid = static_cast<int32_t>(source_idx);

        // shift pointers to the current camera and gaussian
        const float *means_b    = means + bid * 3 * source_N;
        const float *viewmats_b = viewmats + bid * C * 16 + cid * 16;
        const float *Ks_b       = Ks + bid * C * 9 + cid * 9;

        // planar [B, 3, N] layout: coalesced reads across threads
        const vec3 mean_w
            = vec3(means_b[source_gid], means_b[source_N + source_gid], means_b[2 * source_N + source_gid]);

        if(source_colors != nullptr)
        {
            AssignAs<uint2>(colors[gid * 4], source_colors[source_gid * 4]);
        }

        // glm is column-major but input is row-major
        const mat3 R = mat3(
            viewmats_b[0],
            viewmats_b[4],
            viewmats_b[8], // 1st column
            viewmats_b[1],
            viewmats_b[5],
            viewmats_b[9], // 2nd column
            viewmats_b[2],
            viewmats_b[6],
            viewmats_b[10] // 3rd column
        );
        const vec3 t = vec3(viewmats_b[3], viewmats_b[7], viewmats_b[11]);

        // transform Gaussian center to camera space
        vec3 mean_c;
        gsplat::posW2C(R, t, mean_w, mean_c);
        // For FISHEYE, use radial distance so back-hemisphere gaussians (z<0) at
        // FoV > 180 pass; other models keep the pinhole-style z-depth semantics.
        const float depth_val = (camera_model == CameraModelType::FISHEYE)
            ? sqrtf(mean_c.x * mean_c.x + mean_c.y * mean_c.y + mean_c.z * mean_c.z)
            : mean_c.z;
        if(depth_val < near_plane || depth_val > far_plane)
        {
            return false;
        }

        // Wide 128-bit load of packed {quat(4), scale(3), opacity(1)} in half
        __half inference_local[8];
        AssignAs<uint4>(inference_local[0], inference[(bid * source_N + source_gid) * 8]);

        const vec4 quat = vec4(
            __half2float(inference_local[0]),
            __half2float(inference_local[1]),
            __half2float(inference_local[2]),
            __half2float(inference_local[3])
        );
        const vec3 scale = vec3(
            __half2float(inference_local[4]), __half2float(inference_local[5]), __half2float(inference_local[6])
        );
        float opacity = __half2float(inference_local[7]);

        // transform Gaussian covariance to camera space
        mat3 covar;
        if(covars != nullptr)
        {
            const float *covars_g = covars + bid * source_N * 6 + source_gid * 6;

            covar = mat3(
                covars_g[0],
                covars_g[1],
                covars_g[2], // 1st column
                covars_g[1],
                covars_g[3],
                covars_g[4], // 2nd column
                covars_g[2],
                covars_g[4],
                covars_g[5] // 3rd column
            );
        }
        else
        {
            gsplat::quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
        }

        mat3 covar_c;
        gsplat::covarW2C(R, covar, covar_c);

        // perspective projection
        mat2 covar2d;
        vec2 mean2d;

        switch(camera_model)
        {
        case CameraModelType::PINHOLE:
            gsplat::persp_proj(
                mean_c, covar_c, Ks_b[0], Ks_b[4], Ks_b[2], Ks_b[5], image_width, image_height, covar2d, mean2d
            );
            break;
        case CameraModelType::ORTHO:
            gsplat::ortho_proj(
                mean_c, covar_c, Ks_b[0], Ks_b[4], Ks_b[2], Ks_b[5], image_width, image_height, covar2d, mean2d
            );
            break;
        case CameraModelType::FISHEYE:
            fisheye_proj_distorted(
                mean_c, covar_c, Ks_b[0], Ks_b[4], Ks_b[2], Ks_b[5], image_width, image_height,
                k1, k2, k3, k4, covar2d, mean2d
            );
            break;
        }

        float compensation;
        const float det = gsplat::add_blur(eps2d, covar2d, compensation);
        if(det <= 0.f)
        {
            return false;
        }

        // compute the inverse of the 2d covariance
        const mat2 covar2d_inv = glm::inverse(covar2d);

        float extend = MAX_EXTEND;
        if(compensations != nullptr)
        {
            opacity *= compensation;
        }
        if(opacity < ALPHA_THRESHOLD)
        {
            return false;
        }
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));

        // compute tight rectangular bounding box
        float radius_x = extend * sqrtf(covar2d[0][0]);
        float radius_y = extend * sqrtf(covar2d[1][1]);

        // let's do the radius clip test prior ceil for finer fidelity of clipping
        if(radius_x <= radius_clip && radius_y <= radius_clip)
        {
            return false;
        }

        radius_x = ceilf(radius_x);
        radius_y = ceilf(radius_y);

        // mask out gaussians outside the image region
        if(mean2d.x + radius_x <= 0
           || mean2d.x - radius_x >= image_width
           || mean2d.y + radius_y <= 0
           || mean2d.y - radius_y >= image_height)
        {
            return false;
        }

        // write to outputs
        AssignAs<float2>(means2d[idx * 2], mean2d);
        depths[idx]       = depth_val;
        // Store Cholesky factor L of the inverse covariance: Sigma^{-1} = L*L^T,
        // L = [[l0, 0], [l1, l2]].  Consumers reconstruct A=l0², B=l0*l1,
        // C=l1²+l2² when needed.  This lets the rasterizer use the factors
        // directly for a numerically stable sum-of-squares sigma in fp16.
        const float ci_00 = covar2d_inv[0][0];
        const float ci_01 = covar2d_inv[0][1];
        const float ci_11 = covar2d_inv[1][1];
        const float l0    = sqrtf(fmaxf(ci_00, 0.f));
        const float l1    = (l0 > 1e-12f) ? ci_01 / l0 : 0.f;
        const float l2    = sqrtf(fmaxf(ci_11 - l1 * l1, 0.f));

        __half2 conics_opac[2];
        conics_opac[0] = __floats2half2_rn(l0, l1);
        conics_opac[1] = __floats2half2_rn(l2, opacity);
        AssignAs<uint2>(conics[idx * 4], conics_opac);

        if(compensations != nullptr)
        {
            compensations[idx] = compensation;
        }

        // Fused SH evaluation: mean_w, R, t are still in scope from the projection load above.
        if constexpr(FUSE_SH)
        {
            // Camera world position: cam_pos = -R^T * t (R is column-major glm mat3)
            const vec3 cam_w   = -glm::transpose(R) * t;
            const float dx     = mean_w.x - cam_w.x;
            const float dy     = mean_w.y - cam_w.y;
            const float dz     = mean_w.z - cam_w.z;
            const float inorm  = rsqrtf(dx * dx + dy * dy + dz * dz);
            const float3 dir_n = make_float3(dx * inorm, dy * inorm, dz * inorm);
            EvalSHForGaussian<SH_MODE>(
                dir_n,
                source_gid,
                source_N,
                gid,
                degrees_to_use,
                sh_input,
                sh_bias,
                sh_min_value,
                sh_decode_params,
                colors
            );
        }

        return true;
    };

    const bool is_visible   = processGaussian();
    const uint32_t ballot   = __ballot_sync(~0u, is_visible);
    const int32_t word_idx  = idx >> 5;
    const int32_t num_words = (B * C * N + 31) >> 5;
    if(lane == 0 && word_idx < num_words)
    {
        visible[word_idx] = ballot;
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void launch_projection_fwd_kernel(
    // inputs
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::Tensor inference,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const gsplat::CameraModelType camera_model,
    const at::optional<at::Tensor> radial_coeffs,
    // outputs
    at::Tensor visible,
    at::Tensor means2d,
    at::Tensor depths,
    at::Tensor conics,
    at::optional<at::Tensor> compensations
)
{
    uint32_t N = means.size(-1);
    uint32_t C = viewmats.size(-3);
    uint32_t B = means.numel() / (3 * N);

    int64_t n_elements = B * C * N;
    dim3 threads(CTA_SIZE);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    if(n_elements == 0)
    {
        return;
    }

    const float *covars_ptr = nullptr;
    if(covars.has_value())
    {
        covars_ptr = covars.value().data_ptr<float>();
    }
    float *comp_ptr = nullptr;
    if(compensations.has_value())
    {
        comp_ptr = compensations.value().data_ptr<float>();
    }

    float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;
    if(radial_coeffs.has_value())
    {
        const auto rc = radial_coeffs.value().to(at::kCPU).to(at::kFloat).contiguous();
        TORCH_CHECK(rc.numel() >= 4, "radial_coeffs must have at least 4 elements (k1..k4)");
        const float *rc_ptr = rc.data_ptr<float>();
        k1 = rc_ptr[0];
        k2 = rc_ptr[1];
        k3 = rc_ptr[2];
        k4 = rc_ptr[3];
    }

    projection_fwd_kernel<false><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        B,
        C,
        N,
        N,
        means.data_ptr<float>(),
        covars_ptr,
        reinterpret_cast<const __half *>(inference.data_ptr<at::Half>()),
        nullptr,
        false,
        nullptr,
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        k1, k2, k3, k4,
        reinterpret_cast<uint32_t *>(visible.data_ptr<int32_t>()),
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        reinterpret_cast<__half *>(conics.data_ptr<at::Half>()),
        comp_ptr,
        // SH params (unused, dead-code-eliminated)
        0,
        nullptr,
        0.f,
        0.f,
        SHDecodeParams{},
        nullptr
    );
}

void launch_projection_indexed_kernel(
    const at::Tensor means,
    const at::Tensor inference,
    const at::optional<at::Tensor> colors,
    const at::Tensor active_indices,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const gsplat::CameraModelType camera_model,
    at::Tensor visible,
    at::Tensor means2d,
    at::Tensor depths,
    at::Tensor conics,
    const at::optional<at::Tensor> &out_colors
)
{
    const uint32_t source_N = means.size(-1);
    const uint32_t N        = active_indices.numel();
    const uint32_t C        = viewmats.size(-3);
    const uint32_t B        = means.numel() / (3 * source_N);
    const int64_t n_elements = static_cast<int64_t>(B) * C * N;
    if(n_elements == 0)
    {
        return;
    }

    dim3 threads(CTA_SIZE);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    projection_fwd_kernel<false><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        B,
        C,
        N,
        source_N,
        means.data_ptr<float>(),
        nullptr,
        reinterpret_cast<const __half *>(inference.data_ptr<at::Half>()),
        active_indices.data_ptr(),
        active_indices.scalar_type() == at::kLong,
        colors.has_value() ? reinterpret_cast<const __half *>(colors.value().data_ptr<at::Half>()) : nullptr,
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        // indexed path does not carry radial distortion; pass zero coeffs
        0.f, 0.f, 0.f, 0.f,
        reinterpret_cast<uint32_t *>(visible.data_ptr<int32_t>()),
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        reinterpret_cast<__half *>(conics.data_ptr<at::Half>()),
        nullptr,
        0,
        nullptr,
        0.f,
        0.f,
        SHDecodeParams{},
        out_colors.has_value() ? reinterpret_cast<__half *>(out_colors.value().data_ptr<at::Half>()) : nullptr
    );
}

void launch_projection_sh_fused_kernel(
    // inputs
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::Tensor inference,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const gsplat::CameraModelType camera_model,
    // SH inputs
    const int32_t degrees_to_use,
    const at::Tensor sh_input,
    const float bias,
    const float min_value,
    const SHCompressionMode mode,
    const SHDecodeParams *decode_params,
    const at::optional<at::Tensor> &active_indices,
    // outputs
    at::Tensor visible,
    at::Tensor means2d,
    at::Tensor depths,
    at::Tensor conics,
    at::Tensor colors,
    at::optional<at::Tensor> compensations,
    const at::optional<at::Tensor> radial_coeffs
)
{
    uint32_t source_N = means.size(-1);
    uint32_t N = active_indices.has_value() ? active_indices.value().numel() : source_N;
    uint32_t C = viewmats.size(-3);
    uint32_t B = means.numel() / (3 * source_N);

    int64_t n_elements = B * C * N;
    dim3 threads(CTA_SIZE);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    if(n_elements == 0)
    {
        return;
    }

    const float *covars_ptr = nullptr;
    if(covars.has_value())
    {
        covars_ptr = covars.value().data_ptr<float>();
    }
    float *comp_ptr = nullptr;
    if(compensations.has_value())
    {
        comp_ptr = compensations.value().data_ptr<float>();
    }

    float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;
    if(radial_coeffs.has_value())
    {
        const auto rc = radial_coeffs.value().to(at::kCPU).to(at::kFloat).contiguous();
        TORCH_CHECK(rc.numel() >= 4, "radial_coeffs must have at least 4 elements (k1..k4)");
        const float *rc_ptr = rc.data_ptr<float>();
        k1 = rc_ptr[0];
        k2 = rc_ptr[1];
        k3 = rc_ptr[2];
        k4 = rc_ptr[3];
    }

    const SHDecodeParams dp = decode_params ? *decode_params : SHDecodeParams{};
    auto stream             = at::cuda::getCurrentCUDAStream();
    auto *vis_ptr           = reinterpret_cast<uint32_t *>(visible.data_ptr<int32_t>());
    auto *means2d_ptr       = means2d.data_ptr<float>();
    auto *depths_ptr        = depths.data_ptr<float>();
    auto *conics_ptr        = reinterpret_cast<__half *>(conics.data_ptr<at::Half>());
    auto *colors_ptr        = reinterpret_cast<__half *>(colors.data_ptr<at::Half>());

    auto launch = [&](auto mode_tag, const void *sh_data_ptr)
    {
        constexpr SHInputMode MODE = decltype(mode_tag)::value;
        projection_fwd_kernel<true, MODE><<<grid, threads, 0, stream>>>(
            B,
            C,
            N,
            source_N,
            means.data_ptr<float>(),
            covars_ptr,
            reinterpret_cast<const __half *>(inference.data_ptr<at::Half>()),
            active_indices.has_value() ? active_indices.value().data_ptr() : nullptr,
            active_indices.has_value() && active_indices.value().scalar_type() == at::kLong,
            nullptr,
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            camera_model,
            k1, k2, k3, k4,
            vis_ptr,
            means2d_ptr,
            depths_ptr,
            conics_ptr,
            comp_ptr,
            degrees_to_use,
            sh_data_ptr,
            bias,
            min_value,
            dp,
            colors_ptr
        );
    };

    if(mode == SHCompressionMode::COMPRESS_32B)
    {
        launch(
            std::integral_constant<SHInputMode, SHInputMode::COMPRESS_32B>{},
            static_cast<const void *>(sh_input.data_ptr<int32_t>())
        );
    }
    else if(mode == SHCompressionMode::COMPRESS_16B)
    {
        launch(
            std::integral_constant<SHInputMode, SHInputMode::COMPRESS_16B>{},
            static_cast<const void *>(sh_input.data_ptr<int32_t>())
        );
    }
    else if(sh_input.scalar_type() == at::kFloat)
    {
        launch(
            std::integral_constant<SHInputMode, SHInputMode::RAW_FLOAT>{},
            static_cast<const void *>(sh_input.data_ptr<float>())
        );
    }
    else
    {
        launch(
            std::integral_constant<SHInputMode, SHInputMode::RAW_HALF>{},
            static_cast<const void *>(sh_input.data_ptr<at::Half>())
        );
    }
}
} // namespace higs
