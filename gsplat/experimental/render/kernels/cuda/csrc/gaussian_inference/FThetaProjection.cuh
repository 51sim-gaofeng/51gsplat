#pragma once

#include "Common.h"



// OpenCV Brown-Conrady pinhole distortion coefficients (radial k1..k6 + tangential p1,p2),
// passed by value into the projection kernel. active=false -> plain persp_proj.
struct OpenCVDistortion
{
    float k[6]{};
    float p[2]{};
    float r2_max = 1e30f;  // undistorted r^2 limit of the monotonic (valid) mapping region
    // C1 linear extension of r_d(r) beyond r_knee (just inside the monotonic limit), so the
    // whole image maps to real rays instead of cunamespace higs
{
struct FThetaCoefficients
{
    float polynomial[6]{};
    float linear[3]{1.f, 0.f, 0.f};
    float max_angle = 0.f;
    float max_screen_radius = 0.f;
    float max_screen_radius_dist = 0.f;
    // Elongated-splat cull: major axis > soft_screen_radius AND major/minor > max_eccentricity.
    float soft_screen_radius = 0.f;
    float max_eccentricity = 0.f;
    float offscreen_radius = 0.f;
    float ecc_clamp = 0.f;  // >0: shrink major axis instead of culling eccentric splats
};lling past the limit. r_knee <= 0 -> off.
    float r_knee = 0.f;
    float rd_knee = 0.f;
    float slope_knee = 0.f;
    bool  active = false;
};

inline __device__ bool ftheta_project(
    const gsplat::vec3 &mean,
    const gsplat::mat3 &covariance,
    const FThetaCoefficients &coeffs,
    float principal_x,
    float principal_y,
    gsplat::mat2 &covariance_2d,
    gsplat::vec2 &mean_2d
)
{
    if(mean.z <= 0.f)
    {
        return false;
    }
    const float radial = hypotf(mean.x, mean.y);
    const float theta = atan2f(radial, mean.z);
    if(theta >= coeffs.max_angle)
    {
        return false;
    }
    float radius = coeffs.polynomial[5];
    float derivative = 0.f;
    for(int index = 4; index >= 0; --index)
    {
        derivative = fmaf(derivative, theta, radius);
        radius = fmaf(radius, theta, coeffs.polynomial[index]);
    }

    glm::mat3x2 jacobian;
    gsplat::vec2 offset;
    if(radial <= 1e-7f * mean.z)
    {
        const float slope = coeffs.polynomial[1] / mean.z;
        offset = gsplat::vec2(mean.x, mean.y) * slope;
        jacobian = glm::mat3x2(slope, 0.f, 0.f, slope, 0.f, 0.f);
    }
    else
    {
        const float unit_x = mean.x / radial;
        const float unit_y = mean.y / radial;
        const float distance_squared = radial * radial + mean.z * mean.z;
        const float angular_slope = derivative * mean.z / distance_squared;
        const float radial_slope = radius / radial;
        const float difference = angular_slope - radial_slope;
        offset = radius * gsplat::vec2(unit_x, unit_y);
        jacobian = glm::mat3x2(
            radial_slope + unit_x * unit_x * difference,
            unit_x * unit_y * difference,
            unit_x * unit_y * difference,
            radial_slope + unit_y * unit_y * difference,
            -derivative * mean.x / distance_squared,
            -derivative * mean.y / distance_squared
        );
    }
    const gsplat::mat2 linear(coeffs.linear[0], coeffs.linear[2], coeffs.linear[1], 1.f);
    mean_2d = linear * offset + gsplat::vec2(principal_x + .5f, principal_y + .5f);
    jacobian = linear * jacobian;
    covariance_2d = jacobian * covariance * glm::transpose(jacobian);
    return isfinite(mean_2d.x) && isfinite(mean_2d.y)
        && isfinite(covariance_2d[0][0]) && isfinite(covariance_2d[0][1])
        && isfinite(covariance_2d[1][1]);
}
}