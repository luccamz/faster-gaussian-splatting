#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::gradient_render {

    // gradient_render holds ONLY the analytical image-gradient kernel. The projection, depth/tile
    // sorts and per-tile instance lists it consumes are built once by the render pass (inference)
    // and reused here through the shared rasterization buffers -- see gradient_render_from_buffers().
    //
    // For the "Lightweight Gradient-Aware Upscaling of 3DGS Images" (Niedermayr et al.) spline
    // upscaler we need the analytical spatial gradients of the rendered image w.r.t. screen position:
    //   dI/dx, dI/dy, d2I/dxdy   (paper Eqs. 8-10).
    // These are accumulated in the same front-to-back blend traversal as the color (Eqs. 11-14),
    // reading only quantities already resident in the blend loop -- no autograd, loss, or GT.
    //
    // Derivation (delta = mean2d - pixel = (dx, dy); conic = (a, b, c); alpha = opacity * g;
    //             g = exp(min(E, 0)), E = -0.5*(a*dx^2 + c*dy^2) - b*dx*dy):
    //   dE/dx = a*dx + b*dy,  dE/dy = c*dy + b*dx,  d2E/dxdy = -b     (0 when E is clamped, E > 0)
    //   d(alpha)/dx = alpha * dE/dx
    //   d(alpha)/dy = alpha * dE/dy
    //   d2(alpha)/dxdy = alpha * (dE/dx * dE/dy + d2E/dxdy)
    // Image gradients accumulate as (T = transmittance seen by this Gaussian, c_i = its color):
    //   dI/dx    += c_i * (dT/dx * alpha + T * d(alpha)/dx)
    //   dI/dy    += c_i * (dT/dy * alpha + T * d(alpha)/dy)
    //   d2I/dxdy += c_i * (d2T/dxdy * alpha + dT/dx * d(alpha)/dy + dT/dy * d(alpha)/dx + T * d2(alpha)/dxdy)
    // Transmittance derivatives follow the front-to-back recurrence T_{i+1} = T_i * (1 - alpha_i):
    //   d2T/dxdy <- d2T/dxdy*(1-alpha) - dT/dx*d(alpha)/dy - dT/dy*d(alpha)/dx - T*d2(alpha)/dxdy
    //   dT/dx    <- dT/dx*(1-alpha)    - T*d(alpha)/dx
    //   dT/dy    <- dT/dy*(1-alpha)    - T*d(alpha)/dy
    // (the d2T update must run before the first-order updates, since it consumes the old dT/dx, dT/dy).
    // Finally the constant background contributes I += T_final * bg  =>  dI += dT_final * bg.
    //
    // Gradient outputs are CHW float buffers [3, height, width], matching the CHW render layout the
    // spline upscaler consumes.
    __global__ void __launch_bounds__(config::block_size_blend) blend_gradients_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ grad_x,
        float* __restrict__ grad_y,
        float* __restrict__ grad_xy,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory (identical fetch layout to the inference blend kernel)
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage: running transmittance and its spatial derivatives, plus the
        // accumulated image gradients (per color channel)
        float transmittance = 1.0f;
        float dt_dx = 0.0f, dt_dy = 0.0f, dt_dxdy = 0.0f;
        float3 grad_image_x = make_float3(0.0f);
        float3 grad_image_y = make_float3(0.0f);
        float3 grad_image_xy = make_float3(0.0f);
        bool done = !inside;
        // collaborative loading and processing (identical traversal to the inference blend kernel)
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                collected_color[thread_rank] = primitive_color[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // evaluate current Gaussian at pixel (matches the inference blend kernel exactly)
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
                const float alpha = opacity * gaussian;
                if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;

                // analytical spatial derivatives of alpha (zero where the Gaussian is clamped, E > 0)
                float de_dx = 0.0f, de_dy = 0.0f, de_dxdy = 0.0f;
                if (exponent < 0.0f) {
                    de_dx = conic.x * delta.x + conic.y * delta.y;
                    de_dy = conic.z * delta.y + conic.y * delta.x;
                    de_dxdy = -conic.y;
                }
                const float dalpha_dx = alpha * de_dx;
                const float dalpha_dy = alpha * de_dy;
                const float dalpha_dxdy = alpha * (de_dx * de_dy + de_dxdy);

                // accumulate image gradients using the transmittance / transmittance-derivatives
                // seen BEFORE this Gaussian is composited
                const float3 color = collected_color[j];
                grad_image_xy += color * (dt_dxdy * alpha + dt_dx * dalpha_dy + dt_dy * dalpha_dx + transmittance * dalpha_dxdy);
                grad_image_x += color * (dt_dx * alpha + transmittance * dalpha_dx);
                grad_image_y += color * (dt_dy * alpha + transmittance * dalpha_dy);

                // update transmittance derivatives (second order first: it consumes the old first-order
                // terms), then transmittance itself
                const float one_minus_alpha = 1.0f - alpha;
                dt_dxdy = dt_dxdy * one_minus_alpha - dt_dx * dalpha_dy - dt_dy * dalpha_dx - transmittance * dalpha_dxdy;
                dt_dx = dt_dx * one_minus_alpha - transmittance * dalpha_dx;
                dt_dy = dt_dy * one_minus_alpha - transmittance * dalpha_dy;
                transmittance *= one_minus_alpha;

                // early stopping
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
        if (inside) {
            // constant background contributes I += T_final * bg, hence dI += dT_final * bg
            const float3 bg = bg_color[0];
            grad_image_xy += bg * dt_dxdy;
            grad_image_x += bg * dt_dx;
            grad_image_y += bg * dt_dy;
            // store results in CHW layout
            const uint n_pixels = width * height;
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            grad_x[pixel_idx] = grad_image_x.x;
            grad_x[n_pixels + pixel_idx] = grad_image_x.y;
            grad_x[2 * n_pixels + pixel_idx] = grad_image_x.z;
            grad_y[pixel_idx] = grad_image_y.x;
            grad_y[n_pixels + pixel_idx] = grad_image_y.y;
            grad_y[2 * n_pixels + pixel_idx] = grad_image_y.z;
            grad_xy[pixel_idx] = grad_image_xy.x;
            grad_xy[n_pixels + pixel_idx] = grad_image_xy.y;
            grad_xy[2 * n_pixels + pixel_idx] = grad_image_xy.z;
        }
    }

}
