#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::metric_counts {

    // metric_counts holds ONLY the final counting kernel. The projection, depth/tile sorts and
    // per-tile instance lists it consumes are built once by the render pass (inference) and reused
    // here through the shared rasterization buffers -- see metric_counts_from_buffers().

    __global__ void __launch_bounds__(config::block_size_blend) compute_metric_counts_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const int* __restrict__ metric_map,
        float* __restrict__ counts,
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
        // whether this pixel is flagged as high-error; out-of-bounds pixels never count
        const int metric = inside ? metric_map[pixel_coords.y * width + pixel_coords.x] : 0;
        // setup shared memory
        __shared__ uint collected_primitive_idx[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        // initialize local storage
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing (identical traversal to compute_scores_cu)
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_idx[thread_rank] = primitive_idx;
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
                const float alpha = opacity * gaussian;
                if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;

                // this Gaussian contributes to the current pixel; count it if the pixel is high-error
                // (note: a per-tile early-out on `metric` is possible if the scoring pass becomes a
                //  bottleneck, but we keep the full traversal here to match FastGS's forward semantics)
                if (metric == 1) atomicAdd(&counts[collected_primitive_idx[j]], 1.0f);

                // update transmittance
                transmittance *= 1.0f - alpha;

                // early stopping
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
    }

}
