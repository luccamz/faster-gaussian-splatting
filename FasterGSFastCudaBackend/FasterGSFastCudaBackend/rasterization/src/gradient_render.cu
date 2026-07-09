#include "gradient_render.h"
#include "kernels_gradient_render.cuh"  // the analytical image-gradient kernel (blend_gradients_cu)
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"

// Reuses the projection + depth/tile sort + instance lists already built by a render pass
// (inference_with_buffers) for the same view and Gaussian set, so only the gradient kernel runs here
// -- the preprocess, both radix sorts, scan, instance creation and range extraction are skipped
// entirely. Correctness relies on the render and gradient paths sharing byte-identical
// preprocess/instance-building kernels (they do) and on restoring the sorted DoubleBuffer side via
// the selector captured by the render (mirrors metric_counts.cu / backward.cu).
void faster_gs::rasterization::gradient_render_from_buffers(
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    const float3* bg_color,
    float* grad_x,
    float* grad_y,
    float* grad_xy,
    const int n_primitives,
    const int n_instances,
    const int instance_primitive_indices_selector,
    const int width,
    const int height)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);

    auto dispatch_gradient_render = [&](const uint* instance_primitive_indices) {
        kernels::gradient_render::blend_gradients_cu<<<grid, block>>>(
            tile_buffers.instance_ranges,
            instance_primitive_indices,
            primitive_buffers.mean2d,
            primitive_buffers.conic_opacity,
            primitive_buffers.color,
            bg_color,
            grad_x,
            grad_y,
            grad_xy,
            width,
            height,
            grid.x
        );
        CHECK_CUDA(config::debug, "blend_gradients (from_buffers)")
    };
    if (end_bit <= 16) {
        auto instance_buffers = InstanceBuffers<ushort>::from_blob(instance_buffers_blob, n_instances, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_gradient_render(instance_buffers.primitive_indices.Current());
    }
    else {
        auto instance_buffers = InstanceBuffers<uint>::from_blob(instance_buffers_blob, n_instances, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_gradient_render(instance_buffers.primitive_indices.Current());
    }
}
