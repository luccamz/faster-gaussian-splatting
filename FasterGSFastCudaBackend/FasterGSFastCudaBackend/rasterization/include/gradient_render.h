#pragma once

#include "helper_math.h"

namespace faster_gs::rasterization {

    // Computes the analytical spatial gradients (dI/dx, dI/dy, d2I/dxdy) of an already-rendered view,
    // reusing the rasterization buffers produced by inference_with_buffers (same view, same Gaussians)
    // -- mirrors metric_counts_from_buffers: only the gradient kernel runs here, the projection and
    // depth/tile sorts are skipped. Writes three CHW float buffers [3, height, width].
    void gradient_render_from_buffers(
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
        const int height);

}
