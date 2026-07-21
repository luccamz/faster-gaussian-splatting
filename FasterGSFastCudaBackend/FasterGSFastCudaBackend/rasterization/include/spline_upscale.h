#pragma once

namespace faster_gs::rasterization {

    // Gradient-aware bicubic (Hermite) spline upscaling of a rendered view to an arbitrary target
    // resolution, using the analytical image gradients from gradient_render_from_buffers (Niedermayr
    // et al., Eqs. 6-7). Inputs image / grad_* are CHW [3, height, width]; out is [3, out_h, out_w]
    // (CHW) or the HWC equivalent, with out_w >= width, out_h >= height (upscaling only).
    void spline_upscale(
        const float* image,
        const float* grad_x,
        const float* grad_y,
        const float* grad_xy,
        float* out,
        const int width,
        const int height,
        const int out_w,
        const int out_h,
        const bool to_chw,
        const bool clamp_output);

}
