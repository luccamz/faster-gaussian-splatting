#include "spline_upscale.h"
#include "kernels_spline_upscale.cuh"
#include "rasterization_config.h"
#include "utils.h"

void faster_gs::rasterization::spline_upscale(
    const float* image,
    const float* grad_x,
    const float* grad_y,
    const float* grad_xy,
    float* out,
    const int width,
    const int height,
    const int factor,
    const bool to_chw,
    const bool clamp_output)
{
    const int out_w = width * factor;
    const int out_h = height * factor;
    const dim3 block(config::tile_width, config::tile_height, 1);
    const dim3 grid(div_round_up(out_w, config::tile_width), div_round_up(out_h, config::tile_height), 1);
    kernels::spline_upscale::spline_upscale_cu<<<grid, block>>>(
        image,
        grad_x,
        grad_y,
        grad_xy,
        out,
        width,
        height,
        factor,
        to_chw,
        clamp_output
    );
    CHECK_CUDA(config::debug, "spline_upscale")
}
