#pragma once

#include "rasterization_config.h"
#include "helper_math.h"

namespace faster_gs::rasterization::kernels::spline_upscale {

    // Gradient-aware bicubic (Hermite) spline upscaling -- Niedermayr et al., Eqs. 6-7.
    //
    // Classical bicubic interpolation estimates the tangents f_x, f_y, f_xy at the cell corners with
    // finite differences over a 4x4 neighbourhood. Here they come analytically from the renderer
    // (blend_gradients_cu), so only the 2x2 cell surrounding the target sub-pixel is needed -- no
    // finite-difference stencil, hence no outer ring and simpler borders.
    //
    // For a target pixel that falls at fractional position (tx, ty) inside a unit cell with corner
    // data F (values + tangents, Eq. 6), the interpolation coefficients are A = C * F * C^T with the
    // Hermite basis matrix
    //     C = [[ 1, 0, 0, 0],
    //          [ 0, 0, 1, 0],
    //          [-3, 3,-2,-1],
    //          [ 2,-2, 1, 1]]
    // and the value is p = [1, tx, tx^2, tx^3] * A * [1, ty, ty^2, ty^3]^T.
    // The analytical tangents are in per-low-res-pixel units, matching the unit-spaced Hermite cell.

    __device__ __forceinline__ float hermite_eval(
        // corner values and tangents (x0/x1 = left/right column, y0/y1 = top/bottom row)
        const float f00, const float f10, const float f01, const float f11,
        const float fx00, const float fx10, const float fx01, const float fx11,
        const float fy00, const float fy10, const float fy01, const float fy11,
        const float fxy00, const float fxy10, const float fxy01, const float fxy11,
        const float tx, const float ty)
    {
        // F laid out as in Eq. 6 (rows indexed by x-data, columns by y-data)
        const float F[4][4] = {
            {  f00,  f01,  fy00,  fy01 },
            {  f10,  f11,  fy10,  fy11 },
            { fx00, fx01, fxy00, fxy01 },
            { fx10, fx11, fxy10, fxy11 },
        };
        const float C[4][4] = {
            {  1.0f,  0.0f,  0.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f,  0.0f },
            { -3.0f,  3.0f, -2.0f, -1.0f },
            {  2.0f, -2.0f,  1.0f,  1.0f },
        };
        // M = C * F
        float M[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) acc += C[i][k] * F[k][j];
                M[i][j] = acc;
            }
        // A = M * C^T
        float A[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) acc += M[i][k] * C[j][k];
                A[i][j] = acc;
            }
        // p = xvec * A * yvec
        const float xvec[4] = { 1.0f, tx, tx * tx, tx * tx * tx };
        const float yvec[4] = { 1.0f, ty, ty * ty, ty * ty * ty };
        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float row = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) row += A[i][j] * yvec[j];
            p += xvec[i] * row;
        }
        return p;
    }

    // image / grad_* are CHW float buffers [3, height, width]; out is [3, out_h, out_w] (CHW) or
    // [out_h, out_w, 3] (HWC) at an arbitrary target resolution (out_w >= width, out_h >= height --
    // upscaling only, enforced by the caller). One thread per output pixel, all three channels.
    __global__ void spline_upscale_cu(
        const float* __restrict__ image,
        const float* __restrict__ grad_x,
        const float* __restrict__ grad_y,
        const float* __restrict__ grad_xy,
        float* __restrict__ out,
        const int width,
        const int height,
        const int out_w,
        const int out_h,
        const bool to_chw,
        const bool clamp_output)
    {
        const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
        if (out_x >= out_w || out_y >= out_h) return;

        // Map the output pixel centre back to low-res continuous coordinates. Per-axis scale is
        // anisotropy-safe for non-integer / non-square targets; width/out_w = 1/factor when out_w =
        // width*factor, so an integer factor stays bit-identical. The -0.5 turns a half-pixel-centred
        // output coordinate into the low-res sample-index space the Hermite cells live in: low-res
        // sample i sits at continuous position i+0.5, matching the rasterizer's pixel convention
        // (kernels_forward.cuh: pixel = (i, j) + 0.5f).
        const float inv_scale_x = static_cast<float>(width) / out_w;
        const float inv_scale_y = static_cast<float>(height) / out_h;
        const float xl = (out_x + 0.5f) * inv_scale_x - 0.5f;
        const float yl = (out_y + 0.5f) * inv_scale_y - 0.5f;
        // Pick the cell with a single clamp to [0, size-2]: interior yields tx, ty in [0, 1); the outer
        // sub-pixel ring (xl < 0 or xl > width-1) keeps valid corners x0, x0+1 and lets tx/ty fall just
        // outside [0, 1] so the boundary Hermite cubic extrapolates from the analytical edge gradients
        // instead of replicating. Overhang is < 1 low-res pixel. Upscaling guarantees width, height >= 2.
        const int x0 = min(max(static_cast<int>(floorf(xl)), 0), width - 2);
        const int y0 = min(max(static_cast<int>(floorf(yl)), 0), height - 2);
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float tx = xl - x0;
        const float ty = yl - y0;

        const int n_pixels = width * height;
        const int out_pixels = out_w * out_h;
        const int out_idx = out_y * out_w + out_x;

        #pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            const float* im = image + ch * n_pixels;
            const float* gx = grad_x + ch * n_pixels;
            const float* gy = grad_y + ch * n_pixels;
            const float* gxy = grad_xy + ch * n_pixels;
            const int i00 = y0 * width + x0, i10 = y0 * width + x1;
            const int i01 = y1 * width + x0, i11 = y1 * width + x1;
            float value = hermite_eval(
                im[i00], im[i10], im[i01], im[i11],
                gx[i00], gx[i10], gx[i01], gx[i11],
                gy[i00], gy[i10], gy[i01], gy[i11],
                gxy[i00], gxy[i10], gxy[i01], gxy[i11],
                tx, ty
            );
            if (clamp_output) value = __saturatef(value);
            if (to_chw) out[ch * out_pixels + out_idx] = value;
            else out[3 * out_idx + ch] = value;
        }
    }

}
