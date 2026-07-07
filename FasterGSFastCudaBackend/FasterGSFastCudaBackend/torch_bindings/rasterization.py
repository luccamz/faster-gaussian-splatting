from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSFastCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    bg_color: torch.Tensor  # background color in RGB format
    active_sh_bases: (
        int  # number of spherical harmonics bases to use for color computation
    )
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    proper_antialiasing: bool  # whether to use proper antialiasing

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        pixel_denom: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> torch.Tensor:
        (
            image,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
            n_instances,
            n_buckets,
            instance_primitive_indices_selector,
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.pixel_denom = pixel_denom
        ctx.mark_non_differentiable(densification_info)
        ctx.mark_non_differentiable(pixel_denom)
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]":
        (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
        ) = _C.backward(
            ctx.densification_info,
            ctx.pixel_denom,
            grad_image,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            None,  # densification_info
            None,  # pixel_denom
            None,  # rasterizer_settings
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    pixel_denom: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        pixel_denom,
        rasterizer_settings,
    )


def rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    clamp_output: bool = True,
) -> torch.Tensor:
    return _C.inference(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
        to_chw,
        clamp_output,
    )


def rasterize_with_buffers(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    clamp_output: bool = True,
) -> "tuple[torch.Tensor, tuple]":
    """Renders a view like `rasterize`, but also returns the intermediate rasterization buffers.

    The returned `buffers` bundle (opaque byte tensors + `n_instances` + the sorted-instance
    DoubleBuffer selector) can be passed to `count_metric_from_buffers` to compute FastGS metric
    counts without rebuilding the projection and depth/tile sort for the same view.
    """
    (
        image,
        primitive_buffers,
        tile_buffers,
        instance_buffers,
        n_instances,
        instance_primitive_indices_selector,
    ) = _C.inference_with_buffers(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
        to_chw,
        clamp_output,
    )
    buffers = (
        primitive_buffers,
        tile_buffers,
        instance_buffers,
        n_instances,
        instance_primitive_indices_selector,
    )
    return image, buffers


def count_metric_from_buffers(
    counts: torch.Tensor,
    metric_map: torch.Tensor,
    buffers: tuple,
    width: int,
    height: int,
) -> torch.Tensor:
    """Accumulates FastGS metric counts for one view, reusing a render's buffers (see
    `rasterize_with_buffers`). Runs only the counting kernel -- the preprocess, depth/tile sorts and
    instance-list construction are skipped, since the render already built them for this view.
    For every Gaussian contributing to a high-error pixel (`metric_map == 1`), its per-primitive entry
    in `counts` is incremented by one. `counts` is a float32 CUDA tensor of length n_primitives,
    accumulated in place; `metric_map` a contiguous int32 CUDA tensor of length height*width in
    row-major (y*width+x) order.
    """
    (
        primitive_buffers,
        tile_buffers,
        instance_buffers,
        n_instances,
        instance_primitive_indices_selector,
    ) = buffers
    return _C.metric_counts_from_buffers(
        counts,
        metric_map,
        primitive_buffers,
        tile_buffers,
        instance_buffers,
        n_instances,
        instance_primitive_indices_selector,
        width,
        height,
    )


def update_pruning_scores(
    scores: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _C.pruning_scores(
        scores,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
    )


