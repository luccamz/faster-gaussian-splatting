"""FasterGS/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View, apply_background_color
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSFast.Model import FasterGSModel
from Methods.FasterGSFast.FasterGSFastCudaBackend import (
    diff_rasterize,
    rasterize,
    rasterize_with_buffers,
    update_pruning_scores,
    count_metric_from_buffers,
    RasterizerSettings,
)
from Optim.Losses.DSSIM import fused_dssim


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    proper_antialiasing: bool,
    render_scale: float = 1.0,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError(
            "FasterGS renderer only supports perspective cameras"
        )
    if view.camera.distortion is not None:
        Logger.log_warning(
            "found distortion parameters that will be ignored by the rasterizer"
        )
    width, height = view.camera.width, view.camera.height
    focal_x, focal_y = view.camera.focal_x, view.camera.focal_y
    center_x, center_y = view.camera.center_x, view.camera.center_y
    if render_scale > 1.0:
        # DashGaussian coarse-to-fine: rasterize at a reduced size and scale the intrinsics to match
        width, height = int(view.camera.width / render_scale), int(view.camera.height / render_scale)
        scale_x, scale_y = width / view.camera.width, height / view.camera.height
        focal_x, focal_y = focal_x * scale_x, focal_y * scale_y
        center_x, center_y = center_x * scale_x, center_y * scale_y
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        active_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        proper_antialiasing,
    )


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
    PROPER_ANTIALIASING=False,
    FORCE_OPTIMIZED_INFERENCE=False,
)
class FasterGSRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: "BaseModel") -> None:
        super().__init__(model, [FasterGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError(
                "FasterGS renderer not implemented in CPU mode"
            )
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(
                f"FasterGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}"
            )

    def render_image(
        self, view: View, to_chw: bool = False, benchmark: bool = False
    ) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        elif self.model.training:
            raise Framework.RendererError(
                "please directly call render_image_training() instead of render_image() during training"
            )
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(
        self, view: View, update_densification_info: bool, bg_color: torch.Tensor, render_scale: float = 1.0, depth_scale_reference: float = 0.0
    ) -> torch.Tensor:
        """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=self.model.gaussians.densification_info
            if update_densification_info
            else torch.empty(0),
            pixel_denom=self.model.gaussians.pixel_denom
            if (update_densification_info and self.model.gaussians.pixel_denom is not None)
            else torch.empty(0),
            depth_scale_reference=depth_scale_reference,
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                bg_color,
                self.PROPER_ANTIALIASING,
                render_scale,
            ),
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        return image

    @torch.no_grad()
    def render_image_inference(
        self, view: View, to_chw: bool = False
    ) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales
            + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=torch.empty(0),
            pixel_denom=torch.empty(0),
            depth_scale_reference=0.0,
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            ),
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        else:
            image = image.clamp(0.0, 1.0)
        return {"rgb": image if to_chw else image.permute(1, 2, 0)}

    @torch.inference_mode()
    def render_image_benchmark(
        self, view: View, to_chw: bool = False
    ) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            ),
            to_chw=to_chw,
            clamp_output=self.model.ppisp is None,
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        return {"rgb": image}

    def ppisp_controller_distillation(self, view: View) -> torch.Tensor:
        """Renders an image for a given view where only the PPISP module will receive gradients."""
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            ),
            to_chw=True,
            clamp_output=False,
        )
        image = self.model.ppisp(image, view)
        return image

    @torch.inference_mode()
    def compute_pruning_scores(self, dataset: BaseDataset) -> torch.Tensor:
        """Computes the pruning scores for the current dataset."""
        scores = torch.zeros(
            self.model.gaussians.means.shape[0],
            device=self.model.gaussians.means.device,
            dtype=torch.float32,
        )
        for view in dataset:
            update_pruning_scores(
                scores=scores,
                means=self.model.gaussians.means,
                scales=self.model.gaussians.raw_scales,
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                rasterizer_settings=extract_settings(
                    view,
                    self.model.gaussians.active_sh_bases,
                    view.camera.background_color,
                    self.PROPER_ANTIALIASING,
                ),
            )
        return scores

    @torch.no_grad()
    def compute_multiview_scores(
        self,
        views: 'list[View]',
        loss_thresh: float,
        lambda_l1: float,
        lambda_dssim: float,
        need_importance: bool = True,
    ) -> 'tuple[torch.Tensor | None, torch.Tensor]':
        """Computes FastGS multi-view consistency scores over the given sampled views.

        For each view the scene is rendered, a per-pixel L1 error map is thresholded into a
        high-error mask (Eqs 6-8), and `count_metric_from_buffers` accumulates, per Gaussian, the
        number of high-error pixels it contributes to (reusing the render's buffers). Returns:
          - importance (s_d, Eq 9): per-Gaussian floor-average of high-error counts across
            views; used to gate VCD densification. `None` when `need_importance` is False.
          - pruning (s_p, Eq 11): min-max normalized sum of (per-view photometric loss * counts);
            used by VCP pruning.
        Uses `@torch.no_grad()` (not inference_mode) so the DSSIM autograd op can run.
        """
        gaussians = self.model.gaussians
        n_primitives = gaussians.means.shape[0]
        device = gaussians.means.device
        accum_counts = (
            torch.zeros(n_primitives, device=device, dtype=torch.float32)
            if need_importance
            else None
        )
        accum_score = torch.zeros(n_primitives, device=device, dtype=torch.float32)
        n_views = 0
        for view in views:
            n_views += 1
            settings = extract_settings(
                view,
                gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            )
            # render the current view (no-grad inference rasterizer, unclamped to match training domain);
            # keep the rasterization buffers so the metric-count pass below reuses this view's projection
            # and depth/tile sort instead of rebuilding them
            image, buffers = rasterize_with_buffers(
                means=gaussians.means,
                scales=gaussians.raw_scales,
                rotations=gaussians.raw_rotations,
                opacities=gaussians.raw_opacities,
                sh_coefficients_0=gaussians.sh_coefficients_0,
                sh_coefficients_rest=gaussians.sh_coefficients_rest,
                rasterizer_settings=settings,
                to_chw=True,
                clamp_output=False,
            )
            # ground truth, composited onto the same background if it carries an alpha channel
            rgb_gt = view.rgb
            if (alpha_gt := view.alpha) is not None:
                rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
            # per-pixel L1 over channels -> min-max normalized -> high-error mask (Eqs 6-8)
            l1_map = (image - rgb_gt).abs().mean(dim=0)
            l1_min = l1_map.min()
            l1_norm = (l1_map - l1_min) / (l1_map.max() - l1_min).clamp_min(1e-8)
            metric_map = (l1_norm > loss_thresh).to(torch.int32).reshape(-1).contiguous()
            # per-Gaussian count of high-error pixels this Gaussian contributes to, this view;
            # reuses the render's buffers (no re-projection / re-sort)
            counts = torch.zeros(n_primitives, device=device, dtype=torch.float32)
            count_metric_from_buffers(
                counts=counts,
                metric_map=metric_map,
                buffers=buffers,
                width=settings.width,
                height=settings.height,
            )
            if need_importance:
                accum_counts += counts
            # photometric loss weighting for the pruning score (Eqs 10-11)
            e_photo = lambda_l1 * torch.nn.functional.l1_loss(image, rgb_gt) + lambda_dssim * fused_dssim(image, rgb_gt)
            accum_score += e_photo * counts
        # importance = floor(mean counts) (Eq 9); pruning = min-max normalized weighted score (Eq 11)
        importance = (
            torch.div(accum_counts, max(n_views, 1), rounding_mode='floor')
            if need_importance
            else None
        )
        score_min = accum_score.min()
        pruning = (accum_score - score_min) / (accum_score.max() - score_min).clamp_min(1e-8)
        return importance, pruning

    def postprocess_outputs(
        self, outputs: dict[str, torch.Tensor], *_
    ) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {"rgb": outputs["rgb"]}
