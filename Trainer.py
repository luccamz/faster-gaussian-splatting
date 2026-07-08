"""FasterGS/Trainer.py"""

import random

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import (
    pre_training_callback,
    training_callback,
    post_training_callback,
)
from Methods.FasterGSFast.Loss import FasterGSLoss
from Methods.FasterGSFast.ResolutionScheduler import ResolutionScheduler
from Methods.FasterGSFast.utils import enable_expandable_segments, carve
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,  # while official code states 500, densification actually starts at 600 there
    DENSIFICATION_END_ITERATION=14_900,  # should be set to 24900 when using MCMC; while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,  # only used when USE_MCMC=False
    DENSIFICATION_PERCENT_DENSE=0.01,  # only used when USE_MCMC=False
    SPEEDYSPLAT_PRUNING=Framework.ConfigParameterList(
        USE=False,  # only used when USE_MCMC=False
        START_ITERATION=6_000,
        END_ITERATION=30_000,
        INTERVAL=3_000,
        SOFT_PRUNING_RATIO=0.8,
        HARD_PRUNING_RATIO=0.3,
    ),
    FASTGS=Framework.ConfigParameterList(
        # FastGS multi-view consistency densification (VCD) / pruning (VCP); only used when USE_MCMC=False
        N_SAMPLED_VIEWS=10,  # K: number of training views sampled per densify interval for scoring
        LOSS_THRESHOLD=0.1,  # tau: normalized per-pixel L1 above which a pixel counts as high-error
        VCD=Framework.ConfigParameterList(
            USE=False,
            IMPORTANCE_THRESHOLD=5,  # tau_d: min avg high-error-pixel count for a Gaussian to be densified
            NORM_PERCENTILE=1.0,  # normalize the per-pixel error map by this quantile instead of its max (1.0=max=FastGS); <1 (e.g. 0.99) makes the high-error mask robust to outlier pixels
        ),
        VCP=Framework.ConfigParameterList(
            USE=False,
            SOFT_PRUNING_RATIO=0.5,  # fraction of opacity/size-flagged Gaussians pruned during densification
            FINAL_START_ITERATION=15_000,
            FINAL_END_ITERATION=30_000,
            FINAL_INTERVAL=3_000,
            FINAL_MIN_OPACITY=0.1,
            FINAL_SCORE_THRESHOLD=0.9,  # tau_p: prune Gaussians whose pruning score exceeds this
        ),
        ABSGS=Framework.ConfigParameterList(
            USE=False,  # absolute-gradient split channel (FastGS AbsGS); only used when USE_MCMC=False
            GRAD_ABS_THRESHOLD=0.0012,  # tau for the abs-gradient split gate (FastGS grad_abs_thresh)
        ),
        OPACITY_CLAMP=Framework.ConfigParameterList(
            USE=False,  # FastGS post-densify opacity cap: after each densification cap activated opacity at MAX + reset opacity Adam; only used when USE_MCMC=False
            MAX=0.8,
        ),
        REVISED_OPACITY=Framework.ConfigParameterList(
            USE=False,  # Revising Densification (arXiv:2404.06109): on clone, rescale parent+duplicate opacity to 1-sqrt(1-a) so the region keeps its pre-clone alpha-compositing weight; only used when USE_MCMC=False
        ),
        PIXEL_GS=Framework.ConfigParameterList(
            USE=False,  # Pixel-GS (arXiv:2403.15530): weight each Gaussian's clone-gradient average by its per-view pixel coverage so under-reconstructed large Gaussians clone; only used when USE_MCMC=False
            DEPTH_SCALE_GAMMA=0.0,  # Pixel-GS scaled gradient field (Eq. 10): >0 scales the densification gradient by clip((depth / (gamma * scene_extent))^2, 0, 1) to suppress near-camera floaters; 0 disables (paper uses 0.37)
        ),
    ),
    RESOLUTION_SCHEDULE=Framework.ConfigParameterList(
        # DashGaussian coarse-to-fine training-resolution schedule (FFT-driven): ramps the render
        # downsampling factor MAX_SCALE->1 by DENSIFICATION_END_ITERATION for faster early iterations.
        USE=False,
        MAX_SCALE=8,
        N_LEVELS=32,
        START_SIGNIFICANCE_FACTOR=4,
    ),
    USE_MCMC=False,
    MAX_PRIMITIVES=1_000_000,  # only used when USE_MCMC=True
    OPACITY_RESET_INTERVAL=3_000,  # will be skipped when USE_MCMC=True
    EXTRA_OPACITY_RESET_ITERATION=500,  # will be skipped when USE_MCMC=True
    MORTON_ORDERING_INTERVAL=5000,  # lowering to 2500 or 1000 may improve performance when number of Gaussians is high
    MORTON_ORDERING_END_ITERATION=15000,  # should be set to 25000 when using MCMC
    FILTER_3D=Framework.ConfigParameterList(
        USE=False,
        ORIGINAL_FORMULATION=False,  # if True, the original formulation from the Mip-Splatting paper is used
        FILTER_VARIANCE=0.2,
    ),
    USE_RANDOM_BACKGROUND_COLOR=False,  # prevents the model from overfitting to the background color
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,  # if True, the point cloud from the dataset will be ignored
        N_POINTS=100_000,  # number of random points to be sampled within the scene bounding box
        ENABLE_CARVING=True,  # removes points that are never in-frustum in any training view
        CARVING_IN_ALL_FRUSTUMS=False,  # removes points not in-frustum in all views
        CARVING_ENFORCE_ALPHA=False,  # removes points that project to a pixel with alpha=0 in any view where the point is in-frustum
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
        LAMBDA_OPACITY_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
        LAMBDA_SCALE_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
        LAMBDA_FREQ_AMPLITUDE=0.0,  # FreGS (arXiv:2403.06908): weight for the frequency amplitude-discrepancy loss (ortho-normalized FFT, scale/resolution-invariant); 0 disables. Start ~0.1-1
        LAMBDA_FREQ_PHASE=0.0,  # FreGS: weight for the frequency phase-discrepancy loss; keep low or 0 (phase is noisy and wraps at +-pi)
        FREQUENCY_D0_FRACTION=0.15,  # FreGS frequency annealing: initial low-pass band radius as a fraction of the max spectrum radius, grown to full by the densification end
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.000125,  # 0.0025 / 20
        LEARNING_RATE_OPACITIES=0.025,  # use 0.05 (old default in official code) with MCMC densification or Speedy-Splat pruning to match the respective paper
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
    ),
)
class FasterGSTrainer(GuiTrainer):
    """Defines the trainer for the FasterGS variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info(
                    'using "expandable_segments:True" with the torch cuda memory allocator'
                )
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = None

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: "BaseDataset") -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: "BaseDataset") -> None:
        """Sets up the model."""
        if self.USE_MCMC and (self.FASTGS.VCD.USE or self.FASTGS.VCP.USE or self.FASTGS.ABSGS.USE or self.FASTGS.OPACITY_CLAMP.USE or self.FASTGS.REVISED_OPACITY.USE or self.FASTGS.PIXEL_GS.USE):
            raise Framework.TrainingError(
                "FastGS VCD/VCP/AbsGS densification only compose with the ADC path; set USE_MCMC=False"
            )
        if self.FASTGS.VCP.USE and self.SPEEDYSPLAT_PRUNING.USE:
            raise Framework.TrainingError(
                "FastGS VCP and Speedy-Splat pruning are mutually exclusive; enable only one"
            )
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (
            1.1
            * torch.max(
                torch.linalg.norm(
                    camera_centers - torch.mean(camera_centers, dim=0), dim=1
                )
            )
        ).item()
        Logger.log_info(f"training cameras extent: {radius:.2f}")

        if dataset.point_cloud is not None and not self.RANDOM_INITIALIZATION.FORCE:
            point_cloud = dataset.point_cloud
        else:
            samples = torch.rand(
                (self.RANDOM_INITIALIZATION.N_POINTS, 3),
                dtype=torch.float32,
                device=Framework.config.GLOBAL.DEFAULT_DEVICE,
            )
            positions = samples * dataset.bounding_box.size + dataset.bounding_box.min
            if self.RANDOM_INITIALIZATION.ENABLE_CARVING:
                positions = carve(
                    positions,
                    dataset,
                    self.RANDOM_INITIALIZATION.CARVING_IN_ALL_FRUSTUMS,
                    self.RANDOM_INITIALIZATION.CARVING_ENFORCE_ALPHA,
                )
            point_cloud = BasicPointCloud(positions)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, self.USE_MCMC)
        self.model.gaussians.training_setup(self, radius)
        if not self.USE_MCMC:
            self.model.gaussians.reset_densification_info(track_abs_grad=self.FASTGS.ABSGS.USE, track_pixel_counts=self.FASTGS.PIXEL_GS.USE)
        if self.FILTER_3D.USE:
            self.model.gaussians.setup_3d_filter(self.FILTER_3D, dataset)
        if self.model.ppisp is not None:
            self.model.ppisp.initialize(dataset, self.NUM_ITERATIONS)
        self.loss = FasterGSLoss(loss_config=self.LOSS, model=self.model, freq_anneal_end=self.DENSIFICATION_END_ITERATION)
        # DashGaussian coarse-to-fine resolution schedule, built once from the training-image spectra
        self.resolution_scheduler = (
            ResolutionScheduler(
                [view.rgb for view in dataset.train()],
                self.NUM_ITERATIONS,
                self.DENSIFICATION_END_ITERATION,
                self.RESOLUTION_SCHEDULE,
            )
            if self.RESOLUTION_SCHEDULE.USE
            else None
        )

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @torch.no_grad()
    def _sample_scoring_views(self, dataset: "BaseDataset") -> list:
        """Randomly samples up to N_SAMPLED_VIEWS training views for FastGS multi-view scoring."""
        views = list(dataset.train())
        k = min(self.FASTGS.N_SAMPLED_VIEWS, len(views))
        return random.sample(views, k)

    @training_callback(
        priority=100,
        start_iteration="DENSIFICATION_START_ITERATION",
        end_iteration="DENSIFICATION_END_ITERATION",
        iteration_stride="DENSIFICATION_INTERVAL",
    )
    @torch.no_grad()
    def densify(self, iteration: int, dataset: "BaseDataset") -> None:
        """Apply densification."""
        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(
                min_opacity=0.005, cap_max=self.MAX_PRIMITIVES
            )
        else:
            # FastGS VCD/VCP: compute multi-view consistency scores once, feed both densify + soft-prune
            importance_score = None
            pruning_score = None
            if self.FASTGS.VCD.USE or self.FASTGS.VCP.USE:
                views = self._sample_scoring_views(dataset)
                importance_score, pruning_score = self.renderer.compute_multiview_scores(
                    views,
                    self.FASTGS.LOSS_THRESHOLD,
                    self.LOSS.LAMBDA_L1,
                    self.LOSS.LAMBDA_DSSIM,
                    need_importance=self.FASTGS.VCD.USE,
                    norm_percentile=self.FASTGS.VCD.NORM_PERCENTILE,
                )
            self.model.gaussians.adaptive_density_control(
                self.DENSIFICATION_GRAD_THRESHOLD,
                0.005,
                iteration > self.OPACITY_RESET_INTERVAL,
                importance_score=importance_score,
                importance_threshold=self.FASTGS.VCD.IMPORTANCE_THRESHOLD,
                abs_grad_threshold=self.FASTGS.ABSGS.GRAD_ABS_THRESHOLD if self.FASTGS.ABSGS.USE else None,
                pruning_score=pruning_score if self.FASTGS.VCP.USE else None,
                soft_pruning_ratio=self.FASTGS.VCP.SOFT_PRUNING_RATIO,
                revised_opacity=self.FASTGS.REVISED_OPACITY.USE,
            )
            if self.FASTGS.OPACITY_CLAMP.USE:
                self.model.gaussians.clamp_opacities(self.FASTGS.OPACITY_CLAMP.MAX)

            if (
                self.SPEEDYSPLAT_PRUNING.USE
                and self.SPEEDYSPLAT_PRUNING.START_ITERATION
                <= iteration
                < self.SPEEDYSPLAT_PRUNING.END_ITERATION
                and iteration % self.SPEEDYSPLAT_PRUNING.INTERVAL == 0
            ):
                # Soft Pruning (see https://github.com/j-alex-hanson/speedy-splat/blob/e480b2c3944e4aac4e251307216fe1b8d6a0afc3/train.py#L178-L188)
                scores = self.renderer.compute_pruning_scores(dataset.train())
                self.model.gaussians.importance_pruning(
                    scores, pruning_ratio=self.SPEEDYSPLAT_PRUNING.SOFT_PRUNING_RATIO
                )

            if iteration < self.DENSIFICATION_END_ITERATION:
                self.model.gaussians.reset_densification_info(track_abs_grad=self.FASTGS.ABSGS.USE, track_pixel_counts=self.FASTGS.PIXEL_GS.USE)
        if self.requires_empty_cache:
            torch.cuda.empty_cache()
        if self.FILTER_3D.USE:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(
        priority=99,
        end_iteration="MORTON_ORDERING_END_ITERATION",
        iteration_stride="MORTON_ORDERING_INTERVAL",
    )
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        """Apply morton ordering to all Gaussian parameters and their optimizer states."""
        self.model.gaussians.apply_morton_ordering()

    @training_callback(
        active="FILTER_3D.USE",
        priority=95,
        start_iteration="DENSIFICATION_END_ITERATION",
        iteration_stride=100,
    )
    @torch.no_grad()
    def recompute_3d_filter(self, iteration: int, dataset: "BaseDataset") -> None:
        """Recompute 3D filter."""
        if self.DENSIFICATION_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(
        priority=90,
        start_iteration="OPACITY_RESET_INTERVAL",
        end_iteration="DENSIFICATION_END_ITERATION",
        iteration_stride="OPACITY_RESET_INTERVAL",
    )
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        if not self.USE_MCMC:
            self.model.gaussians.reset_opacities()

    @training_callback(
        priority=90,
        start_iteration="EXTRA_OPACITY_RESET_ITERATION",
        end_iteration="EXTRA_OPACITY_RESET_ITERATION",
    )
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: "BaseDataset") -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if not self.USE_MCMC and dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info(
                "resetting opacities one additional time because using non-black background"
            )
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: "BaseDataset") -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)["view"]
        # render
        bg_color = (
            torch.rand_like(view.camera.background_color)
            if self.USE_RANDOM_BACKGROUND_COLOR
            else view.camera.background_color
        )
        render_scale = (
            self.resolution_scheduler.get_res_scale(iteration)
            if self.resolution_scheduler is not None
            else 1
        )
        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC
            and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
            render_scale=render_scale,
            depth_scale_reference=self.FASTGS.PIXEL_GS.DEPTH_SCALE_GAMMA
            * self.model.gaussians.training_cameras_extent,
        )
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        if render_scale > 1:
            # DashGaussian: downsample the GT to the reduced render size before the loss
            rgb_gt = torch.nn.functional.interpolate(
                rgb_gt.unsqueeze(0), size=image.shape[-2:], mode="bicubic", antialias=True, align_corners=False
            ).squeeze(0)
        loss = self.loss(image, rgb_gt, iteration)
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()

    @training_callback(
        active="SPEEDYSPLAT_PRUNING.USE",
        priority=70,
        start_iteration="SPEEDYSPLAT_PRUNING.START_ITERATION",
        end_iteration="SPEEDYSPLAT_PRUNING.END_ITERATION",
        iteration_stride="SPEEDYSPLAT_PRUNING.INTERVAL",
    )
    @torch.no_grad()
    def hard_pruning(self, iteration: int, dataset: "BaseDataset") -> None:
        """Speedy-Splat Hard Pruning (see https://github.com/j-alex-hanson/speedy-splat/blob/e480b2c3944e4aac4e251307216fe1b8d6a0afc3/train.py#L202-L213)."""
        if iteration >= self.DENSIFICATION_END_ITERATION + self.DENSIFICATION_INTERVAL:
            scores = self.renderer.compute_pruning_scores(dataset.train())
            self.model.gaussians.importance_pruning(
                scores, pruning_ratio=self.SPEEDYSPLAT_PRUNING.HARD_PRUNING_RATIO
            )

    @training_callback(
        active="FASTGS.VCP.USE",
        priority=70,
        start_iteration="FASTGS.VCP.FINAL_START_ITERATION",
        end_iteration="FASTGS.VCP.FINAL_END_ITERATION",
        iteration_stride="FASTGS.VCP.FINAL_INTERVAL",
    )
    @torch.no_grad()
    def vcp_final_prune(self, iteration: int, dataset: "BaseDataset") -> None:
        """FastGS VCP final-stage pruning: post-densification, every FINAL_INTERVAL iters, remove
        Gaussians with low opacity or high pruning score (low multi-view contribution)."""
        _, pruning_score = self.renderer.compute_multiview_scores(
            self._sample_scoring_views(dataset),
            self.FASTGS.LOSS_THRESHOLD,
            self.LOSS.LAMBDA_L1,
            self.LOSS.LAMBDA_DSSIM,
            need_importance=False,
        )
        self.model.gaussians.final_prune_vcp(
            min_opacity=self.FASTGS.VCP.FINAL_MIN_OPACITY,
            pruning_score=pruning_score,
            score_threshold=self.FASTGS.VCP.FINAL_SCORE_THRESHOLD,
        )

    @training_callback(
        active="WANDB.ACTIVATE", priority=10, iteration_stride="WANDB.INTERVAL"
    )
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: "BaseDataset") -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log(
            {"#Gaussians": self.model.gaussians.means.shape[0]}, step=iteration
        )
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, _, dataset: "BaseDataset") -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(
            min_opacity=self.MIN_OPACITY_AFTER_TRAINING
        )
        Logger.log_info(f"final number of Gaussians: {n_gaussians:,}")
        with open(
            str(self.output_directory / "n_gaussians.txt"), "w"
        ) as n_gaussians_file:
            n_gaussians_file.write(
                f"Final number of Gaussians: {n_gaussians:,}\n"
                f"\n"
                f"N_Gaussians:{n_gaussians}"
            )
        if (
            self.model.ppisp is not None
            and self.model.ppisp.config.controller_distillation
        ):
            Logger.log_info(f"distilling PPISP controller")
            with torch.enable_grad():
                self.model.train()
                dataset.train()
                self.loss.train()
                for _ in Logger.log_progress(
                    range(self.model.ppisp.config.controller_training_steps)
                ):
                    # get random view
                    view = self.train_sampler.get(dataset=dataset)["view"]
                    # render
                    image = self.renderer.ppisp_controller_distillation(view=view)
                    # calculate loss
                    # compose gt with background color if needed  # FIXME: integrate into data model
                    rgb_gt = view.rgb
                    if (alpha_gt := view.alpha) is not None:
                        rgb_gt = apply_background_color(
                            rgb_gt, alpha_gt, view.camera.background_color
                        )
                    loss = self.loss(image, rgb_gt)
                    # backward
                    loss.backward()
                    # optimizer step
                    self.model.ppisp.step()
            self.model.ppisp.create_report(self.output_directory)
