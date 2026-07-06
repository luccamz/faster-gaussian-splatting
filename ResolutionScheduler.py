"""DashGaussian coarse-to-fine training-resolution schedule.

Builds a per-scene schedule from the training-image frequency spectrum and returns an integer
render-downsampling factor per iteration, ramping from MAX_SCALE down to 1 by `increase_until`.
Implements the resolution scheduler of DashGaussian (Chen et al., CVPR 2025); the primitive-budget
component is not included.
"""
import math

import torch


class ResolutionScheduler:
    """Frequency-driven coarse-to-fine render-resolution schedule."""

    def __init__(self, train_images: 'list[torch.Tensor]', max_steps: int, increase_until: int, config) -> None:
        self.max_steps = max_steps
        self.increase_reso_until = increase_until
        self.start_significance_factor = config.START_SIGNIFICANCE_FACTOR
        self.max_reso_scale = config.MAX_SCALE
        self.reso_sample_num = max(2, config.N_LEVELS)  # must be >= 2
        self.next_i = 2
        self.reso_scales: 'list[float]' = []
        self.reso_level_begin: 'list[int]' = []
        self._init_schedule(train_images)

    def get_res_scale(self, iteration: int) -> int:
        """Integer render-downsampling factor for the given iteration (1 once fully ramped up)."""
        if iteration >= self.increase_reso_until:
            return 1
        if iteration < self.reso_level_begin[1]:
            return int(self.reso_scales[0])
        while iteration >= self.reso_level_begin[self.next_i]:
            self.next_i += 1
        i = self.next_i - 1
        i_now, i_nxt = self.reso_level_begin[i: i + 2]
        s_lst, s_now = self.reso_scales[i - 1: i + 1]
        scale = (1.0 / ((iteration - i_now) / (i_nxt - i_now) * (1.0 / s_now ** 2 - 1.0 / s_lst ** 2) + 1.0 / s_lst ** 2)) ** 0.5
        return int(scale)

    @staticmethod
    def _win_significance(significance_map: torch.Tensor, scale: float) -> float:
        """Spectral energy inside the centered low-frequency window of size (H/scale, W/scale)."""
        h, w = significance_map.shape[-2:]
        c = ((h + 1) // 2, (w + 1) // 2)
        win = (int(h / scale), int(w / scale))
        return significance_map[..., c[0] - win[0] // 2: c[0] + win[0] // 2,
                                     c[1] - win[1] // 2: c[1] + win[1] // 2].sum().item()

    @classmethod
    def _scale_solver(cls, significance_map: torch.Tensor, target: float) -> float:
        """Binary-search the scale whose central window holds `target` energy."""
        lo, hi, mid = 0.0, 1.0, 0.5
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if cls._win_significance(significance_map, 1.0 / mid) < target:
                lo = mid
            else:
                hi = mid
        return 1.0 / mid

    def _init_schedule(self, train_images: 'list[torch.Tensor]') -> None:
        self.max_reso_scale = 8
        self.next_i = 2
        scene_freq = None
        for image in train_images:
            fft = torch.fft.fftshift(torch.fft.fft2(image.float()), dim=(-2, -1))
            magnitude = (fft.real.square() + fft.imag.square()).sqrt()
            scene_freq = magnitude if scene_freq is None else scene_freq + magnitude
            e_total = magnitude.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(self.max_reso_scale, self._scale_solver(magnitude, e_min))
        modulation = math.log
        significance: 'list[float]' = []
        scene_freq /= len(train_images)
        total_energy = scene_freq.sum().item()
        min_energy = self._win_significance(scene_freq, self.max_reso_scale)
        significance.append(min_energy)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)
        for i in range(1, self.reso_sample_num - 1):
            significance.append((total_energy - min_energy) * i / (self.reso_sample_num - 1) + min_energy)
            self.reso_scales.append(self._scale_solver(scene_freq, significance[-1]))
            significance[-2] = modulation(significance[-2] / min_energy)
            self.reso_level_begin.append(int(self.increase_reso_until * significance[-2] / modulation(total_energy / min_energy)))
        significance.append(modulation(total_energy / min_energy))
        self.reso_scales.append(1.0)
        significance[-2] = modulation(significance[-2] / min_energy)
        self.reso_level_begin.append(int(self.increase_reso_until * significance[-2] / modulation(total_energy / min_energy)))
        self.reso_level_begin.append(self.increase_reso_until)
