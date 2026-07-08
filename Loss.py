"""FasterGS/Loss.py"""

import math

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim
from Methods.FasterGSFast.Model import FasterGSModel


class FasterGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, model: FasterGSModel, freq_anneal_end: int = 0) -> None:
        super().__init__()
        self.add_loss_metric(
            "L1_Color", torch.nn.functional.l1_loss, loss_config.LAMBDA_L1
        )
        self.add_loss_metric("DSSIM_Color", fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric(
            "OPACITY_REGULARIZATION",
            model.gaussians.opacity_regularization_loss,
            loss_config.LAMBDA_OPACITY_REGULARIZATION,
        )
        self.add_loss_metric(
            "SCALE_REGULARIZATION",
            model.gaussians.scale_regularization_loss,
            loss_config.LAMBDA_SCALE_REGULARIZATION,
        )
        if model.ppisp is None:
            self.add_loss_metric("PPISP_REGULARIZATION", lambda: 0.0, 0.0)
        else:
            self.add_loss_metric(
                "PPISP_REGULARIZATION", model.ppisp.model.get_regularization_loss, 1.0
            )
        if loss_config.LAMBDA_FREQ_AMPLITUDE > 0.0 or loss_config.LAMBDA_FREQ_PHASE > 0.0:
            self.freq_anneal_end = freq_anneal_end
            self.freq_d0_fraction = loss_config.FREQUENCY_D0_FRACTION
            self.freq_lambda_amplitude = loss_config.LAMBDA_FREQ_AMPLITUDE
            self.freq_lambda_phase = loss_config.LAMBDA_FREQ_PHASE
            self._frequency_radius_cache: dict = {}
            self.add_loss_metric("FREQUENCY", self.frequency_loss, 1.0)
        self.add_quality_metric(
            "PSNR", torchmetrics.functional.image.peak_signal_noise_ratio
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor, iteration: int | None = None) -> torch.Tensor:
        return super().forward(
            {
                "L1_Color": {"input": input, "target": target},
                "DSSIM_Color": {"input": input, "target": target},
                "OPACITY_REGULARIZATION": {},
                "SCALE_REGULARIZATION": {},
                "PPISP_REGULARIZATION": {},
                "FREQUENCY": {"input": input, "target": target, "iteration": iteration},
                "PSNR": {"preds": input, "target": target, "data_range": 1.0},
            }
        )

    def frequency_loss(
        self, input: torch.Tensor, target: torch.Tensor, iteration: int | None
    ) -> torch.Tensor:
        """FreGS progressive frequency-space loss (arXiv:2403.06908, Eqs. 5-6, 13): amplitude and phase
        discrepancy between the rendered and ground-truth spectra over a centered low-pass band whose radius
        anneals from D0 to the full spectrum by `freq_anneal_end`, weighted separately. The FFT is
        ortho-normalized so its magnitude is scale- and resolution-invariant (stable across the DashGaussian
        resolution ramp); amplitude and phase therefore need independent weights (normalization rescales
        amplitude but not phase, which is invariant to a positive-real scale). Returns 0 without an
        iteration (the PPISP distillation loop is not part of the annealed training)."""
        if iteration is None:
            return input.new_zeros(())
        fft_input = torch.fft.fftshift(torch.fft.fft2(input, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        fft_target = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        mask = self._frequency_annealing_mask(
            input.shape[-2], input.shape[-1], iteration, input.device
        )
        loss = input.new_zeros(())
        if self.freq_lambda_amplitude > 0.0:
            amplitude_discrepancy = ((fft_input.abs() - fft_target.abs()).abs() * mask).mean()
            loss = loss + self.freq_lambda_amplitude * amplitude_discrepancy
        if self.freq_lambda_phase > 0.0:
            phase_discrepancy = ((torch.angle(fft_input) - torch.angle(fft_target)).abs() * mask).mean()
            loss = loss + self.freq_lambda_phase * phase_discrepancy
        return loss

    def _frequency_annealing_mask(
        self, height: int, width: int, iteration: int, device: torch.device
    ) -> torch.Tensor:
        """Centered circular low-pass mask whose radius grows linearly from `freq_d0_fraction` of the
        spectrum radius to the full radius by `freq_anneal_end` (FreGS Eq. 13)."""
        key = (height, width, device)
        radius = self._frequency_radius_cache.get(key)
        if radius is None:
            offset_y = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1) - height / 2.0
            offset_x = torch.arange(width, device=device, dtype=torch.float32).view(1, -1) - width / 2.0
            radius = (offset_y * offset_y + offset_x * offset_x).sqrt()
            self._frequency_radius_cache[key] = radius
        max_radius = math.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2)
        d0 = self.freq_d0_fraction * max_radius
        progress = min(iteration / max(self.freq_anneal_end, 1), 1.0)
        cutoff = d0 + progress * (max_radius - d0)
        return (radius <= cutoff).to(radius.dtype)
