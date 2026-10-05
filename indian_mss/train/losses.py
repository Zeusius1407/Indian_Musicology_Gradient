"""Training loss: spectrogram L1 plus waveform L1, weighted from the config.

The two terms catch different failures. Spectrogram L1 punishes energy in the
wrong instrument (bleed) but is blind to phase; waveform L1 punishes phase and
timing errors that make a stem sound smeared even when its spectrum is right.
"""

from __future__ import annotations

import torch

from ..dsp.stft import STFTConfig
from ..models.spec import TorchSTFT

__all__ = ["SeparationLoss"]


class SeparationLoss(torch.nn.Module):
    def __init__(self, stft_cfg: STFTConfig, spectrogram_l1: float = 1.0, waveform_l1: float = 1.0):
        super().__init__()
        self.stft = TorchSTFT(stft_cfg)
        self.w_spec = spectrogram_l1
        self.w_wave = waveform_l1

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """``est, ref: (B, S, T)`` -> ``(total, {"spectrogram_l1", "waveform_l1"})``.

        Both terms are divided by the reference's mean absolute value per
        example, so a quiet recording counts as much as a loud one.
        """
        scale = ref.abs().mean(dim=(1, 2), keepdim=True) + 1e-6
        wave = ((est - ref).abs() / scale).mean()
        spec_scale = scale.unsqueeze(-1)
        spec = ((self.stft(est).abs() - self.stft(ref).abs()).abs() / spec_scale).mean()
        total = self.w_spec * spec + self.w_wave * wave
        return total, {"spectrogram_l1": float(spec.detach()), "waveform_l1": float(wave.detach())}
