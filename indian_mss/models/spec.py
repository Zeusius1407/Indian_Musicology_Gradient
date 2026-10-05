"""Torch STFT that matches ``indian_mss.dsp.stft`` bin for bin.

The models train on GPU, so they need ``torch.stft`` -- but the README promise
is that one ``STFTConfig`` defines the transform everywhere. This module keeps
that promise: same window (periodic Hann), same zero padding at the edges, and
the same ``scaling="spectrum"`` normalization (divide by the window sum), so a
mask predicted here can be applied to a spectrogram from ``dsp.stft.stft``
without any rescaling. ``tests/test_models.py`` checks the two agree.
"""

from __future__ import annotations

import torch

from ..dsp.stft import STFTConfig

__all__ = ["TorchSTFT"]


class TorchSTFT(torch.nn.Module):
    def __init__(self, cfg: STFTConfig):
        super().__init__()
        if cfg.window != "hann":
            raise ValueError(f"TorchSTFT only implements the hann window, got {cfg.window!r}")
        self.cfg = cfg
        window = torch.hann_window(cfg.win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)
        self.scale = float(window.sum())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``(..., T)`` real -> ``(..., F, frames)`` complex."""
        shape = audio.shape
        spec = torch.stft(
            audio.reshape(-1, shape[-1]),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(audio.dtype),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        spec = spec / self.scale
        return spec.reshape(*shape[:-1], *spec.shape[-2:])

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """``(..., F, frames)`` complex -> ``(..., length)`` real."""
        shape = spec.shape
        audio = torch.istft(
            spec.reshape(-1, *shape[-2:]) * self.scale,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        return audio.reshape(*shape[:-2], length)
