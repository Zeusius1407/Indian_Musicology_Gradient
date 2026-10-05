"""Spectrogram U-Net that predicts one soft mask per instrument.

The from-scratch fallback when HTDemucs does not fit in the available memory,
and the most direct reading of the Milestone 2 deliverable: the network output
*is* a set of mask ratios. Masks are a softmax across sources, so at every
time-frequency bin they sum to one -- the mixture energy is divided between
instruments, never created or lost -- and resynthesis uses the mixture phase,
exactly as the Milestone 1 HPSS baseline does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..dsp.stft import STFTConfig
from .spec import TorchSTFT

__all__ = ["MaskUNet"]


def _block(chin: int, chout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(chin, chout, 3, padding=1, bias=False),
        nn.BatchNorm2d(chout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(chout, chout, 3, padding=1, bias=False),
        nn.BatchNorm2d(chout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class MaskUNet(nn.Module):
    """Mixture waveform in, one waveform per source out (via soft masks).

    Parameters
    ----------
    sources:
        Output stem names, in output order.
    stft_cfg:
        The pipeline-wide STFT; masks are defined on this grid.
    channels:
        Width of the first encoder level; doubles at every level.
    depth:
        Number of 2x down-sampling steps. Each one halves both frequency and
        time resolution, so the receptive field grows by 2**depth.
    """

    def __init__(self, sources: list[str], stft_cfg: STFTConfig, channels: int = 32, depth: int = 4):
        super().__init__()
        self.sources = list(sources)
        self.stft_cfg = stft_cfg
        self.depth = depth
        self.stft = TorchSTFT(stft_cfg)

        widths = [channels * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        chin = 1
        for w in widths[:-1]:
            self.down.append(_block(chin, w))
            chin = w
        self.bottom = _block(widths[-2], widths[-1])
        self.up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2))
            self.up_blocks.append(_block(widths[i] * 2, widths[i]))
        self.head = nn.Conv2d(widths[0], len(self.sources), 1)

    def masks(self, mag: torch.Tensor) -> torch.Tensor:
        """``(B, F, frames)`` magnitude -> ``(B, S, F, frames)`` masks summing to 1."""
        # Log compression plus per-example standardization: the network sees
        # the same input for a quiet and a loud take of the same performance.
        x = torch.log(mag + 1e-7)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = ((x - mean) / (std + 1e-5)).unsqueeze(1)

        n_freq, n_frames = x.shape[-2:]
        mult = 2**self.depth
        x = F.pad(x, (0, (-n_frames) % mult, 0, (-n_freq) % mult))

        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, block in zip(self.up, self.up_blocks):
            x = up(x)
            x = block(torch.cat([x, skips.pop()], dim=1))

        logits = self.head(x)[..., :n_freq, :n_frames]
        return torch.softmax(logits, dim=1)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` -> ``(B, S, T)``."""
        spec = self.stft(mix)
        masks = self.masks(spec.abs())
        return self.stft.inverse(masks * spec.unsqueeze(1), length=mix.shape[-1])
