"""A lightweight 2-D CNN over MFCC patches.

Four conv blocks treat the ``(coeffs, frames)`` patch as a one-channel image:
the kernels see cepstral shape along one axis and its evolution -- a stroke's
attack and decay -- along the other. Global average pooling makes the head
independent of patch length. About 100k parameters: the task is four timbres,
and a bigger network would mostly memorise the five training concerts.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TimbreCNN"]


class TimbreCNN(nn.Module):
    def __init__(
        self, n_classes: int, n_coeffs: int, channels: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3
    ):
        super().__init__()
        # Per-coefficient standardization, fitted on the training set and
        # saved with the weights so a checkpoint classifies raw MFCCs.
        self.register_buffer("mean", torch.zeros(n_coeffs, 1))
        self.register_buffer("std", torch.ones(n_coeffs, 1))
        layers, prev = [], 1
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, n_classes))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.reshape(self.mean.shape))
        self.std.copy_(std.reshape(self.std.shape).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, coeffs, frames)`` raw MFCCs -> ``(B, n_classes)`` logits."""
        h = self.features(((x - self.mean) / self.std)[:, None])
        return self.head(h.mean(dim=(2, 3)))
