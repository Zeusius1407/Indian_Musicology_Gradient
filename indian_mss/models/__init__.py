"""Milestone 2: demixing networks for the Sanidha instrument set.

Importing this package requires ``torch`` (``pip install -e ".[train]"``).
"""

from .separator import build_separator, load_separator, predict_masks, save_checkpoint, separate
from .spec import TorchSTFT
from .unet import MaskUNet

__all__ = [
    "MaskUNet",
    "TorchSTFT",
    "build_separator",
    "load_separator",
    "predict_masks",
    "save_checkpoint",
    "separate",
]
