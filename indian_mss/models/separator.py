"""One interface over both backbones: build, run, predict masks, checkpoint.

Every separator is an ``nn.Module`` mapping a mono mixture ``(B, T)`` to one
waveform per source ``(B, S, T)`` and exposing ``.sources``. Training,
validation and the ``imss-separate`` CLI only ever talk to that interface, so
switching ``train.backbone`` between ``htdemucs`` and ``unet`` changes nothing
downstream.

Mask ratios (the Milestone 2 deliverable) are derived the same way for both:
separate the full track, take the magnitude of each estimate on the pipeline
STFT grid, and divide by their sum. For the U-Net that recovers its own
softmax masks; for HTDemucs, which predicts waveforms directly, it is the
ratio mask its output implies. Either way the result lines up frame for frame
with ``dsp.stft.stft`` of the mixture, ready for Milestone 3.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..dsp.stft import STFTConfig, stft

__all__ = [
    "build_separator",
    "load_separator",
    "param_groups",
    "predict_masks",
    "save_checkpoint",
    "separate",
]


def build_separator(
    backbone: str,
    sources: list[str],
    stft_cfg: STFTConfig,
    model_cfg: dict | None = None,
    pretrained: bool = True,
) -> tuple[nn.Module, dict]:
    """Return ``(model, arch)``; ``arch`` is everything needed to rebuild it."""
    model_cfg = model_cfg or {}
    if backbone == "unet":
        from .unet import MaskUNet

        ucfg = model_cfg.get("unet", {})
        arch = {"kind": "unet", "channels": int(ucfg.get("channels", 32)), "depth": int(ucfg.get("depth", 4))}
        return MaskUNet(sources, stft_cfg, arch["channels"], arch["depth"]), arch

    if backbone == "htdemucs":
        from .demucs_backbone import DemucsSeparator, load_htdemucs, retarget_sources

        dcfg = model_cfg.get("htdemucs", {})
        if pretrained:
            base = load_htdemucs(dcfg.get("name", "htdemucs"))
        else:
            from demucs.htdemucs import HTDemucs

            # From scratch: only sensible for tests or ablations -- the whole
            # point of this backbone is the pretrained encoder.
            kwargs = {"samplerate": stft_cfg.sample_rate, **dcfg.get("kwargs", {})}
            base = HTDemucs(["drums", "bass", "other", "vocals"], **kwargs)
        if base.samplerate != stft_cfg.sample_rate:
            raise ValueError(
                f"HTDemucs runs at {base.samplerate} Hz but stft.sample_rate is {stft_cfg.sample_rate}; "
                "set them equal in configs/default.yaml"
            )
        retarget_sources(base, sources, dcfg.get("init"))
        args, kwargs = base._init_args_kwargs
        arch = {"kind": "htdemucs", "args": list(args), "kwargs": dict(kwargs)}
        return DemucsSeparator(base), arch

    raise ValueError(f"unknown backbone {backbone!r}: expected 'htdemucs' or 'unet'")


def _rebuild(arch: dict, sources: list[str], stft_cfg: STFTConfig) -> nn.Module:
    if arch["kind"] == "unet":
        from .unet import MaskUNet

        return MaskUNet(sources, stft_cfg, arch["channels"], arch["depth"])
    if arch["kind"] == "htdemucs":
        from demucs.htdemucs import HTDemucs

        from .demucs_backbone import DemucsSeparator

        return DemucsSeparator(HTDemucs(*arch["args"], **arch["kwargs"]))
    raise ValueError(f"unknown architecture {arch['kind']!r}")


def param_groups(model: nn.Module, phase: int, train_cfg: dict) -> list[dict]:
    """Optimizer groups for a training phase (see ``DemucsSeparator``).

    The U-Net has no pretrained weights to protect, so it trains everything
    from the first step and phases do not apply.
    """
    lr = float(train_cfg.get("lr", 3e-4))
    if hasattr(model, "trainable_groups"):
        return model.trainable_groups(
            phase,
            lr=lr,
            encoder_lr=float(train_cfg.get("encoder_lr", lr / 10)),
            unfreeze=list(train_cfg.get("unfreeze", ["decoder", "tdecoder"])),
        )
    for p in model.parameters():
        p.requires_grad_(True)
    return [{"params": list(model.parameters()), "lr": lr, "name": "all"}]


def _chunk_weight(n: int) -> np.ndarray:
    """Triangular cross-fade; strictly positive so every sample is covered."""
    t = np.arange(n)
    return np.minimum(t + 1, n - t).astype(np.float32)


@torch.no_grad()
def separate(
    model: nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    overlap: float = 0.25,
    batch_size: int = 4,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Separate a full-length mono track, chunk by chunk.

    Chunks must not exceed the training segment (HTDemucs refuses longer
    input); overlapping chunks are cross-faded so chunk boundaries do not
    click. Returns ``(S, T)``.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    chunk = int(chunk_seconds * sample_rate)

    if n <= chunk:
        x = torch.from_numpy(audio)[None].to(device)
        return model(x)[0].cpu().numpy()

    hop = max(1, int(chunk * (1.0 - overlap)))
    starts = list(range(0, n - chunk, hop)) + [n - chunk]
    weight = _chunk_weight(chunk)
    out = np.zeros((len(model.sources), n), dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        x = torch.from_numpy(np.stack([audio[s : s + chunk] for s in batch_starts])).to(device)
        y = model(x).cpu().numpy()
        for s, est in zip(batch_starts, y):
            out[:, s : s + chunk] += est * weight
            norm[s : s + chunk] += weight
    return out / norm


def predict_masks(
    model: nn.Module,
    audio: np.ndarray,
    stft_cfg: STFTConfig,
    chunk_seconds: float,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(masks, estimates)``.

    ``masks`` is ``(S, F, frames)`` on the ``dsp.stft.stft`` grid, in
    ``[0, 1]`` and summing to one over sources at every bin (bins where all
    estimates are silent get ``1/S``). ``estimates`` is ``(S, T)``.
    """
    est = separate(model, audio, stft_cfg.sample_rate, chunk_seconds, device=device)
    mags = np.stack([np.abs(stft(e, stft_cfg)) for e in est])
    total = mags.sum(axis=0, keepdims=True)
    masks = np.where(total > 1e-12, mags / np.maximum(total, 1e-12), 1.0 / len(est))
    return masks.astype(np.float32), est


def save_checkpoint(path: str | Path, model: nn.Module, arch: dict, stft_cfg: STFTConfig, **meta) -> None:
    """Self-contained checkpoint: rebuilding it needs no config and no download."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "sources": list(model.sources),
        "stft": asdict(stft_cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts best.pt


def load_separator(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a separator from ``save_checkpoint`` output; returns ``(model, meta)``.

    ``weights_only=False`` because HTDemucs stores its segment length as a
    ``fractions.Fraction``. Only load checkpoints you produced yourself.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stft_cfg = STFTConfig(**payload["stft"])
    model = _rebuild(payload["arch"], payload["sources"], stft_cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta["stft_config"] = stft_cfg
    return model, meta
