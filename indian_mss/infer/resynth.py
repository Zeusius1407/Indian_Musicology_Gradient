"""Milestone 3: turn predicted mask ratios back into waveforms.

The separator's deliverable is ``masks``: ``(S, F, frames)`` ratios on the
pipeline STFT grid. Multiplying them into the mixture magnitude gives one
magnitude spectrogram per instrument; what is missing is phase. Three ways to
supply it:

``mixture_phase``
    Borrow the mixture's phase for every stem, then ISTFT. Exact wherever one
    instrument dominates a bin, and -- because the masks sum to one -- the
    stems add back to the mixture sample for sample. The baseline.
``griffin_lim``
    Throw the phase away and iterate STFT/ISTFT until a phase consistent with
    the magnitude emerges (Griffin & Lim 1984, with the Perraudin 2013
    momentum). No leakage through the borrowed phase, but no phase
    information either: it is what resynthesis costs when phase is unknown.
``vocoder``
    Hand the masked magnitude, as a mel spectrogram, to a pretrained neural
    vocoder (``infer.vocoder``). It generates its own phase and fine
    structure, so it can sound cleaner than either while scoring worse on
    SDR, which rewards sample-level agreement with the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp.stft import STFTConfig, istft, magphase, stft

__all__ = ["METHODS", "ResynthConfig", "griffin_lim", "mixture_phase", "resynthesize"]

METHODS = ("mixture_phase", "griffin_lim", "vocoder")


@dataclass(frozen=True)
class ResynthConfig:
    """``resynthesis:`` block of ``configs/default.yaml``.

    Attributes
    ----------
    method:
        One of ``METHODS``.
    griffin_lim_iters:
        Iterations per stem. With momentum, 32 is past the knee of the
        spectral-convergence curve; plain Griffin-Lim needs several hundred.
    griffin_lim_init:
        ``random`` starts from random phase -- the honest "phase unknown"
        case. ``mixture`` starts from the mixture phase and only refines it,
        which is mixture-phase ISTFT made self-consistent.
    griffin_lim_momentum:
        Perraudin's fast Griffin-Lim acceleration; ``0`` is the original
        algorithm. Values above ~1 diverge.
    vocoder:
        Hugging Face repo of the BigVGAN checkpoint ``infer.vocoder`` loads.
    """

    method: str = "mixture_phase"
    griffin_lim_iters: int = 32
    griffin_lim_init: str = "random"
    griffin_lim_momentum: float = 0.99
    vocoder: str = "nvidia/bigvgan_v2_44khz_128band_512x"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown resynthesis method {self.method!r}: expected one of {METHODS}")
        if self.griffin_lim_init not in ("random", "mixture"):
            raise ValueError(f"griffin_lim_init must be 'random' or 'mixture', got {self.griffin_lim_init!r}")

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "ResynthConfig":
        r = {**cfg.get("resynthesis", {}), **{k: v for k, v in overrides.items() if v is not None}}
        return cls(
            method=str(r.get("method", cls.method)),
            griffin_lim_iters=int(r.get("griffin_lim_iters", cls.griffin_lim_iters)),
            griffin_lim_init=str(r.get("griffin_lim_init", cls.griffin_lim_init)),
            griffin_lim_momentum=float(r.get("griffin_lim_momentum", cls.griffin_lim_momentum)),
            vocoder=str(r.get("vocoder", cls.vocoder)),
        )


def _check_masks(masks: np.ndarray, spec: np.ndarray) -> None:
    if masks.ndim != 3 or masks.shape[1:] != spec.shape:
        raise ValueError(
            f"masks {masks.shape} do not match the mixture STFT {spec.shape}; "
            "they must come from the same STFT settings and the same audio"
        )


def mixture_phase(masks: np.ndarray, mixture: np.ndarray, stft_cfg: STFTConfig) -> np.ndarray:
    """``mask * |mix| * phase(mix)`` -> ISTFT, for every source. Returns ``(S, T)``."""
    spec = stft(mixture, stft_cfg)
    _check_masks(masks, spec)
    n = len(mixture)
    # mask * |X| * e^{j angle X} is simply mask * X.
    return np.stack([istft(m * spec, stft_cfg, length=n) for m in masks]).astype(np.float32)


def griffin_lim(
    magnitude: np.ndarray,
    stft_cfg: STFTConfig,
    length: int,
    n_iter: int = 32,
    init_phase: np.ndarray | None = None,
    momentum: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """Fast Griffin-Lim: a waveform whose STFT magnitude approximates ``magnitude``.

    ``init_phase`` is a unit-modulus array (``magphase``'s second output);
    ``None`` starts from uniformly random phase drawn with ``seed``.
    """
    if init_phase is None:
        rng = np.random.default_rng(seed)
        phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
    else:
        phase = init_phase
    prev = np.zeros_like(magnitude, dtype=np.complex128)
    for _ in range(n_iter):
        rebuilt = stft(istft(magnitude * phase, stft_cfg, length=length), stft_cfg)
        # Momentum step (Perraudin et al. 2013): extrapolate past the
        # projection, then renormalize to unit modulus.
        accel = rebuilt - (momentum / (1.0 + momentum)) * prev
        phase = accel / np.maximum(np.abs(accel), 1e-16)
        prev = rebuilt
    return istft(magnitude * phase, stft_cfg, length=length).astype(np.float32)


def resynthesize(
    masks: np.ndarray,
    mixture: np.ndarray,
    stft_cfg: STFTConfig,
    cfg: ResynthConfig = ResynthConfig(),
    vocoder=None,
) -> np.ndarray:
    """Waveform per source from ``masks`` with ``cfg.method``. Returns ``(S, T)``.

    ``vocoder`` is an ``infer.vocoder.Vocoder``; pass one in when calling this
    repeatedly so the network is loaded once, not per track.
    """
    if cfg.method == "mixture_phase":
        return mixture_phase(masks, mixture, stft_cfg)

    if cfg.method == "griffin_lim":
        spec = stft(mixture, stft_cfg)
        _check_masks(masks, spec)
        mag, phase = magphase(spec)
        init = phase if cfg.griffin_lim_init == "mixture" else None
        return np.stack(
            [
                griffin_lim(m * mag, stft_cfg, len(mixture), cfg.griffin_lim_iters, init, cfg.griffin_lim_momentum, i)
                for i, m in enumerate(masks)
            ]
        )

    if vocoder is None:
        from .vocoder import Vocoder

        vocoder = Vocoder(cfg.vocoder)
    return vocoder.from_masks(masks, mixture, stft_cfg)
