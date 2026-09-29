"""Harmonic-Percussive Source Separation by median filtering.

Implemented from the Fitzgerald (2010) / Driedger (2014) formulation rather
than calling ``librosa.decompose.hpss``, because Milestone 1 asks for the
filter itself and because the kernel sizes need to be tuned for Indian art
music: a mridangam or tabla stroke is far shorter than a Western kick drum,
and a gamaka-laden violin line is far less horizontal than a Western sustained
note.

Intuition
---------
In a log-magnitude spectrogram a sustained pitched sound is a *horizontal*
ridge (stable frequency, long duration) and a drum stroke is a *vertical*
ridge (broadband, one instant). Median filtering along time therefore
suppresses transients and keeps harmonics; median filtering along frequency
does the reverse. Comparing the two estimates bin-by-bin yields a mask.

The residual matters here. With ``margin > 1`` bins that are not clearly
either become a third "residual" output -- for Carnatic audio the ghatam and
the plucked attack of a veena land there, and throwing them silently into the
harmonic stem is what makes naive HPSS sound wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .stft import STFTConfig, istft, magphase, stft

__all__ = ["HPSSConfig", "hpss_masks", "hpss_spectrograms", "hpss_audio", "HPSSResult"]


@dataclass(frozen=True)
class HPSSConfig:
    """HPSS hyperparameters.

    Attributes
    ----------
    harmonic_ms:
        Length of the time-direction median kernel in milliseconds. Should be
        longer than the longest percussive stroke you want removed and shorter
        than the shortest melodic note you want kept. ~200 ms is a reasonable
        start for Carnatic tempi; drop it for fast tani avartanam passages.
    percussive_hz:
        Width of the frequency-direction median kernel in Hz. Should exceed the
        spacing between adjacent harmonics of the lowest pitched instrument
        present (~500 Hz covers a tanpura down to a low Sa).
    power:
        Exponent of the soft (Wiener) mask. ``power=2`` is the standard soft
        mask; large values approach a binary mask, which is sharper but adds
        musical noise.
    margin:
        Separation factor. ``1.0`` gives complementary masks that sum to one.
        Values above 1 require a bin to dominate by that factor before it is
        assigned, and the unassigned energy is returned as a residual.
    """

    harmonic_ms: float = 200.0
    percussive_hz: float = 500.0
    power: float = 2.0
    margin: float = 1.0

    def kernel_sizes(self, cfg: STFTConfig) -> tuple[int, int]:
        """Convert the physical kernel sizes into (frames, bins), forced odd."""
        frames = int(round(self.harmonic_ms / 1000.0 * cfg.frame_rate))
        bin_hz = cfg.sample_rate / cfg.n_fft
        bins = int(round(self.percussive_hz / bin_hz))
        return max(3, frames | 1), max(3, bins | 1)


@dataclass
class HPSSResult:
    """Container for the Milestone 1 deliverable."""

    harmonic: np.ndarray  # magnitude spectrogram, melodic content
    percussive: np.ndarray  # magnitude spectrogram, transient content
    residual: np.ndarray  # magnitude spectrogram, unassigned energy
    mask_h: np.ndarray
    mask_p: np.ndarray
    phase: np.ndarray  # mixture phase, needed for resynthesis
    stft_config: STFTConfig
    hpss_config: HPSSConfig


def hpss_masks(
    mag: np.ndarray,
    stft_cfg: STFTConfig = STFTConfig(),
    hpss_cfg: HPSSConfig = HPSSConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Soft harmonic and percussive masks for a magnitude spectrogram.

    Returns masks in [0, 1] of the same shape as ``mag``. When
    ``margin == 1`` they sum to one; otherwise the shortfall is residual.
    """
    if mag.ndim != 2:
        raise ValueError(f"expected a 2-D magnitude spectrogram, got {mag.shape}")

    n_frames, n_bins = hpss_cfg.kernel_sizes(stft_cfg)

    # Horizontal filter -> harmonic estimate; vertical filter -> percussive.
    # 'reflect' avoids the edge darkening that 'constant' would introduce at
    # the start and end of a track.
    harm_ref = median_filter(mag, size=(1, n_frames), mode="reflect")
    perc_ref = median_filter(mag, size=(n_bins, 1), mode="reflect")

    p = hpss_cfg.power
    eps = np.finfo(np.float64).tiny

    h_p = harm_ref.astype(np.float64) ** p
    p_p = perc_ref.astype(np.float64) ** p
    total = h_p + p_p + eps

    if hpss_cfg.margin <= 1.0:
        return h_p / total, p_p / total

    # Margin form: a bin is harmonic only if it beats the percussive estimate
    # by the margin, and vice versa. Everything else falls through to residual.
    m = hpss_cfg.margin
    mask_h = (harm_ref > perc_ref * m).astype(np.float64) * (h_p / total)
    mask_p = (perc_ref > harm_ref * m).astype(np.float64) * (p_p / total)
    return mask_h, mask_p


def hpss_spectrograms(
    audio: np.ndarray,
    stft_cfg: STFTConfig = STFTConfig(),
    hpss_cfg: HPSSConfig = HPSSConfig(),
) -> HPSSResult:
    """Milestone 1 deliverable: mix in, melodic and percussive arrays out."""
    spec = stft(audio, stft_cfg)
    mag, phase = magphase(spec)
    mask_h, mask_p = hpss_masks(mag, stft_cfg, hpss_cfg)
    residual_mask = np.clip(1.0 - mask_h - mask_p, 0.0, 1.0)
    return HPSSResult(
        harmonic=mag * mask_h,
        percussive=mag * mask_p,
        residual=mag * residual_mask,
        mask_h=mask_h,
        mask_p=mask_p,
        phase=phase,
        stft_config=stft_cfg,
        hpss_config=hpss_cfg,
    )


def hpss_audio(
    audio: np.ndarray,
    stft_cfg: STFTConfig = STFTConfig(),
    hpss_cfg: HPSSConfig = HPSSConfig(),
) -> dict[str, np.ndarray]:
    """HPSS all the way back to time domain, using the mixture phase.

    This doubles as the Milestone 3 baseline: the same mixture-phase inversion
    is what the neural masks will use before any vocoder is introduced.
    """
    res = hpss_spectrograms(audio, stft_cfg, hpss_cfg)
    n = len(audio)
    return {
        "harmonic": istft(res.harmonic * res.phase, stft_cfg, length=n),
        "percussive": istft(res.percussive * res.phase, stft_cfg, length=n),
        "residual": istft(res.residual * res.phase, stft_cfg, length=n),
    }
