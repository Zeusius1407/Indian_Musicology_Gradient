"""STFT front end.

One place where the time-frequency transform is defined. Everything downstream
(HPSS, the separation model, resynthesis) uses ``STFTConfig`` so that the
analysis and synthesis windows can never drift out of sync -- the single most
common source of resynthesis artefacts.

The complex spectrogram is kept, not just the magnitude: Milestone 3 needs the
mixture phase to invert predicted magnitude masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sps

__all__ = ["STFTConfig", "stft", "istft", "magphase", "log_magnitude"]


@dataclass(frozen=True)
class STFTConfig:
    """Analysis parameters for the whole pipeline.

    The defaults give ~46 ms windows at 44.1 kHz with 75% overlap. That is long
    enough to resolve the low harmonics of a tanpura drone and short enough to
    keep a mridangam stroke inside two or three frames.
    """

    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    window: str = "hann"

    @property
    def win_length(self) -> int:
        return self.n_fft

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frame_rate(self) -> float:
        """Spectrogram frames per second -- needed to size HPSS median kernels."""
        return self.sample_rate / self.hop_length

    def freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)


def _window(cfg: STFTConfig) -> np.ndarray:
    return sps.get_window(cfg.window, cfg.win_length, fftbins=True)


def stft(audio: np.ndarray, cfg: STFTConfig = STFTConfig()) -> np.ndarray:
    """Complex STFT of a mono signal.

    Parameters
    ----------
    audio:
        1-D float array. Multi-channel input should be handled by the caller
        (separate channels, or downmix) so that the channel convention is
        explicit rather than guessed here.

    Returns
    -------
    Complex array of shape ``(n_freqs, n_frames)``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")

    _, _, spec = sps.stft(
        audio,
        fs=cfg.sample_rate,
        window=_window(cfg),
        nperseg=cfg.win_length,
        noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary="zeros",
        padded=True,
        return_onesided=True,
        scaling="spectrum",
    )
    return spec


def istft(spec: np.ndarray, cfg: STFTConfig = STFTConfig(), length: int | None = None) -> np.ndarray:
    """Inverse STFT. ``length`` trims the result back to the original duration.

    Always pass ``length`` when reconstructing a stem you intend to compare
    against a reference -- the padding introduced by ``stft`` otherwise shifts
    the two signals relative to each other and destroys the SDR.
    """
    _, audio = sps.istft(
        spec,
        fs=cfg.sample_rate,
        window=_window(cfg),
        nperseg=cfg.win_length,
        noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft,
        input_onesided=True,
        boundary=True,
        scaling="spectrum",
    )
    if length is not None:
        if len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))
        audio = audio[:length]
    return audio


def magphase(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a complex spectrogram into magnitude and unit-modulus phase.

    The phase is returned as ``exp(j*angle)`` rather than the angle itself so
    that resynthesis is a plain multiplication: ``mask * mag * phase``.
    """
    mag = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))
    return mag, phase


def log_magnitude(mag: np.ndarray, eps: float = 1e-8, ref_db: float = 80.0) -> np.ndarray:
    """Log-compressed magnitude in [0, 1], the usual network input.

    ``ref_db`` sets the dynamic range floor: anything more than ``ref_db``
    below the loudest bin maps to 0. Without a floor the near-silent bins
    dominate the loss and the model wastes capacity modelling noise.
    """
    db = 20.0 * np.log10(np.maximum(mag, eps))
    db = np.maximum(db, db.max() - ref_db)
    return (db - (db.max() - ref_db)) / ref_db
