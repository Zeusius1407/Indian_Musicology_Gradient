"""Feature extraction for the Milestone 4 timbre classifier.

Kept separate from ``stft.py`` because these are *classifier* inputs, computed
on already-separated stems, and they use a different (shorter, cheaper)
analysis than the separation front end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MFCCConfig", "mfcc", "windowed_mfcc"]


@dataclass(frozen=True)
class MFCCConfig:
    sample_rate: int = 22050  # timbre survives downsampling; halves the cost
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 30.0
    fmax: float | None = None
    include_deltas: bool = True


def mfcc(audio: np.ndarray, cfg: MFCCConfig = MFCCConfig()) -> np.ndarray:
    """MFCCs (optionally with first and second differences).

    Deltas matter more than usual here: the distinguishing feature of tabla
    versus mridangam is the *decay shape* of a stroke, which is a temporal
    derivative, not a static spectral envelope.

    Returns array of shape ``(n_coeffs, n_frames)``.
    """
    import librosa

    m = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    if not cfg.include_deltas:
        return m
    return np.concatenate([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)], axis=0)


def windowed_mfcc(
    audio: np.ndarray,
    cfg: MFCCConfig = MFCCConfig(),
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    drop_silent_db: float = -50.0,
) -> np.ndarray:
    """Cut ``audio`` into fixed windows and return a stack of MFCC patches.

    Shape ``(n_windows, n_coeffs, n_frames)`` -- ready to be treated as
    single-channel images by a 2-D CNN.

    Silent windows are dropped. This is not cosmetic: a separated stem is
    mostly silence wherever its instrument is not playing, and feeding those
    windows to the classifier lets it hit high accuracy by learning "silence
    means whichever stem is usually quiet", which would invalidate the
    Milestone 4 bleed argument entirely.

    Note on splitting: build train/test splits by *recording*, never by
    window. Adjacent windows from one take share room, mic and instrument, so
    a random window split leaks and inflates accuracy by a wide margin.
    """
    win = int(window_seconds * cfg.sample_rate)
    hop = int(hop_seconds * cfg.sample_rate)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))

    threshold = 10 ** (drop_silent_db / 20.0)
    patches = []
    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start : start + win]
        if np.sqrt(np.mean(chunk**2)) < threshold:
            continue
        patches.append(mfcc(chunk, cfg))

    if not patches:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.stack(patches).astype(np.float32)
