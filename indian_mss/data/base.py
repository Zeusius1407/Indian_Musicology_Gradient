"""Shared audio loading and the per-track container every corpus returns.

All corpora go through ``load_audio`` so that resampling and downmixing happen
in exactly one place. A stem loaded at 48 kHz next to a mixture loaded at
44.1 kHz is a silent SDR killer: nothing crashes, the numbers are just wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["TrackItem", "load_audio", "AUDIO_EXTENSIONS"]

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")


@dataclass
class TrackItem:
    """One multitrack recording, sample-aligned and at a single rate."""

    track_id: str
    mixture: np.ndarray  # mono, float32
    stems: dict[str, np.ndarray]  # stem name -> mono, same length as mixture
    sample_rate: int
    # True when no mixdown shipped with the track and ``mixture`` is the plain
    # sum of the stems. Such mixes are easier to separate than a real mixdown,
    # so any score computed on them is optimistic and must be labelled.
    mixture_is_synthetic: bool = False
    metadata: dict = field(default_factory=dict)


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sample_rate``.

    Multi-channel files are averaged to mono. The STFT front end is mono-only
    by design (see ``dsp.stft.stft``); downmixing here keeps that convention
    explicit instead of guessed further down.
    """
    import librosa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)
