"""Shared audio loading and the per-track container every corpus returns.

All corpora go through ``load_audio`` so that resampling and downmixing happen
in exactly one place. A stem loaded at 48 kHz next to a mixture loaded at
44.1 kHz is a silent SDR killer: nothing crashes, the numbers are just wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["TrackItem", "load_audio", "load_segment", "audio_duration", "AUDIO_EXTENSIONS"]

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


def audio_duration(path: str | Path) -> float:
    """Duration in seconds, read from the header only."""
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def load_segment(path: str | Path, sample_rate: int, start_seconds: float, seconds: float) -> np.ndarray:
    """Load ``seconds`` of mono audio from ``start_seconds``, at ``sample_rate``.

    Reads only the requested frames from disk, so random training crops do not
    decode whole concert-length files. The result is always exactly
    ``round(seconds * sample_rate)`` samples (zero-padded past the end of the
    file), which keeps stems from different files sample-aligned.
    """
    import soundfile as sf

    n_out = int(round(seconds * sample_rate))
    with sf.SoundFile(str(path)) as f:
        native = f.samplerate
        f.seek(min(int(round(start_seconds * native)), f.frames))
        audio = f.read(int(round(seconds * native)), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if native != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=native, target_sr=sample_rate)
    if len(audio) < n_out:
        audio = np.pad(audio, (0, n_out - len(audio)))
    return audio[:n_out].astype(np.float32)
