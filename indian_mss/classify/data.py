"""MFCC windows for the Milestone 4 timbre classifier.

Training data are Sanidha's clean close-mic stems, one class per stem. The
proposal names IMIDataset, but its ten classes contain neither voice,
mridangam nor ghatam -- three of the four instruments the separator outputs
-- so a classifier trained on it could not score most of the stems. Sanidha
covers all four, at near-zero bleed, which is what a "clean timbre" label
needs.

Two rules from ``dsp/features.py`` are enforced here rather than trusted:

* **Silent windows are dropped by the reference, not the input.** At test time
  one set of windows -- those where the ground-truth instrument is playing --
  is scored on every input (reference, separated stem, mixture). Choosing
  windows from each input instead would let a separated stem that is silent
  where it should not be simply vanish from its own score.
* **Splits are by recording.** Windows only ever come from whole tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp.features import MFCCConfig, mfcc

__all__ = ["WindowConfig", "active_starts", "mfcc_windows", "window_rms_db"]


@dataclass(frozen=True)
class WindowConfig:
    """How a stem is cut into classifier inputs.

    Attributes
    ----------
    window_seconds, hop_seconds:
        Patch length and spacing. One second holds several mridangam strokes
        and at least one full gamaka.
    silence_db:
        Absolute floor (dBFS RMS): quieter windows are silence.
    silence_rel_db:
        Relative floor below the stem's own loud level (its 95th-percentile
        window RMS). Close mics are recorded at very different gains, and a
        quiet stem's low-level bleed would pass a fixed floor alone.
    """

    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    silence_db: float = -50.0
    silence_rel_db: float = -30.0


def window_rms_db(audio: np.ndarray, sample_rate: int, cfg: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, rms_db)`` for every full window of ``audio``."""
    win = int(cfg.window_seconds * sample_rate)
    hop = int(cfg.hop_seconds * sample_rate)
    if len(audio) < win:
        return np.empty(0, dtype=int), np.empty(0)
    starts = np.arange(0, len(audio) - win + 1, hop)
    # Cumulative sum of squares: one pass, however many windows.
    c = np.concatenate([[0.0], np.cumsum(np.asarray(audio, dtype=np.float64) ** 2)])
    rms = np.sqrt((c[starts + win] - c[starts]) / win)
    return starts, 20.0 * np.log10(np.maximum(rms, 1e-10))


def active_starts(reference: np.ndarray, sample_rate: int, cfg: WindowConfig = WindowConfig()) -> np.ndarray:
    """Start samples of the windows where ``reference`` is audibly playing."""
    starts, db = window_rms_db(reference, sample_rate, cfg)
    if not len(starts):
        return starts
    floor = max(cfg.silence_db, float(np.percentile(db, 95)) + cfg.silence_rel_db)
    return starts[db >= floor]


def mfcc_windows(
    audio: np.ndarray, starts: np.ndarray, mfcc_cfg: MFCCConfig, window_seconds: float
) -> np.ndarray:
    """``(n, coeffs, frames)`` MFCC patches at ``starts`` (samples at ``mfcc_cfg.sample_rate``).

    MFCCs are computed once over the whole signal and sliced, rather than per
    window: identical inside a window, and an hour of audio takes seconds
    instead of minutes. Deltas then see real context at window edges.
    """
    n = int(round(window_seconds * mfcc_cfg.sample_rate / mfcc_cfg.hop_length))
    if not len(starts):
        return np.empty((0, 0, n), dtype=np.float32)
    feats = mfcc(audio, mfcc_cfg)
    # librosa centres frames: frame k is centred on sample k * hop.
    first = np.round(np.asarray(starts) / mfcc_cfg.hop_length).astype(int)
    first = np.minimum(first, feats.shape[1] - n)
    return np.stack([feats[:, f : f + n] for f in first]).astype(np.float32)
