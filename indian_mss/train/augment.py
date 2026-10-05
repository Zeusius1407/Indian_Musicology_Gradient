"""Training-time augmentation, applied to one example's stems ``(S, T)``.

Pure numpy, so it runs inside DataLoader workers and is testable without a
GPU. Every function returns new stems and leaves the mixture to the caller,
who always rebuilds it as ``stems.sum(0)`` -- the invariant that keeps the
training target consistent with the input no matter what was augmented.

Simulated bleed -- the augmentation that matters
------------------------------------------------
Sanidha is recorded in isolation booths; Saraga is recorded live, where every
microphone also picks up delayed, room-coloured, attenuated copies of the
other instruments. On a mixdown that leakage shows up as extra reverberant
copies of each source. ``simulate_bleed`` adds exactly that, and labels every
copy as belonging to the instrument that *produced* it: the leaked mridangam
in the violin mic is still mridangam. That is what teaches the masks to
follow the instrument rather than the microphone, and it is what carries a
model trained on studio-clean Sanidha to live concert audio.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

__all__ = ["random_gain", "pitch_shift", "simulate_bleed", "synthetic_rir", "augment_stems"]


def random_gain(stems: np.ndarray, rng: np.random.Generator, db_range: tuple[float, float]) -> np.ndarray:
    """Independent gain per stem -- breaks the fixed balance of one mix engineer."""
    gains_db = rng.uniform(*db_range, size=(len(stems), 1))
    return stems * (10.0 ** (gains_db / 20.0)).astype(stems.dtype)


def pitch_shift(
    stems: np.ndarray, rng: np.random.Generator, sample_rate: int, semitone_range: tuple[float, float]
) -> np.ndarray:
    """Shift every stem by the *same* random interval.

    The same interval for all stems keeps the ensemble in tune with its own
    tonic -- the drone, the voice and a tuned mridangam stay consistent -- while
    exposing the model to tonics it never saw in the corpus.
    """
    import librosa

    n_steps = float(rng.uniform(*semitone_range))
    if abs(n_steps) < 0.05:
        return stems
    out = np.stack([librosa.effects.pitch_shift(s, sr=sample_rate, n_steps=n_steps) for s in stems])
    return out.astype(stems.dtype)


def synthetic_rir(rng: np.random.Generator, sample_rate: int, rir_ms: float) -> np.ndarray:
    """A short room response: a mic-distance delay, then exponentially decaying noise.

    Not a measured room, deliberately -- a fresh random one per example stops
    the model from learning one particular hall.
    """
    n = max(2, int(rir_ms / 1000.0 * sample_rate))
    delay = int(rng.uniform(1.0, 10.0) / 1000.0 * sample_rate)  # 0.3--3.4 m between mics
    t = np.arange(n) / sample_rate
    rt60 = rng.uniform(0.3, 1.0) * rir_ms / 1000.0
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir = np.concatenate([np.zeros(delay), tail])
    return (rir / (np.sqrt(np.sum(rir**2)) + 1e-12)).astype(np.float32)


def simulate_bleed(
    stems: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int,
    neighbour_gain_db: tuple[float, float],
    rir_ms: float,
) -> np.ndarray:
    """Add each source's leakage into every *other* microphone, credited to that source.

    Source ``j`` reaches the ``S - 1`` other mics at independent random gains;
    all those copies go through one random room response and are added to
    stem ``j`` itself (see the module docstring for why).
    """
    n_src, n = stems.shape
    out = stems.copy()
    for j in range(n_src):
        if not np.any(stems[j]):
            continue
        gains = 10.0 ** (rng.uniform(*neighbour_gain_db, size=n_src - 1) / 20.0)
        leaked = fftconvolve(stems[j], synthetic_rir(rng, sample_rate, rir_ms))[:n]
        out[j] += float(gains.sum()) * leaked.astype(stems.dtype)
    return out


def augment_stems(stems: np.ndarray, rng: np.random.Generator, sample_rate: int, cfg: dict) -> np.ndarray:
    """Apply the ``augment`` section of the config. Order: pitch, gain, bleed.

    Bleed comes last so leaked copies inherit each stem's final level, just as
    a louder player leaks more in a real room.
    """
    if cfg.get("pitch_shift_semitones") and rng.random() < float(cfg.get("pitch_shift_prob", 0.0)):
        stems = pitch_shift(stems, rng, sample_rate, tuple(cfg["pitch_shift_semitones"]))
    if cfg.get("random_gain_db"):
        stems = random_gain(stems, rng, tuple(cfg["random_gain_db"]))
    bleed = cfg.get("simulate_bleed") or {}
    if bleed.get("enabled") and rng.random() < float(bleed.get("prob", 1.0)):
        stems = simulate_bleed(
            stems, rng, sample_rate, tuple(bleed["neighbour_gain_db"]), float(bleed["rir_ms"])
        )
    return stems
