"""Tonic (Sa) estimation and tuning normalization.

Indian art music is not tuned to A440. Each performance picks a tonic to suit
the lead artist's range, and every melodic interval in the piece is defined
*relative* to that tonic. A separation model trained without normalizing this
has to learn the same raga phrase separately at a dozen absolute pitches, and
a classifier trained on raw MFCCs partly learns the tonic rather than timbre.

Two strategies are supported:

``shift``
    Pitch-shift the audio so the tonic lands on a canonical frequency. Simple,
    and makes downstream code tonic-agnostic. Costs a resampling artefact and
    is the wrong choice if you later want to report absolute pitch.

``condition``
    Leave the audio alone and carry the tonic as metadata, to be fed to the
    network as a conditioning input. Lossless, but every downstream component
    must accept the extra input.

This module implements estimation plus the ``shift`` strategy, and returns the
tonic so that ``condition`` is available to callers who want it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_tonic", "normalize_tuning", "CANONICAL_TONIC_HZ"]

#: Canonical tonic to normalize to. 146.83 Hz is D3, a common male vocal Sa;
#: the exact value is arbitrary but must be fixed across the whole dataset.
CANONICAL_TONIC_HZ = 146.83


def estimate_tonic(audio: np.ndarray, sample_rate: int, fmin: float = 100.0, fmax: float = 400.0) -> float:
    """Estimate the tonic frequency in Hz from a multi-pitch histogram.

    Method: track pitch with librosa's pYIN over the whole excerpt, fold the
    detected pitches into a one-octave cent histogram, and take the mode. In
    Indian art music the drone sounds continuously on Sa (and Pa), so the
    tonic is the most-occupied pitch class by a wide margin.

    This is a dependency-light stand-in for compIAM's dedicated tonic
    identification models (``compiam.melody.tonic_identification``). Prefer
    those on real data -- they were trained for exactly this task and handle
    the Sa/Pa ambiguity better. Swap the call here and nothing else changes.

    Returns
    -------
    Tonic frequency in Hz, or ``CANONICAL_TONIC_HZ`` if no voiced pitch was
    found (which is what happens on a percussion-only excerpt).
    """
    import librosa  # imported lazily so the core DSP has no heavy dependency

    f0, voiced, _ = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
    )
    f0 = f0[voiced & np.isfinite(f0)]
    if f0.size == 0:
        return CANONICAL_TONIC_HZ

    # Fold into one octave relative to fmin, in cents, then histogram.
    cents = 1200.0 * np.log2(f0 / fmin)
    folded = np.mod(cents, 1200.0)
    hist, edges = np.histogram(folded, bins=120, range=(0.0, 1200.0))  # 10-cent bins
    peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])

    # Map the winning pitch class back to the octave where most energy sits.
    candidate = fmin * 2 ** (peak / 1200.0)
    while candidate < fmin:
        candidate *= 2
    while candidate > fmax:
        candidate /= 2
    return float(candidate)


def normalize_tuning(
    audio: np.ndarray,
    sample_rate: int,
    tonic_hz: float | None = None,
    target_hz: float = CANONICAL_TONIC_HZ,
) -> tuple[np.ndarray, float]:
    """Pitch-shift ``audio`` so its tonic sits at ``target_hz``.

    Returns ``(shifted_audio, estimated_tonic_hz)``. Store the returned tonic
    in your manifest: you need it to undo the shift before delivering audio to
    a listener, and it is the conditioning feature if you switch strategies.
    """
    import librosa

    if tonic_hz is None:
        tonic_hz = estimate_tonic(audio, sample_rate)

    n_steps = 12.0 * np.log2(target_hz / tonic_hz)
    if abs(n_steps) < 0.01:
        return audio, tonic_hz

    shifted = librosa.effects.pitch_shift(
        np.asarray(audio, dtype=np.float32), sr=sample_rate, n_steps=float(n_steps)
    )
    return shifted, tonic_hz
