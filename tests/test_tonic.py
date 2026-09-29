"""Tests for tonic estimation and tuning normalization.

Why this layer needs guarding: every melodic interval in Indian art music is
defined relative to the performance's own Sa. If ``estimate_tonic`` silently
drifts by an octave or a fifth, nothing downstream raises -- the separation
model just quietly learns the same raga phrase twice at different absolute
pitches, and the Milestone 4 classifier partly learns the tonic instead of the
timbre. A wrong tonic is invisible without a test like this one.
"""

from __future__ import annotations

import numpy as np
import pytest

from indian_mss.dsp.tonic import CANONICAL_TONIC_HZ, estimate_tonic, normalize_tuning

from conftest import SR, make_clicks, make_tone

# pYIN is the slow part of this module; keep excerpts short.
DURATION = 3.0


def cents(a: float, b: float) -> float:
    """Absolute interval between two frequencies, in cents."""
    return abs(1200.0 * np.log2(a / b))


@pytest.mark.parametrize("freq", [146.83, 176.2, 220.0])
def test_estimate_tonic_recovers_a_drone(freq):
    """A sustained drone's fundamental is the tonic, within a few cents.

    Tolerance is 25 cents -- a quarter semitone. The histogram in
    ``estimate_tonic`` uses 10-cent bins, so anything tighter would be testing
    the bin width rather than the method.
    """
    audio = make_tone(DURATION, freq=freq)
    assert cents(estimate_tonic(audio, SR), freq) < 25.0


def test_estimate_tonic_falls_back_when_unvoiced():
    """Percussion-only input has no pitch, so the documented fallback applies.

    This is not a corner case: any excerpt of a tani avartanam hits it, and
    returning 0 or NaN instead would make the pitch-shift downstream produce
    silence or crash.
    """
    assert estimate_tonic(make_clicks(DURATION), SR) == CANONICAL_TONIC_HZ


def test_estimate_tonic_stays_in_the_search_band():
    """The returned tonic must be octave-folded into [fmin, fmax]."""
    audio = make_tone(DURATION, freq=220.0)
    tonic = estimate_tonic(audio, SR, fmin=100.0, fmax=400.0)
    assert 100.0 <= tonic <= 400.0


def test_normalize_tuning_is_a_noop_at_the_target():
    """Audio already at the canonical tonic must not be resampled.

    ``pitch_shift`` is lossy, so the short-circuit matters: re-normalizing an
    already-normalized corpus should not degrade it a second time.
    """
    audio = make_tone(DURATION, freq=CANONICAL_TONIC_HZ)
    shifted, tonic = normalize_tuning(audio, SR)
    assert cents(tonic, CANONICAL_TONIC_HZ) < 25.0
    # Identical object, not merely close -- proves no shift was applied.
    assert shifted is audio


def test_normalize_tuning_reports_the_estimate_it_used():
    """The returned tonic is the *source* tonic, which callers must store."""
    audio = make_tone(DURATION, freq=196.0)
    _, tonic = normalize_tuning(audio, SR)
    assert cents(tonic, 196.0) < 25.0


def test_normalize_tuning_accepts_a_known_tonic():
    """Passing ``tonic_hz`` skips estimation and echoes the value back."""
    audio = make_tone(DURATION, freq=196.0)
    _, tonic = normalize_tuning(audio, SR, tonic_hz=196.0)
    assert tonic == 196.0


def test_normalize_tuning_moves_the_tonic_to_the_target():
    """End to end: shift a 196 Hz drone and re-estimate at the canonical Sa.

    Tolerance is loose (50 cents) on purpose. ``librosa.effects.pitch_shift``
    resamples through a phase vocoder, and the artefacts it leaves shift pYIN's
    estimate by more than the histogram bin width.
    """
    audio = make_tone(DURATION, freq=196.0)
    shifted, _ = normalize_tuning(audio, SR)
    assert shifted is not audio
    assert cents(estimate_tonic(shifted, SR), CANONICAL_TONIC_HZ) < 50.0
