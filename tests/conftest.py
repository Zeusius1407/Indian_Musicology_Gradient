"""Shared synthetic fixtures for the Milestone 1 tests.

The signals here are deliberately extreme -- a pure sustained tone plus sparse
broadband clicks -- because if the DSP cannot handle *that*, no amount of
kernel tuning will help on real audio. Keeping them in one place means the
HPSS, tonic and feature tests all describe the same "instrument".
"""

from __future__ import annotations

import numpy as np
import pytest

#: Analysis rate for the tests. Half of the pipeline default, which keeps pYIN
#: and the median filters fast without changing any behaviour under test.
SR = 22050


def make_tone(duration: float = 4.0, freq: float = 220.0, sample_rate: int = SR) -> np.ndarray:
    """Sustained harmonic tone -- stands in for a melodic instrument.

    Four harmonics with 1/k amplitude decay, which is enough structure for
    pYIN to lock onto ``freq`` as the fundamental.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return sum(0.5 / (k + 1) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(4))


def make_clicks(duration: float = 4.0, rate: float = 4.0, sample_rate: int = SR) -> np.ndarray:
    """Sparse decaying broadband bursts -- stands in for tonal percussion."""
    n = int(duration * sample_rate)
    out = np.zeros(n)
    rng = np.random.default_rng(0)
    burst_len = int(0.03 * sample_rate)
    envelope = np.exp(-np.linspace(0, 8, burst_len))
    for start in range(0, n - burst_len, int(sample_rate / rate)):
        out[start : start + burst_len] += rng.standard_normal(burst_len) * envelope
    return out


@pytest.fixture
def mixture():
    """``(mix, harmonic, percussive)`` -- the three signals, sample-aligned."""
    h, p = make_tone(), make_clicks()
    return h + p, h, p
