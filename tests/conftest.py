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


def make_sanidha(root, n_tracks: int = 3, duration: float = 3.0, sample_rate: int = 8000):
    """Write a tiny Sanidha-layout corpus: four stems per track, no mixdown.

    Each "instrument" is distinguishable by construction -- two tones at
    different pitches, and two click trains at different rates -- so a model
    that learns anything at all separates them.
    """
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_tracks):
        track = root / f"track{i:02d}"
        track.mkdir()
        f0 = 150.0 + 20.0 * i  # a different tonic per track
        stems = {
            "vocal": make_tone(duration, f0, sample_rate),
            "violin": 0.6 * make_tone(duration, f0 * 1.5, sample_rate),
            "mridangam": make_clicks(duration, 4.0, sample_rate),
            "ghatam": 0.5 * make_clicks(duration, 6.0, sample_rate),
        }
        for name, audio in stems.items():
            sf.write(track / f"{name}.wav", audio.astype(np.float32), sample_rate)
    return root


def make_saraga(root, n_concerts: int = 2, duration: float = 3.0, sample_rate: int = 8000):
    """Write a tiny Saraga-layout corpus: concert/piece nesting, real mixdown.

    Mirrors the release faithfully in the ways the loader has to cope with --
    two mridangam mics, a second vocal mic, ghatam on only some pieces, the
    ``.mp3.mp3`` double extension on the master, AppleDouble twins beside every
    stem, and the same piece name reused across concerts.
    """
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    for c in range(n_concerts):
        concert = root / f"Concert {c} by Some Artist"
        for p, piece_name in enumerate(("Kanukontini", "Athirum Kazhal")):
            piece = concert / piece_name  # same names in every concert, as in the release
            piece.mkdir(parents=True)
            f0 = 150.0 + 20.0 * (2 * c + p)
            tracks = {
                "multitrack-vocal": make_tone(duration, f0, sample_rate),
                "multitrack-vocal-s": 0.3 * make_tone(duration, f0, sample_rate),
                "multitrack-violin": 0.6 * make_tone(duration, f0 * 1.5, sample_rate),
                "multitrack-mridangam-left": make_clicks(duration, 4.0, sample_rate),
                "multitrack-mridangam-right": 0.8 * make_clicks(duration, 4.0, sample_rate),
            }
            if p == 0:  # ghatam only on a subset, as in the release
                tracks["multitrack-ghatam"] = 0.5 * make_clicks(duration, 6.0, sample_rate)
            for tag, audio in tracks.items():
                sf.write(piece / f"{piece_name}.{tag}.wav", audio.astype(np.float32), sample_rate)
                (piece / f"._{piece_name}.{tag}.wav").write_bytes(b"\x00\x05\x16\x07")

            # The released master: engineered separately, so deliberately NOT
            # the sum of the mics -- that difference is what mixture="mixdown"
            # exposes.
            master = 0.7 * sum(tracks.values())
            sf.write(piece / f"{piece_name}.wav.wav", master.astype(np.float32), sample_rate)
    return root
