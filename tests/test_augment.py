"""Tests for training-time augmentation. Pure numpy -- run without torch."""

from __future__ import annotations

import numpy as np

from indian_mss.train.augment import augment_stems, random_gain, simulate_bleed, synthetic_rir

from conftest import make_clicks, make_tone

SR = 8000


def _stems(duration=1.0):
    return np.stack(
        [make_tone(duration, 200.0, SR), make_tone(duration, 300.0, SR), make_clicks(duration, 4.0, SR),
         np.zeros(int(duration * SR))]
    ).astype(np.float32)


def test_random_gain_stays_in_range():
    stems = _stems()
    out = random_gain(stems, np.random.default_rng(0), (-6.0, 6.0))
    ratio = np.abs(out[:3]).max(axis=1) / np.abs(stems[:3]).max(axis=1)
    assert np.all(ratio >= 10 ** (-6 / 20) - 1e-6) and np.all(ratio <= 10 ** (6 / 20) + 1e-6)


def test_bleed_is_credited_to_its_source_and_spares_silent_stems():
    stems = _stems()
    out = simulate_bleed(stems, np.random.default_rng(0), SR, (-30.0, -12.0), 80.0)
    assert out.shape == stems.shape
    assert not np.any(out[3])  # a silent instrument leaks nothing
    # Leakage is a quieter, delayed copy added on top of the dry source.
    extra = out[0] - stems[0]
    assert 0.0 < np.sqrt(np.mean(extra**2)) < np.sqrt(np.mean(stems[0] ** 2))


def test_synthetic_rir_is_unit_energy_and_delayed():
    rir = synthetic_rir(np.random.default_rng(0), SR, 80.0)
    assert np.isclose(np.sum(rir**2), 1.0, atol=1e-4)
    assert rir[0] == 0.0


def test_augment_is_deterministic_per_seed():
    cfg = {"random_gain_db": [-6, 6], "simulate_bleed": {"enabled": True, "neighbour_gain_db": [-30, -12], "rir_ms": 50}}
    a = augment_stems(_stems(), np.random.default_rng(7), SR, cfg)
    b = augment_stems(_stems(), np.random.default_rng(7), SR, cfg)
    assert np.array_equal(a, b)
