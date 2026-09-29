"""Tests for the Milestone 1 DSP layer.

The synthetic signals come from ``conftest.py`` and are deliberately extreme
-- a pure sustained tone plus sparse clicks -- because if HPSS cannot separate
*that*, no amount of tuning will help on real audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from indian_mss.dsp.hpss import HPSSConfig, hpss_audio, hpss_masks, hpss_spectrograms
from indian_mss.dsp.stft import STFTConfig, istft, log_magnitude, magphase, stft
from indian_mss.eval.sdr import bss_eval, evaluate_stems, si_sdr

from conftest import SR, make_clicks, make_tone

CFG = STFTConfig(sample_rate=SR, n_fft=1024, hop_length=256)


def test_stft_istft_roundtrip():
    audio = make_tone(1.0)
    rec = istft(stft(audio, CFG), CFG, length=len(audio))
    assert rec.shape == audio.shape
    # Perfect reconstruction up to numerical error.
    assert si_sdr(rec, audio) > 60.0


def test_magphase_recombines():
    spec = stft(make_tone(0.5), CFG)
    mag, phase = magphase(spec)
    assert np.allclose(mag * phase, spec, atol=1e-8)


def test_log_magnitude_bounded():
    mag, _ = magphase(stft(make_tone(0.5), CFG))
    lm = log_magnitude(mag)
    assert lm.min() >= 0.0 and lm.max() <= 1.0 + 1e-9


def test_kernel_sizes_are_odd_and_positive():
    frames, bins = HPSSConfig().kernel_sizes(CFG)
    assert frames % 2 == 1 and bins % 2 == 1
    assert frames >= 3 and bins >= 3


def test_masks_are_complementary_at_unit_margin(mixture):
    mix, _, _ = mixture
    mag, _ = magphase(stft(mix, CFG))
    mh, mp = hpss_masks(mag, CFG, HPSSConfig(margin=1.0))
    assert np.allclose(mh + mp, 1.0, atol=1e-6)
    assert mh.min() >= 0.0 and mh.max() <= 1.0


def test_margin_creates_residual(mixture):
    mix, _, _ = mixture
    res = hpss_spectrograms(mix, CFG, HPSSConfig(margin=3.0))
    assert res.residual.sum() > 0.0


def test_energy_is_conserved(mixture):
    """Harmonic + percussive + residual magnitudes must reconstruct the mix."""
    mix, _, _ = mixture
    res = hpss_spectrograms(mix, CFG, HPSSConfig(margin=1.0))
    total = res.harmonic + res.percussive + res.residual
    mag, _ = magphase(stft(mix, CFG))
    assert np.allclose(total, mag, atol=1e-6)


def test_hpss_actually_separates(mixture):
    """The real assertion: each output must resemble its own source."""
    mix, harm, perc = mixture
    est = hpss_audio(mix, CFG, HPSSConfig(harmonic_ms=200.0, percussive_hz=500.0))
    scores = evaluate_stems(
        {"harmonic": est["harmonic"], "percussive": est["percussive"]},
        {"harmonic": harm, "percussive": perc},
    )
    assert scores["harmonic"]["sdr"] > 5.0, scores
    assert scores["percussive"]["sdr"] > 5.0, scores
    # And each estimate must be closer to its own source than to the other.
    assert si_sdr(est["harmonic"], harm) > si_sdr(est["harmonic"], perc)


def test_bss_eval_perfect_estimate():
    h, p = make_tone(1.0), make_clicks(1.0)
    m = bss_eval(h, h, p[None, :])
    assert m.sdr > 60.0 and m.sir > 60.0


def test_si_sdr_is_scale_invariant():
    h = make_tone(1.0)
    assert si_sdr(h * 0.01, h) == pytest.approx(si_sdr(h, h), abs=1e-6)


def test_istft_length_alignment():
    """A length mismatch is the classic silent SDR killer -- guard it."""
    audio = make_tone(1.3)
    rec = istft(stft(audio, CFG), CFG, length=len(audio))
    assert len(rec) == len(audio)
