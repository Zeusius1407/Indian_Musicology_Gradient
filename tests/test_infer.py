"""Tests for Milestone 3 resynthesis.

The synthetic tone-plus-clicks mix from ``conftest.py`` stands in for a
melodic and a percussive instrument, separated with oracle ratio masks so the
tests isolate the resynthesis step from the quality of any model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from indian_mss.cli.compare_resynthesis import ratio_masks
from indian_mss.dsp.stft import STFTConfig, istft, magphase, stft
from indian_mss.eval.sdr import evaluate_stems, magnitude_snr
from indian_mss.infer.resynth import ResynthConfig, griffin_lim, mixture_phase, resynthesize

from conftest import SR, make_clicks, make_tone

CFG = STFTConfig(sample_rate=SR, n_fft=1024, hop_length=256)


@pytest.fixture(scope="module")
def mix():
    tone, clicks = make_tone(2.0), 0.5 * make_clicks(2.0)
    stems = np.stack([tone, clicks])
    return stems.sum(axis=0), stems


def test_mixture_phase_stems_sum_to_mixture(mix):
    mixture, stems = mix
    est = mixture_phase(ratio_masks(stems, CFG), mixture, CFG)
    assert est.shape == stems.shape
    # Masks sum to one, so resynthesis is a partition of the mixture.
    assert np.allclose(est.sum(axis=0), mixture, atol=1e-5)
    scores = evaluate_stems({"tone": est[0], "clicks": est[1]}, {"tone": stems[0], "clicks": stems[1]})
    assert min(s["sdr"] for s in scores.values()) > 10.0


def test_mixture_phase_rejects_mismatched_masks(mix):
    mixture, stems = mix
    masks = ratio_masks(stems, STFTConfig(sample_rate=SR, n_fft=512, hop_length=128))
    with pytest.raises(ValueError, match="do not match"):
        mixture_phase(masks, mixture, CFG)


def test_griffin_lim_converges_on_magnitude():
    tone = make_tone(1.0)
    mag = np.abs(stft(tone, CFG))
    est = griffin_lim(mag, CFG, len(tone), n_iter=32)
    assert est.shape == tone.shape
    assert magnitude_snr(est, tone, CFG) > 15.0
    assert magnitude_snr(est, tone, CFG) > magnitude_snr(griffin_lim(mag, CFG, len(tone), n_iter=0), tone, CFG)


def test_griffin_lim_from_mixture_phase_keeps_alignment(mix):
    mixture, stems = mix
    cfg = ResynthConfig("griffin_lim", griffin_lim_iters=8, griffin_lim_init="mixture")
    est = resynthesize(ratio_masks(stems, CFG), mixture, CFG, cfg)
    # Started from the mixture phase, the tone stays sample-aligned with its
    # reference -- random init would not.
    assert evaluate_stems({"tone": est[0]}, {"tone": stems[0], "clicks": stems[1]})["tone"]["sdr"] > 10.0


def test_magnitude_snr_is_phase_blind():
    tone = make_tone(1.0)
    spec = stft(tone, CFG)
    mag, _ = magphase(spec)
    rng = np.random.default_rng(0)

    scrambled = istft(mag * np.exp(2j * np.pi * rng.random(mag.shape)), CFG, length=len(tone))
    assert magnitude_snr(tone, tone, CFG) > 60.0
    # Random phase wrecks SDR but only partially spoils the magnitude (the
    # overlap-add re-mixes it), so the two metrics disagree.
    assert magnitude_snr(scrambled, tone, CFG) > 0.0
    assert evaluate_stems({"t": scrambled}, {"t": tone, "z": make_clicks(1.0)})["t"]["sdr"] < 0.0


def test_resynth_config_validates_and_reads_overrides():
    with pytest.raises(ValueError, match="unknown resynthesis method"):
        ResynthConfig("phase_vocoder")
    with pytest.raises(ValueError, match="griffin_lim_init"):
        ResynthConfig(griffin_lim_init="zeros")
    cfg = ResynthConfig.from_config({"resynthesis": {"method": "griffin_lim", "griffin_lim_iters": 8}}, method=None)
    assert (cfg.method, cfg.griffin_lim_iters) == ("griffin_lim", 8)
    assert ResynthConfig.from_config({}, method="vocoder").method == "vocoder"


def test_vocoder_mask_grid_is_half_a_hop_late():
    pytest.importorskip("torch")
    from indian_mss.infer.vocoder import Vocoder

    # Exercise the grid mapping without downloading the network.
    voc = object.__new__(Vocoder)
    voc.h = SimpleNamespace(n_fft=1024, hop_size=256)
    voc.device = "cpu"
    frames = 10
    ramp = np.broadcast_to(np.arange(frames, dtype=np.float32), (1, 3, frames))
    grid = voc._masks_on_vocoder_grid(ramp, CFG, frames - 1).numpy()
    assert np.allclose(grid[0, 0], np.arange(frames - 1) + 0.5)


def test_resynthesize_cli_writes_stems(mix, tmp_path):
    sf = pytest.importorskip("soundfile")
    from dataclasses import asdict

    from indian_mss.cli.resynthesize import main

    mixture, stems = mix
    sf.write(tmp_path / "mix.wav", mixture.astype(np.float32), SR)
    np.save(tmp_path / "mix_masks.npy", ratio_masks(stems, CFG))
    (tmp_path / "mix_masks.json").write_text(json.dumps({"sources": ["tone", "clicks"], "stft": asdict(CFG)}))

    main(
        [
            "--masks", str(tmp_path / "mix_masks.npy"),
            "--input", str(tmp_path / "mix.wav"),
            "--outdir", str(tmp_path / "out"),
            "--method", "griffin_lim",
            "--griffin-lim-iters", "4",
        ]
    )  # fmt: skip
    out, rate = sf.read(tmp_path / "out" / "mix_tone.wav")
    assert rate == SR and len(out) == len(mixture)
    assert json.loads((tmp_path / "out" / "mix_resynthesis.json").read_text())["resynthesis"]["method"] == "griffin_lim"
