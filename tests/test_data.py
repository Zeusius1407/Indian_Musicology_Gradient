"""Tests for the data layer and the Milestone 1 CLI, on synthetic files.

The CLI is the Milestone 1 deliverable, so it gets an end-to-end test: write
a mix to disk, run the script, and check both spectrogram arrays come out.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from indian_mss.cli.preprocess import main
from indian_mss.data import Sanidha, load_audio

from conftest import SR, make_clicks, make_tone


def test_load_audio_downmixes_and_resamples(tmp_path):
    stereo = np.stack([make_tone(1.0), make_tone(1.0)], axis=1)
    sf.write(tmp_path / "x.wav", stereo, SR)
    audio = load_audio(tmp_path / "x.wav", SR // 2)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(SR // 2, abs=2)


def test_sanidha_synthesizes_missing_mixture(tmp_path):
    track = tmp_path / "track01"
    track.mkdir()
    sf.write(track / "Vocal.wav", make_tone(1.0), SR)
    sf.write(track / "mridangam_close.wav", make_clicks(1.0), SR)

    ds = Sanidha(tmp_path, SR)
    item = ds.load("track01")
    assert ds.track_ids() == ["track01"]
    assert set(item.stems) == {"vocal", "mridangam"}
    assert item.mixture_is_synthetic
    assert np.allclose(item.mixture, item.stems["vocal"] + item.stems["mridangam"], atol=1e-6)


def test_sanidha_rejects_unknown_layout(tmp_path):
    (tmp_path / "track01").mkdir()
    sf.write(tmp_path / "track01" / "channel_1.wav", make_tone(0.5), SR)
    with pytest.raises(ValueError, match="layout"):
        Sanidha(tmp_path, SR)


def _run_cli(tmp_path, *extra):
    sf.write(tmp_path / "mix.wav", make_tone(2.0) + make_clicks(2.0), SR)
    out = tmp_path / "out"
    main(["--input", str(tmp_path / "mix.wav"), "--outdir", str(out), "--sample-rate", str(SR), *extra])
    return out


def test_cli_writes_melodic_and_percussive_arrays(tmp_path):
    out = _run_cli(tmp_path, "--tonic-strategy", "none")
    melodic = np.load(out / "mix_melodic.npy")
    percussive = np.load(out / "mix_percussive.npy")
    assert melodic.ndim == 2 and melodic.shape == percussive.shape


def test_cli_writes_log_spectrograms_on_a_shared_scale(tmp_path):
    out = _run_cli(tmp_path, "--tonic-strategy", "none")
    melodic_log = np.load(out / "mix_melodic_log.npy")
    percussive_log = np.load(out / "mix_percussive_log.npy")
    assert melodic_log.shape == np.load(out / "mix_melodic.npy").shape
    for arr in (melodic_log, percussive_log):
        assert arr.min() >= 0.0 and arr.max() <= 1.0
    # Shared reference: the stems are not each stretched to touch 1.0.
    assert min(melodic_log.max(), percussive_log.max()) < 1.0


def test_cli_reads_config_and_flags_override_it(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("stft:\n  n_fft: 512\n  hop_length: 128\ntonic:\n  strategy: none\n")
    out = _run_cli(tmp_path, "--config", str(cfg))
    assert np.load(out / "mix_melodic.npy").shape[0] == 512 // 2 + 1

    out = _run_cli(tmp_path, "--config", str(cfg), "--n-fft", "1024", "--hop-length", "256")
    assert np.load(out / "mix_melodic.npy").shape[0] == 1024 // 2 + 1


def test_cli_records_the_tonic(tmp_path):
    out = _run_cli(tmp_path, "--tonic-strategy", "condition")
    record = json.loads((out / "mix_tonic.json").read_text())
    assert record["strategy"] == "condition"
    assert abs(1200 * np.log2(record["tonic_hz"] / 220.0)) < 25.0
