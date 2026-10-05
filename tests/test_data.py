"""Tests for the data layer and the Milestone 1 CLI, on synthetic files.

The CLI is the Milestone 1 deliverable, so it gets an end-to-end test: write
a mix to disk, run the script, and check both spectrogram arrays come out.
"""

from __future__ import annotations

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


def test_cli_writes_melodic_and_percussive_arrays(tmp_path):
    sf.write(tmp_path / "mix.wav", make_tone(2.0) + make_clicks(2.0), SR)
    main(["--input", str(tmp_path / "mix.wav"), "--outdir", str(tmp_path / "out"), "--sample-rate", str(SR)])
    melodic = np.load(tmp_path / "out" / "mix_melodic.npy")
    percussive = np.load(tmp_path / "out" / "mix_percussive.npy")
    assert melodic.ndim == 2 and melodic.shape == percussive.shape
