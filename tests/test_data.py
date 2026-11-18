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


def test_saraga_indexes_concert_piece_nesting(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=2, sample_rate=SR)
    ds = Saraga(tmp_path, SR)
    assert len(ds) == 4
    # Piece names repeat across concerts, so the concert must stay in the id.
    assert ds.track_ids() == [
        "Concert 0 by Some Artist/Athirum Kazhal",
        "Concert 0 by Some Artist/Kanukontini",
        "Concert 1 by Some Artist/Athirum Kazhal",
        "Concert 1 by Some Artist/Kanukontini",
    ]


def test_saraga_sums_multi_mic_stems_and_skips_appledouble(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=1, sample_rate=SR)
    ds = Saraga(tmp_path, SR)
    paths = ds.stem_paths("Concert 0 by Some Artist/Kanukontini")
    assert len(paths["mridangam"]) == 2  # left + right
    assert len(paths["vocal"]) == 2  # main + second vocalist
    assert len(paths["violin"]) == 1
    assert not any(p.name.startswith("._") for ps in paths.values() for p in ps)


def test_saraga_ghatam_only_on_a_subset(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=1, sample_rate=SR)
    ds = Saraga(tmp_path, SR)
    with_ghatam = ds.load("Concert 0 by Some Artist/Kanukontini")
    without = ds.load("Concert 0 by Some Artist/Athirum Kazhal")
    assert set(with_ghatam.stems) == {"vocal", "violin", "mridangam", "ghatam"}
    assert set(without.stems) == {"vocal", "violin", "mridangam"}
    # The gap is recorded, so a four-stem mean cannot silently average over
    # tracks that never had a ghatam.
    assert without.metadata["stems_missing"] == ["ghatam"]


def test_saraga_default_mixture_is_the_sum_of_the_mics(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=1, sample_rate=SR)
    item = Saraga(tmp_path, SR).load("Concert 0 by Some Artist/Kanukontini")
    assert item.mixture_is_synthetic  # linear remix of real bled mics
    assert item.metadata["mixture_mode"] == "stems"
    assert np.allclose(item.mixture, np.sum(list(item.stems.values()), axis=0), atol=1e-5)


def test_saraga_mixdown_mode_uses_the_released_master(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=1, sample_rate=SR)
    item = Saraga(tmp_path, SR, mixture="mixdown").load("Concert 0 by Some Artist/Kanukontini")
    assert not item.mixture_is_synthetic
    assert item.metadata["mixture_mode"] == "mixdown"
    # The master is engineered, not the sum of the mics -- the whole reason the
    # two modes are not interchangeable.
    assert not np.allclose(item.mixture, np.sum(list(item.stems.values()), axis=0), atol=1e-3)


def test_saraga_names_the_hindustani_mistake(tmp_path):
    """A mixture-only release must say so, not just "no tracks found"."""
    from indian_mss.data import Saraga

    piece = tmp_path / "Raag Shree by Someone" / "Raag Shree"
    piece.mkdir(parents=True)
    sf.write(piece / "Raag Shree.wav.wav", make_tone(1.0), SR)
    with pytest.raises(ValueError, match="mixture-only"):
        Saraga(tmp_path, SR)


def test_saraga_rejects_unknown_stem_and_mixture_mode(tmp_path):
    from conftest import make_saraga

    from indian_mss.data import Saraga

    make_saraga(tmp_path, n_concerts=1, sample_rate=SR)
    with pytest.raises(ValueError, match="mixture must be"):
        Saraga(tmp_path, SR, mixture="mixdown_or_something")
    with pytest.raises(ValueError, match="tabla"):
        Saraga(tmp_path, SR, stems=("vocal", "tabla"))
