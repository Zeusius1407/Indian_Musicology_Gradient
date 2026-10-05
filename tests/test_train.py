"""Tests for Milestone 2 training, end to end on a synthetic Sanidha corpus.

Everything runs on CPU in seconds: 8 kHz audio, 1 s segments, a few-thousand-
parameter model. The point is not separation quality but that every piece --
augmentation, split, both backbones, both training phases, checkpointing,
resume, and the separate CLI -- actually runs and produces what it promises.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from conftest import make_sanidha

SR = 8000
SOURCES = ["vocal", "violin", "mridangam", "ghatam"]


torch = pytest.importorskip("torch")

from indian_mss.data.sanidha import Sanidha  # noqa: E402
from indian_mss.train.dataset import SegmentDataset, split_tracks, track_refs  # noqa: E402
from indian_mss.train.loop import train  # noqa: E402

BASE_CFG = {
    "stft": {"sample_rate": SR, "n_fft": 256, "hop_length": 64, "window": "hann"},
    "data": {"stems": SOURCES},
    "augment": {
        "random_gain_db": [-3.0, 3.0],
        "shuffle_stems_across_tracks": True,
        "shuffle_prob": 0.5,
        "simulate_bleed": {"enabled": True, "prob": 0.5, "neighbour_gain_db": [-30.0, -12.0], "rir_ms": 40.0},
    },
    "train": {
        "backbone": "unet",
        "epochs": 2,
        "frozen_encoder_epochs": 1,
        "segment_seconds": 1.0,
        "samples_per_epoch": 8,
        "batch_size": 4,
        "num_workers": 0,
        "val_fraction": 0.34,
        "val_seconds": 2.0,
        "lr": 1e-3,
        "unet": {"channels": 4, "depth": 2},
        "htdemucs": {
            "kwargs": dict(segment=1, channels=8, depth=2, nfft=512, t_layers=1, t_heads=2,
                           t_hidden_scale=1.0, dconv_comp=2)
        },
    },
}


@pytest.fixture
def corpus(tmp_path):
    return make_sanidha(tmp_path / "sanidha", n_tracks=3, duration=3.0, sample_rate=SR)


def test_split_is_by_track_and_deterministic(corpus):
    ds = Sanidha(corpus, SR)
    tr, va = split_tracks(ds, 0.34, seed=0)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 3 and len(va) == 1
    assert split_tracks(ds, 0.34, seed=0) == (tr, va)


def test_segment_dataset_mixture_is_sum_of_stems(corpus):
    ds = Sanidha(corpus, SR)
    data = SegmentDataset(track_refs(ds, ds.track_ids()), SOURCES, SR, 1.0, 4, BASE_CFG["augment"])
    mix, stems = data[0]
    assert stems.shape == (len(SOURCES), SR) and mix.shape == (SR,)
    assert torch.allclose(stems.sum(0), mix, atol=1e-5)
    data.set_epoch(1)
    assert not torch.equal(data[0][0], mix)  # fresh crops every epoch


def test_train_unet_end_to_end_and_resume(corpus, tmp_path):
    out = tmp_path / "run"
    history = train(BASE_CFG, corpus, out, device="cpu", progress=False)
    assert (out / "best.pt").is_file() and (out / "last.pt").is_file()
    saved = json.loads((out / "history.json").read_text())
    assert [e["epoch"] for e in saved["epochs"]] == [1, 2]
    for e in saved["epochs"]:
        assert np.isfinite(e["train_loss"]) and np.isfinite(e["val_loss"])
        assert set(e["val_sdr"]) <= set(SOURCES) and e["val_sdr"]
    assert history["best"]["epoch"] in (1, 2)
    assert set(saved["train_tracks"]).isdisjoint(saved["val_tracks"])

    cfg = copy.deepcopy(BASE_CFG)
    cfg["train"]["epochs"] = 3
    resumed = train(cfg, corpus, out, device="cpu", resume=True, progress=False)
    assert [e["epoch"] for e in resumed["epochs"]] == [1, 2, 3]


def test_train_htdemucs_runs_both_phases(corpus, tmp_path):
    pytest.importorskip("demucs")
    cfg = copy.deepcopy(BASE_CFG)
    cfg["train"]["backbone"] = "htdemucs"
    cfg["train"]["unfreeze"] = ["decoder", "tdecoder"]
    history = train(cfg, corpus, tmp_path / "run", device="cpu", pretrained=False, progress=False)
    assert [e["phase"] for e in history["epochs"]] == [1, 2]
    assert set(history["epochs"][1]["lr"]) == {"head", "body"}


def test_training_reduces_the_loss(corpus, tmp_path):
    """The model must actually learn on data it can obviously separate."""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["augment"] = {}
    cfg["train"].update(epochs=4, samples_per_epoch=16, lr=3e-3)
    history = train(cfg, corpus, tmp_path / "run", device="cpu", progress=False)
    losses = [e["train_loss"] for e in history["epochs"]]
    assert losses[-1] < losses[0], losses


def test_separate_cli_writes_masks_on_the_milestone1_grid(corpus, tmp_path):
    import soundfile as sf

    from indian_mss.cli.separate import main
    from indian_mss.dsp.stft import STFTConfig, stft

    train(BASE_CFG, corpus, tmp_path / "run", device="cpu", progress=False)
    mix = sum(sf.read(p)[0] for p in sorted((corpus / "track00").glob("*.wav")))
    sf.write(tmp_path / "mix.wav", mix, SR)

    main(["--checkpoint", str(tmp_path / "run" / "best.pt"), "--input", str(tmp_path / "mix.wav"),
          "--outdir", str(tmp_path / "sep"), "--device", "cpu"])
    masks = np.load(tmp_path / "sep" / "mix_masks.npy")
    info = json.loads((tmp_path / "sep" / "mix_masks.json").read_text())
    grid = stft(mix, STFTConfig(sample_rate=SR, n_fft=256, hop_length=64))
    assert info["sources"] == SOURCES
    assert masks.shape == (len(SOURCES), *grid.shape)
    assert np.allclose(masks.sum(0), 1.0, atol=1e-4)
    for s in SOURCES:
        assert (tmp_path / "sep" / f"mix_{s}.wav").is_file()
