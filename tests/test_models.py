"""Tests for the Milestone 2 models. Skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from indian_mss.dsp.stft import STFTConfig, stft  # noqa: E402
from indian_mss.models.separator import (  # noqa: E402
    build_separator,
    load_separator,
    param_groups,
    predict_masks,
    save_checkpoint,
    separate,
)
from indian_mss.models.spec import TorchSTFT  # noqa: E402

from conftest import make_clicks, make_tone  # noqa: E402

SR = 8000
CFG = STFTConfig(sample_rate=SR, n_fft=256, hop_length=64)
SOURCES = ["vocal", "violin", "mridangam", "ghatam"]
TINY_UNET = {"unet": {"channels": 4, "depth": 2}}
TINY_DEMUCS = {
    "htdemucs": {
        "kwargs": dict(
            segment=1, channels=8, depth=2, nfft=512, t_layers=1, t_heads=2, t_hidden_scale=1.0, dconv_comp=2
        )
    }
}


def test_torch_stft_matches_numpy_stft():
    """The promise in models/spec.py: one STFT, bin for bin, in both worlds."""
    audio = make_tone(0.5, sample_rate=SR) + make_clicks(0.5, sample_rate=SR)
    audio = audio[: (len(audio) // CFG.hop_length) * CFG.hop_length]  # whole frames
    ref = stft(audio, CFG)
    got = TorchSTFT(CFG)(torch.from_numpy(audio)).numpy()
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-6)


def test_torch_stft_roundtrip():
    t = TorchSTFT(CFG)
    x = torch.from_numpy(make_tone(0.5, sample_rate=SR))
    assert torch.allclose(t.inverse(t(x), len(x)), x, atol=1e-6)


def test_unet_masks_sum_to_one_and_stems_sum_to_mix():
    model, _ = build_separator("unet", SOURCES, CFG, TINY_UNET)
    model.eval()
    mix = torch.from_numpy(make_tone(1.0, sample_rate=SR) + make_clicks(1.0, sample_rate=SR)).float()[None]
    with torch.no_grad():
        masks = model.masks(model.stft(mix).abs())
        est = model(mix)
    assert masks.shape[1] == len(SOURCES)
    assert torch.allclose(masks.sum(1), torch.ones_like(masks[:, 0]), atol=1e-5)
    # Softmax masks + mixture phase: the stems add back up to the mixture.
    assert torch.allclose(est.sum(1), mix, atol=1e-4)


def test_retargeted_demucs_starts_from_the_western_sources():
    """Each Indian stem must initially reproduce its Western donor exactly."""
    pytest.importorskip("demucs")
    from demucs.htdemucs import HTDemucs

    from indian_mss.models.demucs_backbone import WESTERN_INIT, retarget_sources

    torch.manual_seed(0)
    western = ["drums", "bass", "other", "vocals"]
    kw = dict(samplerate=SR, **TINY_DEMUCS["htdemucs"]["kwargs"])
    original = HTDemucs(western, **kw).eval()
    retargeted = HTDemucs(western, **kw)
    retargeted.load_state_dict(original.state_dict())
    retarget_sources(retargeted, SOURCES).eval()

    x = torch.randn(1, 2, SR)
    with torch.no_grad():
        before, after = original(x), retargeted(x)
    assert after.shape[1] == len(SOURCES)
    for j, s in enumerate(SOURCES):
        assert torch.allclose(after[:, j], before[:, western.index(WESTERN_INIT[s])], atol=1e-5), s


def test_demucs_phases_freeze_the_encoder():
    pytest.importorskip("demucs")
    model, _ = build_separator("htdemucs", SOURCES, CFG, TINY_DEMUCS, pretrained=False)
    cfg = {"lr": 1e-3, "encoder_lr": 1e-4, "unfreeze": ["decoder", "tdecoder"]}

    groups = param_groups(model, 1, cfg)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert [g["name"] for g in groups] == ["head"]
    assert trainable == {id(p) for p in model.head_parameters()}

    groups = param_groups(model, 2, cfg)
    assert {g["name"]: g["lr"] for g in groups} == {"head": 1e-3, "body": 1e-4}
    assert not any(p.requires_grad for p in model.model.encoder.parameters())
    assert all(p.requires_grad for p in model.model.decoder.parameters())


def test_chunked_separation_matches_single_pass_for_unet():
    """Overlap-add over chunks must not change a stateless model's output much."""
    model, _ = build_separator("unet", SOURCES, CFG, TINY_UNET)
    audio = (make_tone(3.0, sample_rate=SR) + make_clicks(3.0, sample_rate=SR)).astype(np.float32)
    full = separate(model, audio, SR, chunk_seconds=10.0)
    chunked = separate(model, audio, SR, chunk_seconds=1.0)
    assert chunked.shape == full.shape == (len(SOURCES), len(audio))
    # Stems still sum to the mixture after cross-fading.
    assert np.allclose(chunked.sum(0), audio, atol=1e-3)


@pytest.mark.parametrize("backbone", ["unet", "htdemucs"])
def test_checkpoint_roundtrip_and_masks(tmp_path, backbone):
    if backbone == "htdemucs":
        pytest.importorskip("demucs")
    model_cfg = TINY_UNET if backbone == "unet" else TINY_DEMUCS
    model, arch = build_separator(backbone, SOURCES, CFG, model_cfg, pretrained=False)
    save_checkpoint(tmp_path / "m.pt", model, arch, CFG, epoch=3, segment_seconds=1.0)
    loaded, meta = load_separator(tmp_path / "m.pt")
    assert meta["sources"] == SOURCES and meta["epoch"] == 3

    audio = (make_tone(1.0, sample_rate=SR) + make_clicks(1.0, sample_rate=SR)).astype(np.float32)
    model.eval()
    a, b = separate(model, audio, SR, 1.0), separate(loaded, audio, SR, 1.0)
    assert np.allclose(a, b, atol=1e-5)

    masks, est = predict_masks(loaded, audio, CFG, 1.0)
    assert masks.shape == (len(SOURCES), *stft(audio, CFG).shape)  # Milestone 1 grid
    assert masks.min() >= 0.0 and masks.max() <= 1.0
    assert np.allclose(masks.sum(0), 1.0, atol=1e-5)
