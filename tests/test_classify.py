"""Tests for the Milestone 4 timbre classifier, on a synthetic Sanidha corpus.

The four synthetic "instruments" (two tones, two click trains) are trivially
distinct, so a classifier that trains at all should beat chance on held-out
recordings -- and the mixture should score worse than the clean stems.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_sanidha

torch = pytest.importorskip("torch")

from indian_mss.classify.data import WindowConfig, active_starts, mfcc_windows  # noqa: E402
from indian_mss.classify.evaluate import CONDITIONS, classify_track, summarize  # noqa: E402
from indian_mss.classify.train import load_classifier, split_recordings, train_classifier  # noqa: E402
from indian_mss.data.sanidha import Sanidha  # noqa: E402
from indian_mss.dsp.features import MFCCConfig  # noqa: E402

SR = 8000
CLASSES = ["vocal", "violin", "mridangam", "ghatam"]


def test_active_starts_skips_silence_and_quiet_bleed():
    cfg = WindowConfig(window_seconds=0.5, hop_seconds=0.25)
    loud = np.sin(np.linspace(0, 2000, SR * 2))
    audio = np.concatenate([loud, 1e-3 * loud, np.zeros(SR), loud])  # -60 dB "bleed", then silence
    starts = active_starts(audio, SR, cfg)
    assert len(starts) > 0
    quiet = (starts >= 2 * SR) & (starts + SR // 2 <= 5 * SR)
    assert not np.any(quiet)


def test_mfcc_windows_shape():
    cfg = MFCCConfig(sample_rate=SR, n_mfcc=13, n_fft=256, hop_length=64, n_mels=40)
    audio = np.random.default_rng(0).standard_normal(SR * 3).astype(np.float32)
    patches = mfcc_windows(audio, np.array([0, SR, 2 * SR]), cfg, 1.0)
    assert patches.shape == (3, 39, SR // 64)


def test_split_keeps_takes_together_and_holds_out_test():
    ids = ["C1_01-A", "C1_01-A-(Take-1)", "C1_02-B", "C1_03-C", "C2_01-D", "C2_02-E", "C3_01-F"]
    train, val, test = split_recordings(ids, ["C3"], seed=0)
    assert test == ["C3_01-F"]
    assert sorted(train + val + test) == sorted(ids)
    assert ("C1_01-A" in val) == ("C1_01-A-(Take-1)" in val)
    assert {t.split("_")[0] for t in val} == {"C1", "C2"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = make_sanidha(tmp_path_factory.mktemp("sanidha"), n_tracks=6, duration=6.0, sample_rate=SR)
    # Two training concerts of two recordings each, plus a test concert.
    for i, path in enumerate(sorted(root.iterdir())):
        path.rename(root / f"C{i // 2 + 1}_{i % 2 + 1:02d}-piece{i}")
    return Sanidha(root, SR, stems=tuple(CLASSES))


def _cfg():
    return {
        "data": {"stems": CLASSES},
        "classifier": {
            "sample_rate": SR, "n_mfcc": 13, "n_fft": 256, "hop_length": 64, "n_mels": 40, "window_seconds": 0.5, "hop_seconds": 0.25,
            "windows_per_class": 200, "epochs": 4, "batch_size": 32, "test_concerts": ["C3"],
        },
    }  # fmt: skip


def test_train_and_evaluate_end_to_end(corpus, tmp_path):
    history = train_classifier(corpus, _cfg(), tmp_path, "cpu")
    assert len(history["epochs"]) == 4
    assert set(history["test_tracks"]).isdisjoint(history["train_tracks"] + history["val_tracks"])

    model, meta = load_classifier(tmp_path / "classifier.pt")
    assert meta["classes"] == CLASSES

    item = corpus.load(history["test_tracks"][0])
    probs = classify_track(model, meta, item.stems, item.stems, item.mixture, "cpu")
    summary = summarize({c: {k: [v] for k, v in probs[c].items()} for c in CONDITIONS}, CLASSES)
    # Clean stems passed as "separated" must score exactly like the references.
    assert summary["separated"]["balanced_accuracy"] == summary["reference"]["balanced_accuracy"]
    assert summary["reference"]["balanced_accuracy"] > summary["mixture"]["balanced_accuracy"]
    conf = np.asarray(summary["reference"]["confusion"])
    assert conf.shape == (4, 4) and conf.sum() == sum(len(p) for p in probs["reference"].values())
    json.dumps(summary)  # report-serializable


def test_report_renders(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from indian_mss.classify.report import write_report

    rng = np.random.default_rng(0)
    labels = CLASSES + ["tabla", "flute"]
    probs = {c: {k: [rng.dirichlet(np.ones(6), size=10)] for k in CLASSES} for c in CONDITIONS}
    imi = {"imi_test": {"tabla": [rng.dirichlet(np.ones(6), size=5)], "flute": [rng.dirichlet(np.ones(6), size=5)]}}
    result = {
        "classes": CLASSES,
        "labels": labels,
        "summary": summarize(probs, labels, CLASSES),
        "imi_test": summarize(imi, labels, ["flute", "tabla"], ("imi_test",))["imi_test"],
        "test_concerts": ["C3"],
        "resynthesis_method": "mixture_phase",
    }
    write_report(result, tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 0 and (tmp_path / "r.png").is_file()
    assert (tmp_path / "r_imi.png").is_file()


def _make_imi(root, n_per_class: int = 12, duration: float = 1.5):
    """Tiny IMIDataset layout, including the release's duplicates."""
    import shutil

    import soundfile as sf

    from conftest import make_clicks, make_tone

    rng = np.random.default_rng(1)
    signals = {
        "Bass": lambda i: make_tone(duration, 60.0 + i, SR),
        "drum": lambda i: make_clicks(duration, 3.0, SR) * (1 + 0.01 * i),
        "flute": lambda i: make_tone(duration, 900.0 + 5 * i, SR) + 0.01 * rng.standard_normal(int(duration * SR)),
    }
    for part in ("Imid-01", "Imid-02"):
        (root / part / "dataset").mkdir(parents=True)
    for cls, make in signals.items():
        for i in range(1, n_per_class + 1):
            part = "Imid-01" if i % 2 else "Imid-02"
            sf.write(root / part / "dataset" / f"{cls}_{i}song.wav", make(i).astype(np.float32), SR)
    # Imid-06 is a copy of Imid-02; "(1)" copies; one clip under a second number.
    shutil.copytree(root / "Imid-02", root / "Imid-06")
    shutil.copy(root / "Imid-01" / "dataset" / "Bass_1song.wav", root / "Imid-01" / "dataset" / "Bass_1song(1).wav")
    shutil.copy(root / "Imid-01" / "dataset" / "flute_3song.wav", root / "Imid-02" / "dataset" / "flute_99song.wav")
    return root


def test_imi_loader_dedupes_and_splits_by_block(tmp_path):
    from indian_mss.data.imi import IMIDataset

    imi = IMIDataset(_make_imi(tmp_path / "imi"))
    assert imi.classes() == ["bass", "drums", "flute"]
    # 12 per class; every copy dropped, including flute_99 (same audio as flute_3).
    assert len(imi.clips) == 36
    train, val, test = imi.split(block=4, seed=0)
    assert len(train) + len(val) + len(test) == 36
    for cls in imi.classes():
        groups = [{c.group(4) for c in s if c.cls == cls} for s in (train, val, test)]
        assert all(groups) and not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_train_with_imi_adds_its_classes(corpus, tmp_path):
    from indian_mss.classify.evaluate import classify_clips
    from indian_mss.data.imi import IMIDataset

    imi = IMIDataset(_make_imi(tmp_path / "imi"))
    cfg = _cfg()
    cfg["classifier"]["imi_block"] = 4
    history = train_classifier(corpus, cfg, tmp_path / "run", "cpu", imi=imi)
    assert history["classes"] == CLASSES + ["bass", "drums", "flute"]
    model, meta = load_classifier(tmp_path / "run" / "classifier.pt")
    assert meta["imi_test_clips"] and all(len(c) == 2 for c in meta["imi_test_clips"])
    probs = classify_clips(model, meta, meta["imi_test_clips"], "cpu")
    summary = summarize({"imi_test": probs}, meta["classes"], sorted(probs), ("imi_test",))["imi_test"]
    assert np.asarray(summary["confusion"]).shape == (len(probs), 7)
