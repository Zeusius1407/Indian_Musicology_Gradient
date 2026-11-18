"""Train the timbre classifier on IMIDataset plus Sanidha's clean stems.

The proposal trains on IMIDataset, whose ten classes are bass, drums, flute,
guitar, piano, sarod, sitar, tabla, veena and violin. Three of the four
instruments the separator outputs -- voice, mridangam and ghatam -- are not
among them, so Sanidha's clean close-mic stems supply those classes, and
violin draws on both corpora. The label set is the union (13 classes). With
``data.imi_root`` absent or ``classifier.use_imi: false`` it falls back to
Sanidha alone (the four stems).

Splits never cut inside a recording:

* Sanidha **test** -- the separator's held-out concert(s)
  (``classifier.test_concerts``, default Concert03), so the Milestone 4
  comparison runs on audio neither network trained on. **val** -- one
  recording per remaining concert, for epoch selection only (Concert01 is the
  only training concert with a ghatam). **train** -- the rest.
* IMIDataset -- by block of ``imi_block`` consecutive clip numbers (clips are
  consecutive excerpts of longer recordings; see ``data/imi.py``), 15% of each
  class's blocks to val and 15% to test.

Each class gets the same window budget (``windows_per_class``), split evenly
between the corpora that have it and then over their recordings. Otherwise
the classifier learns the class prior -- and, worse, which corpus a window
came from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..dsp.features import MFCCConfig
from ..train.dataset import concert_of
from .data import WindowConfig, active_starts, mfcc_windows
from .model import TimbreCNN

__all__ = [
    "Unit",
    "collect_windows",
    "configs_from",
    "imi_units",
    "load_classifier",
    "predict_proba",
    "sanidha_units",
    "split_recordings",
    "train_classifier",
]


def configs_from(cfg: dict) -> tuple[MFCCConfig, WindowConfig]:
    c = cfg.get("classifier", {})
    fmax = c.get("fmax", MFCCConfig.fmax)
    mfcc_cfg = MFCCConfig(
        sample_rate=int(c.get("sample_rate", MFCCConfig.sample_rate)),
        n_mfcc=int(c.get("n_mfcc", MFCCConfig.n_mfcc)),
        n_fft=int(c.get("n_fft", MFCCConfig.n_fft)),
        hop_length=int(c.get("hop_length", MFCCConfig.hop_length)),
        n_mels=int(c.get("n_mels", MFCCConfig.n_mels)),
        fmax=float(fmax) if fmax is not None else None,
    )
    win_cfg = WindowConfig(
        window_seconds=float(c.get("window_seconds", 1.0)),
        hop_seconds=float(c.get("hop_seconds", 0.5)),
        silence_db=float(c.get("silence_db", -50.0)),
        silence_rel_db=float(c.get("silence_rel_db", -30.0)),
    )
    return mfcc_cfg, win_cfg


@dataclass
class Unit:
    """One recording: never split across train/val/test. ``load`` -> ``{class: audio}``."""

    uid: str
    source: str
    classes: tuple[str, ...]
    load: Callable[[], dict[str, np.ndarray]]


def sanidha_units(dataset, track_ids: list[str], classes: list[str]) -> list[Unit]:
    def loader(tid):
        return lambda: {c: a for c, a in dataset.load(tid).stems.items() if c in classes}

    return [
        Unit(tid, "sanidha", tuple(c for c in classes if dataset.stem_paths(tid).get(c)), loader(tid))
        for tid in track_ids
    ]


def imi_units(clips, sample_rate: int) -> list[Unit]:
    from ..data.base import load_audio

    def loader(clip):
        return lambda: {clip.cls: load_audio(clip.path, sample_rate)}

    return [Unit(str(clip.path), "imi", (clip.cls,), loader(clip)) for clip in clips]


def _recording(track_id: str) -> str:
    """Alternate takes are one recording: ``..._01-Title-(Take-1)`` -> ``..._01-Title``."""
    return re.sub(r"-\(Take-\d+\)$", "", track_id)


def split_recordings(
    track_ids: list[str], test_concerts: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Sanidha ``(train, val, test)`` track ids; takes of one recording stay together."""
    test = [t for t in track_ids if concert_of(t) in set(test_concerts)]
    rest = [t for t in track_ids if concert_of(t) not in set(test_concerts)]
    rng = np.random.default_rng(seed)
    val_recordings = set()
    for concert in sorted({concert_of(t) for t in rest}):
        recordings = sorted({_recording(t) for t in rest if concert_of(t) == concert})
        if len(recordings) >= 2:
            val_recordings.add(recordings[rng.integers(len(recordings))])
    val = [t for t in rest if _recording(t) in val_recordings]
    return [t for t in rest if t not in val], val, test


def collect_windows(
    units: list[Unit], classes: list[str], mfcc_cfg: MFCCConfig, win_cfg: WindowConfig, windows_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(X, y, unit_of_window)`` from ``units``; ``windows_per_class <= 0`` keeps all."""
    rng = np.random.default_rng(seed)
    holders: dict[tuple[str, str], int] = {}
    for u in units:
        for c in u.classes:
            holders[(c, u.source)] = holders.get((c, u.source), 0) + 1
    n_sources = {c: sum(1 for (cc, _) in holders if cc == c) for c in classes}
    win = int(win_cfg.window_seconds * mfcc_cfg.sample_rate)

    X, y, owner = [], [], []
    for k, u in enumerate(units, 1):
        for c, audio in u.load().items():
            if len(audio) < win:  # IMI violin and flute clips are often just under 1 s
                audio = np.pad(audio, (0, win - len(audio)))
            starts = active_starts(audio, mfcc_cfg.sample_rate, win_cfg)
            if windows_per_class > 0:
                quota = max(1, windows_per_class // n_sources[c] // holders[(c, u.source)])
                if len(starts) > quota:
                    starts = np.sort(rng.choice(starts, quota, replace=False))
            if not len(starts):
                continue
            patches = mfcc_windows(audio, starts, mfcc_cfg, win_cfg.window_seconds)
            X.append(patches.astype(np.float16))
            y += [classes.index(c)] * len(patches)
            owner += [u.uid] * len(patches)
        if k % 500 == 0 or k == len(units) or u.source == "sanidha":
            print(f"  {k}/{len(units)} recordings, {len(y)} windows", flush=True)
    if not X:
        return np.empty((0, 0, 0), dtype=np.float16), np.empty(0, dtype=np.int64), []
    return np.concatenate(X), np.asarray(y, dtype=np.int64), owner


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.head[-1].out_features))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall: the headline, because the classes are unbalanced."""
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(recalls)) if recalls else float("nan")


def train_classifier(dataset, cfg: dict, outdir: str | Path, device, imi=None, cache: str | Path | None = None) -> dict:
    """Build the window sets, train, and write ``classifier.pt`` + ``history.json``.

    ``dataset`` is a ``Sanidha``; ``imi`` an optional ``IMIDataset``.
    """
    c = cfg.get("classifier", {})
    stems = list(cfg.get("data", {}).get("stems", ["vocal", "violin", "mridangam", "ghatam"]))
    classes = stems + ([k for k in imi.classes() if k not in stems] if imi is not None else [])
    mfcc_cfg, win_cfg = configs_from(cfg)
    seed = int(c.get("seed", 0))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = split_recordings(
        sorted(dataset.track_ids()), list(c.get("test_concerts", ["Concert03"])), seed
    )
    if not train_ids or not val_ids:
        raise ValueError(
            "classifier needs at least one concert outside test_concerts with two or more recordings "
            f"(got {len(train_ids)} train / {len(val_ids)} val tracks)"
        )
    print(f"Sanidha split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test recordings")
    print(f"  val: {', '.join(val_ids)}")
    imi_split = ([], [], [])
    if imi is not None:
        imi_split = imi.split(int(c.get("imi_block", 100)), seed=seed)
        print(f"IMIDataset split: {' / '.join(str(len(s)) for s in imi_split)} train / val / test clips")
    print(f"classes ({len(classes)}): {', '.join(classes)}")

    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        z = np.load(cache)
        Xtr, ytr, Xva, yva = z["Xtr"], z["ytr"], z["Xva"], z["yva"]
        print(f"loaded cached windows from {cache}")
    else:
        budget = int(c.get("windows_per_class", 4000))
        print("extracting training windows")
        Xtr, ytr, _ = collect_windows(
            sanidha_units(dataset, train_ids, stems) + imi_units(imi_split[0], mfcc_cfg.sample_rate),
            classes, mfcc_cfg, win_cfg, budget, seed,
        )  # fmt: skip
        print("extracting validation windows")
        Xva, yva, _ = collect_windows(
            sanidha_units(dataset, val_ids, stems) + imi_units(imi_split[1], mfcc_cfg.sample_rate),
            classes, mfcc_cfg, win_cfg, budget // 4, seed + 1,
        )  # fmt: skip
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    counts = {cl: int(np.sum(ytr == i)) for i, cl in enumerate(classes)}
    print(f"train windows per class: {counts}; val windows: {len(yva)}")

    torch.manual_seed(seed)
    model = TimbreCNN(len(classes), Xtr.shape[1]).to(device)
    flat = torch.from_numpy(Xtr.astype(np.float32)).permute(1, 0, 2).reshape(Xtr.shape[1], -1)
    model.set_normalization(flat.mean(dim=1), flat.std(dim=1))
    del flat

    # Inverse-frequency weights on top of the budget: classes short of audio
    # (ghatam, IMI flute and violin) fall below it.
    freq = np.bincount(ytr, minlength=len(classes)).astype(np.float64)
    weights = torch.tensor(freq.sum() / (len(classes) * np.maximum(freq, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.get("lr", 1e-3)), weight_decay=1e-4)
    epochs = int(c.get("epochs", 30))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    batch = int(c.get("batch_size", 128))
    # A gain of g dB shifts every log-mel bin by g, which the orthonormal DCT
    # maps to g * sqrt(n_mels) on c0 and nothing elsewhere. Augmenting c0
    # directly is exact, and stops the network keying on recording level.
    gain_db = float(c.get("gain_aug_db", 6.0))
    c0_per_db = float(np.sqrt(mfcc_cfg.n_mels))

    rng = np.random.default_rng(seed)
    history = {
        "epochs": [],
        "classes": classes,
        "train_tracks": train_ids,
        "val_tracks": val_ids,
        "test_tracks": test_ids,
        "imi_counts": {k: len(s) for k, s in zip(("train", "val", "test"), imi_split)},
    }
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(ytr))
        total, n = 0.0, 0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            xb = torch.from_numpy(Xtr[idx].astype(np.float32)).to(device)
            if gain_db > 0:
                xb[:, 0] += torch.empty(len(idx), 1, device=device).uniform_(-gain_db, gain_db) * c0_per_db
            yb = torch.from_numpy(ytr[idx]).to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        sched.step()

        pred = predict_proba(model, Xva, device).argmax(axis=1)
        val_bal = balanced_accuracy(yva, pred, len(classes))
        per_class = {cl: float(np.mean(pred[yva == i] == i)) for i, cl in enumerate(classes) if np.any(yva == i)}
        is_best = val_bal > best
        history["epochs"].append({"epoch": epoch, "train_loss": total / n, "val_balanced_acc": val_bal, "val_acc": per_class})
        print(
            f"epoch {epoch:3d}  loss {total / n:.4f}  val bal.acc {val_bal:.3f} "
            + " ".join(f"{k} {v:.2f}" for k, v in per_class.items())
            + ("  *best*" if is_best else ""),
            flush=True,
        )
        if is_best:
            best = val_bal
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "n_coeffs": int(Xtr.shape[1]),
                    "mfcc": asdict(mfcc_cfg),
                    "window": asdict(win_cfg),
                    "epoch": epoch,
                    "val_balanced_acc": val_bal,
                    "train_tracks": train_ids,
                    "val_tracks": val_ids,
                    "test_tracks": test_ids,
                    "imi_test_clips": [[clip.cls, str(clip.path)] for clip in imi_split[2]],
                },
                outdir / "classifier.pt",
            )
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def load_classifier(path: str | Path, device="cpu") -> tuple[TimbreCNN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TimbreCNN(len(payload["classes"]), payload["n_coeffs"])
    model.load_state_dict(payload["state_dict"])
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta["mfcc_config"] = MFCCConfig(**payload["mfcc"])
    meta["window_config"] = WindowConfig(**payload["window"])
    return model.to(device).eval(), meta
