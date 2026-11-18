"""Milestone 2 training: fine-tune a separator on Sanidha, keep the best by val SDR.

Run directory layout (``outdir``)::

    best.pt        highest mean validation SDR so far -- the deliverable
    last.pt        end of the latest epoch, with optimizer state, for --resume
    history.json   every logged number; the notebook plots straight from it

Two rules from ``models/README.md`` are enforced here rather than left to
discipline: the checkpoint is selected by validation SDR (not loss, which
keeps falling after quality plateaus), and SDR is logged per stem (a mean can
hide a model that nails the vocal and fails the ghatam).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.sanidha import Sanidha
from ..dsp.stft import STFTConfig
from ..eval.sdr import evaluate_stems
from ..models.separator import build_separator, param_groups, save_checkpoint, separate
from .dataset import SegmentDataset, concert_of, split_tracks, track_refs
from .losses import SeparationLoss

__all__ = ["pick_device", "train", "validate"]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stft_config(cfg: dict) -> STFTConfig:
    s = cfg.get("stft", {})
    return STFTConfig(
        sample_rate=int(s.get("sample_rate", 44100)),
        n_fft=int(s.get("n_fft", 2048)),
        hop_length=int(s.get("hop_length", 512)),
        window=str(s.get("window", "hann")),
    )


def _load_val_items(dataset: Sanidha, val_ids: list[str], seconds: float, sources: list[str]) -> list[dict]:
    """Load each validation track once, cropped to its central ``seconds``.

    The centre rather than the start: Carnatic recordings often open with
    tuning or a solo alapana where the percussion is silent, which would make
    the percussive SDR undefined.
    """
    items = []
    for tid in val_ids:
        item = dataset.load(tid)
        n = len(item.mixture)
        keep = min(n, int(seconds * item.sample_rate)) if seconds else n
        start = (n - keep) // 2
        stems = {s: item.stems[s][start : start + keep] for s in sources if s in item.stems}
        items.append({"track_id": tid, "mixture": item.mixture[start : start + keep], "stems": stems})
    return items


def validate(
    model: torch.nn.Module,
    items: list[dict],
    stft_cfg: STFTConfig,
    chunk_seconds: float,
    loss_fn: SeparationLoss,
    device: torch.device,
) -> dict:
    """Mean SDR per stem over validation tracks, plus the validation loss.

    A stem whose reference is silent in a track is skipped for that track:
    its SDR is undefined, and scoring it would drag the mean towards -inf.
    """
    sources = list(model.sources)
    per_stem: dict[str, list[float]] = {s: [] for s in sources}
    losses = []
    for item in items:
        est = separate(model, item["mixture"], stft_cfg.sample_rate, chunk_seconds, device=device)
        mix_energy = float(np.sum(item["mixture"] ** 2)) + 1e-12
        refs = {
            s: item["stems"][s]
            for s in sources
            if s in item["stems"] and np.sum(item["stems"][s] ** 2) > 1e-6 * mix_energy
        }
        ests = {s: est[i] for i, s in enumerate(sources) if s in refs}
        for stem, scores in evaluate_stems(ests, refs).items():
            per_stem[stem].append(scores["sdr"])

        full_ref = np.stack([item["stems"].get(s, np.zeros_like(item["mixture"])) for s in sources])
        with torch.no_grad():
            loss, _ = loss_fn(torch.from_numpy(est)[None], torch.from_numpy(full_ref)[None])
        losses.append(float(loss))

    sdr = {s: float(np.mean(v)) for s, v in per_stem.items() if v}
    return {
        "val_sdr": sdr,
        "val_sdr_mean": float(np.mean(list(sdr.values()))) if sdr else float("nan"),
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
    }


def _write_history(path: Path, history: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2))
    tmp.replace(path)


def train(
    cfg: dict,
    sanidha_root: str | Path,
    outdir: str | Path,
    *,
    device: str = "auto",
    pretrained: bool = True,
    resume: bool = False,
    progress: bool = True,
) -> dict:
    """Train according to ``cfg`` (the parsed ``configs/default.yaml``); returns the history."""
    tcfg = cfg.get("train", {})
    sources = list(cfg.get("data", {}).get("stems", ["vocal", "violin", "mridangam", "ghatam"]))
    stft_cfg = stft_config(cfg)
    sr = stft_cfg.sample_rate
    seed = int(tcfg.get("seed", 0))
    torch.manual_seed(seed)
    dev = pick_device(device)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- data --------------------------------------------------------------
    corpus = Sanidha(sanidha_root, sr, stems=tuple(sources))
    train_ids, val_ids = split_tracks(
        corpus, float(tcfg.get("val_fraction", 0.15)), seed, tcfg.get("val_concerts")
    )
    if not val_ids:
        raise ValueError("need at least two Sanidha concerts: one to train on, one to validate on")
    held = sorted({concert_of(t) for t in val_ids})
    print(f"validation: {', '.join(held)} held out ({len(val_ids)} tracks; {len(train_ids)} train)")
    # A stem no held-out concert plays is never scored, so it cannot steer
    # checkpoint selection -- say so rather than let it pass silently.
    unvalidated = [s for s in sources if not any(corpus.stem_paths(t).get(s) for t in val_ids)]
    if unvalidated:
        print(f"warning: no validation track has {', '.join(unvalidated)}; best.pt ignores those stems")
    segment = float(tcfg.get("segment_seconds", 7.8))
    train_set = SegmentDataset(
        track_refs(corpus, train_ids),
        sources,
        sr,
        segment,
        samples_per_epoch=int(tcfg.get("samples_per_epoch", 2000)),
        augment=cfg.get("augment", {}),
        seed=seed,
    )
    batch_size = int(tcfg.get("batch_size", 4))
    loader_kw = dict(
        batch_size=batch_size,
        shuffle=False,  # every index is already a fresh random crop
        num_workers=int(tcfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=dev.type == "cuda",
    )
    val_items = _load_val_items(corpus, val_ids, float(tcfg.get("val_seconds", 60.0)), sources)

    # --- model -------------------------------------------------------------
    backbone = str(tcfg.get("backbone", "htdemucs"))
    model, arch = build_separator(backbone, sources, stft_cfg, tcfg, pretrained=pretrained)
    model.to(dev)
    loss_w = tcfg.get("loss", {})
    loss_fn = SeparationLoss(
        stft_cfg, float(loss_w.get("spectrogram_l1", 1.0)), float(loss_w.get("waveform_l1", 1.0))
    ).to(dev)
    val_loss_fn = SeparationLoss(
        stft_cfg, float(loss_w.get("spectrogram_l1", 1.0)), float(loss_w.get("waveform_l1", 1.0))
    )
    epochs = int(tcfg.get("epochs", 60))
    frozen_epochs = int(tcfg.get("frozen_encoder_epochs", 10)) if backbone == "htdemucs" else 0
    grad_clip = float(tcfg.get("grad_clip", 5.0))

    history: dict = {
        "config": cfg,
        "backbone": backbone,
        "sources": sources,
        "train_tracks": train_ids,
        "val_tracks": val_ids,
        "steps": [],
        "epochs": [],
        "best": None,
    }
    start_epoch, phase, opt_state = 1, None, None
    last_path, best_path = outdir / "last.pt", outdir / "best.pt"
    if resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        history = json.loads((outdir / "history.json").read_text())
        start_epoch = ckpt["epoch"] + 1
        opt_state = (ckpt["phase"], ckpt["optimizer"])
        print(f"resuming from epoch {ckpt['epoch']} ({last_path})")

    meta = {"backbone": backbone, "segment_seconds": segment, "train_tracks": train_ids, "val_tracks": val_ids}
    step = history["steps"][-1]["step"] if history["steps"] else 0
    optimizer = None

    for epoch in range(start_epoch, epochs + 1):
        new_phase = 1 if epoch <= frozen_epochs else 2
        if new_phase != phase:
            phase = new_phase
            groups = param_groups(model, phase, tcfg)
            optimizer = torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 0.0)))
            if opt_state and opt_state[0] == phase:
                optimizer.load_state_dict(opt_state[1])
            opt_state = None
            n_train = sum(p.numel() for g in groups for p in g["params"])
            label = f"phase {phase}: " if frozen_epochs else ""
            print(f"{label}training {n_train:,} parameters ({', '.join(g['name'] for g in groups)})")

        t0 = time.time()
        model.train()
        train_set.set_epoch(epoch)
        loader = DataLoader(train_set, **loader_kw)
        if progress:
            try:
                from tqdm.auto import tqdm

                loader = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
            except ImportError:
                pass

        sums = {"loss": 0.0, "spectrogram_l1": 0.0, "waveform_l1": 0.0}
        n_batches = 0
        for mix, ref in loader:
            mix, ref = mix.to(dev), ref.to(dev)
            est = model(mix)
            loss, parts = loss_fn(est, ref)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip
            )
            optimizer.step()
            step += 1
            n_batches += 1
            sums["loss"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] += v
            history["steps"].append({"step": step, "epoch": epoch, "loss": float(loss.detach()), **parts})

        val = validate(model, val_items, stft_cfg, segment, val_loss_fn, dev)
        record = {
            "epoch": epoch,
            "phase": phase,
            **{f"train_{k}" if k != "loss" else "train_loss": v / max(n_batches, 1) for k, v in sums.items()},
            **val,
            "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
            "seconds": time.time() - t0,
        }
        history["epochs"].append(record)

        best = history["best"]
        improved = math.isfinite(val["val_sdr_mean"]) and (best is None or val["val_sdr_mean"] > best["val_sdr_mean"])
        if improved:
            history["best"] = {"epoch": epoch, "val_sdr_mean": val["val_sdr_mean"], "val_sdr": val["val_sdr"]}
            save_checkpoint(best_path, model, arch, stft_cfg, epoch=epoch, val_sdr=val["val_sdr"], **meta)
        save_checkpoint(
            last_path, model, arch, stft_cfg, epoch=epoch, phase=phase, optimizer=optimizer.state_dict(), **meta
        )
        _write_history(outdir / "history.json", history)

        per_stem = "  ".join(f"{k} {v:+.2f}" for k, v in val["val_sdr"].items())
        print(
            f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  val loss {val['val_loss']:.4f}  "
            f"val SDR {val['val_sdr_mean']:+.2f} dB [{per_stem}]" + ("  *best*" if improved else "")
        )

    return history
