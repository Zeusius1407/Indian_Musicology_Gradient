#!/usr/bin/env python3
"""Milestone 2 CLI: fine-tune a separator on Sanidha.

Examples
--------
Fine-tune pretrained HTDemucs with everything from ``configs/default.yaml``::

    python scripts/train.py --outdir checkpoints/htdemucs

The from-scratch U-Net, when HTDemucs does not fit in memory::

    python scripts/train.py --backbone unet --outdir checkpoints/unet

Pick up an interrupted run where it stopped::

    python scripts/train.py --outdir checkpoints/htdemucs --resume

The run directory receives ``best.pt`` (selected by validation SDR -- the
deliverable), ``last.pt`` and ``history.json`` (plotted by
``notebooks/train.ipynb``).
"""

from __future__ import annotations

import argparse
import copy

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Return a copy of ``cfg`` with every flag that was given written into it."""
    cfg = copy.deepcopy(cfg)
    train = cfg.setdefault("train", {})
    for flag, key in (
        ("backbone", "backbone"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("samples_per_epoch", "samples_per_epoch"),
        ("num_workers", "num_workers"),
        ("lr", "lr"),
    ):
        value = getattr(args, flag)
        if value is not None:
            train[key] = value
    if args.sanidha_root is not None:
        cfg.setdefault("data", {})["sanidha_root"] = args.sanidha_root
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--sanidha-root", type=str, default=None, help="overrides data.sanidha_root")
    p.add_argument("--outdir", type=str, default=None, help="run directory (default: checkpoints/<backbone>)")
    p.add_argument("--backbone", choices=("htdemucs", "unet"), default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--no-pretrained", action="store_true", help="HTDemucs from random init (ablation only)")
    p.add_argument("--resume", action="store_true", help="continue from <outdir>/last.pt")
    p.add_argument("--no-progress", action="store_true", help="no per-batch progress bar")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)

    from indian_mss.train.loop import train

    backbone = cfg["train"].get("backbone", "htdemucs")
    outdir = args.outdir or f"checkpoints/{backbone}"
    history = train(
        cfg,
        cfg.get("data", {}).get("sanidha_root", "data/sanidha"),
        outdir,
        device=args.device,
        pretrained=not args.no_pretrained,
        resume=args.resume,
        progress=not args.no_progress,
    )
    best = history["best"]
    if best:
        print(f"\nbest: epoch {best['epoch']}, mean val SDR {best['val_sdr_mean']:+.2f} dB -> {outdir}/best.pt")


if __name__ == "__main__":
    main()
