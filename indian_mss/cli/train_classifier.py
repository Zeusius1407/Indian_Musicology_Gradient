#!/usr/bin/env python3
"""Milestone 4 CLI: train the MFCC + 2-D CNN timbre classifier.

Example::

    python scripts/train_classifier.py --sanidha-root data/sanidha_flat --imi-root data/IMIDataset \
        --outdir checkpoints/classifier

Classes are IMIDataset's ten plus the Sanidha stems it lacks (vocal,
mridangam, ghatam); see ``indian_mss/classify/train.py``. ``--no-imi``
trains on Sanidha's four stems alone.

Writes ``classifier.pt`` (best epoch by validation balanced accuracy) and
``history.json``. Hyperparameters come from the ``classifier:`` block of
``configs/default.yaml``. ``--cache`` stores the extracted MFCC windows, so
a re-run with different training settings skips decoding the corpus.
"""

from __future__ import annotations

import argparse

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--sanidha-root", type=str, default=None, help="overrides data.sanidha_root")
    p.add_argument("--imi-root", type=str, default=None, help="overrides data.imi_root")
    p.add_argument("--no-imi", action="store_true", help="train on Sanidha alone")
    p.add_argument("--outdir", type=str, default="checkpoints/classifier")
    p.add_argument("--cache", type=str, default=None, help="npz of extracted windows, reused if present")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("classifier", {})["epochs"] = args.epochs

    from indian_mss.classify.train import configs_from, train_classifier
    from indian_mss.data.sanidha import Sanidha
    from indian_mss.train.loop import pick_device

    mfcc_cfg, _ = configs_from(cfg)
    root = args.sanidha_root or cfg.get("data", {}).get("sanidha_root", "data/sanidha")
    stems = tuple(cfg.get("data", {}).get("stems", ["vocal", "violin", "mridangam", "ghatam"]))
    dataset = Sanidha(root, mfcc_cfg.sample_rate, stems=stems)
    imi = None
    if not args.no_imi and cfg.get("classifier", {}).get("use_imi", True):
        from indian_mss.data.imi import IMIDataset

        imi = IMIDataset(args.imi_root or cfg.get("data", {}).get("imi_root", "data/IMIDataset"))
    train_classifier(dataset, cfg, args.outdir, pick_device(args.device), imi=imi, cache=args.cache)
    print(f"wrote {args.outdir}/classifier.pt")


if __name__ == "__main__":
    main()
