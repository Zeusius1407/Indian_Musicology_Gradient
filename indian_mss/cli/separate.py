#!/usr/bin/env python3
"""Milestone 2 CLI: run a trained checkpoint on a mix, write masks and stems.

Example::

    python scripts/separate.py --checkpoint checkpoints/htdemucs/best.pt --input concert.wav --outdir out/

Writes, for ``concert.wav``:

``concert_masks.npy``
    ``(S, F, frames)`` mask ratios on the pipeline STFT grid (the same grid as
    the Milestone 1 spectrograms), summing to one over sources at every bin.
``concert_masks.json``
    Source order, STFT settings and checkpoint provenance for the array.
``concert_<source>.wav``
    The model's separated stems (skip with ``--no-audio``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="polyphonic audio file")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--no-audio", action="store_true", help="write masks only")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from indian_mss.data.base import load_audio
    from indian_mss.models.separator import load_separator, predict_masks
    from indian_mss.train.loop import pick_device

    device = pick_device(args.device)
    model, meta = load_separator(args.checkpoint, device)
    stft_cfg = meta["stft_config"]
    audio = load_audio(args.input, stft_cfg.sample_rate)

    masks, estimates = predict_masks(model, audio, stft_cfg, meta["segment_seconds"], device=device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    np.save(outdir / f"{stem}_masks.npy", masks)
    (outdir / f"{stem}_masks.json").write_text(
        json.dumps(
            {
                "sources": meta["sources"],
                "shape": list(masks.shape),
                "stft": asdict(stft_cfg),
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": meta.get("epoch"),
                "checkpoint_val_sdr": meta.get("val_sdr"),
                "backbone": meta.get("backbone"),
            },
            indent=2,
        )
    )
    print(f"wrote masks {masks.shape} for sources {meta['sources']} to {outdir}")

    if not args.no_audio:
        import soundfile as sf

        for name, wav in zip(meta["sources"], estimates):
            sf.write(outdir / f"{stem}_{name}.wav", wav, stft_cfg.sample_rate)
        print(f"wrote {len(estimates)} stems to {outdir}")


if __name__ == "__main__":
    main()
