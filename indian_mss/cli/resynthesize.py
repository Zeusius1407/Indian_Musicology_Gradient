#!/usr/bin/env python3
"""Milestone 3 CLI: mask ratios + mixture in, one ``.wav`` per instrument out.

Takes the ``*_masks.npy`` written by ``scripts/separate.py`` and the mix it
was computed from::

    python scripts/separate.py --checkpoint best.pt --input concert.wav --outdir out/ --no-audio
    python scripts/resynthesize.py --masks out/concert_masks.npy --input concert.wav --outdir out/stems

The method comes from ``resynthesis.method`` in ``configs/default.yaml``
(``mixture_phase``, ``griffin_lim`` or ``vocoder``); ``--method`` overrides
it. Stems are written as ``concert_<source>.wav``, with the method recorded
in ``concert_resynthesis.json``.

``scripts/compare_resynthesis.py`` scores all methods against ground truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config
from indian_mss.infer.resynth import METHODS, ResynthConfig, resynthesize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks", type=str, required=True, help="*_masks.npy from scripts/separate.py")
    p.add_argument("--input", type=str, required=True, help="the mixture the masks were computed from")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--method", choices=METHODS, default=None, help="overrides resynthesis.method")
    p.add_argument("--griffin-lim-iters", type=int, default=None)
    p.add_argument("--griffin-lim-init", choices=("random", "mixture"), default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    rcfg = ResynthConfig.from_config(
        cfg, method=args.method, griffin_lim_iters=args.griffin_lim_iters, griffin_lim_init=args.griffin_lim_init
    )

    import soundfile as sf

    from indian_mss.data.base import load_audio
    from indian_mss.dsp.stft import STFTConfig

    masks_path = Path(args.masks)
    meta_path = masks_path.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found: scripts/separate.py writes it next to the masks")
    meta = json.loads(meta_path.read_text())
    # The masks' own STFT settings, not the config's: resynthesizing on a
    # different grid would misalign every bin.
    stft_cfg = STFTConfig(**meta["stft"])
    masks = np.load(masks_path)
    audio = load_audio(args.input, stft_cfg.sample_rate)

    stems = resynthesize(masks, audio, stft_cfg, rcfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem
    for source, wav in zip(meta["sources"], stems):
        sf.write(outdir / f"{name}_{source}.wav", wav, stft_cfg.sample_rate)
    (outdir / f"{name}_resynthesis.json").write_text(
        json.dumps({"masks": str(masks_path), "input": args.input, "resynthesis": asdict(rcfg)}, indent=2)
    )
    print(f"wrote {len(stems)} stems ({rcfg.method}) to {outdir}")


if __name__ == "__main__":
    main()
