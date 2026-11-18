#!/usr/bin/env python3
"""Milestone 2 CLI: score a trained separator on Saraga, the live-bleed test set.

Examples
--------
Score the fine-tuned checkpoint on every Saraga Carnatic multitrack recording::

    python scripts/evaluate.py --checkpoint checkpoints/htdemucs_full/best.pt

Each track is separated twice, from the two mixture definitions described in
``indian_mss/data/saraga.py``: ``stems`` (the linear sum of the bled close
mics -- the headline number) and ``mixdown`` (the released stereo master --
the realism check). ``--mixture stems`` skips the second.

Scoring matches training validation: the central ``--seconds`` of each track
(default ``train.val_seconds``), and a stem whose reference is silent there is
skipped rather than scored. Results go to ``<outdir>/saraga_eval.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config
from indian_mss.eval.sdr import evaluate_stems

METRICS = ("sdr", "sir", "sar", "si_sdr")
# Ghatam ships on only ~46 Saraga recordings, so the four-stem mean covers a
# much smaller set than the three-stem one. Report both, labelled.
CORE_STEMS = ("vocal", "violin", "mridangam")


def _crop_centre(item, seconds: float) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    n = len(item.mixture)
    keep = min(n, int(seconds * item.sample_rate)) if seconds else n
    start = (n - keep) // 2
    return item.mixture[start : start + keep], {k: v[start : start + keep] for k, v in item.stems.items()}, start


def _audible(stems: dict[str, np.ndarray], mixture: np.ndarray) -> dict[str, np.ndarray]:
    """Drop silent references -- same threshold as ``train.loop.validate``."""
    mix_energy = float(np.sum(mixture**2)) + 1e-12
    return {s: ref for s, ref in stems.items() if np.sum(ref**2) > 1e-6 * mix_energy}


def summarize(rows: list[dict], mode: str) -> dict:
    scored = [r for r in rows if mode in r["scores"]]
    per_stem = {}
    for stem in sorted({s for r in scored for s in r["scores"][mode]}):
        vals = [r["scores"][mode][stem] for r in scored if stem in r["scores"][mode]]
        per_stem[stem] = {"n_tracks": len(vals), **{m: float(np.mean([v[m] for v in vals])) for m in METRICS}}

    def track_mean(stems: tuple[str, ...]) -> dict:
        # Per-track mean over ``stems`` first, then over tracks that have all
        # of them -- so every track in the average contributes the same stems.
        means = [
            np.mean([r["scores"][mode][s]["sdr"] for s in stems])
            for r in scored
            if all(s in r["scores"][mode] for s in stems)
        ]
        return {"n_tracks": len(means), "sdr": float(np.mean(means)) if means else float("nan")}

    return {
        "per_stem": per_stem,
        "mean_3stem": track_mean(CORE_STEMS),
        "mean_4stem": track_mean(CORE_STEMS + ("ghatam",)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--saraga-root", type=str, default=None, help="overrides data.saraga_root")
    p.add_argument("--mixture", choices=("both", "stems", "mixdown"), default="both")
    p.add_argument("--seconds", type=float, default=None, help="central excerpt per track; 0 = whole track")
    p.add_argument("--limit", type=int, default=0, help="score only the first N tracks")
    p.add_argument("--outdir", type=str, default=None, help="default: the checkpoint's directory")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    from indian_mss.data.saraga import Saraga
    from indian_mss.models.separator import load_separator, separate
    from indian_mss.train.loop import pick_device

    device = pick_device(args.device)
    model, meta = load_separator(args.checkpoint, device)
    sources = list(meta["sources"])
    sample_rate = meta["stft_config"].sample_rate
    seconds = args.seconds if args.seconds is not None else float(cfg["train"].get("val_seconds", 60.0))
    modes = ("stems", "mixdown") if args.mixture == "both" else (args.mixture,)

    root = args.saraga_root or cfg.get("data", {}).get("saraga_root", "data/saraga")
    # Loading in mixdown mode returns the master when it shipped *and* the
    # stems, so one decode serves both mixture definitions.
    dataset = Saraga(root, sample_rate, stems=tuple(sources), mixture="mixdown" if "mixdown" in modes else "stems")
    track_ids = dataset.track_ids()[: args.limit] if args.limit else dataset.track_ids()
    print(f"scoring {len(track_ids)} Saraga tracks ({', '.join(modes)}), central {seconds:g} s each")

    rows = []
    for i, tid in enumerate(track_ids, 1):
        item = dataset.load(tid)
        mixture, stems, _ = _crop_centre(item, seconds)
        mixtures = {"stems": np.sum(list(stems.values()), axis=0).astype(np.float32)}
        if "mixdown" in modes and not item.mixture_is_synthetic:
            mixtures["mixdown"] = mixture

        refs = _audible(stems, mixtures["stems"])
        row = {"track_id": tid, "stems_scored": sorted(refs), "metadata": item.metadata, "scores": {}}
        for mode in modes:
            if mode not in mixtures or len(refs) < 2:
                continue
            est = separate(model, mixtures[mode], sample_rate, meta["segment_seconds"], device=device)
            row["scores"][mode] = evaluate_stems({s: est[sources.index(s)] for s in refs}, refs)
        rows.append(row)

        line = "  ".join(
            f"{mode}: " + " ".join(f"{s}={v['sdr']:+.1f}" for s, v in sc.items()) for mode, sc in row["scores"].items()
        )
        print(f"[{i}/{len(track_ids)}] {tid}  {line or 'skipped: fewer than two audible stems'}", flush=True)

    summary = {mode: summarize(rows, mode) for mode in modes}
    print(f"\n=== Saraga test, {Path(args.checkpoint).name} (epoch {meta.get('epoch')}) ===")
    for mode, s in summary.items():
        print(f"\nmixture = {mode}")
        for stem, v in s["per_stem"].items():
            print(
                f"  {stem:10s} SDR {v['sdr']:+6.2f}  SIR {v['sir']:+6.2f}  SAR {v['sar']:+6.2f} dB"
                f"   ({v['n_tracks']} tracks)"
            )
        for key in ("mean_3stem", "mean_4stem"):
            print(f"  {key:10s} SDR {s[key]['sdr']:+6.2f} dB   ({s[key]['n_tracks']} tracks)")

    outdir = Path(args.outdir) if args.outdir else Path(args.checkpoint).parent
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "saraga_eval.json"
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": meta.get("epoch"),
                "checkpoint_val_sdr": meta.get("val_sdr"),
                "sources": sources,
                "sample_rate": sample_rate,
                "seconds": seconds,
                "summary": summary,
                "per_track": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
