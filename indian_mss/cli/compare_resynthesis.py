#!/usr/bin/env python3
"""Milestone 3 CLI: score every resynthesis method against ground truth.

Examples
--------
All methods on the checkpoint's held-out validation concert (central
``train.val_seconds`` of each track, the excerpt training validated on)::

    python scripts/compare_resynthesis.py --checkpoint checkpoints/htdemucs_concertsplit/best.pt \\
        --outdir out/resynthesis

The 30 s deliverable sample instead, writing full-length stems per method::

    python scripts/compare_resynthesis.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \\
        --sample deliverables/sample --clip-seconds 0 --outdir deliverables/milestone_3_resynthesis

Methods (``--methods`` picks a subset):

``direct``                HTDemucs's own waveform output (what Milestone 2 ships)
``mixture_phase``         predicted mask x mixture STFT -> ISTFT (the baseline)
``griffin_lim``           predicted mask x |mixture|, phase from random init
``griffin_lim_mixinit``   the same, Griffin-Lim started from the mixture phase
``vocoder``               predicted mask x |mixture| -> mel -> BigVGAN
``oracle_mixture_phase``  ideal ratio mask from the references x mixture STFT:
                          the ceiling for any mixture-phase method

Each stem gets SDR / SIR / SAR / SI-SDR and ``mag_snr``, a phase-blind
spectral match (``eval.sdr.magnitude_snr``). Compare Griffin-Lim and the
vocoder on ``mag_snr`` and by ear, not on SDR: BSS-Eval needs sample-level
phase agreement that neither method even attempts.

Writes ``resynthesis_comparison.json`` and ``clips/<track>/<method>/<source>.wav``
listening clips (the central ``--clip-seconds`` of the first track, alongside
the mixture and references).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from indian_mss.cli.evaluate import _audible, _crop_centre
from indian_mss.config import DEFAULT_CONFIG_PATH, load_config
from indian_mss.eval.sdr import evaluate_stems, magnitude_snr
from indian_mss.infer.resynth import ResynthConfig, resynthesize

METHODS = ("direct", "mixture_phase", "griffin_lim", "griffin_lim_mixinit", "vocoder", "oracle_mixture_phase")
METRICS = ("sdr", "sir", "sar", "si_sdr", "mag_snr")


def ratio_masks(stems: np.ndarray, stft_cfg) -> np.ndarray:
    """``|STFT(s)| / sum |STFT|`` -- the same mask form ``predict_masks`` derives."""
    from indian_mss.dsp.stft import stft

    mags = np.stack([np.abs(stft(s, stft_cfg)) for s in stems])
    total = mags.sum(axis=0, keepdims=True)
    return np.where(total > 1e-12, mags / np.maximum(total, 1e-12), 1.0 / len(stems)).astype(np.float32)


def run_methods(methods, mixture, refs_all, masks, direct, stft_cfg, rcfg, vocoder) -> dict[str, np.ndarray]:
    """``{method: (S, T)}`` for one track."""
    out = {}
    for method in methods:
        if method == "direct":
            out[method] = direct
        elif method == "oracle_mixture_phase":
            out[method] = resynthesize(ratio_masks(refs_all, stft_cfg), mixture, stft_cfg, replace(rcfg, method="mixture_phase"))
        elif method == "griffin_lim_mixinit":
            gl = replace(rcfg, method="griffin_lim", griffin_lim_init="mixture")
            out[method] = resynthesize(masks, mixture, stft_cfg, gl)
        elif method == "griffin_lim":
            out[method] = resynthesize(masks, mixture, stft_cfg, replace(rcfg, method="griffin_lim", griffin_lim_init="random"))
        else:
            out[method] = resynthesize(masks, mixture, stft_cfg, replace(rcfg, method=method), vocoder=vocoder)
    return out


def score(est: np.ndarray, refs: dict[str, np.ndarray], sources: list[str], stft_cfg) -> dict:
    rows = evaluate_stems({s: est[sources.index(s)] for s in refs}, refs)
    for s, row in rows.items():
        row["mag_snr"] = magnitude_snr(est[sources.index(s)], refs[s], stft_cfg)
    return rows


def summarize(rows: list[dict], methods: list[str]) -> dict:
    summary = {}
    for method in methods:
        per_stem = {}
        for stem in sorted({s for r in rows for s in r["scores"].get(method, {})}):
            vals = [r["scores"][method][stem] for r in rows if stem in r["scores"].get(method, {})]
            per_stem[stem] = {"n_tracks": len(vals), **{m: float(np.mean([v[m] for v in vals])) for m in METRICS}}
        summary[method] = {
            "per_stem": per_stem,
            # Mean over stems of the per-stem means, as training validation reports.
            "mean": {m: float(np.mean([v[m] for v in per_stem.values()])) for m in METRICS} if per_stem else {},
        }
    return summary


def write_clips(outdir: Path, track: str, mixture, refs_all, results, sources, sample_rate, seconds) -> None:
    import soundfile as sf

    n = len(mixture)
    keep = min(n, int(seconds * sample_rate)) if seconds else n
    a = (n - keep) // 2
    base = outdir / "clips" / track
    (base / "reference").mkdir(parents=True, exist_ok=True)
    sf.write(base / "mixture.wav", mixture[a : a + keep], sample_rate)
    for s, ref in zip(sources, refs_all):
        sf.write(base / "reference" / f"{s}.wav", ref[a : a + keep], sample_rate)
    for method, est in results.items():
        (base / method).mkdir(parents=True, exist_ok=True)
        for s, wav in zip(sources, est):
            sf.write(base / method / f"{s}.wav", wav[a : a + keep], sample_rate)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--sanidha-root", type=str, default=None, help="overrides data.sanidha_root")
    p.add_argument(
        "--sample", type=str, default=None, help="directory with sample_mix.wav + reference_<source>.wav, instead of Sanidha"
    )
    p.add_argument("--seconds", type=float, default=None, help="central excerpt per track; 0 = whole track")
    p.add_argument("--limit", type=int, default=0, help="score only the first N validation tracks")
    p.add_argument("--clip-seconds", type=float, default=10.0, help="listening clip length; 0 = whole excerpt")
    p.add_argument("--outdir", type=str, default="out/resynthesis")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    return p


def _tracks(args, cfg, meta, sources, sample_rate):
    """Yield ``(track_id, mixture, {source: reference})``."""
    from indian_mss.data.base import load_audio

    if args.sample:
        root = Path(args.sample)
        refs = {s: load_audio(root / f"reference_{s}.wav", sample_rate) for s in sources}
        yield root.name, load_audio(root / "sample_mix.wav", sample_rate), refs
        return

    from indian_mss.data.sanidha import Sanidha

    val_ids = list(meta.get("val_tracks") or [])
    if not val_ids:
        raise ValueError("checkpoint records no val_tracks; pass --sample instead")
    seconds = args.seconds if args.seconds is not None else float(cfg["train"].get("val_seconds", 60.0))
    root = args.sanidha_root or cfg.get("data", {}).get("sanidha_root", "data/sanidha")
    dataset = Sanidha(root, sample_rate, stems=tuple(sources))
    for tid in val_ids[: args.limit] if args.limit else val_ids:
        _, stems, _ = _crop_centre(dataset.load(tid), seconds)
        # Sanidha ships no mixdown: the mixture is the sum of the close mics,
        # which also makes the references sum exactly to it.
        stems = {s: stems.get(s, np.zeros_like(next(iter(stems.values())))) for s in sources}
        yield tid, np.sum(list(stems.values()), axis=0).astype(np.float32), stems


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    rcfg = ResynthConfig.from_config(cfg)

    from indian_mss.models.separator import load_separator, predict_masks
    from indian_mss.train.loop import pick_device

    device = pick_device(args.device)
    model, meta = load_separator(args.checkpoint, device)
    sources = list(meta["sources"])
    stft_cfg = meta["stft_config"]
    vocoder = None
    if "vocoder" in args.methods:
        from indian_mss.infer.vocoder import Vocoder

        vocoder = Vocoder(rcfg.vocoder, device=device)

    outdir = Path(args.outdir)
    rows = []
    for i, (tid, mixture, stems) in enumerate(_tracks(args, cfg, meta, sources, stft_cfg.sample_rate)):
        masks, direct = predict_masks(model, mixture, stft_cfg, meta["segment_seconds"], device=device)
        refs_all = np.stack([stems[s] for s in sources])
        results = run_methods(args.methods, mixture, refs_all, masks, direct, stft_cfg, rcfg, vocoder)

        refs = _audible(stems, mixture)
        row = {"track_id": tid, "stems_scored": sorted(refs), "scores": {}}
        if len(refs) >= 2:
            row["scores"] = {m: score(est, refs, sources, stft_cfg) for m, est in results.items()}
        rows.append(row)
        if i == 0:
            write_clips(outdir, tid, mixture, refs_all, results, sources, stft_cfg.sample_rate, args.clip_seconds)

        print(f"[{i + 1}] {tid}")
        for m, sc in row["scores"].items():
            print(f"    {m:22s} " + "  ".join(f"{s}={v['sdr']:+.1f}/{v['mag_snr']:+.1f}" for s, v in sc.items()))

    summary = summarize(rows, args.methods)
    print(f"\n=== resynthesis, {Path(args.checkpoint).name}, {len(rows)} track(s): mean over stems ===")
    print(f"  {'method':22s} " + " ".join(f"{m:>8s}" for m in METRICS))
    for method, s in summary.items():
        if s["mean"]:
            print(f"  {method:22s} " + " ".join(f"{s['mean'][m]:+8.2f}" for m in METRICS))

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "resynthesis_comparison.json"
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": meta.get("epoch"),
                "sources": sources,
                "stft": asdict(stft_cfg),
                "resynthesis": asdict(rcfg),
                "source": args.sample or "sanidha validation",
                "summary": summary,
                "per_track": rows,
            },
            indent=2,
        )
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
