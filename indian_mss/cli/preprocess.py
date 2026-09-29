#!/usr/bin/env python3
"""Milestone 1 CLI: polyphonic mix in, melodic + percussive spectrograms out.

Examples
--------
Separate one file and write both spectrogram arrays plus audio previews::

    python scripts/preprocess.py --input concert.wav --outdir out/ --write-audio

Score the HPSS baseline against Sanidha ground truth (harmonic stems vs
percussive stems), which is the number the neural model must beat in
Milestone 2::

    python scripts/preprocess.py --sanidha-root data/sanidha --evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from indian_mss.dsp.hpss import HPSSConfig, hpss_spectrograms
from indian_mss.dsp.stft import STFTConfig, istft
from indian_mss.eval.sdr import evaluate_stems

# Which Sanidha stems count as melodic vs percussive when scoring HPSS.
HARMONIC_STEMS = ("vocal", "violin")
PERCUSSIVE_STEMS = ("mridangam", "ghatam")


def build_configs(args: argparse.Namespace) -> tuple[STFTConfig, HPSSConfig]:
    stft_cfg = STFTConfig(
        sample_rate=args.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length
    )
    hpss_cfg = HPSSConfig(
        harmonic_ms=args.harmonic_ms,
        percussive_hz=args.percussive_hz,
        power=args.power,
        margin=args.margin,
    )
    return stft_cfg, hpss_cfg


def run_single(args: argparse.Namespace) -> None:
    from indian_mss.data.base import load_audio

    stft_cfg, hpss_cfg = build_configs(args)
    audio = load_audio(args.input, stft_cfg.sample_rate)

    if args.normalize_tuning:
        from indian_mss.dsp.tonic import normalize_tuning

        audio, tonic = normalize_tuning(audio, stft_cfg.sample_rate)
        print(f"estimated tonic: {tonic:.2f} Hz (normalized)")

    result = hpss_spectrograms(audio, stft_cfg, hpss_cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    np.save(outdir / f"{stem}_melodic.npy", result.harmonic.astype(np.float32))
    np.save(outdir / f"{stem}_percussive.npy", result.percussive.astype(np.float32))
    np.save(outdir / f"{stem}_residual.npy", result.residual.astype(np.float32))
    print(f"wrote spectrogram arrays of shape {result.harmonic.shape} to {outdir}")

    if args.write_audio:
        import soundfile as sf

        n = len(audio)
        for name, mag in (
            ("melodic", result.harmonic),
            ("percussive", result.percussive),
            ("residual", result.residual),
        ):
            wav = istft(mag * result.phase, stft_cfg, length=n)
            sf.write(outdir / f"{stem}_{name}.wav", wav, stft_cfg.sample_rate)
        print(f"wrote audio previews to {outdir}")


def run_evaluate(args: argparse.Namespace) -> None:
    from indian_mss.data.sanidha import Sanidha
    from indian_mss.dsp.hpss import hpss_audio

    stft_cfg, hpss_cfg = build_configs(args)
    dataset = Sanidha(args.sanidha_root, stft_cfg.sample_rate)

    rows = []
    track_ids = dataset.track_ids()[: args.limit] if args.limit else dataset.track_ids()
    for tid in track_ids:
        item = dataset.load(tid)
        est = hpss_audio(item.mixture, stft_cfg, hpss_cfg)

        # Collapse ground truth to the two HPSS categories.
        refs = {}
        harm = [item.stems[s] for s in HARMONIC_STEMS if s in item.stems]
        perc = [item.stems[s] for s in PERCUSSIVE_STEMS if s in item.stems]
        if harm:
            refs["harmonic"] = np.sum(harm, axis=0)
        if perc:
            refs["percussive"] = np.sum(perc, axis=0)
        if len(refs) < 2:
            print(f"skipping {tid}: needs both a melodic and a percussive reference")
            continue

        scores = evaluate_stems({k: est[k] for k in refs}, refs)
        # Carry the synthetic-mixture flag into the record: a mixture summed
        # from clean stems is easier to separate than a real mixdown, so the
        # baseline is optimistic and the number is meaningless without it.
        rows.append(
            {
                "track_id": tid,
                "mixture_is_synthetic": item.mixture_is_synthetic,
                "stems_scored": sorted(item.stems),
                **scores,
            }
        )
        print(
            f"{tid}: "
            + "  ".join(f"{k} SDR={v['sdr']:+.2f} SIR={v['sir']:+.2f}" for k, v in scores.items())
            + ("  [synthetic mix]" if item.mixture_is_synthetic else "")
        )

    if not rows:
        print("no tracks scored")
        return

    summary = {}
    for category in ("harmonic", "percussive"):
        vals = [r[category] for r in rows if category in r]
        if vals:
            summary[category] = {
                m: float(np.mean([v[m] for v in vals])) for m in ("sdr", "sir", "sar", "si_sdr")
            }

    n_synthetic = sum(r["mixture_is_synthetic"] for r in rows)

    print("\n=== HPSS baseline (mean over tracks) ===")
    for category, metrics in summary.items():
        print(
            f"{category:12s} SDR {metrics['sdr']:+6.2f} dB   "
            f"SIR {metrics['sir']:+6.2f} dB   SAR {metrics['sar']:+6.2f} dB"
        )
    if n_synthetic:
        print(
            f"\nwarning: {n_synthetic}/{len(rows)} mixtures were synthesized by summing "
            "stems -- easier to separate than a real mixdown, so these numbers are optimistic."
        )

    out = Path(args.outdir) / "hpss_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "stft": {
                        "sample_rate": stft_cfg.sample_rate,
                        "n_fft": stft_cfg.n_fft,
                        "hop_length": stft_cfg.hop_length,
                        "window": stft_cfg.window,
                    },
                    "hpss": {
                        "harmonic_ms": hpss_cfg.harmonic_ms,
                        "percussive_hz": hpss_cfg.percussive_hz,
                        "power": hpss_cfg.power,
                        "margin": hpss_cfg.margin,
                    },
                },
                "n_tracks": len(rows),
                "n_synthetic_mixtures": n_synthetic,
                "per_track": rows,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nbaseline written to {out} -- Milestone 2 must beat these numbers")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=str, help="polyphonic audio file to process")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--write-audio", action="store_true", help="also resynthesize .wav previews")
    p.add_argument("--normalize-tuning", action="store_true", help="pitch-shift to the canonical tonic")

    p.add_argument("--evaluate", action="store_true", help="score HPSS against Sanidha ground truth")
    p.add_argument("--sanidha-root", type=str, default="data/sanidha")
    p.add_argument("--limit", type=int, default=0, help="score only the first N tracks")

    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--harmonic-ms", type=float, default=200.0)
    p.add_argument("--percussive-hz", type=float, default=500.0)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.evaluate:
        run_evaluate(args)
    elif args.input:
        run_single(args)
    else:
        p.error("pass --input FILE or --evaluate")


if __name__ == "__main__":
    main()
