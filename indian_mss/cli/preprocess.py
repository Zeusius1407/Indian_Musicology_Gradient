#!/usr/bin/env python3
"""Milestone 1 CLI: polyphonic mix in, melodic + percussive spectrograms out.

Examples
--------
Separate one file and write both spectrogram arrays plus audio previews::

    python scripts/preprocess.py --input concert.wav --outdir out/ --write-audio

Each array is written twice: ``*_melodic.npy`` is the linear magnitude (what
masking and resynthesis need) and ``*_melodic_log.npy`` is the log-compressed
spectrogram in [0, 1] (what a network or a plot wants). All log arrays share
the mixture's peak as reference, so their levels are comparable.

Hyperparameters come from ``configs/default.yaml``; any flag given on the
command line overrides the file::

    python scripts/preprocess.py --input concert.wav --harmonic-ms 120 --tonic-strategy none

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

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config
from indian_mss.dsp.hpss import HPSSConfig, hpss_spectrograms
from indian_mss.dsp.stft import STFTConfig, istft, log_magnitude
from indian_mss.eval.sdr import evaluate_stems

# Which Sanidha stems count as melodic vs percussive when scoring HPSS.
HARMONIC_STEMS = ("vocal", "violin")
PERCUSSIVE_STEMS = ("mridangam", "ghatam")

TONIC_STRATEGIES = ("shift", "condition", "none")

# (argparse attribute, config section, config key, fallback if neither is set).
# Command-line flags win over the config file, which wins over the fallback.
CONFIG_KEYS = (
    ("sample_rate", "stft", "sample_rate", 44100),
    ("n_fft", "stft", "n_fft", 2048),
    ("hop_length", "stft", "hop_length", 512),
    ("window", "stft", "window", "hann"),
    ("harmonic_ms", "hpss", "harmonic_ms", 200.0),
    ("percussive_hz", "hpss", "percussive_hz", 500.0),
    ("power", "hpss", "power", 2.0),
    ("margin", "hpss", "margin", 1.0),
    ("tonic_strategy", "tonic", "strategy", "none"),
    ("canonical_hz", "tonic", "canonical_hz", 146.83),
    ("sanidha_root", "data", "sanidha_root", "data/sanidha"),
)


def resolve_args(args: argparse.Namespace, p: argparse.ArgumentParser) -> None:
    """Fill every unset flag from the config file, then from the fallback."""
    if args.config is not None:
        cfg = load_config(args.config)
    elif DEFAULT_CONFIG_PATH.is_file():
        cfg = load_config(DEFAULT_CONFIG_PATH)
    else:
        cfg = {}
        print(f"note: {DEFAULT_CONFIG_PATH} not found, using built-in defaults")

    for attr, section, key, fallback in CONFIG_KEYS:
        if getattr(args, attr) is None:
            setattr(args, attr, (cfg.get(section) or {}).get(key, fallback))

    if args.tonic_strategy not in TONIC_STRATEGIES:
        p.error(f"tonic strategy must be one of {TONIC_STRATEGIES}, got {args.tonic_strategy!r}")


def build_configs(args: argparse.Namespace) -> tuple[STFTConfig, HPSSConfig]:
    stft_cfg = STFTConfig(
        sample_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        window=str(args.window),
    )
    hpss_cfg = HPSSConfig(
        harmonic_ms=float(args.harmonic_ms),
        percussive_hz=float(args.percussive_hz),
        power=float(args.power),
        margin=float(args.margin),
    )
    return stft_cfg, hpss_cfg


def run_single(args: argparse.Namespace) -> None:
    from indian_mss.data.base import load_audio

    stft_cfg, hpss_cfg = build_configs(args)
    audio = load_audio(args.input, stft_cfg.sample_rate)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    tonic_record = None
    if args.tonic_strategy == "shift":
        from indian_mss.dsp.tonic import normalize_tuning

        target = float(args.canonical_hz)
        audio, tonic = normalize_tuning(audio, stft_cfg.sample_rate, target_hz=target)
        tonic_record = {"strategy": "shift", "tonic_hz": tonic, "target_hz": target}
        print(f"estimated tonic: {tonic:.2f} Hz (shifted to {target:.2f} Hz)")
    elif args.tonic_strategy == "condition":
        from indian_mss.dsp.tonic import estimate_tonic

        tonic = estimate_tonic(audio, stft_cfg.sample_rate)
        tonic_record = {"strategy": "condition", "tonic_hz": tonic}
        print(f"estimated tonic: {tonic:.2f} Hz (recorded, audio unchanged)")
    if tonic_record is not None:
        # The source tonic is needed to undo a shift before delivering audio,
        # and is the conditioning input under the 'condition' strategy.
        (outdir / f"{stem}_tonic.json").write_text(json.dumps(tonic_record, indent=2))

    result = hpss_spectrograms(audio, stft_cfg, hpss_cfg)

    # One reference for all three log arrays: the loudest bin of the mixture.
    mix_peak = float((result.harmonic + result.percussive + result.residual).max())
    for name, mag in (
        ("melodic", result.harmonic),
        ("percussive", result.percussive),
        ("residual", result.residual),
    ):
        np.save(outdir / f"{stem}_{name}.npy", mag.astype(np.float32))
        np.save(outdir / f"{stem}_{name}_log.npy", log_magnitude(mag, ref=mix_peak).astype(np.float32))
    print(f"wrote linear and log spectrogram arrays of shape {result.harmonic.shape} to {outdir}")

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
    # Tonic shifting is deliberately not applied here: it would move the
    # mixture away from the unshifted reference stems and wreck the SDR.

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
                    "config_file": str(args.config or DEFAULT_CONFIG_PATH),
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
    p.add_argument(
        "--config", type=str, default=None, help=f"YAML config (default: {DEFAULT_CONFIG_PATH} if present)"
    )
    p.add_argument("--tonic-strategy", choices=TONIC_STRATEGIES, default=None)
    p.add_argument(
        "--normalize-tuning",
        dest="tonic_strategy",
        action="store_const",
        const="shift",
        help="alias for --tonic-strategy shift",
    )
    p.add_argument("--canonical-hz", type=float, default=None, help="tonic to shift to")

    p.add_argument("--evaluate", action="store_true", help="score HPSS against Sanidha ground truth")
    p.add_argument("--sanidha-root", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="score only the first N tracks")

    # Defaults are None so that "not given" can be told apart from "given the
    # same value as the config" -- unset flags are filled by resolve_args.
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--window", type=str, default=None)
    p.add_argument("--harmonic-ms", type=float, default=None)
    p.add_argument("--percussive-hz", type=float, default=None)
    p.add_argument("--power", type=float, default=None)
    p.add_argument("--margin", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    resolve_args(args, p)

    if args.evaluate:
        run_evaluate(args)
    elif args.input:
        run_single(args)
    else:
        p.error("pass --input FILE or --evaluate")


if __name__ == "__main__":
    main()
