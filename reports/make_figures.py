#!/usr/bin/env python3
"""Regenerate every figure in ``reports/figures/`` for the progress presentation.

Everything here runs on SYNTHETIC audio -- Sanidha access is still pending --
and every figure says so. The synthetic ensemble is built to be Carnatic-like
rather than a toy: a voice singing a raga phrase with gamaka oscillations, a
violin shadowing it a beat behind, a tanpura-style drone, tuned mridangam
strokes and ghatam slaps.

    python reports/make_figures.py              # ~15 min on a laptop CPU
    python reports/make_figures.py --m2-only    # only the training figures

Writes PNGs plus ``reports/figures/numbers.json`` with every figure's numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from indian_mss.dsp.hpss import HPSSConfig, hpss_audio, hpss_spectrograms  # noqa: E402
from indian_mss.dsp.stft import STFTConfig, log_magnitude, magphase, stft  # noqa: E402
from indian_mss.eval.sdr import evaluate_stems  # noqa: E402

OUT = ROOT / "reports" / "figures"
SR = 16000
# Training audio is 8 kHz: four times cheaper per step on a laptop CPU and
# still above every fundamental and stroke in the ensemble (ghatam ~1.4 kHz).
TRAIN_SR = 8000

# Deck palette: ink, paper, and four stem colours that differ in lightness too.
INK, PAPER, MUTED = "#1B2430", "#F7F4EE", "#5B6573"
STEM_COLORS = {"vocal": "#2F4B7C", "violin": "#7FA7D9", "mridangam": "#D9822B", "ghatam": "#7A4A1E"}
ACCENT, ACCENT2 = "#D9822B", "#2F4B7C"

plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --- a synthetic Carnatic ensemble ---------------------------------------------

# Shankarabharanam (major-scale) svaras in semitones above Sa.
SVARAS = [0, 2, 4, 5, 7, 9, 11, 12]


def _harmonic(f0: np.ndarray, sr: int, amps: list[float]) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f0) / sr
    return sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps))


def melody_f0(duration: float, tonic: float, rng: np.random.Generator, sr: int) -> np.ndarray:
    """A raga phrase: held svaras joined by glides, with gamaka oscillation."""
    n = int(duration * sr)
    notes, t = [], 0.0
    while t < duration:
        d = rng.uniform(0.35, 0.9)
        notes.append((t, d, rng.choice(SVARAS)))
        t += d
    semis = np.zeros(n)
    for start, d, s in notes:
        a, b = int(start * sr), min(n, int((start + d) * sr))
        semis[a:b] = s
    # Glide between notes (~80 ms portamento), then gamaka: a 5-7 Hz
    # oscillation of up to a semitone on some notes, the microtonal motion
    # Western models fail to follow.
    k = int(0.08 * sr)
    semis = np.convolve(np.pad(semis, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    t = np.arange(n) / sr
    gamaka_depth = np.repeat(rng.choice([0.0, 0.4, 0.9], size=len(notes)), int(np.ceil(n / len(notes))))[:n]
    semis = semis + gamaka_depth * np.sin(2 * np.pi * rng.uniform(5, 7) * t)
    return tonic * 2 ** (semis / 12)


def stroke(sr: int, freq: float, decay: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    n = int(0.4 * sr)
    t = np.arange(n) / sr
    env = np.exp(-t / decay)
    tone = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2.0 * freq * t)
    burst = rng.standard_normal(n) * np.exp(-t / 0.008)
    return (1 - noise) * tone * env + noise * burst


def ensemble(duration: float, tonic: float, seed: int, sr: int = SR, ghatam: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    f0 = melody_f0(duration, tonic * 2, rng, sr)  # voice an octave above the drone Sa
    phrase_env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * np.arange(n) / sr / rng.uniform(2.5, 4)))
    vocal = _harmonic(f0, sr, [1.0, 0.6, 0.45, 0.2, 0.12, 0.06]) * phrase_env
    # The violin follows the voice ~150 ms late (Carnatic accompaniment), with
    # a brighter, sawtooth-like spectrum.
    lag = int(0.15 * sr)
    vf0 = np.concatenate([np.full(lag, f0[0]), f0[:-lag]])
    violin = 0.5 * _harmonic(vf0 * 2 ** (rng.normal(0, 0.05) / 12), sr, [1 / (k + 1) for k in range(10)])
    drone = 0.25 * sum(_harmonic(np.full(n, f), sr, [1, 0.5, 0.3, 0.2]) for f in (tonic, tonic * 1.5, tonic * 2))
    vocal = vocal + drone  # the shruti box sits on the vocal mic

    mrid = np.zeros(n)
    beat = 60 / rng.uniform(80, 110) / 2
    t = 0.0
    while t < duration - 0.4:
        i = int(t * sr)
        kind = rng.random()
        s = stroke(sr, tonic * (1 if kind < 0.5 else 2), 0.12 if kind < 0.5 else 0.05, 0.3, rng)
        mrid[i : i + len(s)] += rng.uniform(0.6, 1.0) * s
        t += beat * rng.choice([1, 1, 2, 0.5])
    gh = np.zeros(n)
    if ghatam:
        t = beat / 2
        while t < duration - 0.4:
            i = int(t * sr)
            s = stroke(sr, rng.uniform(900, 1400), 0.02, 0.7, rng)
            gh[i : i + len(s)] += 0.5 * s
            t += beat * rng.choice([1, 2])
    stems = {"vocal": vocal, "violin": violin, "mridangam": mrid, "ghatam": gh}
    peak = max(np.abs(sum(stems.values())).max(), 1e-9)
    return {k: (0.8 * v / peak).astype(np.float32) for k, v in stems.items()}


# --- figures -----------------------------------------------------------------


def spec_panel(ax, mag: np.ndarray, cfg: STFTConfig, ref: float, title: str, fmax: float = 4000):
    img = log_magnitude(mag, ref=ref, ref_db=70)
    n_bins = int(fmax / (cfg.sample_rate / cfg.n_fft))
    dur = mag.shape[1] * cfg.hop_length / cfg.sample_rate
    ax.imshow(img[:n_bins], origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, fmax / 1000])
    ax.set_title(title, loc="left")
    ax.set_xlabel("time (s)")


def fig_hpss(numbers: dict) -> None:
    stems = ensemble(8.0, 146.83, seed=1)
    mix = sum(stems.values())
    cfg = STFTConfig(sample_rate=SR, n_fft=1024, hop_length=256)
    hcfg = HPSSConfig(harmonic_ms=200, percussive_hz=500)
    res = hpss_spectrograms(mix, cfg, hcfg)
    peak = float((res.harmonic + res.percussive).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    spec_panel(axes[0], res.harmonic + res.percussive + res.residual, cfg, peak, "Input mix")
    spec_panel(axes[1], res.harmonic, cfg, peak, "Melodic spectrogram")
    spec_panel(axes[2], res.percussive, cfg, peak, "Percussive spectrogram")
    axes[0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(OUT / "m1_hpss_spectrograms.png", dpi=200)
    plt.close(fig)

    est = hpss_audio(mix, cfg, hcfg)
    refs = {"harmonic": stems["vocal"] + stems["violin"], "percussive": stems["mridangam"] + stems["ghatam"]}
    scores = evaluate_stems({k: est[k] for k in refs}, refs)
    numbers["hpss_default"] = scores

    # Kernel sweep: the domain-tuning result.
    h_ms = [50, 100, 200, 400, 800]
    p_hz = [125, 250, 500, 1000, 2000]
    grid = np.zeros((len(h_ms), len(p_hz)))
    for i, h in enumerate(h_ms):
        for j, p in enumerate(p_hz):
            e = hpss_audio(mix, cfg, HPSSConfig(harmonic_ms=h, percussive_hz=p))
            s = evaluate_stems({k: e[k] for k in refs}, refs)
            grid[i, j] = np.mean([s["harmonic"]["sdr"], s["percussive"]["sdr"]])
    numbers["kernel_sweep"] = {"harmonic_ms": h_ms, "percussive_hz": p_hz, "mean_sdr": grid.tolist()}

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, cmap="cividis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(p_hz)), [str(p) for p in p_hz])
    ax.set_yticks(range(len(h_ms)), [str(h) for h in h_ms])
    ax.set_xlabel("frequency kernel (Hz)")
    ax.set_ylabel("time kernel (ms)")
    best = np.unravel_index(grid.argmax(), grid.shape)
    for i in range(len(h_ms)):
        for j in range(len(p_hz)):
            light = grid[i, j] > grid.min() + 0.6 * (grid.max() - grid.min())
            ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=17,
                    color=INK if light else PAPER, fontweight="bold" if (i, j) == best else "normal")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("mean SDR (dB)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "m1_kernel_sweep.png", dpi=200)
    plt.close(fig)
    numbers["kernel_best"] = {"harmonic_ms": h_ms[best[0]], "percussive_hz": p_hz[best[1]], "mean_sdr": float(grid.max())}


def _fold_old(f0: np.ndarray, fmin: float, fmax: float) -> float:
    """``estimate_tonic`` as first committed (a7015fa), octave step included."""
    cents = 1200.0 * np.log2(f0 / fmin)
    folded = np.mod(cents, 1200.0)
    hist, edges = np.histogram(folded, bins=120, range=(0.0, 1200.0))
    peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    candidate = fmin * 2 ** (peak / 1200.0)
    while candidate < fmin:
        candidate *= 2
    while candidate > fmax:
        candidate /= 2
    return float(candidate)


def fig_tonic(numbers: dict) -> None:
    import librosa

    from indian_mss.dsp.tonic import estimate_tonic

    true = np.geomspace(105, 380, 22)
    old, new = [], []
    for f in true:
        t = np.arange(int(2.0 * SR)) / SR
        drone = sum(0.5 / (k + 1) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        f0, voiced, _ = librosa.pyin(drone.astype(np.float32), fmin=100.0, fmax=400.0, sr=SR)
        f0 = f0[voiced & np.isfinite(f0)]
        old.append(_fold_old(f0, 100.0, 400.0))
        new.append(estimate_tonic(drone, SR))
    numbers["tonic"] = {"true": true.tolist(), "old": old, "new": new}

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axvspan(200, 400, color="#E9E2D6", zorder=0)
    ax.text(205, 375, "typical female\nvocal Sa range", va="top", fontsize=16, color=MUTED)
    ax.plot(true, true, color=MUTED, lw=1.5, ls="--", label="correct")
    ax.plot(true, old, "o-", color=ACCENT, lw=2.5, ms=8, label="before fix")
    ax.plot(true, new, "s-", color=ACCENT2, lw=2.5, ms=7, label="after fix")
    ax.set_xlabel("true tonic (Hz)")
    ax.set_ylabel("estimated tonic (Hz)")
    ax.set_xlim(100, 390)
    ax.set_ylim(90, 400)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "m1_tonic_fix.png", dpi=200)
    plt.close(fig)


def fig_training(numbers: dict, workdir: Path) -> None:
    import soundfile as sf
    import torch

    from indian_mss.models.separator import load_separator, predict_masks, separate
    from indian_mss.train.loop import train

    corpus = workdir / "sanidha_synthetic"
    if not corpus.exists():
        for i in range(6):
            stems = ensemble(30.0, [130.8, 146.8, 164.8, 185.0, 207.7, 138.6][i], seed=10 + i, sr=TRAIN_SR,
                             ghatam=i % 2 == 0)
            d = corpus / f"concert{i + 1}"
            d.mkdir(parents=True)
            for name, a in stems.items():
                if name == "ghatam" and not np.any(a):
                    continue  # like Sanidha: no ghatam in some concerts
                sf.write(d / f"{name}.wav", a, TRAIN_SR)

    cfg = {
        "stft": {"sample_rate": TRAIN_SR, "n_fft": 512, "hop_length": 128, "window": "hann"},
        "data": {"stems": list(STEM_COLORS)},
        "augment": {
            "random_gain_db": [-6.0, 6.0],
            "shuffle_stems_across_tracks": True,
            "shuffle_prob": 0.5,
            "simulate_bleed": {"enabled": True, "prob": 0.8, "neighbour_gain_db": [-30.0, -12.0], "rir_ms": 80.0},
        },
        "train": {
            "backbone": "unet", "seed": 0, "epochs": 30, "segment_seconds": 3.0, "samples_per_epoch": 400,
            "batch_size": 8, "num_workers": 0, "val_fraction": 0.2, "val_seconds": 20.0, "lr": 1e-3,
            "unet": {"channels": 16, "depth": 4},
        },
    }
    run = workdir / "run"
    torch.manual_seed(0)
    history = train(cfg, corpus, run, device="cpu", progress=False)
    numbers["training"] = {"epochs": history["epochs"], "best": history["best"], "val_tracks": history["val_tracks"]}

    epochs = history["epochs"]
    x = [e["epoch"] for e in epochs]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    ax = axes[0]
    ax.plot(x, [e["train_loss"] for e in epochs], "o-", color=ACCENT2, lw=2.5, ms=6, label="train")
    ax.plot(x, [e["val_loss"] for e in epochs], "s-", color=ACCENT, lw=2.5, ms=6, label="validation")
    ax.set_title("Loss", loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("spectrogram L1 + waveform L1")
    ax.legend(frameon=False)
    ax = axes[1]
    for stem, c in STEM_COLORS.items():
        y = [e["val_sdr"].get(stem, np.nan) for e in epochs]
        if not np.all(np.isnan(y)):
            ax.plot(x, y, "o-", color=c, lw=2, ms=5, label=stem)
    ax.plot(x, [e["val_sdr_mean"] for e in epochs], color=INK, lw=3.5, label="mean")
    b = history["best"]
    ax.axvline(b["epoch"], color=MUTED, ls=":", lw=1.5)
    ax.annotate(f"kept: epoch {b['epoch']}", (b["epoch"], ax.get_ylim()[0]), xytext=(6, 8),
                textcoords="offset points", fontsize=16, color=MUTED)
    ax.set_title("Validation SDR per instrument (dB)", loc="left")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False, ncol=3, fontsize=15, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "m2_training_curves.png", dpi=200)
    plt.close(fig)

    # Masks on the held-out concert, and the learned model vs HPSS on it.
    model, meta = load_separator(run / "best.pt")
    stft_cfg = meta["stft_config"]
    val_dir = corpus / history["val_tracks"][0]
    stems = {p.stem: sf.read(p)[0].astype(np.float32) for p in sorted(val_dir.glob("*.wav"))}
    a, b_ = int(5 * TRAIN_SR), int(13 * TRAIN_SR)
    stems = {k: v[a:b_] for k, v in stems.items()}
    mix = sum(stems.values()).astype(np.float32)
    masks, _ = predict_masks(model, mix, stft_cfg, meta["segment_seconds"])
    mag, _ = magphase(stft(mix, stft_cfg))

    shown = [s for s in meta["sources"] if s in stems]
    fig, axes = plt.subplots(1, len(shown) + 1, figsize=(16, 5.6), sharey=True)
    spec_panel(axes[0], mag, stft_cfg, float(mag.max()), "Mix")
    for ax, s in zip(axes[1:], shown):
        i = meta["sources"].index(s)
        dur = mag.shape[1] * stft_cfg.hop_length / TRAIN_SR
        n_bins = int(4000 / (TRAIN_SR / stft_cfg.n_fft))
        ax.imshow(masks[i, :n_bins], origin="lower", aspect="auto", cmap="Greys_r", vmin=0, vmax=1,
                  extent=[0, dur, 0, 4])
        ax.set_title(f"{s} mask", loc="left", color=STEM_COLORS[s])
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(OUT / "m2_masks.png", dpi=200)
    plt.close(fig)

    # Same two categories HPSS produces, on the same held-out excerpt.
    full = {k: v for k, v in stems.items()}
    est = separate(model, mix, TRAIN_SR, meta["segment_seconds"])
    est_d = {s: est[i] for i, s in enumerate(meta["sources"])}
    refs = {"melodic": full["vocal"] + full["violin"],
            "percussive": full["mridangam"] + full.get("ghatam", 0 * mix)}
    learned = evaluate_stems({"melodic": est_d["vocal"] + est_d["violin"],
                              "percussive": est_d["mridangam"] + est_d["ghatam"]}, refs)
    # HPSS gets its best kernels from the Milestone 1 sweep, not the default:
    # the comparison should be against the strongest baseline we have.
    best = numbers.get("kernel_best", {"harmonic_ms": 200.0, "percussive_hz": 500.0})
    h = hpss_audio(mix, stft_cfg, HPSSConfig(harmonic_ms=best["harmonic_ms"], percussive_hz=best["percussive_hz"]))
    hp = evaluate_stems({"melodic": h["harmonic"], "percussive": h["percussive"]}, refs)
    numbers["learned_vs_hpss"] = {"track": history["val_tracks"][0], "unet": learned, "hpss": hp}

    fig, ax = plt.subplots(figsize=(10, 6.2))
    cats = ["melodic", "percussive"]
    xs = np.arange(len(cats))
    w = 0.36
    for off, (label, sc, c) in enumerate((("HPSS (Milestone 1)", hp, "#B9B2A6"), ("U-Net (Milestone 2)", learned, ACCENT2))):
        vals = [sc[k]["sdr"] for k in cats]
        bars = ax.bar(xs + (off - 0.5) * w, vals, w, color=c, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.3 if v >= 0 else -0.9), f"{v:+.1f}",
                    ha="center", fontsize=17, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(xs, cats)
    ax.set_ylabel("SDR (dB), held-out concert")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "m2_vs_hpss.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    work = Path(args[0]) if args else ROOT / "out" / "figure_work"
    work.mkdir(parents=True, exist_ok=True)
    numbers_path = OUT / "numbers.json"
    numbers: dict = json.loads(numbers_path.read_text()) if numbers_path.is_file() else {}
    numbers["note"] = "all numbers are on synthetic audio; see reports/make_figures.py"
    if "--m2-only" not in sys.argv:
        fig_hpss(numbers)
        print("milestone 1 figures done", flush=True)
        fig_tonic(numbers)
        print("tonic figure done", flush=True)
        numbers_path.write_text(json.dumps(numbers, indent=2, default=float))
    fig_training(numbers, work)
    print("milestone 2 figures done")
    numbers_path.write_text(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()
