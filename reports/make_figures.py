#!/usr/bin/env python3
"""Regenerate the figures in ``reports/figures/``.

Milestone 1 figures run on REAL audio, with the pipeline's own STFT and HPSS
settings from ``configs/default.yaml``:

* ``m1_hpss_spectrograms.png``: the 30 s deliverable clip (Concert03, never
  trained on), first 10 s: mix, melodic and percussive spectrograms.
* ``m1_kernel_sweep.png``: HPSS SDR over time and frequency kernel sizes, on
  all six Concert03 tracks scored against their ground-truth stems.
* ``m1_tonic_saraga.png``: ``dsp.tonic.estimate_tonic`` against the
  annotated tonic of every Saraga Carnatic recording that has one.

``--m2-synthetic`` additionally runs the old synthetic-audio training demo
(a toy Carnatic-like ensemble trained at 8 kHz). Its figures predate access to
Sanidha. The real training curves are in
``deliverables/milestone_2_model/training_report.pdf``.

    python reports/make_figures.py                  # ~20 min, needs data/sanidha_flat and data/saraga
    python reports/make_figures.py --m2-synthetic   # also the synthetic training demo

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


# --- Milestone 1 figures: real audio -----------------------------------------
#
# The spectrograms use the 30 s deliverable clip (Concert03, held out from
# training); the kernel sweep scores HPSS on all six Concert03 tracks against
# their ground-truth stems; the tonic figure scores ``estimate_tonic`` on every
# Saraga Carnatic recording with an annotated tonic. STFT and HPSS settings
# come from configs/default.yaml, the same ones the pipeline uses.

SANIDHA = ROOT / "data" / "sanidha_flat"
SARAGA = ROOT / "data" / "saraga" / "saraga1.5_carnatic"
SAMPLE = ROOT / "deliverables" / "sample"
SWEEP_SECONDS = 30.0
TONIC_SECONDS = 30.0
TONIC_SR = 22050
MELODIC, PERCUSSIVE = ("vocal", "violin"), ("mridangam", "ghatam")


def _pipeline_configs():
    from indian_mss.config import load_config

    cfg = load_config(ROOT / "configs" / "default.yaml")
    s, h = cfg["stft"], cfg["hpss"]
    stft_cfg = STFTConfig(sample_rate=s["sample_rate"], n_fft=s["n_fft"], hop_length=s["hop_length"])
    hpss_cfg = HPSSConfig(harmonic_ms=h["harmonic_ms"], percussive_hz=h["percussive_hz"], power=h["power"], margin=h["margin"])
    return stft_cfg, hpss_cfg


def _refs(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(next(iter(stems.values())))
    pick = lambda names: np.sum([stems[k] for k in names if k in stems] or [np.zeros(n)], axis=0)  # noqa: E731
    return {"harmonic": pick(MELODIC), "percussive": pick(PERCUSSIVE)}


def _sweep_one(job):
    mix, refs, stft_cfg, h, p = job
    e = hpss_audio(mix, stft_cfg, HPSSConfig(harmonic_ms=h, percussive_hz=p))
    s = evaluate_stems({k: e[k] for k in refs}, refs)
    return h, p, s["harmonic"]["sdr"], s["percussive"]["sdr"]


def fig_hpss(numbers: dict) -> None:
    from multiprocessing import Pool

    from indian_mss.cli.evaluate import _crop_centre
    from indian_mss.data.base import load_audio
    from indian_mss.data.sanidha import Sanidha

    stft_cfg, hpss_cfg = _pipeline_configs()
    sr = stft_cfg.sample_rate

    # Spectrograms: the first 10 s of the deliverable clip, legible at slide size.
    mix = load_audio(SAMPLE / "sample_mix.wav", sr)
    stems = {k: load_audio(SAMPLE / f"reference_{k}.wav", sr) for k in MELODIC + PERCUSSIVE}
    shown = mix[: 10 * sr]
    res = hpss_spectrograms(shown, stft_cfg, hpss_cfg)
    peak = float((res.harmonic + res.percussive + res.residual).max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    spec_panel(axes[0], res.harmonic + res.percussive + res.residual, stft_cfg, peak, "Input mix")
    spec_panel(axes[1], res.harmonic, stft_cfg, peak, "Melodic spectrogram")
    spec_panel(axes[2], res.percussive, stft_cfg, peak, "Percussive spectrogram")
    axes[0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(OUT / "m1_hpss_spectrograms.png", dpi=200)
    plt.close(fig)

    est = hpss_audio(mix, stft_cfg, hpss_cfg)
    refs = _refs(stems)
    numbers["hpss_default_sample_clip"] = evaluate_stems({k: est[k] for k in refs}, refs)

    # Kernel sweep over Concert03 (the held-out concert), central 30 s per track.
    corpus = Sanidha(SANIDHA, sr, stems=MELODIC + PERCUSSIVE)
    tracks = [t for t in corpus.track_ids() if t.startswith("Concert03")]
    h_ms = [50, 100, 200, 400, 800]
    p_hz = [125, 250, 500, 1000, 2000]
    jobs = []
    for tid in tracks:
        m, st, _ = _crop_centre(corpus.load(tid), SWEEP_SECONDS)
        r = _refs(st)
        jobs += [(m.copy(), {k: v.copy() for k, v in r.items()}, stft_cfg, h, p) for h in h_ms for p in p_hz]
    with Pool(6) as pool:
        results = pool.map(_sweep_one, jobs)
    harm = np.zeros((len(h_ms), len(p_hz)))
    perc = np.zeros_like(harm)
    for h, p, hs, ps in results:
        i, j = h_ms.index(h), p_hz.index(p)
        harm[i, j] += hs / len(tracks)
        perc[i, j] += ps / len(tracks)
    grid = (harm + perc) / 2
    numbers["kernel_sweep"] = {
        "data": f"Sanidha Concert03, {len(tracks)} tracks, central {SWEEP_SECONDS:g} s",
        "harmonic_ms": h_ms, "percussive_hz": p_hz,
        "mean_sdr": grid.tolist(), "harmonic_sdr": harm.tolist(), "percussive_sdr": perc.tolist(),
    }  # fmt: skip

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
                    color=INK if light else PAPER, fontweight="bold" if (i, j) == best else "normal")  # fmt: skip
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("mean SDR, melodic + percussive (dB)")
    ax.set_title(f"HPSS on Concert03 ({len(tracks)} tracks)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "m1_kernel_sweep.png", dpi=200)
    plt.close(fig)
    numbers["kernel_best"] = {"harmonic_ms": h_ms[best[0]], "percussive_hz": p_hz[best[1]], "mean_sdr": float(grid.max())}
    d = (h_ms.index(int(hpss_cfg.harmonic_ms)), p_hz.index(int(hpss_cfg.percussive_hz)))
    numbers["kernel_config_default"] = {"harmonic_ms": hpss_cfg.harmonic_ms, "percussive_hz": hpss_cfg.percussive_hz, "mean_sdr": float(grid[d])}


def _tonic_one(job):
    import warnings

    from indian_mss.data.base import audio_duration, load_segment
    from indian_mss.dsp.tonic import estimate_tonic

    audio_path, true_hz = job
    warnings.filterwarnings("ignore")
    dur = audio_duration(audio_path)
    start = max(0.0, (dur - TONIC_SECONDS) / 2)
    x = load_segment(audio_path, TONIC_SR, start, TONIC_SECONDS)
    return str(audio_path), true_hz, float(estimate_tonic(x, TONIC_SR))


def _tonic_jobs() -> list[tuple[Path, float]]:
    jobs = []
    for ann in sorted(SARAGA.rglob("*.ctonic.txt")):
        if ann.name.startswith("._"):
            continue
        piece = ann.name[: -len(".ctonic.txt")]
        audio = next((ann.with_name(piece + ext) for ext in (".mp3.mp3", ".mp3", ".wav") if ann.with_name(piece + ext).is_file()), None)
        if audio is not None:
            jobs.append((audio, float(ann.read_text().split()[0])))
    return jobs


def fig_tonic(numbers: dict) -> None:
    """``estimate_tonic`` against Saraga's annotated tonics (central 30 s of each released recording)."""
    from multiprocessing import Pool

    with Pool(8) as pool:
        rows = pool.map(_tonic_one, _tonic_jobs())
    true = np.array([t for _, t, _ in rows])
    est = np.array([e for *_, e in rows])
    cents = 1200 * np.log2(est / true)
    folded = (cents + 600) % 1200 - 600  # pitch-class error, octave ignored

    def share(mask):
        return float(np.mean(mask))

    exact = np.abs(cents) <= 50
    octave = ~exact & (np.abs(folded) <= 50)
    # Folded to [-600, 600) cents, Pa taken for Sa (a fifth up, or a fourth
    # down) lands at -500 whatever the octave; Ma taken for Sa at +500.
    pa = ~exact & ~octave & (np.abs(folded + 500) <= 50)
    ma = ~exact & ~octave & (np.abs(folded - 500) <= 50)
    other = ~(exact | octave | pa | ma)
    numbers["tonic_saraga"] = {
        "data": f"Saraga Carnatic, {len(rows)} annotated recordings, central {TONIC_SECONDS:g} s of the released audio",
        "n": len(rows),
        "correct_within_50_cents": share(exact),
        "octave_error": share(octave),
        "pa_taken_for_sa": share(pa),
        "ma_taken_for_sa": share(ma),
        "other": share(other),
        "median_abs_cents_when_correct": float(np.median(np.abs(cents[exact]))) if exact.any() else None,
        "per_recording": [{"audio": a, "true_hz": t, "est_hz": e} for a, t, e in rows],
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))
    lo, hi = 90, 420
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, ls="--", label="correct")
    for k, ls in ((2, ":"), (0.5, ":")):
        ax.plot([lo, hi], [lo * k, hi * k], color=MUTED, lw=1, ls=ls)
    ax.plot(true[exact], est[exact], "o", color=ACCENT2, ms=7, label=f"within 50 cents ({exact.mean():.0%})")
    ax.plot(true[octave], est[octave], "s", color=ACCENT, ms=7, label=f"octave error ({octave.mean():.0%})")
    rest = ~(exact | octave)
    ax.plot(true[rest], est[rest], "x", color=INK, ms=8, mew=2, label=f"other ({rest.mean():.0%})")
    ax.set_xlabel("annotated tonic (Hz)")
    ax.set_ylabel("estimated tonic (Hz)")
    ax.set_xlim(lo, max(260.0, float(true.max()) * 1.05))
    ax.set_ylim(lo, hi)
    ax.set_title(f"Tonic estimation on Saraga Carnatic ({len(rows)} recordings)", loc="left")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "m1_tonic_saraga.png", dpi=200)
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
    numbers_path = OUT / "numbers.json"
    numbers: dict = {"note": "Milestone 1 figures are on real audio; see reports/make_figures.py"}
    fig_hpss(numbers)
    print("HPSS figures done", flush=True)
    fig_tonic(numbers)
    print("tonic figure done", flush=True)
    numbers_path.write_text(json.dumps(numbers, indent=2, default=float))
    if "--m2-synthetic" in sys.argv:
        work = Path(args[0]) if args else ROOT / "out" / "figure_work"
        work.mkdir(parents=True, exist_ok=True)
        synthetic: dict = {"note": "synthetic audio"}
        fig_training(synthetic, work)
        numbers["m2_synthetic_demo"] = synthetic
        numbers_path.write_text(json.dumps(numbers, indent=2, default=float))
        print("synthetic milestone 2 figures done")


if __name__ == "__main__":
    main()
