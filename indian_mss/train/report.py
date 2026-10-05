"""Tables and charts for ``notebooks/train.ipynb``, built from ``history.json``.

Kept in the package rather than the notebook so they are tested, and so the
same figures can be regenerated from a finished run without re-training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["dataset_stats", "load_baseline", "plot_losses", "plot_val_sdr", "smooth"]


def dataset_stats(dataset) -> dict:
    """Track count, total hours, and hours of audio per stem for a ``Sanidha``."""
    per_stem: dict[str, float] = {}
    total = 0.0
    for tid in dataset.track_ids():
        duration = dataset.duration(tid)
        total += duration
        for stem in dataset.stem_paths(tid):
            per_stem[stem] = per_stem.get(stem, 0.0) + duration
    return {
        "tracks": len(dataset),
        "hours": total / 3600.0,
        "stem_hours": {k: v / 3600.0 for k, v in sorted(per_stem.items())},
    }


def load_baseline(path: str | Path = "out/hpss_baseline.json") -> dict | None:
    """The Milestone 1 HPSS summary written by ``preprocess.py --evaluate``, if any."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("summary")


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average -- per-step losses are too noisy to read raw."""
    v = np.asarray(values, dtype=float)
    if window <= 1 or len(v) < window:
        return v
    kernel = np.ones(window) / window
    head = np.cumsum(v[: window - 1]) / np.arange(1, window)
    return np.concatenate([head, np.convolve(v, kernel, mode="valid")])


def plot_losses(history: dict, ax=None):
    """Per-step training loss (smoothed) with per-epoch train and validation loss."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    steps = history["steps"]
    epochs = history["epochs"]
    if steps:
        x = [s["step"] for s in steps]
        y = [s["loss"] for s in steps]
        ax.plot(x, y, color="0.8", lw=0.8, label="step loss")
        ax.plot(x, smooth(y, max(1, len(y) // 50)), color="C0", lw=1.5, label="step loss (smoothed)")
    if epochs:
        per_epoch = max(1, len(steps) // len(epochs))
        ex = [e["epoch"] * per_epoch for e in epochs]
        ax.plot(ex, [e["train_loss"] for e in epochs], "o-", color="C1", label="train (epoch mean)")
        ax.plot(ex, [e["val_loss"] for e in epochs], "s-", color="C2", label="validation")
        _shade_phases(ax, epochs, per_epoch)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (spectrogram L1 + waveform L1)")
    ax.set_title("Training loss")
    ax.legend()
    return ax


def plot_val_sdr(history: dict, baseline: dict | None = None, ax=None):
    """Validation SDR per stem and mean, per epoch, with the best epoch marked.

    The HPSS baseline (Milestone 1) has only harmonic/percussive categories,
    so it is drawn as two reference lines rather than per stem.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    epochs = history["epochs"]
    x = [e["epoch"] for e in epochs]
    for i, stem in enumerate(history["sources"]):
        y = [e["val_sdr"].get(stem, np.nan) for e in epochs]
        ax.plot(x, y, "o-", ms=3, color=f"C{i}", label=stem)
    ax.plot(x, [e["val_sdr_mean"] for e in epochs], "k-", lw=2, label="mean")
    best = history.get("best")
    if best:
        ax.axvline(best["epoch"], color="k", ls=":", lw=1)
        ax.annotate(
            f"best {best['val_sdr_mean']:+.2f} dB",
            (best["epoch"], best["val_sdr_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    if baseline:
        for name, style in (("harmonic", "--"), ("percussive", "-.")):
            if name in baseline:
                ax.axhline(baseline[name]["sdr"], color="0.4", ls=style, lw=1, label=f"HPSS {name}")
    _shade_phases(ax, epochs, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("SDR (dB)")
    ax.set_title("Validation SDR (checkpoint selected on the mean)")
    ax.legend(ncol=3, fontsize="small")
    return ax


def _shade_phases(ax, epochs: list[dict], scale: int) -> None:
    """Grey band over the frozen-encoder epochs, so the unfreeze is visible."""
    frozen = [e["epoch"] for e in epochs if e.get("phase") == 1]
    if frozen and len(frozen) < len(epochs):
        ax.axvspan(0, max(frozen) * scale, color="0.92", zorder=0, label="encoder frozen")
