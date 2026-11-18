"""The Milestone 4 measurement: does a separated stem sound like one instrument?

For every test recording and every instrument, the windows where that
instrument's ground-truth stem is playing are classified four times:

``reference``  the clean close-mic stem -- the classifier's own ceiling
``separated``  the separator's stem for that instrument (Milestone 3 output)
``bleed_12db`` the clean stem plus every other instrument 12 dB down -- a
               control: the leakage level the separator was trained against.
               If the classifier scored this like ``reference`` it could not
               see bleed, and ``separated`` matching ``reference`` would prove
               nothing.
``mixture``    the full mix, the same audio for every label -- the floor

If separation removed the other instruments, ``separated`` sits near
``reference`` and far above ``mixture``. The confusion matrix says *where*
the remainder went: mass in row ``violin``, column ``vocal`` is vocal still
audible in the violin stem. Mean probability on the wrong classes
(``bleed_mass``) is the same signal without the argmax threshold.
"""

from __future__ import annotations

import numpy as np

from .data import active_starts, mfcc_windows
from .train import predict_proba

__all__ = ["CONDITIONS", "classify_clips", "classify_track", "summarize"]

CONDITIONS = ("reference", "separated", "bleed_12db", "mixture")
BLEED_DB = -12.0


def classify_track(
    model, meta: dict, references: dict[str, np.ndarray], separated: dict[str, np.ndarray], mixture: np.ndarray, device
) -> dict:
    """Per-class probabilities for one track, all audio at the classifier's rate.

    Returns ``{condition: {class: (n_windows, n_classes) probabilities}}``.
    """
    mfcc_cfg, win_cfg = meta["mfcc_config"], meta["window_config"]
    out: dict[str, dict[str, np.ndarray]] = {c: {} for c in CONDITIONS}
    for cls in meta["classes"]:
        if cls not in references:
            continue
        starts = active_starts(references[cls], mfcc_cfg.sample_rate, win_cfg)
        if not len(starts):
            continue
        # Sanidha's mixture is the sum of the stems, so mixture - reference is
        # exactly the other instruments.
        others = mixture - references[cls]
        inputs = {
            "reference": references[cls],
            "separated": separated[cls],
            "bleed_12db": references[cls] + 10 ** (BLEED_DB / 20) * others,
            "mixture": mixture,
        }
        for cond, audio in inputs.items():
            out[cond][cls] = predict_proba(model, mfcc_windows(audio, starts, mfcc_cfg, win_cfg.window_seconds), device)
    return out


def summarize(
    probs: dict[str, dict[str, list[np.ndarray]]], labels: list[str], rows: list[str] | None = None, conditions=CONDITIONS
) -> dict:
    """Confusion matrices and accuracies from ``{condition: {class: [probs per track]}}``.

    ``labels`` are the classifier's outputs (the matrix columns); ``rows`` the
    classes actually scored, default all of them. Matrices are
    ``(len(rows), len(labels))``.
    """
    rows = list(rows or labels)
    summary = {}
    for cond in conditions:
        confusion = np.zeros((len(rows), len(labels)), dtype=int)
        mean_prob = np.zeros((len(rows), len(labels)))
        per_class = {}
        for r, cls in enumerate(rows):
            chunks = probs.get(cond, {}).get(cls, [])
            if not chunks:
                continue
            p = np.concatenate(chunks)
            i = labels.index(cls)
            confusion[r] = np.bincount(p.argmax(axis=1), minlength=len(labels))
            mean_prob[r] = p.mean(axis=0)
            per_class[cls] = {
                "n_windows": int(len(p)),
                "accuracy": float(np.mean(p.argmax(axis=1) == i)),
                "true_class_prob": float(p[:, i].mean()),
                "bleed_mass": float(1.0 - p[:, i].mean()),
            }
        n = confusion.sum()
        correct = sum(confusion[r, labels.index(cls)] for r, cls in enumerate(rows))
        summary[cond] = {
            "per_class": per_class,
            # Balanced: mean per-class recall. Ghatam has far fewer windows
            # than vocal, and plain accuracy would hide it.
            "balanced_accuracy": float(np.mean([v["accuracy"] for v in per_class.values()])) if per_class else float("nan"),
            "accuracy": float(correct / n) if n else float("nan"),
            "mean_bleed_mass": float(np.mean([v["bleed_mass"] for v in per_class.values()])) if per_class else float("nan"),
            "confusion": confusion.tolist(),
            "mean_prob": mean_prob.tolist(),
        }
    return summary


def classify_clips(model, meta: dict, clips: list[tuple[str, str]], device) -> dict[str, list[np.ndarray]]:
    """Every active window of each labelled clip -> ``{class: [probabilities per clip]}``.

    The in-domain check: IMIDataset test clips, never trained on.
    """
    from ..data.base import load_audio

    mfcc_cfg, win_cfg = meta["mfcc_config"], meta["window_config"]
    win = int(win_cfg.window_seconds * mfcc_cfg.sample_rate)
    out: dict[str, list[np.ndarray]] = {}
    for cls, path in clips:
        audio = load_audio(path, mfcc_cfg.sample_rate)
        if len(audio) < win:
            audio = np.pad(audio, (0, win - len(audio)))
        starts = active_starts(audio, mfcc_cfg.sample_rate, win_cfg)
        if len(starts):
            out.setdefault(cls, []).append(
                predict_proba(model, mfcc_windows(audio, starts, mfcc_cfg, win_cfg.window_seconds), device)
            )
    return out
