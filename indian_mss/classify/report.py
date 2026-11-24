"""Figures for the Milestone 4 evaluation report.

Confusion matrices are row-normalized (each row: where that instrument's
windows went) and drawn on a single-hue sequential ramp, light = 0 to dark =
1, with the share printed in each cell so the figure reads without colour.
Rows are the instruments scored; columns every class the classifier can
answer with, so a mridangam stem called "tabla" is visible as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluate import CONDITIONS

__all__ = ["plot_confusion", "write_report"]

# Sequential blue, steps 100 -> 700 of the reference palette.
_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_INK, _INK_MUTED, _SURFACE = "#1f1f1d", "#6b6a65", "#ffffff"

_TITLES = {
    "reference": "Ground-truth stems",
    "separated": "Separated stems",
    "bleed_12db": "Control: stem + others at -12 dB",
    "mixture": "Raw mixture",
    "close_mic": "Live close mic (its own bleed)",
    "separated_master": "Separated from the released master",
}


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", _BLUES)


def plot_confusion(ax, s: dict, rows: list[str], labels: list[str], title: str) -> None:
    conf = np.asarray(s["confusion"], dtype=float)
    shares = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    ax.imshow(shares, cmap=_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(labels)):
            v = shares[i, j]
            if v >= 0.005:  # blank cells read faster than a wall of 0%
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color=_SURFACE if v > 0.5 else _INK)  # fmt: skip
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", color=_INK, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, color=_INK, fontsize=9)
    ax.set_xlabel("predicted", color=_INK_MUTED, fontsize=9)
    ax.set_title(f"{title}\nbalanced acc. {s['balanced_accuracy']:.1%}", color=_INK, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def write_report(result: dict, path: str | Path) -> None:
    """PDF: Concert03 confusion matrices, IMIDataset test matrix, per-class table."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = result["classes"]
    labels = result.get("labels", rows)
    summary = result["summary"]
    conditions = list(result.get("conditions", CONDITIONS))
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, (len(conditions) + 1) // 2, figsize=(14, 7.5), constrained_layout=True, squeeze=False)
        for ax, cond in zip(axes.flat, conditions):
            plot_confusion(ax, summary[cond], rows, labels, _TITLES.get(cond, cond))
        for ax in list(axes.flat)[len(conditions):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("instrument actually playing", color=_INK_MUTED, fontsize=9)
        fig.suptitle(
            result.get("title")
            or f"Timbre classifier on held-out {', '.join(result['test_concerts'])}: windows where each "
            f"instrument is playing (separator resynthesis: {result['resynthesis_method']})",
            color=_INK,
        )
        pdf.savefig(fig)
        fig.savefig(Path(path).with_suffix(".png"), dpi=150)
        plt.close(fig)

        if result.get("imi_test"):
            imi = result["imi_test"]
            imi_rows = list(imi["per_class"])
            fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
            plot_confusion(ax, imi, imi_rows, labels, "Held-out IMIDataset clips (in-domain check)")
            ax.set_ylabel("labelled instrument", color=_INK_MUTED, fontsize=9)
            pdf.savefig(fig)
            fig.savefig(Path(path).with_name(Path(path).stem + "_imi.png"), dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.axis("off")
        header = ["instrument", "windows"] + [f"{c} acc." for c in conditions] + [f"{c} bleed" for c in conditions]
        table_rows = []
        for cls in rows:
            ref = summary[conditions[0]]["per_class"].get(cls)
            if not ref:
                continue
            table_rows.append(
                [cls, str(ref["n_windows"])]
                + [f"{summary[c]['per_class'][cls]['accuracy']:.1%}" for c in conditions]
                + [f"{summary[c]['per_class'][cls]['bleed_mass']:.2f}" for c in conditions]
            )
        table_rows.append(
            ["balanced", ""]
            + [f"{summary[c]['balanced_accuracy']:.1%}" for c in conditions]
            + [f"{summary[c]['mean_bleed_mass']:.2f}" for c in conditions]
        )
        table = ax.table(cellText=table_rows, colLabels=header, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        ax.set_title(
            "acc. = share of windows given the right instrument; bleed = mean probability on any other class",
            color=_INK_MUTED, fontsize=10,
        )  # fmt: skip
        pdf.savefig(fig)
        plt.close(fig)
