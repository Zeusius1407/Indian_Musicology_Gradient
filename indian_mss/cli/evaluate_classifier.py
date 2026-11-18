#!/usr/bin/env python3
"""Milestone 4 CLI: classify ground-truth, separated and mixed audio.

Example::

    python scripts/evaluate_classifier.py --classifier checkpoints/classifier/classifier.pt \\
        --separator deliverables/milestone_2_model/checkpoint/best.pt \\
        --sanidha-root data/sanidha_flat --outdir deliverables/milestone_4_classifier

Every recording of the classifier's test concerts (the separator's held-out
concert, so both networks see it for the first time) is separated with the
Milestone 2 model and resynthesized with ``resynthesis.method`` (Milestone 3),
then classified as described in ``indian_mss/classify/evaluate.py``.

If the classifier was trained with IMIDataset, its held-out IMIDataset clips
are classified too: the in-domain accuracy, for comparison.

Writes ``classifier_eval.json`` (confusion matrices, per-class accuracy and
bleed mass, per track), ``classifier_report.pdf`` and a ``.png`` of its
confusion matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--separator", type=str, required=True, help="Milestone 2 checkpoint")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--sanidha-root", type=str, default=None, help="overrides data.sanidha_root")
    p.add_argument("--seconds", type=float, default=0.0, help="central excerpt per track; 0 = whole track")
    p.add_argument("--block-seconds", type=float, default=60.0, help="separation block length (bounds memory)")
    p.add_argument("--limit", type=int, default=0, help="only the first N test tracks")
    p.add_argument("--outdir", type=str, default="out/classifier")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    return p


def separate_blocks(model, meta, mixture: np.ndarray, rcfg, block_seconds: float, device, vocoder=None) -> np.ndarray:
    """Separate + resynthesize a long track in blocks. Returns ``(S, T)``."""
    from indian_mss.infer.resynth import resynthesize
    from indian_mss.models.separator import predict_masks

    stft_cfg = meta["stft_config"]
    block = int(block_seconds * stft_cfg.sample_rate)
    out = []
    for a in range(0, len(mixture), block):
        chunk = mixture[a : a + block]
        masks, _ = predict_masks(model, chunk, stft_cfg, meta["segment_seconds"], device=device)
        out.append(resynthesize(masks, chunk, stft_cfg, rcfg, vocoder=vocoder))
    return np.concatenate(out, axis=1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    import librosa

    from indian_mss.classify.evaluate import CONDITIONS, classify_track, summarize
    from indian_mss.classify.report import write_report
    from indian_mss.classify.train import load_classifier
    from indian_mss.cli.evaluate import _crop_centre
    from indian_mss.data.sanidha import Sanidha
    from indian_mss.infer.resynth import ResynthConfig
    from indian_mss.models.separator import load_separator
    from indian_mss.train.loop import pick_device

    device = pick_device(args.device)
    clf, cmeta = load_classifier(args.classifier, device)
    sep, smeta = load_separator(args.separator, device)
    rcfg = ResynthConfig.from_config(cfg)
    vocoder = None
    if rcfg.method == "vocoder":
        from indian_mss.infer.vocoder import Vocoder

        vocoder = Vocoder(rcfg.vocoder, device=device)

    labels, sources = cmeta["classes"], list(smeta["sources"])
    # Score the separator's stems the classifier has a class for; the matrix
    # columns are every class it can answer with.
    classes = [s for s in sources if s in labels]
    if not classes:
        raise ValueError(f"classifier classes {labels} include none of the separator's stems {sources}")
    sr_sep, sr_clf = smeta["stft_config"].sample_rate, cmeta["mfcc_config"].sample_rate

    root = args.sanidha_root or cfg.get("data", {}).get("sanidha_root", "data/sanidha")
    dataset = Sanidha(root, sr_sep, stems=tuple(sources))
    test_ids = cmeta["test_tracks"][: args.limit] if args.limit else cmeta["test_tracks"]
    if overlap := set(test_ids) & set(smeta.get("train_tracks") or []):
        raise ValueError(f"test tracks {sorted(overlap)} were used to train the separator")

    def down(x: np.ndarray) -> np.ndarray:
        return librosa.resample(x, orig_sr=sr_sep, target_sr=sr_clf).astype(np.float32)

    pooled = {c: {} for c in CONDITIONS}
    per_track = []
    for i, tid in enumerate(test_ids, 1):
        mixture, stems, _ = _crop_centre(dataset.load(tid), args.seconds)
        est = separate_blocks(sep, smeta, mixture, rcfg, args.block_seconds, device, vocoder)
        refs = {c: down(stems[c]) for c in classes if c in stems}
        separated = {c: down(est[sources.index(c)]) for c in classes}
        probs = classify_track(clf, cmeta, refs, separated, down(mixture), device)
        for cond, by_class in probs.items():
            for cls, p in by_class.items():
                pooled[cond].setdefault(cls, []).append(p)
        track_summary = summarize({c: {k: [v] for k, v in probs[c].items()} for c in CONDITIONS}, labels, classes)
        per_track.append({"track_id": tid, "summary": track_summary})
        print(
            f"[{i}/{len(test_ids)}] {tid}  "
            + "  ".join(f"{c} {track_summary[c]['balanced_accuracy']:.1%}" for c in CONDITIONS),
            flush=True,
        )

    summary = summarize(pooled, labels, classes)
    print(f"\n=== timbre classifier, {len(test_ids)} test tracks ===")
    print(f"  {'':10s} " + " ".join(f"{c:>11s}" for c in CONDITIONS) + "   (accuracy; bleed mass)")
    for cls in classes:
        if cls in summary["reference"]["per_class"]:
            print(
                f"  {cls:10s} "
                + " ".join(f"{summary[c]['per_class'][cls]['accuracy']:11.1%}" for c in CONDITIONS)
                + "   "
                + " ".join(f"{summary[c]['per_class'][cls]['bleed_mass']:.2f}" for c in CONDITIONS)
            )
    print(f"  {'balanced':10s} " + " ".join(f"{summary[c]['balanced_accuracy']:11.1%}" for c in CONDITIONS))

    imi_summary = None
    if cmeta.get("imi_test_clips"):
        from indian_mss.classify.evaluate import classify_clips

        clips = cmeta["imi_test_clips"]
        print(f"\nclassifying {len(clips)} held-out IMIDataset clips")
        imi_rows = sorted({cls for cls, _ in clips})
        imi_summary = summarize({"imi_test": classify_clips(clf, cmeta, clips, device)}, labels, imi_rows, ("imi_test",))
        imi_summary = imi_summary["imi_test"]
        for cls, v in imi_summary["per_class"].items():
            print(f"  {cls:10s} {v['accuracy']:6.1%}  ({v['n_windows']} windows)")
        print(f"  {'balanced':10s} {imi_summary['balanced_accuracy']:6.1%}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "classifier": args.classifier,
        "classifier_epoch": cmeta.get("epoch"),
        "classifier_val_balanced_acc": cmeta.get("val_balanced_acc"),
        "separator": args.separator,
        "separator_epoch": smeta.get("epoch"),
        "resynthesis_method": rcfg.method,
        "classes": classes,
        "labels": labels,
        "test_concerts": sorted({t.split("_", 1)[0] for t in test_ids}),
        "test_tracks": test_ids,
        "seconds": args.seconds,
        "summary": summary,
        "imi_test": imi_summary,
        "per_track": per_track,
    }
    (outdir / "classifier_eval.json").write_text(json.dumps(result, indent=2))
    write_report(result, outdir / "classifier_report.pdf")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
