#!/usr/bin/env python3
"""Milestone 4 on live audio: classify Saraga's close mics, separated stems and mixture.

Example::

    python scripts/evaluate_classifier_saraga.py --classifier checkpoints/classifier/classifier.pt \\
        --separator deliverables/milestone_2_model/checkpoint/best.pt \\
        --outdir deliverables/milestone_4_classifier/saraga

The Sanidha test (``evaluate_classifier.py``) compares separated stems with
clean references and a *simulated* -12 dB bleed control. Saraga has no clean
reference, but it has something better than the control: real close mics
that carry real live bleed. So for each instrument, on the windows where it is
playing (chosen by ``classify.live``, which needs no clean stem), the same
windows are classified as

``close_mic``         the instrument's own live close mic, bleed and all
``separated``         the separator's stem, from the sum of the close mics
``separated_master``  the separator's stem, from the released stereo master
``mixture``           the sum of the close mics: the floor

If separation removes live bleed, ``separated`` scores *above* ``close_mic``.
The master is not sample-aligned with the mics (see the Milestone 2 Saraga
notes), which ruins SDR but not this test: a 1 s window does not care about a
few milliseconds of offset.

Scored on the central ``--seconds`` of each recording (default 60, the same
excerpt as the Saraga SDR evaluation), so per-track results can be joined with
``saraga_eval.json``. Writes ``classifier_saraga.json`` and a PDF report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from indian_mss.config import DEFAULT_CONFIG_PATH, load_config

CONDITIONS = ("close_mic", "separated", "separated_master", "mixture")
CORE_STEMS = ("vocal", "violin", "mridangam")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--separator", type=str, required=True, help="Milestone 2 checkpoint")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--saraga-root", type=str, default=None, help="overrides data.saraga_root")
    p.add_argument("--seconds", type=float, default=60.0, help="central excerpt per track; 0 = whole track")
    p.add_argument("--floor-percentile", type=float, default=None, help="overrides classifier.live.floor_percentile")
    p.add_argument("--margin-db", type=float, default=None, help="overrides classifier.live.margin_db")
    p.add_argument("--limit-concerts", type=int, default=0, help="only the first N concerts")
    p.add_argument("--outdir", type=str, default="out/classifier_saraga")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    import librosa

    from indian_mss.classify.data import mfcc_windows
    from indian_mss.classify.evaluate import summarize
    from indian_mss.classify.live import LiveWindowConfig, live_active_starts
    from indian_mss.classify.report import write_report
    from indian_mss.classify.train import load_classifier, predict_proba
    from indian_mss.cli.evaluate import _crop_centre
    from indian_mss.cli.evaluate_classifier import separate_blocks
    from indian_mss.data.saraga import Saraga
    from indian_mss.infer.resynth import ResynthConfig
    from indian_mss.models.separator import load_separator
    from indian_mss.train.loop import pick_device

    device = pick_device(args.device)
    clf, cmeta = load_classifier(args.classifier, device)
    sep, smeta = load_separator(args.separator, device)
    rcfg = ResynthConfig.from_config(cfg)
    if rcfg.method == "vocoder":
        raise ValueError("set resynthesis.method to an ISTFT method; the vocoder is not used for this test")

    live_cfg = cfg.get("classifier", {}).get("live", {})
    live = LiveWindowConfig(
        floor_percentile=args.floor_percentile if args.floor_percentile is not None else live_cfg.get("floor_percentile", 5.0),
        margin_db=args.margin_db if args.margin_db is not None else live_cfg.get("margin_db", 10.0),
    )
    labels, sources = cmeta["classes"], list(smeta["sources"])
    classes = [s for s in sources if s in labels]
    mfcc_cfg, win_cfg = cmeta["mfcc_config"], cmeta["window_config"]
    sr_sep, sr_clf = smeta["stft_config"].sample_rate, mfcc_cfg.sample_rate

    def down(x: np.ndarray) -> np.ndarray:
        return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=sr_sep, target_sr=sr_clf).astype(np.float32)

    root = args.saraga_root or cfg.get("data", {}).get("saraga_root", "data/saraga")
    dataset = Saraga(root, sr_sep, stems=tuple(sources), mixture="mixdown")
    concerts: dict[str, list[str]] = defaultdict(list)
    for tid in dataset.track_ids():
        concerts[tid.split("/")[1]].append(tid)
    names = sorted(concerts)[: args.limit_concerts] if args.limit_concerts else sorted(concerts)
    n_tracks = sum(len(concerts[c]) for c in names)
    print(
        f"{n_tracks} Saraga tracks in {len(names)} concerts, central {args.seconds:g} s each; "
        f"bleed floor = p{live.floor_percentile:g} per concert, margin {live.margin_db:g} dB"
    )

    pooled = {c: {} for c in CONDITIONS}
    per_track, floors_by_concert = [], {}
    done = 0
    for concert in names:
        # Pass 1: load the concert's excerpts and choose windows from the
        # close mics alone, with one bleed floor per mic for the concert.
        # Only the excerpts are kept: concert recordings run to tens of
        # minutes across five or six mics, and the crops are views, so they
        # are copied to let the full-length arrays go.
        loaded = []
        for tid in concerts[concert]:
            item = dataset.load(tid)
            master, stems, _ = _crop_centre(item, args.seconds)
            master = None if item.mixture_is_synthetic else master.copy()
            stems = {s: v.copy() for s, v in stems.items()}
            del item
            mics = {s: down(stems[s]) for s in classes if s in stems}
            loaded.append((tid, master, stems, mics))
        windows, floors = live_active_starts([m for *_, m in loaded], sr_clf, win_cfg, live)
        floors_by_concert[concert] = floors

        # Pass 2: separate, then classify every condition on those windows.
        for (tid, master, stems, mics), starts_by_stem in zip(loaded, windows):
            done += 1
            mix = np.sum(list(stems.values()), axis=0).astype(np.float32)
            est = separate_blocks(sep, smeta, mix, rcfg, 60.0, device)
            inputs = {"mixture": down(mix)}
            est_master = None
            if master is not None:
                est_master = separate_blocks(sep, smeta, master, rcfg, 60.0, device)
            track_probs = {c: {} for c in CONDITIONS}
            for cls in classes:
                starts = starts_by_stem.get(cls)
                if starts is None or not len(starts):
                    continue
                audio = {
                    "close_mic": mics[cls],
                    "separated": down(est[sources.index(cls)]),
                    "mixture": inputs["mixture"],
                }
                if est_master is not None:
                    audio["separated_master"] = down(est_master[sources.index(cls)])
                for cond, x in audio.items():
                    p = predict_proba(clf, mfcc_windows(x, starts, mfcc_cfg, win_cfg.window_seconds), device)
                    track_probs[cond][cls] = p
                    pooled[cond].setdefault(cls, []).append(p)
            track_summary = summarize({c: {k: [v] for k, v in track_probs[c].items()} for c in CONDITIONS}, labels, classes, CONDITIONS)
            per_track.append(
                {
                    "track_id": tid,
                    "concert": concert,
                    "n_windows": {c: int(len(s)) for c, s in starts_by_stem.items()},
                    "summary": {
                        c: {k: track_summary[c][k] for k in ("per_class", "balanced_accuracy", "mean_bleed_mass")}
                        for c in CONDITIONS
                    },
                }
            )
            print(
                f"[{done}/{n_tracks}] {tid.split('/')[-1][:40]:40s} windows "
                + " ".join(f"{c[:3]}={len(s)}" for c, s in starts_by_stem.items())
                + "  "
                + "  ".join(f"{c} {track_summary[c]['balanced_accuracy']:.0%}" for c in CONDITIONS),
                flush=True,
            )

    summary = summarize(pooled, labels, classes, CONDITIONS)
    # The window rule cannot isolate ghatam (see classify/live.py), so the
    # headline balanced accuracy is over the three stems it can.
    core = [c for c in CORE_STEMS if c in classes]
    core_summary = summarize({c: {k: pooled[c][k] for k in core if k in pooled[c]} for c in CONDITIONS}, labels, core, CONDITIONS)
    print(f"\n=== timbre classifier on Saraga, {n_tracks} tracks ===")
    print(f"  {'':10s} {'windows':>8s} " + " ".join(f"{c:>17s}" for c in CONDITIONS) + "   (accuracy; bleed mass)")
    for cls in classes:
        if cls in summary["close_mic"]["per_class"]:
            print(
                f"  {cls:10s} {summary['close_mic']['per_class'][cls]['n_windows']:8d} "
                + " ".join(f"{summary[c]['per_class'][cls]['accuracy']:17.1%}" for c in CONDITIONS)
                + "   "
                + " ".join(f"{summary[c]['per_class'][cls]['bleed_mass']:.2f}" for c in CONDITIONS)
            )
    print(f"  {'balanced':10s} {'':8s} " + " ".join(f"{summary[c]['balanced_accuracy']:17.1%}" for c in CONDITIONS))
    print(f"  {'bal. 3-stem':10s} {'':8s} " + " ".join(f"{core_summary[c]['balanced_accuracy']:17.1%}" for c in CONDITIONS))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "classifier": args.classifier,
        "classifier_epoch": cmeta.get("epoch"),
        "separator": args.separator,
        "separator_epoch": smeta.get("epoch"),
        "resynthesis_method": rcfg.method,
        "conditions": list(CONDITIONS),
        "classes": classes,
        "labels": labels,
        "corpus": "saraga",
        "test_concerts": names,
        "seconds": args.seconds,
        "window_rule": {"rule": "bleed-floor ratio, classify/live.py", **live.__dict__},
        "bleed_floors_db": floors_by_concert,
        "title": f"Timbre classifier on Saraga (live, {n_tracks} recordings): windows where each instrument is "
        f"playing, chosen from the close mics (separator resynthesis: {rcfg.method})",
        "summary": summary,
        "balanced_accuracy_3stem": {c: core_summary[c]["balanced_accuracy"] for c in CONDITIONS},
        "per_track": per_track,
    }
    (outdir / "classifier_saraga.json").write_text(json.dumps(result, indent=2))
    write_report(result, outdir / "classifier_saraga_report.pdf")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
