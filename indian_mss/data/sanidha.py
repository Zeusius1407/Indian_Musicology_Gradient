"""Sanidha loader -- studio Carnatic multitrack, the training corpus.

Expected layout (check this against your download before a training run)::

    sanidha_root/
        <track_id>/
            <anything>vocal<anything>.wav
            <anything>violin<anything>.wav
            <anything>mridangam<anything>.wav
            <anything>ghatam<anything>.wav
            mix.wav            # optional; also accepted: mixture.*, *_mix.*
        <track_id>/
            ...

One directory per track; stem files are matched by the stem name appearing in
the filename (case-insensitive), so ``Vocal_Close.wav`` and ``01_vocal.flac``
both count. Tracks with no mixdown get a synthetic one -- the sum of the
stems -- and are flagged ``mixture_is_synthetic`` so evaluation can say so.

If the release you downloaded is laid out differently, ``Sanidha`` raises a
``ValueError`` naming the folder it could not parse rather than silently
returning tracks with missing stems.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AUDIO_EXTENSIONS, TrackItem, load_audio

__all__ = ["Sanidha", "SANIDHA_STEMS"]

SANIDHA_STEMS = ("vocal", "violin", "mridangam", "ghatam")
_MIX_NAMES = ("mix", "mixture", "mixdown")


def _is_mixture(path: Path) -> bool:
    name = path.stem.lower()
    return name in _MIX_NAMES or any(name.endswith(f"_{m}") for m in _MIX_NAMES)


class Sanidha:
    def __init__(self, root: str | Path, sample_rate: int, stems: tuple[str, ...] = SANIDHA_STEMS):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.stems = stems
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Sanidha root not found: {self.root} -- set data.sanidha_root in configs/default.yaml"
            )
        self._tracks = self._index()
        if not self._tracks:
            raise ValueError(
                f"no Sanidha tracks found under {self.root}: expected one sub-directory per track "
                f"containing audio files named after the stems {list(stems)} "
                "(see the layout comment at the top of indian_mss/data/sanidha.py)"
            )

    def _index(self) -> dict[str, dict]:
        tracks: dict[str, dict] = {}
        for track_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = [f for f in track_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
            if not files:
                continue
            mixture = next((f for f in files if _is_mixture(f)), None)
            stem_files: dict[str, Path] = {}
            for stem in self.stems:
                matches = sorted(f for f in files if stem in f.stem.lower() and f != mixture)
                if len(matches) > 1:
                    # Multi-mic tracks (e.g. two mridangam mics): sum them as
                    # one stem rather than picking one arbitrarily.
                    stem_files[stem] = matches  # type: ignore[assignment]
                elif matches:
                    stem_files[stem] = matches[0]
            if not stem_files:
                raise ValueError(
                    f"{track_dir} contains audio but no file names any of the stems "
                    f"{list(self.stems)} -- the layout differs from what this loader expects"
                )
            tracks[track_dir.name] = {"mixture": mixture, "stems": stem_files}
        return tracks

    def track_ids(self) -> list[str]:
        return list(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def load(self, track_id: str) -> TrackItem:
        entry = self._tracks[track_id]

        stems: dict[str, np.ndarray] = {}
        for stem, src in entry["stems"].items():
            paths = src if isinstance(src, list) else [src]
            loaded = [load_audio(p, self.sample_rate) for p in paths]
            n = min(len(a) for a in loaded)
            stems[stem] = np.sum([a[:n] for a in loaded], axis=0).astype(np.float32)

        if entry["mixture"] is not None:
            mixture = load_audio(entry["mixture"], self.sample_rate)
            synthetic = False
        else:
            mixture = None
            synthetic = True

        # Trim everything to a common length so references and estimates stay
        # sample-aligned -- stems from one session can differ by a few samples.
        lengths = [len(a) for a in stems.values()] + ([len(mixture)] if mixture is not None else [])
        n = min(lengths)
        stems = {k: v[:n] for k, v in stems.items()}
        if mixture is None:
            mixture = np.sum(list(stems.values()), axis=0).astype(np.float32)
        else:
            mixture = mixture[:n]

        return TrackItem(
            track_id=track_id,
            mixture=mixture,
            stems=stems,
            sample_rate=self.sample_rate,
            mixture_is_synthetic=synthetic,
        )
