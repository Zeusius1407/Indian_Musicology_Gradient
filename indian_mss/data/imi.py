"""IMIDataset loader -- labelled solo-instrument clips for the timbre classifier.

Layout of the release (six zip parts, each extracting to its own folder)::

    imi_root/
        Imid-01/dataset/<Class>_<N>song.wav
        ...
        Imid-06/dataset/...

Ten classes: Bass, drum, flute, guitar, piano, sarod, sitar, tabla, veena,
violin (normalised here to ``bass``, ``drums`` and lower case). Four things
about the release shape this loader:

* **Duplicates.** ``Imid-06`` is a byte-for-byte copy of ``Imid-02``, a few
  ``Bass_<N>song(1).wav`` files duplicate their originals, and a handful of
  clips appear under two numbers. Clips are de-duplicated by name *and* by
  content, keeping the first, so no audio can sit on both sides of a split.
* **Clips are consecutive excerpts.** For sitar, sarod and veena the sample
  step across the boundary from clip ``N`` to ``N + 1`` is no larger than a
  normal step inside a clip: they were cut from longer recordings in order.
  A random clip split would put adjacent seconds of one performance in train
  and test, so ``group`` buckets clip numbers into blocks and splits are drawn
  per block. Leakage is then limited to the block edges.
* **Sample rate follows class.** All piano clips are 16 kHz and all sitar,
  sarod and veena clips 48 kHz. After resampling, piano alone has nothing
  above 8 kHz -- a cue a classifier will happily learn instead of timbre. Cap
  the MFCC analysis at 8 kHz (``classifier.fmax``) when training on this set.
* **Short clips.** Violin and flute clips are mostly about one second, so
  clips shorter than a window are zero-padded rather than dropped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["IMIClip", "IMIDataset", "IMI_CLASSES"]

IMI_CLASSES = ("bass", "drums", "flute", "guitar", "piano", "sarod", "sitar", "tabla", "veena", "violin")
_ALIASES = {"drum": "drums"}
_NAME = re.compile(r"^(?P<cls>[A-Za-z]+)_(?P<num>\d+)song(?P<copy>\(\d+\))?\.wav$", re.IGNORECASE)


@dataclass(frozen=True)
class IMIClip:
    cls: str
    number: int
    path: Path

    def group(self, block: int) -> str:
        """Split unit: clips ``block`` numbers apart rarely share a recording."""
        return f"{self.cls}/{self.number // block}"


def _quick_key(path: Path) -> tuple[int, str]:
    """Size plus a hash of the first 64 kB: cheap, but clips of equal length that
    open on the same digital silence collide, so it only shortlists."""
    with path.open("rb") as f:
        head = f.read(65536)
    return path.stat().st_size, hashlib.md5(head).hexdigest()


def _full_key(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class IMIDataset:
    def __init__(self, root: str | Path, classes: tuple[str, ...] = IMI_CLASSES):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"IMIDataset root not found: {self.root}")
        self.clips = self._index(set(classes))
        if not self.clips:
            raise ValueError(
                f"no IMIDataset clips under {self.root}: expected <part>/dataset/<Class>_<N>song.wav "
                "(see the layout comment at the top of indian_mss/data/imi.py)"
            )

    def _index(self, wanted: set[str]) -> list[IMIClip]:
        seen_names: set[tuple[str, int]] = set()
        # Kept clips by quick key. A whole file is hashed only when its quick
        # key collides with a kept clip's, which is rare.
        kept: dict[tuple[int, str], list[Path]] = {}
        digests: dict[Path, str] = {}

        def digest(p: Path) -> str:
            if p not in digests:
                digests[p] = _full_key(p)
            return digests[p]

        clips = []
        # Sorted, and originals before "(1)" copies, so the kept copy is stable.
        paths = sorted(self.root.glob("*/dataset/*.wav"), key=lambda p: (p.parent.parent.name, "(" in p.name, p.name))
        for path in paths:
            m = _NAME.match(path.name)
            if not m:
                continue
            cls = m["cls"].lower()
            cls = _ALIASES.get(cls, cls)
            if cls not in wanted:
                continue
            name = (cls, int(m["num"]))
            if name in seen_names:
                continue
            key = _quick_key(path)
            if any(digest(path) == digest(other) for other in kept.get(key, [])):
                continue
            kept.setdefault(key, []).append(path)
            seen_names.add(name)
            clips.append(IMIClip(cls, name[1], path))
        return clips

    def classes(self) -> list[str]:
        return sorted({c.cls for c in self.clips})

    def split(
        self, block: int, fractions: tuple[float, float] = (0.15, 0.15), seed: int = 0
    ) -> tuple[list[IMIClip], list[IMIClip], list[IMIClip]]:
        """``(train, val, test)`` by block of clip numbers, stratified by class.

        ``fractions`` are the val and test shares of each class's blocks; every
        class with three or more blocks gets at least one block in each.
        """
        rng = np.random.default_rng(seed)
        train, val, test = [], [], []
        for cls in self.classes():
            clips = [c for c in self.clips if c.cls == cls]
            groups = sorted({c.group(block) for c in clips})
            rng.shuffle(groups)
            n_val = max(1, round(len(groups) * fractions[0])) if len(groups) >= 3 else 0
            n_test = max(1, round(len(groups) * fractions[1])) if len(groups) >= 3 else 0
            val_g, test_g = set(groups[:n_val]), set(groups[n_val : n_val + n_test])
            for c in clips:
                g = c.group(block)
                (val if g in val_g else test if g in test_g else train).append(c)
        return train, val, test
