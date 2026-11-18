"""Saraga loader -- live Carnatic concerts with real bleed, the test corpus.

This is the evaluation-only half of the data story. Sanidha is recorded in
isolation booths, so its masks learn the instrument; Saraga is recorded live
off close microphones that all hear the whole stage, so the only honest
generalization number this project can quote comes from here. Nothing in
``train/`` should ever point at this loader -- training on real bleed teaches
the model that leakage belongs in the target.

Use the **Carnatic** release (``saraga1.5_carnatic.zip``). Of its 249
recordings, 168 ship multitrack stems:

====================================  ==========================
file                                  maps to
====================================  ==========================
``<piece>.multitrack-vocal.mp3``      ``vocal``
``<piece>.multitrack-vocal-s.mp3``    ``vocal``   (second vocalist; summed)
``<piece>.multitrack-violin.mp3``     ``violin``
``<piece>.multitrack-mridangam-left.mp3``   ``mridangam`` (summed)
``<piece>.multitrack-mridangam-right.mp3``  ``mridangam`` (summed)
``<piece>.multitrack-ghatam.mp3``     ``ghatam``  (only ~46 recordings)
``<piece>.mp3.mp3``                   the released stereo mixdown
====================================  ==========================

Expected layout -- one directory per concert, one per piece inside it::

    saraga_root/
        <concert>/
            <piece>/
                <piece>.mp3.mp3
                <piece>.multitrack-vocal.mp3
                <piece>.multitrack-violin.mp3
                <piece>.multitrack-mridangam-left.mp3
                <piece>.multitrack-mridangam-right.mp3

Track ids are ``"<concert>/<piece>"``. Piece names repeat across concerts (the
same kriti played by different artists), so the concert has to stay in the key;
the slash is only ever a dictionary key and a field in the result JSON, never a
path this project writes to.

Stems are matched by *exact* ``multitrack-*`` tag, not by substring. Substring
matching -- the scheme ``sanidha.py`` uses, which is right for its filenames --
would fold ``.multitrack-vocal-s.mp3`` into ``vocal`` by accident rather than by
decision, and would pick up ``.pitch-vocal.txt`` siblings if the extension list
ever grows. AppleDouble twins (``__MACOSX/``, ``._*``) are skipped: the release
zip carries one for every stem and they would otherwise be indexed as audio.

Ghatam is present on only a subset, so a four-stem mean is computed over
roughly 46 tracks while a three-stem mean covers all 168. Report which.

**``mixture``: the one methodological choice in this file.** The released
mixdown is a separately engineered stereo master -- its own gain staging, EQ and
compression -- so it is *not* the sum of the close mics. BSS-Eval assumes the
references sum to the signal that was separated; score against the released
mixdown and a chunk of the penalty is that mismatch rather than anything the
model did.

* ``mixture="stems"`` (default) sums the close mics. Still the real live room,
  every bit of the acoustic bleed intact, but linear and consistent with the
  references. This is the number to put in the report, flagged
  ``mixture_is_synthetic=True`` -- here that flag means "linear remix of real
  bled microphones", which is a world away from Sanidha's "sum of clean studio
  stems".
* ``mixture="mixdown"`` uses the released master. Lower and not strictly
  comparable, but it is what a listener actually hears, so it is worth quoting
  alongside as the realism check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AUDIO_EXTENSIONS, TrackItem, audio_duration, load_audio

__all__ = ["Saraga", "SARAGA_STEMS", "MULTITRACK_TAGS"]

SARAGA_STEMS = ("vocal", "violin", "mridangam", "ghatam")

#: stem name -> the ``multitrack-*`` tags summed to build it.
MULTITRACK_TAGS: dict[str, tuple[str, ...]] = {
    "vocal": ("multitrack-vocal", "multitrack-vocal-s"),
    "violin": ("multitrack-violin",),
    "mridangam": ("multitrack-mridangam-left", "multitrack-mridangam-right"),
    "ghatam": ("multitrack-ghatam",),
}

_TAG_TO_STEM = {tag: stem for stem, tags in MULTITRACK_TAGS.items() for tag in tags}

MIXTURE_MODES = ("stems", "mixdown")


def _is_audio(path: Path) -> bool:
    # ``._Name.mp3`` is an AppleDouble resource fork, not audio: a few hundred
    # bytes that librosa would either fail on or decode as silence.
    return path.suffix.lower() in AUDIO_EXTENSIONS and not path.name.startswith("._")


def _tag(path: Path) -> str:
    """The ``multitrack-*`` tag in a Saraga filename, or ``""``.

    ``Kanukontini.multitrack-vocal.mp3`` -> ``multitrack-vocal``;
    ``Kanukontini.mp3.mp3`` -> ``mp3``; ``Kanukontini.mp3`` -> ``""``.
    """
    stem = path.stem
    return stem.rsplit(".", 1)[-1].lower() if "." in stem else ""


class Saraga:
    """Index of the Saraga Carnatic multitrack subset.

    Mirrors the ``Sanidha`` surface -- ``track_ids`` / ``stem_paths`` /
    ``duration`` / ``load`` -- so the evaluation code does not care which
    corpus it was handed.
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int,
        stems: tuple[str, ...] = SARAGA_STEMS,
        mixture: str = "stems",
    ):
        if mixture not in MIXTURE_MODES:
            raise ValueError(f"mixture must be one of {list(MIXTURE_MODES)}, got {mixture!r}")
        unknown = [s for s in stems if s not in MULTITRACK_TAGS]
        if unknown:
            raise ValueError(
                f"no Saraga multitrack files map to {unknown}; known stems are {list(MULTITRACK_TAGS)}"
            )
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.stems = tuple(stems)
        self.mixture = mixture
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Saraga root not found: {self.root} -- set data.saraga_root in configs/default.yaml"
            )
        self._tracks = self._index()
        if not self._tracks:
            raise ValueError(self._diagnose_empty())

    # ------------------------------------------------------------------ index

    def _audio_files(self) -> list[Path]:
        return [
            f
            for f in self.root.rglob("*")
            if f.is_file() and _is_audio(f) and "__MACOSX" not in f.parts
        ]

    def _index(self) -> dict[str, dict]:
        files = self._audio_files()
        # A "piece" is any directory holding at least one recognized multitrack
        # file. Deriving it from the files rather than from a fixed
        # concert/piece nesting depth means a re-organized or partially
        # extracted download still indexes.
        by_dir: dict[Path, list[Path]] = {}
        for f in files:
            by_dir.setdefault(f.parent, []).append(f)

        tracks: dict[str, dict] = {}
        for piece_dir in sorted(by_dir):
            siblings = by_dir[piece_dir]
            tagged = [f for f in siblings if _tag(f) in _TAG_TO_STEM]
            if not tagged:
                continue  # mixture-only recording: nothing to score against

            stem_files: dict[str, list[Path]] = {}
            for f in sorted(tagged):
                stem = _TAG_TO_STEM[_tag(f)]
                if stem in self.stems:
                    stem_files.setdefault(stem, []).append(f)
            if not stem_files:
                continue  # has multitracks, but none of the requested stems

            track_id = piece_dir.relative_to(self.root).as_posix()
            tracks[track_id] = {
                "dir": piece_dir,
                "mixdown": self._mixdown_file(siblings, piece_dir.name),
                "stems": stem_files,
            }
        return tracks

    @staticmethod
    def _mixdown_file(siblings: list[Path], piece: str) -> Path | None:
        """The released master among ``siblings``, if it shipped.

        Named ``<piece>.mp3.mp3`` in the 1.5 release -- the double extension is
        in the archive, not a mistake here -- so prefer a name that matches the
        directory before falling back to any untagged audio file.
        """
        candidates = [f for f in siblings if _tag(f) not in _TAG_TO_STEM]
        if not candidates:
            return None
        wanted = {piece.lower(), f"{piece.lower()}.mp3"}
        exact = sorted(f for f in candidates if f.stem.lower() in wanted)
        return (exact or sorted(candidates))[0]

    def _diagnose_empty(self) -> str:
        """Explain an empty index, naming the most likely cause.

        The common mistake is downloading ``saraga1.5_hindustani.zip``, which
        is mixture-only: it parses fine, yields no stems, and would otherwise
        fail with a generic "no tracks found" after a 4 GB download.
        """
        audio = self._audio_files()
        if not audio:
            return (
                f"no audio files under {self.root} -- expected the extracted "
                "saraga1.5_carnatic release (see the layout comment at the top of "
                "indian_mss/data/saraga.py)"
            )
        return (
            f"found {len(audio)} audio files under {self.root} but none named "
            f"'<piece>.multitrack-<instrument>.<ext>', so there are no reference stems "
            "to evaluate against. The Hindustani release (saraga1.5_hindustani.zip) is "
            "mixture-only and looks exactly like this; separation metrics need "
            "saraga1.5_carnatic.zip, whose 168 multitrack recordings carry "
            f"{sorted(_TAG_TO_STEM)}."
        )

    # ------------------------------------------------------------------- read

    def track_ids(self) -> list[str]:
        return list(self._tracks)

    def stem_paths(self, track_id: str) -> dict[str, list[Path]]:
        """Stem name -> the file(s) summed for it (mridangam L+R, both vocals)."""
        return {stem: list(paths) for stem, paths in self._tracks[track_id]["stems"].items()}

    def mixdown_path(self, track_id: str) -> Path | None:
        return self._tracks[track_id]["mixdown"]

    def duration(self, track_id: str) -> float:
        """Shortest stem duration in seconds -- the span where all stems exist."""
        return min(audio_duration(p) for paths in self.stem_paths(track_id).values() for p in paths)

    def __len__(self) -> int:
        return len(self._tracks)

    def load(self, track_id: str) -> TrackItem:
        entry = self._tracks[track_id]

        stems: dict[str, np.ndarray] = {}
        for stem, paths in entry["stems"].items():
            loaded = [load_audio(p, self.sample_rate) for p in paths]
            n = min(len(a) for a in loaded)
            stems[stem] = np.sum([a[:n] for a in loaded], axis=0).astype(np.float32)

        mixdown = entry["mixdown"]
        if self.mixture == "mixdown" and mixdown is not None:
            mixture = load_audio(mixdown, self.sample_rate)
            synthetic = False
        else:
            mixture = None
            synthetic = True

        # Common length, so references and estimates stay sample-aligned: the
        # close mics were cut by hand and the master is often a few hundred
        # samples longer or shorter.
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
            metadata={
                # Everything a reported number needs to be interpretable: which
                # mixture definition, which stems were actually present, and
                # how many microphones went into each.
                "corpus": "saraga",
                "mixture_mode": self.mixture if not synthetic else "stems",
                "mixdown_available": mixdown is not None,
                "stems_present": sorted(stems),
                "stems_missing": [s for s in self.stems if s not in stems],
                "mics_per_stem": {s: len(p) for s, p in self.stem_paths(track_id).items()},
            },
        )
