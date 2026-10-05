"""Random-crop training examples from Sanidha, and the train/validation split.

Split by *track*, never by segment: two crops of one recording share the
room, the microphones and the performers, so a segment-level split leaks and
the validation SDR stops measuring generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.base import load_segment
from ..data.sanidha import Sanidha
from .augment import augment_stems

__all__ = ["TrackRef", "SegmentDataset", "split_tracks", "track_refs"]


@dataclass(frozen=True)
class TrackRef:
    """What a worker needs to read crops from one track -- cheap to pickle."""

    track_id: str
    stems: dict[str, tuple[Path, ...]]
    duration: float


def split_tracks(dataset: Sanidha, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic track-level split. At least one track each way when possible."""
    ids = sorted(dataset.track_ids())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    if len(ids) >= 2:
        n_val = min(max(n_val, 1), len(ids) - 1)
    else:
        n_val = 0
    return sorted(ids[n_val:]), sorted(ids[:n_val])


class SegmentDataset(Dataset):
    """``samples_per_epoch`` random crops, re-drawn every epoch.

    Returns ``(mixture (T,), stems (S, T))`` as float32 tensors, with the
    mixture always equal to the sum of the (augmented) stems.
    """

    def __init__(
        self,
        tracks: list[TrackRef],
        sources: list[str],
        sample_rate: int,
        segment_seconds: float,
        samples_per_epoch: int,
        augment: dict | None = None,
        seed: int = 0,
    ):
        usable = [t for t in tracks if t.duration >= segment_seconds]
        if not usable:
            raise ValueError(f"no training track is at least {segment_seconds} s long")
        self.tracks = usable
        self.sources = list(sources)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment or {}
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _crop(self, track: TrackRef, stem: str, start: float) -> np.ndarray:
        n = int(round(self.segment_seconds * self.sample_rate))
        paths = track.stems.get(stem)
        if not paths:
            return np.zeros(n, dtype=np.float32)  # instrument absent from this track
        return np.sum(
            [load_segment(p, self.sample_rate, start, self.segment_seconds) for p in paths], axis=0
        ).astype(np.float32)

    def _draw(self, rng: np.random.Generator) -> np.ndarray:
        shuffle = self.augment.get("shuffle_stems_across_tracks", False) and rng.random() < float(
            self.augment.get("shuffle_prob", 0.5)
        )
        stems = []
        if shuffle:
            # Each instrument from an unrelated track and time: new
            # combinations the corpus never recorded, at the cost of musical
            # coherence -- hence only part of the time.
            for stem in self.sources:
                candidates = [t for t in self.tracks if stem in t.stems] or self.tracks
                track = candidates[rng.integers(len(candidates))]
                start = rng.uniform(0.0, track.duration - self.segment_seconds)
                stems.append(self._crop(track, stem, start))
        else:
            track = self.tracks[rng.integers(len(self.tracks))]
            start = rng.uniform(0.0, track.duration - self.segment_seconds)
            stems = [self._crop(track, stem, start) for stem in self.sources]
        return np.stack(stems)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, index))
        # Skip near-silent crops (a pause between items): they teach nothing
        # and their normalized loss is dominated by noise.
        for _ in range(10):
            stems = self._draw(rng)
            if np.sqrt(np.mean(stems.sum(axis=0) ** 2)) > 1e-4:
                break
        stems = augment_stems(stems, rng, self.sample_rate, self.augment)
        stems = stems.astype(np.float32)
        return torch.from_numpy(stems.sum(axis=0)), torch.from_numpy(stems)


def track_refs(dataset: Sanidha, track_ids: list[str]) -> list[TrackRef]:
    return [
        TrackRef(
            track_id=tid,
            stems={k: tuple(v) for k, v in dataset.stem_paths(tid).items()},
            duration=dataset.duration(tid),
        )
        for tid in track_ids
    ]
