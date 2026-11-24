"""Choosing "instrument is playing" windows on live recordings, without a clean reference.

On Sanidha the test windows come from the clean close-mic stem
(``data.active_starts``). Saraga has no clean stem: every close mic holds its
own instrument *plus* bleed from the others, so a silent instrument's mic
still looks loud wherever the ensemble is. Picking windows by that mic's level
alone would score bleed-only windows as "violin playing", which is exactly the
mistake the Milestone 4 measurement must not make.

**The rule: bleed-floor ratio.** For mic ``x`` and window ``w``, take

    r_x(w) = 10 log10( E_x(w) / sum_{j != x} E_j(w) )

the energy on ``x``'s mic over the energy on all the other mics. When ``x`` is
silent its mic holds only leakage, so ``r_x`` falls to a floor set by how much
of the room reaches that mic. When ``x`` plays, ``r_x`` rises above it. The
floor is the ``floor_percentile``-th percentile of ``r_x`` over every window of
one *concert* (one mic placement, and enough material that every instrument
rests somewhere), and a window counts as ``x`` playing when

* ``r_x(w) >= floor + margin_db``, and
* ``x``'s mic also passes the usual absolute and relative silence floors.

Pooling the floor per concert rather than per track matters: in a short
excerpt where the mridangam never stops, a per-track floor would sit on the
playing mode itself and reject almost everything.

The rule was validated on Sanidha with simulated bleed: every mic pair
leaked through its own random room response at -20 to -8 dB, and again at
-14 to -2 dB. With the defaults (p5 floor, 10 dB margin), fewer than 5% of the
vocal, violin and mridangam windows it keeps have less than half their mic's
energy from the instrument itself, at either bleed level. **Ghatam is not
reliable:** it is quiet enough that in 30-73% of its kept windows the bleed
outweighs it, so ghatam results from this rule are reported apart. The full
numbers are in ``deliverables/milestone_4_classifier/saraga/README.md``.

What it gives up: recall. Windows where ``x`` plays softly under a loud
ensemble are rejected, so the scored set leans towards passages where ``x``
stands out. Every condition is scored on the *same* windows, so the
comparison between them stays fair; the absolute accuracies are not
comparable with Sanidha's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import WindowConfig, window_rms_db

__all__ = ["LiveWindowConfig", "mic_ratios", "live_active_starts"]


@dataclass(frozen=True)
class LiveWindowConfig:
    """Parameters of the bleed-floor rule (see the module docstring)."""

    floor_percentile: float = 5.0
    margin_db: float = 10.0


def mic_ratios(
    mics: dict[str, np.ndarray], sample_rate: int, cfg: WindowConfig = WindowConfig()
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-window mic-to-rest ratio (dB) and silence-floor pass, for every mic.

    Returns ``(starts, ratio_db, loud_enough)``. All mics must be the same
    length; a recording needs at least two of them.
    """
    if len(mics) < 2:
        raise ValueError("the bleed-floor rule needs at least two microphones")
    names = list(mics)
    energy = {}
    starts = None
    for name in names:
        s, db = window_rms_db(mics[name], sample_rate, cfg)
        if starts is None:
            starts = s
        elif len(s) != len(starts):
            raise ValueError("all microphones must be the same length")
        energy[name] = 10.0 ** (db / 10.0)
    ratio, loud = {}, {}
    for name in names:
        rest = np.sum([energy[j] for j in names if j != name], axis=0)
        ratio[name] = 10.0 * np.log10(np.maximum(energy[name], 1e-20) / np.maximum(rest, 1e-20))
        db = 10.0 * np.log10(np.maximum(energy[name], 1e-20))
        floor = max(cfg.silence_db, float(np.percentile(db, 95)) + cfg.silence_rel_db) if len(db) else 0.0
        loud[name] = db >= floor
    return starts, ratio, loud


def live_active_starts(
    concert: list[dict[str, np.ndarray]],
    sample_rate: int,
    cfg: WindowConfig = WindowConfig(),
    live: LiveWindowConfig = LiveWindowConfig(),
) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
    """Playing windows for every recording of one concert.

    ``concert`` holds one ``{stem: close-mic audio}`` dict per recording; a
    stem may be absent from some of them. Returns per-recording
    ``{stem: start samples}`` and the per-stem bleed floor (dB) that was used.
    """
    per_track = [mic_ratios(mics, sample_rate, cfg) for mics in concert]
    stems = sorted({s for mics in concert for s in mics})
    floors = {}
    for stem in stems:
        pooled = [r[stem] for _, r, _ in per_track if stem in r and len(r[stem])]
        if pooled:
            floors[stem] = float(np.percentile(np.concatenate(pooled), live.floor_percentile))
    out = []
    for starts, ratio, loud in per_track:
        out.append(
            {
                stem: starts[loud[stem] & (ratio[stem] >= floors[stem] + live.margin_db)]
                for stem in ratio
                if stem in floors
            }
        )
    return out, floors
