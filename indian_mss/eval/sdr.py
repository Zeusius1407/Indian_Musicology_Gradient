"""Separation quality metrics: SDR, SIR, SAR.

The BSS-Eval decomposition projects each estimate onto three orthogonal
subspaces:

* the target itself                      -> everything else is error
* the span of all sources                -> leakage from other instruments
* everything                             -> artefacts the algorithm invented

From those projections:

    SDR = 10 log10( ||s_target||^2 / ||e_interf + e_artif||^2 )
    SIR = 10 log10( ||s_target||^2 / ||e_interf||^2 )
    SAR = 10 log10( ||s_target + e_interf||^2 / ||e_artif||^2 )

Report all three. SDR alone hides the trade-off that matters musically: an
aggressive mask raises SIR (less mridangam in the violin) while lowering SAR
(more musical noise), and the two can cancel out to an unchanged SDR while
sounding much worse.

For the final report prefer ``museval`` (the reference implementation, with
the windowed BSS-Eval v4 that separation papers quote). This module is the
lightweight version used inside the training loop, where calling museval every
epoch would dominate runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BSSMetrics", "bss_eval", "evaluate_stems", "si_sdr"]

_EPS = 1e-10


@dataclass
class BSSMetrics:
    sdr: float
    sir: float
    sar: float

    def as_dict(self) -> dict[str, float]:
        return {"sdr": self.sdr, "sir": self.sir, "sar": self.sar}


def _project(signal: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Least-squares projection of ``signal`` onto the rows of ``basis``."""
    gram = basis @ basis.T
    coeffs = np.linalg.solve(gram + _EPS * np.eye(len(gram)), basis @ signal)
    return coeffs @ basis


def bss_eval(estimate: np.ndarray, target: np.ndarray, interferers: np.ndarray) -> BSSMetrics:
    """SDR/SIR/SAR for one estimated stem.

    Parameters
    ----------
    estimate:
        The separated waveform.
    target:
        The ground-truth stem it is meant to be.
    interferers:
        Array ``(n_other_sources, n_samples)`` of the remaining true stems.

    Both estimate and references must be sample-aligned and the same length --
    a one-frame offset from an unpadded ISTFT can cost 10 dB.
    """
    n = min(len(estimate), len(target))
    estimate, target = estimate[:n], target[:n]
    interferers = np.atleast_2d(interferers)[:, :n]

    all_sources = np.vstack([target[None, :], interferers])

    s_target = _project(estimate, target[None, :])
    s_sources = _project(estimate, all_sources)

    e_interf = s_sources - s_target
    e_artif = estimate - s_sources

    def db(num: float, den: float) -> float:
        return float(10.0 * np.log10((num + _EPS) / (den + _EPS)))

    p_target = float(np.sum(s_target**2))
    p_interf = float(np.sum(e_interf**2))
    p_artif = float(np.sum(e_artif**2))

    return BSSMetrics(
        sdr=db(p_target, p_interf + p_artif),
        sir=db(p_target, p_interf),
        sar=db(p_target + p_interf, p_artif),
    )


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant SDR -- immune to a global gain error on the estimate.

    Useful as a training loss/monitor because a model that gets the timbre
    right but the level wrong should not be punished as if it had failed.
    """
    n = min(len(estimate), len(target))
    estimate, target = estimate[:n], target[:n]
    estimate = estimate - estimate.mean()
    target = target - target.mean()

    # Normalize both to unit RMS before the ratio. Without this the fixed
    # epsilon floor below is a *different* relative floor for a quiet signal
    # than for a loud one, and the metric stops being scale-invariant exactly
    # where it is most useful (comparing a model's output against a reference
    # recorded at a different level).
    # (Copies, not in-place: the inputs are views into the caller's arrays.)
    def unit_rms(sig: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(sig**2)))
        return sig / rms if rms > 0 else sig

    estimate, target = unit_rms(estimate), unit_rms(target)

    alpha = float(np.dot(estimate, target) / (np.dot(target, target) + _EPS))
    proj = alpha * target
    noise = estimate - proj
    return float(10.0 * np.log10((np.sum(proj**2) + _EPS) / (np.sum(noise**2) + _EPS)))


def evaluate_stems(
    estimates: dict[str, np.ndarray], references: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Run ``bss_eval`` for every stem present in both dicts.

    Returns ``{stem: {"sdr":…, "sir":…, "sar":…, "si_sdr":…}}``. Stems missing
    from either side are skipped rather than scored as zero, so a partial
    model does not silently look terrible.
    """
    results: dict[str, dict[str, float]] = {}
    shared = [k for k in estimates if k in references]
    for name in shared:
        others = [references[k] for k in references if k != name]
        if not others:
            continue
        metrics = bss_eval(estimates[name], references[name], np.vstack(others))
        row = metrics.as_dict()
        row["si_sdr"] = si_sdr(estimates[name], references[name])
        results[name] = row
    return results
