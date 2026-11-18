# Milestone 1: Spectrogram profiling and preprocessing

**Deliverable:** a script that takes a polyphonic mix and returns a melodic
spectrogram array and a percussive spectrogram array.

**Script:** `scripts/preprocess.py` (implemented in `indian_mss/cli/preprocess.py`,
with HPSS in `indian_mss/dsp/hpss.py` and tonic handling in `indian_mss/dsp/tonic.py`).

## `output/` (run on `../sample/sample_mix.wav`)

| File | Contents |
|---|---|
| `sample_mix_melodic.npy` | melodic (harmonic) magnitude spectrogram, `(1025, 2585)` |
| `sample_mix_percussive.npy` | percussive magnitude spectrogram, same shape |
| `sample_mix_residual.npy` | bins that are neither clearly harmonic nor percussive (ghatam strokes, veena plucks) |
| `*_log.npy` | log-compressed versions in [0, 1], all scaled to the mixture's peak |
| `*.wav` | listening previews, resynthesised with the mixture phase |
| `sample_mix_tonic.json` | estimated tonic (130.81 Hz) and the shift applied to the canonical Sa (146.83 Hz) |

## Evidence

* `hpss_baseline.json`: HPSS scored on all 37 Sanidha tracks (60 s each). Mean
  SDR is 6.84 dB for the harmonic stem against the melodic references and
  −1.28 dB for the percussive stem. Milestone 2 has to beat this.
* `figures/` (from `reports/make_figures.py`, run on synthetic audio):
  * `m1_hpss_spectrograms.png`: mixture, melodic and percussive spectrograms.
  * `m1_kernel_sweep.png`: HPSS SDR across time and frequency kernel sizes.
  * `m1_tonic_fix.png`: the original versus the fixed tonic estimator.
