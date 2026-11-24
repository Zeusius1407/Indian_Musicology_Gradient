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
* `figures/` (from `reports/make_figures.py`, all on real audio with the
  pipeline's own STFT and HPSS settings; every number is in
  `reports/figures/numbers.json`):
  * `m1_hpss_spectrograms.png`: mix, melodic and percussive spectrograms of
    the first 10 s of `../sample/sample_mix.wav`. Over the whole 30 s clip,
    HPSS scores 3.61 dB SDR (melodic) and 2.89 dB (percussive) against the
    reference stems.
  * `m1_kernel_sweep.png`: mean HPSS SDR (melodic and percussive) over time
    and frequency kernel sizes, on all six Concert03 tracks (central 30 s,
    ground-truth stems). The configured kernels (200 ms, 500 Hz) score
    5.07 dB. The best cell is 200 ms / 1000 Hz at 5.55 dB: +0.5 dB, split
    between melodic (+0.4) and percussive (+0.6). The default was left
    unchanged so the HPSS baseline above stays comparable. A 50 ms time kernel
    loses up to 3.4 dB against 200 ms, because a kernel that short cannot tell
    a sustained note from a stroke.
  * `m1_tonic_saraga.png`: `dsp/tonic.py` scored on the 197 Saraga Carnatic
    recordings with an annotated tonic (central 30 s of each released
    recording).

### Tonic estimation on real recordings

| outcome | share |
|---|---|
| within 50 cents of the annotation | 34.5% |
| right pitch class, wrong octave | 14.2% |
| Pa taken for Sa | 20.3% |
| Ma taken for Sa | 10.2% |
| other | 20.8% |

When it is right it is precise (median error 10 cents), but it is right on
only about a third of real concert recordings. The figure also shows
estimates stuck at the 100 Hz search floor. The likely cause, not verified,
is that the pitch histogram of a full mix is dominated by the melody rather
than the drone, so Pa and Ma, which melodies dwell on, can outvote Sa. compIAM's trained tonic models handle this better, but compIAM 0.4.1
cannot be installed in this environment (it pins `numpy<=1.26.4`; see
`indian_mss/dsp/tonic.py`).

**What this affects.** `tonic.strategy: shift`, the config default, pitch-shifts
the preprocessing output by the estimated tonic. On a typical real recording
that shift is likely to be wrong, so treat `sample_mix_tonic.json` as
unverified. Separator training and evaluation do not use the tonic estimate:
they run on raw audio, with ±2 semitone pitch augmentation. Milestones 2–4 are
unaffected.
