# Milestone 3: Waveform resynthesis

**Deliverable:** clean, isolated `.wav` stems, one per target instrument.

## `stems/` (from `../sample/sample_mix.wav`)

`sample_mix_vocal.wav`, `sample_mix_violin.wav`, `sample_mix_mridangam.wav`,
`sample_mix_ghatam.wav`: compare by ear with `../sample/reference_*.wav`.

`sample_metrics.json` scores them against those references:

| | vocal | violin | mridangam | ghatam | mean |
|---|---|---|---|---|---|
| SDR (dB) | 7.02 | 6.36 | 10.09 | 8.18 | **7.91** |
| SIR (dB) | 15.55 | 19.06 | 25.21 | 22.47 | |
| SAR (dB) | 7.80 | 6.65 | 10.24 | 8.37 | |

These scores come from one 30 s excerpt, so treat them as an illustration
rather than a benchmark. SIR is well above SAR for every stem: very little of
the other instruments leaks through, and the remaining error is mostly
artefacts.

## How these stems were produced (and what is still missing)

These are HTDemucs's **direct waveform outputs**. HTDemucs runs its own
internal ISTFT, so they are phase-aligned, but they are *not* yet the
proposal's method of "predicted magnitude × mixture phase → ISTFT".

Still to build for this milestone:

1. `indian_mss/infer/`: resynthesis from `masks.npy` × mixture magnitude × mixture
   phase → ISTFT (the code path already exists for HPSS in `dsp/hpss.py`).
2. Griffin-Lim, and optionally a HiFi-GAN vocoder (`resynthesis.method` in
   `configs/default.yaml` is defined but not read yet).
3. SDR/SIR/SAR for each method on the held-out set, plus listening clips. A
   vocoder can score lower on SDR and still sound better, and the report
   should show that trade-off.
