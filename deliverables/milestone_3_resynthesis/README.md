# Milestone 3: Waveform resynthesis

**Deliverable:** clean, isolated `.wav` stems, one per target instrument,
produced the way the proposal specifies: predicted mask × mixture magnitude,
mixture phase, ISTFT.

## `stems/` (from `../sample/sample_mix.wav`)

`sample_mix_{vocal,violin,mridangam,ghatam}.wav`: the Milestone 2 masks
(`../milestone_2_model/masks/sample_mix_masks.npy`) resynthesized with
`scripts/resynthesize.py` (method `mixture_phase`, recorded in
`sample_mix_resynthesis.json`). Because the masks sum to one, the four stems
add back to the mixture sample for sample. Compare by ear with
`../sample/reference_*.wav`.

`direct_htdemucs/` holds HTDemucs's own waveform outputs for the same clip
(the stems this folder shipped before), kept for comparison.

## Method comparison

`scripts/compare_resynthesis.py` runs every method on the same masks:

| method | what it is |
|---|---|
| `direct` | HTDemucs's direct waveform output (its own internal ISTFT); not mask-based |
| `mixture_phase` | mask × mixture STFT → ISTFT: **the deliverable** |
| `griffin_lim` | mask × \|mixture\|, phase rebuilt from random init (fast Griffin-Lim, 32 iters) |
| `griffin_lim_mixinit` | the same, Griffin-Lim started from the mixture phase |
| `oracle_mixture_phase` | ideal ratio mask from the references × mixture STFT: the ceiling for mixture-phase masking |
| `vocoder` | mask × \|mixture\| → mel → BigVGAN: built, **not run yet** (see below) |

`mag_snr` is a phase-blind spectral match (`eval.sdr.magnitude_snr`): it
compares STFT magnitudes only, so it is the fair column for methods that make
up their own phase.

### `heldout/`: Concert03, all 6 tracks, central 60 s each

Concert03 was held out from training. Mean over stems:

| method | SDR | SIR | SAR | SI-SDR | mag_snr |
|---|---|---|---|---|---|
| direct | 8.89 | 23.03 | 9.25 | 8.90 | 11.52 |
| **mixture_phase** | **8.08** | 18.92 | 8.80 | 8.09 | 10.78 |
| griffin_lim | −30.81 | (15.44) | −28.10 | −30.81 | 10.68 |
| griffin_lim_mixinit | 6.46 | 21.68 | 6.72 | 6.46 | 10.88 |
| oracle_mixture_phase | 9.90 | 18.57 | 10.80 | 9.90 | 13.20 |

Per stem (SDR / mag_snr, dB):

| method | vocal | violin | mridangam | ghatam |
|---|---|---|---|---|
| direct | 9.30 / 12.83 | 7.11 / 9.56 | 12.24 / 14.61 | 6.93 / 9.08 |
| mixture_phase | 7.98 / 11.57 | 6.07 / 8.79 | 11.09 / 13.56 | 7.21 / 9.21 |
| griffin_lim | −36.48 / 11.52 | −29.94 / 8.76 | −29.98 / 13.48 | −26.86 / 8.94 |
| griffin_lim_mixinit | 5.60 / 11.74 | 3.26 / 8.86 | 10.07 / 13.68 | 6.93 / 9.23 |
| oracle_mixture_phase | 9.38 / 12.89 | 7.67 / 11.10 | 12.39 / 15.66 | 10.17 / 13.14 |

The `direct` row reproduces the checkpoint's recorded validation SDR (8.89 dB),
which confirms this evaluation uses the same tracks and excerpts as training.
`heldout/clips/Concert03_01-Vanajakshi-Varnam/` has 10 s listening clips of
every method, the mixture and the references.

### `sample/`: the 30 s deliverable clip

| method | SDR | mag_snr |
|---|---|---|
| direct | 7.91 | 10.70 |
| mixture_phase | 7.41 | 10.18 |
| griffin_lim | −27.83 | 9.68 |
| griffin_lim_mixinit | 6.18 | 10.27 |
| oracle_mixture_phase | 9.47 | 12.50 |

`sample/clips/sample/<method>/` holds full-length stems for every method.

## What the numbers say

* **Mixture-phase ISTFT costs 0.8 dB SDR compared with HTDemucs's direct
  output**: vocal −1.3, mridangam −1.2, violin −1.0, ghatam +0.3. Converting
  HTDemucs's estimates to a ratio mask throws away the magnitude and phase it
  predicted directly. Wherever two instruments share a bin, they get the
  mixture's phase. Vocal and violin overlap most, since they share the raga
  and often the octave.
* **The model is close to the masking ceiling on melody, far from it on
  ghatam.** The oracle mask is only 1.4 / 1.6 dB above the predicted one on
  vocal / violin (1.3 on mridangam), but 3.0 dB above it on ghatam, the stem with the least
  training data. Better masks would help ghatam most. For vocal and violin,
  the next gain has to come from phase, not masks.
* **Griffin-Lim recovers the magnitude, not the waveform.** From random phase
  its mag_snr is within 0.1 dB of mixture-phase ISTFT (10.68 vs 10.78), so the
  spectrum is right, yet its SDR is around −30 dB. BSS-Eval needs sample-level
  phase agreement with the reference, and Griffin-Lim's phase is arbitrary.
  Its SIR in brackets is meaningless for the same reason: both projections are
  near zero. This is the SDR-versus-perception gap the proposal anticipated,
  in its most extreme form. The listening clips are there to judge how it
  actually sounds; no listening test has been done yet.
* **Refining the mixture phase with Griffin-Lim makes things worse**
  (−1.6 dB SDR, mostly violin −2.8), for only +0.1 dB mag_snr. Making the
  phase self-consistent pulls it away from the true sources' phase.
  `mixture_phase` stays the default.

These are SDRs on Sanidha, where the mixture is the plain sum of near-zero-bleed
close mics. Live Saraga audio will score lower for every method.

## Vocoder: built, not yet run

`indian_mss/infer/vocoder.py` wraps NVIDIA's pretrained
**BigVGAN v2** (`nvidia/bigvgan_v2_44khz_128band_512x`). BigVGAN is a HiFi-GAN
generator with anti-aliased Snake activations, trained on music and
environmental sound as well as speech. It runs at 44.1 kHz with n_fft 2048 and
hop 512, the pipeline's own STFT grid, so the masks only need a half-frame time
shift. Public HiFi-GAN checkpoints are 22 kHz speech models, which is why
BigVGAN is used here.

Its model code ships inside the Hugging Face repo, so running it executes
downloaded Python. That step has not been run here. To run it:

```bash
pip install -e ".[vocoder]"
python scripts/compare_resynthesis.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --sanidha-root data/sanidha_flat --outdir deliverables/milestone_3_resynthesis/heldout
```

Expect the vocoder to score low on SDR, for the same reason as Griffin-Lim.
Judge it on `mag_snr` and by ear against the `mixture_phase` clips.
