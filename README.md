# Indian_Musicology — source separation for Indian art music

A separation pipeline for Hindustani and Carnatic ensemble recordings. Off-the-shelf
demixing models are trained on Western multitracks and fail here for two structural
reasons: they read tonal percussion (tabla, mridangam) as generic drums, and they
cannot follow the microtonal glides of a sitar, veena or sarod line. Live ensemble
recording adds a third problem — microphone bleed between channels — that no
frequency filter can undo.

This repository builds the pipeline in four stages: DSP preprocessing, in-domain
fine-tuning of a demixing network, waveform resynthesis, and a timbre classifier
that measures how much cross-instrument bleed actually survived.

## Status

| Milestone | Component | State |
|---|---|---|
| 1 | STFT front end, HPSS, tonic normalization, SDR metrics | **done** |
| 2 | HTDemucs / U-Net fine-tuning on Sanidha, mask prediction | **done** — needs a GPU run on real data |
| 3 | ISTFT / Griffin-Lim / vocoder resynthesis | **done**: all three in `infer/`, compared on held-out Concert03 (vocoder run pending) |
| 4 | MFCC + 2-D CNN instrument classifier | feature extraction done |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install compiam           # install first: most constrained dependency
pip install -e ".[dev]"
pytest                        # no data required; model tests need torch
```

`torch` is deliberately not pulled in by the base install — install the build that
matches your CUDA version from pytorch.org, then `pip install -e ".[train]"`.

## Getting the data

| Corpus | Role | Why |
|---|---|---|
| **Sanidha** ([arXiv:2501.06959](https://arxiv.org/abs/2501.06959)) | training | Studio Carnatic multitrack, artists playing together in isolation booths — near-zero bleed, so the masks learn the instrument and not the room. |
| **Saraga** ([Zenodo 4301737](https://zenodo.org/records/4301737)) | test only | Full live concerts *with* bleed. Training on it teaches the model that leakage belongs in the target; evaluating on it is the only honest generalization number. Use `saraga1.5_carnatic.zip` — 168 of its 249 recordings ship multitrack stems. The Hindustani release is mixture-only and cannot produce a separation metric at all. |
| **IMIDataset** | classifier only | 10 instrument classes, ~33 h. Milestone 4 input. |

Point the paths in `configs/default.yaml` at the extracted folders. Each loader
raises a specific error if the layout differs from what it expects — check the
layout comment at the top of `indian_mss/data/sanidha.py` against your download
before launching a training run.

## Quick start

Milestone 1 deliverable — split a polyphonic mix into melodic and percussive
spectrogram arrays:

```bash
python scripts/preprocess.py --input concert.wav --outdir out/ --write-audio
```

This writes `concert_melodic.npy` / `concert_percussive.npy` (linear magnitude,
used for masking and resynthesis) and `*_log.npy` versions (log-spectrogram in
[0, 1], all scaled against the mixture's peak so stem levels stay comparable).
Settings come from `configs/default.yaml` — including `tonic.strategy: shift`,
which pitch-shifts to the canonical Sa and records the source tonic in
`concert_tonic.json`. Any flag (`--n-fft`, `--harmonic-ms`,
`--tonic-strategy none`, `--config other.yaml`, …) overrides the file.

Establish the HPSS baseline that Milestone 2 has to beat:

```bash
python scripts/preprocess.py --evaluate --sanidha-root data/sanidha
```

Milestone 2 — fine-tune on Sanidha (writes `best.pt`, `last.pt`, `history.json`),
then predict mask ratios for any mix:

```bash
python scripts/train.py --outdir checkpoints/htdemucs                # --backbone unet if VRAM is short
python scripts/separate.py --checkpoint checkpoints/htdemucs/best.pt --input concert.wav --outdir out/
```

`separate.py` writes `concert_masks.npy` — `(sources, freq, frames)` mask ratios on
the same STFT grid as the Milestone 1 spectrograms, summing to one per bin — plus
one `.wav` per instrument. `notebooks/train.ipynb` runs the training and charts the
loss curves and per-stem validation SDR.

Milestone 3: turn those masks into stems, and compare resynthesis methods
against ground truth:

```bash
python scripts/resynthesize.py --masks out/concert_masks.npy --input concert.wav --outdir out/stems  # --method griffin_lim | vocoder
python scripts/compare_resynthesis.py --checkpoint checkpoints/htdemucs/best.pt --outdir out/resynthesis
```

`resynthesize.py` uses `resynthesis.method` from the config (default
`mixture_phase`: mask × mixture STFT → ISTFT). `compare_resynthesis.py` runs
every method on the checkpoint's held-out tracks and writes SDR / SIR / SAR /
SI-SDR plus a phase-blind magnitude SNR per stem, and listening clips. The
vocoder (BigVGAN, `pip install -e ".[vocoder]"`) downloads its weights and
model code from Hugging Face on first use.

Library use:

```python
from indian_mss import STFTConfig, HPSSConfig, hpss_spectrograms

result = hpss_spectrograms(audio, STFTConfig(sample_rate=44100), HPSSConfig(harmonic_ms=150))
result.harmonic     # melodic magnitude spectrogram
result.percussive   # transient magnitude spectrogram
result.phase        # mixture phase, kept for resynthesis
```

## Design decisions worth knowing

**One STFT config, threaded everywhere.** `STFTConfig` is passed to analysis,
masking and synthesis alike. Analysis/synthesis window mismatch is the single most
common cause of "the model trained fine but the audio sounds wrong".

**The complex spectrogram is kept, not just magnitude.** Milestone 3 inverts
predicted magnitudes using the *mixture* phase; discarding phase at the front end
makes that impossible.

**HPSS is hand-written, not `librosa.decompose.hpss`.** Milestone 1 asks for the
filter, and the kernel sizes need domain tuning: a mridangam stroke is much
shorter than a Western kick, and a gamaka-laden violin line is much less
horizontal than a Western sustained note. Kernels are specified in milliseconds
and Hz, then converted to frames and bins, so they stay physically meaningful when
the sample rate changes.

**HPSS emits a residual.** With `margin > 1`, bins that are neither clearly
harmonic nor clearly percussive go to a third output. Ghatam strokes and veena
plucks land there; dumping them into the harmonic stem is what makes naive HPSS
sound wrong on this repertoire.

**Tonic normalization is explicit.** Indian art music has no fixed reference pitch;
each performance picks its own Sa. Unnormalized, the network learns the same
phrase separately at a dozen absolute pitches and the classifier partly learns the
tonic instead of the timbre. `dsp/tonic.py` estimates it from a folded pitch
histogram (the drone makes Sa the modal pitch class) and either shifts or records
it. Swap in compIAM's tonic identification for real runs.

**Silent windows are dropped before classification.** A separated stem is mostly
silence wherever its instrument is not playing. Feed those windows to the
classifier and it scores well by learning "silence ⇒ whichever stem is usually
quiet" — which would invalidate the entire Milestone 4 bleed argument.

**Splits are by recording, never by window.** Adjacent windows from one take share
room, microphone and instrument. A random window split leaks and inflates accuracy
substantially.

**SIR and SAR are reported alongside SDR.** An aggressive mask raises SIR (less
mridangam in the violin) while lowering SAR (more musical noise); the two can
cancel to an unchanged SDR while sounding clearly worse.

## Remaining work

**Milestone 2** — implemented; what remains is the real training run and the
Saraga number. How it is built:

* *Backbone.* HTDemucs rather than Spleeter (PyTorch-native and maintained;
  Spleeter is TF1-era). Its `drums/bass/other/vocals` output head is rebuilt for
  `vocal/violin/mridangam/ghatam`, each new source starting from the closest
  Western one (`vocal←vocals`, `violin←other`, `mridangam`/`ghatam←drums`), so
  training starts from a model that already splits voice, melody and percussion.
* *Parameter-efficient schedule.* Phase 1 (`frozen_encoder_epochs`) trains only
  that head; phase 2 unfreezes the decoders and transformer at a 10× lower rate.
  Both encoders stay frozen throughout.
* *Simulated bleed.* Each source is convolved with a random short room response
  and leaked into the other microphones at −30 to −12 dB, and every leaked copy is
  labelled as the instrument that made it. That teaches the masks to follow the
  instrument, not the microphone — the bridge from studio Sanidha to live Saraga.
* *Other augmentation.* Per-stem gain, one shared pitch shift (the ensemble stays
  in tune with itself), and stems shuffled across tracks.
* *Selection.* `best.pt` is chosen by mean validation SDR, never loss, with SDR
  logged per stem. Validation holds out whole *concerts* (`train.val_concerts`), never
  tracks or segments: songs from one concert share performers, room and mics,
  and alternate takes are near-duplicates, so anything finer leaks.
* *Tonic.* Training does not pitch-normalize whole tracks; the ±2 semitone shift
  augmentation covers tonic variation instead, and separation runs on raw audio.
* *Still to do:* the training run, then evaluate `best.pt` on Saraga. `data/saraga.py`
  is in place: 168 live multitracks, mridangam summed from its two close mics, ghatam
  on only 46 of them (so quote the three-stem mean over all 168 and the four-stem mean
  over the 46 separately, never one mean over both). Its `mixture` argument is a real
  choice, not a default to ignore — the released master is separately engineered and is
  *not* the sum of the close mics, which BSS-Eval assumes; `mixture="stems"` remixes the
  mics linearly, keeping every bit of the live bleed while satisfying that assumption,
  and `mixture="mixdown"` scores the master listeners actually hear. Report both.

**Milestone 3** — implemented in `infer/`: mixture-phase ISTFT (the baseline),
fast Griffin-Lim (random or mixture-phase init) and a pretrained BigVGAN vocoder
(HiFi-GAN generator, trained on music as well as speech, at the pipeline's own
44.1 kHz / hop-512 grid). Results are in
`deliverables/milestone_3_resynthesis/`. *Still to do:* the vocoder run, then
listening tests. Vocoders often score *worse* on SDR while sounding better, so
judge them on `mag_snr` and by ear.

**Milestone 4** — train the 2-D CNN on IMIDataset, then run it on three inputs:
ground-truth Sanidha stems, your separated stems, and the raw mixture. If
separation worked, the separated stems sit near ground truth and far above the
mixture, and the off-diagonal mass in the confusion matrix quantifies exactly how
much bleed is left.

## Layout

```
indian_mss/
  data/     base.py sanidha.py saraga.py          — loaders (imi.py to build)
  dsp/      stft.py hpss.py tonic.py features.py  — Milestone 1
  models/   separator.py demucs_backbone.py unet.py spec.py — Milestone 2 models
  train/    augment.py dataset.py losses.py loop.py report.py — Milestone 2 training
  infer/    resynth.py vocoder.py                  — Milestone 3 resynthesis
  eval/     sdr.py                                 — SDR / SIR / SAR / SI-SDR / magnitude SNR
configs/default.yaml                               — all hyperparameters
scripts/preprocess.py                              — Milestone 1 CLI
scripts/train.py  scripts/separate.py              — Milestone 2 CLIs
scripts/resynthesize.py  scripts/compare_resynthesis.py — Milestone 3 CLIs
notebooks/train.ipynb                              — training run + charts
tests/                                             — run without any dataset
```
