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

| Milestone | Component | 
|---|---|---|
| 1 | STFT front end, HPSS, tonic normalization, SDR metrics |
| 2 | HTDemucs / U-Net fine-tuning on Sanidha, mask prediction | 
| 3 | ISTFT / Griffin-Lim / vocoder resynthesis |
| 4 | MFCC + 2-D CNN instrument classifier | 

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
| **IMIDataset** | classifier only | 10 instrument classes; this release holds 8628 distinct clips (~13.8 h) once its duplicate part `Imid-06` and copied clips are dropped. No voice, mridangam or ghatam class, so Sanidha's clean stems add those three. `data/imi.py` documents the release's quirks. |

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

Milestone 4: train the timbre classifier, then classify ground-truth, separated
and mixed audio of the held-out concert:

```bash
python scripts/train_classifier.py --outdir checkpoints/classifier   # --no-imi: Sanidha's 4 stems only
python scripts/evaluate_classifier.py --classifier checkpoints/classifier/classifier.pt \
    --separator checkpoints/htdemucs/best.pt --outdir out/classifier
```

Library use:

```python
from indian_mss import STFTConfig, HPSSConfig, hpss_spectrograms

result = hpss_spectrograms(audio, STFTConfig(sample_rate=44100), HPSSConfig(harmonic_ms=150))
result.harmonic     # melodic magnitude spectrogram
result.percussive   # transient magnitude spectrogram
result.phase        # mixture phase, kept for resynthesis
```
## Layout

```
indian_mss/
  data/     base.py sanidha.py saraga.py imi.py   — loaders
  dsp/      stft.py hpss.py tonic.py features.py  — Milestone 1
  models/   separator.py demucs_backbone.py unet.py spec.py — Milestone 2 models
  train/    augment.py dataset.py losses.py loop.py report.py — Milestone 2 training
  infer/    resynth.py vocoder.py                  — Milestone 3 resynthesis
  eval/     sdr.py                                 — SDR / SIR / SAR / SI-SDR / magnitude SNR
  classify/ data.py model.py train.py evaluate.py report.py — Milestone 4 timbre classifier
configs/default.yaml                               — all hyperparameters
scripts/preprocess.py                              — Milestone 1 CLI
scripts/train.py  scripts/separate.py              — Milestone 2 CLIs
scripts/resynthesize.py  scripts/compare_resynthesis.py — Milestone 3 CLIs
scripts/train_classifier.py  scripts/evaluate_classifier.py — Milestone 4 CLIs
notebooks/train.ipynb                              — training run + charts
tests/                                             — run without any dataset
```
