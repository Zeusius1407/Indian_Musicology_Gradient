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
| 2 | Demucs fine-tuning on Sanidha | scaffolded (`indian_mss/models/`) |
| 3 | ISTFT / Griffin-Lim / vocoder resynthesis | ISTFT path done, in `dsp/hpss.py` |
| 4 | MFCC + 2-D CNN instrument classifier | feature extraction done |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install compiam           # install first: most constrained dependency
pip install -e ".[dev]"
pytest                        # 11 tests, no data required
```

`torch` is deliberately not pulled in by the base install — install the build that
matches your CUDA version from pytorch.org, then `pip install -e ".[train]"`.

## Getting the data

| Corpus | Role | Why |
|---|---|---|
| **Sanidha** ([arXiv:2501.06959](https://arxiv.org/abs/2501.06959)) | training | Studio Carnatic multitrack, artists playing together in isolation booths — near-zero bleed, so the masks learn the instrument and not the room. |
| **Saraga** | test only | Full live concerts *with* bleed. Training on it teaches the model that leakage belongs in the target; evaluating on it is the only honest generalization number. |
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

Establish the HPSS baseline that Milestone 2 has to beat:

```bash
python scripts/preprocess.py --evaluate --sanidha-root data/sanidha
```

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

**Milestone 2** — start from HTDemucs rather than Spleeter (PyTorch-native and
still maintained; Spleeter is TF1-era). Replace the `vocals/drums/bass/other` head
with `vocal/violin/mridangam/ghatam`, keep the pretrained encoder, and train the
new head with the encoder frozen before unfreezing its top blocks at a 10× lower
learning rate. The augmentation that matters most is **simulated bleed**: convolve
each clean Sanidha stem with a short room impulse response and mix attenuated
copies of its neighbours into every channel. That is what carries the model from
studio-clean training data to Saraga's live conditions. Select checkpoints by
validation SDR, not validation loss.

**Milestone 3** — the mixture-phase ISTFT path already works and is the correct
baseline. Compare it against Griffin-Lim and a HiFi-GAN vocoder both numerically
and by ear; vocoders often score *worse* on SDR while sounding better, and
documenting that tension is a stronger result than picking whichever number wins.

**Milestone 4** — train the 2-D CNN on IMIDataset, then run it on three inputs:
ground-truth Sanidha stems, your separated stems, and the raw mixture. If
separation worked, the separated stems sit near ground truth and far above the
mixture, and the off-diagonal mass in the confusion matrix quantifies exactly how
much bleed is left.

## Layout

```
indian_mss/
  data/     base.py sanidha.py saraga.py imi.py   — one interface, three corpora
  dsp/      stft.py hpss.py tonic.py features.py  — Milestone 1
  models/                                          — Milestone 2 (to build)
  infer/                                           — Milestone 3 CLI (to build)
  eval/     sdr.py                                 — SDR / SIR / SAR / SI-SDR
configs/default.yaml                               — all hyperparameters
scripts/preprocess.py                              — Milestone 1 CLI
tests/test_dsp.py                                  — runs without any dataset
```
