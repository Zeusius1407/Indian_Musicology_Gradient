# Indian_Musicology — source separation for Indian art music

A separation pipeline for Carnatic (and Hindustani) ensemble recordings.
Off-the-shelf demixing models are trained on Western multitracks and fail here
for three reasons. They read tonal percussion (mridangam, ghatam, tabla) as
generic drums. They cannot follow the microtonal glides (gamakas) of a violin or
veena line. And live ensemble recording adds microphone bleed between channels,
which no frequency filter can undo.

The pipeline is built in four milestones:

| # | Stage | Code | Deliverable |
|---|---|---|---|
| 1 | STFT front end, HPSS, tonic normalization | `indian_mss/dsp/` | melodic + percussive spectrogram arrays |
| 2 | HTDemucs fine-tuned on Sanidha → mask ratios | `indian_mss/models/`, `indian_mss/train/` | model checkpoint |
| 3 | Waveform resynthesis (ISTFT / Griffin-Lim / BigVGAN) | `indian_mss/infer/` | isolated `.wav` stems |
| 4 | MFCC + 2-D CNN timbre classifier as a bleed test | `indian_mss/classify/` | classification-accuracy report |

Each milestone has a folder under `deliverables/` with its artefacts, full
results tables and a discussion. This file covers the headline numbers and how
to reproduce them.

---

## Headline results

| Milestone | Result | Where |
|---|---|---|
| 1 | HPSS baseline on 37 Sanidha tracks: **6.84 dB** melodic / **−1.28 dB** percussive SDR | `deliverables/milestone_1_preprocessing/hpss_baseline.json` |
| 1 | Tonic estimate within 50 cents on **34.5%** of 197 annotated Saraga recordings | `reports/figures/numbers.json` |
| 2 | Held-out Concert03 validation SDR **8.89 dB** (vocal 9.30, violin 7.11, mridangam 12.24, ghatam 6.93) | `deliverables/milestone_2_model/checkpoint/history.json` |
| 2 | Live Saraga, 168 recordings: **3.44 dB** 3-stem mean, **−0.62 dB** 4-stem mean (46 tracks) | `deliverables/milestone_2_model/saraga/saraga_eval.json` |
| 3 | Concert03, mixture-phase ISTFT **8.08 dB** SDR (direct 8.89, oracle mask 9.90) | `deliverables/milestone_3_resynthesis/heldout/resynthesis_comparison.json` |
| 4 | Concert03 balanced accuracy: reference **95.7%**, separated **94.8%**, −12 dB bleed control **52.1%**, mixture **22.3%** | `deliverables/milestone_4_classifier/classifier_eval.json` |
| 4 | Held-out IMIDataset clips: **97.6%** balanced accuracy (13 classes) | same file |
| 4 | Live Saraga (vocal, violin, mridangam): separated **86.3%** vs close mics **60.0%** vs mix **16.0%** | `deliverables/milestone_4_classifier/saraga/classifier_saraga.json` |

Every number above is also in the JSON file named next to it. The reproduction
steps below write each of these files again.

---

## 1. Environment

The delivered results were produced on:

* Linux, Python 3.14, PyTorch 2.11 (CUDA)
* one NVIDIA RTX 5060 with **8 GB** of VRAM

Any CUDA GPU with 8 GB should reproduce them. Milestone 1 and the tests run on
CPU only.

```bash
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch first, built for your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchaudio

# 2. The package with every extra the results need
pip install -e ".[train,eval,vocoder,dev]"

# 3. Sanity check: no data needed (model tests are skipped without torch)
pytest
```

| extra | pulls in | needed for |
|---|---|---|
| (base) | numpy, scipy, librosa, soundfile, matplotlib, pyyaml | Milestone 1, figures |
| `train` | torch, torchaudio, demucs, tqdm | Milestones 2–4 |
| `eval` | museval | BSS-Eval metrics |
| `vocoder` | huggingface_hub | the BigVGAN row of Milestone 3 |
| `dev` | pytest | tests |

Do **not** install `requirements.txt` or the `iam` extra into this environment.
Both pull in compIAM 0.4.1, which pins `numpy<=1.26.4` and `torch==2.0.0` and
only installs on Python ≤ 3.12. The pipeline does not use it.

The first Milestone 2 run downloads the pretrained `htdemucs` weights via
`demucs`. The vocoder downloads `nvidia/bigvgan_v2_44khz_128band_512x` from
Hugging Face, including its model code, which is then executed (see
`deliverables/milestone_3_resynthesis/README.md`).

---

## 2. Data

None of the corpora are redistributed here. Download them and place them under
`data/` (git-ignored):

| Corpus | Role | Get it |
|---|---|---|
| **Sanidha** | separator training + Concert03 test | [arXiv:2501.06959](https://arxiv.org/abs/2501.06959) (5 studio Carnatic concerts, 37 tracks, near-zero bleed) |
| **Saraga 1.5 Carnatic** | live test only, never trained on | [Zenodo 4301737](https://zenodo.org/records/4301737), file `saraga1.5_carnatic.zip` |
| **IMIDataset** | classifier training (10 instrument classes) | the `Imid-01` … `Imid-06` release |

Saraga's Hindustani release is mixture-only, so it cannot produce a
separation metric.

### Expected layout

```
data/
  sanidha/Concert0N/<NN-Title>/Audio-Multitracks-Clean/*.wav   # as downloaded
  sanidha_flat/Concert0N_<NN-Title> -> ../sanidha/Concert0N/<NN-Title>/Audio-Multitracks-Clean
  saraga/saraga1.5_carnatic/<concert>/<piece>/...              # extracted zip
  IMIDataset/Imid-01 ... Imid-06/                              # Imid-06 is a duplicate; the loader drops it
```

The loader (`indian_mss/data/sanidha.py`) expects **one folder per track** with
the stems inside. Sanidha ships a nested `Concert/Track/Audio-Multitracks-Clean/`
tree, so every command below points at a flat directory of symlinks. Build it
once:

```bash
mkdir -p data/sanidha_flat
for d in data/sanidha/Concert0*/[0-9]*/Audio-Multitracks-Clean; do
    track=$(basename "$(dirname "$d")"); concert=$(basename "$(dirname "$(dirname "$d")")")
    ln -sfn "$(realpath "$d")" "data/sanidha_flat/${concert}_${track}"
done
find data/sanidha_flat -mindepth 1 -maxdepth 1 | wc -l      # expect 37
```

macOS `._*` resource-fork files in the download are harmless and ignored.
Paths are configured in `configs/default.yaml` (`data:`). Each loader raises a
specific error if the layout is not what it expects.

---

## 3. Reproducing the results

All commands run from the repo root. Every hyperparameter comes from
`configs/default.yaml`, so leave it unchanged to reproduce the reported numbers.
Outputs land in `deliverables/` (overwriting the shipped versions) or in
`checkpoints/` (git-ignored).

There are two ways to reproduce:

* **Evaluate only (§3.1, §3.3–3.6):** use the shipped checkpoints in
  `deliverables/*/checkpoint/`. This skips both trainings and regenerates every
  table. The checkpoints are git-ignored. They are in the hand-in zip
  (`dist/indian_mss_handin.zip`), not in a fresh clone.
* **Full:** also retrain the separator (§3.2) and the classifier (§3.5).
  Expect small numeric differences from GPU non-determinism.

### 3.1 Milestone 1: preprocessing and the HPSS baseline

```bash
# Melodic / percussive spectrograms of the 30 s sample clip
python scripts/preprocess.py --input deliverables/sample/sample_mix.wav \
    --outdir deliverables/milestone_1_preprocessing/output --write-audio

# HPSS baseline over all 37 Sanidha tracks (central 60 s): 6.84 / -1.28 dB
python scripts/preprocess.py --evaluate --sanidha-root data/sanidha_flat --seconds 60 \
    --outdir deliverables/milestone_1_preprocessing

# Figures: HPSS spectrograms, kernel sweep, tonic accuracy on Saraga (~20 min)
python reports/make_figures.py
cp reports/figures/m1_*.png deliverables/milestone_1_preprocessing/figures/
```

`make_figures.py` writes every number it plots to `reports/figures/numbers.json`,
including the tonic table (34.5% within 50 cents) and the kernel sweep
(5.07 dB at the configured 200 ms / 500 Hz).

### 3.2 Milestone 2: train the separator (optional; GPU)

```bash
python scripts/train.py --sanidha-root data/sanidha_flat \
    --outdir checkpoints/htdemucs_concertsplit --batch-size 2 --num-workers 10
```

This is the configuration of the delivered run: HTDemucs, output head rebuilt
for vocal / violin / mridangam / ghatam, 10 head-only epochs then decoders and
transformer unfrozen, simulated-bleed augmentation, Concert03 held out whole.
`--batch-size 2` is what fits in 8 GB, while the config default of 4 needs more
memory. Each epoch prints per-stem validation SDR, and `best.pt` is picked by
mean validation SDR, never by loss.

* **Interruptions:** rerun the same command with `--resume` to continue from
  `last.pt`. The delivered run was resumed twice and stopped by hand during
  epoch 57 of 60. Its `best.pt` is epoch **56**, at 8.89 dB.
* **Use the result:** copy `checkpoints/htdemucs_concertsplit/{best.pt,history.json}`
  to `deliverables/milestone_2_model/checkpoint/`, or pass that path as
  `--checkpoint` / `--separator` in the steps below.
* **Training charts:** `notebooks/train.ipynb` holds the loss curves, per-stem
  SDR and the Saraga summary. To re-render them from a finished run without
  training:

  ```bash
  IMSS_RUN_TRAINING=0 IMSS_RUN_DIR=checkpoints/htdemucs_concertsplit IMSS_SANIDHA_ROOT=data/sanidha_flat \
      jupyter nbconvert --to notebook --execute --inplace notebooks/train.ipynb
  ```

### 3.3 Milestone 2: masks and the live Saraga test

```bash
CKPT=deliverables/milestone_2_model/checkpoint/best.pt

# Mask ratios for the sample clip: (4, 1025, 2585), summing to 1 per bin
python scripts/separate.py --checkpoint $CKPT \
    --input deliverables/sample/sample_mix.wav --outdir deliverables/milestone_2_model/masks
mv deliverables/milestone_2_model/masks/*.wav deliverables/milestone_3_resynthesis/direct_htdemucs/

# All 168 Saraga multitracks, central 60 s: 3.44 dB (3-stem) / -0.62 dB (4-stem, 46 tracks)
python scripts/evaluate.py --checkpoint $CKPT --saraga-root data/saraga \
    --outdir deliverables/milestone_2_model/saraga
```

`evaluate.py` scores two mixtures by default: the sum of the close mics (the
reported number) and the released stereo master. The master scores about
−22 dB because it is misaligned with the mics by 1–7 ms and polarity-flipped on
some tracks. It is kept in the JSON but not reported
(`deliverables/milestone_2_model/README.md` explains why).

### 3.4 Milestone 3: resynthesis and the method comparison

```bash
# The deliverable stems: mask x mixture STFT -> ISTFT
python scripts/resynthesize.py --masks deliverables/milestone_2_model/masks/sample_mix_masks.npy \
    --input deliverables/sample/sample_mix.wav --outdir deliverables/milestone_3_resynthesis/stems

# Every method on held-out Concert03 (6 tracks, central 60 s): mixture_phase 8.08 dB
python scripts/compare_resynthesis.py --checkpoint $CKPT \
    --sanidha-root data/sanidha_flat --outdir deliverables/milestone_3_resynthesis/heldout

# The same on the 30 s sample clip
python scripts/compare_resynthesis.py --checkpoint $CKPT \
    --sample deliverables/sample --clip-seconds 0 --outdir deliverables/milestone_3_resynthesis/sample
```

To leave out the vocoder (and skip the Hugging Face download), add
`--methods direct mixture_phase griffin_lim griffin_lim_mixinit oracle_mixture_phase`.

### 3.5 Milestone 4: train the classifier (optional; GPU)

```bash
python scripts/train_classifier.py --sanidha-root data/sanidha_flat --imi-root data/IMIDataset \
    --outdir checkpoints/classifier --cache checkpoints/classifier/windows.npz
```

This trains 13 classes: IMIDataset's 10 plus Sanidha's vocal, mridangam and
ghatam. Splits are by recording, and IMIDataset is split by blocks of 100
consecutive clips. Best epoch in the delivered run: 25 (99.0% validation
balanced accuracy). `--cache` stores the extracted MFCC windows so reruns skip
feature extraction. Add `--no-imi --outdir checkpoints/classifier_sanidha_only`
for the 4-class comparison in `deliverables/milestone_4_classifier/sanidha_only/`.

### 3.6 Milestone 4: the bleed tests

```bash
CLF=checkpoints/classifier/classifier.pt     # or deliverables/milestone_4_classifier/checkpoint/classifier.pt

# Concert03: reference / separated / -12 dB bleed / mixture = 95.7 / 94.8 / 52.1 / 22.3 %
python scripts/evaluate_classifier.py --classifier $CLF --separator $CKPT \
    --sanidha-root data/sanidha_flat --outdir deliverables/milestone_4_classifier

# Live Saraga, 168 recordings: separated 86.3% vs close mic 60.0% (~1 h on an 8 GB GPU)
python scripts/evaluate_classifier_saraga.py --classifier $CLF --separator $CKPT \
    --saraga-root data/saraga --outdir deliverables/milestone_4_classifier/saraga

# Supporting analyses for the Saraga reading
python reports/saraga_close_mic_bins.py      # bleed vs recording-domain check
python reports/validate_live_windows.py      # validates the window-selection rule
```

Each run writes a JSON with every per-track number, a PDF report with the
confusion matrices, and a PNG of its first page.

### 3.7 Packaging

```bash
python scripts/package_deliverables.py       # -> dist/indian_mss_handin.zip
```

This first checks that every git-ignored artefact the milestone READMEs refer
to exists (checkpoints, `.wav`, `.npy`, logs), and stops if any is missing.

### The sample clip

Every milestone runs on the same input, `deliverables/sample/sample_mix.wav`:
Concert03 / 02-Sharanam-Bhavan, 492–522 s, 44.1 kHz mono. It is the sum of the
close mics, because Sanidha ships no mixdown. The model never trained on it
(`deliverables/sample/sample.json`). To rebuild the clip and its references:

```python
import soundfile as sf
from indian_mss.data.sanidha import Sanidha
item = Sanidha("data/sanidha_flat", 44100).load("Concert03_02-Sharanam-Bhavan")
a, b = 492 * 44100, 522 * 44100
sf.write("deliverables/sample/sample_mix.wav", item.mixture[a:b], 44100)
for name, stem in item.stems.items():
    sf.write(f"deliverables/sample/reference_{name}.wav", stem[a:b], 44100)
```

---

## Known gaps in reproduction

* **HPSS-vs-HTDemucs on Saraga** (+1.0 / −2.7 dB HPSS against +8.5 / +6.8 dB,
  in the Milestone 2 README) was computed on the earlier `htdemucs_full`
  checkpoint. It is stored in `checkpoints/htdemucs_full/saraga_hpss_vs_htdemucs.json`,
  and no committed script regenerates it.
* **Listening tests** for Milestone 3 have not been done. Griffin-Lim and the
  vocoder can only be judged fairly by ear.
* **Tonic estimation** is right on only about a third of real recordings.
  Separator training and evaluation do not use it, so Milestones 2–4 are
  unaffected, but treat `sample_mix_tonic.json` as unverified.
* **Concert03 selected the checkpoint.** It was never trained on, but `best.pt`
  was chosen by validation SDR on it, which slightly flatters Concert03 numbers.
  Saraga is the fully unseen test.

---

## Using the pipeline on your own audio

```bash
python scripts/preprocess.py --input concert.wav --outdir out/ --write-audio
python scripts/separate.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --input concert.wav --outdir out/
python scripts/resynthesize.py --masks out/concert_masks.npy --input concert.wav --outdir out/stems
```

Any `preprocess.py` flag (`--n-fft`, `--harmonic-ms`, `--tonic-strategy none`,
`--config other.yaml`, …) overrides the config. The same CLIs are installed as
`imss-preprocess`, `imss-separate`, `imss-resynthesize` and so on. Library use:

```python
from indian_mss import STFTConfig, HPSSConfig, hpss_spectrograms

result = hpss_spectrograms(audio, STFTConfig(sample_rate=44100), HPSSConfig(harmonic_ms=150))
result.harmonic, result.percussive, result.phase
```

---

## Design decisions

* **One STFT config everywhere.** The same `STFTConfig` is passed to analysis,
  masking and synthesis, which rules out window mismatches between them.
* **Hand-written HPSS with a residual.** Kernels are given in ms and Hz, and
  tuned for short mridangam strokes and gamaka-laden melody. Ambiguous bins
  (ghatam strokes, veena plucks) go to a third output.
* **HTDemucs with a re-targeted head.** The new sources start from the closest
  Western ones (vocal←vocals, violin←other, mridangam/ghatam←drums). Both
  encoders stay frozen.
* **Simulated bleed.** Each source is leaked into the other mics through a
  random room response at −30 to −12 dB, and the leaked copy is labelled as its
  source. This bridges studio Sanidha to live Saraga.
* **Splits by concert or recording, never by window.** Songs from one concert
  share room and mics, and adjacent windows leak.
* **Silent windows are dropped before classification.** Otherwise the
  classifier learns "silence means whichever stem is usually quiet", which
  would invalidate the bleed test.
* **SIR and SAR are reported alongside SDR**, plus a phase-blind magnitude SNR
  for methods that make up their own phase.

---

## Layout

```
indian_mss/
  data/     base.py sanidha.py saraga.py imi.py                       loaders
  dsp/      stft.py hpss.py tonic.py features.py                      Milestone 1
  models/   separator.py demucs_backbone.py unet.py spec.py           Milestone 2 models
  train/    augment.py dataset.py losses.py loop.py report.py         Milestone 2 training
  infer/    resynth.py vocoder.py                                     Milestone 3
  eval/     sdr.py                                                    SDR / SIR / SAR / SI-SDR / mag SNR
  classify/ data.py model.py train.py evaluate.py live.py report.py   Milestone 4
  cli/                                                                implementations of scripts/*
configs/default.yaml        every hyperparameter
scripts/                    one CLI per step (see §3)
reports/                    figure generation and supporting analyses
notebooks/train.ipynb       Milestone 2 training notebook, charts rendered
deliverables/               per-milestone artefacts, results and discussion
report/                     the written project report (LaTeX + PDF)
tests/                      run without any dataset
```
