# Deliverables

One folder per milestone, each holding the artefact the project proposal asks
for plus the evidence behind it. Every milestone is run on the same input: a
30 s excerpt of a Sanidha track that training never saw (`sample/`).

| Milestone | Proposal deliverable | Here | State |
|---|---|---|---|
| 1 | Script returning a melodic and a percussive spectrogram array | `milestone_1_preprocessing/` | done |
| 2 | Model save state predicting mask ratios for Indian instruments | `milestone_2_model/` | done |
| 3 | Isolated `.wav` stems per instrument | `milestone_3_resynthesis/` | stems delivered; ISTFT / Griffin-Lim / vocoder comparison not built |
| 4 | Classification-accuracy report on separated stems | `milestone_4_classifier/` | not started (blocked on IMIDataset access) |

Core software artefacts: the pipeline library is `indian_mss/` at the repo root,
and the training notebook is `milestone_2_model/train.ipynb`.

## `sample/`

* `sample_mix.wav`: Concert03 / 02-Sharanam-Bhavan, 492–522 s, 44.1 kHz mono.
  Concert03 was held out as the validation concert, so the model never trained
  on it. The 30 s window is the one where all four instruments are loudest
  together. Sanidha ships no mixdown, so the mix is the sum of the close mics
  (`sample.json` records this).
* `reference_<instrument>.wav`: the ground-truth stems for the same window.

## Regenerating

From the repo root:

```bash
python scripts/preprocess.py --input deliverables/sample/sample_mix.wav \
    --outdir deliverables/milestone_1_preprocessing/output --write-audio
python scripts/separate.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --input deliverables/sample/sample_mix.wav --outdir deliverables/milestone_2_model/masks
mv deliverables/milestone_2_model/masks/*.wav deliverables/milestone_3_resynthesis/stems/
```

`*.wav`, `*.npy` and `*.pt` are git-ignored, so this folder is for handing in
(zip it), not for committing.
