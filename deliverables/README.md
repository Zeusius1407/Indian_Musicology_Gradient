# Deliverables

One folder per milestone, each holding the artefact the project proposal asks
for plus the evidence behind it. Every milestone is run on the same input: a
30 s excerpt of a Sanidha track that training never saw (`sample/`).

| Milestone | Proposal deliverable | Here | State |
|---|---|---|---|
| 1 | Script returning a melodic and a percussive spectrogram array | `milestone_1_preprocessing/` | done |
| 2 | Model save state predicting mask ratios for Indian instruments | `milestone_2_model/` | done |
| 3 | Isolated `.wav` stems per instrument | `milestone_3_resynthesis/` | done: mixture-phase ISTFT stems, compared with Griffin-Lim and a BigVGAN vocoder on held-out Concert03 |
| 4 | Classification-accuracy report on separated stems | `milestone_4_classifier/` | done: 13-class CNN on IMIDataset + Sanidha, tested on all of Concert03, held-out IMIDataset clips, and all 168 live Saraga recordings |

Core software artefacts: the pipeline library is `indian_mss/` at the repo root,
and the training notebook is `milestone_2_model/train.ipynb` (kept in the repo
at `notebooks/train.ipynb` and copied in when packaging).

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
mv deliverables/milestone_2_model/masks/*.wav deliverables/milestone_3_resynthesis/direct_htdemucs/
python scripts/resynthesize.py --masks deliverables/milestone_2_model/masks/sample_mix_masks.npy \
    --input deliverables/sample/sample_mix.wav --outdir deliverables/milestone_3_resynthesis/stems
python scripts/compare_resynthesis.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --sanidha-root data/sanidha_flat --outdir deliverables/milestone_3_resynthesis/heldout
python scripts/compare_resynthesis.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --sample deliverables/sample --clip-seconds 0 --outdir deliverables/milestone_3_resynthesis/sample
python scripts/train_classifier.py --sanidha-root data/sanidha_flat --imi-root data/IMIDataset \
    --outdir checkpoints/classifier
python scripts/evaluate_classifier.py --classifier checkpoints/classifier/classifier.pt \
    --separator deliverables/milestone_2_model/checkpoint/best.pt \
    --sanidha-root data/sanidha_flat --outdir deliverables/milestone_4_classifier
```

Add `--methods direct mixture_phase griffin_lim griffin_lim_mixinit oracle_mixture_phase`
to either resynthesis comparison to leave out the vocoder.

`*.wav`, `*.npy`, `*.pt` and `*.log` are git-ignored. To hand in, build the zip,
which holds this folder, those artefacts, the notebook and the source:

```bash
python scripts/package_deliverables.py     # -> dist/indian_mss_handin.zip
```
