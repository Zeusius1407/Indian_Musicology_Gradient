# Milestone 2: Model fine-tuning on Sanidha

**Deliverable:** a model save state that predicts mask ratios for Indian
instrument categories.

## `checkpoint/`

* `best.pt`: HTDemucs, with its output head rebuilt for vocal / violin /
  mridangam / ghatam and fine-tuned on Sanidha. This is the concert-split run
  (`checkpoints/htdemucs_concertsplit`): 31 training tracks, with Concert03
  (6 tracks) held out whole. Selected by validation SDR at epoch 56:

  | | vocal | violin | mridangam | ghatam | mean |
  |---|---|---|---|---|---|
  | val SDR (dB) | 9.30 | 7.11 | 12.24 | 6.93 | **8.89** |

* `history.json`: per-step loss, per-epoch validation SDR per stem, and the
  config and splits used.

Load it with `indian_mss.models.separator.load_separator("best.pt")`.

## `masks/` (run on `../sample/sample_mix.wav`)

* `sample_mix_masks.npy`: `(4, 1025, 2585)` mask ratios on the same STFT grid
  as the Milestone 1 spectrograms, summing to 1 over sources in every bin.
* `sample_mix_masks.json`: the source order, STFT settings and which checkpoint produced it.

## Training evidence

* `training_report.pdf`: loss curves and per-stem validation SDR for this run.
* `train.ipynb`: the runnable training notebook (proposal core artefact 2).

The Saraga (live, with bleed) evaluation in the repo was run on an earlier
checkpoint (`checkpoints/htdemucs_full`), not this one, so it is not included here.
