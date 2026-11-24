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
* `train.ipynb`: the runnable training notebook (proposal core artefact 2), with
  its charts rendered. It lives in the repo at `notebooks/train.ipynb` and is
  copied here by `scripts/package_deliverables.py`.

## `saraga/`: live-bleed test on Saraga

`checkpoint/best.pt` scored on all 168 Saraga Carnatic multitrack recordings
(central 60 s each), which no part of training ever saw. Saraga is recorded
live, so every close mic carries the other instruments' bleed.

* `saraga_eval.json`: summary plus SDR / SIR / SAR / SI-SDR per stem and per track.
* `saraga_eval.log`: the run's console output.

Regenerate with:

```bash
python scripts/evaluate.py --checkpoint deliverables/milestone_2_model/checkpoint/best.pt \
    --outdir deliverables/milestone_2_model/saraga
```

### Results (mixture = sum of the close mics)

| stem | tracks | SDR mean | SDR median | SIR | SAR | tracks < 0 dB | Concert03 SDR |
|---|---|---|---|---|---|---|---|
| vocal | 168 | 4.21 | 5.89 | 13.89 | 5.69 | 8% | 9.30 |
| violin | 168 | 1.27 | 0.88 | 12.34 | 2.67 | 41% | 7.11 |
| mridangam | 168 | 4.83 | 8.68 | 16.87 | 6.84 | 14% | 12.24 |
| ghatam | 46 | −8.47 | −5.95 | 10.42 | −6.57 | 87% | 6.93 |

**3-stem mean (vocal, violin, mridangam; 168 tracks): 3.44 dB. 4-stem mean
(46 tracks with a ghatam mic): −0.62 dB.** The two are averaged over
different track sets and should not be merged. SIR and SAR are means.

Against the Milestone 1 HPSS baseline on the same 168 tracks
(`checkpoints/htdemucs_full/saraga_hpss_vs_htdemucs.json`, with stems grouped
into melodic and percussive):

| | melodic | percussive |
|---|---|---|
| HPSS | +1.0 dB | −2.7 dB |
| fine-tuned HTDemucs | +8.5 dB | +6.8 dB |

That comparison was run on the earlier `htdemucs_full` checkpoint. This
checkpoint is within 0.3 dB of it on every Saraga stem except ghatam (3-stem
mean 3.44 vs 3.57 dB), so the 7–9 dB margin carries over.

### What the numbers say

* **Fine-tuning works on live audio.** The model beats HPSS by 7–9 dB on
  bleed-heavy recordings it never saw.
* **Studio to live costs about 5 dB**, unevenly. Vocal and mridangam transfer
  reasonably (mridangam median 8.7 dB, against 12.2 on Concert03). Violin is
  weak, below 0 dB on 41% of tracks. Ghatam fails, the stem with the least
  training data and the one Milestone 3 found furthest from the oracle mask.
* **The mridangam mean is pulled down by pieces where it is silent.** Its 12
  worst tracks (−22 to −44 dB, SIR strongly negative) are nearly all shlokas,
  viruttams and mangalams, free-rhythm pieces with no percussion. The scorer
  keeps any stem above 10⁻⁶ of the mixture's energy. So the mridangam mic,
  which holds only vocal and violin bleed there, is scored as the reference,
  and the model is penalised for correctly staying quiet. This is why the
  median (8.7 dB) sits well above the mean (4.8 dB). Ghatam, which plays only
  in parts of a concert, is probably affected the same way.
* **Live references are not ground truth.** SIR stays at 10–17 dB while SAR
  falls. Every Saraga reference contains room bleed, so a separator that
  removes the bleed cannot match it and is penalised. SDR therefore cannot
  tell a failure apart from a clean separation. The classifier test of
  Milestone 4 is the measure that can, and it has not been run on Saraga yet.
* **Violin and vocal trade off.** Per track, violin SDR is anti-correlated
  with vocal SDR (r = −0.54) and with mridangam SDR (r = −0.49). The worst
  violin tracks have SIR of only 0–5 dB. Carnatic violin shadows the vocal
  line in the same raga, often in the same octave, so the shared melodic
  energy seems to go to one stem or the other. This is a reading of the
  numbers and has not been verified.
* **The piece matters more than the recording.** Concert identity explains
  only 15% of the per-track variance, although the 19 per-concert means range
  from +0.3 to +7.3 dB. Every Saraga multitrack was recorded at one venue
  (Arkay), so this result does not test different rooms.
* **Holding out Concert03 cost nothing on live audio.** This checkpoint and
  `htdemucs_full` (which trained on all tracks) score the same on Saraga. Six
  more studio tracks do not help, so the limit is more likely Sanidha's
  diversity (5 concerts, little ghatam) than model size. There is no run
  without the simulated-bleed augmentation, so these results cannot say
  whether it helped.

### Not reported: the released stereo master

`saraga_eval.json` also holds `mixture = mixdown` scores (3-stem mean
−21.9 dB). They reflect how the master was made, not the separator. Measured
on 6 tracks, the master is shifted from the close mics by 36–311 samples
(1–7 ms), by a different amount each time, and about half the tracks are
polarity-inverted. At zero shift the master and the close-mic sum are almost
uncorrelated (|r| < 0.3), and still only 0.3–0.7 after correcting the shift,
because the master has its own EQ, compression and stereo mix. BSS-Eval needs
sample-level alignment, so these scores collapse whatever the model does. The
earlier checkpoint shows the same −22 dB. Scoring the master fairly needs a
per-track delay and polarity fix first, and then a phase-blind metric.
