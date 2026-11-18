# Milestone 4: Timbre recognition and auto-tagging

**Deliverable:** an evaluation report of classification accuracy on the
separated stems, showing that cross-instrument bleed was removed.

**State: not started.** The only piece in place is MFCC feature extraction
(`indian_mss/dsp/features.py`).

Blocked on: getting access to the IMIDataset (no public download; access has
been requested from the authors / course staff). Once it arrives at `data/imi`:

1. `indian_mss/data/imi.py` loader, with splits by recording.
2. A lightweight 2-D CNN on MFCC windows, dropping silent windows.
3. Run it on three inputs: the ground-truth stems (`../sample/reference_*`),
   the separated stems (`../milestone_3_resynthesis/stems/`) and the raw mix.
   Report the confusion matrices; the off-diagonal mass is the bleed that is left.

Note: IMIDataset's 10 classes do not include vocal, mridangam or ghatam, so
those classes need another labelled source (e.g. Sanidha's clean stems, split
by concert) before the comparison covers all four stems.
