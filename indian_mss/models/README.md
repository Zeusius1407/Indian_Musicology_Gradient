# Milestone 2 — demixing models

| Module | What it is |
|---|---|
| `separator.py` | The one interface both backbones share: `build_separator`, chunked `separate`, `predict_masks`, `save_checkpoint` / `load_separator` |
| `demucs_backbone.py` | Pretrained HTDemucs re-targeted to `vocal/violin/mridangam/ghatam`, with the two-phase freeze schedule |
| `unet.py` | From-scratch spectrogram U-Net with softmax masks — the fallback when HTDemucs will not fit in VRAM |
| `spec.py` | `torch.stft` wrapper that matches `dsp.stft` bin for bin, so masks line up with the Milestone 1 grid |

Training lives in `indian_mss/train/` (`augment.py`, `dataset.py`, `losses.py`, `loop.py`, `report.py`).

Two rules the training loop follows, both learned the hard way:

1. Select the checkpoint by validation SDR, not validation loss. Spectrogram L1
   keeps improving after perceptual quality has plateaued.
2. Log per-stem SDR, not just the mean. A model that nails the vocal and fails
   the ghatam looks acceptable on the average and is useless in practice.
