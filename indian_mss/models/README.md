# Milestone 2 — model code goes here

Planned modules:

- `heads.py`      — replace Demucs' 4-stem output with the Sanidha stem set
- `finetune.py`   — freeze/unfreeze schedule, optimizer groups, checkpointing
- `augment.py`    — gain, pitch, cross-track stem shuffling, simulated bleed
- `unet.py`       — from-scratch spectrogram U-Net fallback if Demucs will not
                    fit in available VRAM

Two rules the training loop must follow, both learned the hard way:

1. Select the checkpoint by validation SDR, not validation loss. Spectrogram L1
   keeps improving after perceptual quality has plateaued.
2. Log per-stem SDR, not just the mean. A model that nails the vocal and fails
   the ghatam looks acceptable on the average and is useless in practice.
