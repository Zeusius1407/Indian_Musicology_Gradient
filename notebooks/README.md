# Notebooks

`train.ipynb` (Milestone 2 deliverable) should contain, in order:

1. Config load from `configs/default.yaml` — no hard-coded numbers.
2. Dataset statistics: track count, total hours, per-stem duration.
3. HPSS baseline table, loaded from `out/hpss_baseline.json`.
4. Training loop with live loss curves (total, spectrogram L1, waveform L1).
5. Per-stem validation SDR per epoch, on one axis with the baseline drawn in.
6. Final evaluation: Sanidha held-out, then Saraga (the generalization gap).
7. Milestone 4 confusion matrices for the three classifier inputs.

Keep the rendered charts committed so the notebook can be read without a GPU.
