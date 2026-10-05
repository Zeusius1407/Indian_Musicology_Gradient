# Notebooks

`train.ipynb` (Milestone 2 deliverable) should contain, in order:

1. Config load from `configs/default.yaml` — no hard-coded numbers. *(done)*
2. Dataset statistics: track count, total hours, per-stem duration. *(done)*
3. HPSS baseline table, loaded from `out/hpss_baseline.json`. *(done)*
4. Training loop with loss curves (per step, per epoch, validation). *(done)*
5. Per-stem validation SDR per epoch, on one axis with the baseline drawn in. *(done)*
6. Final evaluation: Sanidha held-out *(done)*, then Saraga (needs `data/saraga.py`).
7. Milestone 4 confusion matrices for the three classifier inputs.

The charts come from `indian_mss/train/report.py`; set `RUN_TRAINING = False` to
redraw them from a finished run's `history.json` without a GPU.

Keep the rendered charts committed so the notebook can be read without a GPU.
