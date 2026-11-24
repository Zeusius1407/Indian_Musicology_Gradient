# Notebooks

`train.ipynb` is the Milestone 2 deliverable (proposal core artefact 2). It is
committed with its charts rendered, so it reads without a GPU, and
`scripts/package_deliverables.py` copies it into
`deliverables/milestone_2_model/` for hand-in. Edit it here only.

Sections:

1. Dataset statistics: track count, total hours, per-stem duration.
2. The HPSS baseline from `out/hpss_baseline.json`.
3. Training, or loading a finished run's `history.json`.
4. Loss curves per step and per epoch.
5. Per-stem validation SDR per epoch, with the baseline drawn in.
6. The selected checkpoint scored per track on held-out Concert03.
7. The live test on all 168 Saraga recordings (read from
   `deliverables/milestone_2_model/saraga/saraga_eval.json`).

The Milestone 4 confusion matrices are in `deliverables/milestone_4_classifier/`,
not here: they come from a different network.

To re-render the charts from the delivered run, without training:

```bash
IMSS_RUN_TRAINING=0 IMSS_RUN_DIR=checkpoints/htdemucs_concertsplit IMSS_SANIDHA_ROOT=data/sanidha_flat \
    jupyter nbconvert --to notebook --execute --inplace notebooks/train.ipynb
```

The charts come from `indian_mss/train/report.py`.
