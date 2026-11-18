#!/usr/bin/env python3
"""Thin wrapper so the documented ``python scripts/train_classifier.py`` works.

The implementation lives in :mod:`indian_mss.cli.train_classifier` so that the
``imss-train-classifier`` console script survives a non-editable install.
"""

import sys
from pathlib import Path

# Running ``python scripts/train_classifier.py`` puts scripts/ on the path, not the repo
# root, so make the package importable even before ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indian_mss.cli.train_classifier import main  # noqa: E402

if __name__ == "__main__":
    main()
