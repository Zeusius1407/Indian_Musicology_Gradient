#!/usr/bin/env python3
"""Thin wrapper so the documented ``python scripts/compare_resynthesis.py`` works.

The implementation lives in :mod:`indian_mss.cli.compare_resynthesis` so that the
``imss-compare-resynthesis`` console script survives a non-editable install.
"""

import sys
from pathlib import Path

# Running ``python scripts/compare_resynthesis.py`` puts scripts/ on the path, not the repo
# root, so make the package importable even before ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indian_mss.cli.compare_resynthesis import main  # noqa: E402

if __name__ == "__main__":
    main()
