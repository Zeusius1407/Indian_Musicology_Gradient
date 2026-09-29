#!/usr/bin/env python3
"""Thin wrapper kept so the documented ``python scripts/preprocess.py`` works.

The implementation lives in :mod:`indian_mss.cli.preprocess` so that the
``imss-preprocess`` console script survives a non-editable install --
``scripts/`` is not a package and is never shipped in the wheel.
"""

from indian_mss.cli.preprocess import main

if __name__ == "__main__":
    main()
