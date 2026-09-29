"""Command-line entry points.

These live inside the package rather than in ``scripts/`` so that the console
scripts declared in ``pyproject.toml`` survive a non-editable install --
``scripts/`` is not a package and is never shipped in the wheel.
"""
