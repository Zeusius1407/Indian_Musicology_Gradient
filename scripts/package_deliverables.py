#!/usr/bin/env python3
"""Build the hand-in zip: every deliverable plus the source that produced it.

    python scripts/package_deliverables.py            # -> dist/indian_mss_handin.zip

Contents:

* ``deliverables/``, including the git-ignored artefacts the READMEs refer to
  (checkpoints, ``.wav`` stems, ``.npy`` arrays, logs).
* The training notebook, copied to ``deliverables/milestone_2_model/train.ipynb``.
  It is kept once in the repo, at ``notebooks/train.ipynb``, so the two copies
  cannot drift apart.
* The source: every file outside ``deliverables/`` that git does not ignore (library,
  scripts, configs, tests, reports, notebooks, READMEs). No dataset audio is
  included.

Before writing anything, it checks that the large artefacts the milestone
READMEs promise actually exist, and stops if one is missing. Those files are
git-ignored, so a fresh clone does not have them.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "indian_mss_handin.zip"
NOTEBOOK = ROOT / "notebooks" / "train.ipynb"
NOTEBOOK_IN_ZIP = "deliverables/milestone_2_model/train.ipynb"
SKIP_NAMES = {".DS_Store"}

REQUIRED = [
    "deliverables/sample/sample_mix.wav",
    "deliverables/milestone_1_preprocessing/output/sample_mix_melodic.npy",
    "deliverables/milestone_1_preprocessing/output/sample_mix_percussive.npy",
    "deliverables/milestone_2_model/checkpoint/best.pt",
    "deliverables/milestone_2_model/masks/sample_mix_masks.npy",
    "deliverables/milestone_2_model/saraga/saraga_eval.json",
    *(f"deliverables/milestone_3_resynthesis/stems/sample_mix_{s}.wav" for s in ("vocal", "violin", "mridangam", "ghatam")),
    "deliverables/milestone_4_classifier/checkpoint/classifier.pt",
    "deliverables/milestone_4_classifier/saraga/classifier_saraga.json",
]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def main() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        sys.exit("missing deliverable artefacts (regenerate them first, see deliverables/README.md):\n  " + "\n  ".join(missing))

    files: dict[str, Path] = {}
    for path in sorted((ROOT / "deliverables").rglob("*")):
        if path.is_file() and path.name not in SKIP_NAMES and not path.name.startswith("._"):
            files[path.relative_to(ROOT).as_posix()] = path
    for rel in tracked_files():
        if not rel.startswith("deliverables/") and (ROOT / rel).is_file():
            files[rel] = ROOT / rel
    files[NOTEBOOK_IN_ZIP] = NOTEBOOK

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in sorted(files.items()):
            z.write(src, f"indian_mss_handin/{arc}")
    tmp.replace(OUT)
    size = sum(p.stat().st_size for p in files.values())
    print(f"{len(files)} files, {size / 1e6:.0f} MB uncompressed -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
