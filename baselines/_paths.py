"""Resolve the repository root for every script in ``baselines/``.

Resolution order:

1. the ``EWTVD_ROOT`` environment variable, if set;
2. the first ancestor directory of this file that contains ``baselines/`` and
   ``configs/`` -- i.e. the repository root after a normal clone;
3. the current working directory.

Expected layout under the root (see the top-level README)::

    <root>/baselines/   these scripts
    <root>/configs/     dataset yaml files
    <root>/datasets/    WTVD/ and Extreme-WTVD/ after unzipping the archives
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root as a :class:`pathlib.Path`."""
    env = os.environ.get("EWTVD_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "baselines").is_dir() and (parent / "configs").is_dir():
            return parent

    return Path.cwd().resolve()


ROOT = repo_root()


def data_dir() -> Path:
    """Directory holding the unzipped WTVD / Extreme-WTVD roots."""
    return ROOT / "datasets"


def weights_dir() -> Path:
    """Local cache for checkpoints (not versioned)."""
    return ROOT / "weights"
