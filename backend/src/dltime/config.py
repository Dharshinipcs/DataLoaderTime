"""Project paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    # backend/src/dltime/config.py -> parents: [dltime, src, backend, <root>]
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = Path(os.environ.get("DLTIME_PROJECT_ROOT", _find_project_root()))
DATA_DIR: Path = Path(os.environ.get("DLTIME_DATA_DIR", PROJECT_ROOT / "train" / "data"))
