from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _as_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def resolve_data_root(base_dir: str | Path | None = None) -> Path:
    """Resolve the base directory used for data/artifacts."""
    candidate = _as_path(base_dir) or _as_path(os.environ.get("EPM_DATA_ROOT"))
    if candidate:
        return candidate
    return _PACKAGE_ROOT / "data"


def resolve_reports_dir(base_dir: str | Path | None = None) -> Path:
    """Resolve the directory where reports should be written."""
    candidate = _as_path(base_dir) or _as_path(os.environ.get("EPM_REPORTS_DIR"))
    if candidate:
        return candidate
    return _PACKAGE_ROOT / "reports"
