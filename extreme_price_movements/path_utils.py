from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent
MARKET_MODE_SUFFIXES = {"spot": "_spot", "perps": "_perps"}


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


def resolve_market_mode(market_mode: str | None = None, *, use_perps: bool = False) -> str:
    """Resolve spot/perps mode from an explicit value, env, or boolean flag."""
    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "")).strip().lower()
    if mode in {"perp", "perps", "futures"} or use_perps:
        return "perps"
    return "spot"


def mode_suffixed_path(path: str | Path, market_mode: str | None = None) -> Path:
    """Return a sibling path with the active market suffix before the extension."""
    src = Path(path)
    mode = resolve_market_mode(market_mode)
    return src.with_name(f"{src.stem}{MARKET_MODE_SUFFIXES[mode]}{src.suffix}")


def resolve_mode_file(path: str | Path, market_mode: str | None = None) -> Path:
    """Prefer the active market-suffixed file, falling back to the legacy path."""
    src = Path(path)
    mode_path = mode_suffixed_path(src, market_mode)
    if mode_path.exists():
        return mode_path
    return src


def mode_file_candidates(path: str | Path, market_mode: str | None = None) -> list[Path]:
    """Return active market-suffixed path followed by the legacy path."""
    src = Path(path)
    mode_path = mode_suffixed_path(src, market_mode)
    if mode_path == src:
        return [src]
    return [mode_path, src]
