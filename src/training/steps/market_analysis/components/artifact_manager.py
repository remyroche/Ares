"""Utility helpers for persisting component artifacts.

The original version of this module attempted to expose asynchronous APIs
without actually performing asynchronous work which caused confusing calling
patterns.  The simplified implementation provides a synchronous
``save_artifacts`` method with strong input validation and deterministic file
layout.  ``BaseMarketAnalysisComponent`` will offload the blocking work to a
thread when running inside an asynchronous context.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # Pandas and NumPy are optional when running unit tests.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


class ArtifactManager:
    """Persist artifacts for a component execution."""

    def __init__(self, base_dir: Path | str = "artifacts") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ArtifactManager")

    def save_artifacts(
        self,
        component_name: str,
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Persist all artifacts and return a mapping of logical name to path."""

        if not component_name:
            raise ValueError("component_name must be a non-empty string")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("artifacts must be a non-empty dictionary")

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        component_dir = self.base_dir / component_name
        component_dir.mkdir(parents=True, exist_ok=True)

        run_dir = component_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, str] = {}
        for name, value in artifacts.items():
            if not name:
                raise ValueError("artifact names must be non-empty strings")
            path = self._determine_file_path(run_dir, name, value)
            self._write_value(path, value, metadata)
            saved_paths[name] = str(path)
            self.logger.debug("Saved artifact %s -> %s", name, path)

        if metadata:
            meta_path = run_dir / "metadata.json"
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)
            saved_paths["metadata"] = str(meta_path)

        return saved_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _determine_file_path(self, directory: Path, name: str, value: Any) -> Path:
        suffix = "json"
        if pd is not None and isinstance(value, pd.DataFrame):  # type: ignore[attr-defined]
            suffix = "parquet"
        elif np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
            suffix = "npy"
        elif isinstance(value, (bytes, bytearray)):
            suffix = "bin"
        return directory / f"{name}.{suffix}"

    def _write_value(self, path: Path, value: Any, metadata: Optional[Dict[str, Any]]) -> None:
        if pd is not None and isinstance(value, pd.DataFrame):  # type: ignore[attr-defined]
            value.to_parquet(path)
            return
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
            np.save(path, value)
            return
        if isinstance(value, (bytes, bytearray)):
            path.write_bytes(value)
            return

        with path.open("w", encoding="utf-8") as handle:
            json.dump({"value": value, "metadata": metadata or {}}, handle, indent=2, default=str)


__all__ = ["ArtifactManager"]
