"""Shared serialization fallbacks for NAS/TAS modules.

Historically multiple NAS and TAS entry-points attempted to import the rich
serialisation helpers from ``src.utils.serialization_utils`` and re-declared
simple JSON/Pickle fallbacks when the dependency was absent.  The duplicated
logic increased maintenance cost and, in some modules, diverged in behaviour.

The helpers exported here perform the import once and provide battle-tested
fallback classes that maintain the expected ``save``/``load`` interface.  Both
NAS and TAS code can now depend on this single module which guarantees
consistent behaviour across the codebase.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

SERIALIZATION_AVAILABLE = False

try:  # pragma: no cover - passthrough import
    from src.utils.serialization_utils import (  # type: ignore
        JSONSerializer,
        PickleSerializer,
        ParquetSerializer,
        UniversalSerializer,
    )
    SERIALIZATION_AVAILABLE = True
except Exception:  # pragma: no cover - executed only when dependency missing

    class JSONSerializer:  # type: ignore[override]
        """Minimal JSON serialiser used when the shared package is unavailable."""

        @staticmethod
        def save(data: Any, filepath: str | Path) -> bool:
            try:
                with open(filepath, "w") as fh:
                    json.dump(data, fh, indent=2, default=str)
                return True
            except Exception:
                return False

        @staticmethod
        def load(filepath: str | Path) -> Any:
            with open(filepath, "r") as fh:
                return json.load(fh)

    class PickleSerializer:  # type: ignore[override]
        """Minimal Pickle serialiser used when the shared package is unavailable."""

        @staticmethod
        def save(data: Any, filepath: str | Path) -> bool:
            try:
                with open(filepath, "wb") as fh:
                    pickle.dump(data, fh)
                return True
            except Exception:
                return False

        @staticmethod
        def load(filepath: str | Path) -> Any:
            with open(filepath, "rb") as fh:
                return pickle.load(fh)

    class ParquetSerializer:  # pragma: no cover - placeholder
        """Placeholder Parquet serializer when optional dependency missing."""

        @staticmethod
        def save(data: Any, filepath: str | Path) -> bool:
            raise NotImplementedError("Parquet support requires serialization_utils")

        @staticmethod
        def load(filepath: str | Path) -> Any:
            raise NotImplementedError("Parquet support requires serialization_utils")

    class UniversalSerializer:  # pragma: no cover - placeholder
        """Placeholder universal serializer used for graceful degradation."""

        @staticmethod
        def save(data: Any, filepath: str | Path, format_hint: str | None = None) -> bool:
            if format_hint == "json":
                return JSONSerializer.save(data, filepath)
            if format_hint == "pickle":
                return PickleSerializer.save(data, filepath)
            raise NotImplementedError("Universal serialization requires serialization_utils")

        @staticmethod
        def load(filepath: str | Path) -> Any:
            raise NotImplementedError("Universal serialization requires serialization_utils")

def estimate_pickle_size(data: Any) -> int:
    """Estimate the size of an object when serialised with pickle."""

    try:
        return len(pickle.dumps(data))
    except Exception:
        return 0


__all__ = [
    "SERIALIZATION_AVAILABLE",
    "JSONSerializer",
    "PickleSerializer",
    "ParquetSerializer",
    "UniversalSerializer",
    "estimate_pickle_size",
]
