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
            try:
                import pandas as pd
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                # Convert data to DataFrame if it's not already
                if isinstance(data, pd.DataFrame):
                    df = data
                elif isinstance(data, dict):
                    df = pd.DataFrame(data)
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    # For other data types, try to convert to dict first
                    if hasattr(data, '__dict__'):
                        df = pd.DataFrame([data.__dict__])
                    else:
                        df = pd.DataFrame([{'data': str(data)}])
                
                # Save as parquet
                df.to_parquet(filepath, engine='pyarrow')
                return True
            except Exception as e:
                print(f"Parquet save failed: {e}")
                return False

        @staticmethod
        def load(filepath: str | Path) -> Any:
            try:
                import pandas as pd
                return pd.read_parquet(filepath)
            except Exception as e:
                print(f"Parquet load failed: {e}")
                return None

    class UniversalSerializer:  # pragma: no cover - placeholder
        """Placeholder universal serializer used for graceful degradation."""

        @staticmethod
        def save(data: Any, filepath: str | Path, format_hint: str | None = None) -> bool:
            if format_hint == "json":
                return JSONSerializer.save(data, filepath)
            elif format_hint == "pickle":
                return PickleSerializer.save(data, filepath)
            elif format_hint == "parquet":
                return ParquetSerializer.save(data, filepath)
            else:
                # Auto-detect format based on file extension
                filepath_str = str(filepath)
                if filepath_str.endswith('.json'):
                    return JSONSerializer.save(data, filepath)
                elif filepath_str.endswith('.pkl') or filepath_str.endswith('.pickle'):
                    return PickleSerializer.save(data, filepath)
                elif filepath_str.endswith('.parquet'):
                    return ParquetSerializer.save(data, filepath)
                else:
                    # Default to JSON for unknown extensions
                    return JSONSerializer.save(data, filepath)

        @staticmethod
        def load(filepath: str | Path) -> Any:
            filepath_str = str(filepath)
            
            # Try to load based on file extension
            if filepath_str.endswith('.json'):
                return JSONSerializer.load(filepath)
            elif filepath_str.endswith('.pkl') or filepath_str.endswith('.pickle'):
                return PickleSerializer.load(filepath)
            elif filepath_str.endswith('.parquet'):
                return ParquetSerializer.load(filepath)
            else:
                # Try different formats in order of preference
                result = JSONSerializer.load(filepath)
                if result is not None:
                    return result
                
                result = PickleSerializer.load(filepath)
                if result is not None:
                    return result
                
                result = ParquetSerializer.load(filepath)
                if result is not None:
                    return result
                
                return None

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
