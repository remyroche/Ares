"""Utility helpers for persisting component artifacts.

This module centralizes artifact persistence logic so that components can
delegate saving responsibilities to a single implementation. The helper
generates deterministic file names based on content hashes to provide
idempotent persistence and returns a structured :class:`SaveReport` with
metadata about the saved artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

try:  # Optional heavy dependencies
    import pandas as pd  # type: ignore

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas might be unavailable in CI
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - numpy might be unavailable in CI
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

JsonSerializer = Callable[[Any], Any]

@dataclass(frozen=True)
class SaveReport:
    """Structured information returned after saving artifacts."""

    paths: Mapping[str, str]
    bytes: Mapping[str, int]
    duration: float
    checksum: Mapping[str, str]
    correlation_id: str

def persist_artifacts(
    *,
    component_name: str,
    artifacts: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]],
    base_dir: Path,
    logger: logging.Logger,
    json_serializer: Optional[JsonSerializer] = None,
    correlation_id: Optional[str] = None,
) -> SaveReport:
    """Persist artifacts to ``base_dir`` and return a :class:`SaveReport`.

    Args:
        component_name: Name of the component producing the artifacts.
        artifacts: Mapping of artifact names to their payloads.
        metadata: Optional metadata associated with this save operation.
        base_dir: Directory where artifacts will be saved. The directory is
            created if necessary.
        logger: Logger used for emitting structured save logs.
        json_serializer: Optional serializer for non-standard JSON objects.
        correlation_id: Optional correlation identifier. If not provided a new
            UUID4 string is generated.

    Returns:
        SaveReport containing file paths, byte counts, checksums, and timing
        information for the persistence operation.
    """

    serializer = json_serializer or _default_json_serializer
    correlation = correlation_id or str(uuid.uuid4())
    base_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    path_map: Dict[str, str] = {}
    byte_map: Dict[str, int] = {}
    checksum_map: Dict[str, str] = {}
    skipped_map: Dict[str, bool] = {}

    for artifact_name, artifact_data in artifacts.items():
        path, size, checksum, skipped = _persist_single_artifact(
            component_name=component_name,
            artifact_name=artifact_name,
            payload=artifact_data,
            metadata=metadata,
            base_dir=base_dir,
            serializer=serializer,
        )
        path_map[artifact_name] = str(path)
        byte_map[artifact_name] = size
        checksum_map[artifact_name] = checksum
        skipped_map[artifact_name] = skipped

    metadata_path, metadata_size, metadata_checksum, metadata_skipped = _persist_metadata(
        component_name=component_name,
        metadata=metadata,
        correlation_id=correlation,
        base_dir=base_dir,
        serializer=serializer,
    )
    path_map["metadata"] = str(metadata_path)
    byte_map["metadata"] = metadata_size
    checksum_map["metadata"] = metadata_checksum
    skipped_map["metadata"] = metadata_skipped

    duration = perf_counter() - start
    report = SaveReport(
        paths=dict(path_map),
        bytes=dict(byte_map),
        duration=duration,
        checksum=dict(checksum_map),
        correlation_id=correlation,
    )

    log_payload = {
        "event": "artifact_save",
        "component": component_name,
        "correlation_id": correlation,
        "artifacts": list(artifacts.keys()),
        "paths": path_map,
        "bytes": byte_map,
        "checksum": checksum_map,
        "skipped": skipped_map,
        "duration_sec": duration,
    }
    if metadata:
        log_payload["metadata_keys"] = list(metadata.keys())

    try:
        logger.info(json.dumps(log_payload, default=serializer, ensure_ascii=False))
    except TypeError:
        # Fall back to repr for non-serializable log payloads.
        logger.info({**log_payload, "note": "log serialization fallback"})

    return report

def _persist_single_artifact(
    *,
    component_name: str,
    artifact_name: str,
    payload: Any,
    metadata: Optional[Mapping[str, Any]],
    base_dir: Path,
    serializer: JsonSerializer,
) -> Tuple[Path, int, str, bool]:
    extension, content_bytes = _serialize_payload(payload, metadata, serializer)
    checksum = hashlib.sha256(content_bytes).hexdigest()
    filename = f"{component_name}_{artifact_name}_{checksum[:12]}.{extension}"
    file_path = base_dir / filename
    skipped = file_path.exists()

    if not skipped:
        mode = "wb"
        with open(file_path, mode) as handle:
            handle.write(content_bytes)

    return file_path, len(content_bytes), checksum, skipped

def _persist_metadata(
    *,
    component_name: str,
    metadata: Optional[Mapping[str, Any]],
    correlation_id: str,
    base_dir: Path,
    serializer: JsonSerializer,
) -> Tuple[Path, int, str, bool]:
    payload = {
        "component": component_name,
        "correlation_id": correlation_id,
        "metadata": dict(metadata or {}),
    }
    extension, content_bytes = "json", json.dumps(payload, indent=2, default=serializer, ensure_ascii=False).encode("utf-8")
    checksum = hashlib.sha256(content_bytes).hexdigest()
    filename = f"{component_name}_metadata_{checksum[:12]}.json"
    file_path = base_dir / filename
    skipped = file_path.exists()

    if not skipped:
        with open(file_path, "wb") as handle:
            handle.write(content_bytes)

    return file_path, len(content_bytes), checksum, skipped

def _serialize_payload(
    payload: Any,
    metadata: Optional[Mapping[str, Any]],
    serializer: JsonSerializer,
) -> Tuple[str, bytes]:
    if PANDAS_AVAILABLE and isinstance(payload, pd.DataFrame):  # pragma: no cover - requires pandas
        buffer = BytesIO()
        payload.to_parquet(buffer)
        return "parquet", buffer.getvalue()

    if NUMPY_AVAILABLE and isinstance(payload, np.ndarray):  # pragma: no cover - requires numpy
        buffer = BytesIO()
        np.save(buffer, payload)
        return "npy", buffer.getvalue()

    if isinstance(payload, bytes):
        return "bin", payload

    if isinstance(payload, str):
        return "txt", payload.encode("utf-8")

    if isinstance(payload, (int, float, bool)):
        structured = {"value": payload, "metadata": dict(metadata or {})}
        content = json.dumps(structured, indent=2, default=serializer, ensure_ascii=False)
        return "json", content.encode("utf-8")

    # Default to JSON serialization for lists, dicts, and custom objects.
    content = json.dumps(payload, indent=2, default=serializer, ensure_ascii=False)
    return "json", content.encode("utf-8")

def _default_json_serializer(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except TypeError:
            pass

    return str(obj)
