from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ArtifactMetadata:
    file_path: str
    filename: str
    version: str
    timestamp: datetime
    base_name: str
    extension: str
    size_bytes: int
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        parsed = data.copy()
        parsed["timestamp"] = datetime.fromisoformat(parsed["timestamp"]) if parsed.get("timestamp") else datetime.utcnow()
        if parsed.get("created_at"):
            parsed["created_at"] = datetime.fromisoformat(parsed["created_at"])
        if parsed.get("updated_at"):
            parsed["updated_at"] = datetime.fromisoformat(parsed["updated_at"])
        return cls(**parsed)


class EnhancedArtifactManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = artifacts_dir / "enhanced_artifacts_metadata.json"
        self._lock = threading.RLock()
        self._artifacts_metadata: Dict[str, ArtifactMetadata] = {}
        self._load()

    def register_metadata(self, logical_name: str, metadata: ArtifactMetadata) -> None:
        with self._lock:
            self._artifacts_metadata[logical_name] = metadata
            self._persist()

    def get_metadata(self, logical_name: str) -> Optional[ArtifactMetadata]:
        return self._artifacts_metadata.get(logical_name)

    def list_metadata(self) -> Dict[str, ArtifactMetadata]:
        return dict(self._artifacts_metadata)

    def _load(self) -> None:
        if not self._metadata_path.exists():
            return
        try:
            raw = json.loads(self._metadata_path.read_text())
            for key, value in raw.items():
                try:
                    self._artifacts_metadata[key] = ArtifactMetadata.from_dict(value)
                except Exception as exc:
                    self.logger.warning("Failed to load artifact metadata for %s: %s", key, exc)
        except Exception as exc:
            self.logger.error("Failed to load enhanced artifact metadata store: %s", exc)

    def _persist(self) -> None:
        try:
            payload = {key: meta.to_dict() for key, meta in self._artifacts_metadata.items()}
            tmp_path = self._metadata_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2))
            tmp_path.replace(self._metadata_path)
        except Exception as exc:
            self.logger.error("Failed to persist enhanced artifact metadata store: %s", exc)
