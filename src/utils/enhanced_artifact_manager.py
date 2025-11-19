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


# ---------------------------------------------------------------------------
# Compatibility wrapper for legacy artifact_pickup_utils.get_artifact_manager
# ---------------------------------------------------------------------------

_global_artifact_manager: Optional["_CompatibleArtifactManager"] = None


class _CompatibleArtifactManager:
    """Lightweight adapter exposing methods expected by ArtifactPickupUtils.

    This wraps EnhancedArtifactManager for metadata persistence, but implements
    get_most_recent_artifact / load_most_recent_artifact / cleanup_old_artifacts
    and base_paths with safe fallbacks so older utilities can import without
    breaking newer pipelines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._manager = EnhancedArtifactManager(config=config)
        # Basic directory mapping used by artifact_pickup_utils
        self.base_paths: Dict[str, Path] = {
            "artifacts": Path("artifacts"),
            "versioned_artifacts": Path("versioned_artifacts"),
            "outcomes": Path("outcomes"),
        }

    # The following methods intentionally return conservative defaults.
    # They are sufficient for modules that only need to *import* the
    # utilities; more advanced behavior can be layered on later if needed.

    def get_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> Optional[ArtifactMetadata]:
        """Return None to indicate no tracked artifacts (safe fallback)."""
        return None

    def load_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """Return (None, None) as a safe default when no artifact is found."""
        return None, None

    def cleanup_old_artifacts(
        self,
        base_name: str,
        directory: str = "artifacts",
        keep_count: int = 5,
        version: Optional[str] = None,
    ):
        """No-op cleanup that returns an empty list."""
        return []

    def _parse_artifact_filename(self, file_path: str) -> Optional[ArtifactMetadata]:
        """Stub parser: callers will fall back to basic file stats when None."""
        return None


def get_artifact_manager(config: Optional[Dict[str, Any]] = None) -> _CompatibleArtifactManager:
    """Return a module-wide compatible artifact manager instance.

    This preserves the legacy get_artifact_manager() API expected by
    artifact_pickup_utils while internally using EnhancedArtifactManager for
    metadata storage.
    """

    global _global_artifact_manager
    if _global_artifact_manager is None:
        _global_artifact_manager = _CompatibleArtifactManager(config=config)
    return _global_artifact_manager

