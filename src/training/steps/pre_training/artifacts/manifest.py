"""Utilities for persisting and discovering training artifacts via a manifest."""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..settings import get_pre_training_settings

# Import enhanced utilities
from src.utils.enhanced_artifact_manager import EnhancedArtifactManager, ArtifactMetadata
from src.utils.common_operations import (
    ensure_directory, safe_json_load, safe_json_dump,
    get_memory_usage, optimize_memory
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

ARTIFACTS_ENV_VAR = "ARES_ARTIFACTS_DIR"

# Setup logging
logger = logging.getLogger(__name__)

def _default_artifacts_dir() -> Path:
    """Return the base directory for artifacts, creating it if necessary."""
    override = os.getenv(ARTIFACTS_ENV_VAR)
    if override:
        base = Path(override).expanduser()
    else:
        base = get_pre_training_settings().artifacts_root
    base.mkdir(parents=True, exist_ok=True)
    return base

@dataclass
class ArtifactManifestEntry:
    """Single artifact manifest entry."""

    logical_name: str
    path: str
    version: str
    checksum: str

    @property
    def resolved_path(self) -> Path:
        """Return the resolved Path for the artifact."""
        return Path(self.path)

class ArtifactManifest:
    """Read/write helper for ``artifacts/manifest.json`` with enhanced functionality."""

    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        tprint_info("🔧 Initializing ArtifactManifest with enhanced utilities")
        self.base_dir = _default_artifacts_dir()
        self.manifest_path = manifest_path or (self.base_dir / "manifest.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, List[ArtifactManifestEntry]] = {}

        # Initialize enhanced artifact manager
        self.enhanced_manager = EnhancedArtifactManager({
            'artifacts_dir': str(self.base_dir),
            'ares_version': 'v2'  # Enhanced version for pre-training
        })

        # Optimize memory before loading
        optimize_memory()
        self._load()
        tprint_success("✅ ArtifactManifest initialized with enhanced utilities")

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load manifest with enhanced error handling and memory optimization."""
        tprint_info("📥 Loading artifact manifest with enhanced utilities")
        try:
            if not self.manifest_path.exists():
                tprint_info("📋 Manifest file doesn't exist, initializing empty manifest")
                self._artifacts = {}
                return

            # Use safe JSON loading with fallback
            raw = safe_json_load(self.manifest_path, default={})
            if raw is None:
                tprint_warning("⚠️ Failed to load manifest, initializing empty")
                self._artifacts = {}
                return

            artifacts = raw.get("artifacts", {}) if isinstance(raw, dict) else {}
            parsed: Dict[str, List[ArtifactManifestEntry]] = {}

            for logical_name, entries in artifacts.items():
                if isinstance(entries, list):
                    parsed[logical_name] = [
                        ArtifactManifestEntry(
                            logical_name=logical_name,
                            path=str(entry.get("path", "")),
                            version=str(entry.get("version", "")),
                            checksum=str(entry.get("checksum", "")),
                        )
                        for entry in entries
                        if isinstance(entry, dict)
                    ]

            self._artifacts = parsed
            tprint_success(f"✅ Manifest loaded successfully with {len(parsed)} artifact types")

        except Exception as e:
            logger.error(f"❌ Failed to load manifest: {e}")
            tprint_error(f"❌ Failed to load manifest: {e}")
            self._artifacts = {}

    def _write(self) -> None:
        """Write manifest with enhanced error handling and atomic operations."""
        tprint_info("💾 Writing artifact manifest with enhanced utilities")
        try:
            # Prepare data for serialization
            serialised = {
                "artifacts": {
                    logical_name: [
                        {
                            "path": entry.path,
                            "version": entry.version,
                            "checksum": entry.checksum,
                        }
                        for entry in entries
                    ]
                    for logical_name, entries in self._artifacts.items()
                },
                "metadata": {
                    "version": "v2",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "memory_usage_mb": get_memory_usage() / (1024 * 1024),
                    "total_artifacts": sum(len(entries) for entries in self._artifacts.values())
                }
            }

            # Use safe JSON dump with atomic write
            success = safe_json_dump(serialised, self.manifest_path)
            if success:
                tprint_success(f"✅ Manifest written successfully with {serialised['metadata']['total_artifacts']} artifacts")
            else:
                tprint_error("❌ Failed to write manifest")

        except Exception as e:
            logger.error(f"❌ Failed to write manifest: {e}")
            tprint_error(f"❌ Failed to write manifest: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def compute_checksum(path: Path) -> str:
        """Compute a stable SHA-256 checksum for ``path``."""
        digest = sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def list_entries(self, logical_name: str) -> List[ArtifactManifestEntry]:
        """Return all manifest entries for ``logical_name``."""
        return list(self._artifacts.get(logical_name, ()))

    def get_latest(self, logical_name: str) -> Optional[ArtifactManifestEntry]:
        """Return the most recent manifest entry for ``logical_name``."""
        entries = self._artifacts.get(logical_name)
        if not entries:
            return None
        return max(entries, key=lambda entry: entry.version)

    def find(self, logical_name: str, version: Optional[str] = None) -> Optional[ArtifactManifestEntry]:
        """Return the entry matching ``logical_name`` and optionally ``version``."""
        if version is None:
            return self.get_latest(logical_name)
        for entry in self._artifacts.get(logical_name, ()):  # pragma: no branch - tiny loop
            if entry.version == version:
                return entry
        return None

    def register(
        self,
        logical_name: str,
        path: Path,
        *,
        version: Optional[str] = None,
        checksum: Optional[str] = None,
    ) -> ArtifactManifestEntry:
        """Register an artifact in the manifest with enhanced functionality and persist the change."""
        tprint_info(f"📝 Registering artifact '{logical_name}' at path {path}")

        try:
            resolved_path = path.resolve()

            # Validate path exists
            if not resolved_path.exists():
                raise FileNotFoundError(f"Artifact path does not exist: {resolved_path}")

            # Generate version if not provided
            if version is None:
                version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Compute checksum if not provided
            if checksum is None:
                tprint_info("🔐 Computing checksum for artifact")
                checksum = self.compute_checksum(resolved_path)
                tprint_success("✅ Checksum computed")

            # Create enhanced artifact entry
            entry = ArtifactManifestEntry(
                logical_name=logical_name,
                path=str(resolved_path),
                version=version,
                checksum=checksum,
            )

            # Update manifest entries
            entries = self._artifacts.setdefault(logical_name, [])
            # Remove any existing entry with the same path to avoid duplication
            entries = [e for e in entries if e.path != entry.path]
            entries.append(entry)
            entries.sort(key=lambda e: e.version)
            self._artifacts[logical_name] = entries

            # Optimize memory before writing
            optimize_memory()

            # Write manifest
            self._write()

            # Also register with enhanced artifact manager for additional capabilities
            try:
                enhanced_metadata = ArtifactMetadata(
                    file_path=str(resolved_path),
                    filename=resolved_path.name,
                    version=version,
                    timestamp=datetime.now(timezone.utc),
                    base_name=logical_name,
                    extension=resolved_path.suffix,
                    size_bytes=resolved_path.stat().st_size,
                    created_at=datetime.now(timezone.utc)
                )
                self.enhanced_manager._artifacts_metadata[logical_name] = enhanced_metadata
                tprint_success(f"✅ Artifact registered with enhanced metadata: {logical_name}")
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced metadata registration failed: {e}")

            return entry

        except Exception as e:
            logger.error(f"❌ Failed to register artifact '{logical_name}': {e}")
            tprint_error(f"❌ Failed to register artifact '{logical_name}': {e}")
            raise

class DataLocator:
    """Resolve artifact storage locations in a consistent manner with enhanced utilities."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        tprint_info("🔧 Initializing DataLocator with enhanced utilities")
        self.base_dir = Path(base_dir) if base_dir is not None else _default_artifacts_dir()
        ensure_directory(str(self.base_dir))  # Use enhanced directory creation
        self.outcomes_dir = self.base_dir / "outcomes"
        ensure_directory(str(self.outcomes_dir))  # Use enhanced directory creation
        tprint_success("✅ DataLocator initialized with enhanced utilities")

    @staticmethod
    def build_logical_name(
        base_name: str,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> str:
        """Compose a logical name incorporating market identifiers."""
        return f"{base_name}/{symbol.upper()}/{exchange.lower()}/{timeframe}"

    def resolve_artifact_path(
        self,
        base_name: str,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        version: Optional[str] = None,
        extension: str = "json",
    ) -> tuple[Path, str]:
        """Return the path and version for an artifact."""
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        safe_base = base_name.replace(" ", "_")
        container = self.outcomes_dir / f"{symbol.upper()}_{exchange.lower()}_{timeframe}"
        container.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_base}_{version}.{extension}"
        return container / filename, version

    def resolve_multiple(
        self,
        base_name: str,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        versions: Iterable[str],
        extension: str = "json",
    ) -> Dict[str, Path]:
        """Convenience helper returning paths for multiple versions."""
        paths: Dict[str, Path] = {}
        for version in versions:
            path, _ = self.resolve_artifact_path(
                base_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                version=version,
                extension=extension,
            )
            paths[version] = path
        return paths
