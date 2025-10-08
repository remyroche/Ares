"""Utilities for persisting and discovering training artifacts via a manifest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..settings import get_pre_training_settings

ARTIFACTS_ENV_VAR = "ARES_ARTIFACTS_DIR"


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
    """Read/write helper for ``artifacts/manifest.json``."""

    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        self.base_dir = _default_artifacts_dir()
        self.manifest_path = manifest_path or (self.base_dir / "manifest.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, List[ArtifactManifestEntry]] = {}
        self._load()

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.manifest_path.exists():
            self._artifacts = {}
            return

        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        artifacts = raw.get("artifacts", {}) if isinstance(raw, dict) else {}
        parsed: Dict[str, List[ArtifactManifestEntry]] = {}
        for logical_name, entries in artifacts.items():
            parsed[logical_name] = [
                ArtifactManifestEntry(
                    logical_name=logical_name,
                    path=str(entry.get("path")),
                    version=str(entry.get("version")),
                    checksum=str(entry.get("checksum")),
                )
                for entry in entries
                if isinstance(entry, dict)
            ]
        self._artifacts = parsed

    def _write(self) -> None:
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
            }
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(serialised, handle, indent=2, sort_keys=True)

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
        """Register an artifact in the manifest and persist the change."""
        resolved_path = path.resolve()
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if checksum is None:
            checksum = self.compute_checksum(resolved_path)

        entry = ArtifactManifestEntry(
            logical_name=logical_name,
            path=str(resolved_path),
            version=version,
            checksum=checksum,
        )

        entries = self._artifacts.setdefault(logical_name, [])
        # Remove any existing entry with the same path to avoid duplication
        entries = [e for e in entries if e.path != entry.path]
        entries.append(entry)
        entries.sort(key=lambda e: e.version)
        self._artifacts[logical_name] = entries
        self._write()
        return entry


class DataLocator:
    """Resolve artifact storage locations in a consistent manner."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else _default_artifacts_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_dir = self.base_dir / "outcomes"
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)

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
