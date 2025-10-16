"""
Enhanced Artifact Manager with Version and Timestamp Support

This module provides enhanced artifact management functionality including:
- Version and timestamp-based artifact naming
- Automatic artifact discovery and selection
- Most recent artifact selection based on timestamp
- Configuration-driven version management
"""

from __future__ import annotations

import os
import re
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .logger import system_logger
from .common_operations import ensure_directory

@dataclass
class ArtifactMetadata:
    """Metadata for an artifact file."""
    file_path: str
    filename: str
    version: str
    timestamp: datetime
    base_name: str
    extension: str
    size_bytes: int
    created_at: datetime

class EnhancedArtifactManager:
    """Enhanced artifact manager with version and timestamp support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced artifact manager.

        Args:
            config: Configuration dictionary with version and path settings
        """
        self.logger = system_logger.getChild("EnhancedArtifactManager")
        self.config = config or {}

        # Get version from config or default to v1
        self.ares_version = self.config.get("ares_version", "v1")

        # Get base paths from config
        self.base_paths = {
            "data": Path(self.config.get("data_dir", "data")),
            "models": Path(self.config.get("model_dir", "models")),
            "artifacts": Path(self.config.get("artifacts_dir", "artifacts")),
            "cache": Path(self.config.get("cache_dir", "data_cache")),
            "output": Path(self.config.get("output_dir", "output"))
        }

        # Ensure directories exist
        for path in self.base_paths.values():
            ensure_directory(str(path))

    def generate_timestamped_filename(
        self,
        base_name: str,
        extension: str = ".pkl",
        version: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Generate a filename with version and timestamp.

        Args:
            base_name: Base name for the file (without extension)
            extension: File extension (e.g., '.pkl', '.parquet', '.json')
            version: Version string (defaults to self.ares_version)
            timestamp: Timestamp (defaults to current time)

        Returns:
            Generated filename with version and timestamp
        """
        if version is None:
            version = self.ares_version

        if timestamp is None:
            timestamp = datetime.now()

        # Format timestamp as YYYYMMDD_HHMMSS
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Generate filename: base_name_version_timestamp.extension
        filename = f"{base_name}_{version}_{timestamp_str}{extension}"

        self.logger.debug(f"Generated timestamped filename: {filename}")
        return filename

    def save_artifact(
        self,
        data: Any,
        base_name: str,
        extension: str = ".pkl",
        directory: str = "artifacts",
        version: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **save_kwargs
    ) -> str:
        """Save an artifact with version and timestamp in filename.

        Args:
            data: Data to save
            base_name: Base name for the file
            extension: File extension
            directory: Directory to save in (key from base_paths)
            version: Version string (defaults to self.ares_version)
            timestamp: Timestamp (defaults to current time)
            **save_kwargs: Additional arguments for the save method

        Returns:
            Path to the saved file
        """
        if directory not in self.base_paths:
            raise ValueError(f"Unknown directory: {directory}. Available: {list(self.base_paths.keys())}")

        # Generate timestamped filename
        filename = self.generate_timestamped_filename(base_name, extension, version, timestamp)

        # Create full path
        save_dir = self.base_paths[directory]
        ensure_directory(str(save_dir))
        file_path = save_dir / filename

        # Save based on extension
        try:
            if extension == ".pkl":
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, **save_kwargs)
            elif extension == ".parquet":
                try:
                    import pandas as pd
                    if hasattr(data, 'to_parquet'):
                        data.to_parquet(file_path, **save_kwargs)
                    else:
                        pd.DataFrame(data).to_parquet(file_path, **save_kwargs)
                except ImportError:
                    raise ValueError("pandas is required for .parquet files")
            elif extension == ".json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, **save_kwargs)
            elif extension == ".joblib":
                import joblib
                joblib.dump(data, file_path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

            self.logger.info(f"✅ Saved artifact: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to save artifact {file_path}: {e}")
            raise

    def find_artifacts(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None
    ) -> List[ArtifactMetadata]:
        """Find all artifacts matching the criteria.

        Args:
            base_name: Base name to search for
            directory: Directory to search in
            version: Version to filter by (None for all versions)
            extension: Extension to filter by (None for all extensions)

        Returns:
            List of ArtifactMetadata objects
        """
        if directory not in self.base_paths:
            raise ValueError(f"Unknown directory: {directory}. Available: {list(self.base_paths.keys())}")

        search_dir = self.base_paths[directory]
        artifacts = []

        # Build search pattern
        if version:
            pattern = f"{base_name}_{version}_*"
        else:
            pattern = f"{base_name}_*_*"

        if extension:
            pattern += extension

        # Search for files
        search_pattern = str(search_dir / pattern)
        matching_files = glob.glob(search_pattern)

        for file_path in matching_files:
            try:
                metadata = self._parse_artifact_filename(file_path)
                if metadata:
                    artifacts.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to parse artifact filename {file_path}: {e}")

        # Sort by timestamp (most recent first)
        artifacts.sort(key=lambda x: x.timestamp, reverse=True)

        self.logger.debug(f"Found {len(artifacts)} artifacts for base_name '{base_name}'")
        return artifacts

    def get_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None
    ) -> Optional[ArtifactMetadata]:
        """Get the most recent artifact matching the criteria.

        Args:
            base_name: Base name to search for
            directory: Directory to search in
            version: Version to filter by (None for all versions)
            extension: Extension to filter by (None for all extensions)

        Returns:
            Most recent ArtifactMetadata or None if not found
        """
        artifacts = self.find_artifacts(base_name, directory, version, extension)
        return artifacts[0] if artifacts else None

    def load_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None
    ) -> Tuple[Any, Optional[ArtifactMetadata]]:
        """Load the most recent artifact.

        Args:
            base_name: Base name to search for
            directory: Directory to search in
            version: Version to filter by (None for all versions)
            extension: Extension to filter by (None for all extensions)

        Returns:
            Tuple of (loaded_data, metadata) or (None, None) if not found
        """
        metadata = self.get_most_recent_artifact(base_name, directory, version, extension)
        if not metadata:
            self.logger.debug(f"No artifacts found for base_name '{base_name}'")
            return None, None

        try:
            data = self._load_artifact_file(metadata.file_path, metadata.extension)
            self.logger.info(f"✅ Loaded most recent artifact: {metadata.filename}")
            return data, metadata
        except Exception as e:
            self.logger.error(f"❌ Failed to load artifact {metadata.file_path}: {e}")
            return None, None

    def _parse_artifact_filename(self, file_path: str) -> Optional[ArtifactMetadata]:
        """Parse an artifact filename to extract metadata.

        Args:
            file_path: Full path to the artifact file

        Returns:
            ArtifactMetadata object or None if parsing fails
        """
        try:
            path_obj = Path(file_path)
            filename = path_obj.name

            # Pattern: base_name_version_YYYYMMDD_HHMMSS.extension
            pattern = r'^(.+)_([^_]+)_(\d{8}_\d{6})\.(.+)$'
            match = re.match(pattern, filename)

            if not match:
                return None

            base_name, version, timestamp_str, extension = match.groups()

            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            # Get file stats
            stat = path_obj.stat()

            return ArtifactMetadata(
                file_path=str(file_path),
                filename=filename,
                version=version,
                timestamp=timestamp,
                base_name=base_name,
                extension=f".{extension}",
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime)
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse filename {file_path}: {e}")
            return None

    def _load_artifact_file(self, file_path: str, extension: str) -> Any:
        """Load an artifact file based on its extension.

        Args:
            file_path: Path to the file
            extension: File extension

        Returns:
            Loaded data
        """
        if extension == ".pkl":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif extension == ".parquet":
            try:
                return pd.read_parquet(file_path)
            except ImportError:
                raise ValueError("pandas is required for .parquet files")
        elif extension == ".json":
            with open(file_path, 'r') as f:
                return json.load(f)
        elif extension == ".joblib":
            return joblib.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    def cleanup_old_artifacts(
        self,
        base_name: str,
        directory: str = "artifacts",
        keep_count: int = 5,
        version: Optional[str] = None
    ) -> List[str]:
        """Clean up old artifacts, keeping only the most recent ones.

        Args:
            base_name: Base name to clean up
            directory: Directory to clean up
            keep_count: Number of recent artifacts to keep
            version: Version to clean up (None for all versions)

        Returns:
            List of deleted file paths
        """
        artifacts = self.find_artifacts(base_name, directory, version)

        if len(artifacts) <= keep_count:
            self.logger.info(f"No cleanup needed for {base_name}: {len(artifacts)} artifacts <= {keep_count}")
            return []

        # Keep the most recent ones, delete the rest
        to_delete = artifacts[keep_count:]
        deleted_files = []

        for artifact in to_delete:
            try:
                os.remove(artifact.file_path)
                deleted_files.append(artifact.file_path)
                self.logger.info(f"🗑️ Deleted old artifact: {artifact.filename}")
            except Exception as e:
                self.logger.error(f"Failed to delete {artifact.file_path}: {e}")

        self.logger.info(f"🧹 Cleaned up {len(deleted_files)} old artifacts for {base_name}")
        return deleted_files

    def get_artifact_info(self, base_name: str, directory: str = "artifacts") -> Dict[str, Any]:
        """Get information about all artifacts for a base name.

        Args:
            base_name: Base name to get info for
            directory: Directory to search in

        Returns:
            Dictionary with artifact information
        """
        artifacts = self.find_artifacts(base_name, directory)

        if not artifacts:
            return {"count": 0, "artifacts": []}

        # Group by version
        by_version = {}
        for artifact in artifacts:
            if artifact.version not in by_version:
                by_version[artifact.version] = []
            by_version[artifact.version].append(artifact)

        return {
            "count": len(artifacts),
            "versions": list(by_version.keys()),
            "most_recent": {
                "filename": artifacts[0].filename,
                "version": artifacts[0].version,
                "timestamp": artifacts[0].timestamp.isoformat(),
                "size_bytes": artifacts[0].size_bytes
            },
            "by_version": {
                version: {
                    "count": len(version_artifacts),
                    "most_recent": {
                        "filename": version_artifacts[0].filename,
                        "timestamp": version_artifacts[0].timestamp.isoformat()
                    }
                }
                for version, version_artifacts in by_version.items()
            }
        }

# Global instance
_artifact_manager: Optional[EnhancedArtifactManager] = None

def get_artifact_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedArtifactManager:
    """Get the global artifact manager instance.

    Args:
        config: Configuration to initialize with (only used on first call)

    Returns:
        EnhancedArtifactManager instance
    """
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = EnhancedArtifactManager(config)
    return _artifact_manager

def initialize_artifact_manager(config: Dict[str, Any]) -> EnhancedArtifactManager:
    """Initialize the global artifact manager with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedArtifactManager instance
    """
    global _artifact_manager
    _artifact_manager = EnhancedArtifactManager(config)
    return _artifact_manager
