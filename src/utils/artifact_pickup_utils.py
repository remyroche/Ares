"""
Artifact Pickup Utilities

This module provides utilities for automatically picking up the most recent
artifacts from previous pipeline stages based on version and timestamp.
"""

from __future__ import annotations

import os
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .logger import system_logger
from .artifact_manager import setup_enhanced_artifact_manager as get_artifact_manager

# Simple ArtifactMetadata class for compatibility
class ArtifactMetadata:
    """Simple metadata class for artifacts."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ArtifactPickupUtils:
    """Utilities for picking up artifacts from previous pipeline stages."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the artifact pickup utilities."""
        self.logger = system_logger.getChild("ArtifactPickupUtils")
        self.artifact_manager = get_artifact_manager(config or {})

    def find_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None
    ) -> Optional[str]:
        """Find the most recent artifact file path.

        Args:
            base_name: Base name of the artifact
            directory: Directory to search in
            version: Version to filter by (None for all versions)
            extension: Extension to filter by (None for all extensions)

        Returns:
            Path to the most recent artifact or None if not found
        """
        metadata = self.artifact_manager.get_most_recent_artifact(
            base_name, directory, version, extension
        )
        return metadata.file_path if metadata else None

    def load_most_recent_artifact(
        self,
        base_name: str,
        directory: str = "artifacts",
        version: Optional[str] = None,
        extension: Optional[str] = None
    ) -> Tuple[Any, Optional[ArtifactMetadata]]:
        """Load the most recent artifact.

        Args:
            base_name: Base name of the artifact
            directory: Directory to search in
            version: Version to filter by (None for all versions)
            extension: Extension to filter by (None for all extensions)

        Returns:
            Tuple of (loaded_data, metadata) or (None, None) if not found
        """
        return self.artifact_manager.load_most_recent_artifact(
            base_name, directory, version, extension
        )

    def find_artifacts_by_pattern(
        self,
        pattern: str,
        directory: str = "artifacts",
        sort_by_time: bool = True
    ) -> List[str]:
        """Find artifacts matching a pattern.

        Args:
            pattern: Glob pattern to match
            directory: Directory to search in
            sort_by_time: Whether to sort by modification time (most recent first)

        Returns:
            List of matching file paths
        """
        if directory not in self.artifact_manager.base_paths:
            self.logger.error(f"Unknown directory: {directory}")
            return []

        search_dir = self.artifact_manager.base_paths[directory]
        search_pattern = str(search_dir / pattern)
        matching_files = glob.glob(search_pattern)

        if sort_by_time:
            # Sort by modification time (most recent first)
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        self.logger.debug(f"Found {len(matching_files)} files matching pattern '{pattern}'")
        return matching_files

    def get_artifact_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about an artifact file.

        Args:
            file_path: Path to the artifact file

        Returns:
            Dictionary with artifact information or None if parsing fails
        """
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None

            # Try to parse as versioned artifact
            metadata = self.artifact_manager._parse_artifact_filename(file_path)
            if metadata:
                return {
                    "file_path": metadata.file_path,
                    "filename": metadata.filename,
                    "version": metadata.version,
                    "timestamp": metadata.timestamp.isoformat(),
                    "base_name": metadata.base_name,
                    "extension": metadata.extension,
                    "size_bytes": metadata.size_bytes,
                    "created_at": metadata.created_at.isoformat(),
                    "is_versioned": True
                }
            else:
                # Fallback for non-versioned files
                stat = path_obj.stat()
                return {
                    "file_path": str(file_path),
                    "filename": path_obj.name,
                    "version": None,
                    "timestamp": None,
                    "base_name": path_obj.stem,
                    "extension": path_obj.suffix,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "is_versioned": False
                }
        except Exception as e:
            self.logger.error(f"Failed to get artifact info for {file_path}: {e}")
            return None

    def list_available_artifacts(
        self,
        directory: str = "artifacts",
        base_name_filter: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """List all available artifacts in a directory.

        Args:
            directory: Directory to list
            base_name_filter: Optional filter for base names

        Returns:
            Dictionary mapping base names to lists of artifact info
        """
        if directory not in self.artifact_manager.base_paths:
            self.logger.error(f"Unknown directory: {directory}")
            return {}

        search_dir = self.artifact_manager.base_paths[directory]
        artifacts_by_base = {}

        # Find all files in the directory
        for file_path in search_dir.rglob("*"):
            if file_path.is_file():
                artifact_info = self.get_artifact_info(str(file_path))
                if artifact_info:
                    base_name = artifact_info["base_name"]

                    # Apply filter if specified
                    if base_name_filter and base_name_filter not in base_name:
                        continue

                    if base_name not in artifacts_by_base:
                        artifacts_by_base[base_name] = []
                    artifacts_by_base[base_name].append(artifact_info)

        # Sort each base name's artifacts by timestamp (most recent first)
        for base_name in artifacts_by_base:
            artifacts_by_base[base_name].sort(
                key=lambda x: x.get("timestamp", ""), reverse=True
            )

        return artifacts_by_base

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
        return self.artifact_manager.cleanup_old_artifacts(
            base_name, directory, keep_count, version
        )

    def get_pipeline_artifacts(
        self,
        pipeline_stage: str,
        artifact_types: List[str],
        directory: str = "artifacts"
    ) -> Dict[str, Optional[str]]:
        """Get the most recent artifacts for a pipeline stage.

        Args:
            pipeline_stage: Name of the pipeline stage
            artifact_types: List of artifact types to look for
            directory: Directory to search in

        Returns:
            Dictionary mapping artifact types to file paths (None if not found)
        """
        artifacts = {}

        for artifact_type in artifact_types:
            base_name = f"{pipeline_stage}_{artifact_type}"
            file_path = self.find_most_recent_artifact(base_name, directory)
            artifacts[artifact_type] = file_path

            if file_path:
                self.logger.debug(f"Found {artifact_type} artifact: {file_path}")
            else:
                self.logger.warning(f"No {artifact_type} artifact found for {pipeline_stage}")

        return artifacts

# Global instance
_pickup_utils: Optional[ArtifactPickupUtils] = None

def get_artifact_pickup_utils() -> ArtifactPickupUtils:
    """Get the global artifact pickup utils instance.

    Returns:
        ArtifactPickupUtils instance
    """
    global _pickup_utils
    if _pickup_utils is None:
        _pickup_utils = ArtifactPickupUtils()
    return _pickup_utils

def find_most_recent_artifact(
    base_name: str,
    directory: str = "artifacts",
    version: Optional[str] = None,
    extension: Optional[str] = None
) -> Optional[str]:
    """Convenience function to find the most recent artifact.

    Args:
        base_name: Base name of the artifact
        directory: Directory to search in
        version: Version to filter by (None for all versions)
        extension: Extension to filter by (None for all extensions)

    Returns:
        Path to the most recent artifact or None if not found
    """
    return get_artifact_pickup_utils().find_most_recent_artifact(
        base_name, directory, version, extension
    )

def load_most_recent_artifact(
    base_name: str,
    directory: str = "artifacts",
    version: Optional[str] = None,
    extension: Optional[str] = None
) -> Tuple[Any, Optional[ArtifactMetadata]]:
    """Convenience function to load the most recent artifact.

    Args:
        base_name: Base name of the artifact
        directory: Directory to search in
        version: Version to filter by (None for all versions)
        extension: Extension to filter by (None for all extensions)

    Returns:
        Tuple of (loaded_data, metadata) or (None, None) if not found
    """
    return get_artifact_pickup_utils().load_most_recent_artifact(
        base_name, directory, version, extension
    )
