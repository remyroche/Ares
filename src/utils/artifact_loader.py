"""
Artifact loading utilities for the Ares pipeline.

This module provides functionality to load the latest artifacts
with proper versioning and timestamp handling.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from .artifact_naming import get_artifact_naming_manager


class ArtifactLoader:
    """Handles loading of artifacts with version and timestamp management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the artifact loader.
        
        Args:
            config: Configuration dictionary containing bot_version
        """
        self.config = config or {}
        self.bot_version = self.config.get('bot_version', 'aresv1')
        self.naming_manager = get_artifact_naming_manager(config)
        
    def load_latest_outcome(
        self,
        stage: str,
        sub_pipeline: str,
        artifact_dir: str = "outcomes"
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest outcome file for a stage/sub-pipeline.
        
        Args:
            stage: Pipeline stage name
            sub_pipeline: Sub-pipeline name
            artifact_dir: Directory containing artifacts
            
        Returns:
            Outcome data dictionary or None if not found
        """
        artifact_file = self.naming_manager.get_latest_artifact(
            artifact_dir, stage, sub_pipeline, "outcome", "json"
        )
        
        if not artifact_file or not artifact_file.exists():
            return None
            
        try:
            with open(artifact_file, 'r') as f:
                data = json.load(f)
            
            # Verify bot version compatibility
            artifact_version = data.get('bot_version', 'unknown')
            if artifact_version != self.bot_version:
                print(f"⚠️ Warning: Artifact version mismatch. Expected: {self.bot_version}, Found: {artifact_version}")
            
            return data
        except Exception as e:
            print(f"❌ Error loading outcome file {artifact_file}: {e}")
            return None
    
    def load_latest_model(
        self,
        model_type: str,
        stage: str,
        sub_pipeline: str,
        artifact_dir: str = "artifacts"
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest model artifact.
        
        Args:
            model_type: Type of model (analyst, tactician, ensemble, etc.)
            stage: Pipeline stage
            sub_pipeline: Sub-pipeline name
            artifact_dir: Directory containing artifacts
            
        Returns:
            Model data or None if not found
        """
        artifact_file = self.naming_manager.get_latest_artifact(
            artifact_dir, stage, sub_pipeline, f"model_{model_type}", "pkl"
        )
        
        if not artifact_file or not artifact_file.exists():
            return None
            
        try:
            import pickle
            with open(artifact_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"❌ Error loading model file {artifact_file}: {e}")
            return None
    
    def load_latest_data(
        self,
        data_type: str,
        stage: str,
        sub_pipeline: str,
        artifact_dir: str = "data"
    ) -> Optional[Any]:
        """
        Load the latest data artifact.
        
        Args:
            data_type: Type of data (features, labels, processed, etc.)
            stage: Pipeline stage
            sub_pipeline: Sub-pipeline name
            artifact_dir: Directory containing artifacts
            
        Returns:
            Data or None if not found
        """
        artifact_file = self.naming_manager.get_latest_artifact(
            artifact_dir, stage, sub_pipeline, f"data_{data_type}", "parquet"
        )
        
        if not artifact_file or not artifact_file.exists():
            return None
            
        try:
            import pandas as pd
            data = pd.read_parquet(artifact_file)
            return data
        except Exception as e:
            print(f"❌ Error loading data file {artifact_file}: {e}")
            return None
    
    def list_available_artifacts(
        self,
        stage: Optional[str] = None,
        sub_pipeline: Optional[str] = None,
        artifact_type: Optional[str] = None,
        artifact_dir: str = "artifacts"
    ) -> List[Path]:
        """
        List available artifacts matching the criteria.
        
        Args:
            stage: Pipeline stage name (optional)
            sub_pipeline: Sub-pipeline name (optional)
            artifact_type: Type of artifact (optional)
            artifact_dir: Directory to search in
            
        Returns:
            List of artifact file paths
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            return []
        
        # Build pattern based on provided criteria
        pattern_parts = []
        if stage:
            pattern_parts.append(stage)
        if sub_pipeline:
            pattern_parts.append(sub_pipeline)
        if artifact_type:
            pattern_parts.append(artifact_type)
        
        pattern_parts.append("*")  # timestamp
        pattern_parts.append(self.bot_version)
        pattern_parts.append("*")  # extension
        
        pattern = "_".join(pattern_parts)
        matching_files = list(artifact_path.glob(pattern))
        
        # Sort by modification time (newest first)
        return sorted(matching_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    def get_artifact_info(self, artifact_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get information about an artifact file.
        
        Args:
            artifact_path: Path to the artifact file
            
        Returns:
            Artifact information dictionary or None if error
        """
        try:
            stat = artifact_path.stat()
            return {
                'path': str(artifact_path),
                'name': artifact_path.name,
                'size_bytes': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': artifact_path.suffix
            }
        except Exception as e:
            print(f"❌ Error getting artifact info for {artifact_path}: {e}")
            return None
    
    def cleanup_old_artifacts(
        self,
        artifact_dir: str = "artifacts",
        keep_latest: int = 5,
        older_than_days: int = 30
    ) -> int:
        """
        Clean up old artifacts, keeping only the latest ones.
        
        Args:
            artifact_dir: Directory containing artifacts
            keep_latest: Number of latest artifacts to keep per type
            older_than_days: Remove artifacts older than this many days
            
        Returns:
            Number of artifacts removed
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            return 0
        
        removed_count = 0
        cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 3600)
        
        # Group artifacts by type (stage_sub_pipeline_artifact_type)
        artifact_groups = {}
        for artifact_file in artifact_path.glob(f"*_{self.bot_version}.*"):
            try:
                # Extract type from filename
                parts = artifact_file.stem.split('_')
                if len(parts) >= 3:
                    type_key = '_'.join(parts[:-2])  # Everything except timestamp and version
                    if type_key not in artifact_groups:
                        artifact_groups[type_key] = []
                    artifact_groups[type_key].append(artifact_file)
            except Exception:
                continue
        
        # Process each group
        for type_key, artifacts in artifact_groups.items():
            # Sort by modification time (newest first)
            sorted_artifacts = sorted(artifacts, key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Keep only the latest ones
            to_remove = sorted_artifacts[keep_latest:]
            
            for artifact_file in to_remove:
                try:
                    # Check if it's old enough
                    if artifact_file.stat().st_mtime < cutoff_date:
                        artifact_file.unlink()
                        removed_count += 1
                        print(f"🗑️ Removed old artifact: {artifact_file.name}")
                except Exception as e:
                    print(f"❌ Error removing artifact {artifact_file}: {e}")
        
        return removed_count


def get_artifact_loader(config: Optional[Dict[str, Any]] = None) -> ArtifactLoader:
    """
    Get an artifact loader instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ArtifactLoader instance
    """
    return ArtifactLoader(config)


# Convenience functions
def load_latest_outcome(stage: str, sub_pipeline: str, bot_version: str = "aresv1") -> Optional[Dict[str, Any]]:
    """Load the latest outcome for a stage/sub-pipeline."""
    loader = ArtifactLoader({"bot_version": bot_version})
    return loader.load_latest_outcome(stage, sub_pipeline)


def load_latest_model(model_type: str, stage: str, sub_pipeline: str, bot_version: str = "aresv1") -> Optional[Any]:
    """Load the latest model for a stage/sub-pipeline."""
    loader = ArtifactLoader({"bot_version": bot_version})
    return loader.load_latest_model(model_type, stage, sub_pipeline)


def load_latest_data(data_type: str, stage: str, sub_pipeline: str, bot_version: str = "aresv1") -> Optional[Any]:
    """Load the latest data for a stage/sub-pipeline."""
    loader = ArtifactLoader({"bot_version": bot_version})
    return loader.load_latest_data(data_type, stage, sub_pipeline)