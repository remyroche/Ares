"""
Artifact naming utilities with timestamp and bot version support.

This module provides standardized naming conventions for artifacts
created throughout the Ares pipeline, ensuring proper versioning
and timestamping for artifact management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ArtifactNamingManager:
    """Manages artifact naming with timestamp and version information."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the artifact naming manager.
        
        Args:
            config: Configuration dictionary containing bot_version
        """
        self.config = config or {}
        self.bot_version = self.config.get('bot_version', 'aresv1')
        
    def create_artifact_name(
        self, 
        stage: str, 
        sub_pipeline: str, 
        artifact_type: str = "outcome",
        extension: str = "json",
        include_timestamp: bool = True,
        include_version: bool = True
    ) -> str:
        """
        Create a standardized artifact name with timestamp and version.
        
        Args:
            stage: Pipeline stage name
            sub_pipeline: Sub-pipeline name
            artifact_type: Type of artifact (outcome, model, data, etc.)
            extension: File extension
            include_timestamp: Whether to include timestamp
            include_version: Whether to include bot version
            
        Returns:
            Formatted artifact name
        """
        parts = [stage, sub_pipeline, artifact_type]
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            parts.append(timestamp)
            
        if include_version:
            parts.append(self.bot_version)
            
        filename = "_".join(parts) + f".{extension}"
        return filename
    
    def create_model_artifact_name(
        self,
        model_type: str,
        stage: str,
        sub_pipeline: str,
        extension: str = "pkl"
    ) -> str:
        """
        Create a model artifact name with proper versioning.
        
        Args:
            model_type: Type of model (analyst, tactician, ensemble, etc.)
            stage: Pipeline stage
            sub_pipeline: Sub-pipeline name
            extension: File extension
            
        Returns:
            Formatted model artifact name
        """
        return self.create_artifact_name(
            stage=stage,
            sub_pipeline=sub_pipeline,
            artifact_type=f"model_{model_type}",
            extension=extension,
            include_timestamp=True,
            include_version=True
        )
    
    def create_data_artifact_name(
        self,
        data_type: str,
        stage: str,
        sub_pipeline: str,
        extension: str = "parquet"
    ) -> str:
        """
        Create a data artifact name with proper versioning.
        
        Args:
            data_type: Type of data (features, labels, processed, etc.)
            stage: Pipeline stage
            sub_pipeline: Sub-pipeline name
            extension: File extension
            
        Returns:
            Formatted data artifact name
        """
        return self.create_artifact_name(
            stage=stage,
            sub_pipeline=sub_pipeline,
            artifact_type=f"data_{data_type}",
            extension=extension,
            include_timestamp=True,
            include_version=True
        )
    
    def get_latest_artifact(
        self,
        artifact_dir: str,
        stage: str,
        sub_pipeline: str,
        artifact_type: str = "outcome",
        extension: str = "json"
    ) -> Optional[Path]:
        """
        Find the latest artifact matching the criteria.
        
        Args:
            artifact_dir: Directory to search in
            stage: Pipeline stage name
            sub_pipeline: Sub-pipeline name
            artifact_type: Type of artifact
            extension: File extension
            
        Returns:
            Path to the latest artifact or None if not found
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            return None
            
        # Create pattern to match artifacts
        pattern = f"{stage}_{sub_pipeline}_{artifact_type}_*_{self.bot_version}.{extension}"
        matching_files = list(artifact_path.glob(pattern))
        
        if not matching_files:
            return None
            
        # Return the most recent file (by modification time)
        return max(matching_files, key=lambda f: f.stat().st_mtime)
    
    def create_artifact_metadata(
        self,
        stage: str,
        sub_pipeline: str,
        artifact_type: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create metadata for an artifact.
        
        Args:
            stage: Pipeline stage
            sub_pipeline: Sub-pipeline name
            artifact_type: Type of artifact
            additional_metadata: Additional metadata to include
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "stage": stage,
            "sub_pipeline": sub_pipeline,
            "artifact_type": artifact_type,
            "bot_version": self.bot_version,
            "created_at": datetime.now().isoformat(),
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
            
        return metadata


def get_artifact_naming_manager(config: Optional[Dict[str, Any]] = None) -> ArtifactNamingManager:
    """
    Get an artifact naming manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ArtifactNamingManager instance
    """
    return ArtifactNamingManager(config)


# Convenience functions for common operations
def create_outcome_filename(stage: str, sub_pipeline: str, bot_version: str = "aresv1") -> str:
    """Create an outcome filename with timestamp and version."""
    manager = ArtifactNamingManager({"bot_version": bot_version})
    return manager.create_artifact_name(stage, sub_pipeline, "outcome", "json")


def create_model_filename(
    model_type: str, 
    stage: str, 
    sub_pipeline: str, 
    bot_version: str = "aresv1"
) -> str:
    """Create a model filename with timestamp and version."""
    manager = ArtifactNamingManager({"bot_version": bot_version})
    return manager.create_model_artifact_name(model_type, stage, sub_pipeline)


def create_data_filename(
    data_type: str, 
    stage: str, 
    sub_pipeline: str, 
    bot_version: str = "aresv1"
) -> str:
    """Create a data filename with timestamp and version."""
    manager = ArtifactNamingManager({"bot_version": bot_version})
    return manager.create_data_artifact_name(data_type, stage, sub_pipeline)