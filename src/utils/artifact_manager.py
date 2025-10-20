"""Enhanced unified artifact and path management for reads/writes.

Provides a single place to resolve data, reports, cache, optimization, and tmp
paths based on configuration. Ensures directories exist before use.

This is a simplified wrapper around the refactored artifact manager components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass

from .refactored_artifact_manager import RefactoredArtifactManager, ArtifactConfig
from .logger import system_logger

# Step category mapping for organized artifact storage
STEP_CATEGORIES = {
    'data_collection': ['step01', 'data_downloader', 'klines_downloading_processing'],
    'market_analysis': ['step02', 'market_analysis', 'sr_detection', 'regime_discovery'],
    'pre_training': ['step02_5', 'feature_generation', 'pre_training'],
    'models_training': ['step03', 'model_training', 'analyst_models', 'tactician_models'],
    'backtesting': ['step04', 'backtesting', 'real_parameters_optimization']
}


def get_step_category(step_name: str) -> str:
    """Determine the category for a step based on its name."""
    step_name_lower = step_name.lower()
    for category, patterns in STEP_CATEGORIES.items():
        if any(pattern.lower() in step_name_lower for pattern in patterns):
            return category
    return 'pre_training'  # Default fallback


class ArtifactManager:
    """Simplified artifact manager that uses refactored components."""
    
    def __init__(self, config: dict):
        """Initialize the artifact manager.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = system_logger.getChild("ArtifactManager")
        
        # Convert config to ArtifactConfig
        artifact_config = ArtifactConfig(
            base_dir=config.get("paths", {}).get("data_dir", "data"),
            enable_compression=config.get("enable_compression", True),
            enable_caching=config.get("enable_caching", True),
            enable_memory_optimization=config.get("enable_memory_optimization", True),
            enable_thread_safety=config.get("enable_thread_safety", True),
            max_cache_size_mb=config.get("max_cache_size_mb", 512.0),
            max_memory_mb=config.get("max_memory_mb", 2000.0),
            spill_threshold_mb=config.get("spill_threshold_mb", 150.0)
        )
        
        # Initialize the refactored manager
        self._manager = RefactoredArtifactManager(artifact_config)
        
        # Store original config for compatibility
        self.config = config
    
    def set_context(self, step_name: str, symbol: Optional[str] = None, 
                   exchange: Optional[str] = None, datetime: Optional[Any] = None, 
                   information: Optional[str] = None, direction: str = "long", 
                   model: str = "Analyst") -> None:
        """Set the current context for path generation."""
        self._manager.set_context(
            step_name=step_name,
            symbol=symbol,
            exchange=exchange,
            datetime=datetime,
            information=information,
            direction=direction,
            model=model
        )
    
    def save(self, data: Any, artifact_name: str, 
             artifact_type: str = "data", 
             compression: str = "auto",
             metadata: Optional[Dict] = None) -> str:
        """Save an artifact."""
        return self._manager.save(
            data=data,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            compression=compression,
            metadata=metadata
        )
    
    def get_artifact(self, artifact_name: str, 
                    artifact_type: str = "data") -> Optional[Any]:
        """Retrieve an artifact."""
        return self._manager.get_artifact(
            artifact_name=artifact_name,
            artifact_type=artifact_type
        )
    
    def delete_artifact(self, artifact_name: str, artifact_type: str = "data") -> bool:
        """Delete an artifact."""
        return self._manager.delete_artifact(
            artifact_name=artifact_name,
            artifact_type=artifact_type
        )
    
    def list_artifacts(self, pattern: str = "*") -> list[Path]:
        """List artifacts matching a pattern."""
        return self._manager.list_artifacts(pattern)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._manager.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return self._manager.get_stats()
    
    def cleanup(self) -> None:
        """Perform cleanup operations."""
        self._manager.cleanup()
    
    async def run_context(self, run_id: str):
        """Async context manager for automatic cleanup."""
        async for result in self._manager.run_context(run_id):
            yield result
    
    # Compatibility methods for existing code
    def get_data_dir(self, *subdirs: str) -> Path:
        """Get data directory path."""
        return self._manager.base_dir / "data" / Path(*subdirs)
    
    def get_reports_dir(self, *subdirs: str) -> Path:
        """Get reports directory path."""
        return self._manager.base_dir / "reports" / Path(*subdirs)
    
    def get_cache_dir(self, *subdirs: str) -> Path:
        """Get cache directory path."""
        return self._manager.base_dir / "cache" / Path(*subdirs)
    
    def get_optimization_dir(self, *subdirs: str) -> Path:
        """Get optimization directory path."""
        return self._manager.base_dir / "optimization" / Path(*subdirs)
    
    def get_tmp_dir(self, *subdirs: str) -> Path:
        """Get temporary directory path."""
        return self._manager.base_dir / "tmp" / Path(*subdirs)
    
    def get_tmp_path(self, filename: str) -> Path:
        """Get temporary file path."""
        return self.get_tmp_dir() / filename
    
    def reset_run(self) -> None:
        """Reset run state (compatibility method)."""
        # The refactored manager handles this automatically
        pass
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID (compatibility method)."""
        return None
    
    def get_run_dir(self) -> Optional[Path]:
        """Get current run directory (compatibility method)."""
        return self._manager.base_dir