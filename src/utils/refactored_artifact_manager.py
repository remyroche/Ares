"""Refactored Artifact Manager.

Simplified artifact manager that uses separate classes for different responsibilities.
"""

import asyncio
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .artifact_storage import ArtifactStorage
from .compression_manager import CompressionManager, CompressionConfig
from .cache_manager import CacheManager, CacheConfig
from .memory_manager import MemoryManager, MemoryConfig
from .path_manager import PathManager
from .logger import system_logger


@dataclass
class ArtifactConfig:
    """Configuration for the artifact manager."""
    base_dir: str = "artifacts"
    enable_compression: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_thread_safety: bool = True
    max_cache_size_mb: float = 512.0
    max_memory_mb: float = 2000.0
    spill_threshold_mb: float = 150.0


class RefactoredArtifactManager:
    """Simplified artifact manager with separated responsibilities."""
    
    def __init__(self, config: ArtifactConfig):
        """Initialize the artifact manager.
        
        Args:
            config: Configuration for the artifact manager
        """
        self.config = config
        self.logger = system_logger.getChild("RefactoredArtifactManager")
        
        # Initialize base directory
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._storage = ArtifactStorage(self.base_dir)
        self._path_manager = PathManager(self.base_dir)
        
        # Initialize optional components
        if config.enable_compression:
            compression_config = CompressionConfig()
            self._compression = CompressionManager(compression_config)
        else:
            self._compression = None
        
        if config.enable_caching:
            cache_config = CacheConfig(
                max_size_mb=config.max_cache_size_mb,
                enable_thread_safety=config.enable_thread_safety
            )
            self._cache = CacheManager(cache_config)
        else:
            self._cache = None
        
        if config.enable_memory_optimization:
            memory_config = MemoryConfig(
                max_memory_mb=config.max_memory_mb,
                spill_threshold_mb=config.spill_threshold_mb
            )
            spill_dir = self.base_dir / "spilled"
            self._memory = MemoryManager(memory_config, spill_dir)
        else:
            self._memory = None
        
        # Thread safety
        if config.enable_thread_safety:
            self._lock = threading.RLock()
            self._async_lock = asyncio.Lock()
        else:
            self._lock = None
            self._async_lock = None
    
    def _lock_context(self):
        """Get lock context manager."""
        if self._lock is not None:
            return self._lock
        return nullcontext()
    
    async def _async_lock_context(self):
        """Get async lock context manager."""
        if self._async_lock is not None:
            return self._async_lock
        return nullcontext()
    
    def set_context(self, step_name: str, symbol: Optional[str] = None, 
                   exchange: Optional[str] = None, datetime: Optional[Any] = None, 
                   information: Optional[str] = None, direction: str = "long", 
                   model: str = "Analyst") -> None:
        """Set the current context for path generation.
        
        Args:
            step_name: Name of the current step
            symbol: Trading symbol
            exchange: Exchange name
            datetime: Current datetime
            information: Additional information
            direction: Trading direction
            model: Model name
        """
        with self._lock_context():
            self._path_manager.set_context(
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
        """Save an artifact.
        
        Args:
            data: Data to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            compression: Compression method
            metadata: Additional metadata
            
        Returns:
            Path where the artifact was saved
        """
        with self._lock_context():
            try:
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Generate path
                file_path = self._path_manager.get_artifact_path(
                    step_name=step_name,
                    key=artifact_name,
                    file_extension="parquet"
                )
                
                # Optimize data if memory manager is available
                if self._memory and hasattr(data, 'memory_usage'):  # DataFrame
                    data = self._memory.optimize_dataframe(data)
                
                # Save artifact
                success = self._storage.save_artifact(
                    data=data,
                    file_path=file_path,
                    artifact_type=artifact_type,
                    metadata=metadata
                )
                
                if not success:
                    raise Exception(f"Failed to save artifact {artifact_name}")
                
                # Cache if enabled
                if self._cache:
                    self._cache.put(artifact_name, data)
                
                # Profile memory usage if memory manager is available
                if self._memory:
                    self._memory.profile_memory_usage(artifact_name, data)
                
                self.logger.info(f"Saved artifact {artifact_name} to {file_path}")
                return str(file_path)
                
            except Exception as e:
                self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
                raise
    
    def get_artifact(self, artifact_name: str, 
                    artifact_type: str = "data") -> Optional[Any]:
        """Retrieve an artifact.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        with self._lock_context():
            try:
                # Check cache first
                if self._cache:
                    cached_data = self._cache.get(artifact_name)
                    if cached_data is not None:
                        self.logger.debug(f"Retrieved {artifact_name} from cache")
                        return cached_data
                
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Find artifact file
                file_path = self._path_manager.find_artifact(
                    step_name=step_name,
                    key=artifact_name,
                    artifact_type=artifact_type
                )
                
                if file_path is None:
                    self.logger.warning(f"Artifact {artifact_name} not found")
                    return None
                
                # Load artifact
                data = self._storage.load_artifact(file_path)
                
                if data is not None:
                    # Cache if enabled
                    if self._cache:
                        self._cache.put(artifact_name, data)
                    
                    # Profile memory usage if memory manager is available
                    if self._memory:
                        self._memory.profile_memory_usage(artifact_name, data)
                    
                    self.logger.info(f"Retrieved artifact {artifact_name} from {file_path}")
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve artifact {artifact_name}: {e}")
                return None
    
    def delete_artifact(self, artifact_name: str, artifact_type: str = "data") -> bool:
        """Delete an artifact.
        
        Args:
            artifact_name: Name of the artifact to delete
            artifact_type: Type of artifact to delete
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock_context():
            try:
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Find artifact file
                file_path = self._path_manager.find_artifact(
                    step_name=step_name,
                    key=artifact_name,
                    artifact_type=artifact_type
                )
                
                if file_path is None:
                    self.logger.warning(f"Artifact {artifact_name} not found for deletion")
                    return False
                
                # Delete from storage
                success = self._storage.delete_artifact(file_path)
                
                # Remove from cache if enabled
                if self._cache:
                    self._cache.remove(artifact_name)
                
                # Remove from memory profiles if memory manager is available
                if self._memory and artifact_name in self._memory._memory_profiles:
                    profile = self._memory._memory_profiles.pop(artifact_name)
                    self._memory._total_memory_mb -= profile.memory_usage_mb
                
                if success:
                    self.logger.info(f"Deleted artifact {artifact_name}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Failed to delete artifact {artifact_name}: {e}")
                return False
    
    def list_artifacts(self, pattern: str = "*") -> list[Path]:
        """List artifacts matching a pattern.
        
        Args:
            pattern: Glob pattern to match
            
        Returns:
            List of matching artifact paths
        """
        return self._storage.list_artifacts(pattern)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self._cache:
            self._cache.clear()
            self.logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Dictionary with statistics from all components
        """
        stats = {
            "config": {
                "base_dir": str(self.base_dir),
                "enable_compression": self.config.enable_compression,
                "enable_caching": self.config.enable_caching,
                "enable_memory_optimization": self.config.enable_memory_optimization,
                "enable_thread_safety": self.config.enable_thread_safety
            }
        }
        
        # Add cache stats
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        
        # Add memory stats
        if self._memory:
            stats["memory"] = self._memory.get_memory_stats()
        
        # Add compression stats
        if self._compression:
            stats["compression"] = self._compression.get_compression_stats()
        
        return stats
    
    def cleanup(self) -> None:
        """Perform cleanup operations."""
        with self._lock_context():
            # Cleanup cache
            if self._cache:
                self._cache.periodic_cleanup()
            
            # Cleanup memory
            if self._memory:
                self._memory.periodic_cleanup()
            
            self.logger.debug("Cleanup completed")
    
    async def run_context(self, run_id: str):
        """Async context manager for automatic cleanup."""
        async with await self._async_lock_context():
            run_dir = self.base_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                yield run_dir
            finally:
                # Auto-cleanup run directory
                try:
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
                    self.logger.info(f"Cleaned up run directory: {run_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup run directory {run_dir}: {e}")