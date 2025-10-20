"""
Enhanced Base Step Class for Autonomous Pipeline Steps

This module provides the abstract base class that all pipeline steps must inherit from.
Each step becomes autonomous with standardized artifact management and outcome file generation.

ENHANCED FEATURES:
=================

1. STEP-CATEGORY ORGANIZATION:
   - All artifacts are stored in artifacts/STEP-CATEGORY/ structure
   - Categories: data_collection, market_analysis, pre_training, models_training, backtesting
   - Automatic category detection based on step name patterns

2. MULTIPLE FALLBACK MECHANISMS:
   - Primary: Step-category structure (artifacts/STEP-CATEGORY/)
   - Fallback 1: General artifacts/ directory search
   - Fallback 2: Model type variations (Analyst/Tactician)
   - Fallback 3: Direction variations (long/short)
   - Ensures backward compatibility with existing artifacts

3. ADVANCED ARTIFACT MANAGEMENT:
   - Memory optimization and compression
   - Automatic CSV generation for small datasets (< 2000 rows)
   - Enhanced filename generation with context (symbol, exchange, datetime, etc.)
   - Performance monitoring and metrics collection
   - Lazy loading and spill strategies for large datasets

4. ENHANCED CONTEXT MANAGEMENT:
   - Automatic context setting from config parameters
   - Support for symbol, exchange, information, direction, model context
   - Enhanced file naming with full context information

5. CONVENIENCE METHODS:
   - _save_dataframe() / _load_dataframe() for DataFrame operations
   - _save_model() / _load_model() for model persistence
   - _save_metadata() / _load_metadata() for metadata storage
   - _get_performance_metrics() / _get_memory_analytics() for monitoring

USAGE EXAMPLE:
==============

class MyStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Set context for enhanced file naming
        self._set_context(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            information=config.get('information'),
            direction=config.get('direction', 'long'),
            model=config.get('model', 'Analyst')
        )
        
        # Load data with fallback support
        data = self._load_dataframe('market_data')
        if data is None:
            # Handle case where data not found
            return {'success': False, 'error': 'Data not found'}
        
        # Process data...
        processed_data = process_data(data)
        
        # Save with enhanced features
        self._save_dataframe(processed_data, 'processed_data')
        
        return {'success': True, 'artifacts': ['processed_data']}
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback

from src.utils.artifact_manager import ArtifactManager

# Import hardware optimization utilities
from src.utils.hardware import (
    # Core optimization decorators
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    # Memory management
    memory_optimized, gc_optimized, chunked_processing_auto, MemoryOptimizationLevel,
    # Data optimization
    optimize_dataframe_default, optimize_numpy_array_default,
    # Hardware management
    get_integrated_hardware_manager, track_memory_usage,
    # M1 optimizations
    m1_optimized, WorkloadCategory, OptimizationStrategy
)


class BaseStep(ABC):
    """
    Abstract base class for all autonomous pipeline steps.
    
    Each step must:
    - Inherit from this class
    - Implement the execute() method
    - Use artifact_manager for all data I/O
    - Generate Markdown outcome files
    - Be callable only via launcher (no standalone CLI)
    """
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base step with enhanced artifact management.
        
        Args:
            step_name: Unique name for this step (used for artifact paths and outcomes)
            config: Optional configuration dictionary for artifact manager
        """
        self.step_name = step_name
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize artifact manager with enhanced configuration
        artifact_config = config or {}
        self.artifact_manager = ArtifactManager(config=artifact_config)
        
        # Set up artifact manager context with step-category organization
        self.artifact_manager.set_context(
            step_name=step_name,
            datetime=datetime.now()
        )
        
        # Ensure proper directory structure
        self._ensure_directory_structure()
        
        # Ensure all step category directories exist
        self.artifact_manager.ensure_step_category_directories()
        
        self.logger.info(f"🔧 BaseStep initialized: {step_name} with enhanced artifact management")
    
    @memory_optimized(level=MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=3600, max_size=100)
    def _save_dataframe(self, df: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Convenience method to save a DataFrame with automatic optimization.
        
        Args:
            df: DataFrame to save
            name: Name for the artifact
            metadata: Optional metadata
            
        Returns:
            Path where artifact was saved
        """
        # Optimize DataFrame before saving
        if hasattr(df, 'dtypes'):
            df = optimize_dataframe_default(df)
        return self._save_enhanced_artifact(df, name, "data", metadata)
    
    @memory_optimized(level=MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=1800, max_size=50)
    def _load_dataframe(self, name: str) -> Any:
        """
        Convenience method to load a DataFrame with fallback support.
        
        Args:
            name: Name of the artifact to load
            
        Returns:
            Loaded DataFrame or None if not found
        """
        df = self._get_enhanced_artifact(name, "data")
        if df is not None and hasattr(df, 'dtypes'):
            df = optimize_dataframe_default(df)
        return df
    
    @memory_optimized(level=MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=7200, max_size=20)
    def _save_model(self, model: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Convenience method to save a model with enhanced storage.
        
        Args:
            model: Model to save
            name: Name for the artifact
            metadata: Optional metadata
            
        Returns:
            Path where artifact was saved
        """
        return self._save_enhanced_artifact(model, name, "model", metadata)
    
    @memory_optimized(level=MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=3600, max_size=20)
    def _load_model(self, name: str) -> Any:
        """
        Convenience method to load a model with fallback support.
        
        Args:
            name: Name of the artifact to load
            
        Returns:
            Loaded model or None if not found
        """
        return self._get_enhanced_artifact(name, "model")
    
    def _save_metadata(self, metadata: Dict[str, Any], name: str) -> str:
        """
        Convenience method to save metadata.
        
        Args:
            metadata: Metadata to save
            name: Name for the artifact
            
        Returns:
            Path where artifact was saved
        """
        return self._save_enhanced_artifact(metadata, name, "metadata")
    
    def _load_metadata(self, name: str) -> Any:
        """
        Convenience method to load metadata.
        
        Args:
            name: Name of the artifact to load
            
        Returns:
            Loaded metadata or None if not found
        """
        return self._get_enhanced_artifact(name, "metadata")
    
    @abstractmethod
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic.
        
        Args:
            config: Configuration dictionary containing all necessary parameters
                   (symbol, exchange, timeframes, execution_mode, etc.)
        
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': list of artifact paths/metadata created
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
        """
        pass
    
    def _save_artifact(self, data: Any, artifact_name: str, 
                      artifact_type: str = "data", 
                      compression: str = "auto",
                      metadata: Optional[Dict] = None) -> str:
        """
        Save an artifact using the enhanced artifact manager with step-category organization.
        
        This method uses the most advanced functions from Artifact_manager.py including:
        - Step-category based directory organization
        - Automatic CSV generation for small datasets
        - Enhanced filename generation with context
        - Memory optimization and compression
        
        Args:
            data: Data to save (DataFrame, dict, model, etc.)
            artifact_name: Name for the artifact
            artifact_type: Type of artifact ("data", "model", "metadata", etc.)
            compression: Compression method ("auto", "gzip", "lz4", "none")
            metadata: Additional metadata to store with artifact
            
        Returns:
            Path where artifact was saved
        """
        try:
            # Use the enhanced save method with automatic CSV generation
            artifact_path = self.artifact_manager.save(
                data=data,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                compression=compression,
                metadata=metadata
            )
            self.logger.info(f"✅ Saved artifact: {artifact_name} -> {artifact_path}")
            return artifact_path
        except Exception as e:
            self.logger.error(f"❌ Failed to save artifact {artifact_name}: {e}")
            raise
    
    def _get_artifact(self, artifact_name: str, 
                     artifact_type: str = "data") -> Any:
        """
        Retrieve an artifact using multiple fallback mechanisms for backward compatibility.
        
        This method implements a comprehensive fallback strategy:
        1. Primary: Step-category structure (artifacts/STEP-CATEGORY/)
        2. Fallback 1: General artifacts directory search
        3. Fallback 2: Without model type and direction variations (generic search)
        4. Fallback 3: Fuzzy matching for similar names
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Primary: Try step-category structure
            data = self.artifact_manager.get_artifact(
                artifact_name=artifact_name,
                artifact_type=artifact_type
            )
            if data is not None:
                self.logger.info(f"✅ Retrieved artifact from step-category: {artifact_name}")
                return data
            
            # Fallback 1: Try direct artifacts/ directory search
            data = self._get_artifact_fallback_1(artifact_name, artifact_type)
            if data is not None:
                self.logger.info(f"✅ Retrieved artifact from fallback 1: {artifact_name}")
                return data
            
            # Fallback 2: Try without model type and direction variations
            data = self._get_artifact_fallback_2(artifact_name, artifact_type)
            if data is not None:
                self.logger.info(f"✅ Retrieved artifact from fallback 2: {artifact_name}")
                return data
            
            # Fallback 3: Try fuzzy matching for similar names
            data = self._get_artifact_fallback_3(artifact_name, artifact_type)
            if data is not None:
                self.logger.info(f"✅ Retrieved artifact from fallback 3: {artifact_name}")
                return data
            
            self.logger.warning(f"⚠️ Artifact not found with any fallback method: {artifact_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve artifact {artifact_name}: {e}")
            return None
    
    def _get_artifact_fallback_1(self, artifact_name: str, artifact_type: str) -> Any:
        """
        Fallback 1: Search in general artifacts/ directory.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Use the artifact manager's fallback search
            from src.utils.artifact_manager import get_step_category
            step_category = get_step_category(self.step_name)
            
            # Search in artifacts/ directory recursively
            artifacts_dir = self.artifact_manager._artifacts_dir
            if not artifacts_dir.exists():
                return None
            
            # Search for any file containing the artifact name
            for file_path in artifacts_dir.rglob(f"*{artifact_name}*"):
                if file_path.is_file():
                    data = self.artifact_manager._load_artifact_from_path(file_path)
                    if data is not None:
                        return data
            
            return None
        except Exception as e:
            self.logger.debug(f"Fallback 1 failed for {artifact_name}: {e}")
            return None
    
    def _get_artifact_fallback_2(self, artifact_name: str, artifact_type: str) -> Any:
        """
        Fallback 2: Try without model type and direction variations.
        
        This searches for artifacts without the current model type and direction
        in the filename, providing a more generic search.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Clear model and direction context for generic search
            original_model = self.artifact_manager._current_model
            original_direction = self.artifact_manager._current_direction
            
            # Set generic context (no model, no direction)
            self.artifact_manager._current_model = ""
            self.artifact_manager._current_direction = ""
            
            try:
                # Search with generic context
                data = self.artifact_manager.get_artifact(
                    artifact_name=artifact_name,
                    artifact_type=artifact_type
                )
                if data is not None:
                    return data
                
                # Also try searching with just the artifact name in different locations
                data = self._search_generic_artifact(artifact_name, artifact_type)
                if data is not None:
                    return data
                
            finally:
                # Restore original context
                self.artifact_manager._current_model = original_model
                self.artifact_manager._current_direction = original_direction
            
            return None
        except Exception as e:
            self.logger.debug(f"Fallback 2 failed for {artifact_name}: {e}")
            return None
    
    def _get_artifact_fallback_3(self, artifact_name: str, artifact_type: str) -> Any:
        """
        Fallback 3: Try fuzzy matching for similar names.
        
        This searches for artifacts with similar names using fuzzy matching
        across all directories.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Use the artifact manager's fuzzy search
            data = self.artifact_manager._find_artifact_fuzzy(artifact_name, artifact_type)
            if data is not None:
                return self.artifact_manager._load_artifact_from_path(data)
            
            return None
        except Exception as e:
            self.logger.debug(f"Fallback 3 failed for {artifact_name}: {e}")
            return None
    
    def _search_generic_artifact(self, artifact_name: str, artifact_type: str) -> Any:
        """
        Search for artifact with generic naming (no model/direction context).
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Search in artifacts directory with generic patterns
            artifacts_dir = self.artifact_manager._artifacts_dir
            if not artifacts_dir.exists():
                return None
            
            # Search patterns that don't include model/direction
            search_patterns = [
                f"*{artifact_name}*",
                f"*{artifact_name}*.parquet",
                f"*{artifact_name}*.csv",
                f"*{artifact_name}*.pkl",
                f"*{artifact_name}*.json",
            ]
            
            for pattern in search_patterns:
                for file_path in artifacts_dir.rglob(pattern):
                    if file_path.is_file():
                        # Check if filename contains the artifact name
                        if artifact_name.lower() in file_path.name.lower():
                            # Additional check: ensure it doesn't have model/direction in the name
                            filename_lower = file_path.name.lower()
                            has_model = any(model in filename_lower for model in ['analyst', 'tactician'])
                            has_direction = any(direction in filename_lower for direction in ['long', 'short'])
                            
                            # Prefer files without model/direction context
                            if not has_model and not has_direction:
                                return self.artifact_manager._load_artifact_from_path(file_path)
            
            return None
        except Exception as e:
            self.logger.debug(f"Generic search failed for {artifact_name}: {e}")
            return None
    
    def _set_context(self, symbol: Optional[str] = None, exchange: Optional[str] = None, 
                    information: Optional[str] = None, direction: str = "long", 
                    model: str = "Analyst") -> None:
        """
        Set the artifact manager context for enhanced file naming and path management.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            information: Information type
            direction: Trading direction (long/short)
            model: Model type (Analyst/Tactician)
        """
        self.artifact_manager.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            information=information,
            direction=direction,
            model=model
        )
        self.logger.info(f"📁 Context set: symbol={symbol}, exchange={exchange}, information={information}, direction={direction}, model={model}")
    
    def _save_enhanced_artifact(self, data: Any, artifact_name: str, 
                               artifact_type: str = "data", 
                               metadata: Optional[Dict] = None) -> str:
        """
        Save an artifact using the most advanced features from Artifact_manager.
        
        This method uses store_enhanced() which includes:
        - Memory profiling and optimization
        - Automatic spilling for large datasets
        - Enhanced compression strategies
        - Performance monitoring
        
        Args:
            data: Data to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            metadata: Additional metadata
            
        Returns:
            Path where artifact was saved
        """
        try:
            # Use the enhanced storage method
            success = self.artifact_manager.store_enhanced(
                key=artifact_name,
                data=data,
                metadata=metadata
            )
            
            if success:
                # Get the path where it was saved
                step_category = self.artifact_manager.get_step_category(self.step_name)
                artifact_path = self.artifact_manager._get_enhanced_path(
                    self.step_name, artifact_name, "parquet"
                )
                self.logger.info(f"✅ Enhanced artifact saved: {artifact_name} -> {artifact_path}")
                return str(artifact_path)
            else:
                raise Exception("Enhanced storage failed")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced artifact {artifact_name}: {e}")
            # Fallback to regular save
            return self._save_artifact(data, artifact_name, artifact_type, "auto", metadata)
    
    def _get_enhanced_artifact(self, artifact_name: str, 
                              artifact_type: str = "data") -> Any:
        """
        Retrieve an artifact using the most advanced features from Artifact_manager.
        
        This method uses retrieve_enhanced() which includes:
        - Lazy loading from spilled artifacts
        - Memory-optimized retrieval
        - Performance monitoring
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
        """
        try:
            # Try enhanced retrieval first
            data = self.artifact_manager.retrieve_enhanced(artifact_name)
            if data is not None:
                self.logger.info(f"✅ Enhanced artifact retrieved: {artifact_name}")
                return data
            
            # Fallback to regular retrieval with multiple fallbacks
            return self._get_artifact(artifact_name, artifact_type)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve enhanced artifact {artifact_name}: {e}")
            return None
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the artifact manager.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            return self.artifact_manager.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def _get_memory_analytics(self) -> Dict[str, Any]:
        """
        Get memory analytics from the artifact manager.
        
        Returns:
            Dictionary containing memory analytics
        """
        try:
            return self.artifact_manager.get_memory_analytics()
        except Exception as e:
            self.logger.error(f"Failed to get memory analytics: {e}")
            return {}
    
    def _clear_cache(self) -> None:
        """
        Clear the artifact manager cache.
        """
        try:
            self.artifact_manager.clear_cache()
            self.logger.info("🧹 Artifact cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the step with error handling and outcome generation.
        
        This is the main entry point called by the launcher.
        Now includes enhanced artifact management and performance monitoring.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with outcome report path
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Starting execution of {self.step_name}")
            
            # Set context from config if available
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            information = config.get('information')
            direction = config.get('direction', 'long')
            model = config.get('model', 'Analyst')
            
            if any([symbol, exchange, information]):
                self._set_context(symbol, exchange, information, direction, model)
            
            # Execute the step (async)
            execution_result = await self.execute(config)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_result['execution_time'] = execution_time
            
            # Add performance metrics
            try:
                performance_metrics = self._get_performance_metrics()
                memory_analytics = self._get_memory_analytics()
                
                execution_result['performance_metrics'] = performance_metrics
                execution_result['memory_analytics'] = memory_analytics
            except Exception as e:
                self.logger.warning(f"Failed to get performance metrics: {e}")
            
            # Log completion with enhanced information
            if execution_result.get('success', False):
                self.logger.info(f"✅ Successfully completed {self.step_name} in {execution_time:.2f}s")
                if 'performance_metrics' in execution_result:
                    metrics = execution_result['performance_metrics']
                    self.logger.info(f"📊 Performance: Cache hit ratio: {metrics.get('cache_hit_ratio', 0):.2%}, "
                                   f"Compression savings: {metrics.get('compression_savings_mb', 0):.1f}MB")
            else:
                self.logger.error(f"❌ Failed to complete {self.step_name} after {execution_time:.2f}s")
            
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Step {self.step_name} failed: {str(e)}\n{traceback.format_exc()}"
            
            self.logger.error(error_msg)
            
            # Create failure result
            failure_result = {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'artifacts': [],
                'metrics': {}
            }
            
            return failure_result
    
    def _ensure_directory_structure(self) -> None:
        """
        Ensure the proper directory structure exists for step-category organization.
        
        This method creates the necessary directories in the artifacts/STEP-CATEGORY/ structure.
        """
        try:
            from src.utils.artifact_manager import get_step_category
            
            # Get the step category
            step_category = get_step_category(self.step_name)
            
            # Ensure the artifacts directory exists
            artifacts_dir = self.artifact_manager._artifacts_dir
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure the step category directory exists
            category_dir = artifacts_dir / step_category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"📁 Directory structure ensured: {category_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to ensure directory structure: {e}")


class StepRegistry:
    """
    Registry for all autonomous steps.
    
    Used by the launcher to discover and execute steps.
    """
    
    def __init__(self):
        self._steps: Dict[str, type] = {}
    
    def register(self, step_name: str, step_class: type):
        """
        Register a step class.
        
        Args:
            step_name: Unique name for the step
            step_class: Step class that inherits from BaseStep
        """
        if not issubclass(step_class, BaseStep):
            raise ValueError(f"Step class {step_class} must inherit from BaseStep")
        
        self._steps[step_name] = step_class
        logging.getLogger("ares.registry").info(f"Registered step: {step_name}")
    
    def get_step(self, step_name: str) -> type:
        """
        Get a registered step class.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step class
            
        Raises:
            KeyError: If step is not registered
        """
        if step_name not in self._steps:
            raise KeyError(f"Step '{step_name}' not found in registry. Available steps: {list(self._steps.keys())}")
        
        return self._steps[step_name]
    
    def list_steps(self) -> list:
        """
        List all registered step names.
        
        Returns:
            List of step names
        """
        return list(self._steps.keys())
    
    def is_registered(self, step_name: str) -> bool:
        """
        Check if a step is registered.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if step is registered
        """
        return step_name in self._steps


# Global step registry instance
step_registry = StepRegistry()
