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
        # Set context for enhanced file naming and klines operations
        self._set_context(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            information=config.get('information'),
            direction=config.get('direction', 'long'),
            model=config.get('model', 'Analyst')
        )
        
        # Load klines data using context
        klines_data = self._load_klines_with_context('1m')
        if klines_data is None:
            # Handle case where klines data not found
            return {'success': False, 'error': 'Klines data not found'}
        
        # Process klines data...
        processed_klines = process_klines_data(klines_data)
        
        # Store processed klines data
        success = self._store_klines_with_context(processed_klines, '1m')
        if not success:
            return {'success': False, 'error': 'Failed to store klines data'}
        
        # Also save as regular artifact for compatibility
        self._save_dataframe(processed_klines, 'processed_klines')
        
        return {'success': True, 'artifacts': ['processed_klines']}

KLINES PARQUET MANAGER INTEGRATION:
===================================

The BaseStep now includes full integration with KlinesParquetManager for efficient
klines data storage and retrieval. Available methods:

1. _store_klines(df, symbol, exchange, interval, batch_id, metadata) - Store klines data
2. _load_klines(symbol, exchange, interval, start_time, end_time, batch_id) - Load klines data
3. _update_klines(df, symbol, exchange, interval, append_mode) - Update klines data
4. _delete_klines(symbol, exchange, interval, batch_id) - Delete klines data
5. _list_available_klines() - List all available klines datasets
6. _get_klines_storage_stats() - Get storage statistics
7. _get_klines_compression_stats() - Get compression statistics
8. _get_klines_optimization_recommendations(df) - Get optimization recommendations

Context-aware methods (use current symbol/exchange from context):
9. _store_klines_with_context(df, interval, batch_id, metadata)
10. _load_klines_with_context(interval, start_time, end_time, batch_id)

The klines manager is automatically configured with:
- ZSTD compression for optimal storage
- Metadata tracking for data integrity
- Hardware optimization integration
- Automatic directory structure management
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import traceback

from src.utils.artifact_manager import ArtifactManager
from src.utils.hardware.unified_hardware_manager import WorkloadType

# Import KlinesParquetManager with error handling
try:
    from src.utils.kline_parquet import KlinesParquetManager, StorageConfig
    KLINES_PARQUET_AVAILABLE = True
except ImportError as e:
    # Fallback for environments without pandas/pyarrow
    KlinesParquetManager = None
    StorageConfig = None
    KLINES_PARQUET_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"KlinesParquetManager not available: {e}")
# Enhanced hardware optimization imports
try:
    from src.utils.hardware import (
        get_integrated_hardware_manager, IntegratedHardwareConfig,
        m1_optimized, memory_optimized, optimize_dataframe, force_cleanup,
        WorkloadCategory, OptimizationLevel, get_memory_stats
    )
    from src.utils.hardware.memory_optimized_decorators import (
        MemoryOptimizationLevel, comprehensive_memory_optimization,
        memory_efficient, OptimizationConfig
    )
    from src.utils.hardware.optimization_decorators import (
        smart_cache, auto_optimize, performance_tracked, memory_efficient, OptimizationConfig
    )
except ImportError:
    # Fallback to minimal hardware module
    from src.utils.hardware_minimal import (
        get_integrated_hardware_manager, IntegratedHardwareConfig,
        m1_optimized, memory_optimized, optimize_dataframe, force_cleanup,
        WorkloadCategory, OptimizationLevel, get_memory_stats,
        MemoryOptimizationLevel, memory_efficient, OptimizationConfig,
        smart_cache, auto_optimize, performance_tracked
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
    
    @memory_optimized(optimization_level='balanced')
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base step with enhanced artifact management and hardware optimization.
        
        Args:
            step_name: Unique name for this step (used for artifact paths and outcomes)
            config: Optional configuration dictionary for artifact manager
        """
        self.step_name = step_name
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize hardware optimization for all steps
        hardware_config = IntegratedHardwareConfig(
            enable_automatic_optimization=True,
            enable_caching=True,
            enable_memory_monitoring=True,
            memory_limit_gb=4.0,
            cache_memory_limit_mb=256.0
        )
        self.hardware_manager = get_integrated_hardware_manager(hardware_config)
        
        # Initialize artifact manager with enhanced configuration
        artifact_config = config or {}
        artifact_config.update({
            'hardware_optimization': True,
            'memory_optimization': True,
            'compression': 'auto'
        })
        self.artifact_manager = ArtifactManager(config=artifact_config)
        
        # Initialize KlinesParquetManager for klines data operations (if available)
        if KLINES_PARQUET_AVAILABLE:
            klines_config = StorageConfig(
                base_dir=str(self.artifact_manager.base_dir / "klines_data"),
                compression="zstd",
                compression_level=3,
                enable_metadata=True,
                enable_validation=True
            )
            self.klines_manager = KlinesParquetManager(config=klines_config)
            self.logger.info("✅ KlinesParquetManager initialized")
        else:
            self.klines_manager = None
            self.logger.warning("⚠️ KlinesParquetManager not available (pandas/pyarrow required)")
        
        # Integrate hardware manager with artifact manager
        self.artifact_manager._hardware_manager = self.hardware_manager
        
        # Set up artifact manager context with step-category organization
        self.artifact_manager.set_context(
            step_name=step_name,
            datetime_param=datetime.now()
        )
        
        # Ensure proper directory structure
        self._ensure_directory_structure()
        
        # Ensure all step category directories exist
        self.artifact_manager.ensure_step_category_directories()
        
        if self._is_klines_available():
            self.logger.info(f"🔧 BaseStep initialized: {step_name} with enhanced artifact management, klines parquet management, and hardware optimization")
        else:
            self.logger.info(f"🔧 BaseStep initialized: {step_name} with enhanced artifact management and hardware optimization (klines parquet management not available)")
    
    @memory_efficient(
        memory_threshold_mb=100.0,
        enable_compression=True,
        optimization_level=OptimizationLevel.BALANCED
    )
    def _save_dataframe(self, df: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Convenience method to save a DataFrame with automatic optimization and hardware acceleration.
        
        Args:
            df: DataFrame to save
            name: Name for the artifact
            metadata: Optional metadata
            
        Returns:
            Path where artifact was saved
        """
        # Optimize DataFrame with hardware manager
        optimized_df = self.hardware_manager.optimize_dataframe(df)
        return self._save_enhanced_artifact(optimized_df, name, "data", metadata)
    
    @smart_cache(ttl=1800)
    def _load_dataframe(self, name: str) -> Any:
        """
        Convenience method to load a DataFrame with fallback support and memory optimization.
        
        Args:
            name: Name of the artifact to load
            
        Returns:
            Loaded DataFrame or None if not found
        """
        data = self._get_enhanced_artifact(name, "data")
        if data is not None:
            # Apply hardware optimization to loaded data
            return self.hardware_manager.optimize_dataframe(data)
        return data
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
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
    
    @smart_cache(ttl=1800)
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
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _store_klines(self, df: Any, symbol: str, exchange: str, interval: str, 
                     batch_id: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Convenience method to store klines data using KlinesParquetManager.
        
        Args:
            df: DataFrame containing klines data
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "binance")
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return False
            
        try:
            # Optimize DataFrame with hardware manager
            optimized_df = self.hardware_manager.optimize_dataframe(df)
            
            # Store using KlinesParquetManager
            success = self.klines_manager.store_klines(
                df=optimized_df,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                batch_id=batch_id,
                metadata=metadata
            )
            
            if success:
                self.logger.info(f"✅ Klines data stored: {symbol} {exchange} {interval}")
            else:
                self.logger.error(f"❌ Failed to store klines data: {symbol} {exchange} {interval}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error storing klines data: {e}")
            return False
    
    @smart_cache(ttl=1800)
    def _load_klines(self, symbol: str, exchange: str, interval: str, 
                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                    batch_id: Optional[str] = None) -> Any:
        """
        Convenience method to load klines data using KlinesParquetManager.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load
            
        Returns:
            DataFrame containing klines data or None if not found
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return None
            
        try:
            # Load using KlinesParquetManager
            df = self.klines_manager.load_klines(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                batch_id=batch_id
            )
            
            if df is not None and not df.empty:
                # Apply hardware optimization to loaded data
                optimized_df = self.hardware_manager.optimize_dataframe(df)
                self.logger.info(f"✅ Klines data loaded: {symbol} {exchange} {interval} ({len(optimized_df)} records)")
                return optimized_df
            else:
                self.logger.warning(f"⚠️ No klines data found: {symbol} {exchange} {interval}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading klines data: {e}")
            return None
    
    def _update_klines(self, df: Any, symbol: str, exchange: str, interval: str, 
                      append_mode: bool = True) -> bool:
        """
        Convenience method to update klines data using KlinesParquetManager.
        
        Args:
            df: New klines data
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            append_mode: If True, append to existing data; if False, replace
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return False
            
        try:
            # Optimize DataFrame with hardware manager
            optimized_df = self.hardware_manager.optimize_dataframe(df)
            
            # Update using KlinesParquetManager
            success = self.klines_manager.update_klines(
                df=optimized_df,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                append_mode=append_mode
            )
            
            if success:
                self.logger.info(f"✅ Klines data updated: {symbol} {exchange} {interval}")
            else:
                self.logger.error(f"❌ Failed to update klines data: {symbol} {exchange} {interval}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error updating klines data: {e}")
            return False
    
    def _delete_klines(self, symbol: str, exchange: str, interval: str, 
                      batch_id: Optional[str] = None) -> bool:
        """
        Convenience method to delete klines data using KlinesParquetManager.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            batch_id: Optional specific batch to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return False
            
        try:
            # Delete using KlinesParquetManager
            success = self.klines_manager.delete_klines(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                batch_id=batch_id
            )
            
            if success:
                self.logger.info(f"✅ Klines data deleted: {symbol} {exchange} {interval}")
            else:
                self.logger.error(f"❌ Failed to delete klines data: {symbol} {exchange} {interval}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting klines data: {e}")
            return False
    
    def _list_available_klines(self) -> List[Dict[str, Any]]:
        """
        Convenience method to list available klines data using KlinesParquetManager.
        
        Returns:
            List of dictionaries containing available klines data information
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return []
            
        try:
            available_data = self.klines_manager.list_available_data()
            self.logger.info(f"📊 Found {len(available_data)} klines datasets")
            return available_data
            
        except Exception as e:
            self.logger.error(f"❌ Error listing available klines data: {e}")
            return []
    
    def _get_klines_storage_stats(self) -> Dict[str, Any]:
        """
        Convenience method to get klines storage statistics using KlinesParquetManager.
        
        Returns:
            Dictionary containing storage statistics
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return {}
            
        try:
            stats = self.klines_manager.get_storage_stats()
            self.logger.info(f"📊 Klines storage stats: {stats.get('total_files', 0)} files, "
                           f"{stats.get('total_size_mb', 0):.2f}MB, "
                           f"{stats.get('total_records', 0)} records")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting klines storage stats: {e}")
            return {}
    
    def _get_klines_optimization_recommendations(self, df: Any) -> Dict[str, Any]:
        """
        Convenience method to get klines optimization recommendations using KlinesParquetManager.
        
        Args:
            df: DataFrame to analyze for optimization recommendations
            
        Returns:
            Dictionary containing optimization recommendations
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return {}
            
        try:
            recommendations = self.klines_manager.get_optimization_recommendations(df)
            self.logger.info(f"🔧 Klines optimization recommendations: {recommendations.get('compression', 'unknown')} compression, "
                           f"row group size: {recommendations.get('row_group_size', 'unknown')}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error getting klines optimization recommendations: {e}")
            return {}
    
    def _get_klines_compression_stats(self) -> Dict[str, Any]:
        """
        Convenience method to get klines compression statistics using KlinesParquetManager.
        
        Returns:
            Dictionary containing compression statistics
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return {}
            
        try:
            stats = self.klines_manager.get_compression_stats()
            self.logger.info(f"📊 Klines compression stats: {stats.get('overall_compression_ratio', 0):.1f}% compression ratio")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting klines compression stats: {e}")
            return {}
    
    @abstractmethod
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
        Set the artifact manager and klines manager context for enhanced file naming and path management.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            information: Information type
            direction: Trading direction (long/short)
            model: Model type (Analyst/Tactician)
        """
        # Set artifact manager context
        self.artifact_manager.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            information=information,
            direction=direction,
            model=model
        )
        
        # Store context for klines operations (KlinesParquetManager uses these directly)
        self._current_symbol = symbol
        self._current_exchange = exchange
        self._current_direction = direction
        self._current_model = model
        self._current_information = information
        
        self.logger.info(f"📁 Context set: symbol={symbol}, exchange={exchange}, information={information}, direction={direction}, model={model}")
    
    def _is_klines_available(self) -> bool:
        """
        Check if KlinesParquetManager is available.
        
        Returns:
            True if klines manager is available, False otherwise
        """
        return KLINES_PARQUET_AVAILABLE and self.klines_manager is not None
    
    def _get_klines_context(self) -> Dict[str, Optional[str]]:
        """
        Get the current klines context for easy access in step implementations.
        
        Returns:
            Dictionary containing current context (symbol, exchange, direction, model, information)
        """
        return {
            'symbol': getattr(self, '_current_symbol', None),
            'exchange': getattr(self, '_current_exchange', None),
            'direction': getattr(self, '_current_direction', 'long'),
            'model': getattr(self, '_current_model', 'Analyst'),
            'information': getattr(self, '_current_information', None)
        }
    
    def _store_klines_with_context(self, df: Any, interval: str, 
                                 batch_id: Optional[str] = None, 
                                 metadata: Optional[Dict] = None) -> bool:
        """
        Store klines data using current context (symbol, exchange from context).
        
        Args:
            df: DataFrame containing klines data
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return False
            
        context = self._get_klines_context()
        symbol = context.get('symbol')
        exchange = context.get('exchange')
        
        if not symbol or not exchange:
            self.logger.error("❌ Cannot store klines: symbol and exchange must be set in context")
            return False
        
        return self._store_klines(df, symbol, exchange, interval, batch_id, metadata)
    
    def _load_klines_with_context(self, interval: str, 
                                start_time: Optional[datetime] = None, 
                                end_time: Optional[datetime] = None,
                                batch_id: Optional[str] = None) -> Any:
        """
        Load klines data using current context (symbol, exchange from context).
        
        Args:
            interval: Data interval (e.g., "1m")
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load
            
        Returns:
            DataFrame containing klines data or None if not found
        """
        if not self._is_klines_available():
            self.logger.error("❌ KlinesParquetManager not available (pandas/pyarrow required)")
            return None
            
        context = self._get_klines_context()
        symbol = context.get('symbol')
        exchange = context.get('exchange')
        
        if not symbol or not exchange:
            self.logger.error("❌ Cannot load klines: symbol and exchange must be set in context")
            return None
        
        return self._load_klines(symbol, exchange, interval, start_time, end_time, batch_id)
    
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
    
    def _get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics including artifact, klines, and hardware metrics.
        
        Returns:
            Dictionary containing comprehensive statistics
        """
        try:
            stats = {
                'step_name': self.step_name,
                'performance_metrics': self._get_performance_metrics(),
                'memory_analytics': self._get_memory_analytics(),
                'hardware_stats': self._get_hardware_stats(),
                'context': self._get_klines_context(),
                'klines_available': self._is_klines_available()
            }
            
            # Add klines stats only if available
            if self._is_klines_available():
                stats['klines_storage_stats'] = self._get_klines_storage_stats()
                stats['klines_compression_stats'] = self._get_klines_compression_stats()
            else:
                stats['klines_storage_stats'] = {'error': 'KlinesParquetManager not available'}
                stats['klines_compression_stats'] = {'error': 'KlinesParquetManager not available'}
            
            self.logger.info(f"📊 Comprehensive stats generated for {self.step_name}")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get comprehensive stats: {e}")
            return {'error': str(e)}
    
    def _clear_cache(self) -> None:
        """
        Clear the artifact manager cache and hardware caches.
        """
        try:
            self.artifact_manager.clear_cache()
            self.hardware_manager.clear_all_caches()
            force_cleanup()
            self.logger.info("🧹 Artifact and hardware caches cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _optimize_dataframe(self, df) -> Any:
        """
        Optimize DataFrame using hardware acceleration.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        if df is None:
            return df
        try:
            return self.hardware_manager.optimize_dataframe(df)
        except Exception as e:
            self.logger.warning(f"Hardware optimization failed, using fallback: {e}")
            return optimize_dataframe(df)
    
    @smart_cache(ttl=1800)
    def _get_hardware_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive hardware statistics.
        
        Returns:
            Dictionary containing hardware performance metrics
        """
        try:
            return self.hardware_manager.get_performance_metrics()
        except Exception as e:
            self.logger.warning(f"Failed to get hardware stats: {e}")
            return {}
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the step with error handling and outcome generation with hardware optimization.
        
        This is the main entry point called by the launcher.
        Now includes enhanced artifact management, performance monitoring, and hardware optimization.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with outcome report path
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Starting execution of {self.step_name}")
            
            # Optimize hardware for step execution
            if self.hardware_manager is not None:
                self.hardware_manager.optimize_for_workload(WorkloadType.DATA_PROCESSING)
            else:
                self.logger.warning("⚠️ Hardware manager not available, skipping hardware optimization")
            
            # Set context from config if available
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            information = config.get('information')
            direction = config.get('direction', 'long')
            model = config.get('model', 'Analyst')
            
            if any([symbol, exchange, information]):
                self._set_context(symbol, exchange, information, direction, model)
            
            # Execute the step
            execution_result = await self.execute(config)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_result['execution_time'] = execution_time
            
            # Add performance metrics with hardware and klines stats
            try:
                performance_metrics = self._get_performance_metrics()
                memory_analytics = self._get_memory_analytics()
                hardware_stats = get_memory_stats()
                
                execution_result['performance_metrics'] = performance_metrics
                execution_result['memory_analytics'] = memory_analytics
                execution_result['hardware_stats'] = hardware_stats
                
                # Add klines stats only if available
                if self._is_klines_available():
                    execution_result['klines_storage_stats'] = self._get_klines_storage_stats()
                    execution_result['klines_compression_stats'] = self._get_klines_compression_stats()
                else:
                    execution_result['klines_available'] = False
                    
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
        finally:
            # Force cleanup after step execution
            force_cleanup()
    
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
