# src/training/unified_data_orchestrator.py

"""Unified Data Orchestrator - Single Source of Truth for Data Operations.

This module provides a centralized unified approach to all data operations including:
- Data loading from various sources (cache, unified format, raw files): Intelligent source selection with fallback strategies
- Data merging and consolidation: Efficient merging of multiple data sources with conflict resolution
- Multi-timeframe resampling: Cached resampling operations for improved performance
- Data quality validation and repair: Automated detection and correction of data quality issues
- Memory-efficient processing: Optimized memory usage with garbage collection and monitoring
- Caching and optimization: Intelligent caching strategies for frequently accessed data

This serves as the single source of truth for all data operations across the training pipeline.
Enhanced with comprehensive security and troubleshooting decorators.
"""

import asyncio
import contextlib
import gc
import hashlib
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from src.training.data_sharing_manager import DataSharingManager

# Import existing components
from src.training.steps.unified_data_loader import UnifiedDataLoader
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Import training pipeline decorators for security and troubleshooting
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection, debug_training_step, memory_efficient,
    prevent_data_leakage, quality_gate, resource_monitor,
    secure_data_processing, validate_step_output, validate_step_prerequisites)


class UnifiedDataOrchestrator:
    """Unified Data Orchestrator - Single source of truth for all data operations.

    This orchestrator provides a centralized interface for:
    - Data loading with intelligent fallback strategies
    - Multi-timeframe resampling with caching
    - Data merging and consolidation
    - Quality validation and repair
    - Memory-efficient processing
    - Comprehensive logging and monitoring
    """

    def __init__(self, config: dict[str, Any]) -> None:
        start_time = time.time()

        self.config = config
        self.logger = system_logger.getChild("UnifiedDataOrchestrator")

        # Enable memory tracking for troubleshooting
        tracemalloc.start()

        # Initialize components
        self.data_loader = UnifiedDataLoader(config)
        self.data_sharing_manager = DataSharingManager(config)

        # Configuration
        self.orchestrator_config = config.get("unified_data_orchestrator", {})
        self.enable_caching = self.orchestrator_config.get("enable_caching", True)
        self.enable_memory_optimization = self.orchestrator_config.get(
            "enable_memory_optimization", True)
        self.enable_quality_validation = self.orchestrator_config.get(
            "enable_quality_validation", True)
        self.enable_auto_repair = self.orchestrator_config.get(
            "enable_auto_repair", True)

        # Resampling configuration
        self.resampling_config = self.orchestrator_config.get("resampling", {})
        self.default_timeframes = self.resampling_config.get(
            "default_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.resampling_cache: dict[str, pd.DataFrame] = {}
        self.resampling_cache_size = self.resampling_config.get("cache_size", 100)

        # Quality validation configuration
        self.quality_config = self.orchestrator_config.get("quality_validation", {})
        self.min_data_points = self.quality_config.get("min_data_points", 1000)
        self.max_missing_ratio = self.quality_config.get("max_missing_ratio", 0.1)
        self.max_duplicate_ratio = self.quality_config.get("max_duplicate_ratio", 0.05)

        # Statistics
        self.stats: dict[str, Any] = {
            "total_requests": 0, "cache_hits": 0,
            "cache_misses": 0, "resampling_operations": 0, "quality_repairs": 0,
            "memory_cleanups": 0, "total_data_loaded_gb": 0.0, "operation_times": {},
            "memory_usage_history": [],
        }

        # Initialize cleanup task
        self._cache_cleanup_task = None

        init_time = time.time() - start_time
        self.logger.info(f"UnifiedDataOrchestrator initialized in {init_time:.2f}s")

        # Log initial memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Initial memory usage: {memory_mb:.2f} MB")
        except Exception as e:
            self.logger.warning(f"Could not get initial memory usage: {e}")

    def _log_memory_usage(self, context: str) -> None:
        """Log current memory usage with context."""
        memory_mb = self._get_memory_usage_mb()
        self.stats["memory_usage_history"].append(
            {"timestamp": datetime.now(), "context": context, "memory_mb": memory_mb}
        )
        self.logger.debug(f"Memory usage ({context}): {memory_mb:.2f} MB")

    @memory_efficient(
        min_memory_gb=1.0, min_disk_gb=0.5, required_packages=["pandas", "numpy", "asyncio"],
        enable_garbage_collection=True, memory_monitoring=True, disk_monitoring=True,
        backup_before=False, integrity_checks=True, memory_cleanup=True,
        data_validation=False)
    @resource_monitor(
        memory_threshold_gb=2.0, cpu_threshold_percent=50.0,
        disk_threshold_gb=1.0, monitor_interval=10.0, auto_cleanup=True,
        alert_on_threshold=True, log_resource_usage=True)
    @debug_training_step(
        log_intermediate_results=True, save_debug_artifacts=True, performance_profiling=True,
        error_context_preservation=True)
    @circuit_breaker_protection(
        failure_threshold=3, recovery_timeout=60.0,
        expected_exception=Exception, monitor_interval=10.0)
    @secure_data_processing(
        format_validation=False)
    @quality_gate(
        data_quality_metrics={}, validation_score_requirements={"initialization_success": 1.0},
        exceptions=(Exception,), default_return=False,
        validation_timeout=30.0, retry_on_failure=True)
    async def initialize(self) -> bool:
        """Initialize the orchestrator with proper error handling and resource monitoring."""
        start_time = time.time()
        
        try:
            self.logger.info("Initializing UnifiedDataOrchestrator...")
            
            # Initialize data loader
            await self.data_loader.initialize()
            
            # Initialize data sharing manager
            await self.data_sharing_manager.initialize()
            
            # Start cache cleanup task
            self._cache_cleanup_task = asyncio.create_task(
                self._cache_cleanup_loop()
            )
            
            init_time = time.time() - start_time
            self.stats["operation_times"]["initialization"] = init_time
            self.logger.info(f"UnifiedDataOrchestrator initialized successfully in {init_time:.2f}s")
            return True
            
        except Exception as e:
            init_time = time.time() - start_time
            self.logger.error(f"Failed to initialize UnifiedDataOrchestrator after {init_time:.2f}s: {e}")
            await self._handle_initialization_error(e)
            return False

    @memory_efficient(
        min_memory_gb=1.0, min_disk_gb=0.5, required_packages=["pandas", "numpy"],
        enable_garbage_collection=True, memory_monitoring=True, disk_monitoring=True,
        backup_before=False, integrity_checks=False, memory_cleanup=True,
        data_validation=False)
    @resource_monitor(
        memory_threshold_gb=1.0, cpu_threshold_percent=30.0,
        monitor_interval=5.0, auto_cleanup=True)
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True, performance_profiling=True, error_context_preservation=True)
    @circuit_breaker_protection(
        failure_threshold=5, recovery_timeout=30.0, expected_exception=Exception,
        monitor_interval=5.0)
    async def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with proper cleanup and recovery."""
        self.logger.error(f"Handling initialization error: {error}")
        
        try:
            # Cleanup any partially initialized components
            if hasattr(self, 'data_loader'):
                await self.data_loader.cleanup()
            
            if hasattr(self, 'data_sharing_manager'):
                await self.data_sharing_manager.cleanup()
            
            # Cancel cleanup task if it exists
            if self._cache_cleanup_task and not self._cache_cleanup_task.done():
                self._cache_cleanup_task.cancel()
                try:
                    await self._cache_cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear caches
            self.resampling_cache.clear()
            self.stats.clear()
            
            self.logger.info("Initialization error handled successfully")
            
        except Exception as cleanup_error:
            self.logger.error(f"Error during cleanup after initialization failure: {cleanup_error}")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return 0.0

    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up caches and manage memory."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up resampling cache if it's too large
                if len(self.resampling_cache) > self.resampling_cache_size:
                    self._cleanup_resampling_cache()
                
                # Force garbage collection
                if self.enable_memory_optimization:
                    gc.collect()
                    self.stats["memory_cleanups"] += 1
                
                # Log memory usage
                self._log_memory_usage("cache_cleanup")
                
            except asyncio.CancelledError:
                self.logger.info("Cache cleanup loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def _cleanup_resampling_cache(self) -> None:
        """Clean up resampling cache by removing oldest entries."""
        if len(self.resampling_cache) <= self.resampling_cache_size:
            return
        
        # Remove oldest entries (simple FIFO approach)
        items_to_remove = len(self.resampling_cache) - self.resampling_cache_size
        keys_to_remove = list(self.resampling_cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.resampling_cache[key]
        
        self.logger.info(f"Cleaned up {len(keys_to_remove)} items from resampling cache")

    @secure_data_processing(
        format_validation=True)
    @quality_gate(
        data_quality_metrics={"data_completeness": 0.9, "data_consistency": 0.8},
        validation_score_requirements={"data_quality": 0.8},
        exceptions=(ValueError, KeyError), default_return=None)
    async def load_data(self, source: str, **kwargs: Any) -> pd.DataFrame | None:
        """Load data from specified source with quality validation."""
        try:
            self.stats["total_requests"] += 1
            self.logger.info(f"Loading data from source: {source}")
            
            # Load data using the data loader
            data = await self.data_loader.load_data(source, **kwargs)
            
            if data is None or data.empty:
                self.logger.warning(f"No data loaded from source: {source}")
                return None
            
            # Validate data quality
            if self.enable_quality_validation:
                data = await self._validate_and_repair_data(data)
            
            # Update statistics
            data_size_gb = data.memory_usage(deep=True).sum() / 1024 / 1024 / 1024
            self.stats["total_data_loaded_gb"] += data_size_gb
            
            self.logger.info(f"Successfully loaded {len(data)} rows ({data_size_gb:.3f} GB) from {source}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {source}: {e}")
            return None

    async def _validate_and_repair_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and repair data quality issues."""
        original_length = len(data)
        
        # Check for minimum data points
        if len(data) < self.min_data_points:
            self.logger.warning(f"Data has only {len(data)} points, minimum required: {self.min_data_points}")
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > self.max_missing_ratio:
            self.logger.warning(f"High missing value ratio: {missing_ratio:.3f}")
            if self.enable_auto_repair:
                data = self._repair_missing_values(data)
        
        # Check for duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > self.max_duplicate_ratio:
            self.logger.warning(f"High duplicate ratio: {duplicate_ratio:.3f}")
            if self.enable_auto_repair:
                data = data.drop_duplicates()
        
        # Update repair statistics
        if len(data) != original_length:
            self.stats["quality_repairs"] += 1
        
        return data

    def _repair_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair missing values using appropriate strategies."""
        # For numeric columns, use forward fill then backward fill
        numeric_columns = data.select_dtypes(include=['number']).columns
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # For categorical columns, use mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().any():
                mode_value = data[col].mode()
                if not mode_value.empty:
                    data[col] = data[col].fillna(mode_value.iloc[0])
        
        return data

    @memory_efficient(
        min_memory_gb=0.5, min_disk_gb=0.1, required_packages=["pandas"],
        enable_garbage_collection=True, memory_monitoring=True)
    async def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        """Resample data to specified timeframe with caching."""
        try:
            # Create cache key
            cache_key = f"{hashlib.md5(data.to_string().encode()).hexdigest()}_{timeframe}"
            
            # Check cache first
            if cache_key in self.resampling_cache:
                self.stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for resampling to {timeframe}")
                return self.resampling_cache[cache_key].copy()
            
            self.stats["cache_misses"] += 1
            self.stats["resampling_operations"] += 1
            
            # Perform resampling
            resampled_data = await self._perform_resampling(data, timeframe)
            
            if resampled_data is not None:
                # Cache the result
                self.resampling_cache[cache_key] = resampled_data.copy()
                
                # Clean up cache if needed
                if len(self.resampling_cache) > self.resampling_cache_size:
                    self._cleanup_resampling_cache()
            
            return resampled_data
            
        except Exception as e:
            self.logger.error(f"Error resampling data to {timeframe}: {e}")
            return None

    async def _perform_resampling(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        """Perform the actual resampling operation."""
        try:
            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                elif 'time' in data.columns:
                    data = data.set_index('time')
                else:
                    self.logger.error("Data must have datetime index or timestamp/time column")
                    return None
            
            # Perform resampling
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error in resampling operation: {e}")
            return None

    async def cleanup(self) -> None:
        """Clean up resources and stop background tasks."""
        try:
            self.logger.info("Cleaning up UnifiedDataOrchestrator...")
            
            # Cancel cleanup task
            if self._cache_cleanup_task and not self._cache_cleanup_task.done():
                self._cache_cleanup_task.cancel()
                try:
                    await self._cache_cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup components
            await self.data_loader.cleanup()
            await self.data_sharing_manager.cleanup()
            
            # Clear caches
            self.resampling_cache.clear()
            
            # Stop memory tracking
            tracemalloc.stop()
            
            self.logger.info("UnifiedDataOrchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics and performance metrics."""
        current_memory = self._get_memory_usage_mb()
        
        stats = self.stats.copy()
        stats["current_memory_mb"] = current_memory
        stats["cache_size"] = len(self.resampling_cache)
        stats["cache_hit_ratio"] = (
            stats["cache_hits"] / max(stats["total_requests"], 1)
        )
        
        return stats
