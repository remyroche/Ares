# src/training/unified_data_orchestrator.py

"""
Unified Data Orchestrator - Single Source of Truth for Data Operations

This module provides a centralized, unified approach to all data operations including:
- Data loading from various sources (cache, unified format, raw files)
- Data merging and consolidation
- Multi-timeframe resampling
- Data quality validation and repair
- Memory-efficient processing
- Caching and optimization

This serves as the single source of truth for all data operations across the training pipeline.
Enhanced with comprehensive security and troubleshooting decorators.
"""

import asyncio
import gc
import hashlib
import os
import time
import psutil
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, warning, failed, missing

# Import training pipeline decorators for security and troubleshooting
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)

# Import existing components
from src.training.steps.unified_data_loader import UnifiedDataLoader
from src.training.data_sharing_manager import DataSharingManager


class UnifiedDataOrchestrator:
    """
    Unified Data Orchestrator - Single source of truth for all data operations.

    This orchestrator provides a centralized interface for:
    - Data loading with intelligent fallback strategies
    - Multi-timeframe resampling with caching
    - Data merging and consolidation
    - Quality validation and repair
    - Memory-efficient processing
    - Comprehensive logging and monitoring
    """

    def __init__(self, config: dict[str, Any]):
        print(
            f"🔧 [INIT] Starting UnifiedDataOrchestrator initialization at {datetime.now()}"
        )
        start_time = time.time()

        self.config = config
        self.logger = system_logger.getChild("UnifiedDataOrchestrator")

        # Enable memory tracking for troubleshooting
        tracemalloc.start()

        print(f"📊 [INIT] Memory usage at start: {self._get_memory_usage_mb():.2f} MB")

        # Initialize components
        print("🔄 [INIT] Initializing data loader...")
        self.data_loader = UnifiedDataLoader(config)

        print("🔄 [INIT] Initializing data sharing manager...")
        self.data_sharing_manager = DataSharingManager(config)

        # Configuration
        self.orchestrator_config = config.get("unified_data_orchestrator", {})
        self.enable_caching = self.orchestrator_config.get("enable_caching", True)
        self.enable_memory_optimization = self.orchestrator_config.get(
            "enable_memory_optimization", True
        )
        self.enable_quality_validation = self.orchestrator_config.get(
            "enable_quality_validation", True
        )
        self.enable_auto_repair = self.orchestrator_config.get(
            "enable_auto_repair", True
        )

        print(f"⚙️ [INIT] Configuration loaded:")
        print(f"   - Caching enabled: {self.enable_caching}")
        print(f"   - Memory optimization: {self.enable_memory_optimization}")
        print(f"   - Quality validation: {self.enable_quality_validation}")
        print(f"   - Auto repair: {self.enable_auto_repair}")

        # Resampling configuration
        self.resampling_config = self.orchestrator_config.get("resampling", {})
        self.default_timeframes = self.resampling_config.get(
            "default_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        )
        self.resampling_cache = {}
        self.resampling_cache_size = self.resampling_config.get("cache_size", 100)

        print(f"📈 [INIT] Resampling config:")
        print(f"   - Default timeframes: {self.default_timeframes}")
        print(f"   - Cache size: {self.resampling_cache_size}")

        # Quality validation configuration
        self.quality_config = self.orchestrator_config.get("quality_validation", {})
        self.min_data_points = self.quality_config.get("min_data_points", 1000)
        self.max_missing_ratio = self.quality_config.get("max_missing_ratio", 0.1)
        self.max_duplicate_ratio = self.quality_config.get("max_duplicate_ratio", 0.05)

        print(f"🔍 [INIT] Quality validation config:")
        print(f"   - Min data points: {self.min_data_points}")
        print(f"   - Max missing ratio: {self.max_missing_ratio:.2%}")
        print(f"   - Max duplicate ratio: {self.max_duplicate_ratio:.2%}")

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "resampling_operations": 0,
            "quality_repairs": 0,
            "memory_cleanups": 0,
            "total_data_loaded_gb": 0.0,
            "operation_times": {},
            "memory_usage_history": [],
        }

        # Initialize cache cleanup task
        self._cache_cleanup_task = None

        init_time = time.time() - start_time
        print(f"✅ [INIT] UnifiedDataOrchestrator initialized in {init_time:.2f}s")
        print(f"📊 [INIT] Final memory usage: {self._get_memory_usage_mb():.2f} MB")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _log_memory_usage(self, context: str):
        """Log current memory usage with context."""
        memory_mb = self._get_memory_usage_mb()
        self.stats["memory_usage_history"].append(
            {"timestamp": datetime.now(), "context": context, "memory_mb": memory_mb}
        )
        print(f"📊 [MEMORY] {context}: {memory_mb:.2f} MB")
        self.logger.info(f"Memory usage at {context}: {memory_mb:.2f} MB")

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/training"],
        min_memory_gb=1.0,
        min_disk_gb=0.5,
        required_packages=["pandas", "numpy", "asyncio"],
        context="Orchestrator Initialization",
    )
    @secure_data_processing(
        backup_before=False,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=False,
    )
    @resource_monitor(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=50.0,
        disk_threshold_gb=1.0,
        monitor_interval=10.0,
        auto_cleanup=True,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=3,
        recovery_timeout=60.0,
        expected_exception=Exception,
        monitor_interval=10.0,
    )
    @validate_step_output(
        data_quality_checks={},
        performance_thresholds={"initialization_time_seconds": 30.0},
        format_validation=False,
    )
    @quality_gate(
        data_quality_metrics={},
        validation_score_requirements={"initialization_success": 1.0},
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the orchestrator."""
        start_time = time.time()
        print(f"🚀 [INIT] Starting orchestrator initialization at {datetime.now()}")
        self._log_memory_usage("initialize_start")

        try:
            self.logger.info("🚀 Initializing Unified Data Orchestrator")
            print("🔄 [INIT] Step 1: Initializing data sharing manager...")

            # Initialize components
            await self.data_sharing_manager.initialize()
            print("✅ [INIT] Data sharing manager initialized")

            # Start cache cleanup task
            if self.enable_caching:
                print("🔄 [INIT] Step 2: Starting cache cleanup task...")
                self._cache_cleanup_task = asyncio.create_task(
                    self._cache_cleanup_loop()
                )
                print("✅ [INIT] Cache cleanup task started")

            init_time = time.time() - start_time
            self.stats["operation_times"]["initialization"] = init_time

            self._log_memory_usage("initialize_end")

            print(
                f"✅ [INIT] Unified Data Orchestrator initialized successfully in {init_time:.2f}s"
            )
            self.logger.info("✅ Unified Data Orchestrator initialized successfully")
            return True

        except Exception as e:
            init_time = time.time() - start_time
            print(
                f"❌ [INIT] Failed to initialize Unified Data Orchestrator after {init_time:.2f}s: {e}"
            )
            self.logger.error(f"❌ Failed to initialize Unified Data Orchestrator: {e}")
            return False

    @validate_step_prerequisites(
        required_packages=["asyncio", "gc"], context="Orchestrator Cleanup"
    )
    @secure_data_processing(
        backup_before=False,
        integrity_checks=False,
        memory_cleanup=True,
        data_validation=False,
    )
    @resource_monitor(
        memory_threshold_gb=1.0,
        cpu_threshold_percent=30.0,
        monitor_interval=5.0,
        auto_cleanup=True,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=5,
        recovery_timeout=30.0,
        expected_exception=Exception,
        monitor_interval=5.0,
    )
    @validate_step_output(
        data_quality_checks={},
        performance_thresholds={"cleanup_time_seconds": 10.0},
        format_validation=False,
    )
    @quality_gate(
        data_quality_metrics={}, validation_score_requirements={"cleanup_success": 1.0}
    )
    @handle_errors(
        exceptions=(Exception,), default_return=None, context="orchestrator cleanup"
    )
    async def cleanup(self):
        """Cleanup resources."""
        start_time = time.time()
        print(f"🧹 [CLEANUP] Starting cleanup at {datetime.now()}")
        self._log_memory_usage("cleanup_start")

        try:
            if self._cache_cleanup_task:
                print("🔄 [CLEANUP] Cancelling cache cleanup task...")
                self._cache_cleanup_task.cancel()
                try:
                    await self._cache_cleanup_task
                except asyncio.CancelledError:
                    print("✅ [CLEANUP] Cache cleanup task cancelled")
                    pass

            # Clear caches
            print("🔄 [CLEANUP] Clearing resampling cache...")
            cache_size_before = len(self.resampling_cache)
            self.resampling_cache.clear()
            print(f"✅ [CLEANUP] Cleared {cache_size_before} cache entries")

            print("🔄 [CLEANUP] Running garbage collection...")
            self._force_garbage_collection()

            cleanup_time = time.time() - start_time
            self.stats["operation_times"]["cleanup"] = cleanup_time

            self._log_memory_usage("cleanup_end")

            print(f"✅ [CLEANUP] Cleanup completed in {cleanup_time:.2f}s")
            self.logger.info("🧹 Unified Data Orchestrator cleanup completed")

        except Exception as e:
            cleanup_time = time.time() - start_time
            print(f"❌ [CLEANUP] Error during cleanup after {cleanup_time:.2f}s: {e}")
            self.logger.error(f"❌ Error during cleanup: {e}")

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/training"],
        min_memory_gb=2.0,
        min_disk_gb=1.0,
        required_packages=["pandas", "numpy", "pyarrow"],
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Unified Data Loading",
    )
    @secure_data_processing(
        backup_before=True,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=True,
    )
    @prevent_data_leakage(
        temporal_validation=True,
        feature_leakage_detection=True,
        cross_validation_isolation=True,
        lookahead_bias_prevention=True,
    )
    @resource_monitor(
        memory_threshold_gb=4.0,
        cpu_threshold_percent=70.0,
        disk_threshold_gb=2.0,
        monitor_interval=30.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=50000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=100,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=3,
        recovery_timeout=120.0,
        expected_exception=Exception,
        monitor_interval=30.0,
    )
    @validate_step_output(
        data_quality_checks={
            "no_nan_values": False,
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        performance_thresholds={"loading_time_seconds": 60.0, "memory_usage_gb": 2.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={"data_integrity": 0.7},
    )
    @handle_errors(
        exceptions=(Exception,), default_return=None, context="unified data loading"
    )
    async def get_unified_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        lookback_days: Optional[int] = None,
        force_reload: bool = False,
        validate_quality: bool = True,
        auto_repair: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get unified data with comprehensive fallback strategies and quality validation.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            force_reload: Force reload from source
            validate_quality: Validate data quality
            auto_repair: Automatically repair data issues

        Returns:
            DataFrame with unified data or None if loading fails
        """
        start_time = time.time()
        request_id = f"{exchange}_{symbol}_{timeframe}_{int(start_time)}"

        print(f"🔄 [DATA_LOAD] Starting unified data load: {request_id}")
        print(
            f"   Parameters: symbol={symbol}, exchange={exchange}, timeframe={timeframe}"
        )
        print(f"   Options: lookback_days={lookback_days}, force_reload={force_reload}")
        print(f"   Quality: validate={validate_quality}, auto_repair={auto_repair}")

        self._log_memory_usage(f"data_load_start_{request_id}")
        self.stats["total_requests"] += 1

        try:
            self.logger.info(
                f"🔄 Loading unified data: {exchange}_{symbol}_{timeframe}"
            )

            # Step 1: Try data sharing manager first (most efficient)
            if not force_reload:
                print(f"🔄 [DATA_LOAD] Step 1: Trying data sharing manager cache...")
                cache_start = time.time()

                data = await self.data_sharing_manager.get_unified_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    force_reload=False,
                )

                cache_time = time.time() - cache_start
                print(f"📊 [DATA_LOAD] Cache lookup took {cache_time:.2f}s")

                if data is not None and not data.empty:
                    self.stats["cache_hits"] += 1
                    data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                    print(
                        f"✅ [DATA_LOAD] Cache HIT: {data.shape} ({data_size_mb:.2f} MB)"
                    )
                    self.logger.info(f"✅ Data loaded from cache: {data.shape}")

                    if validate_quality:
                        print(
                            f"🔄 [DATA_LOAD] Step 1.5: Validating cached data quality..."
                        )
                        validation_start = time.time()
                        data = await self._validate_and_repair_data(data, auto_repair)
                        validation_time = time.time() - validation_start
                        print(
                            f"📊 [DATA_LOAD] Quality validation took {validation_time:.2f}s"
                        )

                    total_time = time.time() - start_time
                    self.stats["operation_times"][f"cache_hit_{request_id}"] = (
                        total_time
                    )
                    self._log_memory_usage(f"data_load_cache_hit_{request_id}")

                    print(f"✅ [DATA_LOAD] Cache hit completed in {total_time:.2f}s")
                    return data

            self.stats["cache_misses"] += 1
            print(f"❌ [DATA_LOAD] Cache MISS - proceeding to data loader")

            # Step 2: Try unified data loader
            print(f"🔄 [DATA_LOAD] Step 2: Loading from unified data loader...")
            loader_start = time.time()

            data = await self.data_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )

            loader_time = time.time() - loader_start
            print(f"📊 [DATA_LOAD] Unified loader took {loader_time:.2f}s")

            if data is not None and not data.empty:
                data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                print(
                    f"✅ [DATA_LOAD] Unified loader SUCCESS: {data.shape} ({data_size_mb:.2f} MB)"
                )
                self.logger.info(f"✅ Data loaded from unified loader: {data.shape}")

                if validate_quality:
                    print(
                        f"🔄 [DATA_LOAD] Step 2.5: Validating unified data quality..."
                    )
                    validation_start = time.time()
                    data = await self._validate_and_repair_data(data, auto_repair)
                    validation_time = time.time() - validation_start
                    print(
                        f"📊 [DATA_LOAD] Quality validation took {validation_time:.2f}s"
                    )

                # Cache the data
                if self.enable_caching:
                    print(f"🔄 [DATA_LOAD] Step 2.6: Caching unified data...")
                    cache_start = time.time()
                    await self.data_sharing_manager.cache_unified_data(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        lookback_days=lookback_days,
                        data=data,
                    )
                    cache_time = time.time() - cache_start
                    print(f"📊 [DATA_LOAD] Caching took {cache_time:.2f}s")

                total_time = time.time() - start_time
                self.stats["operation_times"][f"unified_loader_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"data_load_unified_success_{request_id}")

                print(f"✅ [DATA_LOAD] Unified loader completed in {total_time:.2f}s")
                return data

            # Step 3: Fallback to raw data loading and conversion
            print(f"🔄 [DATA_LOAD] Step 3: Fallback to raw data conversion...")
            self.logger.warning(
                "⚠️ Unified data not available, trying raw data conversion"
            )
            raw_start = time.time()

            data = await self._load_and_convert_raw_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )

            raw_time = time.time() - raw_start
            print(f"📊 [DATA_LOAD] Raw data conversion took {raw_time:.2f}s")

            if data is not None and not data.empty:
                data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                print(
                    f"✅ [DATA_LOAD] Raw conversion SUCCESS: {data.shape} ({data_size_mb:.2f} MB)"
                )
                self.logger.info(f"✅ Data loaded from raw conversion: {data.shape}")

                if validate_quality:
                    print(f"🔄 [DATA_LOAD] Step 3.5: Validating raw data quality...")
                    validation_start = time.time()
                    data = await self._validate_and_repair_data(data, auto_repair)
                    validation_time = time.time() - validation_start
                    print(
                        f"📊 [DATA_LOAD] Quality validation took {validation_time:.2f}s"
                    )

                total_time = time.time() - start_time
                self.stats["operation_times"][f"raw_conversion_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"data_load_raw_success_{request_id}")

                print(f"✅ [DATA_LOAD] Raw conversion completed in {total_time:.2f}s")
                return data

            # All methods failed
            total_time = time.time() - start_time
            print(f"❌ [DATA_LOAD] All loading methods FAILED after {total_time:.2f}s")
            self.logger.error(
                f"❌ Failed to load data for {exchange}_{symbol}_{timeframe}"
            )
            self._log_memory_usage(f"data_load_failed_{request_id}")
            return None

        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [DATA_LOAD] Exception after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ Error loading unified data: {e}")
            self._log_memory_usage(f"data_load_exception_{request_id}")
            return None

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/training"],
        min_memory_gb=4.0,
        min_disk_gb=2.0,
        required_packages=["pandas", "numpy", "pyarrow"],
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Multi-Timeframe Data Loading",
    )
    @secure_data_processing(
        backup_before=True,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=True,
    )
    @prevent_data_leakage(
        temporal_validation=True,
        feature_leakage_detection=True,
        cross_validation_isolation=True,
        lookahead_bias_prevention=True,
    )
    @resource_monitor(
        memory_threshold_gb=8.0,
        cpu_threshold_percent=80.0,
        disk_threshold_gb=5.0,
        monitor_interval=30.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=25000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=50,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=2,
        recovery_timeout=180.0,
        expected_exception=Exception,
        monitor_interval=30.0,
    )
    @validate_step_output(
        data_quality_checks={
            "no_nan_values": False,
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        performance_thresholds={"loading_time_seconds": 120.0, "memory_usage_gb": 4.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={
            "data_integrity": 0.7,
            "timeframe_alignment": 0.8,
        },
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="multi-timeframe data loading",
    )
    async def get_multi_timeframe_data(
        self,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        force_reload: bool = False,
        validate_quality: bool = True,
        auto_repair: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes with intelligent resampling.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to load
            lookback_days: Number of days to look back
            force_reload: Force reload from source
            validate_quality: Validate data quality
            auto_repair: Automatically repair data issues

        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        start_time = time.time()
        request_id = f"multi_{exchange}_{symbol}_{int(start_time)}"

        print(f"🔄 [MULTI_TF] Starting multi-timeframe data load: {request_id}")
        print(f"   Parameters: symbol={symbol}, exchange={exchange}")
        print(f"   Timeframes: {timeframes or self.default_timeframes}")
        print(f"   Options: lookback_days={lookback_days}, force_reload={force_reload}")
        print(f"   Quality: validate={validate_quality}, auto_repair={auto_repair}")

        self._log_memory_usage(f"multi_tf_start_{request_id}")

        if timeframes is None:
            timeframes = self.default_timeframes

        try:
            self.logger.info(f"🔄 Loading multi-timeframe data: {exchange}_{symbol}")
            self.logger.info(f"   Timeframes: {timeframes}")

            # Sort timeframes by resolution (highest to lowest)
            print(f"🔄 [MULTI_TF] Step 1: Sorting timeframes by resolution...")
            timeframe_order = self._sort_timeframes_by_resolution(timeframes)
            print(f"   Sorted timeframes: {timeframe_order}")

            # Load base timeframe first (usually 1m)
            base_timeframe = self._get_base_timeframe(timeframe_order)
            print(f"🔄 [MULTI_TF] Step 2: Loading base timeframe: {base_timeframe}")
            base_start = time.time()

            base_data = await self.get_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=base_timeframe,
                lookback_days=lookback_days,
                force_reload=force_reload,
                validate_quality=validate_quality,
                auto_repair=auto_repair,
            )

            base_time = time.time() - base_start
            print(f"📊 [MULTI_TF] Base timeframe loading took {base_time:.2f}s")

            if base_data is None or base_data.empty:
                print(f"❌ [MULTI_TF] Failed to load base data for {base_timeframe}")
                self.logger.error(f"❌ Failed to load base data for {base_timeframe}")
                return {}

            base_size_mb = base_data.memory_usage(deep=True).sum() / 1024 / 1024
            print(
                f"✅ [MULTI_TF] Base data loaded: {base_data.shape} ({base_size_mb:.2f} MB)"
            )

            # Load or resample data for each timeframe
            print(
                f"🔄 [MULTI_TF] Step 3: Processing {len(timeframe_order)} timeframes..."
            )
            result = {}
            successful_timeframes = 0

            for i, timeframe in enumerate(timeframe_order, 1):
                print(
                    f"🔄 [MULTI_TF] Processing timeframe {i}/{len(timeframe_order)}: {timeframe}"
                )
                tf_start = time.time()

                if timeframe == base_timeframe:
                    result[timeframe] = base_data
                    print(f"✅ [MULTI_TF] Using base data for {timeframe}")
                    successful_timeframes += 1
                    continue

                # Try to load existing data for this timeframe
                print(f"🔄 [MULTI_TF] Trying to load existing data for {timeframe}...")
                existing_start = time.time()

                existing_data = await self.get_unified_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    force_reload=force_reload,
                    validate_quality=validate_quality,
                    auto_repair=auto_repair,
                )

                existing_time = time.time() - existing_start
                print(
                    f"📊 [MULTI_TF] Existing data lookup for {timeframe} took {existing_time:.2f}s"
                )

                if existing_data is not None and not existing_data.empty:
                    result[timeframe] = existing_data
                    existing_size_mb = (
                        existing_data.memory_usage(deep=True).sum() / 1024 / 1024
                    )
                    print(
                        f"✅ [MULTI_TF] Loaded existing data for {timeframe}: {existing_data.shape} ({existing_size_mb:.2f} MB)"
                    )
                    self.logger.info(
                        f"✅ Loaded existing data for {timeframe}: {existing_data.shape}"
                    )
                    successful_timeframes += 1
                else:
                    # Resample from base data
                    print(f"🔄 [MULTI_TF] Resampling base data to {timeframe}...")
                    resample_start = time.time()

                    resampled_data = await self._resample_data(
                        data=base_data,
                        from_timeframe=base_timeframe,
                        to_timeframe=timeframe,
                        symbol=symbol,
                        exchange=exchange,
                    )

                    resample_time = time.time() - resample_start
                    print(
                        f"📊 [MULTI_TF] Resampling to {timeframe} took {resample_time:.2f}s"
                    )

                    if resampled_data is not None and not resampled_data.empty:
                        result[timeframe] = resampled_data
                        resampled_size_mb = (
                            resampled_data.memory_usage(deep=True).sum() / 1024 / 1024
                        )
                        print(
                            f"✅ [MULTI_TF] Resampled data for {timeframe}: {resampled_data.shape} ({resampled_size_mb:.2f} MB)"
                        )
                        self.logger.info(
                            f"✅ Resampled data for {timeframe}: {resampled_data.shape}"
                        )
                        successful_timeframes += 1

                        # Cache the resampled data
                        if self.enable_caching:
                            print(
                                f"🔄 [MULTI_TF] Caching resampled data for {timeframe}..."
                            )
                            cache_start = time.time()
                            await self.data_sharing_manager.cache_unified_data(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                lookback_days=lookback_days,
                                data=resampled_data,
                            )
                            cache_time = time.time() - cache_start
                            print(
                                f"📊 [MULTI_TF] Caching {timeframe} took {cache_time:.2f}s"
                            )
                    else:
                        print(f"⚠️ [MULTI_TF] Failed to resample data for {timeframe}")
                        self.logger.warning(
                            f"⚠️ Failed to resample data for {timeframe}"
                        )

                tf_time = time.time() - tf_start
                print(
                    f"📊 [MULTI_TF] Timeframe {timeframe} processing took {tf_time:.2f}s"
                )

            total_time = time.time() - start_time
            total_size_mb = (
                sum(df.memory_usage(deep=True).sum() for df in result.values())
                / 1024
                / 1024
            )

            print(f"✅ [MULTI_TF] Multi-timeframe loading completed:")
            print(
                f"   - Success: {successful_timeframes}/{len(timeframe_order)} timeframes"
            )
            print(f"   - Total time: {total_time:.2f}s")
            print(f"   - Total memory: {total_size_mb:.2f} MB")
            print(f"   - Timeframes: {list(result.keys())}")

            self.stats["operation_times"][f"multi_tf_{request_id}"] = total_time
            self._log_memory_usage(f"multi_tf_end_{request_id}")

            self.logger.info(
                f"✅ Multi-timeframe data loading completed: {len(result)} timeframes"
            )
            return result

        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [MULTI_TF] Exception after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ Error loading multi-timeframe data: {e}")
            self._log_memory_usage(f"multi_tf_exception_{request_id}")
            return {}

    @validate_step_prerequisites(
        required_packages=["pandas", "numpy"],
        data_quality_checks={
            "min_rows": 50,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Data Resampling",
    )
    @secure_data_processing(
        backup_before=True,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=True,
    )
    @prevent_data_leakage(temporal_validation=True, lookahead_bias_prevention=True)
    @resource_monitor(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=60.0,
        monitor_interval=15.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=10000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=25,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=Exception,
        monitor_interval=15.0,
    )
    @validate_step_output(
        data_quality_checks={
            "no_nan_values": False,
            "min_rows": 10,
            "required_columns": ["open", "high", "low", "close", "volume"],
        },
        performance_thresholds={"resampling_time_seconds": 30.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.8, "consistency": 0.7},
        validation_score_requirements={"resampling_accuracy": 0.9},
    )
    @handle_errors(
        exceptions=(Exception,), default_return=None, context="data resampling"
    )
    async def _resample_data(
        self,
        data: pd.DataFrame,
        from_timeframe: str,
        to_timeframe: str,
        symbol: str,
        exchange: str,
    ) -> Optional[pd.DataFrame]:
        """
        Resample data from one timeframe to another with caching.

        Args:
            data: Input DataFrame
            from_timeframe: Source timeframe
            to_timeframe: Target timeframe
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Resampled DataFrame or None if resampling fails
        """
        start_time = time.time()
        request_id = f"resample_{from_timeframe}_{to_timeframe}_{int(start_time)}"

        print(f"🔄 [RESAMPLE] Starting resampling: {request_id}")
        print(f"   From: {from_timeframe} ({data.shape})")
        print(f"   To: {to_timeframe}")
        print(f"   Symbol: {symbol}, Exchange: {exchange}")

        self._log_memory_usage(f"resample_start_{request_id}")

        try:
            self.stats["resampling_operations"] += 1

            # Generate cache key
            print(f"🔄 [RESAMPLE] Step 1: Generating cache key...")
            cache_start = time.time()
            cache_key = self._generate_resampling_cache_key(
                data, from_timeframe, to_timeframe, symbol, exchange
            )
            cache_key_time = time.time() - cache_start
            print(f"📊 [RESAMPLE] Cache key generation took {cache_key_time:.2f}s")

            # Check cache
            if cache_key in self.resampling_cache:
                print(f"📋 [RESAMPLE] Cache HIT for {from_timeframe} -> {to_timeframe}")
                cached_data = self.resampling_cache[cache_key].copy()
                cached_size_mb = cached_data.memory_usage(deep=True).sum() / 1024 / 1024
                print(
                    f"✅ [RESAMPLE] Using cached data: {cached_data.shape} ({cached_size_mb:.2f} MB)"
                )
                self.logger.info(
                    f"📋 Using cached resampled data for {from_timeframe} -> {to_timeframe}"
                )

                total_time = time.time() - start_time
                self.stats["operation_times"][f"resample_cache_hit_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"resample_cache_hit_{request_id}")

                print(f"✅ [RESAMPLE] Cache hit completed in {total_time:.2f}s")
                return cached_data

            print(f"❌ [RESAMPLE] Cache MISS - performing resampling...")

            # Perform resampling
            print(f"🔄 [RESAMPLE] Step 2: Performing resampling operation...")
            resample_start = time.time()

            resampled_data = self._perform_resampling(
                data, from_timeframe, to_timeframe
            )

            resample_time = time.time() - resample_start
            print(f"📊 [RESAMPLE] Resampling operation took {resample_time:.2f}s")

            if resampled_data is not None and not resampled_data.empty:
                resampled_size_mb = (
                    resampled_data.memory_usage(deep=True).sum() / 1024 / 1024
                )
                print(
                    f"✅ [RESAMPLE] Resampling successful: {resampled_data.shape} ({resampled_size_mb:.2f} MB)"
                )

                # Cache the result
                print(f"🔄 [RESAMPLE] Step 3: Caching result...")
                cache_start = time.time()

                if len(self.resampling_cache) < self.resampling_cache_size:
                    self.resampling_cache[cache_key] = resampled_data.copy()
                    print(
                        f"✅ [RESAMPLE] Added to cache (size: {len(self.resampling_cache)}/{self.resampling_cache_size})"
                    )
                else:
                    # Remove oldest entry
                    oldest_key = next(iter(self.resampling_cache))
                    del self.resampling_cache[oldest_key]
                    self.resampling_cache[cache_key] = resampled_data.copy()
                    print(
                        f"✅ [RESAMPLE] Replaced oldest cache entry (size: {len(self.resampling_cache)}/{self.resampling_cache_size})"
                    )

                cache_time = time.time() - cache_start
                print(f"📊 [RESAMPLE] Caching took {cache_time:.2f}s")

                total_time = time.time() - start_time
                self.stats["operation_times"][f"resample_success_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"resample_success_{request_id}")

                print(f"✅ [RESAMPLE] Resampling completed in {total_time:.2f}s")
                self.logger.info(
                    f"✅ Resampled {from_timeframe} -> {to_timeframe}: {resampled_data.shape}"
                )
                return resampled_data

            print(f"❌ [RESAMPLE] Resampling failed - no data returned")
            total_time = time.time() - start_time
            self.stats["operation_times"][f"resample_failed_{request_id}"] = total_time
            self._log_memory_usage(f"resample_failed_{request_id}")

            return None

        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [RESAMPLE] Exception after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ Error resampling data: {e}")
            self._log_memory_usage(f"resample_exception_{request_id}")
            return None

    def _perform_resampling(
        self, data: pd.DataFrame, from_timeframe: str, to_timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Perform the actual resampling operation."""
        try:
            print(
                f"🔄 [RESAMPLE_OP] Starting resampling operation: {from_timeframe} -> {to_timeframe}"
            )

            # Ensure we have a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                print(f"🔄 [RESAMPLE_OP] Converting index to DatetimeIndex...")
                if "timestamp" in data.columns:
                    data = data.copy()
                    data.index = pd.to_datetime(data["timestamp"], errors="coerce")
                    data = data.sort_index()
                    print(f"✅ [RESAMPLE_OP] Index converted and sorted")
                else:
                    print(f"❌ [RESAMPLE_OP] No timestamp column found for resampling")
                    self.logger.error("❌ No timestamp column found for resampling")
                    return None

            # Convert timeframes to pandas offset
            timeframe_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D",
            }

            from_offset = timeframe_map.get(from_timeframe, "1min")
            to_offset = timeframe_map.get(to_timeframe, "1min")
            print(f"📊 [RESAMPLE_OP] Offsets: {from_offset} -> {to_offset}")

            # Determine resampling direction
            from_minutes = self._timeframe_to_minutes(from_timeframe)
            to_minutes = self._timeframe_to_minutes(to_timeframe)
            print(f"📊 [RESAMPLE_OP] Minutes: {from_minutes} -> {to_minutes}")

            if from_minutes < to_minutes:
                # Upsampling (e.g., 1m -> 5m)
                print(f"🔄 [RESAMPLE_OP] Performing upsampling...")
                return self._upsample_data(data, to_offset)
            else:
                # Downsampling (e.g., 5m -> 1m) - not supported
                print(
                    f"⚠️ [RESAMPLE_OP] Downsampling not supported: {from_timeframe} -> {to_timeframe}"
                )
                self.logger.warning(
                    f"⚠️ Downsampling not supported: {from_timeframe} -> {to_timeframe}"
                )
                return None

        except Exception as e:
            print(f"❌ [RESAMPLE_OP] Error in resampling operation: {e}")
            self.logger.error(f"❌ Error in resampling operation: {e}")
            return None

    def _upsample_data(
        self, data: pd.DataFrame, target_offset: str
    ) -> Optional[pd.DataFrame]:
        """Upsample data to higher timeframe."""
        try:
            print(f"🔄 [UPSAMPLE] Upsampling to {target_offset}...")
            print(f"   Input shape: {data.shape}")
            print(f"   Input columns: {list(data.columns)}")

            # Resample OHLCV data
            if all(
                col in data.columns
                for col in ["open", "high", "low", "close", "volume"]
            ):
                print(f"🔄 [UPSAMPLE] Resampling OHLCV data...")
                resampled = (
                    data.resample(target_offset)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
                print(f"✅ [UPSAMPLE] OHLCV resampling completed")
            else:
                # Fallback for other data types
                print(f"🔄 [UPSAMPLE] Resampling with fallback method...")
                resampled = data.resample(target_offset).last().dropna()
                print(f"✅ [UPSAMPLE] Fallback resampling completed")

            print(f"✅ [UPSAMPLE] Upsampling completed: {resampled.shape}")
            return resampled

        except Exception as e:
            print(f"❌ [UPSAMPLE] Error upsampling data: {e}")
            self.logger.error(f"❌ Error upsampling data: {e}")
            return None

    @validate_step_prerequisites(
        required_packages=["pandas", "numpy"],
        data_quality_checks={"min_rows": 10},
        context="Data Validation and Repair",
    )
    @secure_data_processing(
        backup_before=True,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=True,
    )
    @prevent_data_leakage(temporal_validation=True, feature_leakage_detection=True)
    @resource_monitor(
        memory_threshold_gb=1.0,
        cpu_threshold_percent=50.0,
        monitor_interval=10.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=5000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=20,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=10,
        recovery_timeout=30.0,
        expected_exception=Exception,
        monitor_interval=10.0,
    )
    @validate_step_output(
        data_quality_checks={"no_nan_values": False, "min_rows": 5},
        performance_thresholds={"validation_time_seconds": 15.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.7, "consistency": 0.6},
        validation_score_requirements={"data_quality": 0.8},
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data validation and repair",
    )
    async def _validate_and_repair_data(
        self,
        data: pd.DataFrame,
        auto_repair: bool = True,
    ) -> pd.DataFrame:
        """
        Validate data quality and repair issues if possible.

        Args:
            data: Input DataFrame
            auto_repair: Automatically repair issues

        Returns:
            Validated and repaired DataFrame
        """
        start_time = time.time()
        request_id = f"validate_{data.shape[0]}_{int(start_time)}"

        print(f"🔍 [VALIDATE] Starting data validation: {request_id}")
        print(f"   Input shape: {data.shape}")
        print(f"   Auto repair: {auto_repair}")

        self._log_memory_usage(f"validate_start_{request_id}")

        try:
            if not self.enable_quality_validation:
                print(
                    f"⚠️ [VALIDATE] Quality validation disabled, returning original data"
                )
                return data

            self.logger.info(f"🔍 Validating data quality: {data.shape}")

            # Check data size
            print(f"🔍 [VALIDATE] Step 1: Checking data size...")
            if len(data) < self.min_data_points:
                print(
                    f"⚠️ [VALIDATE] Insufficient data points: {len(data)} < {self.min_data_points}"
                )
                self.logger.warning(
                    f"⚠️ Insufficient data points: {len(data)} < {self.min_data_points}"
                )
                if not auto_repair:
                    return data

            # Check for missing values
            print(f"🔍 [VALIDATE] Step 2: Checking for missing values...")
            missing_start = time.time()
            missing_counts = data.isnull().sum()
            missing_ratio = missing_counts.sum() / (len(data) * len(data.columns))
            missing_time = time.time() - missing_start
            print(f"📊 [VALIDATE] Missing value analysis took {missing_time:.2f}s")
            print(f"   Missing ratio: {missing_ratio:.2%}")

            if missing_ratio > self.max_missing_ratio:
                print(
                    f"⚠️ [VALIDATE] High missing value ratio: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}"
                )
                self.logger.warning(
                    f"⚠️ High missing value ratio: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}"
                )
                if auto_repair:
                    print(f"🔧 [VALIDATE] Repairing missing values...")
                    repair_start = time.time()
                    data = self._repair_missing_values(data)
                    repair_time = time.time() - repair_start
                    print(f"📊 [VALIDATE] Missing value repair took {repair_time:.2f}s")
                    self.stats["quality_repairs"] += 1

            # Check for duplicates
            print(f"🔍 [VALIDATE] Step 3: Checking for duplicates...")
            duplicate_start = time.time()
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data)
            duplicate_time = time.time() - duplicate_start
            print(f"📊 [VALIDATE] Duplicate analysis took {duplicate_time:.2f}s")
            print(f"   Duplicate ratio: {duplicate_ratio:.2%}")

            if duplicate_ratio > self.max_duplicate_ratio:
                print(
                    f"⚠️ [VALIDATE] High duplicate ratio: {duplicate_ratio:.2%} > {self.max_duplicate_ratio:.2%}"
                )
                self.logger.warning(
                    f"⚠️ High duplicate ratio: {duplicate_ratio:.2%} > {self.max_duplicate_ratio:.2%}"
                )
                if auto_repair:
                    print(f"🔧 [VALIDATE] Removing duplicates...")
                    repair_start = time.time()
                    data = data.drop_duplicates()
                    repair_time = time.time() - repair_start
                    print(f"📊 [VALIDATE] Duplicate removal took {repair_time:.2f}s")
                    self.stats["quality_repairs"] += 1

            # Check for timestamp issues
            print(f"🔍 [VALIDATE] Step 4: Checking timestamp issues...")
            if "timestamp" in data.columns:
                timestamp_start = time.time()
                data = self._repair_timestamp_issues(data)
                timestamp_time = time.time() - timestamp_start
                print(f"📊 [VALIDATE] Timestamp repair took {timestamp_time:.2f}s")

            # Check for price anomalies
            print(f"🔍 [VALIDATE] Step 5: Checking price anomalies...")
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                price_start = time.time()
                data = self._repair_price_anomalies(data)
                price_time = time.time() - price_start
                print(f"📊 [VALIDATE] Price anomaly repair took {price_time:.2f}s")

            total_time = time.time() - start_time
            print(f"✅ [VALIDATE] Data validation completed in {total_time:.2f}s")
            print(f"   Final shape: {data.shape}")

            self.stats["operation_times"][f"validation_{request_id}"] = total_time
            self._log_memory_usage(f"validate_end_{request_id}")

            self.logger.info(f"✅ Data validation completed: {data.shape}")
            return data

        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [VALIDATE] Exception after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ Error validating data: {e}")
            self._log_memory_usage(f"validate_exception_{request_id}")
            return data

    def _repair_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair missing values in the data."""
        try:
            print(f"🔧 [REPAIR_MISSING] Starting missing value repair...")
            print(f"   Input shape: {data.shape}")

            # Forward fill for OHLCV data
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            if all(col in data.columns for col in ohlcv_columns):
                print(f"🔧 [REPAIR_MISSING] Forward filling OHLCV columns...")
                data[ohlcv_columns] = data[ohlcv_columns].fillna(method="ffill")

            # Drop remaining rows with missing values
            print(f"🔧 [REPAIR_MISSING] Dropping remaining rows with missing values...")
            data = data.dropna()

            print(f"✅ [REPAIR_MISSING] Missing value repair completed: {data.shape}")
            self.logger.info(f"🔧 Repaired missing values: {data.shape}")
            return data

        except Exception as e:
            print(f"❌ [REPAIR_MISSING] Error repairing missing values: {e}")
            self.logger.error(f"❌ Error repairing missing values: {e}")
            return data

    def _repair_timestamp_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair timestamp-related issues."""
        try:
            print(f"🔧 [REPAIR_TIMESTAMP] Starting timestamp repair...")
            print(f"   Input shape: {data.shape}")

            # Ensure timestamp column is datetime
            if "timestamp" in data.columns:
                print(f"🔧 [REPAIR_TIMESTAMP] Converting timestamp to datetime...")
                data = data.copy()
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

                # Remove rows with invalid timestamps
                invalid_timestamps = data["timestamp"].isna()
                if invalid_timestamps.sum() > 0:
                    print(
                        f"⚠️ [REPAIR_TIMESTAMP] Removing {invalid_timestamps.sum()} rows with invalid timestamps"
                    )
                    self.logger.warning(
                        f"⚠️ Removing {invalid_timestamps.sum()} rows with invalid timestamps"
                    )
                    data = data[~invalid_timestamps]

                # Sort by timestamp
                print(f"🔧 [REPAIR_TIMESTAMP] Sorting by timestamp...")
                data = data.sort_values("timestamp")

                # Set timestamp as index
                print(f"🔧 [REPAIR_TIMESTAMP] Setting timestamp as index...")
                data = data.set_index("timestamp")

            print(f"✅ [REPAIR_TIMESTAMP] Timestamp repair completed: {data.shape}")
            return data

        except Exception as e:
            print(f"❌ [REPAIR_TIMESTAMP] Error repairing timestamp issues: {e}")
            self.logger.error(f"❌ Error repairing timestamp issues: {e}")
            return data

    def _repair_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair price anomalies in OHLCV data."""
        try:
            print(f"🔧 [REPAIR_PRICE] Starting price anomaly repair...")
            print(f"   Input shape: {data.shape}")

            # Check for negative prices
            price_columns = ["open", "high", "low", "close"]
            negative_prices = (data[price_columns] <= 0).any(axis=1)

            if negative_prices.sum() > 0:
                print(
                    f"⚠️ [REPAIR_PRICE] Removing {negative_prices.sum()} rows with negative prices"
                )
                self.logger.warning(
                    f"⚠️ Removing {negative_prices.sum()} rows with negative prices"
                )
                data = data[~negative_prices]

            # Check for high-low inconsistencies
            if all(col in data.columns for col in ["high", "low"]):
                invalid_hl = data["high"] < data["low"]
                if invalid_hl.sum() > 0:
                    print(
                        f"⚠️ [REPAIR_PRICE] Removing {invalid_hl.sum()} rows with high < low"
                    )
                    self.logger.warning(
                        f"⚠️ Removing {invalid_hl.sum()} rows with high < low"
                    )
                    data = data[~invalid_hl]

            print(f"✅ [REPAIR_PRICE] Price anomaly repair completed: {data.shape}")
            return data

        except Exception as e:
            print(f"❌ [REPAIR_PRICE] Error repairing price anomalies: {e}")
            self.logger.error(f"❌ Error repairing price anomalies: {e}")
            return data

    @validate_step_prerequisites(
        required_directories=["data_cache"],
        min_memory_gb=1.0,
        min_disk_gb=0.5,
        required_packages=["pandas", "numpy"],
        data_quality_checks={"min_rows": 10},
        context="Raw Data Loading and Conversion",
    )
    @secure_data_processing(
        backup_before=True,
        integrity_checks=True,
        memory_cleanup=True,
        data_validation=True,
    )
    @prevent_data_leakage(temporal_validation=True, lookahead_bias_prevention=True)
    @resource_monitor(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=60.0,
        disk_threshold_gb=1.0,
        monitor_interval=20.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=10000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=30,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=5,
        recovery_timeout=90.0,
        expected_exception=Exception,
        monitor_interval=20.0,
    )
    @validate_step_output(
        data_quality_checks={
            "no_nan_values": False,
            "min_rows": 10,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        performance_thresholds={"conversion_time_seconds": 45.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.8, "consistency": 0.7},
        validation_score_requirements={"conversion_accuracy": 0.9},
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="raw data loading and conversion",
    )
    async def _load_and_convert_raw_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        lookback_days: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load and convert raw data to unified format."""
        start_time = time.time()
        request_id = f"raw_{exchange}_{symbol}_{timeframe}_{int(start_time)}"

        print(f"🔄 [RAW_DATA] Starting raw data loading: {request_id}")
        print(
            f"   Parameters: symbol={symbol}, exchange={exchange}, timeframe={timeframe}"
        )
        print(f"   Lookback days: {lookback_days}")

        self._log_memory_usage(f"raw_data_start_{request_id}")

        try:
            self.logger.info(
                f"🔄 Loading raw data for conversion: {exchange}_{symbol}_{timeframe}"
            )

            # Look for raw data files
            print(f"🔄 [RAW_DATA] Step 1: Finding raw data files...")
            find_start = time.time()
            raw_data_paths = self._find_raw_data_files(symbol, exchange, timeframe)
            find_time = time.time() - find_start
            print(f"📊 [RAW_DATA] File search took {find_time:.2f}s")
            print(f"   Found {len(raw_data_paths)} files")

            if not raw_data_paths:
                print(
                    f"⚠️ [RAW_DATA] No raw data files found for {exchange}_{symbol}_{timeframe}"
                )
                self.logger.warning(
                    f"⚠️ No raw data files found for {exchange}_{symbol}_{timeframe}"
                )
                return None

            # Load and combine raw data
            print(f"🔄 [RAW_DATA] Step 2: Loading and combining raw data...")
            load_start = time.time()
            combined_data = []
            successful_files = 0

            for i, file_path in enumerate(raw_data_paths, 1):
                print(
                    f"🔄 [RAW_DATA] Loading file {i}/{len(raw_data_paths)}: {file_path}"
                )
                try:
                    file_start = time.time()
                    raw_data = pd.read_parquet(file_path)
                    file_time = time.time() - file_start
                    print(
                        f"📊 [RAW_DATA] File load took {file_time:.2f}s: {raw_data.shape}"
                    )

                    if not raw_data.empty:
                        combined_data.append(raw_data)
                        successful_files += 1
                except Exception as e:
                    print(f"⚠️ [RAW_DATA] Failed to load {file_path}: {e}")
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

            load_time = time.time() - load_start
            print(f"📊 [RAW_DATA] Total loading took {load_time:.2f}s")
            print(f"   Successful files: {successful_files}/{len(raw_data_paths)}")

            if not combined_data:
                print(
                    f"❌ [RAW_DATA] No valid raw data found for {exchange}_{symbol}_{timeframe}"
                )
                self.logger.error(
                    f"❌ No valid raw data found for {exchange}_{symbol}_{timeframe}"
                )
                return None

            # Combine all data
            print(f"🔄 [RAW_DATA] Step 3: Combining data...")
            combine_start = time.time()
            data = pd.concat(combined_data, ignore_index=True)
            data = data.drop_duplicates()
            combine_time = time.time() - combine_start
            print(f"📊 [RAW_DATA] Data combination took {combine_time:.2f}s")
            print(f"   Combined shape: {data.shape}")

            # Convert to unified format
            print(f"🔄 [RAW_DATA] Step 4: Converting to unified format...")
            convert_start = time.time()
            unified_data = self._convert_to_unified_format(
                data, symbol, exchange, timeframe
            )
            convert_time = time.time() - convert_start
            print(f"📊 [RAW_DATA] Format conversion took {convert_time:.2f}s")

            if unified_data is not None and not unified_data.empty:
                unified_size_mb = (
                    unified_data.memory_usage(deep=True).sum() / 1024 / 1024
                )
                print(
                    f"✅ [RAW_DATA] Conversion successful: {unified_data.shape} ({unified_size_mb:.2f} MB)"
                )
                self.logger.info(
                    f"✅ Converted raw data to unified format: {unified_data.shape}"
                )

                total_time = time.time() - start_time
                self.stats["operation_times"][f"raw_data_{request_id}"] = total_time
                self._log_memory_usage(f"raw_data_success_{request_id}")

                print(f"✅ [RAW_DATA] Raw data loading completed in {total_time:.2f}s")
                return unified_data

            print(f"❌ [RAW_DATA] Conversion failed - no data returned")
            return None

        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [RAW_DATA] Exception after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ Error loading and converting raw data: {e}")
            self._log_memory_usage(f"raw_data_exception_{request_id}")
            return None

    def _find_raw_data_files(
        self, symbol: str, exchange: str, timeframe: str
    ) -> List[str]:
        """Find raw data files for the given parameters."""
        try:
            print(f"🔍 [FIND_FILES] Searching for raw data files...")
            print(f"   Pattern: {exchange}_{symbol}_*_{timeframe}*.parquet")

            # Look in data_cache directory
            cache_dir = Path("data_cache")
            if not cache_dir.exists():
                print(f"⚠️ [FIND_FILES] data_cache directory does not exist")
                return []

            # Search for files matching the pattern
            pattern = f"{exchange}_{symbol}_*_{timeframe}*.parquet"
            files = list(cache_dir.rglob(pattern))

            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            file_paths = [str(f) for f in files]
            print(f"✅ [FIND_FILES] Found {len(file_paths)} files")

            return file_paths

        except Exception as e:
            print(f"❌ [FIND_FILES] Error finding raw data files: {e}")
            self.logger.error(f"❌ Error finding raw data files: {e}")
            return []

    def _convert_to_unified_format(
        self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Convert raw data to unified format."""
        try:
            print(f"🔄 [CONVERT] Converting to unified format...")
            print(f"   Input shape: {data.shape}")
            print(f"   Input columns: {list(data.columns)}")

            # Ensure we have required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Map common column names
            column_mapping = {
                "time": "timestamp",
                "date": "timestamp",
                "datetime": "timestamp",
                "price": "close",
                "amount": "volume",
                "quantity": "volume",
            }

            # Rename columns if needed
            print(f"🔄 [CONVERT] Mapping column names...")
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    print(f"   Mapping {old_col} -> {new_col}")
                    data = data.rename(columns={old_col: new_col})

            # Check if we have the required columns
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                print(f"❌ [CONVERT] Missing required columns: {missing_columns}")
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return None

            # Ensure timestamp is datetime
            print(f"🔄 [CONVERT] Converting timestamp to datetime...")
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            data = data.dropna(subset=["timestamp"])

            # Sort by timestamp
            print(f"🔄 [CONVERT] Sorting by timestamp...")
            data = data.sort_values("timestamp")

            # Add metadata columns
            print(f"🔄 [CONVERT] Adding metadata columns...")
            data["symbol"] = symbol
            data["exchange"] = exchange
            data["timeframe"] = timeframe

            # Set timestamp as index
            print(f"🔄 [CONVERT] Setting timestamp as index...")
            data = data.set_index("timestamp")

            print(f"✅ [CONVERT] Conversion completed: {data.shape}")
            return data

        except Exception as e:
            print(f"❌ [CONVERT] Error converting to unified format: {e}")
            self.logger.error(f"❌ Error converting to unified format: {e}")
            return None

    def _generate_resampling_cache_key(
        self,
        data: pd.DataFrame,
        from_timeframe: str,
        to_timeframe: str,
        symbol: str,
        exchange: str,
    ) -> str:
        """Generate cache key for resampled data."""
        try:
            print(f"🔑 [CACHE_KEY] Generating cache key...")

            # Create a hashable representation of the data
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values
            ).hexdigest()

            cache_key = (
                f"{exchange}_{symbol}_{from_timeframe}_{to_timeframe}_{data_hash}"
            )
            print(f"✅ [CACHE_KEY] Generated key: {cache_key[:50]}...")

            return cache_key

        except Exception:
            # Fallback to simple hash
            print(f"⚠️ [CACHE_KEY] Using fallback cache key generation...")
            fallback_key = f"{exchange}_{symbol}_{from_timeframe}_{to_timeframe}_{hash(str(data.shape))}"
            return fallback_key

    def _sort_timeframes_by_resolution(self, timeframes: List[str]) -> List[str]:
        """Sort timeframes by resolution (highest to lowest)."""
        print(f"📊 [SORT_TF] Sorting timeframes by resolution...")
        print(f"   Input: {timeframes}")

        timeframe_minutes = {tf: self._timeframe_to_minutes(tf) for tf in timeframes}
        sorted_timeframes = sorted(timeframes, key=lambda tf: timeframe_minutes[tf])

        print(f"✅ [SORT_TF] Sorted timeframes: {sorted_timeframes}")
        return sorted_timeframes

    def _get_base_timeframe(self, timeframes: List[str]) -> str:
        """Get the base timeframe (highest resolution)."""
        print(f"📊 [BASE_TF] Getting base timeframe...")
        print(f"   Timeframes: {timeframes}")

        sorted_timeframes = self._sort_timeframes_by_resolution(timeframes)
        base_timeframe = sorted_timeframes[0] if sorted_timeframes else "1m"

        print(f"✅ [BASE_TF] Base timeframe: {base_timeframe}")
        return base_timeframe

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return timeframe_map.get(timeframe, 1)

    async def _cache_cleanup_loop(self):
        """Periodic cache cleanup loop."""
        print(f"🔄 [CACHE_CLEANUP] Starting cache cleanup loop...")

        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                print(f"🔄 [CACHE_CLEANUP] Running periodic cleanup...")
                cleanup_start = time.time()

                # Clean up resampling cache
                if len(self.resampling_cache) > self.resampling_cache_size * 0.8:
                    print(f"🔄 [CACHE_CLEANUP] Cleaning up resampling cache...")
                    # Remove oldest entries
                    keys_to_remove = list(self.resampling_cache.keys())[
                        : len(self.resampling_cache) // 2
                    ]
                    for key in keys_to_remove:
                        del self.resampling_cache[key]

                    print(
                        f"✅ [CACHE_CLEANUP] Cleaned up resampling cache: {len(self.resampling_cache)} entries remaining"
                    )
                    self.logger.info(
                        f"🧹 Cleaned up resampling cache: {len(self.resampling_cache)} entries remaining"
                    )

                # Force garbage collection
                if self.enable_memory_optimization:
                    print(f"🔄 [CACHE_CLEANUP] Running garbage collection...")
                    self._force_garbage_collection()

                cleanup_time = time.time() - cleanup_start
                print(f"📊 [CACHE_CLEANUP] Cleanup completed in {cleanup_time:.2f}s")

            except asyncio.CancelledError:
                print(f"✅ [CACHE_CLEANUP] Cache cleanup loop cancelled")
                break
            except Exception as e:
                print(f"❌ [CACHE_CLEANUP] Error in cache cleanup loop: {e}")
                self.logger.error(f"❌ Error in cache cleanup loop: {e}")

    def _force_garbage_collection(self):
        """Force garbage collection."""
        try:
            print(f"🧹 [GC] Running garbage collection...")
            gc_start = time.time()
            collected = gc.collect()
            gc_time = time.time() - gc_start

            self.stats["memory_cleanups"] += 1
            print(f"📊 [GC] Garbage collection completed in {gc_time:.2f}s")
            print(f"   Collected objects: {collected}")
            self.logger.debug(f"🧹 Garbage collection: collected {collected} objects")

        except Exception as e:
            print(f"❌ [GC] Error during garbage collection: {e}")
            self.logger.error(f"❌ Error during garbage collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        print(f"📊 [STATS] Getting orchestrator statistics...")

        stats = {
            **self.stats,
            "cache_size": len(self.resampling_cache),
            "data_sharing_stats": self.data_sharing_manager.stats,
        }

        print(f"✅ [STATS] Statistics retrieved:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Resampling operations: {stats['resampling_operations']}")
        print(f"   Quality repairs: {stats['quality_repairs']}")
        print(f"   Memory cleanups: {stats['memory_cleanups']}")

        return stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        print(f"📊 [CACHE_INFO] Getting cache information...")

        cache_info = {
            "resampling_cache_size": len(self.resampling_cache),
            "resampling_cache_limit": self.resampling_cache_size,
            "data_sharing_cache_keys": list(
                self.data_sharing_manager._data_cache.keys()
            ),
        }

        print(f"✅ [CACHE_INFO] Cache information retrieved:")
        print(
            f"   Resampling cache: {cache_info['resampling_cache_size']}/{cache_info['resampling_cache_limit']}"
        )
        print(
            f"   Data sharing cache keys: {len(cache_info['data_sharing_cache_keys'])}"
        )

        return cache_info


# Global instance
_unified_data_orchestrator: Optional[UnifiedDataOrchestrator] = None


def get_unified_data_orchestrator(config: dict[str, Any]) -> UnifiedDataOrchestrator:
    """Get or create the global unified data orchestrator instance."""
    global _unified_data_orchestrator

    print(f"🔧 [GLOBAL] Getting unified data orchestrator instance...")

    if _unified_data_orchestrator is None:
        print(f"🔄 [GLOBAL] Creating new orchestrator instance...")
        _unified_data_orchestrator = UnifiedDataOrchestrator(config)
        print(f"✅ [GLOBAL] New orchestrator instance created")
    else:
        print(f"✅ [GLOBAL] Using existing orchestrator instance")

    return _unified_data_orchestrator


async def initialize_unified_data_orchestrator(config: dict[str, Any]) -> bool:
    """Initialize the global unified data orchestrator."""
    global _unified_data_orchestrator

    print(f"🚀 [GLOBAL] Initializing global unified data orchestrator...")

    if _unified_data_orchestrator is None:
        print(f"🔄 [GLOBAL] Creating new orchestrator instance...")
        _unified_data_orchestrator = UnifiedDataOrchestrator(config)

    success = await _unified_data_orchestrator.initialize()

    if success:
        print(f"✅ [GLOBAL] Global orchestrator initialized successfully")
    else:
        print(f"❌ [GLOBAL] Failed to initialize global orchestrator")

    return success


async def cleanup_unified_data_orchestrator():
    """Cleanup the global unified data orchestrator."""
    global _unified_data_orchestrator

    print(f"🧹 [GLOBAL] Cleaning up global unified data orchestrator...")

    if _unified_data_orchestrator is not None:
        await _unified_data_orchestrator.cleanup()
        _unified_data_orchestrator = None
        print(f"✅ [GLOBAL] Global orchestrator cleaned up")
    else:
        print(f"⚠️ [GLOBAL] No orchestrator instance to clean up")
