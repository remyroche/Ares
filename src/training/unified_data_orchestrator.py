# src/training/unified_data_orchestrator.py

from src.core.decorators import (
    cached,
    circuit_breaker,
    handles_errors,
    log_call,
    log_execution_time,
    validates
)

# TODO: These decorators need to be migrated to core decorators or removed
from src.utils.centralized_decorators import (
    prevent_data_leakage,
    quality_gate,
    secure_data_processing
)

"""Unified Data Orchestrator - Single Source of Truth for Data Operations."

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

from src.utils.logger import system_logger

# Import training pipeline decorators for security and troubleshooting
import copy

class UnifiedDataOrchestrator:
    """Unified Data Orchestrator - Single source of truth for all data operations."

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
            "enable_memory_optimization", True,
        )
        self.enable_quality_validation = self.orchestrator_config.get(
            "enable_quality_validation", True,
        )
        self.enable_auto_repair = self.orchestrator_config.get(
            "enable_auto_repair", True,
        )

        # Resampling configuration
        self.resampling_config = self.orchestrator_config.get("resampling", {})
        self.default_timeframes = self.resampling_config.get(
            "default_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        )
        self.resampling_cache: dict[str, pd.DataFrame] = {}
        self.resampling_cache_size = self.resampling_config.get("cache_size", 100)

        # Quality validation configuration
        self.quality_config = self.orchestrator_config.get("quality_validation", {})
        self.min_data_points = self.quality_config.get("min_data_points", 1000)
        self.max_missing_ratio = self.quality_config.get("max_missing_ratio", 0.1)
        self.max_duplicate_ratio = self.quality_config.get("max_duplicate_ratio", 0.05)

        # Statistics
        self.stats: dict[str, Any] = {
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

        _ = time.time() - start_time

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _log_memory_usage(self, context: str) -> None:
        """Log current memory usage with context."""
        memory_mb = self._get_memory_usage_mb()
        self.stats["memory_usage_history"].append(
            {"timestamp": datetime.now(), "context": context, "memory_mb": memory_mb},
        )
        self.logger.info(f"Memory usage at {context}: {memory_mb:.2f} MB")

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=50.0,
        disk_threshold_gb=1.0,
        monitor_interval=10.0,
        auto_cleanup=True,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=3,
        recovery_timeout=60.0,
        expected_exception=Exception,
        monitor_interval=10.0,
    )
    @validates(
        data_quality_checks={},
        performance_thresholds={"initialization_time_seconds": 30.0},
        format_validation=False,
    )
    @quality_gate(
        data_quality_metrics={},
        validation_score_requirements={"initialization_success": 1.0},
    )
    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the orchestrator."""
        start_time = time.time()
        self._log_memory_usage("initialize_start")

        try:
            self.logger.info("🚀 Initializing Unified Data Orchestrator")

            # Initialize components
            await self.data_sharing_manager.initialize()

            # Start cache cleanup task
            if self.enable_caching:
                self._cache_cleanup_task = asyncio.create_task(
                    self._cache_cleanup_loop(),
                )

            init_time = time.time() - start_time
            self.stats["operation_times"]["initialization"] = init_time

            self._log_memory_usage("initialize_end")

            self.logger.info("✅ Unified Data Orchestrator initialized successfully")
            return True

        except Exception as e:
            init_time = time.time() - start_time
            self.logger.exception(f"❌ Failed to initialize Unified Data Orchestrator: {e}")
            return False

    @validates(
        required_packages=["asyncio", "gc"], context="Orchestrator Cleanup",
    )
    @secure_data_processing(
        backup_before=False,
        integrity_checks=False,
        memory_cleanup=True,
        data_validation=False,
    )
    @log_execution_time(
        memory_threshold_gb=1.0,
        cpu_threshold_percent=30.0,
        monitor_interval=5.0,
        auto_cleanup=True,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        expected_exception=Exception,
        monitor_interval=5.0,
    )
    @validates(
        data_quality_checks={},
        performance_thresholds={"cleanup_time_seconds": 10.0},
        format_validation=False,
    )
    @quality_gate(
        data_quality_metrics={}, validation_score_requirements={"cleanup_success": 1.0},
    )
    @handles_errors(
        exceptions=(Exception,), default_return=None, context="orchestrator cleanup",
    )
    async def cleanup(self) -> None:
        """Cleanup resources."""
        start_time = time.time()
        self._log_memory_usage("cleanup_start")

        try:
            if self._cache_cleanup_task:
                self._cache_cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._cache_cleanup_task

            # Clear caches
            _ = len(self.resampling_cache)
            self.resampling_cache.clear()

            self._force_garbage_collection()

            cleanup_time = time.time() - start_time
            self.stats["operation_times"]["cleanup"] = cleanup_time

            self._log_memory_usage("cleanup_end")

            self.logger.info("🧹 Unified Data Orchestrator cleanup completed")

        except Exception as e:
            cleanup_time = time.time() - start_time
            self.logger.exception(f"❌ Error during cleanup: {e}")

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=4.0,
        cpu_threshold_percent=70.0,
        disk_threshold_gb=2.0,
        monitor_interval=30.0,
        auto_cleanup=True,
    )
    @cached(
        chunk_size=50000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=100,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=3,
        recovery_timeout=120.0,
        expected_exception=Exception,
        monitor_interval=30.0,
    )
    @validates(
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
    @handles_errors(
        exceptions=(Exception,), default_return=None, context="unified data loading",
    )
    async def get_unified_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        lookback_days: int | None = None,
        force_reload: bool = False,
        validate_quality: bool = True,
        auto_repair: bool = True,
    ) -> pd.DataFrame | None:
        """Get unified data with comprehensive fallback strategies and quality validation."

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

        self._log_memory_usage(f"data_load_start_{request_id}")
        self.stats["total_requests"] += 1

        try:
            self.logger.info(
                f"🔄 Loading unified data: {exchange}_{symbol}_{timeframe}",
            )

            # Step 1: Try data sharing manager first (most efficient)
            if not force_reload:
                cache_start = time.time()

                data = await self.data_sharing_manager.get_unified_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    force_reload=False,
                )

                _ = time.time() - cache_start

                if data is not None and not data.empty:
                    self.stats["cache_hits"] += 1
                    _ = data.memory_usage(deep=True).sum() / 1024 / 1024
                    self.logger.info(f"✅ Data loaded from cache: {data.shape}")

                    if validate_quality:
                        validation_start = time.time()
                        data = await self._validate_and_repair_data(data, auto_repair)
                        _ = time.time() - validation_start

                    total_time = time.time() - start_time
                    self.stats["operation_times"][f"cache_hit_{request_id}"] = (
                        total_time
                    )
                    self._log_memory_usage(f"data_load_cache_hit_{request_id}")

                    return data

            self.stats["cache_misses"] += 1

            # Step 2: Try unified data loader
            loader_start = time.time()

            data = await self.data_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )

            _ = time.time() - loader_start

            if data is not None and not data.empty:
                _ = data.memory_usage(deep=True).sum() / 1024 / 1024
                self.logger.info(f"✅ Data loaded from unified loader: {data.shape}")

                if validate_quality:
                    validation_start = time.time()
                    data = await self._validate_and_repair_data(data, auto_repair)
                    _ = time.time() - validation_start

                # Cache the data
                if self.enable_caching:
                    cache_start = time.time()
                    await self.data_sharing_manager.cache_unified_data(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        lookback_days=lookback_days,
                        data=data,
                    )
                    _ = time.time() - cache_start

                total_time = time.time() - start_time
                self.stats["operation_times"][f"unified_loader_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"data_load_unified_success_{request_id}")

                return data

            # Step 3: Fallback to raw data loading and conversion
            self.logger.warning(
                "⚠️ Unified data not available, trying raw data conversion",
            )
            raw_start = time.time()

            data = await self._load_and_convert_raw_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )

            _ = time.time() - raw_start

            if data is not None and not data.empty:
                _ = data.memory_usage(deep=True).sum() / 1024 / 1024
                self.logger.info(f"✅ Data loaded from raw conversion: {data.shape}")

                if validate_quality:
                    validation_start = time.time()
                    data = await self._validate_and_repair_data(data, auto_repair)
                    _ = time.time() - validation_start

                total_time = time.time() - start_time
                self.stats["operation_times"][f"raw_conversion_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"data_load_raw_success_{request_id}")

                return data

            # All methods failed
            total_time = time.time() - start_time
            self.logger.error(
                f"❌ Failed to load data for {exchange}_{symbol}_{timeframe}",
            )
            self._log_memory_usage(f"data_load_failed_{request_id}")
            return None

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"❌ Error loading unified data: {e}")
            self._log_memory_usage(f"data_load_exception_{request_id}")
            return None

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=8.0,
        cpu_threshold_percent=80.0,
        disk_threshold_gb=5.0,
        monitor_interval=30.0,
        auto_cleanup=True,
    )
    @cached(
        chunk_size=25000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=50,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=2,
        recovery_timeout=180.0,
        expected_exception=Exception,
        monitor_interval=30.0,
    )
    @validates(
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
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="multi-timeframe data loading",
    )
    async def get_multi_timeframe_data(
        self,
        symbol: str,
        exchange: str,
        timeframes: list[str] | None = None,
        lookback_days: int | None = None,
        force_reload: bool = False,
        validate_quality: bool = True,
        auto_repair: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Get data for multiple timeframes with intelligent resampling."

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

        self._log_memory_usage(f"multi_tf_start_{request_id}")

        if timeframes is None:
            timeframes = self.default_timeframes

        try:
            self.logger.info(f"🔄 Loading multi-timeframe data: {exchange}_{symbol}")
            self.logger.info(f"   Timeframes: {timeframes}")

            # Sort timeframes by resolution (highest to lowest)
            timeframe_order = self._sort_timeframes_by_resolution(timeframes)

            # Load base timeframe first (usually 1m)
            base_timeframe = self._get_base_timeframe(timeframe_order)
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

            _ = time.time() - base_start

            if base_data is None or base_data.empty:
                self.logger.error(f"❌ Failed to load base data for {base_timeframe}")
                return {}

            _ = base_data.memory_usage(deep=True).sum() / 1024 / 1024

            # Load or resample data for each timeframe
            result: dict[str, pd.DataFrame] = {}
            successful_timeframes = 0

            for _i, timeframe in enumerate(timeframe_order, 1):
                tf_start = time.time()

                if timeframe == base_timeframe:
                    result[timeframe] = base_data
                    successful_timeframes += 1
                    _ = time.time() - tf_start
                    continue

                # Try to load existing data for this timeframe
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

                _ = time.time() - existing_start

                if existing_data is not None and not existing_data.empty:
                    result[timeframe] = existing_data
                    _ = (
                        existing_data.memory_usage(deep=True).sum() / 1024 / 1024
                    )
                    self.logger.info(
                        f"✅ Loaded existing data for {timeframe}: {existing_data.shape}",
                    )
                    successful_timeframes += 1
                else:
                    # Resample from base data
                    resample_start = time.time()

                    resampled_data = await self._resample_data(
                        data=base_data,
                        from_timeframe=base_timeframe,
                        to_timeframe=timeframe,
                        symbol=symbol,
                        exchange=exchange,
                    )

                    _ = time.time() - resample_start

                    if resampled_data is not None and not resampled_data.empty:
                        result[timeframe] = resampled_data
                        _ = (
                            resampled_data.memory_usage(deep=True).sum() / 1024 / 1024
                        )
                        self.logger.info(
                            f"✅ Resampled data for {timeframe}: {resampled_data.shape}",
                        )
                        successful_timeframes += 1

                        # Cache the resampled data
                        if self.enable_caching:
                            cache_start = time.time()
                            await self.data_sharing_manager.cache_unified_data(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                lookback_days=lookback_days,
                                data=resampled_data,
                            )
                            _ = time.time() - cache_start
                    else:
                        self.logger.warning(
                            f"⚠️ Failed to resample data for {timeframe}",
                        )

                _ = time.time() - tf_start

            total_time = time.time() - start_time
            _ = (
                sum(df.memory_usage(deep=True).sum() for df in result.values())
                / 1024
                / 1024
            )

            self.stats["operation_times"][f"multi_tf_{request_id}"] = total_time
            self._log_memory_usage(f"multi_tf_end_{request_id}")

            self.logger.info(
                f"✅ Multi-timeframe data loading completed: {len(result)} timeframes",
            )
            return result

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"❌ Error loading multi-timeframe data: {e}")
            self._log_memory_usage(f"multi_tf_exception_{request_id}")
            return {}

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=60.0,
        monitor_interval=15.0,
        auto_cleanup=True,
    )
    @cached(
        chunk_size=10000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=25,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=Exception,
        monitor_interval=15.0,
    )
    @validates(
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
    @handles_errors(
        exceptions=(Exception,), default_return=None, context="data resampling",
    )
    async def _resample_data(
        self,
        data: pd.DataFrame,
        from_timeframe: str,
        to_timeframe: str,
        symbol: str,
        exchange: str,
    ) -> pd.DataFrame | None:
        """Resample data from one timeframe to another with caching."

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

        self._log_memory_usage(f"resample_start_{request_id}")

        try:
            self.stats["resampling_operations"] += 1

            # Generate cache key
            cache_start = time.time()
            cache_key = self._generate_resampling_cache_key(
                data, from_timeframe, to_timeframe, symbol, exchange,
            )
            _ = time.time() - cache_start

            # Check cache
            if cache_key in self.resampling_cache:
                cached_data = self.resampling_cache[cache_key].copy()
                _ = cached_data.memory_usage(deep=True).sum() / 1024 / 1024
                self.logger.info(
                    f"📋 Using cached resampled data for {from_timeframe} -> {to_timeframe}",
                )

                total_time = time.time() - start_time
                self.stats["operation_times"][f"resample_cache_hit_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"resample_cache_hit_{request_id}")

                return cached_data

            # Perform resampling
            resample_start = time.time()

            resampled_data = self._perform_resampling(
                data, from_timeframe, to_timeframe,
            )

            _ = time.time() - resample_start

            if resampled_data is not None and not resampled_data.empty:
                _ = (
                    resampled_data.memory_usage(deep=True).sum() / 1024 / 1024
                )

                # Cache the result
                cache_start = time.time()

                if len(self.resampling_cache) < self.resampling_cache_size:
                    self.resampling_cache[cache_key] = resampled_data.copy()
                else:
                    # Remove oldest entry
                    oldest_key = next(iter(self.resampling_cache))
                    del self.resampling_cache[oldest_key]
                    self.resampling_cache[cache_key] = resampled_data.copy()

                _ = time.time() - cache_start

                total_time = time.time() - start_time
                self.stats["operation_times"][f"resample_success_{request_id}"] = (
                    total_time
                )
                self._log_memory_usage(f"resample_success_{request_id}")

                self.logger.info(
                    f"✅ Resampled {from_timeframe} -> {to_timeframe}: {resampled_data.shape}",
                )
                return resampled_data

            total_time = time.time() - start_time
            self.stats["operation_times"][f"resample_failed_{request_id}"] = total_time
            self._log_memory_usage(f"resample_failed_{request_id}")

            return None

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"❌ Error resampling data: {e}")
            self._log_memory_usage(f"resample_exception_{request_id}")
            return None

    def _perform_resampling(
        self, data: pd.DataFrame, from_timeframe: str, to_timeframe: str,
    ) -> pd.DataFrame | None:
        """Perform the actual resampling operation."""
        try:

            # Ensure we have a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                if "timestamp" in data.columns:
                    data = data.copy()
                    data.index = pd.to_datetime(data["timestamp"], errors="coerce")
                    data = data.sort_index()
                else:
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

            _ = timeframe_map.get(from_timeframe, "1min")
            to_offset = timeframe_map.get(to_timeframe, "1min")

            # Determine resampling direction
            from_minutes = self._timeframe_to_minutes(from_timeframe)
            to_minutes = self._timeframe_to_minutes(to_timeframe)

            if from_minutes < to_minutes:
                # Upsampling (e.g., 1m -> 5m)
                return self._upsample_data(data, to_offset)
            # Downsampling (e.g., 5m -> 1m) - not supported
            self.logger.warning(
                f"⚠️ Downsampling not supported: {from_timeframe} -> {to_timeframe}",
            )
            return None

        except Exception as e:
            self.logger.exception(f"❌ Error in resampling operation: {e}")
            return None

    def _upsample_data(
        self, data: pd.DataFrame, target_offset: str,
    ) -> pd.DataFrame | None:
        """Upsample data to higher timeframe."""
        try:

            # Resample OHLCV data
            if all(
                col in data.columns
                for col in ["open", "high", "low", "close", "volume"]
            ):
                resampled = (
                    data.resample(target_offset)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        },
                    )
                    .dropna()
                )
            else:
                # Fallback for other data types
                resampled = data.resample(target_offset).last().dropna()

            return resampled

        except Exception as e:
            self.logger.exception(f"❌ Error upsampling data: {e}")
            return None

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=1.0,
        cpu_threshold_percent=50.0,
        monitor_interval=10.0,
        auto_cleanup=True,
    )
    @cached(
        chunk_size=5000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=20,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=10,
        recovery_timeout=30.0,
        expected_exception=Exception,
        monitor_interval=10.0,
    )
    @validates(
        data_quality_checks={"no_nan_values": False, "min_rows": 5},
        performance_thresholds={"validation_time_seconds": 15.0},
        format_validation=True,
    )
    @quality_gate(
        data_quality_metrics={"completeness": 0.7, "consistency": 0.6},
        validation_score_requirements={"data_quality": 0.8},
    )
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data validation and repair",
    )
    async def _validate_and_repair_data(
        self,
        data: pd.DataFrame,
        auto_repair: bool = True,
    ) -> pd.DataFrame:
        """Validate data quality and repair issues if possible."

        Args:
            data: Input DataFrame
            auto_repair: Automatically repair issues

        Returns:
            Validated and repaired DataFrame

        """
        start_time = time.time()
        request_id = f"validate_{data.shape[0]}_{int(start_time)}"

        self._log_memory_usage(f"validate_start_{request_id}")

        try:
            if not self.enable_quality_validation:
                return data

            self.logger.info(f"🔍 Validating data quality: {data.shape}")

            # Check data size
            if len(data) < self.min_data_points:
                self.logger.warning(
                    f"⚠️ Insufficient data points: {len(data)} < {self.min_data_points}",
                )
                if not auto_repair:
                    return data

            # Check for missing values
            missing_start = time.time()
            missing_counts = data.isnull().sum()
            missing_ratio = missing_counts.sum() / (len(data) * len(data.columns))
            _ = time.time() - missing_start

            if missing_ratio > self.max_missing_ratio:
                self.logger.warning(
                    f"⚠️ High missing value ratio: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}",
                )
                if auto_repair:
                    repair_start = time.time()
                    data = self._repair_missing_values(data)
                    _ = time.time() - repair_start
                    self.stats["quality_repairs"] += 1

            # Check for duplicates
            duplicate_start = time.time()
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data)
            _ = time.time() - duplicate_start

            if duplicate_ratio > self.max_duplicate_ratio:
                self.logger.warning(
                    f"⚠️ High duplicate ratio: {duplicate_ratio:.2%} > {self.max_duplicate_ratio:.2%}",
                )
                if auto_repair:
                    repair_start = time.time()
                    data = data.drop_duplicates()
                    _ = time.time() - repair_start
                    self.stats["quality_repairs"] += 1

            # Check for timestamp issues
            if "timestamp" in data.columns:
                timestamp_start = time.time()
                data = self._repair_timestamp_issues(data)
                _ = time.time() - timestamp_start

            # Check for price anomalies
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                price_start = time.time()
                data = self._repair_price_anomalies(data)
                _ = time.time() - price_start

            total_time = time.time() - start_time

            self.stats["operation_times"][f"validation_{request_id}"] = total_time
            self._log_memory_usage(f"validate_end_{request_id}")

            self.logger.info(f"✅ Data validation completed: {data.shape}")
            return data

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"❌ Error validating data: {e}")
            self._log_memory_usage(f"validate_exception_{request_id}")
            return data

    def _repair_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair missing values in the data."""
        try:

            # Forward fill for OHLCV data
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            if all(col in data.columns for col in ohlcv_columns):
                data[ohlcv_columns] = data[ohlcv_columns].fillna(method="ffill")

            # Drop remaining rows with missing values
            data = data.dropna()

            self.logger.info(f"🔧 Repaired missing values: {data.shape}")
            return data

        except Exception as e:
            self.logger.exception(f"❌ Error repairing missing values: {e}")
            return data

    def _repair_timestamp_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair timestamp-related issues."""
        try:

            # Ensure timestamp column is datetime
            if "timestamp" in data.columns:
                data = data.copy()
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

                # Remove rows with invalid timestamps
                invalid_timestamps = data["timestamp"].isna()
                if invalid_timestamps.sum() > 0:
                    self.logger.warning(
                        f"⚠️ Removing {invalid_timestamps.sum()} rows with invalid timestamps",
                    )
                    data = data[~invalid_timestamps]

                # Sort by timestamp
                data = data.sort_values("timestamp")

                # Set timestamp as index
                data = data.set_index("timestamp")

            return data

        except Exception as e:
            self.logger.exception(f"❌ Error repairing timestamp issues: {e}")
            return data

    def _repair_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair price anomalies in OHLCV data."""
        try:

            # Check for negative prices
            price_columns = ["open", "high", "low", "close"]
            negative_prices = (data[price_columns] <= 0).any(axis=1)

            if negative_prices.sum() > 0:
                self.logger.warning(
                    f"⚠️ Removing {negative_prices.sum()} rows with negative prices",
                )
                data = data[~negative_prices]

            # Check for high-low inconsistencies
            if all(col in data.columns for col in ["high", "low"]):
                invalid_hl = data["high"] < data["low"]
                if invalid_hl.sum() > 0:
                    self.logger.warning(
                        f"⚠️ Removing {invalid_hl.sum()} rows with high < low",
                    )
                    data = data[~invalid_hl]

            return data

        except Exception as e:
            self.logger.exception(f"❌ Error repairing price anomalies: {e}")
            return data

    @validates(
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
    @log_execution_time(
        memory_threshold_gb=2.0,
        cpu_threshold_percent=60.0,
        disk_threshold_gb=1.0,
        monitor_interval=20.0,
        auto_cleanup=True,
    )
    @cached(
        chunk_size=10000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=30,
    )
    @log_call(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker(
        failure_threshold=5,
        recovery_timeout=90.0,
        expected_exception=Exception,
        monitor_interval=20.0,
    )
    @validates(
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
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="raw data loading and conversion",
    )
    async def _load_and_convert_raw_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        lookback_days: int | None,
    ) -> pd.DataFrame | None:
        """Load and convert raw data to unified format."""
        start_time = time.time()
        request_id = f"raw_{exchange}_{symbol}_{timeframe}_{int(start_time)}"

        self._log_memory_usage(f"raw_data_start_{request_id}")

        try:
            self.logger.info(
                f"🔄 Loading raw data for conversion: {exchange}_{symbol}_{timeframe}",
            )

            # Look for raw data files
            find_start = time.time()
            raw_data_paths = self._find_raw_data_files(symbol, exchange, timeframe)
            _ = time.time() - find_start

            if not raw_data_paths:
                self.logger.warning(
                    f"⚠️ No raw data files found for {exchange}_{symbol}_{timeframe}",
                )
                return None

            # Load and combine raw data
            load_start = time.time()
            combined_data: list[pd.DataFrame] = []
            successful_files = 0

            for _i, file_path in enumerate(raw_data_paths, 1):
                try:
                    file_start = time.time()
                    raw_data = pd.read_parquet(file_path)
                    _ = time.time() - file_start

                    if not raw_data.empty:
                        combined_data.append(raw_data)
                        successful_files += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

            _ = time.time() - load_start

            if not combined_data:
                self.logger.error(
                    f"❌ No valid raw data found for {exchange}_{symbol}_{timeframe}",
                )
                return None

            # Combine all data
            combine_start = time.time()
            data = pd.concat(combined_data, ignore_index=True)
            data = data.drop_duplicates()
            _ = time.time() - combine_start

            # Convert to unified format
            convert_start = time.time()
            unified_data = self._convert_to_unified_format(
                data, symbol, exchange, timeframe,
            )
            _ = time.time() - convert_start

            if unified_data is not None and not unified_data.empty:
                _ = (
                    unified_data.memory_usage(deep=True).sum() / 1024 / 1024
                )
                self.logger.info(
                    f"✅ Converted raw data to unified format: {unified_data.shape}",
                )

                total_time = time.time() - start_time
                self.stats["operation_times"][f"raw_data_{request_id}"] = total_time
                self._log_memory_usage(f"raw_data_success_{request_id}")

                return unified_data

            return None

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"❌ Error loading and converting raw data: {e}")
            self._log_memory_usage(f"raw_data_exception_{request_id}")
            return None

    def _find_raw_data_files(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> list[str]:
        """Find raw data files for the given parameters."""
        try:

            # Look in data_cache directory
            cache_dir = Path("data_cache")
            if not cache_dir.exists():
                return []

            # Search for files matching the pattern
            pattern = f"{exchange}_{symbol}_*_{timeframe}*.parquet"
            files = list(cache_dir.rglob(pattern))

            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            return [str(f) for f in files]

        except Exception as e:
            self.logger.exception(f"❌ Error finding raw data files: {e}")
            return []

    def _convert_to_unified_format(
        self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Convert raw data to unified format."""
        try:

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
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # Check if we have the required columns
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return None

            # Ensure timestamp is datetime
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            data = data.dropna(subset=["timestamp"])

            # Sort by timestamp
            data = data.sort_values("timestamp")

            # Add metadata columns
            data["symbol"] = symbol
            data["exchange"] = exchange
            data["timeframe"] = timeframe

            # Set timestamp as index
            data = data.set_index("timestamp")

            return data

        except Exception as e:
            self.logger.exception(f"❌ Error converting to unified format: {e}")
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

            # Create a hashable representation of the data
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values,
            ).hexdigest()

            return (
                f"{exchange}_{symbol}_{from_timeframe}_{to_timeframe}_{data_hash}"
            )

        except Exception:
            # Fallback to simple hash
            return f"{exchange}_{symbol}_{from_timeframe}_{to_timeframe}_{hash(str(data.shape))}"

    def _sort_timeframes_by_resolution(self, timeframes: list[str]) -> list[str]:
        """Sort timeframes by resolution (highest to lowest)."""
        timeframe_minutes = {tf: self._timeframe_to_minutes(tf) for tf in timeframes}
        return sorted(timeframes, key=lambda tf: timeframe_minutes[tf])

    def _get_base_timeframe(self, timeframes: list[str]) -> str:
        """Get the base timeframe (highest resolution)."""
        sorted_timeframes = self._sort_timeframes_by_resolution(timeframes)
        return sorted_timeframes[0] if sorted_timeframes else "1m"

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

    async def _cache_cleanup_loop(self) -> None:
        """Periodic cache cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cleanup_start = time.time()

                # Clean up resampling cache
                if len(self.resampling_cache) > self.resampling_cache_size * 0.8:
                    # Remove oldest entries
                    keys_to_remove = list(self.resampling_cache.keys())[
                        : len(self.resampling_cache) // 2
                    ]
                    for key in keys_to_remove:
                        del self.resampling_cache[key]

                    self.logger.info(
                        f"🧹 Cleaned up resampling cache: {len(self.resampling_cache)} entries remaining",
                    )

                # Force garbage collection
                if self.enable_memory_optimization:
                    self._force_garbage_collection()

                _ = time.time() - cleanup_start

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"❌ Error in cache cleanup loop: {e}")

    def _force_garbage_collection(self) -> None:
        """Force garbage collection."""
        try:
            gc_start = time.time()
            collected = gc.collect()
            _ = time.time() - gc_start

            self.stats["memory_cleanups"] += 1
            self.logger.debug(f"🧹 Garbage collection: collected {collected} objects")

        except Exception as e:
            self.logger.exception(f"❌ Error during garbage collection: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            "cache_size": len(self.resampling_cache),
            "data_sharing_stats": self.data_sharing_manager.stats,
        }

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        return {
            "resampling_cache_size": len(self.resampling_cache),
            "resampling_cache_limit": self.resampling_cache_size,
            "data_sharing_cache_keys": list(
                self.data_sharing_manager._data_cache.keys(),
            ),
        }

# Global instance
_unified_data_orchestrator: UnifiedDataOrchestrator | None = None

def get_unified_data_orchestrator(config: dict[str, Any]) -> UnifiedDataOrchestrator:
    """Get or create the global unified data orchestrator instance."""
    global _unified_data_orchestrator

    if _unified_data_orchestrator is None:
        _unified_data_orchestrator = UnifiedDataOrchestrator(config)
    else:
        _unified_data_orchestrator.config = config

    return _unified_data_orchestrator

async def initialize_unified_data_orchestrator(config: dict[str, Any]) -> bool:
    """Initialize the global unified data orchestrator."""
    global _unified_data_orchestrator

    if _unified_data_orchestrator is None:
        _unified_data_orchestrator = UnifiedDataOrchestrator(config)

    success = await _unified_data_orchestrator.initialize()

    if success:
        _unified_data_orchestrator.stats["initialized_at"] = datetime.now()
    else:
        system_logger.getChild("UnifiedDataOrchestrator").error("Initialization failed")

    return success

async def cleanup_unified_data_orchestrator() -> None:
    """Cleanup the global unified data orchestrator."""
    global _unified_data_orchestrator

    if _unified_data_orchestrator is not None:
        await _unified_data_orchestrator.cleanup()
        _unified_data_orchestrator = None
    else:
        system_logger.getChild("UnifiedDataOrchestrator").debug("No orchestrator to cleanup")
