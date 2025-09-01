# src/training/data_sharing_manager.py

import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
    validate_step_prerequisites,
)


class DataSharingManager:
    """Manages data sharing between training steps to eliminate redundant data loading.

    This manager provides a centralized way to load and share data between steps,
    with intelligent caching and memory management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataSharingManager")

        # Data cache - stores loaded data by key
        self._data_cache: dict[str, Any] = {}
        self._cache_metadata: dict[str, dict[str, Any]] = {}

        # Cache configuration
        self.cache_config = config.get("data_sharing", {})
        self.max_cache_size_gb = self.cache_config.get("max_cache_size_gb", 8.0)
        self.cache_ttl_hours = self.cache_config.get("cache_ttl_hours", 24)
        self.enable_memory_optimization = self.cache_config.get(
            "enable_memory_optimization", True,
        )

        # Unified data loader
        self.data_loader = get_unified_data_loader(config)

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_data_loaded_gb": 0.0,
            "memory_saved_gb": 0.0,
        }

    def _generate_cache_key(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        lookback_days: int,
        data_type: str = "unified",
    ) -> str:
        """Generate a unique cache key for data."""
        return f"{exchange}_{symbol}_{timeframe}_{lookback_days}_{data_type}"

    def _remove_from_cache(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._data_cache:
            # Estimate size before removal
            data_size = self._get_data_size_gb(self._data_cache[key])
            self.stats["memory_saved_gb"] += data_size

            # Remove data and metadata
            del self._data_cache[key]
            if key in self._cache_metadata:
                del self._cache_metadata[key]

    def _evict_if_needed(self, required_size_gb: float) -> None:
        """Evict cache entries if needed to make space."""
        current_cache_size = sum(
            self._get_data_size_gb(data) for data in self._data_cache.values()
        )

        if current_cache_size + required_size_gb > self.max_cache_size_gb:
            self.logger.info(
                f"⚠️ Cache full ({current_cache_size:.2f}GB), evicting old entries...",
            )

            # Sort by timestamp (oldest first)
            sorted_keys = sorted(
                self._cache_metadata.keys(),
                key=lambda k: self._cache_metadata[k]["timestamp"],
            )

            # Remove oldest entries until we have enough space
            for key in sorted_keys:
                self._remove_from_cache(key)
                current_cache_size = sum(
                    self._get_data_size_gb(data) for data in self._data_cache.values()
                )
                if current_cache_size + required_size_gb <= self.max_cache_size_gb:
                    break

    @validate_step_prerequisites(
        required_directories=["data_cache", "data_cache/unified"],
        min_memory_gb=4.0,
        min_disk_gb=5.0,
        required_packages=["pandas", "numpy"],
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Data Sharing Manager",
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
        lookahead_bias_prevention=True,
    )
    @resource_monitor(
        memory_threshold_gb=8.0,
        cpu_threshold_percent=70.0,
        disk_threshold_gb=10.0,
        monitor_interval=30.0,
        auto_cleanup=True,
    )
    @memory_efficient(
        chunk_size=15000,
        streaming_processing=True,
        memory_pool=True,
        cleanup_frequency=35,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling=True,
        error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=3,
        recovery_timeout=180.0,
        expected_exception=Exception,
        monitor_interval=30.0,
    )
    @validate_step_output(
        required_files=["data_cache/unified/{exchange}/{symbol}/{timeframe}/*.parquet"],
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        performance_thresholds={"loading_time_minutes": 15.0, "memory_usage_gb": 4.0},
        format_validation=True,
    )
    @quality_gate(
        model_performance_thresholds={"cache_hit_rate": 0.7, "data_completeness": 0.9},
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={"data_quality_score": 0.8},
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data sharing manager get unified data",
    )
    def clear_cache(self) -> None:
        """Clear all cached data."""
        cache_size = sum(
            self._get_data_size_gb(data) for data in self._data_cache.values()
        )

        self._data_cache.clear()
        self._cache_metadata.clear()

        if self.enable_memory_optimization:
            gc.collect()

        self.logger.info(f"🧹 Cleared cache ({cache_size:.2f}GB freed)")


# Global instance for easy access
_data_sharing_manager: DataSharingManager | None = None


