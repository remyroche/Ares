# src/training/data_sharing_manager.py

import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from src.training.steps.unified_data_loader import get_unified_data_loader
from src.core.decorators import handles_errors
from src.utils.logger import system_logger

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
import asyncio

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
    """Manages data sharing between training steps to eliminate redundant data loading."

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

    def _get_data_size_gb(self, data: pd.DataFrame | dict[str, Any]) -> float:
        """Estimate the size of data in GB."""
        try:
            if isinstance(data, pd.DataFrame):
                # Estimate DataFrame size
                return data.memory_usage(deep=True).sum() / (1024**3)
            if isinstance(data, dict):
                # Estimate dict size (rough approximation)
                total_size = 0
                for value in data.values():
                    if isinstance(value, pd.DataFrame):
                        total_size += value.memory_usage(deep=True).sum()
                    elif isinstance(value, np.ndarray | list):
                        total_size += len(str(value)) * 8  # Rough estimate
                return total_size / (1024**3)
            return 0.1  # Default small size
        except Exception:
            return 0.1  # Default fallback

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, metadata in self._cache_metadata.items():
            if current_time - metadata["timestamp"] > (self.cache_ttl_hours * 3600):
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_from_cache(key)
            self.logger.info(f"🧹 Removed expired cache entry: {key}")

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

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSharingManager to a JSON-serializable dictionary."""
        return {
            "config": self.config,
            "cache_config": self.cache_config,
            "max_cache_size_gb": self.max_cache_size_gb,
            "cache_ttl_hours": self.cache_ttl_hours,
            "enable_memory_optimization": self.enable_memory_optimization,
            "stats": self.stats,
            "cache_keys": list(self._data_cache.keys()),
            "cache_metadata_keys": list(self._cache_metadata.keys()),
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"DataSharingManager(cache_size={len(self._data_cache)}, stats={self.stats})"

    def _force_garbage_collection(self) -> None:
        """Force garbage collection if memory optimization is enabled."""
        if self.enable_memory_optimization:
            gc.collect()

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
    @handles_errors(fallback=None)
    async def get_unified_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        lookback_days: int = 180,
        force_reload: bool = False,
    ) -> pd.DataFrame | None:
        """Get unified data, either from cache or by loading it."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            force_reload: Force reload even if cached

        Returns:
            DataFrame with unified data or None if loading fails

        """
        cache_key = self._generate_cache_key(symbol, exchange, timeframe, lookback_days)

        # Check if data is already cached and not expired
        if not force_reload and cache_key in self._data_cache:
            metadata = self._cache_metadata.get(cache_key, {})
            current_time = time.time()

            # Check if cache entry is still valid
            if current_time - metadata.get("timestamp", 0) < (
                self.cache_ttl_hours * 3600
            ):
                self.stats["cache_hits"] += 1
                self.logger.info(
                    f"✅ Cache hit for {cache_key} ({metadata.get('rows', 'unknown')} rows)",
                )
                return self._data_cache[cache_key]

        # Cache miss or force reload
        self.stats["cache_misses"] += 1
        self.logger.info(f"🔄 Loading unified data for {cache_key}...")

        # Clean up expired entries before loading
        self._cleanup_expired_cache()

        # Load data using unified data loader
        start_time = time.time()
        data = await self.data_loader.load_unified_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True,
        )

        if data is None or data.empty:
            self.logger.error(f"❌ Failed to load unified data for {cache_key}")
            return None

        # Calculate data size and check if we need to evict
        data_size_gb = self._get_data_size_gb(data)
        self._evict_if_needed(data_size_gb)

        # Cache the data
        self._data_cache[cache_key] = data
        self._cache_metadata[cache_key] = {
            "timestamp": time.time(),
            "rows": len(data),
            "columns": len(data.columns),
            "size_gb": data_size_gb,
            "load_time": time.time() - start_time,
        }

        self.stats["total_data_loaded_gb"] += data_size_gb

        self.logger.info(
            f"✅ Loaded and cached {cache_key}: {len(data)} rows, "
            f"{data_size_gb:.2f}GB in {time.time() - start_time:.2f}s",
        )

        return data

    def get_cached_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        lookback_days: int = 180,
    ) -> pd.DataFrame | None:
        """Get data from cache only (no loading)."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            lookback_days: Number of days to look back

        Returns:
            DataFrame from cache or None if not cached

        """
        cache_key = self._generate_cache_key(symbol, exchange, timeframe, lookback_days)

        if cache_key in self._data_cache:
            metadata = self._cache_metadata.get(cache_key, {})
            current_time = time.time()

            # Check if cache entry is still valid
            if current_time - metadata.get("timestamp", 0) < (
                self.cache_ttl_hours * 3600
            ):
                self.stats["cache_hits"] += 1
                self.logger.info(f"✅ Cache hit for {cache_key}")
                return self._data_cache[cache_key]

        return None

    def cache_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        lookback_days: int,
        data: pd.DataFrame,
        data_type: str = "unified",
    ) -> None:
        """Manually cache data."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            data: Data to cache
            data_type: Type of data being cached

        """
        cache_key = self._generate_cache_key(
            symbol, exchange, timeframe, lookback_days, data_type,
        )

        # Calculate data size and check if we need to evict
        data_size_gb = self._get_data_size_gb(data)
        self._evict_if_needed(data_size_gb)

        # Cache the data
        self._data_cache[cache_key] = data
        self._cache_metadata[cache_key] = {
            "timestamp": time.time(),
            "rows": len(data),
            "columns": len(data.columns),
            "size_gb": data_size_gb,
            "data_type": data_type,
        }

        self.logger.info(
            f"💾 Cached {cache_key}: {len(data)} rows, {data_size_gb:.2f}GB",
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        current_cache_size = sum(
            self._get_data_size_gb(data) for data in self._data_cache.values()
        )

        return {
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
            * 100,
            "current_cache_size_gb": current_cache_size,
            "max_cache_size_gb": self.max_cache_size_gb,
            "total_data_loaded_gb": self.stats["total_data_loaded_gb"],
            "memory_saved_gb": self.stats["memory_saved_gb"],
            "cached_entries": len(self._data_cache),
        }

    def log_cache_stats(self) -> None:
        """Log current cache statistics."""
        stats = self.get_cache_stats()
        self.logger.info("📊 Data Sharing Cache Statistics:")
        self.logger.info(f"   Cache hits: {stats['cache_hits']}")
        self.logger.info(f"   Cache misses: {stats['cache_misses']}")
        self.logger.info(f"   Hit rate: {stats['hit_rate']:.1f}%")
        self.logger.info(
            f"   Current cache size: {stats['current_cache_size_gb']:.2f}GB",
        )
        self.logger.info(f"   Memory saved: {stats['memory_saved_gb']:.2f}GB")
        self.logger.info(f"   Cached entries: {stats['cached_entries']}")

# Global instance for easy access
_data_sharing_manager: DataSharingManager | None = None

def get_data_sharing_manager(config: dict[str, Any]) -> DataSharingManager:
    """Get or create the global data sharing manager instance."""
    global _data_sharing_manager
    if _data_sharing_manager is None:
        _data_sharing_manager = DataSharingManager(config)
    return _data_sharing_manager

def reset_data_sharing_manager() -> None:
    """Reset the global data sharing manager instance."""
    global _data_sharing_manager
    if _data_sharing_manager is not None:
        _data_sharing_manager.clear_cache()
    _data_sharing_manager = None
