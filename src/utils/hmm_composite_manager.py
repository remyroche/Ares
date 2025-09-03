#!/usr/bin/env python3
"""
HMM Composite Cluster Manager

Centralized manager for HMM composite cluster files that can be used by:
- step3_hmm_regime_discovery (to create files)
- VectorizedAdvancedFeatureEngineering (to check if files exist)
- CompositeHMMRegimeSystem (to load files)

This ensures consistent behavior and prevents infinite loops.
"""

from __future__ import annotations

import contextlib
import json
import os
import os.path
import time
from typing import Any

import pandas as pd

from src.core.decorators import handles_errors
from src.training.steps.market_analysis.step03_hmm_regime_discovery import run_step as run_step3
from src.utils.logger import system_logger

# Module-level sets to avoid duplicate logs across multiple instances
# This prevents log spam when different components instantiate the manager separately
_GLOBAL_LOGGED_LOADS: set[str] = set()
_GLOBAL_LOGGED_EVENTS: set[str] = set()


class HMMCompositeManager:
    """Centralized manager for HMM composite cluster files."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("HMMCompositeManager")
        self._cache: dict[str, dict[str, Any]] = (
            {}
        )  # Simple cache to avoid repeated file checks/loads
        # Use shared global sets so multiple instances do not re-log the same events
        self._logged_loads = _GLOBAL_LOGGED_LOADS
        self._logged_events = _GLOBAL_LOGGED_EVENTS

        # Enhanced features
        self._file_metadata_cache: dict[str, dict[str, Any]] = (
            {}
        )  # Cache for file metadata
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup cache every hour

    def _get_file_paths(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> dict[str, str]:
        """Get all required file paths for HMM composite clusters."""
        base_name = f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}"
        return {
            "composite_clusters": os.path.join(data_dir, f"{base_name}.parquet"),
            "block_states": os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
            ),
            "intensity": os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            ),
            "meta": os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
            ),
            "basic_meta": os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_basic_meta_{timeframe}.json",
            ),
        }

    def _check_files_exist(self, file_paths: dict[str, str]) -> tuple[bool, list[str]]:
        """Check if all required files exist."""
        missing_files: list[str] = []
        all_exist = True

        for file_type, file_path in file_paths.items():
            if not os.path.exists(file_path):
                all_exist = False
                missing_files.append(f"{file_type} ({file_path})")

        return all_exist, missing_files

    def _cleanup_cache_if_needed(self) -> None:
        """Clean up cache if it's been too long since last cleanup."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            # Remove old cache entries (older than 1 hour)
            cutoff_time = current_time - 3600
            old_keys = [
                k
                for k, v in self._cache.items()
                if isinstance(v, dict) and v.get("timestamp", 0) < cutoff_time
            ]
            for key in old_keys:
                with contextlib.suppress(Exception):
                    del self._cache[key]

            self._last_cleanup = current_time
            self.logger.debug(
                f"🧹 Cache cleanup completed - removed {len(old_keys)} old entries",
            )

    @handles_errors(fallback=None)
    def load_block_states(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> pd.DataFrame | None:
        """
        Load HMM block states if they exist.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path

        Returns:
            DataFrame with block states if found, None otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        block_states_path = file_paths["block_states"]
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|block_states"

        # Cleanup cache if needed
        self._cleanup_cache_if_needed()

        # Return cached DataFrame if already loaded during this run
        if cache_key in self._cache:
            return self._cache[cache_key]["data"]  # type: ignore[return-value]

        if not os.path.exists(block_states_path):
            self.logger.info(
                f"HMM block states not found for {exchange}_{symbol}_{timeframe}",
            )
            return None

        try:
            df = pd.read_parquet(block_states_path)
            # Cache for subsequent calls and log only once per key
            self._cache[cache_key] = {"data": df, "timestamp": time.time()}
            if cache_key not in self._logged_loads:
                self.logger.info(
                    f"✅ Loaded HMM block states for {exchange}_{symbol}_{timeframe} ({len(df)} rows)",
                )
                self._logged_loads.add(cache_key)
            return df
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to load HMM block states: {e}")
            return None

    @handles_errors(fallback=None)
    def load_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
        auto_create: bool = False,
    ) -> pd.DataFrame | None:
        """
        Load HMM composite clusters if they exist.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path
            auto_create: If True, automatically create clusters if they don't exist

        Returns:
            DataFrame with composite clusters if found, None otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        composite_path = file_paths["composite_clusters"]
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|composite"

        # Cleanup cache if needed
        self._cleanup_cache_if_needed()

        # Return cached DataFrame if already loaded during this run
        if cache_key in self._cache:
            return self._cache[cache_key]["data"]  # type: ignore[return-value]

        if not os.path.exists(composite_path):
            event_key = (
                f"{cache_key}|not_found|{'auto' if auto_create else 'meta_only'}"
            )
            if auto_create:
                if event_key not in self._logged_events:
                    self.logger.info(
                        f"HMM composite clusters not found for {exchange}_{symbol}_{timeframe}; will create them",
                    )
                    self._logged_events.add(event_key)
                # Return None to indicate they need to be created
                return None
            if event_key not in self._logged_events:
                self.logger.info(
                    f"HMM composite clusters not found for {exchange}_{symbol}_{timeframe}; using meta-only",
                )
                self._logged_events.add(event_key)
            return None

        try:
            df = pd.read_parquet(composite_path)
            # Cache for subsequent calls and log only once per key
            self._cache[cache_key] = {"data": df, "timestamp": time.time()}
            if cache_key not in self._logged_loads:
                self.logger.info(
                    f"✅ Loaded HMM composite clusters for {exchange}_{symbol}_{timeframe} ({len(df)} rows)",
                )
                self._logged_loads.add(cache_key)
            return df
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to load HMM composite clusters: {e}")
            return None

    @handles_errors(fallback=None)
    def load_meta(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> dict[str, Any] | None:
        """
        Load HMM meta information if it exists.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path

        Returns:
            Dictionary with meta information if found, None otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        meta_path = file_paths["meta"]
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|meta"

        # Cleanup cache if needed
        self._cleanup_cache_if_needed()

        # Return cached meta if already loaded during this run
        if cache_key in self._cache:
            return self._cache[cache_key]["data"]  # type: ignore[return-value]

        if not os.path.exists(meta_path):
            self.logger.info(f"HMM meta not found for {exchange}_{symbol}_{timeframe}")
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Cache for subsequent calls and log only once per key
            self._cache[cache_key] = {"data": meta, "timestamp": time.time()}
            if cache_key not in self._logged_loads:
                self.logger.info(
                    f"✅ Loaded HMM meta for {exchange}_{symbol}_{timeframe}",
                )
                self._logged_loads.add(cache_key)
            return meta
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to load HMM meta: {e}")
            return None

    @handles_errors(fallback=None)
    def load_intensity(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> pd.DataFrame | None:
        """
        Load HMM composite intensity if it exists.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path

        Returns:
            DataFrame with intensity data if found, None otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        intensity_path = file_paths["intensity"]
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|intensity"

        # Cleanup cache if needed
        self._cleanup_cache_if_needed()

        # Return cached DataFrame if already loaded during this run
        if cache_key in self._cache:
            return self._cache[cache_key]["data"]  # type: ignore[return-value]

        if not os.path.exists(intensity_path):
            self.logger.info(
                f"HMM intensity not found for {exchange}_{symbol}_{timeframe}",
            )
            return None

        try:
            df = pd.read_parquet(intensity_path)
            # Cache for subsequent calls and log only once per key
            self._cache[cache_key] = {"data": df, "timestamp": time.time()}
            if cache_key not in self._logged_loads:
                self.logger.info(
                    f"✅ Loaded HMM intensity for {exchange}_{symbol}_{timeframe} ({len(df)} rows)",
                )
                self._logged_loads.add(cache_key)
            return df
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to load HMM intensity: {e}")
            return None

    @handles_errors(fallback=None)
    def load_basic_meta(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> dict[str, Any] | None:
        """
        Load HMM basic meta information if it exists.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path

        Returns:
            Dictionary with basic meta information if found, None otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        basic_meta_path = file_paths["basic_meta"]
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|basic_meta"

        # Cleanup cache if needed
        self._cleanup_cache_if_needed()

        # Return cached meta if already loaded during this run
        if cache_key in self._cache:
            return self._cache[cache_key]["data"]  # type: ignore[return-value]

        if not os.path.exists(basic_meta_path):
            self.logger.info(
                f"HMM basic meta not found for {exchange}_{symbol}_{timeframe}",
            )
            return None

        try:
            with open(basic_meta_path) as f:
                meta = json.load(f)
            # Cache for subsequent calls and log only once per key
            self._cache[cache_key] = {"data": meta, "timestamp": time.time()}
            if cache_key not in self._logged_loads:
                self.logger.info(
                    f"✅ Loaded HMM basic meta for {exchange}_{symbol}_{timeframe}",
                )
                self._logged_loads.add(cache_key)
            return meta
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to load HMM basic meta: {e}")
            return None

    @handles_errors(fallback=False)
    async def create_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
        force_rerun: bool = False,
        lookback_days: int = 180,
    ) -> bool:
        """
        Create HMM composite clusters if they don't exist or if force_rerun is True.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path
            force_rerun: If True, recreate files even if they exist
            lookback_days: Number of days to look back for data

        Returns:
            True if files were created successfully, False otherwise
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)

        # Check if files already exist
        all_exist, missing_files = self._check_files_exist(file_paths)

        if all_exist and not force_rerun:
            self.logger.info(
                f"✅ All HMM composite cluster files already exist for {exchange}_{symbol}_{timeframe} - skipping creation",
            )
            return True

        if not all_exist:
            self.logger.info(
                f"⚠️ Some HMM files missing - will create: {', '.join(missing_files)}",
            )
        else:
            self.logger.info(
                "🔄 Force rerun enabled - will recreate all HMM composite cluster files",
            )

        self.logger.info(
            f"🚀 Creating HMM composite clusters for {exchange}_{symbol}_{timeframe}",
        )

        success = await run_step3(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
            lookback_days=lookback_days,
        )

        if success:
            self.logger.info(
                f"✅ Successfully created HMM composite clusters for {exchange}_{symbol}_{timeframe}",
            )
            return True
        self.logger.error(
            f"❌ Failed to create HMM composite clusters for {exchange}_{symbol}_{timeframe}",
        )
        return False

    @handles_errors(fallback=None)
    async def get_or_create_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
        force_rerun: bool = False,
        lookback_days: int = 180,
    ) -> pd.DataFrame | None:
        """
        Get HMM composite clusters, creating them if they don't exist.

        Args:
            exchange: Exchange name (e.g., 'BINANCE')
            symbol: Symbol name (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m')
            data_dir: Data directory path
            force_rerun: If True, recreate files even if they exist
            lookback_days: Number of days to look back for data

        Returns:
            DataFrame with composite clusters if successful, None otherwise
        """
        # Try to load existing files first
        df = self.load_composite_clusters(exchange, symbol, timeframe, data_dir)

        if df is not None and not force_rerun:
            return df

        # If files don't exist or force_rerun is True, create them
        success = await self.create_composite_clusters(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
            lookback_days=lookback_days,
        )

        if success:
            # Try to load the newly created files
            return self.load_composite_clusters(exchange, symbol, timeframe, data_dir)

        return None

    def get_file_info(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> dict[str, Any]:
        """
        Get information about HMM files for a given symbol/exchange/timeframe.

        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            data_dir: Data directory path

        Returns:
            Dictionary with file information
        """
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        file_info: dict[str, Any] = {}

        for file_type, file_path in file_paths.items():
            if os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    file_info[file_type] = {
                        "exists": True,
                        "size_bytes": stat.st_size,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": stat.st_mtime,
                        "path": file_path,
                    }
                except Exception as e:  # pragma: no cover - defensive
                    file_info[file_type] = {
                        "exists": True,
                        "error": str(e),
                        "path": file_path,
                    }
            else:
                file_info[file_type] = {"exists": False, "path": file_path}

        return file_info

    def validate_files(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/training",
    ) -> dict[str, Any]:
        """
        Validate HMM files for a given symbol/exchange/timeframe.

        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            data_dir: Data directory path

        Returns:
            Dictionary with validation results
        """
        validation_results: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": self.get_file_info(exchange, symbol, timeframe, data_dir),
        }

        # Check if all required files exist
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        all_exist, missing_files = self._check_files_exist(file_paths)

        if not all_exist:
            validation_results["valid"] = False
            validation_results["errors"].extend(
                [f"Missing file: {f}" for f in missing_files],
            )

        # Try to load each file to validate they can be read
        for file_type, file_path in file_paths.items():
            if os.path.exists(file_path):
                try:
                    if file_type in ["composite_clusters", "block_states", "intensity"]:
                        df = pd.read_parquet(file_path)
                        if df.empty:
                            validation_results["warnings"].append(
                                f"{file_type} is empty",
                            )
                    elif file_type in ["meta", "basic_meta"]:
                        with open(file_path) as f:
                            json.load(f)  # Validate JSON
                except Exception as e:
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Failed to read {file_type}: {e}",
                    )

        return validation_results

    def clear_cache(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """
        Clear cache entries for specific or all files.

        Args:
            exchange: Exchange name (optional; if None clears all)
            symbol: Symbol name (optional; if None clears all)
            timeframe: Timeframe (optional; if None clears all)
        """
        if exchange is None and symbol is None and timeframe is None:
            # Fallback implementation for exchange is None and symbol is None and timeframe
            # Clear all cache
            self._cache.clear()
            self.logger.info("🧹 Cleared all HMM composite manager cache")
        else:
            # Clear specific cache entries
            keys_to_remove: list[str] = []
            for key in list(self._cache.keys()):
                if exchange and exchange not in key:
                    continue
                if symbol and symbol not in key:
                    continue
                if timeframe and timeframe not in key:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                with contextlib.suppress(Exception):
                    del self._cache[key]

            self.logger.info(
                f"🧹 Cleared {len(keys_to_remove)} cache entries for {exchange}_{symbol}_{timeframe}",
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._cache)
        total_size_mb = sum(
            len(str(v.get("data", ""))) / (1024 * 1024)
            for v in self._cache.values()
            if isinstance(v, dict) and "data" in v
        )

        return {
            "total_entries": total_entries,
            "total_size_mb": total_size_mb,
            "logged_loads": len(self._logged_loads),
            "logged_events": len(self._logged_events),
            "last_cleanup": self._last_cleanup,
        }


# Global instance for easy access
_hmm_composite_manager: HMMCompositeManager | None = None


def get_hmm_composite_manager() -> HMMCompositeManager:
    """Get the global HMM composite manager instance."""
    global _hmm_composite_manager
    if _hmm_composite_manager is None:
        # Fallback implementation for _hmm_composite_manager
        _hmm_composite_manager = HMMCompositeManager()
    return _hmm_composite_manager
