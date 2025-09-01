#!/usr/bin/env python3
"""
HMM Composite Cluster Manager

Centralized manager for HMM composite cluster files that can be used by:
- step03_hmm_regime_discovery (to create files)
- VectorizedAdvancedFeatureEngineering (to check if files exist)
- CompositeHMMRegimeSystem (to load files)

This ensures consistent behavior and prevents infinite loops.
"""


import json
import os
import time
from typing import Any

import pandas as pd

from src.training.steps.step03_hmm_regime_discovery import run_step as run_step3
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Module-level sets to avoid duplicate logs across multiple instances
# This prevents log spam when different components instantiate the manager separately
_GLOBAL_LOGGED_LOADS: set[str] = set()
_GLOBAL_LOGGED_EVENTS: set[str] = set()


class HMMCompositeManager:
    """Centralized manager for HMM composite cluster files."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("HMMCompositeManager")
        self._cache: dict[str, dict[str, Any]] = {}  # Simple cache to avoid repeated file checks/loads
        # Use shared global sets so multiple instances do not re-log the same events
        self._logged_loads = _GLOBAL_LOGGED_LOADS
        self._logged_events = _GLOBAL_LOGGED_EVENTS

        # Enhanced features
        self._file_metadata_cache: dict[str, dict[str, Any]] = {}  # Cache for file metadata
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup cache every hour

    def _check_files_exist(self, file_paths: dict[str, str]) -> tuple[bool, list[str]]:
        """Check if all required files exist."""
        missing_files: list[str] = []
        all_exist = True

        for file_type, file_path in file_paths.items():
            if not os.path.exists(file_path):
                all_exist = False
                missing_files.append(f"{file_type} ({file_path})")

        return all_exist, missing_files

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM block states loading",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM composite cluster loading",
    )
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM meta loading",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM intensity loading",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM basic meta loading",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="HMM composite cluster creation",
    )
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM composite cluster management",
    )
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


# Global instance for easy access
_hmm_composite_manager: HMMCompositeManager | None = None

