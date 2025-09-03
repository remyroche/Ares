"""Refactored Data Converter Step
This is a refactored version of step01_5_data_converter.py that uses the extracted components.
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
import os.path
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil

# Import the extracted components
from src.training.steps.data_preparation_components import (
    DataFormatConverter,
    DataValidator,
    DataCleaner
)

# Ensure project root is on path
from src.core.decorators import traced, validates

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import pipeline_standards

# Import other utilities
from src.utils.logger import system_logger
from src.training.steps.data_downloader import download_all_data_with_consolidation
from src.utils.file_operations import ensure_directory, safe_json_dump, safe_json_load


# ----------------------------------------------------------------------------
# Utilities: Timing and Memory trackers (keeping these as they're specific utilities)
# ----------------------------------------------------------------------------
class TimingTracker:
    def __init__(self) -> None:
        self.start_time = None
        self.checkpoints = {}
        self.phase_times = {}

    def start(self, phase_name: str) -> None:
        self.start_time = time.time()
        self.phase_times[phase_name] = {"start": self.start_time, "end": None}

    def checkpoint(self, checkpoint_name: str) -> None:
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        self.checkpoints[checkpoint_name] = elapsed
        print(f"⏱️ {checkpoint_name}: {elapsed:.2f}s")

    def end_phase(self, phase_name: str) -> None:
        if phase_name in self.phase_times:
            self.phase_times[phase_name]["end"] = time.time()
            duration = (
                self.phase_times[phase_name]["end"]
                - self.phase_times[phase_name]["start"]
            )
            print(f"⏱️ Phase '{phase_name}' completed in {duration:.2f}s")

    def get_total_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def print_summary(self) -> None:
        print("\n📊 Timing Summary:")
        for name, elapsed in self.checkpoints.items():
            print(f"  - {name}: {elapsed:.2f}s")
        print(f"  - Total: {self.get_total_time():.2f}s")


class MemoryTracker:
    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @staticmethod
    def log_memory_usage(context: str = "") -> None:
        usage = MemoryTracker.get_memory_usage()
        print(
            f"💾 Memory Usage {context}: RSS={usage['rss_mb']:.1f}MB, "
            f"VMS={usage['vms_mb']:.1f}MB, {usage['percent']:.1f}%"
        )


# ----------------------------------------------------------------------------
# UnifiedDataConverter - Main class using the extracted components
# ----------------------------------------------------------------------------
class UnifiedDataConverter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("UnifiedDataConverter")
        self.standards = pipeline_standards
        
        # Initialize extracted components
        self.format_converter = DataFormatConverter(logger=self.logger)
        self.data_validator = DataValidator(logger=self.logger)
        self.data_cleaner = DataCleaner(logger=self.logger)
        
        # Initialize with default data_cache, will be updated in execute method
        self.data_cache_dir = "data_cache"
        self.unified_dir = os.path.join(self.data_cache_dir, "unified")
        self.backup_dir = os.path.join(self.data_cache_dir, "backup_pre_unified")
        ensure_directory(self.unified_dir)
        ensure_directory(self.backup_dir)

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")
        # Environment validation logic here

    async def initialize(self) -> None:
        self.logger.info("🚀 Initializing Unified Data Converter...")
        self.logger.info(f"📁 Unified data directory: {self.unified_dir}")
        self.logger.info(f"📁 Backup directory: {self.backup_dir}")

    async def execute(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        data_dir: str = None,
        force_rerun: bool = False,
    ) -> bool:
        """Execute the data conversion process using extracted components."""
        try:
            # Use standardized path construction
            self.data_cache_dir = self.standards.build_path(
                "raw_data", exchange, symbol
            )
            self.unified_dir = self.standards.build_path(
                "unified_data", exchange, symbol
            )
            self.backup_dir = self.standards.build_path("backup", exchange, symbol)

            timing = TimingTracker()
            timing.start("conversion")

            self.logger.info(
                f"🔄 Starting unified data conversion for {symbol} on {exchange}"
            )

            # Check if already converted
            if not force_rerun and await self._check_already_converted(
                symbol, exchange, timeframe
            ):
                self.logger.info(
                    "✅ Data already converted and up-to-date. Use force_rerun=True to reconvert."
                )
                return True

            # Load klines data
            klines_data = await self._load_klines_data(symbol, exchange, timeframe)
            if klines_data is None or klines_data.empty:
                self.logger.error("❌ No klines data found to convert")
                return False

            timing.checkpoint("klines_loaded")

            # Clean the data using DataCleaner
            self.logger.info("🧹 Cleaning data...")
            klines_data = self.data_cleaner.remove_duplicates(
                klines_data, subset=["timestamp"]
            )
            klines_data = self.data_cleaner.fill_missing_values(
                klines_data, method="auto"
            )
            
            # Validate data using DataValidator
            self.logger.info("✅ Validating data...")
            missing_info = self.data_validator.verify_missing_columns(
                klines_data, data_type="klines"
            )
            
            if not missing_info["verification_passed"]:
                self.logger.error(
                    f"❌ Data validation failed: {missing_info['missing_required']}"
                )
                return False
                
            # Calculate missing columns if needed
            if any(missing_info["can_calculate"].values()):
                klines_data = self.data_validator.calculate_missing_columns(
                    klines_data, missing_info
                )

            # Process and save the data
            success = await self._process_and_save_unified_data(
                klines_data, symbol, exchange, timeframe
            )

            timing.end_phase("conversion")
            timing.print_summary()

            return success

        except Exception as e:
            self.logger.exception(f"❌ Unified data conversion failed: {e}")
            return False

    async def _check_already_converted(
        self, symbol: str, exchange: str, timeframe: str
    ) -> bool:
        """Check if data is already converted and up-to-date."""
        try:
            base_dir = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe
            )
            
            if not os.path.exists(base_dir):
                return False
                
            # Check latest timestamp using DataFormatConverter
            latest_unified = self.format_converter.get_latest_timestamp(base_dir)
            if latest_unified is None:
                return False
                
            # Compare with source data
            klines_dir = os.path.join(
                self.data_cache_dir, "parquet", f"klines_{exchange}_{symbol}_{timeframe}"
            )
            latest_klines = self.format_converter.get_latest_timestamp(klines_dir)
            
            if latest_klines is None or latest_unified < latest_klines:
                return False
                
            return True
            
        except Exception:
            return False

    async def _load_klines_data(
        self, symbol: str, exchange: str, timeframe: str
    ) -> pd.DataFrame | None:
        """Load klines data using DataFormatConverter."""
        try:
            parquet_dir = os.path.join(
                self.data_cache_dir, "parquet", f"klines_{exchange}_{symbol}_{timeframe}"
            )
            
            if not os.path.exists(parquet_dir):
                self.logger.warning(f"⚠️ Parquet directory not found: {parquet_dir}")
                return None
                
            # Use DataFormatConverter to scan the dataset
            self.logger.info(f"📂 Loading klines from: {parquet_dir}")
            klines_data = self.format_converter.scan_dataset(
                parquet_dir,
                columns=None,
                to_pandas=True
            )
            
            if klines_data is None or klines_data.empty:
                self.logger.warning("⚠️ No klines data found in parquet files")
                return None
                
            # Enforce schema
            klines_data = self.format_converter.enforce_schema(klines_data, "klines")
            
            self.logger.info(
                f"✅ Loaded {len(klines_data):,} klines rows, "
                f"columns: {list(klines_data.columns)}"
            )
            
            return klines_data
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to load klines data: {e}")
            return None

    async def _process_and_save_unified_data(
        self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str
    ) -> bool:
        """Process and save unified data using DataFormatConverter."""
        try:
            # Add metadata columns
            data["exchange"] = exchange.upper()
            data["symbol"] = symbol
            data["timeframe"] = timeframe
            
            # Ensure timestamp is properly formatted
            if "timestamp" in data.columns:
                data = data.sort_values("timestamp").reset_index(drop=True)
                
            base_dir = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe
            )
            ensure_directory(base_dir)
            
            # Write partitioned dataset using DataFormatConverter
            self.logger.info(f"💾 Writing unified data to: {base_dir}")
            
            self.format_converter.write_partitioned_dataset(
                data,
                base_dir,
                partition_cols=["year", "month"],
                schema_name="unified",
                compression="snappy",
                update_manifest=True,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "conversion_time": datetime.now(UTC).isoformat(),
                }
            )
            
            # Write configuration
            config_path = self.get_unified_config_path(symbol, exchange, timeframe)
            config_data = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "conversion_time": datetime.now(UTC).isoformat(),
                "source_columns": list(data.columns),
                "row_count": len(data),
                "date_range": {
                    "start": str(data.index.min()) if isinstance(data.index, pd.DatetimeIndex) else None,
                    "end": str(data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else None,
                }
            }
            safe_json_dump(config_data, config_path, indent=2)
            
            self.logger.info(
                f"✅ Successfully saved unified data: {len(data):,} rows"
            )
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to save unified data: {e}")
            return False

    def get_unified_data_path(self, symbol: str, exchange: str, timeframe: str) -> str:
        return os.path.join(self.unified_dir, exchange.lower(), symbol, timeframe)

    def get_unified_config_path(
        self, symbol: str, exchange: str, timeframe: str
    ) -> str:
        base_path = self.get_unified_data_path(symbol, exchange, timeframe)
        return os.path.join(base_path, "unified_config.json")


# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------
async def main():
    """Example usage of the refactored UnifiedDataConverter."""
    config = {
        "symbols": ["BTCUSDT"],
        "exchanges": ["BYBIT"],
        "timeframes": ["1m"],
    }
    
    converter = UnifiedDataConverter(config)
    await converter.initialize()
    
    for symbol in config["symbols"]:
        for exchange in config["exchanges"]:
            for timeframe in config["timeframes"]:
                success = await converter.execute(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=False
                )
                
                if success:
                    print(f"✅ Successfully converted {symbol} {exchange} {timeframe}")
                else:
                    print(f"❌ Failed to convert {symbol} {exchange} {timeframe}")


if __name__ == "__main__":
    asyncio.run(main())