"""
Data downloader adapter for training steps.

This module provides a unified interface for downloading data
using the centralized unified downloader with comprehensive type safety
and error handling.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from src.config import CONFIG
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning
from src.utils.error_handler import handles_errors as utils_handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation

# Import our custom types
from ..types import (
    StepConfig, ExecutionResult, DataLoadResult, ValidationResult,
    DataLoadError, ConfigurationError, validate_config, create_error_result, create_success_result
)

# Import the unified downloader
from .unified_data_downloader import UnifiedDataDownloader

@utils_handles_errors(fallback=False)
async def download_all_data_with_consolidation(
    symbol: str,
    exchange_name: str,
    interval: str = "1m",
    data_dir: Optional[str] = None,
) -> DataLoadResult:
    """
    Unified entrypoint used by training steps to download raw data.

    Uses the centralized unified downloader for all data types with comprehensive
    error handling and type safety.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange_name: Exchange name (e.g., 'binance')
        interval: Data interval (e.g., '1m', '15m')
        data_dir: Optional data directory path

    Returns:
        DataLoadResult containing success status, data, and error information

    Raises:
        DataLoadError: If data loading fails
        ConfigurationError: If configuration is invalid
    """
    try:
        # Validate input parameters
        if not symbol or not isinstance(symbol, str):
            raise DataLoadError("Symbol must be a non-empty string")
        if not exchange_name or not isinstance(exchange_name, str):
            raise DataLoadError("Exchange name must be a non-empty string")
        if not interval or not isinstance(interval, str):
            raise DataLoadError("Interval must be a non-empty string")

        tprint_info(f"📥 Starting data download for {symbol} on {exchange_name}")

        # Get lookback years from config
        lookback_years: int = 2
        try:
            if isinstance(CONFIG, dict):
                model_training_cfg: Optional[Dict[str, Any]] = CONFIG.get("MODEL_TRAINING")
                if model_training_cfg and isinstance(model_training_cfg.get("lookback_years"), int):
                    lookback_years = int(model_training_cfg["lookback_years"])
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get lookback_years from config: {e}, using default: {lookback_years}")

        # Initialize downloader
        try:
            downloader = UnifiedDataDownloader(data_dir or "data_cache")
            tprint_success("✅ UnifiedDataDownloader initialized")
        except Exception as e:
            raise DataLoadError(f"Failed to initialize UnifiedDataDownloader: {e}") from e

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        tprint_info(f"📅 Date range: {start_date} to {end_date}")
        tprint_info(f"📊 Lookback period: {lookback_years} years")

        # Download klines data
        klines_success = False
        klines_data = None
        klines_error = None

        try:
            tprint_info(f"📈 Downloading klines data for {symbol}...")
            klines_success, klines_data, klines_error = await downloader.download_klines(
                symbol, exchange_name, interval, start_date, end_date
            )
            
            if klines_success and klines_data is not None:
                tprint_success(f"✅ Downloaded {len(klines_data)} klines records")
            else:
                tprint_error(f"❌ Klines download failed: {klines_error}")
                
        except Exception as e:
            klines_error = f"Klines download exception: {e}"
            tprint_error(f"❌ Klines download exception: {e}")

        # Skip aggtrades download as per new setup
        tprint_info(f"⚠️ Skipping aggtrades download for {symbol} - aggtrades downloads disabled")

        # Determine overall success
        if klines_success and klines_data is not None:
            tprint_success(f"📊 Data download completed successfully: {len(klines_data)} records")
            return create_success_result(
                data=klines_data,
                source="unified_downloader",
                rows=len(klines_data),
                columns=len(klines_data.columns) if hasattr(klines_data, 'columns') else 0
            )
        else:
            error_msg = f"Data download failed: {klines_error or 'Unknown error'}"
            tprint_error(f"❌ {error_msg}")
            return create_error_result(DataLoadError(error_msg), "download_all_data_with_consolidation")

    except DataLoadError:
        raise
    except Exception as e:
        tprint_error(f"❌ Unexpected error in data download: {e}")
        raise DataLoadError(f"Data download failed: {e}") from e
