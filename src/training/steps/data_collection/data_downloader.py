from ...core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Data downloader adapter for training steps.

This module provides a unified interface for downloading data
using the centralized unified downloader.
"""

from typing import Any

from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors as utils_handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation

# Import the unified downloader
from .unified_data_downloader import UnifiedDataDownloader

@handles_errors(fallback = False)
async def download_all_data_with_consolidation(
    symbol: str,
    exchange_name: str,
    interval: str = "1m",
    data_dir: str = None,
) -> bool:
    """Unified entrypoint used by training steps to download raw data.

    Uses the centralized unified downloader for all data types.
    Returns True on success, False otherwise.
    """
    logger = system_logger.getChild("DataDownloaderAdapter")

    lookback_years: int = 2
    try:
        if isinstance(CONFIG, dict):
            model_training_cfg: dict[str, Any] | None = CONFIG.get("MODEL_TRAINING")  # type: ignore[assignment]
            if model_training_cfg and isinstance(
                model_training_cfg.get("lookback_years"),
                int,
            ):
                lookback_years = int(model_training_cfg["lookback_years"])  # type: ignore[arg-type]
    except Exception:
        # Keep default lookback_years
        pass

    # Use unified downloader
    try:
        from datetime import datetime, timedelta
        
        downloader = UnifiedDataDownloader(data_dir or "data_cache")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        
        logger.info(f"📥 Downloading data for {symbol} on {exchange_name}")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        # Download all data types
        success_count = 0
        total_types = 3
        
        # Download klines
        klines_success, klines_data, klines_error = await downloader.download_klines(
            symbol, exchange_name, interval, start_date, end_date
        )
        if klines_success:
            success_count += 1
            logger.info(f"✅ Downloaded {len(klines_data)} klines records")
        else:
            logger.error(f"❌ Klines download failed: {klines_error}")
        
        # Download aggtrades
        aggtrades_success, aggtrades_data, aggtrades_error = await downloader.download_aggtrades(
            symbol, exchange_name, start_date, end_date
        )
        if aggtrades_success:
            success_count += 1
            logger.info(f"✅ Downloaded {len(aggtrades_data)} aggtrades records")
        else:
            logger.error(f"❌ Aggtrades download failed: {aggtrades_error}")
        
        # Download futures
        futures_success, futures_data, futures_error = await downloader.download_futures(
            symbol, exchange_name, start_date, end_date
        )
        if futures_success:
            success_count += 1
            logger.info(f"✅ Downloaded {len(futures_data)} futures records")
        else:
            logger.error(f"❌ Futures download failed: {futures_error}")
        
        # Return success if at least one data type was downloaded successfully
        overall_success = success_count > 0
        logger.info(f"📊 Download summary: {success_count}/{total_types} data types successful")
        
        return overall_success
        
    except Exception as e:
        logger.exception(f"Unified downloader failed: {e}")
        return False
