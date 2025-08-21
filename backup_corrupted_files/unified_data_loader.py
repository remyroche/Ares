# src/training/steps/unified_data_loader.py

"""Unified Data Loader for Training Pipeline.

This module provides a unified interface for loading data across all training steps.
It uses step1 and step1_5 functions for data downloading and resampling to ensure
consistency and avoid duplication of data loading logic.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CONFIG
from src.training.steps.step1.aggtrades_validator import AggtradesValidator
from src.training.steps.step1.data_resampler import DataPreparation
from src.training.steps.step1.missing_data_downloader_and_gap_filler import (
    MissingDataDownloaderAndGapFiller,
)
from src.training.steps.step1.step1_orchestrator import Step1Orchestrator
from src.training.steps.step1_5_data_converter import UnifiedDataConverter
from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors,
    secure_data_processing,
    validate_data_quality,
    with_tracing_span,
)
from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("UnifiedDataLoader")


class UnifiedDataLoader:
    """Unified data loader that uses step1 and step1_5 functions for consistent data access.

    This class provides a single interface for all training steps to access:
    - Raw data (aggtrades, klines, futures)
    - Resampled data (multiple timeframes)
    - Processed data (step1_5 converted data)
    - Missing data downloading and gap filling
    """

    def __init__(
        self,
        data_cache_path: str = "data_cache",
        data_dir: str = "data/training",
    ) -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_dir = Path(data_dir)

        # Initialize step1 components
        self.step1_orchestrator = Step1Orchestrator(data_cache_path)
        self.data_preparation = DataPreparation(data_cache_path)
        self.data_downloader = MissingDataDownloaderAndGapFiller(data_cache_path)
        self.aggtrades_validator = AggtradesValidator(data_cache_path)

        # Initialize step1_5 converter
        self.step1_5_converter = UnifiedDataConverter(CONFIG)

        # Ensure directories exist
        self.data_cache_path.mkdir(exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @secure_data_processing
    @with_tracing_span("ensure_data_availability")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            FileNotFoundError,
            PermissionError,
        ),
        default_return=False,
        context="unified_data_loader.ensure_data_availability",
    )
    async def ensure_data_availability(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        auto_download: bool = True,
        auto_fix: bool = True,
    ) -> bool:
        """Ensure all required data is available for the given symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for data (default: 2 years ago)
            end_date: End date for data (default: 2 days ago)
            auto_download: Whether to automatically download missing data
            auto_fix: Whether to automatically fix data issues

        Returns:
            True if data is available, False otherwise

        """
        logger.info(f"🔍 Ensuring data availability for {exchange}_{symbol}")

        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 2)
        if end_date is None:
            end_date = datetime.now() - timedelta(days=2)

        # Step 1: Check current data status
        status = self.step1_orchestrator.get_step1_status(symbol, exchange)
        logger.info(f"📊 Current data status: {status['overall_status']}")

        if status["overall_status"] == "complete":
            logger.info("✅ All required data is already available")
            return True

        # Step 2: Run comprehensive step1 process if needed
        if status["overall_status"] in ["partial", "missing"]:
            logger.info("📥 Running comprehensive step1 data collection")

            step1_results = self.step1_orchestrator.run_complete_step1(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                auto_fix=auto_fix,
            )

            if not step1_results["success"]:
                logger.error("❌ Step1 data collection failed")
                return False

            # Download missing data if requested
            if auto_download:
                logger.info("📥 Downloading missing data")
                download_results = (
                    await self.data_downloader.download_missing_data_comprehensive(
                        symbol=symbol,
                        exchange=exchange,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )

                if not download_results["success"]:
                    logger.error("❌ Missing data download failed")
                    return False

        logger.info("✅ Data availability check completed")
        return True

    @validate_data_quality
    @guard_dataframe_nulls
    @handle_errors(
        exceptions=(FileNotFoundError, ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="unified_data_loader.load_klines_data",
    )
    def load_klines_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load klines data using step1 functions.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with klines data

        """
        logger.info(f"📊 Loading klines data for {exchange}_{symbol}_{timeframe}")

        # Use step1 data preparation for loading
        data = self.data_preparation.load_klines_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"✅ Loaded {len(data)} klines records")
        return data

    @validate_data_quality
    @guard_dataframe_nulls
    @handle_errors(
        exceptions=(FileNotFoundError, ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="unified_data_loader.load_aggtrades_data",
    )
    def load_aggtrades_data(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load aggtrades data using step1 functions.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with aggtrades data

        """
        logger.info(f"📊 Loading aggtrades data for {exchange}_{symbol}")

        # Use step1 data preparation for loading
        data = self.data_preparation.load_aggtrades_data(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"✅ Loaded {len(data)} aggtrades records")
        return data

    @validate_data_quality
    @guard_dataframe_nulls
    @handle_errors(
        exceptions=(FileNotFoundError, ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="unified_data_loader.load_step1_5_data",
    )
    def load_step1_5_data(
        self,
        symbol: str,
        exchange: str,
        data_type: str = "klines",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load step1_5 converted data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data (klines, aggtrades, futures)
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with step1_5 data

        """
        logger.info(f"📊 Loading step1_5 data for {exchange}_{symbol}_{data_type}")

        # Use step1_5 converter for loading
        data = self.step1_5_converter.load_converted_data(
            symbol=symbol,
            exchange=exchange,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"✅ Loaded {len(data)} step1_5 records")
        return data

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="unified_data_loader.get_data_status",
    )
    def get_data_status(
        self,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Get comprehensive data status for a symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with data status information

        """
        logger.info(f"📊 Getting data status for {exchange}_{symbol}")

        # Get step1 status
        step1_status = self.step1_orchestrator.get_step1_status(symbol, exchange)

        # Get step1_5 status
        step1_5_status = self.step1_5_converter.get_conversion_status(symbol, exchange)

        # Combine status information
        status = {
            "symbol": symbol,
            "exchange": exchange,
            "step1_status": step1_status,
            "step1_5_status": step1_5_status,
            "overall_status": "unknown",
        }

        # Determine overall status
        if (
            step1_status["overall_status"] == "complete"
            and step1_5_status["overall_status"] == "complete"
        ):
            status["overall_status"] = "complete"
        elif step1_status["overall_status"] in ["partial", "missing"] or step1_5_status[
            "overall_status"
        ] in ["partial", "missing"]:
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "missing"

        logger.info(f"📊 Data status: {status['overall_status']}")
        return status

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified_data_loader.download_missing_data",
    )
    async def download_missing_data(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> bool:
        """Download missing data using step1 functions.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for data
            end_date: End date for data

        Returns:
            True if download successful, False otherwise

        """
        logger.info(f"📥 Downloading missing data for {exchange}_{symbol}")

        # Use step1 data downloader
        results = await self.data_downloader.download_missing_data_comprehensive(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )

        if results["success"]:
            logger.info("✅ Missing data download completed successfully")
        else:
            logger.error("❌ Missing data download failed")

        return results["success"]

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified_data_loader.resample_data",
    )
    def resample_data(
        self,
        symbol: str,
        exchange: str,
        timeframes: list[str] | None = None,
    ) -> bool:
        """Resample data to multiple timeframes using step1 functions.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to resample to

        Returns:
            True if resampling successful, False otherwise

        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        logger.info(f"📊 Resampling data for {exchange}_{symbol} to {timeframes}")

        # Use step1 data preparation for resampling
        results = self.data_preparation.resample_all_timeframes(
            symbol=symbol,
            exchange=exchange,
            timeframes=timeframes,
        )

        if results["success"]:
            logger.info("✅ Data resampling completed successfully")
        else:
            logger.error("❌ Data resampling failed")

        return results["success"]


def get_unified_data_loader(
    data_cache_path: str = "data_cache",
    data_dir: str = "data/training",
) -> UnifiedDataLoader:
    """Get a UnifiedDataLoader instance.

    Args:
        data_cache_path: Path to data cache directory
        data_dir: Path to training data directory

    Returns:
        UnifiedDataLoader instance

    """
    return UnifiedDataLoader(data_cache_path=data_cache_path, data_dir=data_dir)
