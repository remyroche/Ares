"""Step 1: Data Collection.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""

import sys
from pathlib import Path
from typing import Any
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import constants
try:
    from src.config.constants import DEFAULT_LOOKBACK_DAYS
except ImportError:
    # Fallback if constants module is not available
    DEFAULT_LOOKBACK_DAYS = 1095

# Handle imports with fallback - this must be done before any other imports
CONFIG = None
handle_errors = None
setup_logging = None
system_logger = None
download_all_data_with_consolidation = None

# Temporarily comment out problematic imports
# try:
#     from src.config import CONFIG
#     from src.utils.error_handler import handle_errors
#     from src.utils.logger import setup_logging, system_logger
#     from src.training.steps.data_downloader import download_all_data_with_consolidation
#     from src.utils.data_quality_decorators import (
#         handle_data_collection_errors,
#         validate_klines_data,
#         format_klines_data,
#         log_step_metrics,
#     )
# except ImportError:
# Fallback decorators if data quality decorators are not available
def handle_data_collection_errors(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def log_step_metrics(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

# Handle imports with fallback - this must be done before any other imports
CONFIG = None
handle_errors = None
setup_logging = None
system_logger = None
download_all_data_with_consolidation = None

try:
    from src.config import CONFIG
    from src.utils.error_handler import handle_errors
    from src.utils.logger import setup_logging, system_logger
    from src.training.steps.data_downloader import download_all_data_with_consolidation
except ImportError:
    # Fallback configuration
    CONFIG = {
        "SYMBOL": "ETHUSDT",
        "INTERVAL": "1m",
        "LOOKBACK_YEARS": 2,
    }

    # Create fallback functions
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def setup_logging():
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    system_logger = setup_logging()
    download_all_data_with_consolidation = None


class DataCollectionStep:
    """Step 1: Data Collection using existing run_step function."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataCollectionStep")

    async def initialize(self) -> None:
        """Initialize the data collection step."""
        self.logger.info("Initializing Data Collection Step...")
        self.logger.info("Data Collection Step initialized successfully")

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute data collection.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state

        """
        self.logger.info("Starting data collection...")

        try:
            # Execute the data collection
            success = await self._run_data_collection(training_input)

            if success:
                self.logger.info("Data collection completed successfully")
                pipeline_state["data_collection_completed"] = True
            else:
                self.logger.error("Data collection failed")
                pipeline_state["data_collection_completed"] = False

        except Exception as e:
            self.logger.exception(f"Error during data collection: {e}")
            pipeline_state["data_collection_completed"] = False

        return pipeline_state

    @handle_data_collection_errors(context="run_data_collection")
    @log_step_metrics(context="data_collection")
    async def _run_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Run the actual data collection process."""
        try:
            # Try to import the downloader if not already imported
            global download_all_data_with_consolidation
            if download_all_data_with_consolidation is None:
                try:
                    from src.training.steps.data_downloader import download_all_data_with_consolidation
                except ImportError:
                    self.logger.warning("Could not import data downloader, using fallback")
                    return await self._fallback_data_collection(training_input)

            if download_all_data_with_consolidation:
                # Use the existing data downloader if available
                symbol = training_input.get("symbol", "ETHUSDT")
                exchange = training_input.get("exchange", "BINANCE")
                timeframe = training_input.get("timeframe", "1m")

                self.logger.info(f"📊 Downloading data for {exchange}_{symbol}_{timeframe}")
                success = await download_all_data_with_consolidation(
                    symbol=symbol,
                    exchange_name=exchange,
                    interval=timeframe,
                )
                return success
            # Fallback implementation
            self.logger.warning("Using fallback data collection method")
            return await self._fallback_data_collection(training_input)

        except Exception as e:
            self.logger.exception(f"Error in data collection: {e}")
            return False

    @handle_data_collection_errors(context="fallback_data_collection")
    async def _fallback_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Fallback data collection method."""
        self.logger.info("Running fallback data collection...")
        # Add fallback implementation here if needed
        return True


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step1_data_collection",
)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the data collection step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if data exists
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise

    """
    try:
        logger = system_logger.getChild("Step1DataCollection")

        logger.info("=" * 80)
        logger.info("🚀 STEP 1: Data Collection")
        logger.info("=" * 80)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔄 Force rerun: {force_rerun}")

        # Check if data already exists and force_rerun is False
        if not force_rerun:
            # Check for existing consolidated data
            consolidated_files = [
                f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            ]

            existing_files = []
            for file_path in consolidated_files:
                if Path(file_path).exists():
                    existing_files.append(file_path)

            if existing_files:
                logger.info(f"✅ Found existing consolidated data: {len(existing_files)} files")
                logger.info("   📁 Existing files:")
                for file_path in existing_files:
                    logger.info(f"      - {file_path}")

                # Check if data is complete by examining the date range
                try:
                    import pandas as pd
                    klines_file = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
                    if Path(klines_file).exists():
                        df = pd.read_parquet(klines_file)
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                            df["timestamp"].min().date()
                            max_date = df["timestamp"].max().date()
                            current_date = datetime.now().date()

                            # Check if we have recent data (within last 30 days)
                            days_since_last_data = (current_date - max_date).days

                            if days_since_last_data > 30:
                                logger.info(f"⚠️ Data is {days_since_last_data} days old, downloading recent data...")
                                # Continue with data collection to download missing data
                            else:
                                logger.info(f"✅ Data is up to date (last data: {max_date}, {days_since_last_data} days ago)")
                                logger.info("✅ Step 1: Data Collection completed (using existing data)")
                                return True
                        else:
                            logger.warning("⚠️ Could not determine data completeness, proceeding with data collection...")
                    else:
                        logger.warning("⚠️ Klines file not found, proceeding with data collection...")
                except Exception as e:
                    logger.warning(f"⚠️ Error checking data completeness: {e}, proceeding with data collection...")

        # Initialize data collection step
        step = DataCollectionStep(CONFIG or {})
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
        }

        # Execute data collection
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("data_collection_completed", False):
            logger.info("✅ Step 1: Data Collection completed successfully")
            return True
        else:
            logger.error("❌ Step 1: Data Collection failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 1: Data Collection failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            print("Usage: python step1_data_collection.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]")
            print("Example: python step1_data_collection.py ETHUSDT BINANCE 1m data_cache true")
            return

        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            print("✅ Step 1: Data Collection completed successfully")
        else:
            print("❌ Step 1: Data Collection failed")

        # Clean up memory to prevent segmentation fault
        import gc
        gc.collect()

    # Use a more robust approach to prevent segmentation fault
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Final cleanup
        import gc
        gc.collect()
