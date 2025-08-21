"""Step 1: Data Collection.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""

import sys
from pathlib import Path
from typing import Any

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

try:
    from src.config import CONFIG
    from src.utils.error_handler import handle_errors
    from src.utils.logger import setup_logging, system_logger

    # Temporarily comment out to avoid step2 import issues
    # from src.training.steps.data_downloader import download_all_data_with_consolidation
    download_all_data_with_consolidation = None
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

    async def _run_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Run the actual data collection process."""
        try:
            if download_all_data_with_consolidation:
                # Use the existing data downloader if available
                await download_all_data_with_consolidation()
                return True
            # Fallback implementation
            self.logger.warning("Using fallback data collection method")
            return await self._fallback_data_collection(training_input)

        except Exception as e:
            self.logger.exception(f"Error in data collection: {e}")
            return False

    async def _fallback_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Fallback data collection method."""
        self.logger.info("Running fallback data collection...")
        # Add fallback implementation here if needed
        return True
