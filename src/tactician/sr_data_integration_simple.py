# src/tactician/sr_data_integration_simple.py

"""
Simplified S/R Data Integration Module

This module provides S/R backtesting validation with proper data access patterns
without depending on problematic training modules. It ensures the S/R system uses
consistent lookback periods and data access patterns.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Simplified imports without problematic training modules
try:
    from src.utils.logger import system_logger
except ImportError:
    # Fallback logging
    import logging
    system_logger = logging.getLogger(__name__)

# Default constants
DEFAULT_LOOKBACK_DAYS = 730  # 2 years default
TRAINING_MODES = {
    "light": {"lookback_days": 30, "name": "light", "description": "Light training mode"},
    "blank": {"lookback_days": 180, "name": "blank", "description": "Blank training mode"},
    "full": {"lookback_days": 730, "name": "full", "description": "Full training mode"},
}


class SRDataIntegrationSimple:
    """
    Simplified S/R data integration that doesn't depend on training modules.

    This class ensures that:
    1. S/R validation uses consistent data access patterns
    2. Lookback periods are managed properly
    3. Data loading follows simple, reliable patterns
    4. Timeframe-specific data is properly handled
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simplified S/R data integration system.

        Args:
            config: Configuration dictionary with data access parameters
        """
        self.config = config or {}
        self.logger = system_logger.getChild("SRDataIntegrationSimple") if system_logger else None

        # Data access configuration
        self.data_config = self.config.get("data_integration", {})
        self.symbol = self.data_config.get("symbol", "BTCUSDT")
        self.exchange = self.data_config.get("exchange", "binance")
        self.timeframes = self.data_config.get("timeframes", ["1m", "5m", "15m", "30m"])

        # Lookback period configuration
        self.lookback_days = self.data_config.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
        self.training_mode = self.data_config.get("training_mode", "blank")

        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._last_load_time: Dict[str, datetime] = {}

        # Data validation settings
        self.min_data_points = self.data_config.get("min_data_points", 1000)
        self.max_data_age_hours = self.data_config.get("max_data_age_hours", 24)

    async def _validate_configuration(self) -> bool:
        """Validate the configuration parameters.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate symbol
            if not self.symbol or not isinstance(self.symbol, str):
                if self.logger:
                    self.logger.error("❌ Invalid symbol configuration")
                return False

            # Validate exchange
            if not self.exchange or not isinstance(self.exchange, str):
                if self.logger:
                    self.logger.error("❌ Invalid exchange configuration")
                return False

            # Validate timeframes
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            for tf in self.timeframes:
                if tf not in valid_timeframes:
                    if self.logger:
                        self.logger.error(f"❌ Invalid timeframe: {tf}")
                    return False

            # Validate lookback period
            if self.lookback_days <= 0 or self.lookback_days > 1095:  # Max 3 years
                if self.logger:
                    self.logger.error(f"❌ Invalid lookback days: {self.lookback_days}")
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    async def _ensure_data_availability(self) -> bool:
        """Ensure that required data is available for all timeframes.

        Returns:
            True if data is available, False otherwise
        """
        try:
            if self.logger:
                self.logger.info("📊 Checking data availability...")

            # For simplified version, we'll assume data is available
            # In a real implementation, this would check actual data files
            if self.logger:
                self.logger.info("✅ Data availability check completed (simplified)")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Data availability check failed: {e}")
            return False

    async def _load_timeframe_data(self, timeframe: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Load data for a specific timeframe.

        Args:
            timeframe: The timeframe to load
            lookback_days: Number of days to look back

        Returns:
            DataFrame with market data or None if failed
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Try to load from file system
            data = await self._load_from_file_system(timeframe, start_date, end_date)

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to load timeframe data: {e}")
            return None

    async def _load_from_file_system(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data directly from file system.

        Args:
            timeframe: The timeframe to load
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with market data or None if failed
        """
        try:
            # Construct file path
            data_dir = Path("data") / self.exchange / self.symbol / timeframe
            if not data_dir.exists():
                if self.logger:
                    self.logger.debug(f"Data directory not found: {data_dir}")
                return None

            # Find the most recent data file
            data_files = list(data_dir.glob("*.parquet"))
            if not data_files:
                if self.logger:
                    self.logger.debug(f"No data files found in {data_dir}")
                return None

            # Load the most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)

            # Load data
            data = pd.read_parquet(latest_file)

            # Filter by date range
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data = data[
                    (data['timestamp'] >= start_date) &
                    (data['timestamp'] <= end_date)
                ]

            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                if self.logger:
                    self.logger.warning(f"Missing required columns in {latest_file}")
                return None

            return data.sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            if self.logger:
                self.logger.debug(f"File system loading failed for {timeframe}: {e}")
            return None

    async def validate_data_quality(self, data: pd.DataFrame, timeframe: str) -> bool:
        """Validate the quality of loaded data.

        Args:
            data: The data to validate
            timeframe: The timeframe the data represents

        Returns:
            True if data quality is acceptable, False otherwise
        """
        try:
            if data is None or len(data) == 0:
                if self.logger:
                    self.logger.error(f"❌ No data provided for validation")
                return False

            # Check minimum data points
            min_points = self._get_min_data_points_for_timeframe(timeframe)
            if len(data) < min_points:
                if self.logger:
                    self.logger.error(f"❌ Insufficient data points: {len(data)} < {min_points}")
                return False

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                if self.logger:
                    self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return False

            # Check for data gaps
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp')
                time_diffs = data_sorted['timestamp'].diff().dropna()

                # Calculate expected time difference based on timeframe
                expected_diff = self._get_expected_time_diff(timeframe)
                max_gap_multiplier = 5  # Allow gaps up to 5x expected interval

                large_gaps = time_diffs > (expected_diff * max_gap_multiplier)
                if large_gaps.sum() > len(data) * 0.1:  # More than 10% gaps
                    if self.logger:
                        self.logger.warning(f"⚠️ Large data gaps detected in {timeframe}")

            # Check for price anomalies
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if data[col].isnull().sum() > len(data) * 0.05:  # More than 5% nulls
                    if self.logger:
                        self.logger.warning(f"⚠️ High null count in {col}: {timeframe}")

            if self.logger:
                self.logger.info(f"✅ Data quality validation passed for {timeframe}")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Data quality validation failed: {e}")
            return False


# Convenience function for easy integration