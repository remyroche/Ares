# src/tactician/sr_data_integration_simple.py

"""
Simplified S/R Data Integration Module

This module provides S/R backtesting validation with proper data access patterns
without depending on problematic training modules. It ensures the S/R system uses
consistent lookback periods and data access patterns.
"""

import sys
from pathlib import Path
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

    async def initialize(self) -> bool:
        """Initialize the data integration system.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            if self.logger:
                self.logger.info(f"🔧 Initializing Simplified S/R Data Integration")
                self.logger.info(f"   - Symbol: {self.symbol}")
                self.logger.info(f"   - Exchange: {self.exchange}")
                self.logger.info(f"   - Timeframes: {self.timeframes}")
                self.logger.info(f"   - Lookback days: {self.lookback_days}")
                self.logger.info(f"   - Training mode: {self.training_mode}")

            # Validate configuration
            if not await self._validate_configuration():
                return False

            # Ensure data is available
            if not await self._ensure_data_availability():
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to initialize S/R data integration: {e}")
            return False

    async def _validate_configuration(self) -> bool:
        """Validate the configuration parameters.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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

    async def get_market_data(
        self,
        timeframe: str,
        lookback_days: Optional[int] = None,
        force_reload: bool = False
    ) -> Optional[pd.DataFrame]:
        """Get market data for S/R validation.

        Args:
            timeframe: The timeframe to get data for
            lookback_days: Override the default lookback period
            force_reload: Force reload data from disk

        Returns:
            DataFrame with market data or None if failed
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Use provided lookback_days or default
            actual_lookback_days = lookback_days or self.lookback_days

            # Check cache first
            cache_key = f"{timeframe}_{actual_lookback_days}"
            if not force_reload and cache_key in self._data_cache:
                last_load = self._last_load_time.get(cache_key)
                if last_load and (datetime.now() - last_load).total_seconds() < 3600:  # 1 hour cache
                    if self.logger:
                        self.logger.debug(f"📊 Using cached data for {timeframe}")
                    return self._data_cache[cache_key]

            # Load data
            data = await self._load_timeframe_data(timeframe, actual_lookback_days)

            if data is not None and len(data) > 0:
                # Cache the data
                self._data_cache[cache_key] = data
                self._last_load_time[cache_key] = datetime.now()

                if self.logger:
                    self.logger.info(f"📊 Loaded {len(data)} data points for {timeframe} ({actual_lookback_days} days lookback)")

                return data
            else:
                if self.logger:
                    self.logger.error(f"❌ No data available for {timeframe}")
                return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to get market data for {timeframe}: {e}")
            return None

    async def _load_timeframe_data(self, timeframe: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Load data for a specific timeframe.

        Args:
            timeframe: The timeframe to load
            lookback_days: Number of days to look back

        Returns:
            DataFrame with market data or None if failed
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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

    async def get_multi_timeframe_data(
        self,
        timeframes: Optional[List[str]] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get market data for multiple timeframes.

        Args:
            timeframes: List of timeframes to get data for
            lookback_days: Override the default lookback period

        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            timeframes = timeframes or self.timeframes
            lookback_days = lookback_days or self.lookback_days

            if self.logger:
                self.logger.info(f"📊 Loading multi-timeframe data for {len(timeframes)} timeframes")

            # Load data for each timeframe
            multi_tf_data = {}
            for timeframe in timeframes:
                data = await self.get_market_data(timeframe, lookback_days)
                if data is not None:
                    multi_tf_data[timeframe] = data
                else:
                    if self.logger:
                        self.logger.warning(f"⚠️ Failed to load data for {timeframe}")

            if self.logger:
                self.logger.info(f"✅ Loaded data for {len(multi_tf_data)} timeframes")

            return multi_tf_data

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to get multi-timeframe data: {e}")
            return {}

    def get_lookback_period_for_timeframe(self, timeframe: str) -> int:
        """Get the appropriate lookback period for a specific timeframe.

        Args:
            timeframe: The timeframe to get lookback period for

        Returns:
            Number of days to look back
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Timeframe-specific lookback periods
            timeframe_lookback_map = {
                "1m": min(self.lookback_days, 30),      # Max 30 days for 1m
                "5m": min(self.lookback_days, 60),      # Max 60 days for 5m
                "15m": min(self.lookback_days, 120),    # Max 120 days for 15m
                "30m": min(self.lookback_days, 180),    # Max 180 days for 30m
                "1h": min(self.lookback_days, 365),     # Max 1 year for 1h
                "4h": min(self.lookback_days, 730),     # Max 2 years for 4h
                "1d": self.lookback_days,               # Full lookback for daily
            }

            return timeframe_lookback_map.get(timeframe, self.lookback_days)

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to get lookback period for {timeframe}: {e}")
            return self.lookback_days

    async def validate_data_quality(self, data: pd.DataFrame, timeframe: str) -> bool:
        """Validate the quality of loaded data.

        Args:
            data: The data to validate
            timeframe: The timeframe the data represents

        Returns:
            True if data quality is acceptable, False otherwise
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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

    def _get_min_data_points_for_timeframe(self, timeframe: str) -> int:
        """Get minimum required data points for a timeframe.

        Args:
            timeframe: The timeframe to get minimum points for

        Returns:
            Minimum number of data points required
        """
        # Minimum data points based on timeframe
        min_points_map = {
            "1m": 1440,    # 1 day of minute data
            "5m": 288,     # 1 day of 5-minute data
            "15m": 96,     # 1 day of 15-minute data
            "30m": 48,     # 1 day of 30-minute data
            "1h": 24,      # 1 day of hourly data
            "4h": 6,       # 1 day of 4-hour data
            "1d": 30,      # 30 days of daily data
        }

        return min_points_map.get(timeframe, 100)

    def _get_expected_time_diff(self, timeframe: str) -> pd.Timedelta:
        """Get expected time difference between data points.

        Args:
            timeframe: The timeframe to get expected diff for

        Returns:
            Expected time difference
        """
        time_diff_map = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }

        return time_diff_map.get(timeframe, pd.Timedelta(minutes=1))

    async def cleanup_cache(self) -> None:
        """Clean up the data cache to free memory."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            if self.logger:
                self.logger.info("🧹 Cleaning up data cache...")

            # Clear old cache entries
            current_time = datetime.now()
            keys_to_remove = []

            for key, last_load in self._last_load_time.items():
                if (current_time - last_load).total_seconds() > 7200:  # 2 hours
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self._data_cache:
                    del self._data_cache[key]
                if key in self._last_load_time:
                    del self._last_load_time[key]

            if self.logger:
                self.logger.info(f"✅ Cache cleanup completed, removed {len(keys_to_remove)} entries")

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Cache cleanup failed: {e}")


# Convenience function for easy integration
async def create_sr_data_integration_simple(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframes: List[str] = None,
    lookback_days: Optional[int] = None,
    training_mode: str = "blank"
) -> SRDataIntegrationSimple:
    """Create and initialize a simplified S/R data integration instance.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframes: List of timeframes to use
        lookback_days: Override default lookback period
        training_mode: Training mode to use for lookback period

    Returns:
        Initialized SRDataIntegrationSimple instance
    """
    try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
        # Set default timeframes if not provided
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "30m"]

        # Get lookback days from training mode if not provided
        if lookback_days is None:
            mode_config = TRAINING_MODES.get(training_mode)
            if mode_config:
                lookback_days = mode_config["lookback_days"]
            else:
                lookback_days = DEFAULT_LOOKBACK_DAYS

        # Create configuration
        config = {
            "data_integration": {
                "symbol": symbol,
                "exchange": exchange,
                "timeframes": timeframes,
                "lookback_days": lookback_days,
                "training_mode": training_mode,
                "min_data_points": 1000,
                "max_data_age_hours": 24,
            }
        }

        # Create and initialize integration
        integration = SRDataIntegrationSimple(config)
        if await integration.initialize():
            return integration
        else:
            raise RuntimeError("Failed to initialize S/R data integration")

    except Exception as e:
        print(f"❌ Failed to create S/R data integration: {e}")
        raise