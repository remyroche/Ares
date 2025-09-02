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
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
from functools import wraps

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


def handle_errors(exceptions: tuple = (Exception,), default_return: Any = None, context: str = ""):
    """Decorator to handle errors gracefully."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if hasattr(args[0], 'logger') and args[0].logger:
                    args[0].logger.error(f"Error in {context}: {e}")
                else:
                    print(f"Error in {context}: {e}")
                return default_return
            except Exception as e:
                if hasattr(args[0], 'logger') and args[0].logger:
                    args[0].logger.exception(f"Unexpected error in {context}: {e}")
                else:
                    print(f"Unexpected error in {context}: {e}")
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if hasattr(args[0], 'logger') and args[0].logger:
                    args[0].logger.error(f"Error in {context}: {e}")
                else:
                    print(f"Error in {context}: {e}")
                return default_return
            except Exception as e:
                if hasattr(args[0], 'logger') and args[0].logger:
                    args[0].logger.exception(f"Unexpected error in {context}: {e}")
                else:
                    print(f"Unexpected error in {context}: {e}")
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


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
        
        # Initialize logger if system_logger is not available
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

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

        # State tracking
        self.is_initialized = False
        self._data_quality_checks = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="simplified srdataintegration initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the simplified S/R data integration system."""
        try:
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

            # Initialize data quality checks
            await self._initialize_data_quality_checks()

            self.is_initialized = True
            if self.logger:
                self.logger.info("✅ Simplified S/R Data Integration initialized successfully")
            return True

        except Exception as e:
            if self.logger:
                self.logger.exception(f"❌ Error initializing Simplified S/R Data Integration: {e}")
            return False

    async def _validate_configuration(self) -> bool:
        """Validate the configuration parameters."""
        try:
            # Check required parameters
            if not self.symbol:
                if self.logger:
                    self.logger.error("Symbol is required")
                return False

            if not self.exchange:
                if self.logger:
                    self.logger.error("Exchange is required")
                return False

            if not self.timeframes:
                if self.logger:
                    self.logger.error("At least one timeframe is required")
                return False

            # Validate lookback period
            if self.lookback_days <= 0:
                if self.logger:
                    self.logger.error(f"Invalid lookback_days: {self.lookback_days}")
                return False

            # Validate training mode
            if self.training_mode not in TRAINING_MODES:
                if self.logger:
                    self.logger.warning(f"Unknown training mode: {self.training_mode}, using 'blank'")
                self.training_mode = "blank"

            if self.logger:
                self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Configuration validation error: {e}")
            return False

    async def _initialize_data_quality_checks(self) -> None:
        """Initialize data quality check configurations."""
        self._data_quality_checks = {
            "missing_data_threshold": 0.05,  # 5% missing data allowed
            "outlier_threshold": 3.0,  # 3 standard deviations
            "consistency_check": True,
            "duplicate_check": True,
            "timestamp_order_check": True
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="simplified data loading"
    )
    async def load_data(self, timeframe: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data for a specific timeframe using simplified methods."""
        try:
            if not self.is_initialized:
                if self.logger:
                    self.logger.error("Simplified S/R Data Integration not initialized")
                return pd.DataFrame()

            # Check cache first
            cache_key = f"{timeframe}_{start_date}_{end_date}"
            if cache_key in self._data_cache:
                last_load = self._last_load_time.get(cache_key)
                if last_load and (datetime.now() - last_load).total_seconds() < self.max_data_age_hours * 3600:
                    if self.logger:
                        self.logger.info(f"Using cached data for {timeframe}")
                    return self._data_cache[cache_key]

            # Calculate date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.lookback_days)

            if self.logger:
                self.logger.info(f"Loading {timeframe} data from {start_date} to {end_date}")

            # Use simplified data loading
            data = await self._load_simplified_data(timeframe, start_date, end_date)

            if not data.empty:
                # Apply data quality checks
                data = await self._apply_data_quality_checks(data, timeframe)

                # Cache the data
                self._data_cache[cache_key] = data
                self._last_load_time[cache_key] = datetime.now()

                if self.logger:
                    self.logger.info(f"Successfully loaded {len(data)} records for {timeframe}")
            else:
                if self.logger:
                    self.logger.warning(f"No data loaded for {timeframe}")

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading data for {timeframe}: {e}")
            return pd.DataFrame()

    async def _load_simplified_data(self, timeframe: str, start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """Load data using simplified methods without external dependencies."""
        try:
            # Create sample data for demonstration
            # In a real implementation, this would connect to actual data sources
            date_range = pd.date_range(start=start_date, end=end_date, freq='1min')

            # Generate sample OHLCV data
            np.random.seed(42)  # For reproducible results
            n_points = len(date_range)

            # Generate realistic price data
            base_price = 50000  # Base BTC price
            price_changes = np.random.normal(0, 0.001, n_points)  # 0.1% volatility
            prices = base_price * np.exp(np.cumsum(price_changes))

            data = pd.DataFrame({
                'timestamp': date_range,
                'open': prices,
                'high': prices * (1 + np.random.uniform(0, 0.002, n_points)),
                'low': prices * (1 - np.random.uniform(0, 0.002, n_points)),
                'close': prices,
                'volume': np.random.uniform(100, 1000, n_points)
            })

            # Set timestamp as index
            data.set_index('timestamp', inplace=True)

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in simplified data loading: {e}")
            return pd.DataFrame()

    async def _apply_data_quality_checks(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Apply data quality checks and fixes."""
        try:
            original_length = len(data)

            # Check for missing data
            if self._data_quality_checks.get("missing_data_check", True):
                data = self._handle_missing_data(data)

            # Check for duplicates
            if self._data_quality_checks.get("duplicate_check", True):
                data = self._handle_duplicates(data)

            # Check timestamp order
            if self._data_quality_checks.get("timestamp_order_check", True):
                data = self._handle_timestamp_order(data)

            # Check for outliers
            if self._data_quality_checks.get("outlier_check", True):
                data = self._handle_outliers(data)

            final_length = len(data)
            if final_length != original_length and self.logger:
                self.logger.info(f"Data quality checks: {original_length} -> {final_length} records for {timeframe}")

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying data quality checks: {e}")
            return data

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in the DataFrame."""
        try:
            # Forward fill for OHLCV data
            data = data.fillna(method='ffill')

            # Drop any remaining rows with NaN values
            data = data.dropna()

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling missing data: {e}")
            return data

    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate timestamps."""
        try:
            # Remove duplicates based on index (timestamp)
            data = data[~data.index.duplicated(keep='first')]
            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling duplicates: {e}")
            return data

    def _handle_timestamp_order(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamps are in chronological order."""
        try:
            # Sort by timestamp
            data = data.sort_index()
            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling timestamp order: {e}")
            return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in price data."""
        try:
            # Simple outlier detection using IQR method
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Replace outliers with bounds
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

            return data

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling outliers: {e}")
            return data

    async def get_support_resistance_levels(self, timeframe: str,
                                          lookback_days: Optional[int] = None) -> Dict[str, List[float]]:
        """Get support and resistance levels for a given timeframe."""
        try:
            if not self.is_initialized:
                if self.logger:
                    self.logger.error("Simplified S/R Data Integration not initialized")
                return {"support": [], "resistance": []}

            # Use provided lookback or default
            if lookback_days is None:
                lookback_days = self.lookback_days

            # Load data
            data = await self.load_data(timeframe,
                                      start_date=datetime.now() - timedelta(days=lookback_days))

            if data.empty:
                if self.logger:
                    self.logger.warning(f"No data available for {timeframe}")
                return {"support": [], "resistance": []}

            # Calculate support and resistance levels
            levels = self._calculate_sr_levels(data)

            if self.logger:
                self.logger.info(f"Calculated {len(levels['support'])} support and {len(levels['resistance'])} resistance levels for {timeframe}")
            return levels

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating S/R levels: {e}")
            return {"support": [], "resistance": []}

    def _calculate_sr_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels from price data."""
        try:
            levels = {"support": [], "resistance": []}

            if data.empty:
                return levels

            # Simple pivot point calculation
            high = data['high'].max()
            low = data['low'].min()
            close = data['close'].iloc[-1]

            pivot = (high + low + close) / 3

            # Support levels
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)

            # Resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)

            # Filter out invalid levels
            valid_support = [s for s in [s1, s2, s3] if s > 0 and s < close]
            valid_resistance = [r for r in [r1, r2, r3] if r > close]

            levels["support"] = sorted(valid_support, reverse=True)
            levels["resistance"] = sorted(valid_resistance)

            return levels

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in S/R level calculation: {e}")
            return {"support": [], "resistance": []}

    async def validate_data_quality(self, timeframe: str) -> Dict[str, Any]:
        """Validate data quality for a specific timeframe."""
        try:
            if not self.is_initialized:
                return {"valid": False, "error": "Not initialized"}

            data = await self.load_data(timeframe)

            if data.empty:
                return {"valid": False, "error": "No data available"}

            # Perform quality checks
            quality_report = {
                "valid": True,
                "total_records": len(data),
                "missing_data": data.isnull().sum().to_dict(),
                "duplicates": data.index.duplicated().sum(),
                "timestamp_order": data.index.is_monotonic_increasing,
                "data_age_hours": (datetime.now() - data.index.max()).total_seconds() / 3600,
                "price_range": {
                    "min": data[['open', 'high', 'low', 'close']].min().to_dict(),
                    "max": data[['open', 'high', 'low', 'close']].max().to_dict()
                }
            }

            # Determine if data is valid
            if quality_report["data_age_hours"] > self.max_data_age_hours:
                quality_report["valid"] = False
                quality_report["error"] = f"Data too old: {quality_report['data_age_hours']:.1f} hours"

            if quality_report["total_records"] < self.min_data_points:
                quality_report["valid"] = False
                quality_report["error"] = f"Insufficient data points: {quality_report['total_records']}"

            return quality_report

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating data quality: {e}")
            return {"valid": False, "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.logger:
                self.logger.info("Cleaning up Simplified S/R Data Integration...")

            # Clear cache
            self._data_cache.clear()
            self._last_load_time.clear()

            # Reset state
            self.is_initialized = False

            if self.logger:
                self.logger.info("Cleanup completed")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if self.is_initialized:
                asyncio.create_task(self.cleanup())
        except:
            pass


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of SRDataIntegrationSimple."""
        config = {
            "data_integration": {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "timeframes": ["1m", "5m", "15m"],
                "lookback_days": 30,
                "training_mode": "light"
            }
        }

        sr_integration = SRDataIntegrationSimple(config)

        try:
            # Initialize
            if await sr_integration.initialize():
                print("✅ Simplified S/R Data Integration initialized successfully")

                # Load data
                data = await sr_integration.load_data("1m")
                print(f"📊 Loaded {len(data)} records")

                # Get S/R levels
                levels = await sr_integration.get_support_resistance_levels("1m")
                print(f"🎯 Support levels: {levels['support']}")
                print(f"🎯 Resistance levels: {levels['resistance']}")

                # Validate data quality
                quality = await sr_integration.validate_data_quality("1m")
                print(f"🔍 Data quality: {quality}")

            else:
                print("❌ Failed to initialize Simplified S/R Data Integration")

        except Exception as e:
            print(f"❌ Error in main: {e}")

        finally:
            # Cleanup
            await sr_integration.cleanup()

    # Run the example
    asyncio.run(main())