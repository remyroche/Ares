"""
Data loader utility for loading existing market data.

This module provides utilities to load existing market data from the historical_data directory
and prepare it for use in the market analysis sub-pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils

class DataLoader:
    """Utility class for loading existing market data."""

    def __init__(self):
        self.logger = system_logger.getChild('DataLoader')
        self.parquet_utils = ParquetUtils()

    def load_ethusdt_1h_data(self, data_dir: str = "historical_data") -> Optional[pd.DataFrame]:
        """
        Load ETHUSDT 1h processed data.

        Args:
            data_dir: Base data directory path

        Returns:
            DataFrame with 1h ETHUSDT data or None if not found
        """
        try:
            # Try to load from processed 1h data
            processed_path = Path(data_dir) / "binance" / "ethusdt" / "processed" / "ethusdt_1h"

            if processed_path.exists():
                self.logger.info(f"📂 Loading ETHUSDT 1h data from: {processed_path}")

                # Load all parquet files from the partitioned directory
                dataframes = []
                for year_dir in processed_path.iterdir():
                    if year_dir.is_dir() and year_dir.name.startswith('year='):
                        for month_dir in year_dir.iterdir():
                            if month_dir.is_dir() and month_dir.name.startswith('month='):
                                for file_path in month_dir.glob('*.parquet'):
                                    df = self.parquet_utils.safe_read_parquet(str(file_path))
                                    if df is not None and not df.empty:
                                        dataframes.append(df)

                if dataframes:
                    # Concatenate while preserving timestamp index
                    combined_df = pd.concat(dataframes, ignore_index=False)

                    # Handle timestamp index - convert to column for consistency
                    if combined_df.index.name == 'timestamp' or (hasattr(combined_df.index, 'name') and combined_df.index.name == 'timestamp'):
                        # Timestamp is already the index, convert to column
                        combined_df = combined_df.reset_index()
                        self.logger.info("Converted timestamp index to column")
                    elif 'timestamp' not in combined_df.columns and hasattr(combined_df.index, 'dtype') and 'datetime' in str(combined_df.index.dtype):
                        # If the index is datetime but not named 'timestamp', rename it
                        combined_df = combined_df.reset_index()
                        combined_df = combined_df.rename(columns={'index': 'timestamp'})
                        self.logger.info("Converted datetime index to timestamp column")
                    elif 'timestamp' not in combined_df.columns:
                        # No timestamp found, create a default one
                        combined_df = combined_df.reset_index(drop=True)
                        combined_df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(combined_df), freq='1H')
                        self.logger.warning("No timestamp found, created default timestamp")

                    # Sort by timestamp if it exists
                    if 'timestamp' in combined_df.columns:
                        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    else:
                        combined_df = combined_df.reset_index(drop=True)

                    self.logger.info(f"✅ Loaded {len(combined_df)} rows of ETHUSDT 1h data")
                    return combined_df

            # Fallback: try to load from raw data and resample
            self.logger.info("📂 Trying to load from raw 1m data and resample to 1h")
            raw_path = Path(data_dir) / "binance" / "ethusdt" / "raw"

            if raw_path.exists():
                dataframes = []
                for file_path in raw_path.glob('ethusdt_1m_*.parquet'):
                    df = self.parquet_utils.safe_read_parquet(str(file_path))
                    if df is not None and not df.empty:
                        dataframes.append(df)

                if dataframes:
                    # Concatenate while preserving timestamp index
                    combined_df = pd.concat(dataframes, ignore_index=False)

                    # Handle timestamp index - convert to column for consistency
                    if combined_df.index.name == 'timestamp' or (hasattr(combined_df.index, 'name') and combined_df.index.name == 'timestamp'):
                        # Timestamp is already the index, convert to column
                        combined_df = combined_df.reset_index()
                        self.logger.info("Converted timestamp index to column")
                    elif 'timestamp' not in combined_df.columns and hasattr(combined_df.index, 'dtype') and 'datetime' in str(combined_df.index.dtype):
                        # If the index is datetime but not named 'timestamp', rename it
                        combined_df = combined_df.reset_index()
                        combined_df = combined_df.rename(columns={'index': 'timestamp'})
                        self.logger.info("Converted datetime index to timestamp column")
                    elif 'timestamp' not in combined_df.columns:
                        # No timestamp found, create a default one
                        combined_df = combined_df.reset_index(drop=True)
                        combined_df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(combined_df), freq='1min')
                        self.logger.warning("No timestamp found, created default timestamp")

                    # Ensure timestamp is in proper format for resampling
                    if 'timestamp' in combined_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
                            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
                        combined_df.set_index('timestamp', inplace=True)

                    # Resample to 1h OHLCV
                    resampled = combined_df.resample('1H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    resampled.reset_index(inplace=True)
                    resampled['timestamp'] = resampled['timestamp'].astype('int64') // 10**6  # Convert back to milliseconds

                    self.logger.info(f"✅ Loaded and resampled {len(resampled)} rows of ETHUSDT 1h data")
                    return resampled

            self.logger.warning("❌ No ETHUSDT data found in expected locations")
            return None

        except Exception as e:
            self.logger.error(f"❌ Error loading ETHUSDT 1h data: {e}")
            return None

    def load_ethusdt_1m_data(self, data_dir: str = "historical_data") -> Optional[pd.DataFrame]:
        """
        Load ETHUSDT 1m raw data.

        Args:
            data_dir: Base data directory path

        Returns:
            DataFrame with 1m ETHUSDT data or None if not found
        """
        try:
            raw_path = Path(data_dir) / "binance" / "ethusdt" / "raw"

            if not raw_path.exists():
                self.logger.warning(f"❌ Raw data directory not found: {raw_path}")
                return None

            self.logger.info(f"📂 Loading ETHUSDT 1m data from: {raw_path}")

            dataframes = []
            for file_path in raw_path.glob('ethusdt_1m_*.parquet'):
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not df.empty:
                    dataframes.append(df)

            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                self.logger.info(f"✅ Loaded {len(combined_df)} rows of ETHUSDT 1m data")
                return combined_df

            self.logger.warning("❌ No ETHUSDT 1m data found")
            return None

        except Exception as e:
            self.logger.error(f"❌ Error loading ETHUSDT 1m data: {e}")
            return None

    def prepare_data_for_market_analysis(self, data: pd.DataFrame, symbol: str = "ETHUSDT",
                                       exchange: str = "binance", timeframe: str = "1h") -> Dict[str, Any]:
        """
        Prepare data for market analysis sub-pipeline.

        Args:
            data: Raw market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe

        Returns:
            Dictionary with prepared data and metadata
        """
        try:
            if data is None or data.empty:
                return {}

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                return {}

            # Add metadata columns
            data = data.copy()
            data['symbol'] = symbol
            data['exchange'] = exchange
            data['timeframe'] = timeframe

            # Ensure timestamp is in the correct format
            if 'timestamp' in data.columns:
                if data['timestamp'].dtype == 'int64':
                    # Convert from milliseconds to datetime for processing
                    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
                else:
                    data['datetime'] = pd.to_datetime(data['timestamp'])
            else:
                self.logger.warning("No timestamp column found in data")
                return {}

            self.logger.info(f"✅ Prepared data for market analysis: {len(data)} rows")

            return {
                'dataframe': data,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_points': len(data)
            }

        except Exception as e:
            self.logger.error(f"❌ Error preparing data for market analysis: {e}")
            return {}

# Convenience function for quick data loading
def load_ethusdt_data_for_analysis(data_dir: str = "historical_data",
                                  timeframe: str = "1h") -> Dict[str, Any]:
    """
    Convenience function to load ETHUSDT data for market analysis.

    Args:
        data_dir: Base data directory path
        timeframe: Data timeframe ('1h' or '1m')

    Returns:
        Dictionary with prepared data and metadata
    """
    loader = DataLoader()

    if timeframe == "1h":
        data = loader.load_ethusdt_1h_data(data_dir)
    elif timeframe == "1m":
        data = loader.load_ethusdt_1m_data(data_dir)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if data is not None:
        return loader.prepare_data_for_market_analysis(data, timeframe=timeframe)
    else:
        return {}
