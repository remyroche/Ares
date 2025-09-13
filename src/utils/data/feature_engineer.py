"""
Feature Engineering and Resampling Tools for Historical Data

This module provides tools to add features and resample historical klines data
to different timeframes with optimized parquet storage.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.processing.data_processing import DataProcessor


class FeatureEngineer:
    """Feature engineering and resampling for historical klines data."""

    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the feature engineer.
        
        Args:
            data_dir: Base directory for historical data
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "binance"
        self.processed_data_dir = self.data_dir / "binance"
        self.logger = system_logger.getChild("FeatureEngineer")
        self.parquet_utils = ParquetUtils()
        self.data_processor = DataProcessor()
        
        # Create processed data directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_symbol_data(
        self,
        symbol: str,
        interval: str = "1m",
        target_intervals: List[str] = None
    ) -> Dict[str, Any]:
        """Process historical data for a symbol with feature engineering and resampling.
        
        Args:
            symbol: Trading symbol
            interval: Source interval (e.g., '1m')
            target_intervals: List of target intervals for resampling
            
        Returns:
            Dictionary with processing results
        """
        if target_intervals is None:
            target_intervals = ["5m", "15m", "30m", "1h"]
        
        try:
            self.logger.info(f"🔧 Processing {symbol} data with feature engineering")
            self.logger.info(f"📊 Target intervals: {target_intervals}")
            
            # Load all raw data
            raw_data = self._load_all_raw_data(symbol, interval)
            
            if raw_data is None or raw_data.empty:
                self.logger.warning(f"No raw data found for {symbol}")
                return {"success": False, "error": "No raw data found"}
            
            # Add features to raw data
            featured_data = self._add_features(raw_data)
            
            # Save featured 1m data
            self._save_processed_data(featured_data, symbol, interval)
            
            # Resample to target intervals
            resampling_results = {}
            for target_interval in target_intervals:
                try:
                    resampled_data = self._resample_data(featured_data, target_interval)
                    if resampled_data is not None and not resampled_data.empty:
                        self._save_processed_data(resampled_data, symbol, target_interval)
                        resampling_results[target_interval] = {
                            "success": True,
                            "records": len(resampled_data)
                        }
                        self.logger.info(f"✅ Resampled to {target_interval}: {len(resampled_data)} records")
                    else:
                        resampling_results[target_interval] = {
                            "success": False,
                            "error": "Empty resampled data"
                        }
                except Exception as e:
                    resampling_results[target_interval] = {
                        "success": False,
                        "error": str(e)
                    }
                    self.logger.error(f"❌ Failed to resample to {target_interval}: {e}")
            
            return {
                "success": True,
                "source_records": len(raw_data),
                "featured_records": len(featured_data),
                "resampling_results": resampling_results
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Feature engineering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_all_raw_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load all raw historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            
        Returns:
            Combined DataFrame or None if no data
        """
        try:
            symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
            if not symbol_dir.exists():
                return None
            
            # Find all parquet files for this symbol and interval
            pattern = f"{symbol.lower()}_{interval}_*.parquet"
            files = list(symbol_dir.glob(pattern))
            
            if not files:
                return None
            
            # Load and combine all files
            dataframes = []
            for file_path in sorted(files):
                try:
                    df = self.parquet_utils.safe_read_parquet(str(file_path))
                    if df is not None and not df.empty:
                        dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
            
            if not dataframes:
                return None
            
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=False)
            combined_df = combined_df.sort_index()
            
            # Remove duplicates
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            self.logger.info(f"📊 Loaded {len(combined_df)} raw records from {len(files)} files")
            return combined_df
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to load raw data for {symbol}: {e}")
            return None
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features to the DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        try:
            featured_df = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in featured_df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return df
            
            # Price returns
            featured_df['close_return'] = featured_df['close'].pct_change()
            featured_df['open_return'] = featured_df['open'].pct_change()
            featured_df['high_return'] = featured_df['high'].pct_change()
            featured_df['low_return'] = featured_df['low'].pct_change()
            
            # Log returns (more stable for financial data)
            featured_df['close_log_return'] = np.log(featured_df['close'] / featured_df['close'].shift(1))
            featured_df['open_log_return'] = np.log(featured_df['open'] / featured_df['open'].shift(1))
            
            # Volume returns
            featured_df['volume_return'] = featured_df['volume'].pct_change()
            featured_df['volume_log_return'] = np.log(featured_df['volume'] / featured_df['volume'].shift(1))
            
            # Price features
            featured_df['price_range'] = featured_df['high'] - featured_df['low']
            featured_df['price_range_pct'] = featured_df['price_range'] / featured_df['close']
            featured_df['body_size'] = abs(featured_df['close'] - featured_df['open'])
            featured_df['body_size_pct'] = featured_df['body_size'] / featured_df['close']
            
            # Upper and lower shadows
            featured_df['upper_shadow'] = featured_df['high'] - featured_df[['open', 'close']].max(axis=1)
            featured_df['lower_shadow'] = featured_df[['open', 'close']].min(axis=1) - featured_df['low']
            
            # Volume features
            featured_df['volume_sma_20'] = featured_df['volume'].rolling(window=20).mean()
            featured_df['volume_ratio'] = featured_df['volume'] / featured_df['volume_sma_20']
            
            # Price momentum features
            featured_df['close_sma_5'] = featured_df['close'].rolling(window=5).mean()
            featured_df['close_sma_20'] = featured_df['close'].rolling(window=20).mean()
            featured_df['close_ema_12'] = featured_df['close'].ewm(span=12).mean()
            featured_df['close_ema_26'] = featured_df['close'].ewm(span=26).mean()
            
            # RSI (simplified)
            featured_df['rsi_14'] = self._calculate_rsi(featured_df['close'], 14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(featured_df['close'], 20, 2)
            featured_df['bb_upper'] = bb_upper
            featured_df['bb_middle'] = bb_middle
            featured_df['bb_lower'] = bb_lower
            featured_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            featured_df['bb_position'] = (featured_df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volatility features
            featured_df['volatility_20'] = featured_df['close_return'].rolling(window=20).std()
            featured_df['volatility_5'] = featured_df['close_return'].rolling(window=5).std()
            
            # Time-based features
            featured_df['hour'] = featured_df.index.hour
            featured_df['day_of_week'] = featured_df.index.dayofweek
            featured_df['is_weekend'] = featured_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                featured_df[f'close_lag_{lag}'] = featured_df['close'].shift(lag)
                featured_df[f'volume_lag_{lag}'] = featured_df['volume'].shift(lag)
            
            # Forward-looking features (for analysis, not trading)
            featured_df['close_future_1'] = featured_df['close'].shift(-1)
            featured_df['close_future_5'] = featured_df['close'].shift(-5)
            featured_df['close_future_10'] = featured_df['close'].shift(-10)
            
            # Future returns
            featured_df['future_return_1'] = featured_df['close_future_1'] / featured_df['close'] - 1
            featured_df['future_return_5'] = featured_df['close_future_5'] / featured_df['close'] - 1
            featured_df['future_return_10'] = featured_df['close_future_10'] / featured_df['close'] - 1
            
            # Optimize data types
            featured_df = self.data_processor.optimize_feature_engineering_pipeline(
                featured_df, stage="output"
            )
            
            self.logger.info(f"✅ Added {len(featured_df.columns) - len(df.columns)} features")
            return featured_df
            
        except Exception as e:
            self.logger.exception(f"❌ Feature engineering failed: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Price series
            window: RSI window
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        window: int = 20, 
        std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def _resample_data(self, df: pd.DataFrame, target_interval: str) -> Optional[pd.DataFrame]:
        """Resample data to target interval.
        
        Args:
            df: Input DataFrame
            target_interval: Target interval (e.g., '5m', '15m', '1h')
            
        Returns:
            Resampled DataFrame or None if failed
        """
        try:
            # Convert interval string to pandas frequency
            freq_map = {
                '1m': '1T',
                '3m': '3T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '2h': '2H',
                '4h': '4H',
                '6h': '6H',
                '8h': '8H',
                '12h': '12H',
                '1d': '1D',
                '1w': '1W',
                '1M': '1M'
            }
            
            freq = freq_map.get(target_interval)
            if not freq:
                self.logger.error(f"Unknown interval: {target_interval}")
                return None
            
            # Resample OHLCV data
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'taker_buy_base': 'sum',
                'taker_buy_quote': 'sum'
            }
            
            # Resample basic OHLCV columns
            resampled_df = df[list(ohlc_dict.keys())].resample(freq).agg(ohlc_dict)
            
            # Resample feature columns (use appropriate aggregation)
            feature_columns = [col for col in df.columns if col not in ohlc_dict.keys()]
            
            for col in feature_columns:
                if col in df.columns:
                    if 'return' in col or 'log_return' in col:
                        # For returns, use mean
                        resampled_df[col] = df[col].resample(freq).mean()
                    elif 'sma' in col or 'ema' in col or 'rsi' in col or 'bb_' in col:
                        # For technical indicators, use last value
                        resampled_df[col] = df[col].resample(freq).last()
                    elif 'volatility' in col or 'std' in col:
                        # For volatility, use mean
                        resampled_df[col] = df[col].resample(freq).mean()
                    elif 'lag_' in col or 'future_' in col:
                        # For lagged/future features, use last value
                        resampled_df[col] = df[col].resample(freq).last()
                    elif col in ['hour', 'day_of_week', 'is_weekend', 'year', 'month', 'day']:
                        # For time features, use first value
                        resampled_df[col] = df[col].resample(freq).first()
                    else:
                        # Default to last value
                        resampled_df[col] = df[col].resample(freq).last()
            
            # Add metadata
            resampled_df['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
            resampled_df['interval'] = target_interval
            resampled_df['year'] = resampled_df.index.year
            resampled_df['month'] = resampled_df.index.month
            resampled_df['day'] = resampled_df.index.day
            
            # Remove rows with all NaN values
            resampled_df = resampled_df.dropna(how='all')
            
            # Optimize data types
            resampled_df = self.data_processor.optimize_feature_engineering_pipeline(
                resampled_df, stage="output"
            )
            
            return resampled_df
            
        except Exception as e:
            self.logger.exception(f"❌ Resampling failed for {target_interval}: {e}")
            return None
    
    def _save_processed_data(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Save processed data with optimized parquet partitioning.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create processed data directory structure
            processed_dir = self.processed_data_dir / symbol.lower() / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Group by year and month for partitioning
            df_with_partitions = df.copy()
            df_with_partitions['year'] = df_with_partitions.index.year
            df_with_partitions['month'] = df_with_partitions.index.month
            
            # Save as partitioned parquet
            output_path = processed_dir / f"{symbol.lower()}_{interval}"
            
            # Use pyarrow for better partitioning support
            df_with_partitions.to_parquet(
                output_path,
                partition_cols=['year', 'month'],
                index=True,
                compression='snappy',
                engine='pyarrow'
            )
            
            self.logger.info(f"💾 Saved processed data: {symbol} {interval} ({len(df)} records)")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to save processed data: {e}")
            return False


# Convenience functions
def process_ethusdt_data(
    data_dir: str = "historical_data",
    target_intervals: List[str] = None
) -> Dict[str, Any]:
    """Process ETHUSDT data with feature engineering and resampling.
    
    Args:
        data_dir: Base directory for data storage
        target_intervals: List of target intervals for resampling
        
    Returns:
        Dictionary with processing results
    """
    if target_intervals is None:
        target_intervals = ["5m", "15m", "30m", "1h"]
    
    engineer = FeatureEngineer(data_dir)
    return engineer.process_symbol_data("ETHUSDT", "1m", target_intervals)


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    results = engineer.process_symbol_data("ETHUSDT", "1m", ["5m", "15m", "30m", "1h"])
    print(f"Processing results: {results}")