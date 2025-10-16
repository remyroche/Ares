"""
Basic Returns Feature Engineer

This module provides tools to add basic return features to historical klines data
and resample to different timeframes with optimized parquet storage.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from src.utils.parquet_utils import get_parquet_utils
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.hardware.memory_optimization import MemoryMonitor, optimize_dataframe_dtypes
from src.utils.hardware.m1_optimizations import get_m1_memory_optimizer


class BasicReturnsEngineer:
    """Basic returns feature engineering and resampling for historical klines data."""

    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the basic returns engineer.
        
        Args:
            data_dir: Base directory for historical data
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "binance"
        self.processed_data_dir = self.data_dir / "binance"
        self.logger = system_logger.getChild("BasicReturnsEngineer")
        self.parquet_utils = get_parquet_utils()
        self.data_processor = DataProcessor()
        self.memory_monitor = MemoryMonitor()
        self.m1_optimizer = get_m1_memory_optimizer()
        
        # Create processed data directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_symbol_data(
        self,
        symbol: str,
        interval: str = "1m",
        target_intervals: List[str] = None
    ) -> Dict[str, Any]:
        """Process historical data for a symbol with basic returns and resampling.
        
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
            self.logger.info(f"🔧 Processing {symbol} data with basic returns")
            self.logger.info(f"📊 Target intervals: {target_intervals}")
            self.logger.info(f"📁 Data directory: {self.data_dir}")
            self.logger.info(f"⏰ Starting data processing pipeline...")

            # Load all raw data
            self.logger.info(f"📖 Loading raw {symbol} {interval} data from disk...")
            raw_data = self._load_all_raw_data(symbol, interval)
            
            if raw_data is None or raw_data.empty:
                self.logger.warning(f"❌ No raw data found for {symbol}")
                return {"success": False, "error": "No raw data found"}

            self.logger.info(f"✅ Loaded {len(raw_data):,} raw records spanning {raw_data.index.min()} to {raw_data.index.max()}")
            self.logger.info(f"🔄 Starting feature engineering...")

            # Add basic returns features
            self.logger.info(f"📈 Adding basic returns and technical features...")
            featured_data = self._add_basic_returns(raw_data)
            
            # Save featured 1m data
            self.logger.info(f"💾 Saving processed {symbol} {interval} data...")
            self._save_processed_data(featured_data, symbol, interval)
            self.logger.info(f"✅ Saved {symbol} {interval} data: {len(featured_data):,} records")

            # Resample to target intervals
            self.logger.info(f"🔄 Starting resampling to target intervals: {target_intervals}")
            resampling_results = {}
            for i, target_interval in enumerate(target_intervals, 1):
                try:
                    self.logger.info(f"📊 Resampling {i}/{len(target_intervals)}: {interval} → {target_interval}...")
                    resampled_data = self._resample_data(featured_data, target_interval)
                    if resampled_data is not None and not resampled_data.empty:
                        self.logger.info(f"💾 Saving {symbol} {target_interval} data...")
                        self._save_processed_data(resampled_data, symbol, target_interval)
                        resampling_results[target_interval] = {
                            "success": True,
                            "records": len(resampled_data)
                        }
                        self.logger.info(f"✅ Completed {target_interval}: {len(resampled_data):,} records")
                    else:
                        resampling_results[target_interval] = {
                            "success": False,
                            "error": "Empty resampled data"
                        }
                        self.logger.warning(f"⚠️ Empty resampled data for {target_interval}")
                except Exception as e:
                    resampling_results[target_interval] = {
                        "success": False,
                        "error": str(e)
                    }
                    self.logger.error(f"❌ Failed to resample to {target_interval}: {e}")
            
            # Final summary
            total_processed = len(featured_data) + sum(result.get("records", 0) for result in resampling_results.values() if result.get("success"))
            self.logger.info(f"🎉 Processing complete! Total records processed: {total_processed:,}")
            self.logger.info(f"📊 Summary: {len(featured_data):,} {interval} + {sum(result.get('records', 0) for result in resampling_results.values() if result.get('success')):,} resampled records")

            return {
                "success": True,
                "source_records": len(raw_data),
                "featured_records": len(featured_data),
                "resampling_results": resampling_results
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Basic returns processing failed: {e}")
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

            # Remove duplicates with improved strategy for financial data
            initial_count = len(combined_df)
            duplicate_mask = combined_df.index.duplicated(keep=False)  # Mark all duplicates

            if duplicate_mask.any():
                duplicate_count = duplicate_mask.sum()
                self.logger.info(f"🔍 Found {duplicate_count} duplicate timestamp entries to resolve")

                # More robust duplicate handling
                if duplicate_count > 0:
                    # Group by index (timestamp) and keep the most complete record
                    groups = combined_df.groupby(level=0)
                    resolved_dfs = []

                    for timestamp, group in groups:
                        if len(group) > 1:
                            # For each duplicate group, keep the record with most non-null values
                            # Prioritize OHLCV columns for completeness
                            priority_cols = ['open', 'high', 'low', 'close', 'volume']
                            completeness_scores = []

                            for idx in group.index:
                                row = group.loc[idx]
                                score = 0
                                # Higher score for non-null priority columns
                                for col in priority_cols:
                                    if col in row.index and pd.notna(row[col]):
                                        score += 2
                                # Additional score for any non-null values
                                score += row.notna().sum()
                                completeness_scores.append((idx, score))

                            # Keep the row with highest completeness score
                            best_idx = max(completeness_scores, key=lambda x: x[1])[0]
                            resolved_dfs.append(group.loc[[best_idx]])
                        else:
                            resolved_dfs.append(group)

                    # Reconstruct the dataframe
                    combined_df = pd.concat(resolved_dfs)

                # Final safety deduplication
                final_duplicates = combined_df.index.duplicated(keep='first')
                if final_duplicates.any():
                    combined_df = combined_df[~final_duplicates]
                    self.logger.info(f"🧹 Removed {final_duplicates.sum()} remaining duplicates")

                removed_count = initial_count - len(combined_df)
                if removed_count > 0:
                    self.logger.info(f"✅ Resolved {removed_count} duplicate timestamps, kept most complete records")

            # Filter out aggregated trade columns that shouldn't be in klines data
            # These columns can cause data type conversion issues during resampling
            aggregated_trade_columns = ['taker_buy_base', 'taker_buy_quote']
            columns_to_drop = [col for col in aggregated_trade_columns if col in combined_df.columns]
            if columns_to_drop:
                combined_df = combined_df.drop(columns=columns_to_drop)
                self.logger.info(f"🧹 Dropped aggregated trade columns: {columns_to_drop}")
            
            self.logger.info(f"📊 Loaded {len(combined_df)} raw records from {len(files)} files")
            return combined_df
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to load raw data for {symbol}: {e}")
            return None
    
    def _add_basic_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic return features to the DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added basic return features
        """
        try:
            self.logger.info(f"🔧 Starting feature engineering on {len(df)} records...")
            featured_df = df.copy()

            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in featured_df.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return df

            self.logger.info(f"✅ Found all required OHLCV columns")

            # Basic price returns (percentage) - only on close
            self.logger.info(f"📈 Calculating close returns...")
            featured_df['close_return'] = featured_df['close'].pct_change()

            # Log returns (more stable for financial data) - only on close
            self.logger.info(f"📈 Calculating log returns...")
            featured_df['close_log_return'] = np.log(featured_df['close'] / featured_df['close'].shift(1))

            # Volume returns (with safe handling for zero volumes)
            self.logger.info(f"📊 Calculating volume returns...")
            # Use safe_pct_change to handle zero volumes
            featured_df['volume_return'] = self._safe_pct_change(featured_df['volume'])

            # Safe log return calculation with comprehensive edge case handling
            current_volume = featured_df['volume']
            prev_volume = featured_df['volume'].shift(1)

            # Initialize with NaN values
            volume_log_return = np.full(len(featured_df), np.nan)

            # Valid cases: both current and previous volume > 0
            valid_mask = (current_volume > 0) & (prev_volume > 0)
            # Additional check: ensure ratio is positive and finite before taking log
            ratio = current_volume[valid_mask] / prev_volume[valid_mask]
            finite_ratio_mask = np.isfinite(ratio) & (ratio > 0)
            if finite_ratio_mask.any():
                volume_log_return[valid_mask] = np.where(
                    finite_ratio_mask,
                    np.log(np.clip(ratio, 1e-10, 1e10)),  # Clip ratio to prevent extreme log values
                    0.0
                )

            # Handle cases where current volume is 0 but previous was > 0
            zero_current_mask = (current_volume == 0) & (prev_volume > 0)
            volume_log_return[zero_current_mask] = -9.0  # Large negative value instead of -inf

            # Handle cases where previous volume was 0 but current is > 0
            zero_prev_mask = (current_volume > 0) & (prev_volume == 0)
            volume_log_return[zero_prev_mask] = 9.0  # Large positive value instead of +inf

            # Handle cases where both volumes are 0
            both_zero_mask = (current_volume == 0) & (prev_volume == 0)
            volume_log_return[both_zero_mask] = 0.0  # No change

            # Handle any potential NaN or infinite values from calculations
            volume_log_return = np.nan_to_num(volume_log_return, nan=0.0, posinf=9.0, neginf=-9.0)

            # Apply final clipping to ensure no infinite values remain
            volume_log_return = np.clip(volume_log_return, -9.0, 9.0)

            # Additional safety: replace any remaining non-finite values
            volume_log_return = np.where(np.isfinite(volume_log_return), volume_log_return, 0.0)

            featured_df['volume_log_return'] = volume_log_return

            # Price range features
            self.logger.info(f"📊 Calculating price range features...")
            featured_df['price_range'] = featured_df['high'] - featured_df['low']
            featured_df['price_range_pct'] = featured_df['price_range'] / featured_df['close']

            # Calculate body features
            featured_df = self._calculate_body_features(featured_df)

            # Finalize feature engineering
            featured_df = self._finalize_feature_engineering(featured_df, df)

            return featured_df

        except Exception as e:
            self.logger.exception(f"❌ Basic returns feature engineering failed: {e}")
            return df

    def _safe_pct_change(self, series: pd.Series) -> pd.Series:
        """Calculate percentage change with safe handling for zero values."""
        current = series
        prev = series.shift(1)

        # Initialize with NaN values
        pct_change = np.full(len(series), np.nan)

        # Valid cases: both current and previous > 0
        valid_mask = (current > 0) & (prev > 0)
        pct_change[valid_mask] = (current[valid_mask] - prev[valid_mask]) / prev[valid_mask]

        # Handle cases where current is 0 but previous was > 0
        zero_current_mask = (current == 0) & (prev > 0)
        pct_change[zero_current_mask] = -1.0  # -100% change

        # Handle cases where previous was 0 but current is > 0
        zero_prev_mask = (current > 0) & (prev == 0)
        pct_change[zero_prev_mask] = 9.0  # Large positive value instead of infinity

        # Handle cases where both are 0
        both_zero_mask = (current == 0) & (prev == 0)
        pct_change[both_zero_mask] = 0.0  # No change

        # Handle any potential NaN or infinite values from original data
        pct_change = np.nan_to_num(pct_change, nan=0.0, posinf=9.0, neginf=-9.0)

        # Apply final clipping to ensure no infinite values remain
        pct_change = np.clip(pct_change, -9.0, 9.0)

        # Additional safety: replace any remaining non-finite values
        pct_change = np.where(np.isfinite(pct_change), pct_change, 0.0)

        return pd.Series(pct_change, index=series.index)

    def _calculate_body_features(self, featured_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candle body features."""
        # Body size (absolute difference between open and close)
        self.logger.info(f"📊 Calculating candle body features...")
        featured_df['body_size'] = abs(featured_df['close'] - featured_df['open'])
        featured_df['body_size_pct'] = featured_df['body_size'] / featured_df['close']

        # Time-based features
        self.logger.info(f"🕐 Adding time-based features...")
        featured_df['hour'] = featured_df.index.hour
        featured_df['day_of_week'] = featured_df.index.dayofweek
        featured_df['is_weekend'] = featured_df['day_of_week'].isin([5, 6]).astype(int)

        return featured_df

    def _finalize_feature_engineering(self, featured_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Finalize feature engineering with optimization."""
        # Removed lagged and future features as requested
        self.logger.info(f"✅ Feature engineering complete: {len(featured_df.columns) - len(original_df.columns)} new features added")

        # Optimize data types using M1 optimizer (exclude string columns)
        numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
        string_cols = featured_df.select_dtypes(exclude=[np.number]).columns

        # Only optimize numeric columns
        if len(numeric_cols) > 0:
            numeric_df = featured_df[numeric_cols]
            optimized_numeric = self.m1_optimizer.create_memory_efficient_array(numeric_df, dtype=np.float32)
            featured_df[numeric_cols] = optimized_numeric

        # Keep string columns as-is
        if len(string_cols) > 0:
            featured_df[string_cols] = featured_df[string_cols].astype('string')

        return featured_df
    
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
            
            # Resample OHLCV data (exclude aggregated trade columns that shouldn't be in klines)
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum'
                # Note: taker_buy_base and taker_buy_quote are aggregated trade columns
                # that shouldn't be in klines data - they're being excluded
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
                    # Remove SMA and ratio handling as requested
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
            resampled_df = optimize_dataframe_dtypes(resampled_df)
            
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
            
            # Save as partitioned parquet using M1 optimizer
            output_path = processed_dir / f"{symbol.lower()}_{interval}"
            
            # Use memory-efficient saving
            with self.m1_optimizer.memory_checkpoint(f"save_{symbol}_{interval}"):
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
def process_ethusdt_basic_returns(
    data_dir: str = "historical_data",
    target_intervals: List[str] = None
) -> Dict[str, Any]:
    """Process ETHUSDT data with basic returns and resampling.
    
    Args:
        data_dir: Base directory for data storage
        target_intervals: List of target intervals for resampling
        
    Returns:
        Dictionary with processing results
    """
    if target_intervals is None:
        target_intervals = ["5m", "15m", "30m", "1h"]
    
    engineer = BasicReturnsEngineer(data_dir)
    return engineer.process_symbol_data("ETHUSDT", "1m", target_intervals)


if __name__ == "__main__":
    # Example usage
    engineer = BasicReturnsEngineer()
    results = engineer.process_symbol_data("ETHUSDT", "1m", ["5m", "15m", "30m", "1h"])
    print(f"Processing results: {results}")
