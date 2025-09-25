"""
Kline Parquet Utilities

This module provides utilities for working with Kline (candlestick) data in Parquet format.
It includes functions for reading, writing, processing, and analyzing Kline data with
comprehensive error handling and tprint logging.

Key Features:
- Kline data reading and writing
- Data validation and quality checks
- Time series processing
- Technical indicator calculations
- Data aggregation and resampling
- Memory optimization for large datasets
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

# Import utility modules with error handling
try:
    from .tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_performance, tprint_structured, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: tprint not available: {e}")
    TPRINT_AVAILABLE = False
    
    # Fallback tprint functions
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")
    def tprint_debug(msg): print(f"DEBUG: {msg}")
    def tprint_performance(msg): print(f"PERFORMANCE: {msg}")
    def tprint_structured(data): print(f"STRUCTURED: {data}")
    def tprint_timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"TIMER: {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper

try:
    from .common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        validate_finite, validate_positive, safe_divide
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: common_operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

try:
    from .math_validation import (
        MathValidation, safe_correlation, safe_covariance,
        safe_mean, safe_std, validate_finite as math_validate_finite
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: math_validation not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class KlineDataProcessor:
    """
    Kline data processor with comprehensive error handling.
    
    This class provides utilities for processing Kline (candlestick) data
    with robust error handling and performance optimization.
    """
    
    def __init__(self, memory_limit_gb: Optional[float] = None):
        """
        Initialize Kline data processor.
        
        Args:
            memory_limit_gb: Memory limit in GB for processing
        """
        try:
            tprint_info("🚀 Initializing Kline Data Processor...")
            
            self.memory_limit_gb = memory_limit_gb
            self.math_validator = MathValidation() if MATH_VALIDATION_AVAILABLE else None
            
            # Data quality metrics
            self.quality_metrics = {
                'total_records': 0,
                'missing_values': 0,
                'duplicate_records': 0,
                'invalid_ohlc': 0,
                'time_gaps': 0,
                'data_quality_score': 0.0
            }
            
            tprint_success("✅ Kline Data Processor initialized successfully")
            
        except Exception as e:
            tprint_error(f"Kline Data Processor initialization failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Kline Data Processor: {e}")
    
    def read_kline_parquet(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Read Kline data from Parquet file with comprehensive error handling.
        
        Args:
            filepath: Path to Parquet file
            
        Returns:
            DataFrame containing Kline data
        """
        try:
            tprint_info(f"📖 Reading Kline data from {filepath}")
            
            # Validate file path
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            if not filepath.suffix.lower() == '.parquet':
                tprint_warning(f"File extension is not .parquet: {filepath}")
            
            # Read Parquet file with error handling
            try:
                df = pd.read_parquet(filepath)
                tprint_success(f"✅ Successfully read {len(df)} records from {filepath}")
            except Exception as e:
                tprint_error(f"Failed to read Parquet file: {e}")
                raise
            
            # Validate Kline data structure
            try:
                self._validate_kline_data(df)
                tprint_success("✅ Kline data validation passed")
            except Exception as e:
                tprint_warning(f"Kline data validation failed: {e}")
                # Continue with processing
            
            # Calculate data quality metrics
            try:
                self._calculate_quality_metrics(df)
                tprint_info(f"📊 Data quality score: {self.quality_metrics['data_quality_score']:.2f}")
            except Exception as e:
                tprint_warning(f"Quality metrics calculation failed: {e}")
            
            return df
            
        except Exception as e:
            tprint_error(f"Failed to read Kline Parquet file: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def write_kline_parquet(self, df: pd.DataFrame, filepath: Union[str, Path], 
                          compression: str = 'snappy') -> bool:
        """
        Write Kline data to Parquet file with comprehensive error handling.
        
        Args:
            df: DataFrame containing Kline data
            filepath: Output file path
            compression: Compression algorithm
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Writing Kline data to {filepath}")
            
            # Validate input data
            if df is None or df.empty:
                tprint_error("❌ Empty DataFrame provided")
                return False
            
            # Validate file path
            filepath = Path(filepath)
            ensure_directory(filepath.parent) if COMMON_OPS_AVAILABLE else filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate Kline data structure
            try:
                self._validate_kline_data(df)
                tprint_success("✅ Kline data validation passed")
            except Exception as e:
                tprint_warning(f"Kline data validation failed: {e}")
                # Continue with writing
            
            # Write Parquet file with error handling
            try:
                df.to_parquet(filepath, compression=compression, index=False)
                tprint_success(f"✅ Successfully wrote {len(df)} records to {filepath}")
                return True
            except Exception as e:
                tprint_error(f"Failed to write Parquet file: {e}")
                return False
                
        except Exception as e:
            tprint_error(f"Failed to write Kline Parquet file: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_kline_data(self, df: pd.DataFrame) -> bool:
        """
        Validate Kline data structure with comprehensive error handling.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            tprint_info("🔍 Validating Kline data structure...")
            
            # Check if DataFrame is empty
            if df.empty:
                tprint_error("❌ DataFrame is empty")
                return False
            
            # Required columns for Kline data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            optional_columns = ['timestamp', 'datetime', 'time', 'date']
            
            # Check required columns
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                tprint_error(f"❌ Missing required columns: {missing_required}")
                return False
            
            # Check for timestamp column
            timestamp_cols = [col for col in optional_columns if col in df.columns]
            if not timestamp_cols:
                tprint_warning("⚠️ No timestamp column found")
            else:
                tprint_info(f"✅ Found timestamp column: {timestamp_cols[0]}")
            
            # Validate OHLC data
            try:
                self._validate_ohlc_data(df)
                tprint_success("✅ OHLC data validation passed")
            except Exception as e:
                tprint_warning(f"OHLC data validation failed: {e}")
            
            # Validate volume data
            try:
                self._validate_volume_data(df)
                tprint_success("✅ Volume data validation passed")
            except Exception as e:
                tprint_warning(f"Volume data validation failed: {e}")
            
            tprint_success("✅ Kline data structure validation completed")
            return True
            
        except Exception as e:
            tprint_error(f"Kline data validation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLC (Open, High, Low, Close) data.
        
        Args:
            df: DataFrame containing OHLC data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for negative values
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        tprint_warning(f"⚠️ Found {negative_count} negative values in {col}")
            
            # Check OHLC relationships
            invalid_ohlc = 0
            for col in ohlc_columns:
                if col in df.columns:
                    # High should be >= Low
                    if 'high' in df.columns and 'low' in df.columns:
                        invalid_high_low = (df['high'] < df['low']).sum()
                        if invalid_high_low > 0:
                            tprint_warning(f"⚠️ Found {invalid_high_low} records where high < low")
                            invalid_ohlc += invalid_high_low
                    
                    # High should be >= Open and Close
                    if 'high' in df.columns and col in ['open', 'close']:
                        invalid_high = (df['high'] < df[col]).sum()
                        if invalid_high > 0:
                            tprint_warning(f"⚠️ Found {invalid_high} records where high < {col}")
                            invalid_ohlc += invalid_high
                    
                    # Low should be <= Open and Close
                    if 'low' in df.columns and col in ['open', 'close']:
                        invalid_low = (df['low'] > df[col]).sum()
                        if invalid_low > 0:
                            tprint_warning(f"⚠️ Found {invalid_low} records where low > {col}")
                            invalid_ohlc += invalid_low
            
            if invalid_ohlc > 0:
                tprint_warning(f"⚠️ Total invalid OHLC relationships: {invalid_ohlc}")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"OHLC data validation failed: {e}")
            return False
    
    def _validate_volume_data(self, df: pd.DataFrame) -> bool:
        """
        Validate volume data.
        
        Args:
            df: DataFrame containing volume data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if 'volume' not in df.columns:
                tprint_warning("⚠️ No volume column found")
                return True
            
            # Check for negative volume
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                tprint_warning(f"⚠️ Found {negative_volume} negative volume values")
            
            # Check for infinite volume
            infinite_volume = np.isinf(df['volume']).sum()
            if infinite_volume > 0:
                tprint_warning(f"⚠️ Found {infinite_volume} infinite volume values")
            
            return True
            
        except Exception as e:
            tprint_error(f"Volume data validation failed: {e}")
            return False
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data quality metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            tprint_info("📊 Calculating data quality metrics...")
            
            # Basic metrics
            total_records = len(df)
            missing_values = df.isnull().sum().sum()
            duplicate_records = df.duplicated().sum()
            
            # OHLC validation
            invalid_ohlc = 0
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = ((df['high'] < df['low']) | 
                              (df['high'] < df['open']) | 
                              (df['high'] < df['close']) |
                              (df['low'] > df['open']) | 
                              (df['low'] > df['close'])).sum()
            
            # Time gaps (if timestamp available)
            time_gaps = 0
            timestamp_cols = [col for col in ['timestamp', 'datetime', 'time', 'date'] if col in df.columns]
            if timestamp_cols:
                try:
                    time_col = timestamp_cols[0]
                    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        time_diff = df[time_col].diff()
                        expected_interval = time_diff.mode().iloc[0] if not time_diff.empty else None
                        if expected_interval:
                            time_gaps = (time_diff != expected_interval).sum()
                except Exception as e:
                    tprint_warning(f"Time gap calculation failed: {e}")
            
            # Calculate quality score
            quality_score = 1.0
            if total_records > 0:
                quality_score -= (missing_values / total_records) * 0.3
                quality_score -= (duplicate_records / total_records) * 0.2
                quality_score -= (invalid_ohlc / total_records) * 0.3
                quality_score -= (time_gaps / total_records) * 0.2
                quality_score = max(0.0, quality_score)
            
            # Update metrics
            self.quality_metrics = {
                'total_records': total_records,
                'missing_values': missing_values,
                'duplicate_records': duplicate_records,
                'invalid_ohlc': invalid_ohlc,
                'time_gaps': time_gaps,
                'data_quality_score': quality_score
            }
            
            tprint_success(f"✅ Quality metrics calculated: score = {quality_score:.2f}")
            return self.quality_metrics
            
        except Exception as e:
            tprint_error(f"Quality metrics calculation failed: {e}")
            return self.quality_metrics
    
    def resample_kline_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample Kline data to different frequency with comprehensive error handling.
        
        Args:
            df: DataFrame containing Kline data
            frequency: Target frequency (e.g., '1H', '1D', '1W')
            
        Returns:
            Resampled DataFrame
        """
        try:
            tprint_info(f"🔄 Resampling Kline data to {frequency}")
            
            # Find timestamp column
            timestamp_cols = [col for col in ['timestamp', 'datetime', 'time', 'date'] if col in df.columns]
            if not timestamp_cols:
                tprint_error("❌ No timestamp column found for resampling")
                return df
            
            time_col = timestamp_cols[0]
            
            # Ensure timestamp column is datetime
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    df[time_col] = pd.to_datetime(df[time_col])
                tprint_info("✅ Timestamp column converted to datetime")
            except Exception as e:
                tprint_error(f"Failed to convert timestamp column: {e}")
                return df
            
            # Set timestamp as index
            df_indexed = df.set_index(time_col)
            
            # Resample data
            try:
                resampled = df_indexed.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                tprint_success(f"✅ Successfully resampled to {frequency}: {len(resampled)} records")
                return resampled.reset_index()
                
            except Exception as e:
                tprint_error(f"Resampling failed: {e}")
                return df
                
        except Exception as e:
            tprint_error(f"Kline data resampling failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for Kline data.
        
        Args:
            df: DataFrame containing Kline data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            tprint_info("📈 Calculating technical indicators...")
            
            result_df = df.copy()
            
            # Simple Moving Averages
            try:
                for period in [5, 10, 20, 50]:
                    result_df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                tprint_info("✅ Simple Moving Averages calculated")
            except Exception as e:
                tprint_warning(f"SMA calculation failed: {e}")
            
            # Exponential Moving Averages
            try:
                for period in [12, 26]:
                    result_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                tprint_info("✅ Exponential Moving Averages calculated")
            except Exception as e:
                tprint_warning(f"EMA calculation failed: {e}")
            
            # RSI
            try:
                result_df['rsi'] = self._calculate_rsi(df['close'])
                tprint_info("✅ RSI calculated")
            except Exception as e:
                tprint_warning(f"RSI calculation failed: {e}")
            
            # MACD
            try:
                macd_line, signal_line, histogram = self._calculate_macd(df['close'])
                result_df['macd'] = macd_line
                result_df['macd_signal'] = signal_line
                result_df['macd_histogram'] = histogram
                tprint_info("✅ MACD calculated")
            except Exception as e:
                tprint_warning(f"MACD calculation failed: {e}")
            
            # Bollinger Bands
            try:
                upper, middle, lower = self._calculate_bollinger_bands(df['close'])
                result_df['bb_upper'] = upper
                result_df['bb_middle'] = middle
                result_df['bb_lower'] = lower
                tprint_info("✅ Bollinger Bands calculated")
            except Exception as e:
                tprint_warning(f"Bollinger Bands calculation failed: {e}")
            
            tprint_success("✅ Technical indicators calculation completed")
            return result_df
            
        except Exception as e:
            tprint_error(f"Technical indicators calculation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            tprint_warning(f"RSI calculation failed: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            tprint_warning(f"MACD calculation failed: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
        except Exception as e:
            tprint_warning(f"Bollinger Bands calculation failed: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series


# Convenience functions
def read_kline_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Convenience function to read Kline Parquet file."""
    processor = KlineDataProcessor()
    return processor.read_kline_parquet(filepath)


def write_kline_parquet(df: pd.DataFrame, filepath: Union[str, Path], 
                       compression: str = 'snappy') -> bool:
    """Convenience function to write Kline Parquet file."""
    processor = KlineDataProcessor()
    return processor.write_kline_parquet(df, filepath, compression)


def resample_kline_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Convenience function to resample Kline data."""
    processor = KlineDataProcessor()
    return processor.resample_kline_data(df, frequency)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to calculate technical indicators."""
    processor = KlineDataProcessor()
    return processor.calculate_technical_indicators(df)


# Example usage
if __name__ == "__main__":
    # Example usage
    tprint_info("🧪 Testing Kline Parquet utilities...")
    
    # Create sample Kline data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })
    
    # Test processor
    processor = KlineDataProcessor()
    
    # Test data validation
    processor._validate_kline_data(sample_data)
    
    # Test technical indicators
    sample_data_with_indicators = processor.calculate_technical_indicators(sample_data)
    
    tprint_success("✅ Kline Parquet utilities test completed")