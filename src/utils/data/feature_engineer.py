"""
Feature Engineering and Resampling Tools for Historical Data

This module provides tools to add features and resample historical klines data
to different timeframes with optimized parquet storage.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.processing.data_processing import DataProcessor

# Import unified matrix operations for optimized calculations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
except ImportError:
    get_unified_matrix_operations = None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    system_logger.warning("Unified matrix operations not available, falling back to numpy")

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

        # Initialize unified matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=True,
                enable_memory_optimization=True,
                enable_parallel=True
            )
            self.logger.info("✅ Unified matrix operations initialized for feature engineering")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Unified matrix operations not available, using numpy fallback")

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

            if raw_data is None or len(raw_data) == 0:
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
                    if resampled_data is not None and not len(resampled_data) == 0:
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
                    if df is not None and not len(df) == 0:
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

            # Log returns (more stable for financial data)
            featured_df['close_log_return'] = np.log(featured_df['close'] / featured_df['close'].shift(1))

            # Volume returns (with safe handling for zero volumes)
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

            # VWAP (Volume Weighted Average Price)
            featured_df['vwap'] = self._calculate_vwap(featured_df)
            featured_df['vwap_price_ratio'] = featured_df['close'] / featured_df['vwap']

            # VWAP Momentum features
            featured_df['vwap_momentum_5'] = featured_df['vwap'].pct_change(5)
            featured_df['vwap_momentum_10'] = featured_df['vwap'].pct_change(10)
            featured_df['vwap_momentum_20'] = featured_df['vwap'].pct_change(20)

            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_histogram = self._calculate_macd(featured_df['close'])
            featured_df['macd'] = macd
            featured_df['macd_signal'] = macd_signal
            featured_df['macd_histogram'] = macd_histogram

            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(featured_df)
            featured_df['stoch_k'] = stoch_k
            featured_df['stoch_d'] = stoch_d

            # Williams %R
            featured_df['williams_r'] = self._calculate_williams_r(featured_df)

            # Commodity Channel Index (CCI)
            featured_df['cci'] = self._calculate_cci(featured_df)

            # Average True Range (ATR)
            featured_df['atr'] = self._calculate_atr(featured_df)

            # Average Directional Index (ADX)
            featured_df['adx'] = self._calculate_adx(featured_df)

            # On-Balance Volume (OBV)
            featured_df['obv'] = self._calculate_obv(featured_df)

            # Chaikin Money Flow (CMF)
            featured_df['cmf'] = self._calculate_chaikin_mf(featured_df)

            # Price Volume Trend (PVT)
            featured_df['pvt'] = self._calculate_pvt(featured_df)

            # Rate of Change (ROC)
            featured_df['roc_10'] = self._calculate_roc(featured_df['close'], 10)
            featured_df['roc_20'] = self._calculate_roc(featured_df['close'], 20)

            # Momentum indicators
            featured_df['momentum_10'] = featured_df['close'] / featured_df['close'].shift(10) - 1
            featured_df['momentum_20'] = featured_df['close'] / featured_df['close'].shift(20) - 1

            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                featured_df[f'close_lag_{lag}'] = featured_df['close'].shift(lag)
                featured_df[f'volume_lag_{lag}'] = featured_df['volume'].shift(lag)

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
        middle_band = rolling_mean(prices, window=window) if VECTORBT_AVAILABLE and len(prices) > 1000 else prices.rolling(window=window).mean()
        std = rolling_std(prices, window=window) if VECTORBT_AVAILABLE and len(prices) > 1000 else prices.rolling(window=window).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP) using matrix operations and safe math.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            VWAP series
        """
        try:
            # VWAP = (Price * Volume) / Volume (cumulative)
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0

            # Use matrix operations for element-wise multiplication
            if self.matrix_ops:
                price_volume = typical_price.values * df['volume'].values
            else:
                price_volume = typical_price * df['volume']

            # Vectorized cumulative sums
            cumulative_price_volume = np.cumsum(price_volume)
            cumulative_volume = np.cumsum(df['volume'].values)

            # Safe division using matrix operations if available
            if self.matrix_ops and hasattr(self.matrix_ops, 'safe_correlation_matrix'):
                # Create safe division matrix
                division_matrix = np.column_stack([cumulative_price_volume, cumulative_volume])
                # Use correlation-like operation for safe division (approximation)
                vwap_values = np.divide(cumulative_price_volume, cumulative_volume,
                                       out=np.full_like(cumulative_price_volume, np.nan, dtype=float),
                                       where=(cumulative_volume != 0))
            else:
                # Standard numpy safe division
                vwap_values = np.divide(cumulative_price_volume, cumulative_volume,
                                       out=np.full_like(cumulative_price_volume, np.nan, dtype=float),
                                       where=(cumulative_volume != 0))

            return pd.Series(vwap_values, index=df.index, name='vwap')
        except Exception as e:
            self.logger.warning(f"VWAP calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence) using optimized vectorized operations.

        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (macd, signal, histogram)
        """
        try:
            # Use pandas ewm for EMA calculations (highly optimized)
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

            # Vectorized MACD calculation
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal

            return macd, signal, histogram
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {e}")
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator using vectorized operations and safe math.

        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D period (SMA of %K)

        Returns:
            Tuple of (%K, %D)
        """
        try:
            # Vectorized rolling min/max calculations
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()

            # Safe division for %K calculation using matrix operations if available
            denominator = highest_high - lowest_low
            if self.matrix_ops and hasattr(self.matrix_ops, 'batch_process'):
                # Use safe normalization for %K calculation
                price_range = df['close'] - lowest_low
                stoch_k = 100 * self.matrix_ops.batch_process(
                    np.column_stack([price_range, denominator]),
                    operation='normalize'
                )[:, 0]
            else:
                # Standard numpy safe division
                stoch_k = np.where(
                    denominator != 0,
                    100 * (df['close'] - lowest_low) / denominator,
                    50.0  # Neutral value when denominator is zero
                )

            # Vectorized SMA for %D
            stoch_d = pd.Series(stoch_k, index=df.index).rolling(window=d_period).mean()

            return pd.Series(stoch_k, index=df.index, name='stoch_k'), stoch_d
        except Exception as e:
            self.logger.warning(f"Stochastic calculation failed: {e}")
            nan_series = pd.Series([np.nan] * len(df), index=df.index)
            return nan_series, nan_series

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R using vectorized operations.

        Args:
            df: DataFrame with OHLC data
            period: Lookback period

        Returns:
            Williams %R series
        """
        try:
            # Vectorized rolling calculations
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()

            # Safe division for Williams %R
            denominator = highest_high - lowest_low
            williams_r = np.where(
                denominator != 0,
                -100 * (highest_high - df['close']) / denominator,
                -50.0  # Neutral value when denominator is zero
            )

            return pd.Series(williams_r, index=df.index, name='williams_r')
        except Exception as e:
            self.logger.warning(f"Williams %R calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI) using vectorized operations.

        Args:
            df: DataFrame with OHLC data
            period: CCI period

        Returns:
            CCI series
        """
        try:
            # Vectorized Typical Price calculation
            tp = (df['high'] + df['low'] + df['close']) / 3.0

            # Vectorized SMA of Typical Price
            sma_tp = rolling_mean(tp, window=period) if VECTORBT_AVAILABLE and len(tp) > 1000 else tp.rolling(window=period).mean()

            # Vectorized Mean Deviation (more efficient than apply)
            rolling_mean = rolling_mean(tp, window=period) if VECTORBT_AVAILABLE and len(tp) > 1000 else tp.rolling(window=period).mean()
            rolling_std = rolling_std(tp, window=period) if VECTORBT_AVAILABLE and len(tp) > 1000 else tp.rolling(window=period).std()
            # Approximation: mean deviation ≈ 0.8 * std for normal distribution
            mean_deviation = 0.8 * rolling_std

            # Safe CCI calculation
            denominator = 0.015 * mean_deviation
            cci = np.divide(
                tp - sma_tp,
                denominator,
                out=np.full_like(tp, np.nan, dtype=float),
                where=(denominator != 0)
            )

            return pd.Series(cci, index=df.index, name='cci')
        except Exception as e:
            self.logger.warning(f"CCI calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            ATR series
        """
        try:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR = EMA of True Range
            atr = true_range.ewm(span=period).mean()

            return atr
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        Args:
            df: DataFrame with OHLC data
            period: ADX period

        Returns:
            ADX series
        """
        try:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            plus_dm = df['high'] - df['high'].shift(1)
            minus_dm = df['low'].shift(1) - df['low']

            # Only count positive directional movement
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # Smoothed averages
            atr = true_range.ewm(span=period).mean()
            plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)

            # DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period).mean()

            return adx
        except Exception as e:
            self.logger.warning(f"ADX calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            OBV series
        """
        try:
            obv = pd.Series(0.0, index=df.index)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            return obv
        except Exception as e:
            self.logger.warning(f"OBV calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_chaikin_mf(self, df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Calculate Chaikin Money Flow (CMF).

        Args:
            df: DataFrame with OHLCV data
            period: CMF period

        Returns:
            CMF series
        """
        try:
            # Money Flow Multiplier
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])

            # Money Flow Volume
            mfv = mfm * df['volume']

            # Chaikin Money Flow
            cmf = rolling_sum(mfv, window=period) if VECTORBT_AVAILABLE and len(mfv) > 1000 else mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

            return cmf
        except Exception as e:
            self.logger.warning(f"CMF calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_pvt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend (PVT).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            PVT series
        """
        try:
            # PVT = Previous PVT + (Volume * (Close - Previous Close) / Previous Close)
            price_change = df['close'].pct_change()
            pvt = (price_change * df['volume']).cumsum()

            return pvt
        except Exception as e:
            self.logger.warning(f"PVT calculation failed: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    def _calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change (ROC).

        Args:
            prices: Price series
            period: ROC period

        Returns:
            ROC series
        """
        try:
            # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
            roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
            return roc
        except Exception as e:
            self.logger.warning(f"ROC calculation failed: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)

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

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
