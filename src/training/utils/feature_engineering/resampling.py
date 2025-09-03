"""Optimized data resampling utilities for feature engineering.

This module provides efficient resampling functionality for time series data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit, prange

from src.utils.logger import system_logger


class OptimizedResampler:
    """Optimized resampling with Numba JIT compilation and efficient memory usage."""
    
    def __init__(self):
        self.logger = system_logger.getChild("OptimizedResampler")
        
    @staticmethod
    @njit(parallel=True)
    def _resample_ohlcv_numba(
        timestamps: np.ndarray,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volumes: np.ndarray,
        bin_edges: np.ndarray
    ) -> tuple:
        """Numba-optimized OHLCV resampling."""
        n_bins = len(bin_edges) - 1
        
        # Initialize output arrays
        out_open = np.empty(n_bins)
        out_high = np.empty(n_bins)
        out_low = np.empty(n_bins)
        out_close = np.empty(n_bins)
        out_volume = np.empty(n_bins)
        out_count = np.zeros(n_bins, dtype=np.int32)
        
        # Process each bin in parallel
        for i in prange(n_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            # Find data points in this bin
            mask = (timestamps >= bin_start) & (timestamps < bin_end)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # OHLC aggregation
                out_open[i] = open_prices[indices[0]]
                out_high[i] = np.max(high_prices[indices])
                out_low[i] = np.min(low_prices[indices])
                out_close[i] = close_prices[indices[-1]]
                out_volume[i] = np.sum(volumes[indices])
                out_count[i] = len(indices)
            else:
                # No data in this bin - forward fill
                if i > 0:
                    out_open[i] = out_close[i-1]
                    out_high[i] = out_close[i-1]
                    out_low[i] = out_close[i-1]
                    out_close[i] = out_close[i-1]
                    out_volume[i] = 0.0
                else:
                    # First bin with no data
                    out_open[i] = np.nan
                    out_high[i] = np.nan
                    out_low[i] = np.nan
                    out_close[i] = np.nan
                    out_volume[i] = 0.0
                    
        return out_open, out_high, out_low, out_close, out_volume, out_count
    
    def resample_data(
        self,
        data: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str
    ) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe.
        
        Args:
            data: DataFrame with OHLCV data
            source_timeframe: Source timeframe (e.g., '1m')
            target_timeframe: Target timeframe (e.g., '5m')
            
        Returns:
            Resampled DataFrame
        """
        if source_timeframe == target_timeframe:
            return data.copy()
        
        self.logger.info(f"Resampling from {source_timeframe} to {target_timeframe}")
        
        # Convert to numpy arrays for performance
        timestamps = data.index.values.astype(np.int64) // 10**9  # Convert to seconds
        
        # Create bin edges based on target timeframe
        freq_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        
        bin_size = freq_map.get(target_timeframe, 300)
        bin_edges = np.arange(
            timestamps[0] - (timestamps[0] % bin_size),
            timestamps[-1] + bin_size,
            bin_size
        )
        
        # Perform resampling
        out_open, out_high, out_low, out_close, out_volume, out_count = \
            self._resample_ohlcv_numba(
                timestamps,
                data['open'].values,
                data['high'].values,
                data['low'].values,
                data['close'].values,
                data['volume'].values,
                bin_edges
            )
        
        # Create output DataFrame
        result_timestamps = pd.to_datetime(bin_edges[:-1], unit='s')
        
        result = pd.DataFrame({
            'open': out_open,
            'high': out_high,
            'low': out_low,
            'close': out_close,
            'volume': out_volume,
            'count': out_count
        }, index=result_timestamps)
        
        # Remove rows with no data
        result = result[result['count'] > 0].drop(columns=['count'])
        
        # Forward fill any remaining NaN values
        result = result.fillna(method='ffill')
        
        self.logger.info(
            f"Resampled {len(data)} rows to {len(result)} rows "
            f"({target_timeframe} timeframe)"
        )
        
        return result
    
    def create_multi_timeframe_features(
        self,
        data: pd.DataFrame,
        base_timeframe: str,
        target_timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Create features for multiple timeframes.
        
        Args:
            data: Base DataFrame with OHLCV data
            base_timeframe: Base timeframe
            target_timeframes: List of target timeframes
            
        Returns:
            Dictionary mapping timeframe to resampled data
        """
        multi_tf_data = {base_timeframe: data}
        
        for tf in target_timeframes:
            if tf != base_timeframe:
                multi_tf_data[tf] = self.resample_data(data, base_timeframe, tf)
        
        return multi_tf_data