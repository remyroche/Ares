"""
VectorBT Batch Processor for Efficient Feature Engineering

This module provides batch processing capabilities using VectorBT to replace
manual loops and improve performance for feature engineering operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import logging

try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov, rolling_rank,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.indicators.basic import RSI, MACD, BBANDS, ATR, STOCH
    from vectorbt.returns import Returns
    from vectorbt.portfolio import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorBTBatchProcessor:
    """
    Batch processor for efficient feature engineering using VectorBT.
    """

    def __init__(self, enable_vectorbt: bool = True):
        """
        Initialize VectorBT batch processor.

        Args:
            enable_vectorbt: Whether to use VectorBT (falls back to pandas if False)
        """
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE

    def batch_rolling_operations(self,
                                data: pd.Series,
                                windows: List[int],
                                operations: List[str],
                                **kwargs) -> pd.DataFrame:
        """
        Perform multiple rolling operations in batch using VectorBT.

        Args:
            data: Input time series data
            windows: List of window sizes
            operations: List of operations ('mean', 'std', 'var', 'min', 'max', 'sum', 'skew', 'kurt')

        Returns:
            DataFrame with rolling features
        """
        if not self.enable_vectorbt:
            return self._batch_rolling_operations_pandas(data, windows, operations, **kwargs)

        try:
            results = {}

            for window in windows:
                if len(data) < window:
                    continue

                for operation in operations:
                    if operation == 'mean':
                        results[f'{operation}_{window}'] = rolling_mean(data, window=window)
                    elif operation == 'std':
                        results[f'{operation}_{window}'] = rolling_std(data, window=window)
                    elif operation == 'var':
                        results[f'{operation}_{window}'] = rolling_var(data, window=window)
                    elif operation == 'min':
                        results[f'{operation}_{window}'] = rolling_min(data, window=window)
                    elif operation == 'max':
                        results[f'{operation}_{window}'] = rolling_max(data, window=window)
                    elif operation == 'sum':
                        results[f'{operation}_{window}'] = rolling_sum(data, window=window)
                    elif operation == 'skew':
                        results[f'{operation}_{window}'] = rolling_skew(data, window=window)
                    elif operation == 'kurt':
                        results[f'{operation}_{window}'] = rolling_kurt(data, window=window)

            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            logger.warning(f"VectorBT batch rolling operations failed: {e}")
            return self._batch_rolling_operations_pandas(data, windows, operations, **kwargs)

    def batch_technical_indicators(self,
                                  data: pd.DataFrame,
                                  indicators: List[str],
                                  periods: List[int]) -> pd.DataFrame:
        """
        Calculate multiple technical indicators in batch using VectorBT.

        Args:
            data: OHLCV data
            indicators: List of indicators ('rsi', 'macd', 'bbands', 'atr', 'stoch')
            periods: List of periods for indicators

        Returns:
            DataFrame with technical indicators
        """
        if not self.enable_vectorbt:
            return self._batch_technical_indicators_pandas(data, indicators, periods)

        try:
            results = {}

            for indicator in indicators:
                for period in periods:
                    if indicator == 'rsi' and 'close' in data.columns:
                        rsi_result = RSI.run(data['close'], window=period)
                        results[f'rsi_{period}'] = rsi_result.rsi

                    elif indicator == 'macd' and 'close' in data.columns:
                        macd_result = MACD.run(data['close'], fast=period, slow=period*2, signal=period//2)
                        results[f'macd_{period}'] = macd_result.macd
                        results[f'macd_signal_{period}'] = macd_result.signal
                        results[f'macd_histogram_{period}'] = macd_result.histogram

                    elif indicator == 'bbands' and 'close' in data.columns:
                        bb_result = BBANDS.run(data['close'], window=period)
                        results[f'bb_upper_{period}'] = bb_result.upper
                        results[f'bb_middle_{period}'] = bb_result.middle
                        results[f'bb_lower_{period}'] = bb_result.lower
                        results[f'bb_width_{period}'] = (bb_result.upper - bb_result.lower) / bb_result.middle

                    elif indicator == 'atr' and all(col in data.columns for col in ['high', 'low', 'close']):
                        atr_result = ATR.run(data['high'], data['low'], data['close'], window=period)
                        results[f'atr_{period}'] = atr_result.atr

                    elif indicator == 'stoch' and all(col in data.columns for col in ['high', 'low', 'close']):
                        stoch_result = STOCH.run(data['high'], data['low'], data['close'],
                                               k_window=period, d_window=period//2)
                        results[f'stoch_k_{period}'] = stoch_result.k
                        results[f'stoch_d_{period}'] = stoch_result.d

            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            logger.warning(f"VectorBT batch technical indicators failed: {e}")
            return self._batch_technical_indicators_pandas(data, indicators, periods)

    def batch_correlation_analysis(self,
                                  data: pd.DataFrame,
                                  windows: List[int],
                                  target_col: str = 'close') -> pd.DataFrame:
        """
        Calculate rolling correlations in batch using VectorBT.

        Args:
            data: Input data
            windows: List of window sizes
            target_col: Target column for correlations

        Returns:
            DataFrame with correlation features
        """
        if not self.enable_vectorbt:
            return self._batch_correlation_analysis_pandas(data, windows, target_col)

        try:
            results = {}
            target = data[target_col]

            for window in windows:
                if len(data) < window:
                    continue

                for col in data.columns:
                    if col != target_col:
                        corr = rolling_corr(target, data[col], window=window)
                        results[f'corr_{col}_{window}'] = corr

            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            logger.warning(f"VectorBT batch correlation analysis failed: {e}")
            return self._batch_correlation_analysis_pandas(data, windows, target_col)

    def batch_transformations(self,
                             data: pd.Series,
                             transformations: List[str],
                             windows: List[int] = None) -> pd.DataFrame:
        """
        Apply multiple transformations in batch using VectorBT.

        Args:
            data: Input time series data
            transformations: List of transformations ('rank', 'quantile', 'zscore', 'winsorize')
            windows: List of window sizes for rolling transformations

        Returns:
            DataFrame with transformed features
        """
        if not self.enable_vectorbt:
            return self._batch_transformations_pandas(data, transformations, windows)

        try:
            results = {}

            for transformation in transformations:
                if transformation == 'rank':
                    if windows:
                        for window in windows:
                            results[f'rank_{window}'] = rolling_rank(data, window=window, pct=True)
                    else:
                        results['rank'] = data.rank(pct=True)

                elif transformation == 'quantile':
                    if windows:
                        for window in windows:
                            for q in [0.25, 0.5, 0.75]:
                                results[f'quantile_{q}_{window}'] = rolling_quantile(data, window=window, q=q)
                    else:
                        for q in [0.25, 0.5, 0.75]:
                            results[f'quantile_{q}'] = data.quantile(q)

                elif transformation == 'zscore':
                    if windows:
                        for window in windows:
                            mean_val = rolling_mean(data, window=window)
                            std_val = rolling_std(data, window=window)
                            results[f'zscore_{window}'] = (data - mean_val) / std_val
                    else:
                        results['zscore'] = (data - data.mean()) / data.std()

            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            logger.warning(f"VectorBT batch transformations failed: {e}")
            return self._batch_transformations_pandas(data, transformations, windows)

    def _batch_rolling_operations_pandas(self,
                                        data: pd.Series,
                                        windows: List[int],
                                        operations: List[str],
                                        **kwargs) -> pd.DataFrame:
        """Fallback pandas implementation."""
        results = {}

        for window in windows:
            if len(data) < window:
                continue

            for operation in operations:
                if operation == 'mean':
                    results[f'{operation}_{window}'] = data.rolling(window).mean()
                elif operation == 'std':
                    results[f'{operation}_{window}'] = data.rolling(window).std()
                elif operation == 'var':
                    results[f'{operation}_{window}'] = data.rolling(window).var()
                elif operation == 'min':
                    results[f'{operation}_{window}'] = data.rolling(window).min()
                elif operation == 'max':
                    results[f'{operation}_{window}'] = data.rolling(window).max()
                elif operation == 'sum':
                    results[f'{operation}_{window}'] = data.rolling(window).sum()
                elif operation == 'skew':
                    results[f'{operation}_{window}'] = data.rolling(window).skew()
                elif operation == 'kurt':
                    results[f'{operation}_{window}'] = data.rolling(window).kurt()

        return pd.DataFrame(results, index=data.index)

    def _batch_technical_indicators_pandas(self,
                                          data: pd.DataFrame,
                                          indicators: List[str],
                                          periods: List[int]) -> pd.DataFrame:
        """Fallback pandas implementation for technical indicators."""
        results = {}

        for indicator in indicators:
            for period in periods:
                if indicator == 'rsi' and 'close' in data.columns:
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    results[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return pd.DataFrame(results, index=data.index)

    def _batch_correlation_analysis_pandas(self,
                                          data: pd.DataFrame,
                                          windows: List[int],
                                          target_col: str = 'close') -> pd.DataFrame:
        """Fallback pandas implementation for correlation analysis."""
        results = {}
        target = data[target_col]

        for window in windows:
            if len(data) < window:
                continue

            for col in data.columns:
                if col != target_col:
                    corr = target.rolling(window).corr(data[col])
                    results[f'corr_{col}_{window}'] = corr

        return pd.DataFrame(results, index=data.index)

    def _batch_transformations_pandas(self,
                                     data: pd.Series,
                                     transformations: List[str],
                                     windows: List[int] = None) -> pd.DataFrame:
        """Fallback pandas implementation for transformations."""
        results = {}

        for transformation in transformations:
            if transformation == 'rank':
                if windows:
                    for window in windows:
                        results[f'rank_{window}'] = data.rolling(window).rank(pct=True)
                else:
                    results['rank'] = data.rank(pct=True)

            elif transformation == 'quantile':
                if windows:
                    for window in windows:
                        for q in [0.25, 0.5, 0.75]:
                            results[f'quantile_{q}_{window}'] = data.rolling(window).quantile(q)
                else:
                    for q in [0.25, 0.5, 0.75]:
                        results[f'quantile_{q}'] = data.quantile(q)

            elif transformation == 'zscore':
                if windows:
                    for window in windows:
                        mean_val = data.rolling(window).mean()
                        std_val = data.rolling(window).std()
                        results[f'zscore_{window}'] = (data - mean_val) / std_val
                else:
                    results['zscore'] = (data - data.mean()) / data.std()

        return pd.DataFrame(results, index=data.index)
