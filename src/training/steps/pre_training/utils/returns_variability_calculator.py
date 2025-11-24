"""
Returns Variability Calculator

This module provides utilities for calculating returns variability distribution
statistics for use in comprehensive reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class ReturnsVariabilityCalculator:
    """
    Calculator for returns variability distribution statistics.

    This class provides methods to calculate comprehensive statistics about
    returns variability, including distribution metrics, percentiles, and
    volatility regime classification.
    """

    def __init__(self):
        """Initialize the returns variability calculator."""
        pass

    def calculate_returns_variability(
        self,
        data: pd.DataFrame,
        returns_column: str = 'close_return',
        window: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate returns variability distribution statistics.

        Args:
            data: DataFrame containing returns data
            returns_column: Column name containing returns
            window: Rolling window for volatility calculation

        Returns:
            Dictionary containing variability statistics
        """
        if returns_column not in data.columns:
            return self._empty_result(f"Column '{returns_column}' not found in data")

        returns = data[returns_column].dropna()

        if len(returns) == 0:
            return self._empty_result("No valid returns data available")

        # Calculate rolling volatility (returns variability)
        volatility = returns.rolling(window=window).std()
        valid_volatility = volatility.dropna()

        if len(valid_volatility) == 0:
            return self._empty_result(f"Insufficient data for window size {window}")

        # Basic statistics
        mean_vol = float(np.mean(valid_volatility))
        std_vol = float(np.std(valid_volatility, ddof=1))
        median_vol = float(np.median(valid_volatility))
        min_vol = float(np.min(valid_volatility))
        max_vol = float(np.max(valid_volatility))

        # Percentiles
        percentiles = {
            'p5': float(np.percentile(valid_volatility, 5)),
            'p25': float(np.percentile(valid_volatility, 25)),
            'p50': float(np.percentile(valid_volatility, 50)),
            'p75': float(np.percentile(valid_volatility, 75)),
            'p95': float(np.percentile(valid_volatility, 95))
        }

        # Distribution shape
        skewness = float(pd.Series(valid_volatility).skew())
        kurtosis = float(pd.Series(valid_volatility).kurtosis())

        # Range and spread metrics
        iqr = percentiles['p75'] - percentiles['p25']
        range_val = max_vol - min_vol

        # Coefficient of variation
        if mean_vol != 0:
            cv = std_vol / mean_vol
        else:
            cv = 'inf'

        # Volatility regime classification
        volatility_regime = self._classify_volatility_regime(
            mean_vol, std_vol, percentiles
        )

        # Interpretation
        interpretation = self._generate_interpretation(
            mean_vol, std_vol, skewness, kurtosis, volatility_regime
        )

        return {
            'mean': mean_vol,
            'std': std_vol,
            'median': median_vol,
            'min': min_vol,
            'max': max_vol,
            'percentiles': percentiles,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'iqr': iqr,
            'range': range_val,
            'coefficient_of_variation': cv,
            'volatility_regime': volatility_regime,
            'interpretation': interpretation,
            'sample_size': len(valid_volatility),
            'window_size': window
        }

    def calculate_returns_variability_4h(
        self,
        data: pd.DataFrame,
        resample_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate returns variability distribution for 4h timeframe.

        Args:
            data: DataFrame containing price/returns data
            resample_from: Original timeframe to resample from (e.g., '15m', '1h')
                          If None, assumes data is already at 4h timeframe

        Returns:
            Dictionary containing 4h returns variability statistics
        """
        # If resampling is needed
        if resample_from is not None:
            data_4h = self._resample_to_4h(data, resample_from)
        else:
            data_4h = data.copy()

        # Calculate returns if not present
        if 'close_return' not in data_4h.columns and 'close' in data_4h.columns:
            data_4h['close_return'] = data_4h['close'].pct_change()

        # Calculate variability
        return self.calculate_returns_variability(
            data_4h,
            returns_column='close_return',
            window=20  # 20 periods at 4h = ~3.3 days
        )

    def _resample_to_4h(
        self,
        data: pd.DataFrame,
        from_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to 4h timeframe.

        Args:
            data: DataFrame with datetime index
            from_timeframe: Original timeframe

        Returns:
            Resampled DataFrame at 4h timeframe
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for resampling")

        # Resample OHLCV data
        resampled = data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in data.columns else 'sum'
        })

        # Remove rows with NaN close prices
        resampled = resampled.dropna(subset=['close'])

        return resampled

    def _classify_volatility_regime(
        self,
        mean_vol: float,
        std_vol: float,
        percentiles: Dict[str, float]
    ) -> str:
        """
        Classify the volatility regime based on statistics.

        Args:
            mean_vol: Mean volatility
            std_vol: Standard deviation of volatility
            percentiles: Percentile values

        Returns:
            Volatility regime classification
        """
        # Use coefficient of variation as primary indicator
        if mean_vol != 0:
            cv = std_vol / mean_vol
        else:
            return 'UNKNOWN'

        # Use percentiles to determine regime
        p95 = percentiles['p95']
        p50 = percentiles['p50']

        if p95 < 0.01:  # Very low volatility
            return 'LOW'
        elif p95 < 0.02 and p50 < 0.01:  # Normal volatility
            return 'NORMAL'
        elif p95 < 0.05:  # Elevated volatility
            return 'HIGH'
        else:  # Extreme volatility
            return 'EXTREME'

    def _generate_interpretation(
        self,
        mean_vol: float,
        std_vol: float,
        skewness: float,
        kurtosis: float,
        regime: str
    ) -> str:
        """
        Generate human-readable interpretation of volatility metrics.

        Args:
            mean_vol: Mean volatility
            std_vol: Standard deviation of volatility
            skewness: Distribution skewness
            kurtosis: Distribution kurtosis
            regime: Volatility regime

        Returns:
            Interpretation string
        """
        interpretations = []

        # Regime interpretation
        regime_text = {
            'LOW': 'Market exhibits low volatility, suggesting stable conditions',
            'NORMAL': 'Market shows normal volatility levels',
            'HIGH': 'Elevated volatility detected, indicating increased market activity',
            'EXTREME': 'Extreme volatility present, suggesting high market uncertainty'
        }
        interpretations.append(regime_text.get(regime, 'Unknown regime'))

        # Skewness interpretation
        if abs(skewness) > 1.0:
            if skewness > 0:
                interpretations.append('Distribution is right-skewed (positive tail)')
            else:
                interpretations.append('Distribution is left-skewed (negative tail)')

        # Kurtosis interpretation
        if kurtosis > 3.0:
            interpretations.append('Fat-tailed distribution (higher than normal peak)')
        elif kurtosis < -1.0:
            interpretations.append('Thin-tailed distribution (flatter than normal)')

        return '. '.join(interpretations) + '.'

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """
        Return an empty result with error reason.

        Args:
            reason: Reason for empty result

        Returns:
            Empty result dictionary
        """
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'percentiles': {
                'p5': 0.0,
                'p25': 0.0,
                'p50': 0.0,
                'p75': 0.0,
                'p95': 0.0
            },
            'skewness': 0.0,
            'kurtosis': 0.0,
            'iqr': 0.0,
            'range': 0.0,
            'coefficient_of_variation': 'N/A',
            'volatility_regime': 'UNKNOWN',
            'interpretation': reason,
            'sample_size': 0,
            'window_size': 0
        }


def calculate_returns_variability_4h(
    data: pd.DataFrame,
    resample_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate returns variability at 4h timeframe.

    Args:
        data: DataFrame containing price/returns data
        resample_from: Original timeframe to resample from

    Returns:
        Dictionary containing 4h returns variability statistics
    """
    calculator = ReturnsVariabilityCalculator()
    return calculator.calculate_returns_variability_4h(data, resample_from)
