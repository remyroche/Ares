"""
Advanced Statistical Feature Generator

This module provides feature generators for advanced statistical indicators,
including Hurst exponent, jump indicators, CVaR, drawdown measures, and other
sophisticated statistical features for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

class HurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for Hurst exponent using R/S analysis."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"hurst_exponent_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Hurst exponent using R/S analysis over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate Hurst exponent
        hurst = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 10:  # Need enough data for R/S analysis
                hurst[i] = self._calculate_hurst_exponent(valid_returns)
        
        return pd.Series(hurst, index=data.index)
    
    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            n = len(returns)
            if n < 4:
                return 0.5
            
            # Calculate mean
            mean_return = np.mean(returns)
            
            # Calculate deviations from mean
            deviations = returns - mean_return
            
            # Calculate cumulative deviations
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns, ddof=1)
            
            if S == 0:
                return 0.5
            
            # R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent approximation
            # H = log(R/S) / log(n)
            if rs_ratio > 0:
                hurst = np.log(rs_ratio) / np.log(n)
                return np.clip(hurst, 0.0, 1.0)
            else:
                return 0.5
        except:
            return 0.5

class JumpIndicatorsGenerator(VectorizedFeatureGenerator):
    """Generator for jump indicators (tail count and bipower variation)."""
    
    def __init__(self, window: int = 20, k_multiplier: float = 3.0):
        config = FeatureConfig(
            name=f"jump_indicators_{window}_{k_multiplier}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Jump indicators over {window} periods (k={k_multiplier})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window, 'k_multiplier': k_multiplier},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.k_multiplier = k_multiplier
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate jump indicators
        jump_indicators = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 2:
                # Calculate jump indicator
                jump_indicator = self._calculate_jump_indicator(valid_returns, self.k_multiplier)
                jump_indicators[i] = jump_indicator
        
        return pd.Series(jump_indicators, index=data.index)
    
    def _calculate_jump_indicator(self, returns: np.ndarray, k: float) -> float:
        """Calculate jump indicator using tail count method."""
        try:
            # Calculate standard deviation
            sigma = np.std(returns, ddof=1)
            if sigma == 0:
                return 0.0
            
            # Count returns beyond k*sigma
            threshold = k * sigma
            tail_count = np.sum(np.abs(returns) > threshold)
            
            # Normalize by window size
            jump_indicator = tail_count / len(returns)
            
            return jump_indicator
        except:
            return 0.0

class CVaRGenerator(VectorizedFeatureGenerator):
    """Generator for Conditional Value at Risk (CVaR)."""
    
    def __init__(self, window: int = 20, confidence_level: float = 0.05):
        config = FeatureConfig(
            name=f"cvar_{window}_{confidence_level}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Conditional Value at Risk over {window} periods (confidence {confidence_level})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window, 'confidence_level': confidence_level},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.confidence_level = confidence_level
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate CVaR
        cvar = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 2:
                cvar[i] = self._calculate_cvar(valid_returns, self.confidence_level)
        
        return pd.Series(cvar, index=data.index)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate VaR (Value at Risk)
            var_index = int(confidence_level * len(sorted_returns))
            var = sorted_returns[var_index]
            
            # Calculate CVaR (average of returns below VaR)
            tail_returns = sorted_returns[:var_index]
            if len(tail_returns) > 0:
                cvar = np.mean(tail_returns)
            else:
                cvar = var
            
            return cvar
        except:
            return 0.0

class MaxDrawdownGenerator(VectorizedFeatureGenerator):
    """Generator for maximum drawdown and time under water."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"max_drawdown_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Maximum drawdown over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate maximum drawdown
        max_drawdown = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_prices = close[i - self.window + 1:i + 1]
            drawdown = self._calculate_max_drawdown(window_prices)
            max_drawdown[i] = drawdown
        
        return pd.Series(max_drawdown, index=data.index)
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            # Calculate running maximum
            running_max = np.maximum.accumulate(prices)
            
            # Calculate drawdown
            drawdown = (prices - running_max) / running_max
            
            # Return maximum drawdown (most negative)
            max_dd = np.min(drawdown)
            
            return max_dd
        except:
            return 0.0

class RollingSkewnessKurtosisGenerator(VectorizedFeatureGenerator):
    """Generator for rolling skewness and kurtosis of returns."""
    
    def __init__(self, window: int = 20, stat_type: str = 'skewness'):
        config = FeatureConfig(
            name=f"rolling_{stat_type}_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Rolling {stat_type} of returns over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window, 'stat_type': stat_type},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.stat_type = stat_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate rolling statistics
        rolling_stats = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 2:
                if self.stat_type == 'skewness':
                    rolling_stats[i] = self._calculate_skewness(valid_returns)
                elif self.stat_type == 'kurtosis':
                    rolling_stats[i] = self._calculate_kurtosis(valid_returns)
        
        return pd.Series(rolling_stats, index=data.index)
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return == 0:
                return 0.0
            
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
            return skewness
        except:
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return == 0:
                return 0.0
            
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return kurtosis
        except:
            return 0.0

class TrendPersistenceGenerator(VectorizedFeatureGenerator):
    """Generator for trend persistence (run length and fraction of up bars)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"trend_persistence_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Trend persistence over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate trend persistence
        trend_persistence = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                persistence = self._calculate_trend_persistence(valid_returns)
                trend_persistence[i] = persistence
        
        return pd.Series(trend_persistence, index=data.index)
    
    def _calculate_trend_persistence(self, returns: np.ndarray) -> float:
        """Calculate trend persistence metrics."""
        try:
            # Calculate signs
            signs = np.sign(returns)
            
            # Calculate fraction of up bars
            up_fraction = np.sum(signs > 0) / len(signs)
            
            # Calculate average run length
            run_lengths = []
            current_run = 1
            current_sign = signs[0]
            
            for i in range(1, len(signs)):
                if signs[i] == current_sign:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
                    current_sign = signs[i]
            
            run_lengths.append(current_run)
            avg_run_length = np.mean(run_lengths) if run_lengths else 1.0
            
            # Combine metrics (normalized)
            persistence = (up_fraction - 0.5) * avg_run_length
            
            return persistence
        except:
            return 0.0

def create_default_advanced_statistical_generators() -> List[FeatureGenerator]:
    """Create default advanced statistical feature generators."""
    generators = []
    
    # Hurst exponent generators
    for window in [20, 50]:
        generators.append(HurstExponentGenerator(window))
    
    # Jump indicators generators
    for window in [20]:
        for k_multiplier in [2.0, 3.0]:
            generators.append(JumpIndicatorsGenerator(window, k_multiplier))
    
    # CVaR generators
    for window in [20]:
        for confidence_level in [0.05, 0.01]:
            generators.append(CVaRGenerator(window, confidence_level))
    
    # Max drawdown generators
    for window in [20, 50]:
        generators.append(MaxDrawdownGenerator(window))
    
    # Rolling skewness generators
    for window in [20]:
        generators.append(RollingSkewnessKurtosisGenerator(window, 'skewness'))
    
    # Rolling kurtosis generators
    for window in [20]:
        generators.append(RollingSkewnessKurtosisGenerator(window, 'kurtosis'))
    
    # Trend persistence generators
    for window in [20]:
        generators.append(TrendPersistenceGenerator(window))
    
    return generators

# Export all generators
__all__ = [
    'HurstExponentGenerator',
    'JumpIndicatorsGenerator',
    'CVaRGenerator',
    'MaxDrawdownGenerator',
    'RollingSkewnessKurtosisGenerator',
    'TrendPersistenceGenerator',
    'create_default_advanced_statistical_generators'
]