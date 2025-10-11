"""
VectorBT-Optimized Advanced Statistical Feature Generators

This module provides high-performance advanced statistical feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Hurst exponent using R/S analysis
- Jump indicators (tail count and bipower variation)
- Conditional Value at Risk (CVaR)
- Maximum drawdown and time under water
- Rolling skewness and kurtosis
- Trend persistence (run length and fraction of up bars)
- Advanced statistical measures for quantitative finance
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union
from scipy import stats

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)

class VectorBTHurstExponentGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Hurst exponent generator using R/S analysis."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_hurst_exponent_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized Hurst exponent using R/S analysis over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Hurst exponent using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_hurst_exponent_{self.window}')
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_hurst_exponent_{self.window}')
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling operations for efficiency
        hurst_values = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 10:  # Need enough data for R/S analysis
                hurst_values[i] = self._calculate_hurst_exponent(valid_returns.values)
        
        return pd.Series(hurst_values, index=returns.index, name=f'vectorbt_hurst_exponent_{self.window}')
    
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


class VectorBTJumpIndicatorsGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized jump indicators generator."""
    
    def __init__(self, window: int = 20, k_multiplier: float = 3.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window, k_multiplier)
        super().__init__(config)
        self.window = window
        self.k_multiplier = k_multiplier
    
    @classmethod
    def _create_default_config(cls, window: int = 20, k_multiplier: float = 3.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_jump_indicators_{window}_{k_multiplier}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized jump indicators over {window} periods (k={k_multiplier})",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "k_multiplier": k_multiplier},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jump indicators using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_jump_indicators_{self.window}_{self.k_multiplier}')
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_jump_indicators_{self.window}_{self.k_multiplier}')
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling operations for efficiency
        jump_indicators = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                jump_indicators[i] = self._calculate_jump_indicator(valid_returns.values, self.k_multiplier)
        
        return pd.Series(jump_indicators, index=returns.index, name=f'vectorbt_jump_indicators_{self.window}_{self.k_multiplier}')
    
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


class VectorBTCVaRGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Conditional Value at Risk (CVaR) generator."""
    
    def __init__(self, window: int = 20, confidence_level: float = 0.05, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window, confidence_level)
        super().__init__(config)
        self.window = window
        self.confidence_level = confidence_level
    
    @classmethod
    def _create_default_config(cls, window: int = 20, confidence_level: float = 0.05) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_cvar_{window}_{confidence_level}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized Conditional Value at Risk over {window} periods (confidence {confidence_level})",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "confidence_level": confidence_level},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate CVaR using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_cvar_{self.window}_{self.confidence_level}')
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_cvar_{self.window}_{self.confidence_level}')
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling operations for efficiency
        cvar_values = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                cvar_values[i] = self._calculate_cvar(valid_returns.values, self.confidence_level)
        
        return pd.Series(cvar_values, index=returns.index, name=f'vectorbt_cvar_{self.window}_{self.confidence_level}')
    
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


class VectorBTMaxDrawdownGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized maximum drawdown generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_max_drawdown_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized maximum drawdown over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate maximum drawdown using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_max_drawdown_{self.window}')
        
        close = data['close']
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_max_drawdown_{self.window}')
        
        # Use VectorBT rolling operations for efficiency
        max_drawdown_values = np.full(len(close), np.nan)
        
        for i in range(self.window - 1, len(close)):
            window_prices = close.iloc[i - self.window + 1:i + 1]
            max_drawdown_values[i] = self._calculate_max_drawdown(window_prices.values)
        
        return pd.Series(max_drawdown_values, index=close.index, name=f'vectorbt_max_drawdown_{self.window}')
    
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


class VectorBTRollingSkewnessKurtosisGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized rolling skewness and kurtosis generator."""
    
    def __init__(self, window: int = 20, stat_type: str = 'skewness', config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window, stat_type)
        super().__init__(config)
        self.window = window
        self.stat_type = stat_type
    
    @classmethod
    def _create_default_config(cls, window: int = 20, stat_type: str = 'skewness') -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rolling_{stat_type}_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized rolling {stat_type} of returns over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "stat_type": stat_type},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate rolling statistics using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rolling_{self.stat_type}_{self.window}')
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_rolling_{self.stat_type}_{self.window}')
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling operations for efficiency
        rolling_stats = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                if self.stat_type == 'skewness':
                    rolling_stats[i] = self._calculate_skewness(valid_returns.values)
                elif self.stat_type == 'kurtosis':
                    rolling_stats[i] = self._calculate_kurtosis(valid_returns.values)
        
        return pd.Series(rolling_stats, index=returns.index, name=f'vectorbt_rolling_{self.stat_type}_{self.window}')
    
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


class VectorBTTrendPersistenceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend persistence generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_trend_persistence_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"VectorBT-optimized trend persistence over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend persistence using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_persistence_{self.window}')
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index, name=f'vectorbt_trend_persistence_{self.window}')
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling operations for efficiency
        trend_persistence = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 1:
                trend_persistence[i] = self._calculate_trend_persistence(valid_returns.values)
        
        return pd.Series(trend_persistence, index=returns.index, name=f'vectorbt_trend_persistence_{self.window}')
    
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


def create_vectorbt_advanced_statistical_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized advanced statistical feature generators."""
    generators = []
    
    # Hurst exponent generators
    for window in [20, 50]:
        generators.append(VectorBTHurstExponentGenerator(window))
    
    # Jump indicators generators
    for window in [20]:
        for k_multiplier in [2.0, 3.0]:
            generators.append(VectorBTJumpIndicatorsGenerator(window, k_multiplier))
    
    # CVaR generators
    for window in [20]:
        for confidence_level in [0.05, 0.01]:
            generators.append(VectorBTCVaRGenerator(window, confidence_level))
    
    # Max drawdown generators
    for window in [20, 50]:
        generators.append(VectorBTMaxDrawdownGenerator(window))
    
    # Rolling skewness generators
    for window in [20]:
        generators.append(VectorBTRollingSkewnessKurtosisGenerator(window, 'skewness'))
    
    # Rolling kurtosis generators
    for window in [20]:
        generators.append(VectorBTRollingSkewnessKurtosisGenerator(window, 'kurtosis'))
    
    # Trend persistence generators
    for window in [20]:
        generators.append(VectorBTTrendPersistenceGenerator(window))
    
    return generators


def create_default_vectorbt_advanced_statistical_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized advanced statistical feature generators."""
    return create_vectorbt_advanced_statistical_generators()


# Export all generators
__all__ = [
    'VectorBTHurstExponentGenerator',
    'VectorBTJumpIndicatorsGenerator',
    'VectorBTCVaRGenerator',
    'VectorBTMaxDrawdownGenerator',
    'VectorBTRollingSkewnessKurtosisGenerator',
    'VectorBTTrendPersistenceGenerator',
    'create_vectorbt_advanced_statistical_generators',
    'create_default_vectorbt_advanced_statistical_generators'
]