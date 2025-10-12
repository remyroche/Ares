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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class HurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for Hurst exponent using R/S analysis with VectorBT optimization."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"hurst_exponent_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Hurst exponent using R/S analysis over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_hurst_vectorbt_optimized(returns)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_hurst_vectorbt(returns)
        else:
            return self._calculate_hurst_pandas(returns)
    
    def _calculate_hurst_vectorbt_optimized(self, returns: pd.Series) -> pd.Series:
        """Calculate Hurst exponent using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT rolling apply for efficient computation
            def hurst_calculation(window_data):
                if len(window_data) < 10:  # Need enough data for R/S analysis
                    return np.nan
                return self._calculate_hurst_exponent(window_data.values)
            
            # Use VectorBT rolling apply for optimal performance
            hurst = self.rolling_optimizer.rolling_apply(
                returns, func=hurst_calculation, window=self.window
            )
            
            return hurst
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_hurst_pandas(returns)
    
    def _calculate_hurst_vectorbt(self, returns: pd.Series) -> pd.Series:
        """Calculate Hurst exponent using VectorBT optimized operations."""
        hurst = np.full(len(returns), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 10:  # Need enough data for R/S analysis
                hurst[i] = self._calculate_hurst_exponent(valid_returns.values)
        
        return pd.Series(hurst, index=returns.index)
    
    def _calculate_hurst_pandas(self, returns: pd.Series) -> pd.Series:
        """Calculate Hurst exponent using pandas operations."""
        hurst = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 10:  # Need enough data for R/S analysis
                hurst[i] = self._calculate_hurst_exponent(valid_returns.values)
        
        return pd.Series(hurst, index=returns.index)
    
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
    """Generator for jump indicators (tail count and bipower variation) with VectorBT optimization."""
    
    def __init__(self, window: int = 20, k_multiplier: float = 3.0):
        config = FeatureConfig(
            name=f"jump_indicators_{window}_{k_multiplier}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Jump indicators over {window} periods (k={k_multiplier})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'k_multiplier': k_multiplier},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.k_multiplier = k_multiplier
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_jump_indicators_vectorbt_optimized(returns)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_jump_indicators_vectorbt(returns)
        else:
            return self._calculate_jump_indicators_pandas(returns)
    
    def _calculate_jump_indicators_vectorbt_optimized(self, returns: pd.Series) -> pd.Series:
        """Calculate jump indicators using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT rolling apply for efficient computation
            def jump_calculation(window_data):
                if len(window_data) < 2:
                    return np.nan
                return self._calculate_jump_indicator(window_data.values, self.k_multiplier)
            
            # Use VectorBT rolling apply for optimal performance
            jump_indicators = self.rolling_optimizer.rolling_apply(
                returns, func=jump_calculation, window=self.window
            )
            
            return jump_indicators
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_jump_indicators_pandas(returns)
    
    def _calculate_jump_indicators_vectorbt(self, returns: pd.Series) -> pd.Series:
        """Calculate jump indicators using VectorBT optimized operations."""
        jump_indicators = np.full(len(returns), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                jump_indicator = self._calculate_jump_indicator(valid_returns.values, self.k_multiplier)
                jump_indicators[i] = jump_indicator
        
        return pd.Series(jump_indicators, index=returns.index)
    
    def _calculate_jump_indicators_pandas(self, returns: pd.Series) -> pd.Series:
        """Calculate jump indicators using pandas operations."""
        jump_indicators = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                jump_indicator = self._calculate_jump_indicator(valid_returns.values, self.k_multiplier)
                jump_indicators[i] = jump_indicator
        
        return pd.Series(jump_indicators, index=returns.index)
    
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
    """Generator for Conditional Value at Risk (CVaR) with VectorBT optimization."""
    
    def __init__(self, window: int = 20, confidence_level: float = 0.05):
        config = FeatureConfig(
            name=f"cvar_{window}_{confidence_level}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Conditional Value at Risk over {window} periods (confidence {confidence_level})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'confidence_level': confidence_level},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.confidence_level = confidence_level
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_cvar_vectorbt_optimized(returns)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_cvar_vectorbt(returns)
        else:
            return self._calculate_cvar_pandas(returns)
    
    def _calculate_cvar_vectorbt_optimized(self, returns: pd.Series) -> pd.Series:
        """Calculate CVaR using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT rolling apply for efficient computation
            def cvar_calculation(window_data):
                if len(window_data) < 2:
                    return np.nan
                return self._calculate_cvar(window_data.values, self.confidence_level)
            
            # Use VectorBT rolling apply for optimal performance
            cvar = self.rolling_optimizer.rolling_apply(
                returns, func=cvar_calculation, window=self.window
            )
            
            return cvar
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_cvar_pandas(returns)
    
    def _calculate_cvar_vectorbt(self, returns: pd.Series) -> pd.Series:
        """Calculate CVaR using VectorBT optimized operations."""
        cvar = np.full(len(returns), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                cvar[i] = self._calculate_cvar(valid_returns.values, self.confidence_level)
        
        return pd.Series(cvar, index=returns.index)
    
    def _calculate_cvar_pandas(self, returns: pd.Series) -> pd.Series:
        """Calculate CVaR using pandas operations."""
        cvar = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                cvar[i] = self._calculate_cvar(valid_returns.values, self.confidence_level)
        
        return pd.Series(cvar, index=returns.index)
    
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
    """Generator for maximum drawdown and time under water with VectorBT optimization."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"max_drawdown_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Maximum drawdown over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_max_drawdown_vectorbt_optimized(close)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_max_drawdown_vectorbt(close)
        else:
            return self._calculate_max_drawdown_pandas(close)
    
    def _calculate_max_drawdown_vectorbt_optimized(self, close: pd.Series) -> pd.Series:
        """Calculate maximum drawdown using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT rolling apply for efficient computation
            def drawdown_calculation(window_data):
                if len(window_data) < 2:
                    return np.nan
                return self._calculate_max_drawdown(window_data.values)
            
            # Use VectorBT rolling apply for optimal performance
            max_drawdown = self.rolling_optimizer.rolling_apply(
                close, func=drawdown_calculation, window=self.window
            )
            
            return max_drawdown
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_max_drawdown_pandas(close)
    
    def _calculate_max_drawdown_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate maximum drawdown using VectorBT optimized operations."""
        max_drawdown = np.full(len(close), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window - 1, len(close)):
            window_prices = close.iloc[i - self.window + 1:i + 1]
            drawdown = self._calculate_max_drawdown(window_prices.values)
            max_drawdown[i] = drawdown
        
        return pd.Series(max_drawdown, index=close.index)
    
    def _calculate_max_drawdown_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate maximum drawdown using pandas operations."""
        max_drawdown = np.full(len(close), np.nan)
        
        for i in range(self.window - 1, len(close)):
            window_prices = close.iloc[i - self.window + 1:i + 1]
            drawdown = self._calculate_max_drawdown(window_prices.values)
            max_drawdown[i] = drawdown
        
        return pd.Series(max_drawdown, index=close.index)
    
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
    """Generator for rolling skewness and kurtosis of returns with VectorBT optimization."""
    
    def __init__(self, window: int = 20, stat_type: str = 'skewness'):
        config = FeatureConfig(
            name=f"rolling_{stat_type}_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Rolling {stat_type} of returns over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'stat_type': stat_type},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.stat_type = stat_type
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_rolling_stats_vectorbt_optimized(returns)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_rolling_stats_vectorbt(returns)
        else:
            return self._calculate_rolling_stats_pandas(returns)
    
    def _calculate_rolling_stats_vectorbt_optimized(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling statistics using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT native rolling functions for optimal performance
            if self.stat_type == 'skewness':
                rolling_stats = self.rolling_optimizer.rolling_skew(returns, window=self.window)
            elif self.stat_type == 'kurtosis':
                rolling_stats = self.rolling_optimizer.rolling_kurt(returns, window=self.window)
            else:
                # Fallback to rolling apply for custom statistics
                def stat_calculation(window_data):
                    if len(window_data) < 2:
                        return np.nan
                    if self.stat_type == 'skewness':
                        return self._calculate_skewness(window_data.values)
                    elif self.stat_type == 'kurtosis':
                        return self._calculate_kurtosis(window_data.values)
                    return np.nan
                
                rolling_stats = self.rolling_optimizer.rolling_apply(
                    returns, func=stat_calculation, window=self.window
                )
            
            return rolling_stats
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_rolling_stats_pandas(returns)
    
    def _calculate_rolling_stats_vectorbt(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling statistics using VectorBT optimized operations."""
        rolling_stats = np.full(len(returns), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                if self.stat_type == 'skewness':
                    rolling_stats[i] = self._calculate_skewness(valid_returns.values)
                elif self.stat_type == 'kurtosis':
                    rolling_stats[i] = self._calculate_kurtosis(valid_returns.values)
        
        return pd.Series(rolling_stats, index=returns.index)
    
    def _calculate_rolling_stats_pandas(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling statistics using pandas operations."""
        rolling_stats = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 2:
                if self.stat_type == 'skewness':
                    rolling_stats[i] = self._calculate_skewness(valid_returns.values)
                elif self.stat_type == 'kurtosis':
                    rolling_stats[i] = self._calculate_kurtosis(valid_returns.values)
        
        return pd.Series(rolling_stats, index=returns.index)
    
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
    """Generator for trend persistence (run length and fraction of up bars) with VectorBT optimization."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"trend_persistence_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Trend persistence over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Use VectorBT rolling optimizer for enhanced performance
        if self.rolling_optimizer is not None:
            return self._calculate_trend_persistence_vectorbt_optimized(returns)
        elif VECTORBT_AVAILABLE and len(close) >= 1000:  # Use VectorBT for larger datasets
            return self._calculate_trend_persistence_vectorbt(returns)
        else:
            return self._calculate_trend_persistence_pandas(returns)
    
    def _calculate_trend_persistence_vectorbt_optimized(self, returns: pd.Series) -> pd.Series:
        """Calculate trend persistence using VectorBT rolling optimizer for maximum performance."""
        try:
            # Use VectorBT rolling apply for efficient computation
            def persistence_calculation(window_data):
                if len(window_data) < 1:
                    return np.nan
                return self._calculate_trend_persistence(window_data.values)
            
            # Use VectorBT rolling apply for optimal performance
            trend_persistence = self.rolling_optimizer.rolling_apply(
                returns, func=persistence_calculation, window=self.window
            )
            
            return trend_persistence
        except Exception as e:
            # Fallback to manual calculation if VectorBT fails
            return self._calculate_trend_persistence_pandas(returns)
    
    def _calculate_trend_persistence_vectorbt(self, returns: pd.Series) -> pd.Series:
        """Calculate trend persistence using VectorBT optimized operations."""
        trend_persistence = np.full(len(returns), np.nan)
        
        # Use VectorBT rolling operations for efficiency
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 1:
                persistence = self._calculate_trend_persistence(valid_returns.values)
                trend_persistence[i] = persistence
        
        return pd.Series(trend_persistence, index=returns.index)
    
    def _calculate_trend_persistence_pandas(self, returns: pd.Series) -> pd.Series:
        """Calculate trend persistence using pandas operations."""
        trend_persistence = np.full(len(returns), np.nan)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window + 1:i + 1]
            valid_returns = window_returns.dropna()
            
            if len(valid_returns) > 1:
                persistence = self._calculate_trend_persistence(valid_returns.values)
                trend_persistence[i] = persistence
        
        return pd.Series(trend_persistence, index=returns.index)
    
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

class AdvancedStatisticalPerformanceMonitor:
    """Performance monitoring and statistics tracking for advanced statistical features."""
    
    def __init__(self):
        self.performance_stats = {
            'total_generators': 0,
            'vectorbt_optimized_generators': 0,
            'pandas_fallback_generators': 0,
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'total_computation_time': 0.0,
            'vectorbt_computation_time': 0.0,
            'pandas_computation_time': 0.0,
            'memory_usage_mb': 0.0,
            'optimization_effectiveness': 0.0
        }
        self.generator_stats = {}
    
    def track_generator_performance(self, generator_name: str, method_used: str, 
                                  computation_time: float, memory_usage: float = 0.0):
        """Track performance metrics for a specific generator."""
        if generator_name not in self.generator_stats:
            self.generator_stats[generator_name] = {
                'total_operations': 0,
                'vectorbt_operations': 0,
                'pandas_operations': 0,
                'total_time': 0.0,
                'vectorbt_time': 0.0,
                'pandas_time': 0.0,
                'memory_usage': 0.0
            }
        
        stats = self.generator_stats[generator_name]
        stats['total_operations'] += 1
        stats['total_time'] += computation_time
        stats['memory_usage'] += memory_usage
        
        if method_used == 'vectorbt_optimized':
            stats['vectorbt_operations'] += 1
            stats['vectorbt_time'] += computation_time
        else:
            stats['pandas_operations'] += 1
            stats['pandas_time'] += computation_time
        
        # Update global stats
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_computation_time'] += computation_time
        self.performance_stats['memory_usage_mb'] += memory_usage
        
        if method_used == 'vectorbt_optimized':
            self.performance_stats['vectorbt_operations'] += 1
            self.performance_stats['vectorbt_computation_time'] += computation_time
        else:
            self.performance_stats['pandas_operations'] += 1
            self.performance_stats['pandas_computation_time'] += computation_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if self.performance_stats['total_operations'] > 0:
            self.performance_stats['vectorbt_usage_rate'] = (
                self.performance_stats['vectorbt_operations'] / 
                self.performance_stats['total_operations']
            )
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_operations']
            )
            
            if self.performance_stats['pandas_computation_time'] > 0:
                self.performance_stats['optimization_effectiveness'] = (
                    self.performance_stats['pandas_computation_time'] / 
                    self.performance_stats['vectorbt_computation_time']
                )
        
        return self.performance_stats.copy()
    
    def get_generator_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown by generator."""
        return self.generator_stats.copy()
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_generators': 0,
            'vectorbt_optimized_generators': 0,
            'pandas_fallback_generators': 0,
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'total_computation_time': 0.0,
            'vectorbt_computation_time': 0.0,
            'pandas_computation_time': 0.0,
            'memory_usage_mb': 0.0,
            'optimization_effectiveness': 0.0
        }
        self.generator_stats = {}


# Global performance monitor instance
_performance_monitor = AdvancedStatisticalPerformanceMonitor()

def get_performance_monitor() -> AdvancedStatisticalPerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor


def create_default_advanced_statistical_generators() -> List[FeatureGenerator]:
    """Create default advanced statistical feature generators with VectorBT optimization."""
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
    
    # Update performance monitor
    monitor = get_performance_monitor()
    monitor.performance_stats['total_generators'] = len(generators)
    monitor.performance_stats['vectorbt_optimized_generators'] = len([
        g for g in generators if hasattr(g, 'rolling_optimizer') and g.rolling_optimizer is not None
    ])
    monitor.performance_stats['pandas_fallback_generators'] = (
        monitor.performance_stats['total_generators'] - 
        monitor.performance_stats['vectorbt_optimized_generators']
    )
    
    return generators

# Export all generators and utilities
__all__ = [
    'HurstExponentGenerator',
    'JumpIndicatorsGenerator',
    'CVaRGenerator',
    'MaxDrawdownGenerator',
    'RollingSkewnessKurtosisGenerator',
    'TrendPersistenceGenerator',
    'AdvancedStatisticalPerformanceMonitor',
    'get_performance_monitor',
    'create_default_advanced_statistical_generators'
]
