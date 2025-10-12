"""
Refactored Momentum Feature Generator

This module provides refactored momentum feature generators that use centralized
utilities to eliminate code duplication and improve performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.unified_feature_generator import UnifiedFeatureGenerator, UnifiedFeatureConfig
from ..core.feature_generator import FeatureResult, FeatureCategory
from ..utils.centralized_rolling_manager import get_centralized_rolling_manager, RollingOperation
from ..utils.scaler_factory import get_scaler_factory, ScalerType
from ..utils.common_operations import get_common_operations

logger = logging.getLogger(__name__)

class RefactoredMomentumFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored momentum feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Refactored momentum features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "rsi_periods": [14],
                "macd_fast": [12],
                "macd_slow": [26],
                "stochastic_periods": [14],
                "momentum_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Calculate momentum using centralized rolling operations
        close_prices = data['close']
        momentum = self.rolling_mean(close_prices, window=20)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            momentum = self.normalize_feature(momentum, feature_type='momentum')
        
        return momentum.rename('refactored_momentum_20')

class RefactoredRSIGenerator(UnifiedFeatureGenerator):
    """Refactored RSI generator using centralized utilities."""
    
    def __init__(self, period: int = 14, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored RSI with period {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='oscillator',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use centralized rolling operations
        avg_gain = self.rolling_mean(gain, window=self.period)
        avg_loss = self.rolling_mean(loss, window=self.period)
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            rsi = self.normalize_feature(rsi, feature_type='oscillator')
        
        return rsi.rename(f'refactored_rsi_{self.period}')

class RefactoredMACDGenerator(UnifiedFeatureGenerator):
    """Refactored MACD generator using centralized utilities."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast_period, slow_period, signal_period)
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    @classmethod
    def _create_default_config(cls, fast_period: int, slow_period: int, signal_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_macd_{fast_period}_{slow_period}_{signal_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored MACD using centralized utilities",
            required_columns=["close"],
            default_lookback=slow_period + signal_period,
            min_lookback=slow_period + signal_period,
            max_lookback=100,
            parameters={"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate EMAs using centralized operations
        ema_fast = close.ewm(span=self.fast_period).mean()
        ema_slow = close.ewm(span=self.slow_period).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Return MACD line as primary feature
        result = macd_line.rename(f'refactored_macd_{self.fast_period}_{self.slow_period}')
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            result = self.normalize_feature(result, feature_type='momentum')
        
        return result

class RefactoredStochasticGenerator(UnifiedFeatureGenerator):
    """Refactored Stochastic generator using centralized utilities."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(k_period, d_period)
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
    
    @classmethod
    def _create_default_config(cls, k_period: int, d_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Stochastic using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=k_period + d_period,
            min_lookback=k_period + d_period,
            max_lookback=100,
            parameters={"k_period": k_period, "d_period": d_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='oscillator',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use centralized rolling operations
        lowest_low = self.rolling_min(low, window=self.k_period)
        highest_high = self.rolling_max(high, window=self.k_period)
        
        # Calculate %K
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (smoothed %K)
        d_percent = self.rolling_mean(k_percent, window=self.d_period)
        
        # Return %K as primary feature
        result = k_percent.rename(f'refactored_stochastic_k_{self.k_period}')
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            result = self.normalize_feature(result, feature_type='oscillator')
        
        return result

class RefactoredWilliamsRGenerator(UnifiedFeatureGenerator):
    """Refactored Williams %R generator using centralized utilities."""
    
    def __init__(self, period: int = 14, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_williams_r_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Williams %R with period {period} using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='oscillator',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Williams %R using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use centralized rolling operations
        highest_high = self.rolling_max(high, window=self.period)
        lowest_low = self.rolling_min(low, window=self.period)
        
        # Calculate Williams %R
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            williams_r = self.normalize_feature(williams_r, feature_type='oscillator')
        
        return williams_r.rename(f'refactored_williams_r_{self.period}')

class RefactoredMomentumOscillatorGenerator(UnifiedFeatureGenerator):
    """Refactored Momentum Oscillator generator using centralized utilities."""
    
    def __init__(self, period: int = 10, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_momentum_oscillator_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Momentum Oscillator with period {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Momentum Oscillator using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate momentum
        momentum = close - close.shift(self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            momentum = self.normalize_feature(momentum, feature_type='momentum')
        
        return momentum.rename(f'refactored_momentum_oscillator_{self.period}')

class RefactoredRateOfChangeGenerator(UnifiedFeatureGenerator):
    """Refactored Rate of Change generator using centralized utilities."""
    
    def __init__(self, period: int = 10, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_rate_of_change_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Rate of Change with period {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Rate of Change using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate rate of change
        roc = (close / close.shift(self.period) - 1) * 100
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            roc = self.normalize_feature(roc, feature_type='momentum')
        
        return roc.rename(f'refactored_rate_of_change_{self.period}')

class RefactoredVectorBTMomentumFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored VectorBT momentum feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_vectorbt_momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Refactored VectorBT momentum features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "rsi_periods": [14],
                "macd_fast": [12],
                "macd_slow": [26],
                "stochastic_periods": [14],
                "momentum_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive momentum features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Use common operations for comprehensive momentum calculation
        momentum_indicators = self.common_operations.calculate_momentum_indicators(data, window=14)
        
        # Combine momentum indicators
        if momentum_indicators:
            # Use ROC as primary feature
            primary_feature = momentum_indicators.get('roc', pd.Series(dtype=float, index=data.index))
            result = primary_feature.rename('refactored_vectorbt_momentum')
            
            # Apply normalization if enabled
            if self.unified_config.auto_normalize:
                result = self.normalize_feature(result, feature_type='momentum')
            
            return result
        else:
            # Fallback to simple momentum calculation
            close = data['close']
            momentum = close - close.shift(20)
            
            if self.unified_config.auto_normalize:
                momentum = self.normalize_feature(momentum, feature_type='momentum')
            
            return momentum.rename('refactored_vectorbt_momentum_fallback')

# Batch momentum generator for multiple features
class RefactoredBatchMomentumGenerator(UnifiedFeatureGenerator):
    """Refactored batch momentum generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_batch_momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Refactored batch momentum features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "rsi_periods": [14, 21],
                "macd_fast": [12],
                "macd_slow": [26],
                "stochastic_periods": [14],
                "momentum_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='momentum',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch momentum features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Create batch configuration for multiple momentum features
        batch_configs = [
            {'name': 'rsi_14', 'operation': 'technical_indicator', 'indicator': 'rsi', 'params': {'period': 14}},
            {'name': 'macd_12_26', 'operation': 'technical_indicator', 'indicator': 'macd', 'params': {'fast_period': 12, 'slow_period': 26}},
            {'name': 'stochastic_14', 'operation': 'technical_indicator', 'indicator': 'stochastic', 'params': {'k_period': 14}},
            {'name': 'momentum_20', 'operation': 'rolling_mean', 'column': 'close', 'params': {'window': 20}}
        ]
        
        # Process in batch
        batch_results = self.batch_process_features(data, batch_configs)
        
        if batch_results:
            # Combine results (use RSI as primary feature)
            primary_feature = batch_results.get('rsi_14', pd.Series(dtype=float, index=data.index))
            result = primary_feature.rename('refactored_batch_momentum')
            
            # Apply normalization if enabled
            if self.unified_config.auto_normalize:
                result = self.normalize_feature(result, feature_type='momentum')
            
            return result
        else:
            # Fallback to simple momentum
            close = data['close']
            momentum = close - close.shift(20)
            
            if self.unified_config.auto_normalize:
                momentum = self.normalize_feature(momentum, feature_type='momentum')
            
            return momentum.rename('refactored_batch_momentum_fallback')