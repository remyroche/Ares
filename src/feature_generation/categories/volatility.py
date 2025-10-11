"""
Volatility Feature Generator

This module provides feature generators for volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Supports different base calculations: price returns, returns-based VWAP, etc.

Enhanced with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class VolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility-based features with batch processing and optimization."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None, base_calculation: Optional[BaseCalculationType] = None):
        self.period = period
        self.base_calculation = base_calculation
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Volatility measure over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                "period": period
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'volatility_{self.period}')

        close_prices = data['close'].astype(float).values
        state = self.get_state()
        history = state.get('close_history') or []

        if history:
            try:
                history_array = np.asarray(history, dtype=float)
            except Exception:
                history_array = np.array(history, dtype=float)
            combined_closes = np.concatenate([history_array, close_prices])
        else:
            combined_closes = close_prices

        combined_volatility = self._calculate_volatility(combined_closes, period=self.period)
        volatility = combined_volatility[-len(close_prices):] if len(close_prices) else np.array([])

        return pd.Series(volatility, index=data.index, name=f'volatility_{self.period}')
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window=period-1).std().values
        return np.concatenate([[np.nan], volatility])

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(self.period, 1)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'close_history': close_history
            }
            self.update_state(state_update)


class VectorBTVolatilityFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility feature generator with comprehensive indicators."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volatility_comprehensive_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized comprehensive volatility features over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive volatility features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_{self.period}')
        
        # Generate multiple volatility indicators using VectorBT
        operations = [
            {'type': 'indicator', 'name': 'atr', 'params': {'indicator': 'atr', 'window': self.period}},
            {'type': 'indicator', 'name': 'bbands_width', 'params': {'indicator': 'bbands_width', 'window': self.period}},
            {'type': 'indicator', 'name': 'bbands_percent', 'params': {'indicator': 'bbands_percent', 'window': self.period}},
            {'type': 'rolling', 'name': 'volatility_std', 'params': {'operation': 'std', 'window': self.period, 'column': 'close'}},
            {'type': 'rolling', 'name': 'volatility_var', 'params': {'operation': 'var', 'window': self.period, 'column': 'close'}}
        ]
        
        # Use batch operations for efficiency
        results = self._vectorbt_batch_operations(data, operations)
        
        # Combine results into a single volatility measure
        if not results.empty:
            # Weighted combination of different volatility measures
            volatility = (
                0.3 * results.get('atr', 0) +
                0.2 * results.get('bbands_width', 0) +
                0.2 * results.get('bbands_percent', 0) +
                0.2 * results.get('volatility_std', 0) +
                0.1 * results.get('volatility_var', 0)
            )
        else:
            # Fallback to simple ATR
            volatility = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        
        return volatility.rename(f'vectorbt_volatility_{self.period}')


class VectorBTBollingerBandsGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Bollinger Bands generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_bbands_{period}_{std_dev}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Bollinger Bands over {period} periods with {std_dev} std dev",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_bbands_{self.period}')
        
        # Generate Bollinger Bands using VectorBT
        bb_result = self._vectorbt_technical_indicator(data, 'bbands_percent', 
                                                     window=self.period, 
                                                     alpha=self.std_dev)
        
        return bb_result.rename(f'vectorbt_bbands_{self.period}')


class VectorBTAverageTrueRangeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Average True Range generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_atr_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Average True Range over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ATR using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_atr_{self.period}')
        
        # Generate ATR using VectorBT
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        
        return atr.rename(f'vectorbt_atr_{self.period}')


def create_default_volatility_generators() -> List[FeatureGenerator]:
    """Create default volatility feature generators with VectorBT optimization."""
    generators = []
    
    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VectorBTVolatilityFeatureGenerator(period))
            generators.append(VectorBTAverageTrueRangeGenerator(period))
            
        # Bollinger Bands with different parameters
        for period in [20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                generators.append(VectorBTBollingerBandsGenerator(period, std_dev))
    else:
        # Fallback to original generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VolatilityFeatureGenerator(period))
    
    return generators