"""
Refactored Momentum Feature Generator

This module demonstrates how feature generators should be refactored to use
centralized utilities from feature_generation/ and features_common/ to eliminate
code duplication and ensure consistency.

Key Improvements:
- Uses CentralizedIndicators for all technical calculations
- Uses VectorBTRollingOptimizer for rolling operations
- Uses VectorBTScaler for normalization
- Eliminates duplicate RSI, MACD, EMA, SMA implementations
- Consistent error handling and fallback strategies
- Memory-efficient batch processing
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE

# Centralized utility imports
from ..utils.centralized_indicators import get_centralized_indicators
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)


class RefactoredRSIGenerator(VectorizedFeatureGenerator):
    """
    Refactored RSI generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all RSI calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, period: int = 14, normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored RSI generator.
        
        Args:
            period: RSI period
            normalize: Whether to normalize the output
            normalization_method: Normalization method ('zscore', 'minmax', 'robust')
        """
        config = FeatureConfig(
            name=f"refactored_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored RSI {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={
                'period': period,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized RSI calculation
        rsi = self.indicators.calculate_rsi(data['close'], self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            rsi = self.scaler.fit_transform(rsi)
        
        return rsi.rename(f'refactored_rsi_{self.period}')


class RefactoredMACDGenerator(VectorizedFeatureGenerator):
    """
    Refactored MACD generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all MACD calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                 normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored MACD generator.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name=f"refactored_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored MACD {fast}/{slow}/{signal} using centralized utilities",
            required_columns=["close"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized MACD calculation
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            data['close'], self.fast, self.slow, self.signal
        )
        
        # Return MACD line (can be extended to return signal_line or histogram)
        result = macd_line
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            result = self.scaler.fit_transform(result)
        
        return result.rename(f'refactored_macd_{self.fast}_{self.slow}_{self.signal}')


class RefactoredStochasticGenerator(VectorizedFeatureGenerator):
    """
    Refactored Stochastic generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all Stochastic calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, 
                 normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored Stochastic generator.
        
        Args:
            k_period: %K period
            d_period: %D period (smoothing)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name=f"refactored_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Stochastic {k_period}/{d_period} using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period,
            parameters={
                'k_period': k_period,
                'd_period': d_period,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.k_period = k_period
        self.d_period = d_period
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized Stochastic calculation
        k_percent, d_percent = self.indicators.calculate_stochastic(
            data['high'], data['low'], data['close'], self.k_period, self.d_period
        )
        
        # Return %K (can be extended to return %D)
        result = k_percent
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            result = self.scaler.fit_transform(result)
        
        return result.rename(f'refactored_stochastic_k_{self.k_period}_{self.d_period}')


class RefactoredWilliamsRGenerator(VectorizedFeatureGenerator):
    """
    Refactored Williams %R generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all Williams %R calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, period: int = 14, normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored Williams %R generator.
        
        Args:
            period: Williams %R period
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name=f"refactored_williams_r_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Williams %R {period} using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Williams %R using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized Williams %R calculation
        williams_r = self.indicators.calculate_williams_r(
            data['high'], data['low'], data['close'], self.period
        )
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            williams_r = self.scaler.fit_transform(williams_r)
        
        return williams_r.rename(f'refactored_williams_r_{self.period}')


class RefactoredMomentumGenerator(VectorizedFeatureGenerator):
    """
    Refactored Momentum generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all momentum calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, period: int = 10, normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored Momentum generator.
        
        Args:
            period: Momentum period
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name=f"refactored_momentum_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Momentum {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Momentum using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized Momentum calculation
        momentum = self.indicators.calculate_momentum(data['close'], self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            momentum = self.scaler.fit_transform(momentum)
        
        return momentum.rename(f'refactored_momentum_{self.period}')


class RefactoredROCGenerator(VectorizedFeatureGenerator):
    """
    Refactored Rate of Change generator using centralized utilities.
    
    This generator eliminates code duplication by using CentralizedIndicators
    for all ROC calculations, ensuring consistency across the codebase.
    """
    
    def __init__(self, period: int = 10, normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored ROC generator.
        
        Args:
            period: ROC period
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name=f"refactored_roc_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Refactored Rate of Change {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ROC using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized ROC calculation
        roc = self.indicators.calculate_roc(data['close'], self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            roc = self.scaler.fit_transform(roc)
        
        return roc.rename(f'refactored_roc_{self.period}')


class RefactoredBatchMomentumGenerator(VectorizedFeatureGenerator):
    """
    Refactored batch momentum generator using centralized utilities.
    
    This generator demonstrates how to efficiently generate multiple momentum
    indicators in batch using centralized utilities, eliminating code duplication
    and improving performance.
    """
    
    def __init__(self, indicators: List[Dict[str, Any]], normalize: bool = False, normalization_method: str = 'zscore'):
        """
        Initialize refactored batch momentum generator.
        
        Args:
            indicators: List of indicator configurations
            normalize: Whether to normalize the output
            normalization_method: Normalization method
        """
        config = FeatureConfig(
            name="refactored_batch_momentum",
            category=FeatureCategory.MOMENTUM,
            description="Refactored batch momentum indicators using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=50,
            min_lookback=10,
            max_lookback=100,
            parameters={
                'indicators': indicators,
                'normalize': normalize,
                'normalization_method': normalization_method
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.indicators = indicators
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.centralized_indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch momentum indicators using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Use centralized batch calculation
        results = self.centralized_indicators.calculate_batch_indicators(data, self.indicators)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            results = self.scaler.fit_transform(results)
        
        # Return the first indicator as the main feature (can be extended)
        if not results.empty:
            return results.iloc[:, 0].rename('refactored_batch_momentum')
        else:
            return pd.Series(np.nan, index=data.index, name='refactored_batch_momentum')


def create_refactored_momentum_generators(periods: Dict[str, List[int]] = None, 
                                        normalize: bool = False, 
                                        normalization_method: str = 'zscore') -> List[FeatureGenerator]:
    """
    Create refactored momentum generators using centralized utilities.
    
    Args:
        periods: Dictionary of periods for different indicators
        normalize: Whether to normalize outputs
        normalization_method: Normalization method
        
    Returns:
        List of refactored momentum generators
    """
    if periods is None:
        periods = {
            'rsi': [14, 21],
            'macd_fast': [12],
            'macd_slow': [26],
            'stochastic': [14],
            'williams_r': [14],
            'momentum': [10, 20],
            'roc': [10, 20]
        }
    
    generators = []
    
    # RSI generators
    for period in periods.get('rsi', [14]):
        generators.append(RefactoredRSIGenerator(period, normalize, normalization_method))
    
    # MACD generators
    fast_periods = periods.get('macd_fast', [12])
    slow_periods = periods.get('macd_slow', [26])
    for fast in fast_periods:
        for slow in slow_periods:
            generators.append(RefactoredMACDGenerator(fast, slow, normalize=normalize, normalization_method=normalization_method))
    
    # Stochastic generators
    for period in periods.get('stochastic', [14]):
        generators.append(RefactoredStochasticGenerator(period, normalize=normalize, normalization_method=normalization_method))
    
    # Williams %R generators
    for period in periods.get('williams_r', [14]):
        generators.append(RefactoredWilliamsRGenerator(period, normalize=normalize, normalization_method=normalization_method))
    
    # Momentum generators
    for period in periods.get('momentum', [10]):
        generators.append(RefactoredMomentumGenerator(period, normalize=normalize, normalization_method=normalization_method))
    
    # ROC generators
    for period in periods.get('roc', [10]):
        generators.append(RefactoredROCGenerator(period, normalize=normalize, normalization_method=normalization_method))
    
    # Batch generator
    batch_indicators = [
        {'type': 'rsi', 'name': 'rsi_14', 'params': {'window': 14}},
        {'type': 'macd', 'name': 'macd_12_26', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        {'type': 'stochastic', 'name': 'stoch_14', 'params': {'k_period': 14, 'd_period': 3}},
        {'type': 'williams_r', 'name': 'willr_14', 'params': {'period': 14}},
        {'type': 'momentum', 'name': 'momentum_10', 'params': {'period': 10}},
        {'type': 'roc', 'name': 'roc_10', 'params': {'period': 10}}
    ]
    generators.append(RefactoredBatchMomentumGenerator(batch_indicators, normalize, normalization_method))
    
    return generators


def create_default_refactored_momentum_generators() -> List[FeatureGenerator]:
    """Create default refactored momentum generators."""
    return create_refactored_momentum_generators()