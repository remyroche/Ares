"""
Returns Feature Generator

This module provides feature generators for various types of returns,
including price returns, log returns, and other return-based features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

class ReturnsFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator for returns-based features.
    
    This generator creates various types of returns features including:
    - Simple returns
    - Log returns
    - Cumulative returns
    - Rolling returns
    - Return volatility
    - Return skewness and kurtosis
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the returns feature generator.
        
        Args:
            config: Feature configuration (uses default if None)
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        """Create default configuration for returns features."""
        return FeatureConfig(
            name="returns_features",
            category=FeatureCategory.RETURNS,
            description="Comprehensive returns-based features including simple returns, log returns, and return statistics",
            required_columns=["close"],
            optional_columns=["open", "high", "low", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=252,
            parameters={
                "return_types": ["simple", "log", "cumulative"],
                "volatility_windows": [5, 10, 20],
                "statistics_windows": [10, 20, 50]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'ReturnsFeatureGenerator':
        """Create a default returns feature generator."""
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate returns features.
        
        Args:
            data: Input data with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            Combined returns features (placeholder - actual implementation would return multiple features)
        """
        # This is a simplified implementation that returns a single feature
        # In practice, this would generate multiple return-based features
        
        close_prices = data['close'].values
        
        # Generate simple returns
        simple_returns = self._calculate_simple_returns(close_prices)
        
        # Generate log returns
        log_returns = self._calculate_log_returns(close_prices)
        
        # Generate return volatility
        volatility = self._calculate_return_volatility(simple_returns)
        
        # For now, return the volatility as the main feature
        # In a full implementation, this would be handled differently
        return pd.Series(volatility, index=data.index, name='return_volatility')
    
    def _calculate_simple_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate simple returns."""
        if len(prices) < 2:
            return np.array([])
        
        returns = np.diff(prices) / prices[:-1]
        return np.concatenate([[np.nan], returns])  # Add NaN for first value
    
    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate log returns."""
        if len(prices) < 2:
            return np.array([])
        
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        return np.concatenate([[np.nan], log_returns])  # Add NaN for first value
    
    def _calculate_return_volatility(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling return volatility."""
        if len(returns) < window:
            return np.full(len(returns), np.nan)
        
        # Use vectorized rolling standard deviation
        if self.enable_matrix_ops and self.matrix_ops:
            try:
                # Convert to pandas Series for rolling operations
                returns_series = pd.Series(returns)
                volatility = returns_series.rolling(window=window).std()
                return volatility.values
            except Exception:
                pass
        
        # Fallback to manual calculation
        volatility = np.full(len(returns), np.nan)
        for i in range(window - 1, len(returns)):
            window_returns = returns[i - window + 1:i + 1]
            valid_returns = window_returns[~np.isnan(window_returns)]
            if len(valid_returns) > 1:
                volatility[i] = np.std(valid_returns)
        
        return volatility
    
    def _generate_feature_with_lookback(self, data: pd.DataFrame, lookback: int, **kwargs) -> pd.Series:
        """
        Generate returns features with specific lookback period.
        
        Args:
            data: Input data
            lookback: Lookback period
            **kwargs: Additional parameters
            
        Returns:
            Returns features with specified lookback
        """
        # Update the volatility window based on lookback
        volatility_window = min(lookback, 20)  # Cap at 20 for stability
        
        close_prices = data['close'].values
        simple_returns = self._calculate_simple_returns(close_prices)
        volatility = self._calculate_return_volatility(simple_returns, window=volatility_window)
        
        return pd.Series(volatility, index=data.index, name=f'return_volatility_{lookback}')

class SimpleReturnsGenerator(FeatureGenerator):
    """Generator for simple price returns."""
    
    def __init__(self, lookback: int = 1):
        """Initialize simple returns generator."""
        config = FeatureConfig(
            name=f"simple_returns_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Simple returns over {lookback} period(s)",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=lookback
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate simple returns."""
        close_prices = data['close']
        
        if self.lookback == 1:
            returns = close_prices.pct_change()
        else:
            returns = close_prices.pct_change(periods=self.lookback)
        
        return returns

class LogReturnsGenerator(FeatureGenerator):
    """Generator for log returns."""
    
    def __init__(self, lookback: int = 1):
        """Initialize log returns generator."""
        config = FeatureConfig(
            name=f"log_returns_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Log returns over {lookback} period(s)",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=lookback
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate log returns."""
        close_prices = data['close']
        
        if self.lookback == 1:
            log_returns = np.log(close_prices / close_prices.shift(1))
        else:
            log_returns = np.log(close_prices / close_prices.shift(self.lookback))
        
        return log_returns

class CumulativeReturnsGenerator(FeatureGenerator):
    """Generator for cumulative returns."""
    
    def __init__(self, lookback: int = 20):
        """Initialize cumulative returns generator."""
        config = FeatureConfig(
            name=f"cumulative_returns_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Cumulative returns over {lookback} periods",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=252
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cumulative returns."""
        close_prices = data['close']
        simple_returns = close_prices.pct_change()
        
        # Calculate rolling cumulative returns
        cumulative_returns = simple_returns.rolling(window=self.lookback).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        
        return cumulative_returns

class ReturnVolatilityGenerator(FeatureGenerator):
    """Generator for return volatility."""
    
    def __init__(self, lookback: int = 20):
        """Initialize return volatility generator."""
        config = FeatureConfig(
            name=f"return_volatility_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling volatility of returns over {lookback} periods",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=2,
            max_lookback=252
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate return volatility."""
        close_prices = data['close']
        simple_returns = close_prices.pct_change()
        
        # Calculate rolling volatility
        volatility = simple_returns.rolling(window=self.lookback).std()
        
        return volatility

class ReturnSkewnessGenerator(FeatureGenerator):
    """Generator for return skewness."""
    
    def __init__(self, lookback: int = 20):
        """Initialize return skewness generator."""
        config = FeatureConfig(
            name=f"return_skewness_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling skewness of returns over {lookback} periods",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=3,
            max_lookback=252
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate return skewness."""
        close_prices = data['close']
        simple_returns = close_prices.pct_change()
        
        # Calculate rolling skewness
        skewness = simple_returns.rolling(window=self.lookback).skew()
        
        return skewness

class ReturnKurtosisGenerator(FeatureGenerator):
    """Generator for return kurtosis."""
    
    def __init__(self, lookback: int = 20):
        """Initialize return kurtosis generator."""
        config = FeatureConfig(
            name=f"return_kurtosis_{lookback}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling kurtosis of returns over {lookback} periods",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=4,
            max_lookback=252
        )
        super().__init__(config)
        self.lookback = lookback
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate return kurtosis."""
        close_prices = data['close']
        simple_returns = close_prices.pct_change()
        
        # Calculate rolling kurtosis
        kurtosis = simple_returns.rolling(window=self.lookback).kurt()
        
        return kurtosis

# Factory functions for creating returns generators
def create_returns_generators(lookback_periods: List[int] = None) -> List[FeatureGenerator]:
    """
    Create a set of returns feature generators.
    
    Args:
        lookback_periods: List of lookback periods to use
        
    Returns:
        List of returns feature generators
    """
    if lookback_periods is None:
        lookback_periods = [1, 5, 10, 20]
    
    generators = []
    
    # Create generators for each lookback period
    for lookback in lookback_periods:
        generators.extend([
            SimpleReturnsGenerator(lookback),
            LogReturnsGenerator(lookback),
            CumulativeReturnsGenerator(lookback),
            ReturnVolatilityGenerator(lookback),
            ReturnSkewnessGenerator(lookback),
            ReturnKurtosisGenerator(lookback)
        ])
    
    return generators

def create_default_returns_generators() -> List[FeatureGenerator]:
    """Create default returns feature generators."""
    return create_returns_generators([1, 5, 10, 20])