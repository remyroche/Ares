"""
Returns Feature Generator

This module provides feature generators for return-based indicators,
including log returns, cumulative returns, rolling returns, and return statistics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class ReturnsFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for return-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="returns_features",
            category=FeatureCategory.RETURNS,
            description="Comprehensive return-based features including log returns, cumulative returns, and return statistics",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "return_periods": [1, 5, 10, 20],
                "log_return_periods": [1, 5, 10],
                "cumulative_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'ReturnsFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        returns = self._calculate_returns(close_prices, period=1)
        return pd.Series(returns, index=data.index, name='returns_1')
    
    def _calculate_returns(self, prices: np.ndarray, period: int = 1) -> np.ndarray:
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        returns = (prices - np.roll(prices, period)) / np.roll(prices, period)
        returns[:period] = np.nan
        return returns

class LogReturnsGenerator(FeatureGenerator):
    """Generator for Log Returns with different base calculations."""
    
    def __init__(self, 
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Log Returns generator.
        
        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"log_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Log returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate log returns based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate log returns
        log_returns = np.log(base_values / base_values.shift(self.period))
        
        return log_returns

class SimpleReturnsGenerator(FeatureGenerator):
    """Generator for Simple Returns with different base calculations."""
    
    def __init__(self, 
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Simple Returns generator.
        
        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"simple_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Simple returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate simple returns based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate simple returns
        simple_returns = (base_values - base_values.shift(self.period)) / base_values.shift(self.period)
        
        return simple_returns

class CumulativeReturnsGenerator(FeatureGenerator):
    """Generator for Cumulative Returns with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Cumulative Returns generator.
        
        Args:
            window: Rolling window for cumulative returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cumulative_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Cumulative returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cumulative returns based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate returns
        returns = base_values.pct_change()
        
        # Calculate cumulative returns over rolling window
        cumulative_returns = returns.rolling(window=self.window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        
        return cumulative_returns

class RollingReturnsGenerator(FeatureGenerator):
    """Generator for Rolling Returns with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Rolling Returns generator.
        
        Args:
            window: Rolling window for returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"rolling_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate rolling returns based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate rolling returns
        rolling_returns = base_values.rolling(window=self.window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], raw=False
        )
        
        return rolling_returns

class ReturnsVolatilityGenerator(FeatureGenerator):
    """Generator for Returns Volatility with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Volatility generator.
        
        Args:
            window: Rolling window for volatility calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_volatility_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns volatility over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate returns volatility based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate returns
        returns = base_values.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.window).std()
        
        return volatility

class ReturnsSkewnessGenerator(FeatureGenerator):
    """Generator for Returns Skewness with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Skewness generator.
        
        Args:
            window: Rolling window for skewness calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_skewness_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns skewness over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate returns skewness based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate returns
        returns = base_values.pct_change()
        
        # Calculate rolling skewness
        skewness = returns.rolling(window=self.window).skew()
        
        return skewness

class ReturnsKurtosisGenerator(FeatureGenerator):
    """Generator for Returns Kurtosis with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Kurtosis generator.
        
        Args:
            window: Rolling window for kurtosis calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_kurtosis_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns kurtosis over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate returns kurtosis based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate returns
        returns = base_values.pct_change()
        
        # Calculate rolling kurtosis
        kurtosis = returns.rolling(window=self.window).kurt()
        
        return kurtosis

class SharpeRatioGenerator(FeatureGenerator):
    """Generator for Sharpe Ratio with different base calculations."""
    
    def __init__(self, 
                 window: int = 20,
                 risk_free_rate: float = 0.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Sharpe Ratio generator.
        
        Args:
            window: Rolling window for Sharpe ratio calculation
            risk_free_rate: Risk-free rate (annualized)
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"sharpe_ratio_{window}_{risk_free_rate}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Sharpe ratio over {window} periods with risk-free rate {risk_free_rate} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'risk_free_rate': risk_free_rate,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Sharpe ratio based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate returns
        returns = base_values.pct_change()
        
        # Calculate rolling Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_returns.rolling(window=self.window).mean() / returns.rolling(window=self.window).std()
        
        return sharpe_ratio

def create_returns_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of returns feature generators."""
    if periods is None:
        periods = {
            'log_returns': [1, 5, 10],
            'simple_returns': [1, 5, 10],
            'cumulative_returns': [10, 20],
            'rolling_returns': [10, 20],
            'volatility': [20],
            'skewness': [20],
            'kurtosis': [20],
            'sharpe_ratio': [20]
        }
    
    generators = []
    
    # Log returns generators
    for period in periods.get('log_returns', [1, 5, 10]):
        generators.append(LogReturnsGenerator(period))
    
    # Simple returns generators
    for period in periods.get('simple_returns', [1, 5, 10]):
        generators.append(SimpleReturnsGenerator(period))
    
    # Cumulative returns generators
    for window in periods.get('cumulative_returns', [10, 20]):
        generators.append(CumulativeReturnsGenerator(window))
    
    # Rolling returns generators
    for window in periods.get('rolling_returns', [10, 20]):
        generators.append(RollingReturnsGenerator(window))
    
    # Volatility generators
    for window in periods.get('volatility', [20]):
        generators.append(ReturnsVolatilityGenerator(window))
    
    # Skewness generators
    for window in periods.get('skewness', [20]):
        generators.append(ReturnsSkewnessGenerator(window))
    
    # Kurtosis generators
    for window in periods.get('kurtosis', [20]):
        generators.append(ReturnsKurtosisGenerator(window))
    
    # Sharpe ratio generators
    for window in periods.get('sharpe_ratio', [20]):
        generators.append(SharpeRatioGenerator(window))
    
    return generators

def create_default_returns_generators() -> List[FeatureGenerator]:
    """Create default returns generators."""
    return create_returns_generators()
