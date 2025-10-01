"""
Volatility Feature Generator

This module provides feature generators for volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Supports different base calculations: price returns, returns-based VWAP, etc.
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

class VolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility-based features with batch processing."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Comprehensive volatility-based features including Bollinger Bands, ATR, and volatility measures",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "bb_periods": [20],
                "bb_std": [2.0],
                "atr_periods": [14],
                "volatility_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        volatility = self._calculate_volatility(close_prices, period=20)
        return pd.Series(volatility, index=data.index, name='volatility_20')
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window=period-1).std().values
        return np.concatenate([[np.nan], volatility])
    
    @classmethod
    def generate_batch_features(cls,
                               data: pd.DataFrame,
                               periods: List[int] = [5, 10, 20, 30],
                               volatility_types: List[str] = ["returns", "price"],
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate volatility features for multiple periods and types in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            volatility_types: List of volatility calculation types
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to Series
        """
        features = {}
        close_prices = data['close']
        
        for period in periods:
            for vol_type in volatility_types:
                if vol_type == "returns":
                    # Calculate volatility based on returns
                    returns = close_prices.pct_change().dropna()
                    volatility = returns.rolling(window=period).std()
                    # Pad with NaN to match original length
                    volatility = pd.concat([pd.Series([np.nan] * (len(close_prices) - len(volatility)), index=close_prices.index[:len(close_prices) - len(volatility)]), volatility])
                elif vol_type == "price":
                    # Calculate volatility based on price levels
                    volatility = close_prices.rolling(window=period).std()
                else:
                    continue
                
                feature_name = f"volatility_{vol_type}_{period}"
                features[feature_name] = volatility
        
        return features

class BollingerBandsGenerator(FeatureGenerator):
    """Generator for Bollinger Bands with different base calculations and batch processing."""
    
    def __init__(self, 
                 period: int = 20, 
                 std_dev: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 band_type: str = "upper",  # "upper", "lower", "middle"
                 **base_kwargs):
        """
        Initialize Bollinger Bands generator.
        
        Args:
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            band_type: Type of band to generate ("upper", "lower", "middle")
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"bb_{band_type}_{period}_{std_dev}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Bollinger Bands {band_type} with period={period}, std={std_dev} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'std_dev': std_dev,
                'base_calculation': base_calculation.value,
                'band_type': band_type,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
        self.base_calculation = base_calculation
        self.band_type = band_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate Bollinger Bands on base values
        sma = base_values.rolling(window=self.period).mean()
        std = base_values.rolling(window=self.period).std()
        
        if self.band_type == "upper":
            band = sma + (std * self.std_dev)
        elif self.band_type == "lower":
            band = sma - (std * self.std_dev)
        elif self.band_type == "middle":
            band = sma
        else:
            raise ValueError(f"Invalid band_type: {self.band_type}")
        
        return band
    
    @classmethod
    def generate_batch_features(cls, 
                               data: pd.DataFrame,
                               periods: List[int] = [10, 20, 30],
                               std_devs: List[float] = [1.5, 2.0, 2.5],
                               band_types: List[str] = ["upper", "lower", "middle"],
                               base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                               **base_kwargs) -> Dict[str, pd.Series]:
        """
        Generate Bollinger Bands features for multiple periods, std_devs, and band_types in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            std_devs: List of standard deviation multipliers
            band_types: List of band types to generate
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
            
        Returns:
            Dictionary mapping feature names to Series
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        base_values = base_calculator.calculate(data)
        
        features = {}
        
        # Vectorized calculation for all combinations
        for period in periods:
            # Calculate rolling mean and std for this period
            rolling_mean = base_values.rolling(window=period).mean()
            rolling_std = base_values.rolling(window=period).std()
            
            for std_dev in std_devs:
                for band_type in band_types:
                    if band_type == "upper":
                        band = rolling_mean + (rolling_std * std_dev)
                    elif band_type == "lower":
                        band = rolling_mean - (rolling_std * std_dev)
                    elif band_type == "middle":
                        band = rolling_mean
                    else:
                        continue
                    
                    feature_name = f"bb_{band_type}_{period}_{std_dev}_{base_calculation.value}"
                    features[feature_name] = band
        
        return features

class ATRGenerator(FeatureGenerator):
    """Generator for Average True Range with different base calculations and batch processing."""
    
    def __init__(self, 
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize ATR generator.
        
        Args:
            period: ATR period
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
            name=f"atr_{period}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Average True Range over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
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
        """Generate ATR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional ATR calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=self.period).mean()
            return atr
        else:
            # For other base calculations, calculate ATR on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate rolling standard deviation as volatility measure
            atr = base_values.rolling(window=self.period).std()
            return atr
    
    @classmethod
    def generate_batch_features(cls,
                               data: pd.DataFrame,
                               periods: List[int] = [7, 14, 21],
                               base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                               **base_kwargs) -> Dict[str, pd.Series]:
        """
        Generate ATR features for multiple periods in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
            
        Returns:
            Dictionary mapping feature names to Series
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        features = {}
        
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional ATR calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR for each period
            for period in periods:
                atr = true_range.rolling(window=period).mean()
                feature_name = f"atr_{period}_{base_calculation.value}"
                features[feature_name] = atr
        else:
            # For other base calculations, calculate ATR on the base values
            base_calculator = create_base_calculator(base_calculation, **base_kwargs)
            base_values = base_calculator.calculate(data)
            
            # Calculate rolling standard deviation for each period
            for period in periods:
                atr = base_values.rolling(window=period).std()
                feature_name = f"atr_{period}_{base_calculation.value}"
                features[feature_name] = atr
        
        return features


class VolatilityBandsGenerator(FeatureGenerator):
    """Generator for Volatility Bands with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 std_multiplier: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Volatility Bands generator.
        
        Args:
            period: Period for volatility calculation
            std_multiplier: Standard deviation multiplier for bands
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_bands_{period}_{std_multiplier}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Volatility Bands with period={period}, multiplier={std_multiplier} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'std_multiplier': std_multiplier,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.std_multiplier = std_multiplier
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volatility Bands upper band based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate moving average and standard deviation
        sma = base_values.rolling(window=self.period).mean()
        volatility = base_values.rolling(window=self.period).std()
        
        # Return upper band as the main feature
        # Lower band would be: sma - (volatility * multiplier)
        # Middle line would be: sma
        upper_band = sma + (volatility * self.std_multiplier)
        
        return upper_band

def create_volatility_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of volatility feature generators."""
    if periods is None:
        periods = {
            'bb': [20],
            'atr': [14],
            'volatility': [10, 20],
            'volatility_bands': [20]
        }
    
    generators = []
    
    # Bollinger Bands generators
    for period in periods.get('bb', [20]):
        generators.append(BollingerBandsGenerator(period))
    
    # ATR generators
    for period in periods.get('atr', [14]):
        generators.append(ATRGenerator(period))
    
    # Volatility Bands generators
    for period in periods.get('volatility_bands', [20]):
        generators.append(VolatilityBandsGenerator(period))
    
    return generators

def create_default_volatility_generators() -> List[FeatureGenerator]:
    return create_volatility_generators()