"""
Interaction Feature Generator

This module provides feature generators for interaction features,
including cross-timeframe interactions, feature combinations, and
polynomial features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from itertools import combinations, product

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

class InteractionFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator for interaction-based features.
    
    This generator creates various interaction features including:
    - Cross-timeframe interactions
    - Feature combinations (ratios, differences, products)
    - Polynomial features
    - Correlation-based interactions
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the interaction feature generator.
        
        Args:
            config: Feature configuration (uses default if None)
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        """Create default configuration for interaction features."""
        return FeatureConfig(
            name="interaction_features",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description="Comprehensive interaction features including cross-timeframe and feature combinations",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                "interaction_types": ["ratio", "difference", "product", "correlation"],
                "max_interaction_depth": 2,
                "top_k_features": 50,
                "correlation_threshold": 0.95,
                "polynomial_degree": 2
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'InteractionFeatureGenerator':
        """Create a default interaction feature generator."""
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate interaction features.
        
        Args:
            data: Input data with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            Combined interaction features (placeholder - actual implementation would return multiple features)
        """
        # This is a simplified implementation that returns a single feature
        # In practice, this would generate multiple interaction features
        
        close_prices = data['close'].values
        
        # Generate a simple interaction feature (price momentum interaction)
        momentum_short = pd.Series(close_prices).pct_change(5)
        momentum_long = pd.Series(close_prices).pct_change(20)
        interaction = momentum_short * momentum_long
        
        return pd.Series(interaction, index=data.index, name='momentum_interaction')

class CrossTimeframeInteractionGenerator(FeatureGenerator):
    """Generator for cross-timeframe interaction features."""
    
    def __init__(self, 
                 short_period: int = 5,
                 long_period: int = 20,
                 interaction_type: str = "ratio"):
        """
        Initialize cross-timeframe interaction generator.
        
        Args:
            short_period: Short timeframe period
            long_period: Long timeframe period
            interaction_type: Type of interaction ("ratio", "difference", "product")
        """
        config = FeatureConfig(
            name=f"cross_timeframe_{interaction_type}_{short_period}_{long_period}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe {interaction_type} between {short_period} and {long_period} periods",
            required_columns=["close"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=long_period,
            parameters={
                'short_period': short_period,
                'long_period': long_period,
                'interaction_type': interaction_type
            }
        )
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
        self.interaction_type = interaction_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe interaction feature."""
        close = data['close']
        
        # Calculate short and long period values
        short_value = close.rolling(window=self.short_period).mean()
        long_value = close.rolling(window=self.long_period).mean()
        
        # Calculate interaction based on type
        if self.interaction_type == "ratio":
            interaction = short_value / long_value
        elif self.interaction_type == "difference":
            interaction = short_value - long_value
        elif self.interaction_type == "product":
            interaction = short_value * long_value
        else:
            raise ValueError(f"Invalid interaction_type: {self.interaction_type}")
        
        return interaction

class FeatureRatioGenerator(FeatureGenerator):
    """Generator for feature ratio interactions."""
    
    def __init__(self, 
                 numerator_period: int = 5,
                 denominator_period: int = 20,
                 feature_type: str = "sma"):
        """
        Initialize feature ratio generator.
        
        Args:
            numerator_period: Period for numerator feature
            denominator_period: Period for denominator feature
            feature_type: Type of feature ("sma", "ema", "rsi", etc.)
        """
        config = FeatureConfig(
            name=f"{feature_type}_ratio_{numerator_period}_{denominator_period}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Ratio of {feature_type} {numerator_period} to {denominator_period} periods",
            required_columns=["close"],
            default_lookback=denominator_period,
            min_lookback=denominator_period,
            max_lookback=denominator_period,
            parameters={
                'numerator_period': numerator_period,
                'denominator_period': denominator_period,
                'feature_type': feature_type
            }
        )
        super().__init__(config)
        self.numerator_period = numerator_period
        self.denominator_period = denominator_period
        self.feature_type = feature_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate feature ratio."""
        close = data['close']
        
        # Calculate features based on type
        if self.feature_type == "sma":
            numerator = close.rolling(window=self.numerator_period).mean()
            denominator = close.rolling(window=self.denominator_period).mean()
        elif self.feature_type == "ema":
            numerator = close.ewm(span=self.numerator_period).mean()
            denominator = close.ewm(span=self.denominator_period).mean()
        elif self.feature_type == "volatility":
            numerator = close.rolling(window=self.numerator_period).std()
            denominator = close.rolling(window=self.denominator_period).std()
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")
        
        # Calculate ratio
        ratio = numerator / denominator
        
        return ratio

class PolynomialFeatureGenerator(FeatureGenerator):
    """Generator for polynomial interaction features."""
    
    def __init__(self, 
                 period: int = 20,
                 degree: int = 2,
                 feature_type: str = "returns"):
        """
        Initialize polynomial feature generator.
        
        Args:
            period: Period for base feature calculation
            degree: Polynomial degree
            feature_type: Type of base feature ("returns", "volatility", "momentum")
        """
        config = FeatureConfig(
            name=f"polynomial_{feature_type}_{period}_deg{degree}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Polynomial {feature_type} feature of degree {degree} over {period} periods",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'degree': degree,
                'feature_type': feature_type
            }
        )
        super().__init__(config)
        self.period = period
        self.degree = degree
        self.feature_type = feature_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate polynomial feature."""
        close = data['close']
        
        # Calculate base feature
        if self.feature_type == "returns":
            base_feature = close.pct_change(self.period)
        elif self.feature_type == "volatility":
            base_feature = close.rolling(window=self.period).std()
        elif self.feature_type == "momentum":
            base_feature = close - close.shift(self.period)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")
        
        # Calculate polynomial feature
        polynomial_feature = base_feature ** self.degree
        
        return polynomial_feature

class CorrelationInteractionGenerator(FeatureGenerator):
    """Generator for correlation-based interaction features."""
    
    def __init__(self, 
                 period1: int = 5,
                 period2: int = 20,
                 feature1: str = "returns",
                 feature2: str = "volume"):
        """
        Initialize correlation interaction generator.
        
        Args:
            period1: Period for first feature
            period2: Period for second feature
            feature1: First feature type
            feature2: Second feature type
        """
        required_columns = ["close"]
        if feature2 == "volume":
            required_columns.append("volume")
        
        config = FeatureConfig(
            name=f"correlation_{feature1}_{period1}_{feature2}_{period2}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Correlation between {feature1} ({period1}) and {feature2} ({period2})",
            required_columns=required_columns,
            default_lookback=max(period1, period2),
            min_lookback=max(period1, period2),
            max_lookback=max(period1, period2),
            parameters={
                'period1': period1,
                'period2': period2,
                'feature1': feature1,
                'feature2': feature2
            }
        )
        super().__init__(config)
        self.period1 = period1
        self.period2 = period2
        self.feature1 = feature1
        self.feature2 = feature2
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate correlation interaction feature."""
        close = data['close']
        
        # Calculate first feature
        if self.feature1 == "returns":
            feature1_values = close.pct_change(self.period1)
        elif self.feature1 == "volatility":
            feature1_values = close.rolling(window=self.period1).std()
        else:
            raise ValueError(f"Invalid feature1: {self.feature1}")
        
        # Calculate second feature
        if self.feature2 == "volume":
            feature2_values = data['volume'].rolling(window=self.period2).mean()
        elif self.feature2 == "returns":
            feature2_values = close.pct_change(self.period2)
        else:
            raise ValueError(f"Invalid feature2: {self.feature2}")
        
        # Calculate rolling correlation
        correlation = feature1_values.rolling(window=max(self.period1, self.period2)).corr(feature2_values)
        
        return correlation

# Factory functions for creating interaction generators
def create_interaction_generators(config: Dict[str, Any] = None) -> List[FeatureGenerator]:
    """
    Create a set of interaction feature generators.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of interaction feature generators
    """
    if config is None:
        config = {
            'cross_timeframe': {
                'short_periods': [5, 10],
                'long_periods': [20, 50],
                'interaction_types': ['ratio', 'difference', 'product']
            },
            'feature_ratios': {
                'periods': [(5, 20), (10, 30)],
                'feature_types': ['sma', 'ema', 'volatility']
            },
            'polynomial': {
                'periods': [10, 20],
                'degrees': [2, 3],
                'feature_types': ['returns', 'volatility']
            },
            'correlation': {
                'combinations': [
                    (5, 20, 'returns', 'volume'),
                    (10, 30, 'volatility', 'returns')
                ]
            }
        }
    
    generators = []
    
    # Cross-timeframe interactions
    for short_period in config.get('cross_timeframe', {}).get('short_periods', [5]):
        for long_period in config.get('cross_timeframe', {}).get('long_periods', [20]):
            for interaction_type in config.get('cross_timeframe', {}).get('interaction_types', ['ratio']):
                generators.append(CrossTimeframeInteractionGenerator(
                    short_period, long_period, interaction_type
                ))
    
    # Feature ratios
    for numerator_period, denominator_period in config.get('feature_ratios', {}).get('periods', [(5, 20)]):
        for feature_type in config.get('feature_ratios', {}).get('feature_types', ['sma']):
            generators.append(FeatureRatioGenerator(
                numerator_period, denominator_period, feature_type
            ))
    
    # Polynomial features
    for period in config.get('polynomial', {}).get('periods', [20]):
        for degree in config.get('polynomial', {}).get('degrees', [2]):
            for feature_type in config.get('polynomial', {}).get('feature_types', ['returns']):
                generators.append(PolynomialFeatureGenerator(
                    period, degree, feature_type
                ))
    
    # Correlation interactions
    for period1, period2, feature1, feature2 in config.get('correlation', {}).get('combinations', []):
        generators.append(CorrelationInteractionGenerator(
            period1, period2, feature1, feature2
        ))
    
    return generators

def create_default_interaction_generators() -> List[FeatureGenerator]:
    """Create default interaction feature generators."""
    return create_interaction_generators()

# Polynomial Feature Generator
class PolynomialFeatureGenerator(FeatureGenerator):
    """Generator for polynomial features (squared, cubed, etc.)."""
    
    def __init__(self,
                 feature_name: str,
                 degree: int = 2,
                 base_calculation: str = "price_returns"):
        """
        Initialize polynomial feature generator.
        
        Args:
            feature_name: Name of the base feature
            degree: Polynomial degree (2 for squared, 3 for cubed, etc.)
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature_name}_polynomial_{degree}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Polynomial feature of degree {degree} for {feature_name}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature_name': feature_name,
                'degree': degree,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature_name = feature_name
        self.degree = degree
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate polynomial feature."""
        # This is a placeholder - in practice, you'd get the base feature
        # and apply the polynomial transformation
        base_values = data['close'].pct_change()  # Placeholder
        
        # Apply polynomial transformation
        polynomial_feature = base_values ** self.degree
        
        return polynomial_feature

# Feature Ratio Generator
class FeatureRatioGenerator(FeatureGenerator):
    """Generator for feature ratio combinations."""
    
    def __init__(self,
                 feature1_name: str,
                 feature2_name: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize feature ratio generator.
        
        Args:
            feature1_name: Name of the first feature
            feature2_name: Name of the second feature
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature1_name}_{feature2_name}_ratio",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Ratio of {feature1_name} to {feature2_name}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature1_name': feature1_name,
                'feature2_name': feature2_name,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature1_name = feature1_name
        self.feature2_name = feature2_name
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate feature ratio."""
        # This is a placeholder - in practice, you'd get both features
        # and calculate their ratio
        feature1 = data['close'].pct_change()  # Placeholder
        feature2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate ratio with safe division
        ratio = feature1 / feature2.replace(0, np.nan)
        
        return ratio

# Feature Difference Generator
class FeatureDifferenceGenerator(FeatureGenerator):
    """Generator for feature difference combinations."""
    
    def __init__(self,
                 feature1_name: str,
                 feature2_name: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize feature difference generator.
        
        Args:
            feature1_name: Name of the first feature
            feature2_name: Name of the second feature
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature1_name}_{feature2_name}_diff",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Difference of {feature1_name} and {feature2_name}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature1_name': feature1_name,
                'feature2_name': feature2_name,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature1_name = feature1_name
        self.feature2_name = feature2_name
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate feature difference."""
        # This is a placeholder - in practice, you'd get both features
        # and calculate their difference
        feature1 = data['close'].pct_change()  # Placeholder
        feature2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate difference
        difference = feature1 - feature2
        
        return difference

# Feature Product Generator
class FeatureProductGenerator(FeatureGenerator):
    """Generator for feature product combinations."""
    
    def __init__(self,
                 feature1_name: str,
                 feature2_name: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize feature product generator.
        
        Args:
            feature1_name: Name of the first feature
            feature2_name: Name of the second feature
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature1_name}_{feature2_name}_product",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Product of {feature1_name} and {feature2_name}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature1_name': feature1_name,
                'feature2_name': feature2_name,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature1_name = feature1_name
        self.feature2_name = feature2_name
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate feature product."""
        # This is a placeholder - in practice, you'd get both features
        # and calculate their product
        feature1 = data['close'].pct_change()  # Placeholder
        feature2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate product
        product = feature1 * feature2
        
        return product

# Cross-timeframe Ratio Generator
class CrossTimeframeRatioGenerator(FeatureGenerator):
    """Generator for cross-timeframe ratio features."""
    
    def __init__(self,
                 feature_name: str,
                 timeframe1: str,
                 timeframe2: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize cross-timeframe ratio generator.
        
        Args:
            feature_name: Name of the base feature
            timeframe1: First timeframe
            timeframe2: Second timeframe
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature_name}_{timeframe1}_{timeframe2}_ratio",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe ratio of {feature_name} between {timeframe1} and {timeframe2}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature_name': feature_name,
                'timeframe1': timeframe1,
                'timeframe2': timeframe2,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature_name = feature_name
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe ratio."""
        # This is a placeholder - in practice, you'd get the feature
        # from both timeframes and calculate their ratio
        feature_tf1 = data['close'].pct_change()  # Placeholder
        feature_tf2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate cross-timeframe ratio
        ratio = feature_tf1 / feature_tf2.replace(0, np.nan)
        
        return ratio

# Cross-timeframe Difference Generator
class CrossTimeframeDifferenceGenerator(FeatureGenerator):
    """Generator for cross-timeframe difference features."""
    
    def __init__(self,
                 feature_name: str,
                 timeframe1: str,
                 timeframe2: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize cross-timeframe difference generator.
        
        Args:
            feature_name: Name of the base feature
            timeframe1: First timeframe
            timeframe2: Second timeframe
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature_name}_{timeframe1}_{timeframe2}_diff",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe difference of {feature_name} between {timeframe1} and {timeframe2}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature_name': feature_name,
                'timeframe1': timeframe1,
                'timeframe2': timeframe2,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature_name = feature_name
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe difference."""
        # This is a placeholder - in practice, you'd get the feature
        # from both timeframes and calculate their difference
        feature_tf1 = data['close'].pct_change()  # Placeholder
        feature_tf2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate cross-timeframe difference
        difference = feature_tf1 - feature_tf2
        
        return difference

# Cross-timeframe Product Generator
class CrossTimeframeProductGenerator(FeatureGenerator):
    """Generator for cross-timeframe product features."""
    
    def __init__(self,
                 feature_name: str,
                 timeframe1: str,
                 timeframe2: str,
                 base_calculation: str = "price_returns"):
        """
        Initialize cross-timeframe product generator.
        
        Args:
            feature_name: Name of the base feature
            timeframe1: First timeframe
            timeframe2: Second timeframe
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature_name}_{timeframe1}_{timeframe2}_product",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe product of {feature_name} between {timeframe1} and {timeframe2}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={
                'feature_name': feature_name,
                'timeframe1': timeframe1,
                'timeframe2': timeframe2,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature_name = feature_name
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe product."""
        # This is a placeholder - in practice, you'd get the feature
        # from both timeframes and calculate their product
        feature_tf1 = data['close'].pct_change()  # Placeholder
        feature_tf2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate cross-timeframe product
        product = feature_tf1 * feature_tf2
        
        return product

# Correlation Interaction Generator
class CorrelationInteractionGenerator(FeatureGenerator):
    """Generator for correlation-based interaction features."""
    
    def __init__(self,
                 feature1_name: str,
                 feature2_name: str,
                 window: int = 20,
                 base_calculation: str = "price_returns"):
        """
        Initialize correlation interaction generator.
        
        Args:
            feature1_name: Name of the first feature
            feature2_name: Name of the second feature
            window: Correlation window
            base_calculation: Base calculation type
        """
        config = FeatureConfig(
            name=f"{feature1_name}_{feature2_name}_correlation_{window}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Rolling correlation between {feature1_name} and {feature2_name} over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'feature1_name': feature1_name,
                'feature2_name': feature2_name,
                'window': window,
                'base_calculation': base_calculation
            }
        )
        super().__init__(config)
        self.feature1_name = feature1_name
        self.feature2_name = feature2_name
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate correlation interaction."""
        # This is a placeholder - in practice, you'd get both features
        # and calculate their rolling correlation
        feature1 = data['close'].pct_change()  # Placeholder
        feature2 = data['close'].rolling(20).mean()  # Placeholder
        
        # Calculate rolling correlation
        correlation = feature1.rolling(window=self.window).corr(feature2)
        
        return correlation