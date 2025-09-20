"""
Cross-Timeframe Feature Generators

This module provides feature generators for cross-timeframe analysis,
capturing relationships and patterns across different time horizons.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class CrossTimeframeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for cross-timeframe features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="cross_timeframe_features",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description="Comprehensive cross-timeframe features across multiple time horizons",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=30,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "timeframes": [5, 15, 30],
                "feature_types": ["momentum", "volatility", "volume", "trend"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

# Cross-Timeframe Momentum Generator
class CrossTimeframeMomentumGenerator(FeatureGenerator):
    """Generator for cross-timeframe momentum features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_momentum_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe momentum over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe momentum."""
        base_values = self.base_calculator.calculate(data)
        momentum = base_values.pct_change(self.timeframe)
        return momentum

# Cross-Timeframe Volatility Generator
class CrossTimeframeVolatilityGenerator(FeatureGenerator):
    """Generator for cross-timeframe volatility features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_volatility_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe volatility over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe volatility."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.timeframe).std()
        return volatility

# Cross-Timeframe Volume Generator
class CrossTimeframeVolumeGenerator(FeatureGenerator):
    """Generator for cross-timeframe volume features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_volume_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe volume over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe volume."""
        base_values = self.base_calculator.calculate(data)
        volume_ma = base_values.rolling(window=self.timeframe).mean()
        return volume_ma

# Cross-Timeframe Trend Generator
class CrossTimeframeTrendGenerator(FeatureGenerator):
    """Generator for cross-timeframe trend features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_trend_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe trend over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe trend."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        trend = base_values.rolling(window=self.timeframe).apply(calculate_trend_strength, raw=False)
        return trend

# Cross-Timeframe High-Low Generator
class CrossTimeframeHighLowGenerator(FeatureGenerator):
    """Generator for cross-timeframe high-low range features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = ["high", "low"]
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_hl_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe high-low range over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe high-low range."""
        hl_range = (data['high'] - data['low']).rolling(window=self.timeframe).mean()
        return hl_range

# Cross-Timeframe Ratio Generator
class CrossTimeframeRatioGenerator(FeatureGenerator):
    """Generator for cross-timeframe ratio features."""
    
    def __init__(self, short_timeframe: int = 5, long_timeframe: int = 20, feature_type: str = "momentum", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_ratio_{feature_type}_{short_timeframe}_{long_timeframe}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe {feature_type} ratio between {short_timeframe} and {long_timeframe} periods",
            required_columns=required_columns,
            default_lookback=long_timeframe,
            min_lookback=long_timeframe,
            max_lookback=long_timeframe,
            parameters={'short_timeframe': short_timeframe, 'long_timeframe': long_timeframe, 'feature_type': feature_type, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe ratio."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            short_feature = base_values.pct_change(self.short_timeframe)
            long_feature = base_values.pct_change(self.long_timeframe)
        elif self.feature_type == "volatility":
            short_feature = base_values.rolling(window=self.short_timeframe).std()
            long_feature = base_values.rolling(window=self.long_timeframe).std()
        elif self.feature_type == "sma":
            short_feature = base_values.rolling(window=self.short_timeframe).mean()
            long_feature = base_values.rolling(window=self.long_timeframe).mean()
        else:  # Default to momentum
            short_feature = base_values.pct_change(self.short_timeframe)
            long_feature = base_values.pct_change(self.long_timeframe)
        
        # Calculate ratio with safe division
        ratio = short_feature / (long_feature + 1e-8)  # Add small epsilon to prevent division by zero
        return ratio

# Cross-Timeframe Correlation Generator
class CrossTimeframeCorrelationGenerator(FeatureGenerator):
    """Generator for cross-timeframe correlation features."""
    
    def __init__(self, timeframe1: int = 5, timeframe2: int = 15, feature_type: str = "momentum", correlation_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_corr_{feature_type}_{timeframe1}_{timeframe2}_{correlation_window}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe correlation of {feature_type} between {timeframe1} and {timeframe2} periods over {correlation_window} window",
            required_columns=required_columns,
            default_lookback=max(timeframe1, timeframe2, correlation_window),
            min_lookback=max(timeframe1, timeframe2, correlation_window),
            max_lookback=max(timeframe1, timeframe2, correlation_window),
            parameters={'timeframe1': timeframe1, 'timeframe2': timeframe2, 'feature_type': feature_type, 'correlation_window': correlation_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.feature_type = feature_type
        self.correlation_window = correlation_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe correlation."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            feature1 = base_values.pct_change(self.timeframe1)
            feature2 = base_values.pct_change(self.timeframe2)
        elif self.feature_type == "volatility":
            feature1 = base_values.rolling(window=self.timeframe1).std()
            feature2 = base_values.rolling(window=self.timeframe2).std()
        else:  # Default to momentum
            feature1 = base_values.pct_change(self.timeframe1)
            feature2 = base_values.pct_change(self.timeframe2)
        
        # Calculate rolling correlation
        correlation = feature1.rolling(window=self.correlation_window).corr(feature2)
        return correlation

# Cross-Timeframe Divergence Generator
class CrossTimeframeDivergenceGenerator(FeatureGenerator):
    """Generator for cross-timeframe divergence features."""
    
    def __init__(self, short_timeframe: int = 5, long_timeframe: int = 20, feature_type: str = "momentum", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_divergence_{feature_type}_{short_timeframe}_{long_timeframe}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe {feature_type} divergence between {short_timeframe} and {long_timeframe} periods",
            required_columns=required_columns,
            default_lookback=long_timeframe,
            min_lookback=long_timeframe,
            max_lookback=long_timeframe,
            parameters={'short_timeframe': short_timeframe, 'long_timeframe': long_timeframe, 'feature_type': feature_type, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe divergence."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            short_feature = base_values.pct_change(self.short_timeframe)
            long_feature = base_values.pct_change(self.long_timeframe)
        elif self.feature_type == "volatility":
            short_feature = base_values.rolling(window=self.short_timeframe).std()
            long_feature = base_values.rolling(window=self.long_timeframe).std()
        elif self.feature_type == "sma":
            short_feature = base_values.rolling(window=self.short_timeframe).mean()
            long_feature = base_values.rolling(window=self.long_timeframe).mean()
        else:  # Default to momentum
            short_feature = base_values.pct_change(self.short_timeframe)
            long_feature = base_values.pct_change(self.long_timeframe)
        
        # Calculate divergence (difference)
        divergence = short_feature - long_feature
        return divergence

def create_cross_timeframe_generators() -> List[FeatureGenerator]:
    """Create all cross-timeframe feature generators."""
    generators = []
    
    # Cross-timeframe momentum for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeMomentumGenerator(timeframe=timeframe))
    
    # Cross-timeframe volatility for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeVolatilityGenerator(timeframe=timeframe))
    
    # Cross-timeframe volume for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeVolumeGenerator(timeframe=timeframe))
    
    # Cross-timeframe trend for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeTrendGenerator(timeframe=timeframe))
    
    # Cross-timeframe high-low for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeHighLowGenerator(timeframe=timeframe))
    
    # Cross-timeframe ratios
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=5, long_timeframe=20, feature_type="momentum"))
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=5, long_timeframe=20, feature_type="volatility"))
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=10, long_timeframe=50, feature_type="sma"))
    
    # Cross-timeframe correlations
    generators.append(CrossTimeframeCorrelationGenerator(timeframe1=5, timeframe2=15, feature_type="momentum", correlation_window=20))
    generators.append(CrossTimeframeCorrelationGenerator(timeframe1=15, timeframe2=30, feature_type="volatility", correlation_window=20))
    
    # Cross-timeframe divergences
    generators.append(CrossTimeframeDivergenceGenerator(short_timeframe=5, long_timeframe=20, feature_type="momentum"))
    generators.append(CrossTimeframeDivergenceGenerator(short_timeframe=5, long_timeframe=20, feature_type="volatility"))
    
    return generators

def create_default_cross_timeframe_generators() -> List[FeatureGenerator]:
    """Create default set of cross-timeframe generators."""
    return create_cross_timeframe_generators()

# Export all generators
__all__ = [
    'CrossTimeframeFeatureGenerator',
    'CrossTimeframeMomentumGenerator',
    'CrossTimeframeVolatilityGenerator',
    'CrossTimeframeVolumeGenerator',
    'CrossTimeframeTrendGenerator',
    'CrossTimeframeHighLowGenerator',
    'CrossTimeframeRatioGenerator',
    'CrossTimeframeCorrelationGenerator',
    'CrossTimeframeDivergenceGenerator',
    'create_cross_timeframe_generators',
    'create_default_cross_timeframe_generators'
]