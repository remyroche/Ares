"""
Cross-Timeframe Feature Generators

This module provides feature generators for cross-timeframe analysis,
capturing relationships and patterns across different time horizons.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class CrossTimeframeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for cross-timeframe features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
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
                "timeframes": [1, 5, 15, 30, 60],
                "feature_types": ["momentum", "volatility", "volume", "trend", "range"],
                "lag_handling": True,
                "fractional_changes": True,
                "learned_projections": True,
                "regime_aware": True,
                "alignment_methods": ["lag", "resample", "interpolate"],
                "projection_methods": ["pca", "autoencoder", "patchtst"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

# Cross-Timeframe Momentum Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe momentum."""
        base_values = self.base_calculator.calculate(data)
        momentum = base_values.pct_change(self.timeframe)
        return momentum

# Cross-Timeframe Volatility Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe volatility."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.timeframe).std()
        return volatility

# Cross-Timeframe Volume Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe volume."""
        base_values = self.base_calculator.calculate(data)
        volume_ma = base_values.rolling(window=self.timeframe).mean()
        return volume_ma

# Cross-Timeframe Trend Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe high-low range."""
        hl_range = (data['high'] - data['low']).rolling(window=self.timeframe).mean()
        return hl_range

# Cross-Timeframe Ratio Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.feature_type = feature_type
        self.correlation_window = correlation_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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

# Enhanced Cross-Timeframe Generators for Better Aggregation

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class CrossTimeframeFractionalChangeGenerator(FeatureGenerator):
    """Generator for fractional change features across timeframes."""

    def __init__(self, fast_tf: int = 5, slow_tf: int = 15, feature_type: str = "volatility"):
        config = FeatureConfig(
            name=f"ctf_fractional_{feature_type}_{fast_tf}m_{slow_tf}m",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Fractional change of {feature_type} from {fast_tf}m to {slow_tf}m timeframe",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=max(fast_tf, slow_tf),
            min_lookback=max(fast_tf, slow_tf),
            max_lookback=max(fast_tf, slow_tf),
            parameters={"fast_tf": fast_tf, "slow_tf": slow_tf, "feature_type": feature_type}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_tf = fast_tf
        self.slow_tf = slow_tf
        self.feature_type = feature_type

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate fractional change feature across timeframes."""
        if self.feature_type == "volatility":
            fast_vol = data["close"].pct_change().rolling(window=self.fast_tf).std()
            slow_vol = data["close"].pct_change().rolling(window=self.slow_tf).std()
            fractional_change = fast_vol / slow_vol
        elif self.feature_type == "momentum":
            fast_momentum = data["close"].pct_change(self.fast_tf)
            slow_momentum = data["close"].pct_change(self.slow_tf)
            fractional_change = fast_momentum / slow_momentum
        elif self.feature_type == "volume":
            if "volume" in data.columns:
                fast_volume = data["volume"].rolling(window=self.fast_tf).mean()
                slow_volume = data["volume"].rolling(window=self.slow_tf).mean()
                fractional_change = fast_volume / slow_volume
            else:
                fractional_change = pd.Series(np.zeros(len(data)), index=data.index)
        else:
            fractional_change = pd.Series(np.zeros(len(data)), index=data.index)

        return fractional_change.fillna(0)


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class CrossTimeframeAlignmentGenerator(FeatureGenerator):
    """Generator for properly aligned cross-timeframe features."""

    def __init__(self, source_tf: int = 1, target_tf: int = 5, alignment_method: str = "lag"):
        config = FeatureConfig(
            name=f"ctf_aligned_{source_tf}m_to_{target_tf}m",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Align {source_tf}m features to {target_tf}m timeframe using {alignment_method}",
            required_columns=["close"],
            default_lookback=target_tf,
            min_lookback=target_tf,
            max_lookback=target_tf,
            parameters={"source_tf": source_tf, "target_tf": target_tf, "alignment_method": alignment_method}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.source_tf = source_tf
        self.target_tf = target_tf
        self.alignment_method = alignment_method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate properly aligned cross-timeframe feature."""
        # Calculate lag needed for alignment
        lag_bars = self.target_tf // self.source_tf - 1

        if self.alignment_method == "lag":
            # Lag fast timeframe features by appropriate number of bars
            returns = data["close"].pct_change()
            aligned_returns = returns.shift(lag_bars)
            return aligned_returns.fillna(0)
        elif self.alignment_method == "resample":
            # Resample to target timeframe
            resampled = data["close"].resample(f'{self.target_tf}min').last()
            # Forward fill to original frequency
            aligned = resampled.reindex(data.index, method='ffill')
            return (aligned / aligned.shift(1) - 1).fillna(0)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class CrossTimeframeLearnedProjectionGenerator(FeatureGenerator):
    """Generator for learned projections across timeframes using PCA/dimensionality reduction."""

    def __init__(self, timeframes: List[int] = [1, 5, 15], n_components: int = 3):
        config = FeatureConfig(
            name=f"ctf_learned_projection_{'_'.join(map(str, timeframes))}_{n_components}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Learned projection across {timeframes} timeframes using {n_components} components",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=max(timeframes) * 10,
            min_lookback=max(timeframes) * 5,
            max_lookback=max(timeframes) * 20,
            parameters={"timeframes": timeframes, "n_components": n_components}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframes = timeframes
        self.n_components = n_components

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate learned projection features across timeframes."""
        try:
            from sklearn.decomposition import PCA

            # Create features for each timeframe
            tf_features = []
            for tf in self.timeframes:
                # Calculate returns for this timeframe
                returns = data["close"].pct_change(tf)

                # Calculate volatility for this timeframe
                volatility = returns.rolling(window=tf).std()

                # Calculate momentum for this timeframe
                momentum = data["close"].pct_change(tf * 5)

                tf_features.append(pd.concat([returns, volatility, momentum], axis=1))

            # Combine features from all timeframes
            feature_matrix = pd.concat(tf_features, axis=1).fillna(0)

            # Apply PCA for dimensionality reduction
            if len(feature_matrix.columns) >= self.n_components:
                pca = PCA(n_components=self.n_components)
                pca_result = pca.fit_transform(feature_matrix)

                # Return first principal component as representative feature
                return pd.Series(pca_result[:, 0], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.warning(f"Error in learned projection: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)


# Enhanced Cross-Timeframe Features

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class EnhancedCrossTimeframeFeatureGenerator(VectorizedFeatureGenerator):
    """Enhanced cross-timeframe feature generator with proper lag handling and fractional changes."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="enhanced_cross_timeframe_features",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description="Enhanced cross-timeframe features with proper lag handling and learned projections",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=100,
            min_lookback=50,
            max_lookback=500,
            parameters={
                "timeframes": [1, 5, 15, 30, 60],
                "feature_types": ["momentum", "volatility", "volume", "trend", "range"],
                "lag_handling": True,
                "fractional_changes": True,
                "learned_projections": True,
                "regime_aware": True,
                "alignment_methods": ["lag", "resample", "interpolate"],
                "projection_methods": ["pca", "autoencoder", "patchtst"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate enhanced cross-timeframe features."""
        try:
            # Generate all enhanced cross-timeframe features
            features_dict = self.generate_enhanced_cross_timeframe_features(data, **kwargs)

            # Return first feature as representative for base class
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.error(f"Error generating enhanced cross-timeframe features: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_enhanced_cross_timeframe_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive enhanced cross-timeframe features."""
        features = {}

        try:
            # Fractional change features with proper lag handling
            features.update(self._generate_fractional_change_features(data))

            # Cross-timeframe alignment features
            features.update(self._generate_alignment_features(data))

            # Learned projection features
            features.update(self._generate_learned_projection_features(data))

            # Regime-aware cross-timeframe features
            features.update(self._generate_regime_aware_cross_timeframe_features(data))

            # Multi-scale correlation features
            features.update(self._generate_multi_scale_correlation_features(data))

            logger.info(f"Generated {len(features)} enhanced cross-timeframe features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_enhanced_cross_timeframe_features: {e}")
            return {}

    def _generate_fractional_change_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate fractional change features across timeframes with proper lag handling."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        feature_types = self.config.parameters.get("feature_types", ["momentum", "volatility", "volume", "trend"])

        for fast_tf in timeframes:
            for slow_tf in timeframes:
                if fast_tf >= slow_tf:
                    continue

                for feature_type in feature_types:
                    # Calculate features with proper lag handling
                    fast_feature = self._calculate_feature_with_lag(data, fast_tf, feature_type)
                    slow_feature = self._calculate_feature_with_lag(data, slow_tf, feature_type)

                    if fast_feature is not None and slow_feature is not None:
                        # Fractional change
                        fractional_change = fast_feature / (slow_feature + 1e-8)
                        features[f"frac_change_{feature_type}_{fast_tf}m_{slow_tf}m"] = fractional_change.fillna(0).values

                        # Relative change
                        relative_change = (fast_feature - slow_feature) / (slow_feature + 1e-8)
                        features[f"rel_change_{feature_type}_{fast_tf}m_{slow_tf}m"] = relative_change.fillna(0).values

                        # Momentum divergence
                        momentum_div = fast_feature - slow_feature
                        features[f"momentum_div_{feature_type}_{fast_tf}m_{slow_tf}m"] = momentum_div.fillna(0).values

        return features

    def _calculate_feature_with_lag(self, data: pd.DataFrame, timeframe: int, feature_type: str) -> Optional[pd.Series]:
        """Calculate feature with proper lag handling to avoid lookahead bias."""
        try:
            if feature_type == "momentum":
                # Calculate momentum with lag
                lag_bars = max(1, timeframe // 5)  # Lag by 20% of timeframe
                returns = data["close"].pct_change(timeframe)
                return returns.shift(lag_bars)

            elif feature_type == "volatility":
                # Calculate volatility with lag
                lag_bars = max(1, timeframe // 5)
                returns = data["close"].pct_change()
                vol = returns.rolling(window=timeframe).std()
                return vol.shift(lag_bars)

            elif feature_type == "volume":
                if "volume" in data.columns:
                    lag_bars = max(1, timeframe // 5)
                    vol_ma = data["volume"].rolling(window=timeframe).mean()
                    return vol_ma.shift(lag_bars)
                else:
                    return None

            elif feature_type == "trend":
                # Calculate trend strength with lag
                lag_bars = max(1, timeframe // 5)
                trend = self._calculate_trend_strength(data["close"], timeframe)
                return trend.shift(lag_bars)

            elif feature_type == "range":
                # Calculate high-low range with lag
                lag_bars = max(1, timeframe // 5)
                if "high" in data.columns and "low" in data.columns:
                    hl_range = (data["high"] - data["low"]).rolling(window=timeframe).mean()
                    return hl_range.shift(lag_bars)
                else:
                    return None

            else:
                return None

        except Exception as e:
            logger.warning(f"Error calculating {feature_type} for timeframe {timeframe}: {e}")
            return None

    def _calculate_trend_strength(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calc_slope(x):
            if len(x) < 2:
                return 0.0
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except:
                return 0.0

        return series.rolling(window=window).apply(calc_slope, raw=False)

    def _generate_alignment_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe alignment features."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        alignment_methods = self.config.parameters.get("alignment_methods", ["lag", "resample", "interpolate"])

        for source_tf in timeframes:
            for target_tf in timeframes:
                if source_tf >= target_tf:
                    continue

                for method in alignment_methods:
                    aligned_feature = self._align_timeframes(data, source_tf, target_tf, method)
                    if aligned_feature is not None:
                        features[f"aligned_{source_tf}m_to_{target_tf}m_{method}"] = aligned_feature.fillna(0).values

        return features

    def _align_timeframes(self, data: pd.DataFrame, source_tf: int, target_tf: int, method: str) -> Optional[pd.Series]:
        """Align features from source timeframe to target timeframe."""
        try:
            if method == "lag":
                # Lag fast timeframe features by appropriate number of bars
                lag_bars = target_tf // source_tf - 1
                returns = data["close"].pct_change()
                return returns.shift(lag_bars)

            elif method == "resample":
                # Resample to target timeframe
                resampled = data["close"].resample(f'{target_tf}min').last()
                # Forward fill to original frequency
                aligned = resampled.reindex(data.index, method='ffill')
                return (aligned / aligned.shift(1) - 1).fillna(0)

            elif method == "interpolate":
                # Interpolate between timeframes
                returns = data["close"].pct_change()
                # Simple interpolation (in practice, would use more sophisticated methods)
                return returns.rolling(window=target_tf//source_tf).mean()

            else:
                return None

        except Exception as e:
            logger.warning(f"Error aligning timeframes {source_tf} to {target_tf} with method {method}: {e}")
            return None

    def _generate_learned_projection_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate learned projection features across timeframes."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        projection_methods = self.config.parameters.get("projection_methods", ["pca", "autoencoder", "patchtst"])

        for method in projection_methods:
            if method == "pca":
                features.update(self._generate_pca_projection_features(data, timeframes))
            elif method == "autoencoder":
                features.update(self._generate_autoencoder_projection_features(data, timeframes))
            elif method == "patchtst":
                features.update(self._generate_patchtst_projection_features(data, timeframes))

        return features

    def _generate_pca_projection_features(self, data: pd.DataFrame, timeframes: List[int]) -> Dict[str, np.ndarray]:
        """Generate PCA projection features across timeframes."""
        features = {}

        try:
            from sklearn.decomposition import PCA

            # Create features for each timeframe
            tf_features = []
            for tf in timeframes:
                # Calculate returns for this timeframe
                returns = data["close"].pct_change(tf).fillna(0)

                # Calculate volatility for this timeframe
                vol = data["close"].pct_change().rolling(window=tf).std().fillna(0)

                # Calculate momentum for this timeframe
                momentum = data["close"].pct_change(tf * 2).fillna(0)

                # Calculate trend for this timeframe
                trend = self._calculate_trend_strength(data["close"], tf).fillna(0)

                tf_features.append(pd.concat([returns, vol, momentum, trend], axis=1))

            # Combine features from all timeframes
            feature_matrix = pd.concat(tf_features, axis=1).fillna(0)

            # Apply PCA for dimensionality reduction
            if len(feature_matrix.columns) >= 3:
                pca = PCA(n_components=min(3, len(feature_matrix.columns)))
                pca_result = pca.fit_transform(feature_matrix)

                for i in range(pca_result.shape[1]):
                    features[f"pca_component_{i+1}"] = pca_result[:, i]

                # Explained variance ratio
                for i, ratio in enumerate(pca.explained_variance_ratio_):
                    features[f"pca_explained_var_{i+1}"] = np.full(len(data), ratio)

        except Exception as e:
            logger.warning(f"Error in PCA projection: {e}")

        return features

    def _generate_autoencoder_projection_features(self, data: pd.DataFrame, timeframes: List[int]) -> Dict[str, np.ndarray]:
        """Generate autoencoder projection features across timeframes."""
        features = {}

        try:
            # Create input features
            input_features = []
            for tf in timeframes:
                returns = data["close"].pct_change(tf).fillna(0)
                vol = data["close"].pct_change().rolling(window=tf).std().fillna(0)
                input_features.extend([returns, vol])

            feature_matrix = pd.concat(input_features, axis=1).fillna(0)

            # Simple autoencoder using PCA as proxy
            if len(feature_matrix.columns) >= 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(2, len(feature_matrix.columns)))
                encoded = pca.fit_transform(feature_matrix)

                for i in range(encoded.shape[1]):
                    features[f"autoencoder_component_{i+1}"] = encoded[:, i]

        except Exception as e:
            logger.warning(f"Error in autoencoder projection: {e}")

        return features

    def _generate_patchtst_projection_features(self, data: pd.DataFrame, timeframes: List[int]) -> Dict[str, np.ndarray]:
        """Generate PatchTST projection features across timeframes."""
        features = {}

        try:
            # Create patches for each timeframe
            patch_length = 16
            num_patches = 8

            for tf in timeframes:
                # Create patches from price sequence
                price_sequence = data["close"].values
                patches = self._create_patches(price_sequence, patch_length, num_patches)

                if patches is not None:
                    # Calculate patch statistics
                    patch_means = patches.mean(axis=1)
                    patch_stds = patches.std(axis=1)
                    patch_trends = np.polyfit(np.arange(patch_length), patches.T, 1)[0]

                    features[f"patchtst_mean_{tf}"] = patch_means
                    features[f"patchtst_std_{tf}"] = patch_stds
                    features[f"patchtst_trend_{tf}"] = patch_trends

        except Exception as e:
            logger.warning(f"Error in PatchTST projection: {e}")

        return features

    def _create_patches(self, sequence: np.ndarray, patch_length: int, num_patches: int) -> Optional[np.ndarray]:
        """Create patches from price sequence."""
        seq_len = len(sequence)
        patch_size = patch_length * num_patches

        if seq_len < patch_size:
            return None

        # Take the most recent data
        recent_data = sequence[-patch_size:]
        
        # Reshape into patches
        patches = recent_data.reshape(num_patches, patch_length)
        return patches

    def _generate_regime_aware_cross_timeframe_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-aware cross-timeframe features."""
        features = {}

        # Detect market regimes
        returns = data["close"].pct_change()
        vol_regime = returns.rolling(window=20).std()
        
        # Define regimes
        low_vol_threshold = vol_regime.quantile(0.33)
        high_vol_threshold = vol_regime.quantile(0.67)
        
        low_vol_regime = (vol_regime <= low_vol_threshold).astype(int)
        high_vol_regime = (vol_regime >= high_vol_threshold).astype(int)

        # Cross-timeframe features for each regime
        timeframes = [5, 15, 30]
        
        for tf1 in timeframes:
            for tf2 in timeframes:
                if tf1 >= tf2:
                    continue

                # Calculate features
                feature1 = data["close"].pct_change(tf1)
                feature2 = data["close"].pct_change(tf2)

                # Low volatility regime features
                low_vol_mask = low_vol_regime == 1
                if low_vol_mask.sum() > 0:
                    low_vol_ratio = np.zeros(len(data))
                    low_vol_ratio[low_vol_mask] = (feature1 / (feature2 + 1e-8))[low_vol_mask]
                    features[f"regime_low_vol_ratio_{tf1}_{tf2}"] = low_vol_ratio

                # High volatility regime features
                high_vol_mask = high_vol_regime == 1
                if high_vol_mask.sum() > 0:
                    high_vol_ratio = np.zeros(len(data))
                    high_vol_ratio[high_vol_mask] = (feature1 / (feature2 + 1e-8))[high_vol_mask]
                    features[f"regime_high_vol_ratio_{tf1}_{tf2}"] = high_vol_ratio

        return features

    def _generate_multi_scale_correlation_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate multi-scale correlation features."""
        features = {}

        timeframes = [5, 10, 20, 50]
        correlation_window = 20

        for tf1 in timeframes:
            for tf2 in timeframes:
                if tf1 >= tf2:
                    continue

                # Calculate features
                feature1 = data["close"].pct_change(tf1)
                feature2 = data["close"].pct_change(tf2)

                # Rolling correlation
                correlation = feature1.rolling(window=correlation_window).corr(feature2)
                features[f"correlation_{tf1}_{tf2}_{correlation_window}"] = correlation.fillna(0).values

                # Correlation stability
                corr_std = correlation.rolling(window=correlation_window).std()
                features[f"corr_stability_{tf1}_{tf2}_{correlation_window}"] = corr_std.fillna(0).values

        return features


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
    'CrossTimeframeFractionalChangeGenerator',
    'CrossTimeframeAlignmentGenerator',
    'CrossTimeframeLearnedProjectionGenerator',
    'EnhancedCrossTimeframeFeatureGenerator',
    'create_cross_timeframe_generators',
    'create_default_cross_timeframe_generators'
]