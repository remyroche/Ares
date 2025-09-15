"""
Entropy Feature Generator

This module provides feature generators for entropy-based indicators,
including price, volume, and return entropy features.
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

class EntropyFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for entropy-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="entropy_features",
            category=FeatureCategory.ENTROPY,
            description="Comprehensive entropy features including price, volume, and return entropy",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "entropy_windows": [5, 10, 20],
                "entropy_types": ["price", "volume", "return"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'EntropyFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        entropy = np.zeros_like(close_prices)
        return pd.Series(entropy, index=data.index, name='entropy_placeholder')

# Price Entropy Generator
class PriceEntropyGenerator(FeatureGenerator):
    """Generator for price entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"price_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Price entropy over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price entropy."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        price_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        return price_entropy

# Volume Entropy Generator
class VolumeEntropyGenerator(FeatureGenerator):
    """Generator for volume entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Volume entropy over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume entropy."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        volume_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        return volume_entropy

# Return Entropy Generator
class ReturnEntropyGenerator(FeatureGenerator):
    """Generator for return entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"return_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Return entropy over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate return entropy."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        return return_entropy

# Price Entropy MA Generator
class PriceEntropyMAGenerator(FeatureGenerator):
    """Generator for price entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"price_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Price entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price entropy MA."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        price_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        price_entropy_ma = price_entropy.rolling(window=self.ma_window).mean()
        return price_entropy_ma

# Volume Entropy MA Generator
class VolumeEntropyMAGenerator(FeatureGenerator):
    """Generator for volume entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Volume entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume entropy MA."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        volume_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        volume_entropy_ma = volume_entropy.rolling(window=self.ma_window).mean()
        return volume_entropy_ma

# Return Entropy MA Generator
class ReturnEntropyMAGenerator(FeatureGenerator):
    """Generator for return entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"return_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Return entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate return entropy MA."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                # Discretize the series into bins
                bins = np.histogram(series, bins=10)[0]
                # Normalize to get probabilities
                probs = bins / np.sum(bins)
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return_entropy = base_values.rolling(window=self.window).apply(calculate_entropy, raw=False)
        return_entropy_ma = return_entropy.rolling(window=self.ma_window).mean()
        return return_entropy_ma

def create_default_entropy_generators() -> List[FeatureGenerator]:
    """Create default entropy feature generators."""
    windows = [5, 10, 20]
    ma_windows = [5, 10]
    
    generators = []
    
    # Create generators for each window
    for window in windows:
        generators.extend([
            PriceEntropyGenerator(window),
            VolumeEntropyGenerator(window),
            ReturnEntropyGenerator(window),
        ])
        
        # Create MA generators
        for ma_window in ma_windows:
            generators.extend([
                PriceEntropyMAGenerator(window, ma_window),
                VolumeEntropyMAGenerator(window, ma_window),
                ReturnEntropyMAGenerator(window, ma_window),
            ])
    
    return generators