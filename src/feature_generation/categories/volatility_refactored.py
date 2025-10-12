"""
Refactored Volatility Feature Generator

This module provides refactored volatility feature generators that use centralized
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

class RefactoredVolatilityFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored volatility feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Refactored volatility features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "volatility_periods": [20, 50, 100],
                "bollinger_periods": [20],
                "atr_periods": [14],
                "volatility_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Use common operations for comprehensive volatility calculation
        volatility_measures = self.calculate_volatility_measures(data, window=20)
        
        # Use price volatility as primary feature
        if 'price_volatility' in volatility_measures:
            result = volatility_measures['price_volatility']
        else:
            # Fallback to simple returns volatility
            close = data['close']
            returns = close.pct_change()
            result = self.rolling_std(returns, window=20)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            result = self.normalize_feature(result, feature_type='volatility')
        
        return result.rename('refactored_volatility_20')

class RefactoredBollingerBandsGenerator(UnifiedFeatureGenerator):
    """Refactored Bollinger Bands generator using centralized utilities."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int, std_dev: float) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_bollinger_bands_{period}_{std_dev}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Bollinger Bands using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate SMA and standard deviation using centralized rolling operations
        sma = self.rolling_mean(close, window=self.period)
        std = self.rolling_std(close, window=self.period)
        
        # Calculate Bollinger Bands
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        # Calculate band width
        band_width = upper_band - lower_band
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            band_width = self.normalize_feature(band_width, feature_type='volatility')
        
        return band_width.rename(f'refactored_bollinger_width_{self.period}_{self.std_dev}')

class RefactoredATRGenerator(UnifiedFeatureGenerator):
    """Refactored ATR generator using centralized utilities."""
    
    def __init__(self, period: int = 14, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_atr_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored ATR with period {period} using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ATR using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using centralized rolling operations
        atr = self.rolling_mean(true_range, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            atr = self.normalize_feature(atr, feature_type='volatility')
        
        return atr.rename(f'refactored_atr_{self.period}')

class RefactoredGarmanKlassVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored Garman-Klass Volatility generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_garman_klass_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Garman-Klass Volatility with period {period} using centralized utilities",
            required_columns=["open", "high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Garman-Klass Volatility using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Garman-Klass volatility
        gk_volatility = 0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open) ** 2)
        
        # Calculate rolling mean using centralized rolling operations
        volatility = self.rolling_mean(gk_volatility, window=self.period)
        volatility = np.sqrt(volatility)  # Convert variance to volatility
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volatility = self.normalize_feature(volatility, feature_type='volatility')
        
        return volatility.rename(f'refactored_garman_klass_volatility_{self.period}')

class RefactoredParkinsonVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored Parkinson Volatility generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_parkinson_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Parkinson Volatility with period {period} using centralized utilities",
            required_columns=["high", "low"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parkinson Volatility using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        
        # Calculate Parkinson volatility
        parkinson_volatility = (1 / (4 * np.log(2))) * (np.log(high / low) ** 2)
        
        # Calculate rolling mean using centralized rolling operations
        volatility = self.rolling_mean(parkinson_volatility, window=self.period)
        volatility = np.sqrt(volatility)  # Convert variance to volatility
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volatility = self.normalize_feature(volatility, feature_type='volatility')
        
        return volatility.rename(f'refactored_parkinson_volatility_{self.period}')

class RefactoredRogersSatchellVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored Rogers-Satchell Volatility generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_rogers_satchell_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Rogers-Satchell Volatility with period {period} using centralized utilities",
            required_columns=["open", "high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Rogers-Satchell Volatility using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Rogers-Satchell volatility
        rs_volatility = (np.log(high / close) * np.log(high / open) + 
                        np.log(low / close) * np.log(low / open))
        
        # Calculate rolling mean using centralized rolling operations
        volatility = self.rolling_mean(rs_volatility, window=self.period)
        volatility = np.sqrt(volatility)  # Convert variance to volatility
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volatility = self.normalize_feature(volatility, feature_type='volatility')
        
        return volatility.rename(f'refactored_rogers_satchell_volatility_{self.period}')

class RefactoredYangZhangVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored Yang-Zhang Volatility generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_yang_zhang_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Yang-Zhang Volatility with period {period} using centralized utilities",
            required_columns=["open", "high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Yang-Zhang Volatility using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Yang-Zhang volatility components
        # Close-to-close volatility
        cc_volatility = (np.log(close / close.shift(1)) ** 2)
        
        # Open-to-close volatility
        oc_volatility = (np.log(open / close.shift(1)) ** 2)
        
        # Garman-Klass volatility
        gk_volatility = 0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open) ** 2)
        
        # Rogers-Satchell volatility
        rs_volatility = (np.log(high / close) * np.log(high / open) + 
                        np.log(low / close) * np.log(low / open))
        
        # Yang-Zhang volatility
        yz_volatility = cc_volatility + oc_volatility + gk_volatility + rs_volatility
        
        # Calculate rolling mean using centralized rolling operations
        volatility = self.rolling_mean(yz_volatility, window=self.period)
        volatility = np.sqrt(volatility)  # Convert variance to volatility
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volatility = self.normalize_feature(volatility, feature_type='volatility')
        
        return volatility.rename(f'refactored_yang_zhang_volatility_{self.period}')

class RefactoredVolatilityOfVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored Volatility of Volatility generator using centralized utilities."""
    
    def __init__(self, inner_period: int = 20, outer_period: int = 50, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(inner_period, outer_period)
        super().__init__(config)
        self.inner_period = inner_period
        self.outer_period = outer_period
    
    @classmethod
    def _create_default_config(cls, inner_period: int, outer_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volatility_of_volatility_{inner_period}_{outer_period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Refactored Volatility of Volatility using centralized utilities",
            required_columns=["close"],
            default_lookback=outer_period + inner_period,
            min_lookback=outer_period + inner_period,
            max_lookback=100,
            parameters={"inner_period": inner_period, "outer_period": outer_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volatility of Volatility using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate inner volatility using centralized rolling operations
        inner_volatility = self.rolling_std(returns, window=self.inner_period)
        
        # Calculate volatility of volatility using centralized rolling operations
        volatility_of_volatility = self.rolling_std(inner_volatility, window=self.outer_period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volatility_of_volatility = self.normalize_feature(volatility_of_volatility, feature_type='volatility')
        
        return volatility_of_volatility.rename(f'refactored_volatility_of_volatility_{self.inner_period}_{self.outer_period}')

# Batch volatility generator for multiple features
class RefactoredBatchVolatilityGenerator(UnifiedFeatureGenerator):
    """Refactored batch volatility generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_batch_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Refactored batch volatility features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "volatility_periods": [20, 50, 100],
                "bollinger_periods": [20],
                "atr_periods": [14],
                "volatility_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volatility',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch volatility features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Use common operations for comprehensive volatility calculation
        volatility_measures = self.calculate_volatility_measures(data, window=20)
        
        if volatility_measures:
            # Use price volatility as primary feature
            primary_feature = volatility_measures.get('price_volatility', pd.Series(dtype=float, index=data.index))
            result = primary_feature.rename('refactored_batch_volatility')
            
            # Apply normalization if enabled
            if self.unified_config.auto_normalize:
                result = self.normalize_feature(result, feature_type='volatility')
            
            return result
        else:
            # Fallback to simple returns volatility
            close = data['close']
            returns = close.pct_change()
            volatility = self.rolling_std(returns, window=20)
            
            if self.unified_config.auto_normalize:
                volatility = self.normalize_feature(volatility, feature_type='volatility')
            
            return volatility.rename('refactored_batch_volatility_fallback')