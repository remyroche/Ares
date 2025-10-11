"""
VectorBT-Optimized Legacy Feature Generators

This module provides high-performance legacy feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Legacy features are traditional technical indicators that have been used in 
financial analysis for decades. These include classic indicators like:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators

These features maintain backward compatibility with existing trading systems
and provide a baseline for comparison with newer, enhanced indicators.

All legacy generators now use VectorBT's optimized C++ backend for optimal performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)

class VectorBTLegacyRSIGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy RSI generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_rsi_{period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy RSI {period} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy RSI using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_rsi_{self.period}')
        
        close = data['close']
        
        # Use VectorBT RSI for optimized calculation
        rsi = self._vectorbt_technical_indicator(data, 'rsi', window=self.period)
        
        return rsi.rename(f'vectorbt_legacy_rsi_{self.period}')


class VectorBTLegacyMACDGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy MACD generator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast, slow, signal)
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @classmethod
    def _create_default_config(cls, fast: int = 12, slow: int = 26, signal: int = 9) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy MACD {fast}/{slow}/{signal} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={"fast": fast, "slow": slow, "signal": signal},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy MACD using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_macd_{self.fast}_{self.slow}_{self.signal}')
        
        # Use VectorBT MACD for optimized calculation
        macd = self._vectorbt_technical_indicator(data, 'macd', 
                                                fast_window=self.fast, 
                                                slow_window=self.slow, 
                                                signal_window=self.signal)
        
        return macd.rename(f'vectorbt_legacy_macd_{self.fast}_{self.slow}_{self.signal}')


class VectorBTLegacyMACDSignalGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy MACD signal generator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast, slow, signal)
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @classmethod
    def _create_default_config(cls, fast: int = 12, slow: int = 26, signal: int = 9) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_macd_signal_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy MACD signal {fast}/{slow}/{signal} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={"fast": fast, "slow": slow, "signal": signal},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy MACD signal using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_macd_signal_{self.fast}_{self.slow}_{self.signal}')
        
        # Use VectorBT MACD signal for optimized calculation
        macd_signal = self._vectorbt_technical_indicator(data, 'macd_signal', 
                                                       fast_window=self.fast, 
                                                       slow_window=self.slow, 
                                                       signal_window=self.signal)
        
        return macd_signal.rename(f'vectorbt_legacy_macd_signal_{self.fast}_{self.slow}_{self.signal}')


class VectorBTLegacyMACDHistogramGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy MACD histogram generator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast, slow, signal)
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @classmethod
    def _create_default_config(cls, fast: int = 12, slow: int = 26, signal: int = 9) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_macd_histogram_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy MACD histogram {fast}/{slow}/{signal} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={"fast": fast, "slow": slow, "signal": signal},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy MACD histogram using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_macd_histogram_{self.fast}_{self.slow}_{self.signal}')
        
        # Use VectorBT MACD histogram for optimized calculation
        macd_histogram = self._vectorbt_technical_indicator(data, 'macd_histogram', 
                                                          fast_window=self.fast, 
                                                          slow_window=self.slow, 
                                                          signal_window=self.signal)
        
        return macd_histogram.rename(f'vectorbt_legacy_macd_histogram_{self.fast}_{self.slow}_{self.signal}')


class VectorBTLegacyBollingerBandsUpperGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Bollinger Bands upper generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_bollinger_upper_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Bollinger Bands upper {period}/{std_dev} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Bollinger Bands upper using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_bollinger_upper_{self.period}_{self.std_dev}')
        
        # Use VectorBT Bollinger Bands upper for optimized calculation
        bb_upper = self._vectorbt_technical_indicator(data, 'bbands_upper', 
                                                    window=self.period, 
                                                    alpha=self.std_dev)
        
        return bb_upper.rename(f'vectorbt_legacy_bollinger_upper_{self.period}_{self.std_dev}')


class VectorBTLegacyBollingerBandsLowerGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Bollinger Bands lower generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_bollinger_lower_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Bollinger Bands lower {period}/{std_dev} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Bollinger Bands lower using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_bollinger_lower_{self.period}_{self.std_dev}')
        
        # Use VectorBT Bollinger Bands lower for optimized calculation
        bb_lower = self._vectorbt_technical_indicator(data, 'bbands_lower', 
                                                    window=self.period, 
                                                    alpha=self.std_dev)
        
        return bb_lower.rename(f'vectorbt_legacy_bollinger_lower_{self.period}_{self.std_dev}')


class VectorBTLegacyBollingerBandsMiddleGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Bollinger Bands middle generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_bollinger_middle_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Bollinger Bands middle {period}/{std_dev} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Bollinger Bands middle using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_bollinger_middle_{self.period}_{self.std_dev}')
        
        # Use VectorBT Bollinger Bands middle for optimized calculation
        bb_middle = self._vectorbt_technical_indicator(data, 'bbands_middle', 
                                                     window=self.period, 
                                                     alpha=self.std_dev)
        
        return bb_middle.rename(f'vectorbt_legacy_bollinger_middle_{self.period}_{self.std_dev}')


class VectorBTLegacySMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy SMA generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_sma_{period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy SMA {period} - traditional implementation",
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
        """Generate legacy SMA using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_sma_{self.period}')
        
        close = data['close']
        
        # Use VectorBT rolling mean for SMA
        sma = self._vectorbt_rolling_operation(close, 'mean', window=self.period)
        
        return sma.rename(f'vectorbt_legacy_sma_{self.period}')


class VectorBTLegacyEMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy EMA generator."""
    
    def __init__(self, period: int = 21, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 21) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_ema_{period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy EMA {period} - traditional implementation",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy EMA using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_ema_{self.period}')
        
        close = data['close']
        
        # Use VectorBT EMA calculation
        ema = close.ewm(span=self.period).mean()
        
        return ema.rename(f'vectorbt_legacy_ema_{self.period}')


class VectorBTLegacyATRGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy ATR generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_atr_{period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy ATR {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy ATR using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_atr_{self.period}')
        
        # Use VectorBT ATR for optimized calculation
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        
        return atr.rename(f'vectorbt_legacy_atr_{self.period}')


class VectorBTLegacyStochasticKGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Stochastic %K generator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(k_period, d_period)
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
    
    @classmethod
    def _create_default_config(cls, k_period: int = 14, d_period: int = 3) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_stochastic_k_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Stochastic %K {k_period}/{d_period} - traditional implementation",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period,
            parameters={"k_period": k_period, "d_period": d_period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Stochastic %K using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_stochastic_k_{self.k_period}_{self.d_period}')
        
        # Use VectorBT Stochastic %K for optimized calculation
        stoch_k = self._vectorbt_technical_indicator(data, 'stoch_k', 
                                                   k_window=self.k_period, 
                                                   d_window=self.d_period)
        
        return stoch_k.rename(f'vectorbt_legacy_stochastic_k_{self.k_period}_{self.d_period}')


class VectorBTLegacyStochasticDGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Stochastic %D generator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(k_period, d_period)
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
    
    @classmethod
    def _create_default_config(cls, k_period: int = 14, d_period: int = 3) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_stochastic_d_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Stochastic %D {k_period}/{d_period} - traditional implementation",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period,
            parameters={"k_period": k_period, "d_period": d_period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Stochastic %D using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_stochastic_d_{self.k_period}_{self.d_period}')
        
        # Use VectorBT Stochastic %D for optimized calculation
        stoch_d = self._vectorbt_technical_indicator(data, 'stoch_d', 
                                                   k_window=self.k_period, 
                                                   d_window=self.d_period)
        
        return stoch_d.rename(f'vectorbt_legacy_stochastic_d_{self.k_period}_{self.d_period}')


class VectorBTLegacyWilliamsRGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Williams %R generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_williams_r_{period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Williams %R {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy Williams %R using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_williams_r_{self.period}')
        
        # Use VectorBT Williams %R for optimized calculation
        willr = self._vectorbt_technical_indicator(data, 'willr', window=self.period)
        
        return willr.rename(f'vectorbt_legacy_williams_r_{self.period}')


class VectorBTLegacyOBVGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy OBV generator."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="vectorbt_legacy_obv",
            category=FeatureCategory.LEGACY,
            description="VectorBT-optimized legacy OBV - traditional implementation",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate legacy OBV using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='vectorbt_legacy_obv')
        
        # Use VectorBT OBV for optimized calculation
        obv = self._vectorbt_technical_indicator(data, 'obv')
        
        return obv.rename('vectorbt_legacy_obv')


def create_vectorbt_legacy_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized legacy feature generators."""
    generators = []
    
    # Classic indicators with standard parameters
    generators.extend([
        VectorBTLegacyRSIGenerator(14),
        VectorBTLegacyMACDGenerator(12, 26, 9),
        VectorBTLegacyMACDSignalGenerator(12, 26, 9),
        VectorBTLegacyMACDHistogramGenerator(12, 26, 9),
        VectorBTLegacyBollingerBandsUpperGenerator(20, 2.0),
        VectorBTLegacyBollingerBandsLowerGenerator(20, 2.0),
        VectorBTLegacyBollingerBandsMiddleGenerator(20, 2.0),
        VectorBTLegacySMAGenerator(20),
        VectorBTLegacyEMAGenerator(21),
        VectorBTLegacyATRGenerator(14),
        VectorBTLegacyStochasticKGenerator(14, 3),
        VectorBTLegacyStochasticDGenerator(14, 3),
        VectorBTLegacyWilliamsRGenerator(14),
        VectorBTLegacyOBVGenerator(),
    ])
    
    # Additional legacy moving averages
    sma_periods = [5, 10, 50, 100, 200]
    for period in sma_periods:
        generators.append(VectorBTLegacySMAGenerator(period))
    
    # Additional legacy EMAs
    ema_periods = [8, 12, 26, 50, 100]
    for period in ema_periods:
        generators.append(VectorBTLegacyEMAGenerator(period))
    
    # Additional legacy RSI periods
    rsi_periods = [9, 21, 25]
    for period in rsi_periods:
        generators.append(VectorBTLegacyRSIGenerator(period))
    
    # Additional Bollinger Bands variations
    bb_periods = [10, 30, 50]
    bb_std_devs = [1.5, 2.5]
    for period in bb_periods:
        for std_dev in bb_std_devs:
            generators.extend([
                VectorBTLegacyBollingerBandsUpperGenerator(period, std_dev),
                VectorBTLegacyBollingerBandsLowerGenerator(period, std_dev),
                VectorBTLegacyBollingerBandsMiddleGenerator(period, std_dev),
            ])
    
    # Additional MACD variations
    macd_configs = [(8, 17, 9), (19, 39, 9)]
    for fast, slow, signal in macd_configs:
        generators.extend([
            VectorBTLegacyMACDGenerator(fast, slow, signal),
            VectorBTLegacyMACDSignalGenerator(fast, slow, signal),
            VectorBTLegacyMACDHistogramGenerator(fast, slow, signal),
        ])
    
    return generators


def create_default_vectorbt_legacy_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized legacy feature generators."""
    return create_vectorbt_legacy_generators()


# Export all generators
__all__ = [
    'VectorBTLegacyRSIGenerator',
    'VectorBTLegacyMACDGenerator',
    'VectorBTLegacyMACDSignalGenerator',
    'VectorBTLegacyMACDHistogramGenerator',
    'VectorBTLegacyBollingerBandsUpperGenerator',
    'VectorBTLegacyBollingerBandsLowerGenerator',
    'VectorBTLegacyBollingerBandsMiddleGenerator',
    'VectorBTLegacySMAGenerator',
    'VectorBTLegacyEMAGenerator',
    'VectorBTLegacyATRGenerator',
    'VectorBTLegacyStochasticKGenerator',
    'VectorBTLegacyStochasticDGenerator',
    'VectorBTLegacyWilliamsRGenerator',
    'VectorBTLegacyOBVGenerator',
    'create_vectorbt_legacy_generators',
    'create_default_vectorbt_legacy_generators'
]