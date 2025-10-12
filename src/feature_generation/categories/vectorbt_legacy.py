"""
VectorBT-Optimized Legacy Feature Generators

This module provides high-performance legacy feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators
- ATR, Stochastic, Williams %R, OBV
- All legacy indicators with VectorBT optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
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
        """Generate RSI using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_rsi_{self.period}')
        
        close = data['close']
        
        # Use VectorBT for optimized RSI calculation
        rsi = self._calculate_rsi_vectorbt(close)
        
        return rsi.rename(f'vectorbt_legacy_rsi_{self.period}')
    
    def _calculate_rsi_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using VectorBT optimized operations."""
        try:
            # Use VectorBT RSI if available
            import vectorbt as vbt
            rsi_result = vbt.RSI.run(close, window=self.period)
            return rsi_result.rsi.rename(f'vectorbt_legacy_rsi_{self.period}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_rsi_manual(close)
    
    def _calculate_rsi_manual(self, close: pd.Series) -> pd.Series:
        """Calculate RSI manually using VectorBT rolling operations."""
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate rolling means using VectorBT
        avg_gains = self._vectorbt_rolling_operation(gains, 'mean', window=self.period)
        avg_losses = self._vectorbt_rolling_operation(losses, 'mean', window=self.period)
        
        # Calculate RSI
        rs = safe_divide(avg_gains, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


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
        """Generate MACD using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_macd_{self.fast}_{self.slow}_{self.signal}')
        
        close = data['close']
        
        # Use VectorBT for optimized MACD calculation
        macd = self._calculate_macd_vectorbt(close)
        
        return macd.rename(f'vectorbt_legacy_macd_{self.fast}_{self.slow}_{self.signal}')
    
    def _calculate_macd_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate MACD using VectorBT optimized operations."""
        try:
            # Use VectorBT MACD if available
            import vectorbt as vbt
            macd_result = vbt.MACD.run(close, fast_window=self.fast, slow_window=self.slow, signal_window=self.signal)
            return macd_result.macd.rename(f'vectorbt_legacy_macd_{self.fast}_{self.slow}_{self.signal}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_macd_manual(close)
    
    def _calculate_macd_manual(self, close: pd.Series) -> pd.Series:
        """Calculate MACD manually using VectorBT rolling operations."""
        # Calculate EMAs using VectorBT
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        return macd


class VectorBTLegacyBollingerBandsGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Bollinger Bands generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_bollinger_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Bollinger Bands {period}/{std_dev} - traditional implementation",
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
        """Generate Bollinger Bands using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_bollinger_upper_{self.period}_{self.std_dev}')
        
        close = data['close']
        
        # Use VectorBT for optimized Bollinger Bands calculation
        upper_band = self._calculate_bollinger_bands_vectorbt(close)
        
        return upper_band.rename(f'vectorbt_legacy_bollinger_upper_{self.period}_{self.std_dev}')
    
    def _calculate_bollinger_bands_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Bands using VectorBT optimized operations."""
        try:
            # Use VectorBT Bollinger Bands if available
            import vectorbt as vbt
            bb_result = vbt.BBANDS.run(close, window=self.period, alpha=self.std_dev)
            return bb_result.upper.rename(f'vectorbt_legacy_bollinger_upper_{self.period}_{self.std_dev}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_bollinger_bands_manual(close)
    
    def _calculate_bollinger_bands_manual(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Bands manually using VectorBT rolling operations."""
        # Calculate SMA using VectorBT
        sma = self._vectorbt_rolling_operation(close, 'mean', window=self.period)
        
        # Calculate rolling standard deviation using VectorBT
        std = self._vectorbt_rolling_operation(close, 'std', window=self.period)
        
        # Calculate upper band
        upper_band = sma + (std * self.std_dev)
        
        return upper_band


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
        """Generate SMA using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_sma_{self.period}')
        
        close = data['close']
        
        # Use VectorBT for optimized SMA calculation
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
        """Generate EMA using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_ema_{self.period}')
        
        close = data['close']
        
        # Use VectorBT for optimized EMA calculation
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
        """Generate ATR using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_atr_{self.period}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized ATR calculation
        atr = self._calculate_atr_vectorbt(high, low, close)
        
        return atr.rename(f'vectorbt_legacy_atr_{self.period}')
    
    def _calculate_atr_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR using VectorBT optimized operations."""
        try:
            # Use VectorBT ATR if available
            import vectorbt as vbt
            atr_result = vbt.ATR.run(high, low, close, window=self.period)
            return atr_result.atr.rename(f'vectorbt_legacy_atr_{self.period}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_atr_manual(high, low, close)
    
    def _calculate_atr_manual(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR manually using VectorBT rolling operations."""
        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # True Range is the maximum of the three components
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as rolling mean of True Range using VectorBT
        atr = self._vectorbt_rolling_operation(tr, 'mean', window=self.period)
        
        return atr


class VectorBTLegacyStochasticGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized legacy Stochastic generator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(k_period, d_period)
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
    
    @classmethod
    def _create_default_config(cls, k_period: int = 14, d_period: int = 3) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_legacy_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"VectorBT-optimized legacy Stochastic {k_period}/{d_period} - traditional implementation",
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
        """Generate Stochastic using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_stochastic_k_{self.k_period}_{self.d_period}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized Stochastic calculation
        k_percent = self._calculate_stochastic_vectorbt(high, low, close)
        
        return k_percent.rename(f'vectorbt_legacy_stochastic_k_{self.k_period}_{self.d_period}')
    
    def _calculate_stochastic_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Stochastic using VectorBT optimized operations."""
        try:
            # Use VectorBT Stochastic if available
            import vectorbt as vbt
            stoch_result = vbt.STOCH.run(high, low, close, k_window=self.k_period, d_window=self.d_period)
            return stoch_result.stoch_k.rename(f'vectorbt_legacy_stochastic_k_{self.k_period}_{self.d_period}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_stochastic_manual(high, low, close)
    
    def _calculate_stochastic_manual(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Stochastic manually using VectorBT rolling operations."""
        # Calculate rolling min and max using VectorBT
        lowest_low = self._vectorbt_rolling_operation(low, 'min', window=self.k_period)
        highest_high = self._vectorbt_rolling_operation(high, 'max', window=self.k_period)
        
        # Calculate %K
        denominator = highest_high - lowest_low
        k_percent = np.where(
            denominator != 0,
            100 * ((close - lowest_low) / denominator),
            0
        )
        
        return pd.Series(k_percent, index=close.index)


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
        """Generate Williams %R using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_legacy_williams_r_{self.period}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized Williams %R calculation
        williams_r = self._calculate_williams_r_vectorbt(high, low, close)
        
        return williams_r.rename(f'vectorbt_legacy_williams_r_{self.period}')
    
    def _calculate_williams_r_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R using VectorBT optimized operations."""
        try:
            # Use VectorBT Williams %R if available
            import vectorbt as vbt
            willr_result = vbt.WILLR.run(high, low, close, window=self.period)
            return willr_result.willr.rename(f'vectorbt_legacy_williams_r_{self.period}')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_williams_r_manual(high, low, close)
    
    def _calculate_williams_r_manual(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R manually using VectorBT rolling operations."""
        # Calculate rolling min and max using VectorBT
        lowest_low = self._vectorbt_rolling_operation(low, 'min', window=self.period)
        highest_high = self._vectorbt_rolling_operation(high, 'max', window=self.period)
        
        # Calculate Williams %R
        denominator = highest_high - lowest_low
        williams_r = np.where(
            denominator != 0,
            -100 * ((highest_high - close) / denominator),
            0
        )
        
        return pd.Series(williams_r, index=close.index)


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
        """Generate OBV using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='vectorbt_legacy_obv')
        
        close = data['close']
        volume = data['volume']
        
        # Use VectorBT for optimized OBV calculation
        obv = self._calculate_obv_vectorbt(close, volume)
        
        return obv.rename('vectorbt_legacy_obv')
    
    def _calculate_obv_vectorbt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using VectorBT optimized operations."""
        try:
            # Use VectorBT OBV if available
            import vectorbt as vbt
            obv_result = vbt.OBV.run(close, volume)
            return obv_result.obv.rename('vectorbt_legacy_obv')
        except Exception as e:
            # Fallback to manual calculation
            return self._calculate_obv_manual(close, volume)
    
    def _calculate_obv_manual(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV manually using VectorBT operations."""
        # Calculate price changes
        price_change = close.diff()
        
        # Calculate OBV based on price direction
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        
        # Cumulative sum
        obv_cumsum = pd.Series(obv, index=close.index).cumsum()
        
        return obv_cumsum


def create_vectorbt_legacy_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized legacy feature generators."""
    generators = []
    
    # Classic indicators with standard parameters
    generators.extend([
        VectorBTLegacyRSIGenerator(14),
        VectorBTLegacyMACDGenerator(12, 26, 9),
        VectorBTLegacyBollingerBandsGenerator(20, 2.0),
        VectorBTLegacySMAGenerator(20),
        VectorBTLegacyEMAGenerator(21),
        VectorBTLegacyATRGenerator(14),
        VectorBTLegacyStochasticGenerator(14, 3),
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
    
    return generators


def create_default_vectorbt_legacy_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized legacy feature generators."""
    return create_vectorbt_legacy_generators()


# Export all generators
__all__ = [
    'VectorBTLegacyRSIGenerator',
    'VectorBTLegacyMACDGenerator',
    'VectorBTLegacyBollingerBandsGenerator',
    'VectorBTLegacySMAGenerator',
    'VectorBTLegacyEMAGenerator',
    'VectorBTLegacyATRGenerator',
    'VectorBTLegacyStochasticGenerator',
    'VectorBTLegacyWilliamsRGenerator',
    'VectorBTLegacyOBVGenerator',
    'create_vectorbt_legacy_generators',
    'create_default_vectorbt_legacy_generators'
]