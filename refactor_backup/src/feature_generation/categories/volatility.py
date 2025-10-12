"""
Advanced Volatility Feature Generator

This module provides feature generators for advanced volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Fully optimized with VectorBT for maximum performance.

Key Features:
- VectorBT-optimized volatility calculations
- Advanced volatility indicators
- Memory-efficient processing
- GPU acceleration support
- Comprehensive volatility analysis
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

class VolatilityFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Advanced feature generator for volatility-based features with VectorBT optimization."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None, base_calculation: Optional[BaseCalculationType] = None):
        self.period = period
        self.base_calculation = base_calculation
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"advanced_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Advanced volatility measure over {period} periods with VectorBT optimization",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                "period": period
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility feature using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volatility_{self.period}')

        close_prices = data['close'].astype(float)
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        if len(returns) < self.period:
            return pd.Series(np.nan, index=data.index, name=f'volatility_{self.period}')
        
        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer:
            try:
                volatility = self.rolling_optimizer.rolling_std(returns, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                # Align with original data index
                volatility = volatility.reindex(data.index)
                return volatility
            except Exception as e:
                self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1
        
        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                volatility = rolling_std(returns, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                # Align with original data index
                volatility = volatility.reindex(data.index)
                return volatility
            except Exception as e:
                self.logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1
        
        # Final fallback to pandas
        volatility = returns.rolling(window=self.period).std()
        return volatility.reindex(data.index)

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(self.period, 1)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'close_history': close_history
            }
            self.update_state(state_update)


class VectorBTVolatilityFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility feature generator with comprehensive indicators."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volatility_comprehensive_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized comprehensive volatility features over {period} periods",
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
        """Generate comprehensive volatility features using VectorBT."""
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_{self.period}')
        
        try:
            # Calculate returns for volatility
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < self.period:
                return pd.Series(np.nan, index=data.index, name=f'vectorbt_volatility_{self.period}')
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    # Calculate multiple volatility measures
                    volatility_std = self.rolling_optimizer.rolling_std(returns, window=self.period)
                    volatility_var = self.rolling_optimizer.rolling_var(returns, window=self.period)
                    
                    # Combine volatility measures
                    volatility = (volatility_std + volatility_var) / 2
                    
                    # Align with original data index
                    volatility = volatility.reindex(data.index)
                    return volatility
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility_std = rolling_std(returns, window=self.period)
                    volatility_var = rolling_var(returns, window=self.period)
                    
                    # Combine volatility measures
                    volatility = (volatility_std + volatility_var) / 2
                    
                    # Align with original data index
                    volatility = volatility.reindex(data.index)
                    return volatility
                except Exception as e:
                    self.logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = returns.rolling(window=self.period).std()
            return volatility.reindex(data.index)
            
        except Exception as e:
            self.logger.error(f"Error generating volatility features: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_volatility_{self.period}')


class VectorBTBollingerBandsGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Bollinger Bands generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_bbands_{period}_{std_dev}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Bollinger Bands over {period} periods with {std_dev} std dev",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_bbands_{self.period}')
        
        # Generate Bollinger Bands using VectorBT
        bb_result = self._vectorbt_technical_indicator(data, 'bbands_percent', 
                                                     window=self.period, 
                                                     alpha=self.std_dev)
        
        return bb_result.rename(f'vectorbt_bbands_{self.period}')


class VectorBTAverageTrueRangeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Average True Range generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_atr_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Average True Range over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ATR using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_atr_{self.period}')
        
        # Generate ATR using VectorBT
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        
        return atr.rename(f'vectorbt_atr_{self.period}')


class VectorBTGarmanKlassVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Garman-Klass Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_garman_klass_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Garman-Klass Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Garman-Klass Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_garman_klass_volatility_{self.period}')
        
        try:
            # Calculate Garman-Klass volatility components
            log_hl = np.log(data['high'] / data['low'])
            log_co = np.log(data['close'] / data['open'])
            
            # Garman-Klass formula: 0.5 * (log(high/low))^2 - (2*log(2)-1) * (log(close/open))^2
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(gk_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(gk_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Garman-Klass calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = gk_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Garman-Klass volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_garman_klass_volatility_{self.period}')


class VectorBTParkinsonVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Parkinson Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_parkinson_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Parkinson Volatility over {period} periods",
            required_columns=["high", "low"],
            optional_columns=["open", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parkinson Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_parkinson_volatility_{self.period}')
        
        try:
            # Calculate Parkinson volatility: (1/(4*ln(2))) * ln(high/low)^2
            log_hl = np.log(data['high'] / data['low'])
            parkinson_volatility = (1 / (4 * np.log(2))) * log_hl**2
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(parkinson_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(parkinson_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Parkinson calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = parkinson_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Parkinson volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_parkinson_volatility_{self.period}')


class VectorBTRogersSatchellVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Rogers-Satchell Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rogers_satchell_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Rogers-Satchell Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Rogers-Satchell Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rogers_satchell_volatility_{self.period}')
        
        try:
            # Calculate Rogers-Satchell volatility components
            log_ho = np.log(data['high'] / data['open'])
            log_hc = np.log(data['high'] / data['close'])
            log_lo = np.log(data['low'] / data['open'])
            log_lc = np.log(data['low'] / data['close'])
            
            # Rogers-Satchell formula
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(rs_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(rs_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Rogers-Satchell calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = rs_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Rogers-Satchell volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_rogers_satchell_volatility_{self.period}')


class VectorBTYangZhangVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Yang-Zhang Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_yang_zhang_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Yang-Zhang Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Yang-Zhang Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_yang_zhang_volatility_{self.period}')
        
        try:
            # Calculate Yang-Zhang volatility components
            # Overnight volatility
            log_co = np.log(data['close'] / data['open'])
            overnight_vol = log_co**2
            
            # Rogers-Satchell volatility (already calculated above)
            log_ho = np.log(data['high'] / data['open'])
            log_hc = np.log(data['high'] / data['close'])
            log_lo = np.log(data['low'] / data['open'])
            log_lc = np.log(data['low'] / data['close'])
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Garman-Klass volatility
            log_hl = np.log(data['high'] / data['low'])
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Yang-Zhang formula: overnight + Rogers-Satchell + Garman-Klass
            yz_volatility = overnight_vol + rs_volatility + gk_volatility
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(yz_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(yz_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Yang-Zhang calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = yz_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Yang-Zhang volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_yang_zhang_volatility_{self.period}')


def create_default_volatility_generators() -> List[FeatureGenerator]:
    """Create default volatility feature generators with VectorBT optimization."""
    generators = []
    
    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VectorBTVolatilityFeatureGenerator(period))
            generators.append(VectorBTAverageTrueRangeGenerator(period))
            
        # Bollinger Bands with different parameters
        for period in [20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                generators.append(VectorBTBollingerBandsGenerator(period, std_dev))
        
        # Advanced volatility indicators
        for period in [10, 14, 20, 30]:
            generators.append(VectorBTGarmanKlassVolatilityGenerator(period))
            generators.append(VectorBTParkinsonVolatilityGenerator(period))
            generators.append(VectorBTRogersSatchellVolatilityGenerator(period))
            generators.append(VectorBTYangZhangVolatilityGenerator(period))
    else:
        # Fallback to original generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VolatilityFeatureGenerator(period))
    
    return generators