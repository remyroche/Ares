"""
Feature Interaction Generators

This module provides feature generators for feature interactions, combinations,
and derived features that capture relationships between different indicators.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
import warnings
import logging

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Unified Vectorization Manager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager,
        OperationType,
        OptimizationStrategy
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class OptimizedInteractionFeatureGenerator(VectorizedFeatureGenerator):
    """Optimized feature generator for interaction-based features using VectorBT and UnifiedVectorizationManager."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optimization components
        self.rolling_optimizer = None
        self.unified_manager = None
        
        if OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="optimized_interaction_features",
            category=FeatureCategory.INTERACTION,
            description="VectorBT-optimized interaction features between different indicators",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "interaction_types": ["momentum_divergence", "momentum_volume", "momentum_volatility", "volatility_volume"],
                "optimization_strategy": "vectorbt_parallel"
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBTRollingOptimizer with advanced features."""
        if self.rolling_optimizer:
            try:
                if operation == 'mean':
                    return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.rolling_optimizer.rolling_quantile(data, window, q, **kwargs)
                elif operation == 'skew':
                    return self.rolling_optimizer.rolling_skew(data, window, **kwargs)
                elif operation == 'kurt':
                    return self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer failed: {e}, using fallback")
        
        # Fallback to basic pandas operations
        return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing using UnifiedVectorizationManager."""
        if self.unified_manager:
            try:
                # Use unified manager for intelligent optimization
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=1024.0,
                    precision_requirement="medium"
                )
                
                # Optimize data types and structure
                optimized_data = self.unified_manager.optimize_dataframe(data, config)
                return optimized_data
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager optimization failed: {e}")
        
        # Fallback to basic optimization
        return data.copy()


                except Exception as e:
                    pass
class InteractionFeatureGenerator(OptimizedInteractionFeatureGenerator):
    """Feature generator for interaction-based features (backward compatibility)."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="interaction_features",
            category=FeatureCategory.INTERACTION,
            description="Comprehensive interaction features between different indicators",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "interaction_types": ["momentum_divergence", "momentum_volume", "momentum_volatility", "volatility_volume"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

# Momentum Divergence Generator
    
class MomentumDivergenceGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for momentum divergence between price and volume using VectorBT optimization."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"momentum_divergence_{period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized momentum divergence between price and volume over {period} periods",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum divergence using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        divergence = price_momentum - volume_momentum
        return divergence

# Momentum Volume Generator
    
class MomentumVolumeGenerator(FeatureGenerator):
    """Generator for momentum-volume interaction."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"momentum_volume_{period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Momentum-volume interaction over {period} periods",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        self.use_vectorbt = True
        self.vectorbt_threshold = 1000
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum-volume interaction using VectorBT."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        interaction = price_momentum * volume_momentum
        return interaction

# Momentum Volatility Generator
    
class MomentumVolatilityGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for momentum-volatility interaction using VectorBT optimization."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_volatility_{period}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized momentum-volatility interaction over {period} periods with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=max(period, volatility_window),
            min_lookback=max(period, volatility_window),
            max_lookback=max(period, volatility_window),
            parameters={'period': period, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum-volatility interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        # Normalize momentum by volatility
        interaction = price_momentum / (volatility + 1e-8)  # Add small epsilon to prevent division by zero
        return interaction

    def generate_advanced_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced interaction features using VectorBT optimization.
        
        This method creates sophisticated interaction features that capture complex
        relationships between different indicators and market conditions.
        
        Args:
            data: OHLCV data with additional features
            
        Returns:
            DataFrame of advanced interaction features
        """
        features = {}
        
        # Advanced momentum interactions
        momentum_features = self._generate_advanced_momentum_interactions(data)
        features.update(momentum_features)
        
        # Advanced volatility interactions
        volatility_features = self._generate_advanced_volatility_interactions(data)
        features.update(volatility_features)
        
        # Advanced volume interactions
        if 'volume' in data.columns:
            volume_features = self._generate_advanced_volume_interactions(data)
            features.update(volume_features)
        
        # Advanced technical indicator interactions
        tech_features = self._generate_advanced_technical_interactions(data)
        features.update(tech_features)
        
        # Advanced regime interactions
        regime_features = self._generate_advanced_regime_interactions(data)
        features.update(regime_features)
        
        return pd.DataFrame(features, index=data.index)

    def _generate_advanced_momentum_interactions(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced momentum interaction features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        try:
            # Multi-timeframe momentum analysis (15m-based timeframes)
            timeframes = [15, 30, 60, 120]  # 15m, 30m, 1h, 2h
            momentum_series = {}
            
            for tf in timeframes:
                if tf < len(close):
                    momentum_series[tf] = close.pct_change(tf)
            
            # Momentum convergence/divergence analysis
            if len(momentum_series) >= 3:
                momentum_df = pd.DataFrame(momentum_series)
                
                # Momentum convergence score
                momentum_corr = momentum_df.rolling(window=20).corr()
                convergence_score = momentum_corr.mean().mean()
                
                if self._is_valid_feature(convergence_score):
                    features['momentum_convergence_score'] = pd.Series(
                        [convergence_score] * len(close), 
                        index=close.index, 
                        name='momentum_convergence_score'
                    )
                
                # Momentum divergence detection
                momentum_std = momentum_df.std(axis=1)
                if self._is_valid_feature(momentum_std):
                    features['momentum_divergence_std'] = momentum_std.rename('momentum_divergence_std')
            
            # Momentum acceleration
            if len(momentum_series) >= 2:
                short_momentum = momentum_series.get(5)
                long_momentum = momentum_series.get(20)
                
                if short_momentum is not None and long_momentum is not None:
                    momentum_acceleration = short_momentum.diff() - long_momentum.diff()
                    if self._is_valid_feature(momentum_acceleration):
                        features['momentum_acceleration'] = momentum_acceleration.rename('momentum_acceleration')
            
        except Exception as e:
            self.logger.warning(f"Advanced momentum interactions failed: {e}")
        
        return features

    def _generate_advanced_volatility_interactions(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced volatility interaction features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        try:
            # Multi-timeframe volatility analysis (15m-based timeframes)
            timeframes = [15, 30, 60, 120]  # 15m, 30m, 1h, 2h
            volatility_series = {}
            
            for tf in timeframes:
                if tf < len(close):
                    returns = close.pct_change()
                    volatility_series[tf] = self._optimized_rolling_operation(returns, 'std', tf)
            
            # Volatility regime detection
            if len(volatility_series) >= 2:
                vol_df = pd.DataFrame(volatility_series)
                
                # Volatility clustering
                vol_clustering = vol_df.rolling(window=10).corr().mean().mean()
                if self._is_valid_feature(vol_clustering):
                    features['volatility_clustering'] = pd.Series(
                        [vol_clustering] * len(close), 
                        index=close.index, 
                        name='volatility_clustering'
                    )
                
                # Volatility mean reversion
                short_vol = volatility_series.get(5)
                long_vol = volatility_series.get(20)
                
                if short_vol is not None and long_vol is not None:
                    vol_mean_reversion = (short_vol - long_vol) / (long_vol + 1e-08)
                    if self._is_valid_feature(vol_mean_reversion):
                        features['volatility_mean_reversion'] = vol_mean_reversion.rename('volatility_mean_reversion')
            
            # Volatility of volatility
            if len(volatility_series) >= 1:
                main_vol = list(volatility_series.values())[0]
                vol_of_vol = self._optimized_rolling_operation(main_vol, 'std', 20)
                
                if self._is_valid_feature(vol_of_vol):
                    features['volatility_of_volatility'] = vol_of_vol.rename('volatility_of_volatility')
            
        except Exception as e:
            self.logger.warning(f"Advanced volatility interactions failed: {e}")
        
        return features

    def _generate_advanced_volume_interactions(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced volume interaction features."""
        features = {}
        
        if 'close' not in data.columns or 'volume' not in data.columns:
            return features
        
        close = data['close']
        volume = data['volume']
        
        try:
            # Price-Volume interaction analysis
            price_change = close.pct_change()
            volume_change = volume.pct_change()
            
            # Price-Volume correlation
            price_volume_corr = self._optimized_rolling_operation(
                price_change, 'corr', 20, other=volume_change
            )
            
            if self._is_valid_feature(price_volume_corr):
                features['price_volume_correlation'] = price_volume_corr.rename('price_volume_correlation')
            
            # Volume-weighted price momentum
            vwap = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
            price_vwap_ratio = close / (vwap + 1e-08)
            
            if self._is_valid_feature(price_vwap_ratio):
                features['price_vwap_ratio'] = price_vwap_ratio.rename('price_vwap_ratio')
            
            # Volume momentum divergence
            price_momentum = close.pct_change(5)
            volume_momentum = volume.pct_change(5)
            volume_divergence = price_momentum - volume_momentum
            
            if self._is_valid_feature(volume_divergence):
                features['volume_momentum_divergence'] = volume_divergence.rename('volume_momentum_divergence')
            
            # Volume volatility interaction
            volume_volatility = self._optimized_rolling_operation(volume, 'std', 20)
            price_volatility = self._optimized_rolling_operation(close.pct_change(), 'std', 20)
            
            vol_vol_interaction = volume_volatility * price_volatility
            if self._is_valid_feature(vol_vol_interaction):
                features['volume_volatility_interaction'] = vol_vol_interaction.rename('volume_volatility_interaction')
            
        except Exception as e:
            self.logger.warning(f"Advanced volume interactions failed: {e}")
        
        return features

    def _generate_advanced_technical_interactions(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced technical indicator interaction features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        try:
            # RSI calculation
            rsi_14 = self._calculate_rsi(close, 14)
            
            # MACD calculation
            macd_line = self._calculate_macd(close, 12, 26)
            
            if rsi_14 is not None and macd_line is not None:
                # RSI-MACD divergence
                rsi_macd_divergence = rsi_14 - macd_line
                if self._is_valid_feature(rsi_macd_divergence):
                    features['rsi_macd_divergence'] = rsi_macd_divergence.rename('rsi_macd_divergence')
                
                # RSI-MACD momentum
                rsi_macd_momentum = rsi_14 * macd_line
                if self._is_valid_feature(rsi_macd_momentum):
                    features['rsi_macd_momentum'] = rsi_macd_momentum.rename('rsi_macd_momentum')
            
            # Bollinger Bands analysis
            bb_window = 20
            bb_std = 2.0
            
            sma = self._optimized_rolling_operation(close, 'mean', bb_window)
            std_dev = self._optimized_rolling_operation(close, 'std', bb_window)
            
            upper_band = sma + (std_dev * bb_std)
            lower_band = sma - (std_dev * bb_std)
            bb_width = upper_band - lower_band
            bb_position = (close - lower_band) / (bb_width + 1e-08)
            
            # Bollinger Bands squeeze detection
            bb_squeeze = bb_width < self._optimized_rolling_operation(bb_width, 'mean', 20) * 0.8
            if self._is_valid_feature(bb_squeeze):
                features['bollinger_squeeze'] = bb_squeeze.astype(float).rename('bollinger_squeeze')
            
            # Bollinger Bands position
            if self._is_valid_feature(bb_position):
                features['bollinger_position'] = bb_position.rename('bollinger_position')
            
        except Exception as e:
            self.logger.warning(f"Advanced technical interactions failed: {e}")
        
        return features

    def _generate_advanced_regime_interactions(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced regime interaction features."""
        features = {}
        
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        try:
            # Market regime detection
            short_ma = self._optimized_rolling_operation(close, 'mean', 20)
            long_ma = self._optimized_rolling_operation(close, 'mean', 50)
            
            # Trend regime
            trend_regime = np.where(close > short_ma, 1, np.where(close < short_ma, -1, 0))
            trend_regime_series = pd.Series(trend_regime, index=close.index)
            
            # Volatility regime
            returns = close.pct_change()
            volatility = self._optimized_rolling_operation(returns, 'std', 20)
            vol_regime = np.where(volatility > volatility.rolling(50).mean(), 1, 0)
            vol_regime_series = pd.Series(vol_regime, index=close.index)
            
            # Regime interaction
            regime_interaction = trend_regime_series * vol_regime_series
            if self._is_valid_feature(regime_interaction):
                features['trend_volatility_regime_interaction'] = regime_interaction.rename('trend_volatility_regime_interaction')
            
            # Regime persistence
            regime_persistence = trend_regime_series.rolling(window=10).apply(lambda x: len(set(x)) == 1, raw=False)
            if self._is_valid_feature(regime_persistence):
                features['regime_persistence'] = regime_persistence.rename('regime_persistence')
            
        except Exception as e:
            self.logger.warning(f"Advanced regime interactions failed: {e}")
        
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using optimized rolling operations."""
        if len(prices) < period:
            return pd.Series([50.0] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        gains_series = pd.Series(gains, index=prices.index)
        losses_series = pd.Series(losses, index=prices.index)
        
        avg_gains = gains_series.ewm(span=period, adjust=False).mean()
        avg_losses = losses_series.ewm(span=period, adjust=False).mean()
        
        rs = avg_gains / (avg_losses + 1e-08)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate MACD using optimized rolling operations."""
        if len(prices) < max(fast_period, slow_period):
            return pd.Series([0.0] * len(prices), index=prices.index)
        
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        return fast_ema - slow_ema

    def _is_valid_feature(self, series: pd.Series) -> bool:
        """Check if a feature series is valid."""
        if series is None or series.empty:
            return False
        
        # Check for all NaN values
        if series.isna().all():
            return False
        
        # Check for infinite values
        if np.isinf(series).any():
            return False
        
        # Check for constant values (no variance)
        if series.nunique() <= 1:
            return False
        
        return True

            except Exception as e:
                pass
# Momentum Trend Generator
    
class MomentumTrendGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for momentum-trend interaction using VectorBT optimization."""
    
    def __init__(self, momentum_period: int = 5, trend_window: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_trend_{momentum_period}_{trend_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized momentum-trend interaction over {momentum_period} momentum periods with {trend_window} trend window",
            required_columns=required_columns,
            default_lookback=max(momentum_period, trend_window),
            min_lookback=max(momentum_period, trend_window),
            max_lookback=max(momentum_period, trend_window),
            parameters={'momentum_period': momentum_period, 'trend_window': trend_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.momentum_period = momentum_period
        self.trend_window = trend_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum-trend interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.momentum_period)
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        # Use optimized rolling apply
        trend_strength = self._optimized_rolling_operation(
            base_values, 'apply', self.trend_window, func=calculate_trend_strength
        )
        
        interaction = price_momentum * trend_strength
        return interaction

                except Exception as e:
                    pass
# Volatility Volume Generator
    
class VolatilityVolumeGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for volatility-volume interaction using VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, volume_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"volatility_volume_{volatility_window}_{volume_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized volatility-volume interaction with {volatility_window} volatility window and {volume_window} volume window",
            required_columns=required_columns,
            default_lookback=max(volatility_window, volume_window),
            min_lookback=max(volatility_window, volume_window),
            max_lookback=max(volatility_window, volume_window),
            parameters={'volatility_window': volatility_window, 'volume_window': volume_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-volume interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        volume_ma = self._optimized_rolling_operation(data['volume'], 'mean', self.volume_window)
        interaction = volatility * volume_ma
        return interaction

# Volatility Price Generator
    
class VolatilityPriceGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for volatility-price interaction using VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_price_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized volatility-price interaction with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=volatility_window,
            min_lookback=volatility_window,
            max_lookback=volatility_window,
            parameters={'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-price interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        # Use close price for interaction
        if 'close' in data.columns:
            interaction = volatility * data['close']
        else:
            interaction = volatility * base_values
        return interaction

# Volatility High-Low Generator
    
class VolatilityHighLowGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for volatility-high-low range interaction using VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["high", "low", "close"]
        
        config = FeatureConfig(
            name=f"volatility_hl_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized volatility-high-low range interaction with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=volatility_window,
            min_lookback=volatility_window,
            max_lookback=volatility_window,
            parameters={'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-high-low range interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        hl_range_pct = (data['high'] - data['low']) / data['close']
        interaction = volatility * hl_range_pct
        return interaction

# Volatility Momentum Generator
    
class VolatilityMomentumGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for volatility-momentum interaction using VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, momentum_period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_momentum_{volatility_window}_{momentum_period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized volatility-momentum interaction with {volatility_window} volatility window and {momentum_period} momentum period",
            required_columns=required_columns,
            default_lookback=max(volatility_window, momentum_period),
            min_lookback=max(volatility_window, momentum_period),
            max_lookback=max(volatility_window, momentum_period),
            parameters={'volatility_window': volatility_window, 'momentum_period': momentum_period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.momentum_period = momentum_period
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-momentum interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        momentum = base_values.pct_change(self.momentum_period)
        interaction = volatility * momentum
        return interaction

# Volatility Trend Generator
    
class VolatilityTrendGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for volatility-trend interaction using VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, trend_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_trend_{volatility_window}_{trend_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized volatility-trend interaction with {volatility_window} volatility window and {trend_window} trend window",
            required_columns=required_columns,
            default_lookback=max(volatility_window, trend_window),
            min_lookback=max(volatility_window, trend_window),
            max_lookback=max(volatility_window, trend_window),
            parameters={'volatility_window': volatility_window, 'trend_window': trend_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-trend interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        volatility = self._optimized_rolling_operation(base_values, 'std', self.volatility_window)
        
def create_interaction_generators() -> List[FeatureGenerator]:
    """Create all interaction feature generators."""
    generators = []
    
    # Momentum divergence
    generators.append(MomentumDivergenceGenerator(period=5))
    
    # Momentum-volume interaction
    generators.append(MomentumVolumeGenerator(period=5))
    
    # Momentum-volatility interaction
    generators.append(MomentumVolatilityGenerator(period=5, volatility_window=20))
    
    # Momentum-trend interaction
    generators.append(MomentumTrendGenerator(momentum_period=5, trend_window=10))
    
    # Volatility-volume interaction
    generators.append(VolatilityVolumeGenerator(volatility_window=20, volume_window=20))
    
    # Volatility-price interaction
    generators.append(VolatilityPriceGenerator(volatility_window=20))
    
    # Volatility-high-low interaction
    generators.append(VolatilityHighLowGenerator(volatility_window=20))
    
    # Volatility-momentum interaction
    generators.append(VolatilityMomentumGenerator(volatility_window=20, momentum_period=20))
    
    # Volatility-trend interaction
    generators.append(VolatilityTrendGenerator(volatility_window=20, trend_window=20))
    
    return generators


# Legacy Interaction Generators
# =============================

    
class CrossTimeframeInteractionGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for cross-timeframe feature interactions using VectorBT optimization."""
    
    def __init__(self, short_period: int = 5, long_period: int = 20, interaction_type: str = "ratio", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cross_timeframe_{interaction_type}_{short_period}_{long_period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized cross-timeframe {interaction_type} interaction between {short_period} and {long_period} periods",
            required_columns=required_columns,
            default_lookback=max(short_period, long_period),
            min_lookback=max(short_period, long_period),
            max_lookback=max(short_period, long_period),
            parameters={'short_period': short_period, 'long_period': long_period, 'interaction_type': interaction_type, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
        self.interaction_type = interaction_type
        self.base_calculation = base_calculation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)
        
        # Calculate short and long period features using optimized rolling operations
        short_ma = self._optimized_rolling_operation(base_values, 'mean', self.short_period)
        long_ma = self._optimized_rolling_operation(base_values, 'mean', self.long_period)
        
        if self.interaction_type == "ratio":
            interaction = short_ma / (long_ma + 1e-8)  # Add small epsilon to prevent division by zero
        elif self.interaction_type == "difference":
            interaction = short_ma - long_ma
        elif self.interaction_type == "momentum":
            interaction = (short_ma - long_ma) / (long_ma + 1e-8)
        elif self.interaction_type == "crossover":
            # Binary signal: 1 if short > long, 0 otherwise
            interaction = (short_ma > long_ma).astype(float)
        else:
            raise ValueError(f"Unknown interaction_type: {self.interaction_type}")
        
        return interaction


    
class FeatureRatioGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for ratios between different features using VectorBT optimization."""
    
    def __init__(self, numerator_column: str = "close", denominator_column: str = "volume", window: int = 1):
        config = FeatureConfig(
            name=f"ratio_{numerator_column}_{denominator_column}_{window}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized ratio between {numerator_column} and {denominator_column} with {window} period smoothing",
            required_columns=[numerator_column, denominator_column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'numerator_column': numerator_column, 'denominator_column': denominator_column, 'window': window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.window = window
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate feature ratio using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        numerator = data[self.numerator_column]
        denominator = data[self.denominator_column]
        
        if self.window > 1:
            numerator = self._optimized_rolling_operation(numerator, 'mean', self.window)
            denominator = self._optimized_rolling_operation(denominator, 'mean', self.window)
        
        ratio = numerator / (denominator + 1e-8)  # Add small epsilon to prevent division by zero
        return ratio


    
class PolynomialFeatureGenerator(FeatureGenerator):
    """Generator for polynomial transformations of features."""
    
    def __init__(self, column: str = "close", degree: int = 2, include_bias: bool = False):
        config = FeatureConfig(
            name=f"poly_{column}_deg{degree}{'_bias' if include_bias else ''}",
            category=FeatureCategory.INTERACTION,
            description=f"Polynomial transformation of {column} with degree {degree}",
            required_columns=[column],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={'column': column, 'degree': degree, 'include_bias': include_bias}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.column = column
        self.degree = degree
        self.include_bias = include_bias
        self.use_vectorbt = True
        self.vectorbt_threshold = 1000
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
                                      pass
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
                                     pass
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate polynomial transformation using VectorBT."""
        values = data[self.column]
        
        # Normalize values to prevent numerical overflow using VectorBT if available
        if self._should_use_vectorbt(values):
            try:
                values_mean = rolling_mean(values, window=len(values)).iloc[-1] if len(values) > 0 else values.mean()
                values_std = rolling_std(values, window=len(values)).iloc[-1] if len(values) > 0 else values.std()
                values_normalized = (values - values_mean) / (values_std + 1e-8)
            except Exception:
                values_normalized = (values - values.mean()) / (values.std() + 1e-8)
        else:
            values_normalized = (values - values.mean()) / (values.std() + 1e-8)
        
        # Create polynomial features
        result = values_normalized ** self.degree
        
        if self.include_bias:
            result = result + 1.0
        
        return result


    
class CorrelationInteractionGenerator(OptimizedInteractionFeatureGenerator):
    """Generator for correlation-based feature interactions using VectorBT optimization."""
    
    def __init__(self, column1: str = "close", column2: str = "volume", window: int = 20, method: str = "pearson"):
        config = FeatureConfig(
            name=f"corr_{column1}_{column2}_{window}_{method}",
            category=FeatureCategory.INTERACTION,
            description=f"VectorBT-optimized {method.capitalize()} correlation between {column1} and {column2} over {window} periods",
            required_columns=[column1, column2],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'column1': column1, 'column2': column2, 'window': window, 'method': method},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.column1 = column1
        self.column2 = column2
        self.window = window
        self.method = method
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate correlation interaction using VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)

        values1 = data[self.column1]
        values2 = data[self.column2]
        
        # Calculate rolling correlation using optimized rolling operations
        correlation = self._optimized_rolling_operation(
            values1, 'corr', self.window, other=values2
        )
        
        return correlation


def create_default_interaction_generators() -> List[FeatureGenerator]:
    """Create default set of legacy interaction generators."""
    generators = []
    
    # Cross-timeframe interactions
    generators.append(CrossTimeframeInteractionGenerator(short_period=5, long_period=20, interaction_type="ratio"))
    generators.append(CrossTimeframeInteractionGenerator(short_period=10, long_period=50, interaction_type="momentum"))
    
    # Feature ratios
    generators.append(FeatureRatioGenerator(numerator_column="close", denominator_column="volume"))
    generators.append(FeatureRatioGenerator(numerator_column="high", denominator_column="low"))
    
    # Polynomial features
    generators.append(PolynomialFeatureGenerator(column="close", degree=2))
    generators.append(PolynomialFeatureGenerator(column="volume", degree=2))
    
    # Correlation interactions
    generators.append(CorrelationInteractionGenerator(column1="close", column2="volume", window=20))
    generators.append(CorrelationInteractionGenerator(column1="high", column2="low", window=10))
    
    return generators

# Export all generators
__all__ = [
    # Optimized generators
    'OptimizedInteractionFeatureGenerator',
    'InteractionFeatureGenerator',
    'MomentumDivergenceGenerator',
    'MomentumVolumeGenerator',
    'MomentumVolatilityGenerator',
    'MomentumTrendGenerator',
    'VolatilityVolumeGenerator',
    'VolatilityPriceGenerator',
    'VolatilityHighLowGenerator',
    'VolatilityMomentumGenerator',
    'VolatilityTrendGenerator',
    'create_interaction_generators',
    # Legacy interaction generators (now optimized)
    'CrossTimeframeInteractionGenerator',
    'FeatureRatioGenerator',
    'PolynomialFeatureGenerator',
    'CorrelationInteractionGenerator',
    'create_default_interaction_generators'
    # Note: RegimeDependentFeatureGenerator, CointegrationResidualGenerator,
    # StructuralRatioGenerator, PairwiseInteractionGenerator are not implemented yet
    # They are referenced but not defined - removed from exports to prevent import errors

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
]
_should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
                                      pass
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
                                     pass
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

            except Exception as e:
                pass