"""
VectorBT Base Classes and Integration Layer

This module provides base classes and utilities for VectorBT-enhanced feature engineering,
enabling comprehensive use of VectorBT's capabilities throughout the feature engineering system.
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt")

# Import existing framework components
from src.feature_generation.core.feature_generator import (
    FeatureGenerator, FeatureCategory, FeatureConfig, FeatureResult, VectorizedFeatureGenerator
)
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import safe_divide, safe_mean, safe_std, validate_finite
from src.utils.matrix_operations import vectorized_rolling_features

logger = logging.getLogger(__name__)


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT feature generation."""
    
    # VectorBT specific settings
    enable_optimization: bool = True
    optimization_runs: int = 100
    optimization_method: str = 'grid'  # 'grid', 'random', 'bayesian'
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel: bool = True
    n_jobs: int = -1
    
    # Data validation
    validate_inputs: bool = True
    handle_nan: str = 'forward_fill'  # 'forward_fill', 'backward_fill', 'drop', 'interpolate'
    handle_inf: str = 'replace'  # 'replace', 'drop', 'error'
    
    # Output settings
    return_metadata: bool = True
    include_optimization_results: bool = True
    include_performance_metrics: bool = True


class VectorBTFeatureGenerator(VectorizedFeatureGenerator):
    """
    Base class for VectorBT-enhanced feature generators.
    
    Provides comprehensive VectorBT integration with optimization, caching,
    and performance monitoring capabilities.
    """
    
    def __init__(
        self, 
        config: FeatureConfig, 
        vectorbt_config: Optional[VectorBTConfig] = None,
        **kwargs
    ):
        """
        Initialize VectorBT feature generator.
        
        Args:
            config: Feature configuration
            vectorbt_config: VectorBT specific configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.vectorbt_config = vectorbt_config or VectorBTConfig()
        self.vbt = vbt
        self._cache = {} if self.vectorbt_config.enable_caching else None
        self._optimization_cache = {}
        
        # Initialize VectorBT settings
        self._setup_vectorbt()
        
        tprint_info(f"🔧 VectorBT feature generator initialized: {config.name}")
        tprint_info(f"   → Optimization: {self.vectorbt_config.enable_optimization}")
        tprint_info(f"   → Caching: {self.vectorbt_config.enable_caching}")
        tprint_info(f"   → Parallel: {self.vectorbt_config.enable_parallel}")
    
    def _setup_vectorbt(self) -> None:
        """Setup VectorBT configuration and settings."""
        try:
            # Configure VectorBT settings
            if self.vectorbt_config.enable_parallel:
                vbt.settings.set_theme('dark')
                vbt.settings['array_wrapper']['freq'] = '1min'
                vbt.settings['array_wrapper']['grouping'] = True
                vbt.settings['array_wrapper']['group_by'] = True
                vbt.settings['array_wrapper']['freq'] = '1min'
            
            # Set up optimization settings
            if self.vectorbt_config.enable_optimization:
                vbt.settings['array_wrapper']['freq'] = '1min'
                vbt.settings['array_wrapper']['grouping'] = True
                
        except Exception as e:
            tprint_warning(f"⚠️ Warning setting up VectorBT: {e}")
    
    def _validate_vectorbt_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare data for VectorBT processing.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Validated and cleaned data
        """
        if not self.vectorbt_config.validate_inputs:
            return data
        
        # Create a copy to avoid modifying original data
        validated_data = data.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in validated_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for VectorBT: {missing_columns}")
        
        # Handle NaN values
        if self.vectorbt_config.handle_nan == 'forward_fill':
            validated_data = validated_data.fillna(method='ffill')
        elif self.vectorbt_config.handle_nan == 'backward_fill':
            validated_data = validated_data.fillna(method='bfill')
        elif self.vectorbt_config.handle_nan == 'drop':
            validated_data = validated_data.dropna()
        elif self.vectorbt_config.handle_nan == 'interpolate':
            validated_data = validated_data.interpolate()
        
        # Handle infinite values
        if self.vectorbt_config.handle_inf == 'replace':
            validated_data = validated_data.replace([np.inf, -np.inf], np.nan)
            validated_data = validated_data.fillna(method='ffill')
        elif self.vectorbt_config.handle_inf == 'drop':
            validated_data = validated_data.replace([np.inf, -np.inf], np.nan)
            validated_data = validated_data.dropna()
        elif self.vectorbt_config.handle_inf == 'error':
            if np.isinf(validated_data.select_dtypes(include=[np.number]).values).any():
                raise ValueError("Infinite values found in data")
        
        # Ensure numeric types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(validated_data[col]):
                validated_data[col] = pd.to_numeric(validated_data[col], errors='coerce')
        
        return validated_data
    
    def _get_cache_key(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Generate cache key for data and parameters."""
        if not self.vectorbt_config.enable_caching:
            return None
        
        # Create hash from data shape, index, and parameters
        data_hash = hash((data.shape, str(data.index[0]), str(data.index[-1])))
        params_hash = hash(str(sorted(params.items())))
        return f"{self.config.name}_{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, pd.Series]]:
        """Get result from cache if available."""
        if not self.vectorbt_config.enable_caching or not cache_key:
            return None
        
        if cache_key in self._cache:
            tprint_info(f"📦 Cache hit for {self.config.name}")
            return self._cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: Dict[str, pd.Series]) -> None:
        """Store result in cache."""
        if not self.vectorbt_config.enable_caching or not cache_key:
            return
        
        # Limit cache size
        if len(self._cache) >= self.vectorbt_config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
        tprint_info(f"💾 Cached result for {self.config.name}")
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        param_ranges: Dict[str, List[Any]],
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize parameters using VectorBT's optimization capabilities.
        
        Args:
            data: Input data for optimization
            param_ranges: Parameter ranges to optimize
            target_metric: Target metric for optimization
            
        Returns:
            Optimized parameters
        """
        if not self.vectorbt_config.enable_optimization:
            return self.config.parameters
        
        cache_key = f"opt_{self.config.name}_{hash(str(param_ranges))}"
        if cache_key in self._optimization_cache:
            return self._optimization_cache[cache_key]
        
        try:
            tprint_info(f"🔍 Optimizing parameters for {self.config.name}")
            
            # Create optimization function
            def optimization_func(data, **params):
                return self._evaluate_parameters(data, params, target_metric)
            
            # Run optimization
            if self.vectorbt_config.optimization_method == 'grid':
                result = vbt.run(
                    optimization_func,
                    data,
                    param_ranges,
                    n_jobs=self.vectorbt_config.n_jobs
                )
            else:
                # Use random search for other methods
                result = vbt.run(
                    optimization_func,
                    data,
                    param_ranges,
                    n_jobs=self.vectorbt_config.n_jobs,
                    param_product=False
                )
            
            # Get best parameters
            best_params = result.best_params()
            self._optimization_cache[cache_key] = best_params
            
            tprint_success(f"✅ Optimized parameters: {best_params}")
            return best_params
            
        except Exception as e:
            tprint_error(f"❌ Parameter optimization failed: {e}")
            return self.config.parameters
    
    def _evaluate_parameters(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, Any], 
        target_metric: str
    ) -> float:
        """
        Evaluate parameters for optimization.
        
        Args:
            data: Input data
            params: Parameters to evaluate
            target_metric: Target metric
            
        Returns:
            Metric value
        """
        try:
            # Generate features with given parameters
            features = self.generate_vectorbt_features(data, params)
            
            # Calculate target metric
            if target_metric == 'sharpe_ratio':
                if 'returns' in features:
                    returns = features['returns'].dropna()
                    if len(returns) > 1:
                        return returns.mean() / returns.std() if returns.std() > 0 else 0
                return 0
            elif target_metric == 'information_ratio':
                if 'returns' in features and 'benchmark_returns' in features:
                    excess_returns = features['returns'] - features['benchmark_returns']
                    excess_returns = excess_returns.dropna()
                    if len(excess_returns) > 1:
                        return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
                return 0
            else:
                # Default to feature stability
                feature_values = []
                for feature in features.values():
                    if isinstance(feature, pd.Series):
                        feature_values.extend(feature.dropna().values)
                
                if feature_values:
                    return -np.std(feature_values)  # Negative for minimization
                return 0
                
        except Exception as e:
            tprint_warning(f"⚠️ Parameter evaluation failed: {e}")
            return 0
    
    @abstractmethod
    def generate_vectorbt_features(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate features using VectorBT indicators.
        
        Args:
            data: OHLCV data
            params: Optional parameters override
            
        Returns:
            Dictionary of feature Series
        """
        pass
    
    def generate(self, data: pd.DataFrame, lookback: Optional[int] = None) -> FeatureResult:
        """
        Generate VectorBT-enhanced features.
        
        Args:
            data: OHLCV data
            lookback: Override default lookback period
            
        Returns:
            FeatureResult with generated features
        """
        start_time = time.time()
        
        try:
            # Validate and prepare data
            validated_data = self._validate_vectorbt_data(data)
            
            # Use provided lookback or default
            effective_lookback = lookback or self.config.default_lookback
            
            # Prepare parameters
            params = self.config.parameters.copy()
            if lookback:
                params['lookback'] = effective_lookback
            
            # Check cache
            cache_key = self._get_cache_key(validated_data, params)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                features = cached_result
            else:
                # Generate features using VectorBT
                features = self.generate_vectorbt_features(validated_data, params)
                
                # Store in cache
                if cache_key:
                    self._store_in_cache(cache_key, features)
            
            # Select primary feature
            primary_feature = self._select_primary_feature(features)
            
            computation_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'lookback_used': effective_lookback,
                'all_features': list(features.keys()),
                'feature_stats': self._calculate_feature_stats(primary_feature),
                'vectorbt_optimized': True,
                'cache_used': cached_result is not None
            }
            
            if self.vectorbt_config.include_performance_metrics:
                metadata['performance_metrics'] = self._calculate_performance_metrics(features)
            
            return FeatureResult(
                name=self.config.name,
                data=primary_feature,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata=metadata
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            tprint_error(f"❌ VectorBT feature generation failed: {e}")
            
            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )
    
    def _select_primary_feature(self, features: Dict[str, pd.Series]) -> pd.Series:
        """Select primary feature from generated features."""
        # Priority order for primary feature selection
        priority_features = [
            'ratio', 'grade', 'score', 'signal', 'trend', 'momentum',
            'volatility', 'efficiency', 'coherence', 'strength'
        ]
        
        for priority in priority_features:
            for feature_name, feature_data in features.items():
                if priority in feature_name.lower() and isinstance(feature_data, pd.Series):
                    return feature_data
        
        # Fallback to first numeric series
        for feature_data in features.values():
            if isinstance(feature_data, pd.Series) and pd.api.types.is_numeric_dtype(feature_data):
                return feature_data
        
        # Last resort - return first series
        return next(iter(features.values()))
    
    def _calculate_feature_stats(self, feature: pd.Series) -> Dict[str, float]:
        """Calculate statistics for a feature."""
        try:
            clean_feature = feature.dropna()
            if len(clean_feature) == 0:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            
            return {
                'mean': float(clean_feature.mean()),
                'std': float(clean_feature.std()),
                'min': float(clean_feature.min()),
                'max': float(clean_feature.max()),
                'count': len(clean_feature),
                'missing': len(feature) - len(clean_feature)
            }
        except Exception:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'missing': 0}
    
    def _calculate_performance_metrics(self, features: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate performance metrics for features."""
        try:
            metrics = {}
            
            # Calculate correlation matrix
            numeric_features = {
                name: data for name, data in features.items() 
                if isinstance(data, pd.Series) and pd.api.types.is_numeric_dtype(data)
            }
            
            if len(numeric_features) > 1:
                feature_df = pd.DataFrame(numeric_features)
                correlation_matrix = feature_df.corr()
                metrics['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Calculate feature stability
            stability_scores = {}
            for name, data in numeric_features.items():
                if len(data.dropna()) > 1:
                    stability_scores[name] = 1.0 / (1.0 + data.std())
            
            metrics['stability_scores'] = stability_scores
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {}
    
    def get_all_features(self, data: pd.DataFrame, lookback: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Generate all VectorBT features.
        
        Args:
            data: OHLCV data
            lookback: Override default lookback period
            
        Returns:
            Dictionary of all generated features
        """
        validated_data = self._validate_vectorbt_data(data)
        effective_lookback = lookback or self.config.default_lookback
        
        params = self.config.parameters.copy()
        if lookback:
            params['lookback'] = effective_lookback
        
        return self.generate_vectorbt_features(validated_data, params)
    
    def cleanup(self) -> None:
        """Clean up resources and caches."""
        if self._cache:
            self._cache.clear()
        if self._optimization_cache:
            self._optimization_cache.clear()
        tprint_info(f"🧹 Cleaned up VectorBT feature generator: {self.config.name}")


class VectorBTTechnicalIndicators:
    """
    Comprehensive collection of VectorBT technical indicators.
    
    Provides easy access to a wide range of technical indicators
    with consistent parameter handling and optimization.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """Initialize VectorBT technical indicators."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available")
        
        self.config = config or VectorBTConfig()
        self.vbt = vbt
        
        tprint_info("📊 VectorBT Technical Indicators initialized")
    
    def get_trend_indicators(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, pd.Series]:
        """Get comprehensive trend indicators."""
        indicators = {}
        
        try:
            # Moving Averages
            for window in windows:
                sma = self.vbt.MA.run(data['close'], window=window, short_name='SMA').ma
                ema = self.vbt.MA.run(data['close'], window=window, short_name='EMA').ma
                wma = self.vbt.MA.run(data['close'], window=window, short_name='WMA').ma
                
                indicators[f'sma_{window}'] = sma
                indicators[f'ema_{window}'] = ema
                indicators[f'wma_{window}'] = wma
                
                # Moving average slopes
                indicators[f'sma_slope_{window}'] = sma.diff()
                indicators[f'ema_slope_{window}'] = ema.diff()
                indicators[f'wma_slope_{window}'] = wma.diff()
            
            # ADX for trend strength
            adx = self.vbt.ADX.run(data['high'], data['low'], data['close']).adx
            indicators['adx'] = adx
            indicators['adx_plus'] = self.vbt.ADX.run(data['high'], data['low'], data['close']).plus_di
            indicators['adx_minus'] = self.vbt.ADX.run(data['high'], data['low'], data['close']).minus_di
            
            # Parabolic SAR
            psar = self.vbt.PARABOLIC.run(data['high'], data['low'], data['close']).sar
            indicators['psar'] = psar
            indicators['psar_signal'] = (data['close'] > psar).astype(int)
            
            # Ichimoku Cloud
            ichimoku = self.vbt.ICHIMOKU.run(data['high'], data['low'], data['close'])
            indicators['ichimoku_conversion'] = ichimoku.conversion
            indicators['ichimoku_base'] = ichimoku.base
            indicators['ichimoku_span_a'] = ichimoku.span_a
            indicators['ichimoku_span_b'] = ichimoku.span_b
            indicators['ichimoku_signal'] = ichimoku.signal
            
        except Exception as e:
            tprint_error(f"❌ Error calculating trend indicators: {e}")
        
        return indicators
    
    def get_momentum_indicators(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [14, 21, 30]
    ) -> Dict[str, pd.Series]:
        """Get comprehensive momentum indicators."""
        indicators = {}
        
        try:
            # RSI
            for window in windows:
                rsi = self.vbt.RSI.run(data['close'], window=window).rsi
                indicators[f'rsi_{window}'] = rsi
                indicators[f'rsi_overbought_{window}'] = (rsi > 70).astype(int)
                indicators[f'rsi_oversold_{window}'] = (rsi < 30).astype(int)
            
            # MACD
            macd = self.vbt.MACD.run(data['close'])
            indicators['macd'] = macd.macd
            indicators['macd_signal'] = macd.signal
            indicators['macd_histogram'] = macd.histogram
            indicators['macd_crossover'] = (macd.macd > macd.signal).astype(int)
            
            # Stochastic Oscillator
            stoch = self.vbt.STOCH.run(data['high'], data['low'], data['close'])
            indicators['stoch_k'] = stoch.k
            indicators['stoch_d'] = stoch.d
            indicators['stoch_overbought'] = (stoch.k > 80).astype(int)
            indicators['stoch_oversold'] = (stoch.k < 20).astype(int)
            
            # Williams %R
            willr = self.vbt.WILLR.run(data['high'], data['low'], data['close'])
            indicators['willr'] = willr.willr
            indicators['willr_overbought'] = (willr.willr > -20).astype(int)
            indicators['willr_oversold'] = (willr.willr < -80).astype(int)
            
            # CCI
            cci = self.vbt.CCI.run(data['high'], data['low'], data['close'])
            indicators['cci'] = cci.cci
            indicators['cci_overbought'] = (cci.cci > 100).astype(int)
            indicators['cci_oversold'] = (cci.cci < -100).astype(int)
            
        except Exception as e:
            tprint_error(f"❌ Error calculating momentum indicators: {e}")
        
        return indicators
    
    def get_volatility_indicators(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [14, 20, 30]
    ) -> Dict[str, pd.Series]:
        """Get comprehensive volatility indicators."""
        indicators = {}
        
        try:
            # ATR
            for window in windows:
                atr = self.vbt.ATR.run(data['high'], data['low'], data['close'], window=window).atr
                indicators[f'atr_{window}'] = atr
                indicators[f'atr_ratio_{window}'] = atr / data['close']
            
            # Bollinger Bands
            for window in windows:
                bb = self.vbt.BBANDS.run(data['close'], window=window)
                indicators[f'bb_upper_{window}'] = bb.upper
                indicators[f'bb_middle_{window}'] = bb.middle
                indicators[f'bb_lower_{window}'] = bb.lower
                indicators[f'bb_width_{window}'] = (bb.upper - bb.lower) / bb.middle
                indicators[f'bb_position_{window}'] = (data['close'] - bb.lower) / (bb.upper - bb.lower)
            
            # Keltner Channels
            for window in windows:
                kc = self.vbt.KELTNER.run(data['high'], data['low'], data['close'], window=window)
                indicators[f'kc_upper_{window}'] = kc.upper
                indicators[f'kc_middle_{window}'] = kc.middle
                indicators[f'kc_lower_{window}'] = kc.lower
                indicators[f'kc_width_{window}'] = (kc.upper - kc.lower) / kc.middle
            
            # Donchian Channels
            for window in windows:
                dc = self.vbt.DONCHIAN.run(data['high'], data['low'], window=window)
                indicators[f'dc_upper_{window}'] = dc.upper
                indicators[f'dc_lower_{window}'] = dc.lower
                indicators[f'dc_width_{window}'] = dc.upper - dc.lower
            
        except Exception as e:
            tprint_error(f"❌ Error calculating volatility indicators: {e}")
        
        return indicators
    
    def get_volume_indicators(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [14, 20, 30]
    ) -> Dict[str, pd.Series]:
        """Get comprehensive volume indicators."""
        indicators = {}
        
        try:
            # Volume moving averages
            for window in windows:
                vma = data['volume'].rolling(window=window).mean()
                indicators[f'volume_ma_{window}'] = vma
                indicators[f'volume_ratio_{window}'] = data['volume'] / vma
            
            # VWAP
            vwap = self.vbt.VWAP.run(data['high'], data['low'], data['close'], data['volume']).vwap
            indicators['vwap'] = vwap
            indicators['vwap_deviation'] = (data['close'] - vwap) / vwap
            
            # Volume Rate of Change
            for window in windows:
                vroc = data['volume'].pct_change(window)
                indicators[f'vroc_{window}'] = vroc
            
            # On Balance Volume
            obv = self.vbt.OBV.run(data['close'], data['volume']).obv
            indicators['obv'] = obv
            indicators['obv_ma'] = obv.rolling(window=20).mean()
            
            # Accumulation/Distribution Line
            adl = self.vbt.ADL.run(data['high'], data['low'], data['close'], data['volume']).adl
            indicators['adl'] = adl
            indicators['adl_ma'] = adl.rolling(window=20).mean()
            
        except Exception as e:
            tprint_error(f"❌ Error calculating volume indicators: {e}")
        
        return indicators
    
    def get_all_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all available technical indicators."""
        all_indicators = {}
        
        # Combine all indicator categories
        all_indicators.update(self.get_trend_indicators(data))
        all_indicators.update(self.get_momentum_indicators(data))
        all_indicators.update(self.get_volatility_indicators(data))
        all_indicators.update(self.get_volume_indicators(data))
        
        return all_indicators


# Convenience functions
def create_vectorbt_config(**kwargs) -> VectorBTConfig:
    """Create VectorBT configuration with given parameters."""
    return VectorBTConfig(**kwargs)


def get_vectorbt_indicators(config: Optional[VectorBTConfig] = None) -> VectorBTTechnicalIndicators:
    """Get VectorBT technical indicators instance."""
    return VectorBTTechnicalIndicators(config)