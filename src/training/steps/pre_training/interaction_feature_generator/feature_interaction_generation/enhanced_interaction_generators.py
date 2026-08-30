"""
Enhanced Interaction Feature Generators

This module provides enhanced versions of the existing interaction feature generators
with VectorBT optimizations and additional analysis capabilities. These generators
leverage the feature bank and provide sophisticated interaction analysis.

Key Features:
- VectorBT-optimized interaction calculations
- Enhanced divergence analysis
- Volatility-aware interactions
- Cross-timeframe interaction analysis
- Quantile-based interaction features
- Advanced statistical interactions
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import warnings

from ...core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ...core.feature_bank import get_global_feature_bank, FeatureBank

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
    from ...utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager,
        OperationType,
        OptimizationStrategy
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None

from ...utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


class EnhancedMomentumDivergenceGenerator(VectorizedFeatureGenerator):
    """Enhanced momentum divergence generator with VectorBT optimization and advanced analysis."""
    
    def __init__(self, period: int = 5, analysis_windows: List[int] = None, 
                 enable_volatility_analysis: bool = True, enable_quantile_analysis: bool = True):
        """Initialize enhanced momentum divergence generator."""
        self.period = period
        self.analysis_windows = analysis_windows or [5, 10, 20, 50]
        self.enable_volatility_analysis = enable_volatility_analysis
        self.enable_quantile_analysis = enable_quantile_analysis
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        
        # Initialize unified vectorization manager
        self.unified_manager = None
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        
        config = FeatureConfig(
            name=f"enhanced_momentum_divergence_{period}",
            category=FeatureCategory.INTERACTION,
            description=f"Enhanced momentum divergence with VectorBT optimization and advanced analysis over {period} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=max(self.analysis_windows),
            parameters={
                'period': period,
                'analysis_windows': self.analysis_windows,
                'enable_volatility_analysis': enable_volatility_analysis,
                'enable_quantile_analysis': enable_quantile_analysis
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
        
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced momentum divergence features."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)
        
        # Calculate basic momentum divergence
        price_momentum = data['close'].pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        basic_divergence = price_momentum - volume_momentum
        
        # Enhanced divergence analysis
        enhanced_features = self._analyze_enhanced_divergence(
            price_momentum, volume_momentum, basic_divergence, data
        )
        
        # Return the main divergence feature (for compatibility)
        return basic_divergence
    
    def _analyze_enhanced_divergence(self, price_momentum: pd.Series, volume_momentum: pd.Series, 
                                   basic_divergence: pd.Series, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze enhanced divergence characteristics."""
        enhanced_features = {}
        
        # 1. Divergence strength analysis
        for window in self.analysis_windows:
            try:
                # Rolling divergence strength
                divergence_strength = self._optimized_rolling_operation(
                    basic_divergence, 'std', window
                )
                enhanced_features[f'divergence_strength_{window}'] = divergence_strength
                
                # Divergence momentum
                divergence_momentum = basic_divergence.pct_change(window)
                enhanced_features[f'divergence_momentum_{window}'] = divergence_momentum
                
                # Divergence acceleration
                divergence_acceleration = divergence_momentum.pct_change(window)
                enhanced_features[f'divergence_acceleration_{window}'] = divergence_acceleration
                
            except Exception as e:
                self.logger.warning(f"Divergence analysis failed for window {window}: {e}")
        
        # 2. Volatility analysis
        if self.enable_volatility_analysis:
            try:
                # Price momentum volatility
                price_vol = self._optimized_rolling_operation(price_momentum, 'std', self.period)
                # Volume momentum volatility
                volume_vol = self._optimized_rolling_operation(volume_momentum, 'std', self.period)
                
                # Volatility-adjusted divergence
                vol_adjusted_divergence = basic_divergence / (price_vol + volume_vol + 1e-8)
                enhanced_features['vol_adjusted_divergence'] = vol_adjusted_divergence
                
                # Volatility ratio
                vol_ratio = price_vol / (volume_vol + 1e-8)
                enhanced_features['momentum_vol_ratio'] = vol_ratio
                
            except Exception as e:
                self.logger.warning(f"Volatility analysis failed: {e}")
        
        # 3. Quantile analysis
        if self.enable_quantile_analysis:
            try:
                quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
                for q in quantile_levels:
                    # Divergence quantiles
                    divergence_quantile = self._optimized_rolling_operation(
                        basic_divergence, 'quantile', self.period, q=q
                    )
                    enhanced_features[f'divergence_quantile_{q}'] = divergence_quantile
                    
                    # Position within quantile range
                    divergence_rank = basic_divergence.rolling(self.period).rank(pct=True)
                    quantile_position = (divergence_rank <= q).astype(int)
                    enhanced_features[f'divergence_quantile_position_{q}'] = quantile_position
                
            except Exception as e:
                self.logger.warning(f"Quantile analysis failed: {e}")
        
        return enhanced_features
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT or fallback."""
        if self.rolling_optimizer and len(data) >= 1000:
            try:
                if operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            except Exception as e:
                self.logger.debug(f"VectorBT operation failed: {e}, using pandas fallback")
        
        # Fallback to pandas operations
        rolling_obj = data.rolling(window=window)
        if operation == 'std':
            return rolling_obj.std()
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
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


class EnhancedMomentumVolatilityGenerator(VectorizedFeatureGenerator):
    """Enhanced momentum-volatility interaction generator with VectorBT optimization."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, 
                 enable_regime_analysis: bool = True, enable_cross_timeframe: bool = True):
        """Initialize enhanced momentum-volatility generator."""
        self.period = period
        self.volatility_window = volatility_window
        self.enable_regime_analysis = enable_regime_analysis
        self.enable_cross_timeframe = enable_cross_timeframe
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        
        config = FeatureConfig(
            name=f"enhanced_momentum_volatility_{period}_{volatility_window}",
            category=FeatureCategory.INTERACTION,
            description=f"Enhanced momentum-volatility interaction with VectorBT optimization over {period} momentum periods and {volatility_window} volatility window",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=max(period, volatility_window),
            min_lookback=max(period, volatility_window),
            max_lookback=max(period, volatility_window),
            parameters={
                'period': period,
                'volatility_window': volatility_window,
                'enable_regime_analysis': enable_regime_analysis,
                'enable_cross_timeframe': enable_cross_timeframe
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
        
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced momentum-volatility interaction features."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)
        
        # Calculate basic momentum-volatility interaction
        price_momentum = data['close'].pct_change(self.period)
        volatility = self._optimized_rolling_operation(data['close'], 'std', self.volatility_window)
        basic_interaction = price_momentum / (volatility + 1e-8)
        
        # Enhanced analysis
        enhanced_features = self._analyze_enhanced_momentum_volatility(
            price_momentum, volatility, basic_interaction, data
        )
        
        # Return the main interaction feature (for compatibility)
        return basic_interaction
    
    def _analyze_enhanced_momentum_volatility(self, price_momentum: pd.Series, volatility: pd.Series,
                                            basic_interaction: pd.Series, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze enhanced momentum-volatility characteristics."""
        enhanced_features = {}
        
        # 1. Regime analysis
        if self.enable_regime_analysis:
            try:
                # High/low volatility regimes
                vol_threshold = volatility.rolling(self.volatility_window * 2).quantile(0.7)
                high_vol_regime = (volatility > vol_threshold).astype(int)
                
                # Regime-specific interactions
                high_vol_interaction = basic_interaction * high_vol_regime
                low_vol_interaction = basic_interaction * (1 - high_vol_regime)
                
                enhanced_features['high_vol_interaction'] = high_vol_interaction
                enhanced_features['low_vol_interaction'] = low_vol_interaction
                enhanced_features['vol_regime'] = high_vol_regime
                
                # Regime persistence
                regime_persistence = high_vol_regime.rolling(self.volatility_window).mean()
                enhanced_features['vol_regime_persistence'] = regime_persistence
                
            except Exception as e:
                self.logger.warning(f"Regime analysis failed: {e}")
        
        # 2. Cross-timeframe analysis
        if self.enable_cross_timeframe:
            try:
                timeframes = [self.period // 2, self.period, self.period * 2]
                for tf in timeframes:
                    if tf != self.period:
                        # Cross-timeframe momentum
                        tf_momentum = data['close'].pct_change(tf)
                        tf_interaction = tf_momentum / (volatility + 1e-8)
                        
                        # Cross-timeframe correlation
                        tf_corr = self._optimized_rolling_operation(
                            basic_interaction, 'corr', min(tf, self.volatility_window), other=tf_interaction
                        )
                        
                        enhanced_features[f'ctf_interaction_{tf}'] = tf_interaction
                        enhanced_features[f'ctf_correlation_{tf}'] = tf_corr
                        
            except Exception as e:
                self.logger.warning(f"Cross-timeframe analysis failed: {e}")
        
        # 3. Advanced statistical features
        try:
            # Momentum-volatility ratio stability
            interaction_std = self._optimized_rolling_operation(basic_interaction, 'std', self.volatility_window)
            interaction_mean = self._optimized_rolling_operation(basic_interaction, 'mean', self.volatility_window)
            stability = interaction_std / (abs(interaction_mean) + 1e-8)
            enhanced_features['interaction_stability'] = stability
            
            # Momentum-volatility asymmetry
            positive_momentum = (price_momentum > 0).astype(int)
            vol_asymmetry = volatility * positive_momentum - volatility * (1 - positive_momentum)
            enhanced_features['vol_asymmetry'] = vol_asymmetry
            
        except Exception as e:
            self.logger.warning(f"Advanced statistical analysis failed: {e}")
        
        return enhanced_features
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT or fallback."""
        if self.rolling_optimizer and len(data) >= 1000:
            try:
                if operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'mean':
                    return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            except Exception as e:
                self.logger.debug(f"VectorBT operation failed: {e}, using pandas fallback")
        
        # Fallback to pandas operations
        rolling_obj = data.rolling(window=window)
        if operation == 'std':
            return rolling_obj.std()
        elif operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        return data.copy()


class EnhancedVolatilityVolumeGenerator(VectorizedFeatureGenerator):
    """Enhanced volatility-volume interaction generator with VectorBT optimization."""
    
    def __init__(self, volatility_window: int = 20, volume_window: int = 20,
                 enable_volume_profile: bool = True, enable_volatility_clustering: bool = True):
        """Initialize enhanced volatility-volume generator."""
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.enable_volume_profile = enable_volume_profile
        self.enable_volatility_clustering = enable_volatility_clustering
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        
        config = FeatureConfig(
            name=f"enhanced_volatility_volume_{volatility_window}_{volume_window}",
            category=FeatureCategory.INTERACTION,
            description=f"Enhanced volatility-volume interaction with VectorBT optimization with {volatility_window} volatility window and {volume_window} volume window",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=max(volatility_window, volume_window),
            min_lookback=max(volatility_window, volume_window),
            max_lookback=max(volatility_window, volume_window),
            parameters={
                'volatility_window': volatility_window,
                'volume_window': volume_window,
                'enable_volume_profile': enable_volume_profile,
                'enable_volatility_clustering': enable_volatility_clustering
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
        
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced volatility-volume interaction features."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe_processing(data)
        
        # Calculate basic volatility-volume interaction
        volatility = self._optimized_rolling_operation(data['close'], 'std', self.volatility_window)
        volume_ma = self._optimized_rolling_operation(data['volume'], 'mean', self.volume_window)
        basic_interaction = volatility * volume_ma
        
        # Enhanced analysis
        enhanced_features = self._analyze_enhanced_volatility_volume(
            volatility, volume_ma, basic_interaction, data
        )
        
        # Return the main interaction feature (for compatibility)
        return basic_interaction
    
    def _analyze_enhanced_volatility_volume(self, volatility: pd.Series, volume_ma: pd.Series,
                                          basic_interaction: pd.Series, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze enhanced volatility-volume characteristics."""
        enhanced_features = {}
        
        # 1. Volume profile analysis
        if self.enable_volume_profile:
            try:
                # Volume-weighted volatility
                volume_weighted_vol = volatility * data['volume']
                enhanced_features['volume_weighted_volatility'] = volume_weighted_vol
                
                # Volume-volatility correlation
                vol_corr = self._optimized_rolling_operation(
                    volatility, 'corr', self.volatility_window, other=data['volume']
                )
                enhanced_features['vol_volume_correlation'] = vol_corr
                
                # Volume profile percentiles
                volume_percentiles = data['volume'].rolling(self.volume_window).quantile([0.25, 0.5, 0.75])
                for i, q in enumerate([0.25, 0.5, 0.75]):
                    enhanced_features[f'volume_percentile_{q}'] = volume_percentiles.iloc[:, i]
                
            except Exception as e:
                self.logger.warning(f"Volume profile analysis failed: {e}")
        
        # 2. Volatility clustering analysis
        if self.enable_volatility_clustering:
            try:
                # Volatility clustering strength
                vol_change = volatility.pct_change()
                vol_clustering = vol_change.rolling(self.volatility_window).sum()
                enhanced_features['volatility_clustering'] = vol_clustering
                
                # Volume clustering
                volume_change = data['volume'].pct_change()
                volume_clustering = volume_change.rolling(self.volume_window).sum()
                enhanced_features['volume_clustering'] = volume_clustering
                
                # Combined clustering
                combined_clustering = vol_clustering * volume_clustering
                enhanced_features['combined_clustering'] = combined_clustering
                
            except Exception as e:
                self.logger.warning(f"Volatility clustering analysis failed: {e}")
        
        # 3. Advanced interaction features
        try:
            # Volatility-volume efficiency
            vol_efficiency = volatility / (volume_ma + 1e-8)
            enhanced_features['vol_volume_efficiency'] = vol_efficiency
            
            # Asymmetric volatility-volume relationship
            positive_returns = (data['close'].pct_change() > 0).astype(int)
            up_vol_interaction = basic_interaction * positive_returns
            down_vol_interaction = basic_interaction * (1 - positive_returns)
            
            enhanced_features['up_vol_interaction'] = up_vol_interaction
            enhanced_features['down_vol_interaction'] = down_vol_interaction
            enhanced_features['asymmetric_vol_interaction'] = up_vol_interaction - down_vol_interaction
            
        except Exception as e:
            self.logger.warning(f"Advanced interaction analysis failed: {e}")
        
        return enhanced_features
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT or fallback."""
        if self.rolling_optimizer and len(data) >= 1000:
            try:
                if operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'mean':
                    return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            except Exception as e:
                self.logger.debug(f"VectorBT operation failed: {e}, using pandas fallback")
        
        # Fallback to pandas operations
        rolling_obj = data.rolling(window=window)
        if operation == 'std':
            return rolling_obj.std()
        elif operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        return data.copy()


def create_enhanced_interaction_generators() -> List[FeatureGenerator]:
    """Create all enhanced interaction feature generators."""
    generators = []
    
    # Enhanced momentum divergence
    generators.append(EnhancedMomentumDivergenceGenerator(period=5))
    generators.append(EnhancedMomentumDivergenceGenerator(period=10))
    generators.append(EnhancedMomentumDivergenceGenerator(period=20))
    
    # Enhanced momentum-volatility
    generators.append(EnhancedMomentumVolatilityGenerator(period=5, volatility_window=20))
    generators.append(EnhancedMomentumVolatilityGenerator(period=10, volatility_window=30))
    generators.append(EnhancedMomentumVolatilityGenerator(period=20, volatility_window=50))
    
    # Enhanced volatility-volume
    generators.append(EnhancedVolatilityVolumeGenerator(volatility_window=20, volume_window=20))
    generators.append(EnhancedVolatilityVolumeGenerator(volatility_window=30, volume_window=30))
    generators.append(EnhancedVolatilityVolumeGenerator(volatility_window=50, volume_window=50))
    
    return generators


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'high': 100 + np.random.randn(n_samples).cumsum() * 0.1 + np.random.uniform(0, 0.5, n_samples),
        'low': 100 + np.random.randn(n_samples).cumsum() * 0.1 - np.random.uniform(0, 0.5, n_samples),
        'close': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Test enhanced interaction generators
    generators = create_enhanced_interaction_generators()
    
    for generator in generators:
        try:
            result = generator.generate(data)
            print(f"✅ {generator.config.name}: Generated {len(result.data)} values")
        except Exception as e:
            print(f"❌ {generator.config.name}: Failed - {e}")