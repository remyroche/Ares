"""
Regime Statistical Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on statistical
regime characteristics rather than short-term trading signals.

Key Features:
- Distribution shape changes (skewness, kurtosis)
- Regime persistence measures
- Cross-correlation stability
- Regime transition probabilities
- Statistical regime stability
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, jarque_bera

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ..utils.optimization.unified_optimizer import get_feature_optimizer, FeatureOptimizationConfig
    UNIFIED_OPTIMIZER_AVAILABLE = True
except ImportError:
    UNIFIED_OPTIMIZER_AVAILABLE = False
    get_feature_optimizer = None
    FeatureOptimizationConfig = None

# VectorBT Optimization Mixin
try:
    from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin
    VECTORBT_MIXIN_AVAILABLE = True
except ImportError:
    VECTORBT_MIXIN_AVAILABLE = False
    VectorBTOptimizationMixin = None

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

# Import tprint for consistent logging
from src.utils.tprint import tprint

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class RegimeStatisticalFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for statistical regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        
        # Initialize VectorBT optimization mixin if available
        if VECTORBT_MIXIN_AVAILABLE:
            VectorBTOptimizationMixin.__init__(self)
        
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT Rolling Optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=getattr(self, 'enable_gpu', False),
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_OPTIMIZER_AVAILABLE:
            self.unified_optimizer = get_feature_optimizer()
        else:
            self.unified_optimizer = None
        
        # Enhanced performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'chunked_operations': 0,
            'gpu_operations': 0,
            'cache_hits': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_statistical_features",
            category=FeatureCategory.REGIME,
            description="Statistical regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,  # 8 hours in 15m periods
            min_lookback=8,       # 2 hours minimum
            max_lookback=128,     # 32 hours maximum
            parameters={
                "distribution_windows": [16, 48, 128],  # 4h, 12h, 32h in 15m periods (original min, middle, new max)
                "correlation_windows": [20, 60, 160],  # 5h, 15h, 40h (original min, middle, new max)
                "persistence_windows": [12, 30, 96],  # 3h, 7.5h, 24h (original min, middle, new max)
                "transition_windows": [8, 20, 64],  # 2h, 5h, 16h (original min, middle, new max)
                "batch_processing": True,
                "chunked_processing": True,
                "gpu_acceleration": False
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single statistical regime feature as required by the base class."""
        try:
            # Generate all statistical regime features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple statistical feature if no features generated
                if 'close' in data.columns and len(data) > 1:
                    stat_feature = data['close'].rolling(window=5).std().fillna(0).values
                    return pd.Series(stat_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            tprint(f"_generate_feature: Statistical regime feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate statistical regime features using optimized VectorBT operations."""
        features = {}
        
        try:
            # Validate price data
            if 'close' not in data.columns:
                tprint("Warning: 'close' column not found in data")
                return features
            
            close_prices = data['close'].values
            if len(close_prices) < 8:
                tprint(f"Warning: Insufficient data points: {len(close_prices)} < 8")
                return features
            
            # Calculate returns for statistical analysis
            returns = np.diff(np.log(close_prices))
            returns_series = pd.Series(returns, index=data.index[1:])
            
            # Use batch processing for multiple features if enabled
            if (self.config.parameters.get("batch_processing", True) and 
                self.rolling_optimizer and ROLLING_OPTIMIZER_AVAILABLE):
                features.update(self._generate_features_batch(returns_series, data))
            else:
                # Fallback to individual feature generation
                features.update(self._generate_features_individual(returns_series, data))
            
        except Exception as e:
            tprint(f"Error in statistical feature generation: {e}")
        
        return features
    
    def generate_features_optimized(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate statistical regime features using fully optimized VectorBT operations."""
        return self.generate_features(data, **kwargs)
    
    def _generate_features_batch(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate features using VectorBT batch processing for maximum efficiency."""
        features = {}
        
        if not self.rolling_optimizer:
            return self._generate_features_individual(returns, data)
        
        try:
            # Prepare batch operations for all windows
            batch_operations = []
            windows = self.config.parameters["distribution_windows"]
            
            # Add rolling operations for all windows
            for window in windows:
                if len(returns) >= window:
                    batch_operations.extend([
                        {
                            'type': 'rolling',
                            'name': f'returns_mean_{window}',
                            'params': {'column': 'returns', 'operation': 'mean', 'window': window}
                        },
                        {
                            'type': 'rolling',
                            'name': f'returns_std_{window}',
                            'params': {'column': 'returns', 'operation': 'std', 'window': window}
                        },
                        {
                            'type': 'rolling',
                            'name': f'returns_skew_{window}',
                            'params': {'column': 'returns', 'operation': 'skew', 'window': window}
                        },
                        {
                            'type': 'rolling',
                            'name': f'returns_kurt_{window}',
                            'params': {'column': 'returns', 'operation': 'kurt', 'window': window}
                        }
                    ])
            
            # Execute batch operations
            if batch_operations:
                # Create a DataFrame with returns for batch processing
                returns_df = pd.DataFrame({'returns': returns})
                batch_results = self.rolling_optimizer._vectorbt_batch_operations(returns_df, batch_operations)
                
                # Process results
                for col in batch_results.columns:
                    if col.startswith('returns_'):
                        features[col] = batch_results[col].fillna(0).values
                
                self.performance_stats['batch_operations'] += 1
            
            # Generate additional statistical features
            features.update(self._generate_distribution_features_optimized(returns, data))
            features.update(self._generate_persistence_features_optimized(returns, data))
            features.update(self._generate_correlation_features_optimized(returns, data))
            features.update(self._generate_transition_features_optimized(returns, data))
            features.update(self._generate_stability_features_optimized(returns, data))
            
        except Exception as e:
            tprint(f"Error in batch feature generation: {e}")
            # Fallback to individual processing
            features.update(self._generate_features_individual(returns, data))
        
        return features
    
    def _generate_features_individual(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Fallback individual feature generation."""
        features = {}
        
        try:
            # 1. Distribution Shape Features
            try:
                features.update(self._generate_distribution_features(returns.values, data))
            except Exception as e:
                tprint(f"Error in distribution features: {e}")
            
            # 2. Statistical Regime Persistence
            try:
                features.update(self._generate_statistical_persistence_features(returns.values, data))
            except Exception as e:
                tprint(f"Error in persistence features: {e}")
            
            # 3. Cross-Correlation Features
            try:
                features.update(self._generate_correlation_features(returns.values, data))
            except Exception as e:
                tprint(f"Error in correlation features: {e}")
            
            # 4. Statistical Regime Transitions
            try:
                features.update(self._generate_statistical_transition_features(returns.values, data))
            except Exception as e:
                tprint(f"Error in transition features: {e}")
            
            # 5. Statistical Regime Stability
            try:
                features.update(self._generate_statistical_stability_features(returns.values, data))
            except Exception as e:
                tprint(f"Error in stability features: {e}")
        except Exception as e:
            tprint(f"Error in individual feature generation: {e}")
        
        return features
    
    def _generate_distribution_features_optimized(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate distribution shape features using optimized VectorBT operations."""
        features = {}
        windows = self.config.parameters["distribution_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            try:
                # Use VectorBT native rolling functions when available
                if self.rolling_optimizer and VECTORBT_AVAILABLE:
                    # Skewness using VectorBT native function
                    skewness = self.rolling_optimizer.rolling_skew(returns, window=window)
                    
                    # Kurtosis using VectorBT native function
                    kurtosis = self.rolling_optimizer.rolling_kurt(returns, window=window)
                    
                    # Normality test using optimized approach
                    normality = self._calculate_normality_optimized(returns, window)
                    
                else:
                    # Fallback to custom implementation
                    skewness = self._calculate_rolling_skewness_optimized(returns, window)
                    kurtosis = self._calculate_rolling_kurtosis_optimized(returns, window)
                    normality = self._calculate_distribution_normality_optimized(returns, window)
                
                # Pad to match data length
                data_len = len(data)
                skewness_padded = self._pad_series(skewness, data_len, window)
                kurtosis_padded = self._pad_series(kurtosis, data_len, window)
                normality_padded = self._pad_series(normality, data_len, window)
                
                features[f'returns_skewness_{window}'] = skewness_padded
                features[f'returns_kurtosis_{window}'] = kurtosis_padded
                features[f'distribution_normality_{window}'] = normality_padded
                
                # Persistence features
                skew_persistence = self._calculate_skewness_persistence_optimized(returns, window)
                kurt_persistence = self._calculate_kurtosis_persistence_optimized(returns, window)
                
                features[f'skewness_persistence_{window}'] = self._pad_series(skew_persistence, data_len, window)
                features[f'kurtosis_persistence_{window}'] = self._pad_series(kurt_persistence, data_len, window)
                
            except Exception as e:
                tprint(f"Error in distribution features for window {window}: {e}")
                continue
        
        return features
    
    def _generate_persistence_features_optimized(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical persistence features using optimized operations."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            try:
                # Statistical persistence using optimized approach
                stat_persistence = self._calculate_statistical_persistence_optimized(returns, window)
                dist_stability = self._calculate_distribution_stability_optimized(returns, window)
                stat_strength = self._calculate_statistical_strength_optimized(returns, window)
                
                data_len = len(data)
                features[f'statistical_persistence_{window}'] = self._pad_series(stat_persistence, data_len, window)
                features[f'distribution_stability_{window}'] = self._pad_series(dist_stability, data_len, window)
                features[f'statistical_strength_{window}'] = self._pad_series(stat_strength, data_len, window)
                
            except Exception as e:
                tprint(f"Error in persistence features for window {window}: {e}")
                continue
        
        return features
    
    def _generate_correlation_features_optimized(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-correlation features using optimized operations."""
        features = {}
        windows = self.config.parameters["correlation_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            try:
                # Autocorrelation using optimized approach
                autocorr = self._calculate_returns_autocorrelation_optimized(returns, window)
                corr_stability = self._calculate_correlation_stability_optimized(returns, window)
                cross_corr = self._calculate_cross_correlation_features_optimized(returns, window)
                
                data_len = len(data)
                features[f'returns_autocorr_{window}'] = self._pad_series(autocorr, data_len, window)
                features[f'correlation_stability_{window}'] = self._pad_series(corr_stability, data_len, window)
                features[f'cross_correlation_{window}'] = self._pad_series(cross_corr, data_len, window)
                
            except Exception as e:
                tprint(f"Error in correlation features for window {window}: {e}")
                continue
        
        return features
    
    def _generate_transition_features_optimized(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical transition features using optimized operations."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            try:
                # Statistical regime change detection
                stat_change = self._detect_statistical_regime_changes_optimized(returns, window)
                dist_transition = self._calculate_distribution_transition_probability_optimized(returns, window)
                stat_momentum = self._calculate_statistical_momentum_optimized(returns, window)
                
                data_len = len(data)
                features[f'statistical_regime_change_{window}'] = self._pad_series(stat_change, data_len, window * 2)
                features[f'distribution_transition_{window}'] = self._pad_series(dist_transition, data_len, window * 2)
                features[f'statistical_momentum_{window}'] = self._pad_series(stat_momentum, data_len, window * 2)
                
            except Exception as e:
                tprint(f"Error in transition features for window {window}: {e}")
                continue
        
        return features
    
    def _generate_stability_features_optimized(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical stability features using optimized operations."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            try:
                # Statistical stability features
                stat_stability = self._calculate_statistical_stability_optimized(returns, window)
                dist_entropy = self._calculate_distribution_entropy_optimized(returns, window)
                stat_consistency = self._calculate_statistical_consistency_optimized(returns, window)
                
                data_len = len(data)
                features[f'statistical_stability_{window}'] = self._pad_series(stat_stability, data_len, window)
                features[f'distribution_entropy_{window}'] = self._pad_series(dist_entropy, data_len, window)
                features[f'statistical_consistency_{window}'] = self._pad_series(stat_consistency, data_len, window)
                
            except Exception as e:
                tprint(f"Error in stability features for window {window}: {e}")
                continue
        
        return features
    
    def _generate_distribution_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate distribution shape features."""
        features = {}
        windows = self.config.parameters["distribution_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Skewness regime features
            skewness = self._calculate_rolling_skewness(returns, window)
            skewness_persistence = self._calculate_skewness_persistence(returns, window)

            # Kurtosis regime features
            kurtosis = self._calculate_rolling_kurtosis(returns, window)
            kurtosis_persistence = self._calculate_kurtosis_persistence(returns, window)

            # Distribution normality
            normality = self._calculate_distribution_normality(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            skewness_padded = np.full(data_len, np.nan)
            skew_persist_padded = np.full(data_len, np.nan)
            kurtosis_padded = np.full(data_len, np.nan)
            kurt_persist_padded = np.full(data_len, np.nan)
            normality_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(skewness), data_len - window)
            if valid_indices > 0:
                skewness_padded[window:window + valid_indices] = skewness[:valid_indices]
                skew_persist_padded[window:window + valid_indices] = skewness_persistence[:valid_indices]
                kurtosis_padded[window:window + valid_indices] = kurtosis[:valid_indices]
                kurt_persist_padded[window:window + valid_indices] = kurtosis_persistence[:valid_indices]
                normality_padded[window:window + valid_indices] = normality[:valid_indices]

            features[f'returns_skewness_{window}'] = skewness_padded
            features[f'skewness_persistence_{window}'] = skew_persist_padded
            features[f'returns_kurtosis_{window}'] = kurtosis_padded
            features[f'kurtosis_persistence_{window}'] = kurt_persist_padded
            features[f'distribution_normality_{window}'] = normality_padded

        return features
    
    def _generate_statistical_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical persistence features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Statistical regime persistence
            stat_persistence = self._calculate_statistical_persistence(returns, window)

            # Distribution stability
            dist_stability = self._calculate_distribution_stability(returns, window)

            # Statistical regime strength
            stat_strength = self._calculate_statistical_strength(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            persistence_padded = np.full(data_len, np.nan)
            stability_padded = np.full(data_len, np.nan)
            strength_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_persistence), data_len - window)
            if valid_indices > 0:
                persistence_padded[window:window + valid_indices] = stat_persistence[:valid_indices]
                stability_padded[window:window + valid_indices] = dist_stability[:valid_indices]
                strength_padded[window:window + valid_indices] = stat_strength[:valid_indices]

            features[f'statistical_persistence_{window}'] = persistence_padded
            features[f'distribution_stability_{window}'] = stability_padded
            features[f'statistical_strength_{window}'] = strength_padded

        return features
    
    def _generate_correlation_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-correlation features."""
        features = {}
        windows = self.config.parameters["correlation_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Returns autocorrelation
            autocorr = self._calculate_returns_autocorrelation(returns, window)

            # Correlation stability
            corr_stability = self._calculate_correlation_stability(returns, window)

            # Cross-correlation regime features
            cross_corr = self._calculate_cross_correlation_features(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            autocorr_padded = np.full(data_len, np.nan)
            stability_padded = np.full(data_len, np.nan)
            cross_corr_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(autocorr), data_len - window)
            if valid_indices > 0:
                autocorr_padded[window:window + valid_indices] = autocorr[:valid_indices]
                stability_padded[window:window + valid_indices] = corr_stability[:valid_indices]
                cross_corr_padded[window:window + valid_indices] = cross_corr[:valid_indices]

            features[f'returns_autocorr_{window}'] = autocorr_padded
            features[f'correlation_stability_{window}'] = stability_padded
            features[f'cross_correlation_{window}'] = cross_corr_padded

        return features
    
    def _generate_statistical_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(returns) < window * 2:
                continue

            # Statistical regime change detection
            stat_change = self._detect_statistical_regime_changes(returns, window)

            # Distribution transition probability
            dist_transition = self._calculate_distribution_transition_probability(returns, window)

            # Statistical regime momentum
            stat_momentum = self._calculate_statistical_momentum(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            change_padded = np.full(data_len, np.nan)
            transition_padded = np.full(data_len, np.nan)
            momentum_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_change), data_len - window * 2)
            if valid_indices > 0:
                change_padded[window*2:window*2 + valid_indices] = stat_change[:valid_indices]
                transition_padded[window*2:window*2 + valid_indices] = dist_transition[:valid_indices]
                momentum_padded[window*2:window*2 + valid_indices] = stat_momentum[:valid_indices]

            features[f'statistical_regime_change_{window}'] = change_padded
            features[f'distribution_transition_{window}'] = transition_padded
            features[f'statistical_momentum_{window}'] = momentum_padded

        return features
    
    def _generate_statistical_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Statistical regime stability
            stat_stability = self._calculate_statistical_stability(returns, window)

            # Distribution entropy
            dist_entropy = self._calculate_distribution_entropy(returns, window)

            # Statistical regime consistency
            stat_consistency = self._calculate_statistical_consistency(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            stability_padded = np.full(data_len, np.nan)
            entropy_padded = np.full(data_len, np.nan)
            consistency_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_stability), data_len - window)
            if valid_indices > 0:
                stability_padded[window:window + valid_indices] = stat_stability[:valid_indices]
                entropy_padded[window:window + valid_indices] = dist_entropy[:valid_indices]
                consistency_padded[window:window + valid_indices] = stat_consistency[:valid_indices]

            features[f'statistical_stability_{window}'] = stability_padded
            features[f'distribution_entropy_{window}'] = entropy_padded
            features[f'statistical_consistency_{window}'] = consistency_padded

        return features
    
    def _calculate_rolling_skewness(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized skewness calculation
        returns_series = pd.Series(returns)
        
        # Vectorized skewness using rolling statistics
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        
        # Simplified skewness approximation using third moment
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        
        return skewness.fillna(0).values
    
    def _calculate_rolling_kurtosis(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized kurtosis calculation
        returns_series = pd.Series(returns)
        
        # Vectorized kurtosis using rolling statistics
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        
        # Simplified kurtosis approximation using fourth moment
        centered = returns_series - rolling_mean
        kurt = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        return kurt.fillna(0).values
    
    def _calculate_skewness_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate skewness persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized skewness persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling skewness for the entire series at once
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        
        # Calculate autocorrelation of skewness using vectorized operations
        skewness_shifted = skewness.shift(1)
        skewness_autocorr = skewness.rolling(window=window//4).corr(skewness_shifted).fillna(0)
        
        return skewness_autocorr.values
    
    def _calculate_autocorrelation(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate autocorrelation for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized skewness calculation
        returns_series = pd.Series(returns_window)
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        
        # Vectorized skewness using third moment
        centered = returns_series - rolling_mean
        skew_values = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        skew_values = skew_values.dropna()
        
        if len(skew_values) > 1:
            corr = np.corrcoef(skew_values[:-1], skew_values[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_kurtosis_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate kurtosis persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized kurtosis persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling kurtosis for the entire series at once
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        centered = returns_series - rolling_mean
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate autocorrelation of kurtosis using vectorized operations
        kurtosis_shifted = kurtosis.shift(1)
        kurtosis_autocorr = kurtosis.rolling(window=window//4).corr(kurtosis_shifted).fillna(0)
        
        return kurtosis_autocorr.values
    
    def _calculate_kurtosis_autocorrelation(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate kurtosis autocorrelation for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized kurtosis calculation
        returns_series = pd.Series(returns_window)
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        
        # Vectorized kurtosis using fourth moment
        centered = returns_series - rolling_mean
        kurt_values = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        kurt_values = kurt_values.dropna()
        
        if len(kurt_values) > 1:
            corr = np.corrcoef(kurt_values[:-1], kurt_values[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_distribution_normality(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution normality using Jarque-Bera test - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized normality approximation
        returns_series = pd.Series(returns)
        
        # Vectorized normality test using skewness and kurtosis approximation
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        
        # Calculate skewness and kurtosis for normality test
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Simplified normality test using JB statistic approximation
        jb_stat = (skewness ** 2 + (kurtosis ** 2) / 4) * window / 6
        normality = np.exp(-jb_stat / 2)  # Approximate p-value
        
        return normality.fillna(0).values
    
    def _calculate_jarque_bera_pvalue(self, returns_window: pd.Series) -> float:
        """Calculate Jarque-Bera p-value for a returns window."""
        try:
            jb_stat, p_value = jarque_bera(returns_window)
            return p_value
        except:
            return 0.0
    
    def _calculate_statistical_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate squared returns
        squared_returns = returns_series ** 2
        
        # Calculate autocorrelation of squared returns using vectorized operations
        squared_returns_shifted = squared_returns.shift(1)
        autocorr = squared_returns.rolling(window=window).corr(squared_returns_shifted).fillna(0)
        
        return autocorr.values
    
    def _calculate_squared_returns_autocorr(self, returns_window: pd.Series) -> float:
        """Calculate autocorrelation of squared returns for a window."""
        if len(returns_window) < 3:
            return 0.0
        
        # Persistence based on autocorrelation of squared returns
        squared_returns = returns_window ** 2
        if len(squared_returns) > 1:
            corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_distribution_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized distribution stability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        skew_cv = self._vectorbt_rolling_operation(skewness, "std", window) / (self._vectorbt_rolling_operation(skewness, "mean", window).abs() + 1e-8)
        kurt_cv = self._vectorbt_rolling_operation(kurtosis, "std", window) / (self._vectorbt_rolling_operation(kurtosis, "mean", window).abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0).values
    
    def _calculate_moment_stability(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate moment stability for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized moment calculations
        returns_series = pd.Series(returns_window)
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        
        # Vectorized skewness and kurtosis
        centered = returns_series - rolling_mean
        skew_vals = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurt_vals = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        skew_vals = skew_vals.dropna()
        kurt_vals = kurt_vals.dropna()
        
        if len(skew_vals) > 1 and len(kurt_vals) > 1:
            skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
            kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
            return max(0, 1 - (skew_cv + kurt_cv) / 2)
        return 0.0
    
    def _calculate_statistical_strength(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime strength - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical strength calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Strength based on deviation from normal distribution
        deviation = np.abs(skewness) + np.abs(kurtosis - 3)
        strength = np.maximum(0, 1 - deviation / 10)  # Normalize to 0-1
        
        return strength.fillna(0).values
    
    def _calculate_distribution_strength(self, returns_window: pd.Series) -> float:
        """Calculate distribution strength for a returns window."""
        if len(returns_window) < 3:
            return 0.0
        
        # Strength based on how well-defined the distribution is
        skewness = skew(returns_window)
        kurtosis_val = kurtosis(returns_window)
        
        # Strength based on deviation from normal distribution
        deviation = abs(skewness) + abs(kurtosis_val - 3)
        return max(0, 1 - deviation / 10)  # Normalize to 0-1
    
    def _calculate_returns_autocorrelation(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate returns autocorrelation - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        returns_series = pd.Series(returns)
        
        # Vectorized autocorrelation using pandas built-in method
        autocorr = returns_series.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        return autocorr.values
    
    def _calculate_window_autocorr(self, returns_window: pd.Series) -> float:
        """Calculate autocorrelation for a returns window."""
        if len(returns_window) < 2:
            return 0.0
        
        corr = np.corrcoef(returns_window[:-1], returns_window[1:])[0, 1]
        return corr if not np.isnan(corr) else 0
    
    def _calculate_correlation_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate correlation stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized correlation stability calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling autocorrelation using vectorized operations
        returns_shifted = returns_series.shift(1)
        autocorr = returns_series.rolling(window=window).corr(returns_shifted)
        
        # Calculate stability as inverse of autocorrelation variance
        autocorr_variance = autocorr.rolling(window=window//4).std()
        stability = np.maximum(0, 1 - autocorr_variance)
        
        return stability.fillna(0).values
    
    def _calculate_correlation_stability_window(self, returns_window: np.ndarray) -> float:
        """Helper function for correlation stability calculation."""
        if len(returns_window) < 3:
            return 0.0
        
        try:
            # Calculate rolling autocorrelation
            autocorr_vals = []
            sub_window = max(2, len(returns_window) // 4)
            step = max(1, sub_window // 2)
            
            for j in range(0, len(returns_window) - sub_window, step):
                sub_returns = returns_window[j:j+sub_window]
                if len(sub_returns) > 1:
                    corr = np.corrcoef(sub_returns[:-1], sub_returns[1:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_vals.append(corr)
            
            if len(autocorr_vals) > 1:
                # Stability based on low variance of autocorrelations
                stability = max(0, 1 - np.std(autocorr_vals))
                return stability
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_cross_correlation_features(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate cross-correlation features - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        
        # OPTIMIZED: Use vectorized cross-correlation calculation
        # Calculate lagged correlation using vectorized operations
        returns_lagged = returns_series.shift(1)
        cross_corr = returns_series.rolling(window=window).corr(returns_lagged).fillna(0)
        
        return cross_corr.values
    
    def _calculate_cross_corr_window(self, returns_window: np.ndarray) -> float:
        """Helper function for cross-correlation calculation."""
        if len(returns_window) < 3:
            return 0.0
        
        try:
            # Cross-correlation between returns and absolute returns
            abs_returns = np.abs(returns_window)
            corr = np.corrcoef(returns_window, abs_returns)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _detect_statistical_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect statistical regime changes - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        
        # OPTIMIZED: Use vectorized moment calculations
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", window)
        centered = returns_series - rolling_mean
        
        # Vectorized skewness and kurtosis
        skew1 = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurt1 = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Shift to get second window
        skew2 = skew1.shift(-window)
        kurt2 = kurt1.shift(-window)
        
        # Calculate change ratios
        skew_change = ((skew2 - skew1).abs() / (skew1.abs() + 1e-8)).fillna(0)
        kurt_change = ((kurt2 - kurt1).abs() / (kurt1.abs() + 1e-8)).fillna(0)
        
        # Apply threshold (50% change)
        changes = ((skew_change > 0.5) | (kurt_change > 0.5)).astype(int)
        
        return changes.fillna(0).values
    
    def _calculate_distribution_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution transition probability - FULLY VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized distribution transition probability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 2)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate volatility of statistical moments
        skew_vol = skewness.rolling(window=window*2).std()
        kurt_vol = kurtosis.rolling(window=window*2).std()
        
        # Transition probability based on moment volatility
        transition_prob = np.minimum(1, (skew_vol + kurt_vol) / 2)
        
        return transition_prob.fillna(0).values
    
    def _calculate_transition_prob_window(self, recent_returns: np.ndarray, sub_window: int) -> float:
        """Helper function for distribution transition probability calculation."""
        if len(recent_returns) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Calculate rolling skewness and kurtosis
            skew_vals = self._calculate_rolling_skewness(recent_returns, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(recent_returns, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Probability based on volatility of statistical moments
                skew_vol = np.std(skew_vals)
                kurt_vol = np.std(kurt_vals)
                return min(1, (skew_vol + kurt_vol) / 2)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_statistical_momentum(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical momentum - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical momentum calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate trend in statistical moments using linear regression approximation
        x = np.arange(len(skewness))
        skew_trend = skewness.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        kurt_trend = kurtosis.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        momentum = (skew_trend + kurt_trend) / 2
        
        return momentum.fillna(0).values
    
    def _calculate_momentum_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical momentum calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Momentum based on trend in statistical moments
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Calculate trend in statistical moments
                x = np.arange(len(skew_vals))
                skew_trend = np.polyfit(x, skew_vals, 1)[0]
                kurt_trend = np.polyfit(x, kurt_vals, 1)[0]
                return (skew_trend + kurt_trend) / 2
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_statistical_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical stability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        skew_cv = self._vectorbt_rolling_operation(skewness, "std", window) / (self._vectorbt_rolling_operation(skewness, "mean", window).abs() + 1e-8)
        kurt_cv = self._vectorbt_rolling_operation(kurtosis, "std", window) / (self._vectorbt_rolling_operation(kurtosis, "mean", window).abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0).values
    
    def _calculate_stability_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical stability calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Stability based on consistency of statistical properties
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Stability based on low coefficient of variation
                skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
                kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
                return max(0, 1 - (skew_cv + kurt_cv) / 2)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_distribution_entropy(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution entropy - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized entropy calculation using histogram approximation
        returns_series = pd.Series(returns)
        
        # Calculate rolling min and max for bin boundaries
        rolling_min = self._vectorbt_rolling_operation(returns_series, "min", window)
        rolling_max = self._vectorbt_rolling_operation(returns_series, "max", window)
        
        # Create 10 bins for each window
        n_bins = 10
        bin_width = (rolling_max - rolling_min) / n_bins
        
        # Calculate entropy using variance approximation (much faster than histogram)
        # Entropy is approximated as log of variance for normal-like distributions
        rolling_var = self._vectorbt_rolling_operation(returns_series, "var", window)
        entropy_approx = np.log(rolling_var + 1e-8)
        
        # Normalize entropy to 0-1 range
        entropy_normalized = entropy_approx / (entropy_approx.rolling(window=window*2).std() + 1e-8)
        entropy_normalized = np.clip(entropy_normalized, 0, 1)
        
        return entropy_normalized.fillna(0).values
    
    def _calculate_entropy_window(self, returns_window: np.ndarray) -> float:
        """Helper function for distribution entropy calculation."""
        if len(returns_window) < 2:
            return 0.0
        
        try:
            # Calculate entropy of returns distribution
            # Discretize returns into bins
            bins = np.linspace(returns_window.min(), returns_window.max(), 10)
            hist, _ = np.histogram(returns_window, bins=bins)
            # Normalize to probabilities
            probs = hist / (np.sum(hist) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            return entropy
        except:
            return 0.0
    
    def _calculate_statistical_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical consistency - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical consistency calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = self._vectorbt_rolling_operation(returns_series, "mean", sub_window)
        rolling_std = self._vectorbt_rolling_operation(returns_series, "std", sub_window)
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate autocorrelation of statistical moments using vectorized operations
        skewness_shifted = skewness.shift(1)
        kurtosis_shifted = kurtosis.shift(1)
        
        skew_corr = skewness.rolling(window=window).corr(skewness_shifted).fillna(0)
        kurt_corr = kurtosis.rolling(window=window).corr(kurtosis_shifted).fillna(0)
        
        consistency = (skew_corr + kurt_corr) / 2
        
        return consistency.values
    
    def _calculate_consistency_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical consistency calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Consistency based on autocorrelation of statistical moments
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                skew_corr = np.corrcoef(skew_vals[:-1], skew_vals[1:])[0, 1]
                kurt_corr = np.corrcoef(kurt_vals[:-1], kurt_vals[1:])[0, 1]
                
                skew_corr = skew_corr if not np.isnan(skew_corr) else 0
                kurt_corr = kurt_corr if not np.isnan(kurt_corr) else 0
                
                return (skew_corr + kurt_corr) / 2
            else:
                return 0.0
        except:
            return 0.0
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
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
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _vectorbt_rolling_operation_optimized(self, data: pd.Series, operation: str, 
                                            window: int, **kwargs) -> pd.Series:
        """Perform optimized VectorBT rolling operation using VectorBTRollingOptimizer."""
        if self.rolling_optimizer and ROLLING_OPTIMIZER_AVAILABLE:
            return self.rolling_optimizer._rolling_operation(data, operation, window, **kwargs)
        else:
            return self._vectorbt_rolling_operation(data, operation, window, **kwargs)
    
    def _calculate_rolling_skewness_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling skewness using optimized VectorBT operations."""
        if self.rolling_optimizer and VECTORBT_AVAILABLE:
            return self.rolling_optimizer.rolling_skew(returns, window=window)
        else:
            # Fallback to pandas implementation
            return returns.rolling(window=window).skew()
    
    def _calculate_rolling_kurtosis_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling kurtosis using optimized VectorBT operations."""
        if self.rolling_optimizer and VECTORBT_AVAILABLE:
            return self.rolling_optimizer.rolling_kurt(returns, window=window)
        else:
            # Fallback to pandas implementation
            return returns.rolling(window=window).kurt()
    
    def _calculate_normality_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate distribution normality using optimized approach."""
        if self.rolling_optimizer and VECTORBT_AVAILABLE:
            # Use VectorBT native functions
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=window)
            
            # Calculate skewness and kurtosis
            skewness = self.rolling_optimizer.rolling_skew(returns, window=window)
            kurtosis = self.rolling_optimizer.rolling_kurt(returns, window=window)
            
            # Simplified normality test using JB statistic approximation
            jb_stat = (skewness ** 2 + (kurtosis ** 2) / 4) * window / 6
            normality = np.exp(-jb_stat / 2)  # Approximate p-value
            
            return normality
        else:
            # Fallback implementation
            return self._calculate_distribution_normality_optimized(returns, window)
    
    def _calculate_distribution_normality_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate distribution normality using optimized approach."""
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=window)
        else:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
        
        # Calculate skewness and kurtosis for normality test
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Simplified normality test using JB statistic approximation
        jb_stat = (skewness ** 2 + (kurtosis ** 2) / 4) * window / 6
        normality = np.exp(-jb_stat / 2)  # Approximate p-value
        
        return normality.fillna(0)
    
    def _calculate_skewness_persistence_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate skewness persistence using optimized operations."""
        if self.rolling_optimizer:
            skewness = self.rolling_optimizer.rolling_skew(returns, window=window)
        else:
            skewness = returns.rolling(window=window).skew()
        
        # Calculate autocorrelation of skewness
        skewness_shifted = skewness.shift(1)
        skewness_autocorr = skewness.rolling(window=window//4).corr(skewness_shifted).fillna(0)
        
        return skewness_autocorr
    
    def _calculate_kurtosis_persistence_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate kurtosis persistence using optimized operations."""
        if self.rolling_optimizer:
            kurtosis = self.rolling_optimizer.rolling_kurt(returns, window=window)
        else:
            kurtosis = returns.rolling(window=window).kurt()
        
        # Calculate autocorrelation of kurtosis
        kurtosis_shifted = kurtosis.shift(1)
        kurtosis_autocorr = kurtosis.rolling(window=window//4).corr(kurtosis_shifted).fillna(0)
        
        return kurtosis_autocorr
    
    def _calculate_statistical_persistence_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate statistical regime persistence using optimized operations."""
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Calculate autocorrelation of squared returns
        squared_returns_shifted = squared_returns.shift(1)
        autocorr = squared_returns.rolling(window=window).corr(squared_returns_shifted).fillna(0)
        
        return autocorr
    
    def _calculate_distribution_stability_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate distribution stability using optimized operations."""
        sub_window = max(2, window // 4)
        
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=sub_window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=sub_window)
        else:
            rolling_mean = returns.rolling(window=sub_window).mean()
            rolling_std = returns.rolling(window=sub_window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        if self.rolling_optimizer:
            skew_cv = self.rolling_optimizer.rolling_std(skewness, window=window) / (self.rolling_optimizer.rolling_mean(skewness, window=window).abs() + 1e-8)
            kurt_cv = self.rolling_optimizer.rolling_std(kurtosis, window=window) / (self.rolling_optimizer.rolling_mean(kurtosis, window=window).abs() + 1e-8)
        else:
            skew_cv = skewness.rolling(window=window).std() / (skewness.rolling(window=window).mean().abs() + 1e-8)
            kurt_cv = kurtosis.rolling(window=window).std() / (kurtosis.rolling(window=window).mean().abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0)
    
    def _calculate_statistical_strength_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate statistical regime strength using optimized operations."""
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=window)
        else:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Strength based on deviation from normal distribution
        deviation = np.abs(skewness) + np.abs(kurtosis - 3)
        strength = np.maximum(0, 1 - deviation / 10)  # Normalize to 0-1
        
        return strength.fillna(0)
    
    def _calculate_returns_autocorrelation_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate returns autocorrelation using optimized operations."""
        if self.rolling_optimizer:
            # Use VectorBT optimized correlation
            returns_shifted = returns.shift(1)
            autocorr = self.rolling_optimizer.rolling_corr(returns, returns_shifted, window=window)
        else:
            # Fallback to pandas implementation
            autocorr = returns.rolling(window=window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            ).fillna(0)
        
        return autocorr
    
    def _calculate_correlation_stability_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate correlation stability using optimized operations."""
        returns_shifted = returns.shift(1)
        
        if self.rolling_optimizer:
            autocorr = self.rolling_optimizer.rolling_corr(returns, returns_shifted, window=window)
            autocorr_variance = self.rolling_optimizer.rolling_std(autocorr, window=window//4)
        else:
            autocorr = returns.rolling(window=window).corr(returns_shifted)
            autocorr_variance = autocorr.rolling(window=window//4).std()
        
        # Calculate stability as inverse of autocorrelation variance
        stability = np.maximum(0, 1 - autocorr_variance)
        
        return stability.fillna(0)
    
    def _calculate_cross_correlation_features_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate cross-correlation features using optimized operations."""
        returns_lagged = returns.shift(1)
        
        if self.rolling_optimizer:
            cross_corr = self.rolling_optimizer.rolling_corr(returns, returns_lagged, window=window)
        else:
            cross_corr = returns.rolling(window=window).corr(returns_lagged)
        
        return cross_corr.fillna(0)
    
    def _detect_statistical_regime_changes_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Detect statistical regime changes using optimized operations."""
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=window)
        else:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
        
        centered = returns - rolling_mean
        skew1 = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurt1 = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Shift to get second window
        skew2 = skew1.shift(-window)
        kurt2 = kurt1.shift(-window)
        
        # Calculate change ratios
        skew_change = ((skew2 - skew1).abs() / (skew1.abs() + 1e-8)).fillna(0)
        kurt_change = ((kurt2 - kurt1).abs() / (kurt1.abs() + 1e-8)).fillna(0)
        
        # Apply threshold (50% change)
        changes = ((skew_change > 0.5) | (kurt_change > 0.5)).astype(int)
        
        return changes.fillna(0)
    
    def _calculate_distribution_transition_probability_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate distribution transition probability using optimized operations."""
        sub_window = max(2, window // 2)
        
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=sub_window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=sub_window)
        else:
            rolling_mean = returns.rolling(window=sub_window).mean()
            rolling_std = returns.rolling(window=sub_window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate volatility of statistical moments
        skew_vol = skewness.rolling(window=window*2).std()
        kurt_vol = kurtosis.rolling(window=window*2).std()
        
        # Transition probability based on moment volatility
        transition_prob = np.minimum(1, (skew_vol + kurt_vol) / 2)
        
        return transition_prob.fillna(0)
    
    def _calculate_statistical_momentum_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate statistical momentum using optimized operations."""
        sub_window = max(2, window // 4)
        
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=sub_window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=sub_window)
        else:
            rolling_mean = returns.rolling(window=sub_window).mean()
            rolling_std = returns.rolling(window=sub_window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate trend in statistical moments using linear regression approximation
        x = np.arange(len(skewness))
        skew_trend = skewness.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        kurt_trend = kurtosis.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        momentum = (skew_trend + kurt_trend) / 2
        
        return momentum.fillna(0)
    
    def _calculate_statistical_stability_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate statistical stability using optimized operations."""
        sub_window = max(2, window // 4)
        
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=sub_window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=sub_window)
        else:
            rolling_mean = returns.rolling(window=sub_window).mean()
            rolling_std = returns.rolling(window=sub_window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        if self.rolling_optimizer:
            skew_cv = self.rolling_optimizer.rolling_std(skewness, window=window) / (self.rolling_optimizer.rolling_mean(skewness, window=window).abs() + 1e-8)
            kurt_cv = self.rolling_optimizer.rolling_std(kurtosis, window=window) / (self.rolling_optimizer.rolling_mean(kurtosis, window=window).abs() + 1e-8)
        else:
            skew_cv = skewness.rolling(window=window).std() / (skewness.rolling(window=window).mean().abs() + 1e-8)
            kurt_cv = kurtosis.rolling(window=window).std() / (kurtosis.rolling(window=window).mean().abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0)
    
    def _calculate_distribution_entropy_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate distribution entropy using optimized operations."""
        if self.rolling_optimizer:
            rolling_min = self.rolling_optimizer.rolling_min(returns, window=window)
            rolling_max = self.rolling_optimizer.rolling_max(returns, window=window)
            rolling_var = self.rolling_optimizer.rolling_var(returns, window=window)
        else:
            rolling_min = returns.rolling(window=window).min()
            rolling_max = returns.rolling(window=window).max()
            rolling_var = returns.rolling(window=window).var()
        
        # Calculate entropy using variance approximation (much faster than histogram)
        entropy_approx = np.log(rolling_var + 1e-8)
        
        # Normalize entropy to 0-1 range
        entropy_normalized = entropy_approx / (entropy_approx.rolling(window=window*2).std() + 1e-8)
        entropy_normalized = np.clip(entropy_normalized, 0, 1)
        
        return entropy_normalized.fillna(0)
    
    def _calculate_statistical_consistency_optimized(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate statistical consistency using optimized operations."""
        sub_window = max(2, window // 4)
        
        if self.rolling_optimizer:
            rolling_mean = self.rolling_optimizer.rolling_mean(returns, window=sub_window)
            rolling_std = self.rolling_optimizer.rolling_std(returns, window=sub_window)
        else:
            rolling_mean = returns.rolling(window=sub_window).mean()
            rolling_std = returns.rolling(window=sub_window).std()
        
        centered = returns - rolling_mean
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate autocorrelation of statistical moments
        skewness_shifted = skewness.shift(1)
        kurtosis_shifted = kurtosis.shift(1)
        
        if self.rolling_optimizer:
            skew_corr = self.rolling_optimizer.rolling_corr(skewness, skewness_shifted, window=window)
            kurt_corr = self.rolling_optimizer.rolling_corr(kurtosis, kurtosis_shifted, window=window)
        else:
            skew_corr = skewness.rolling(window=window).corr(skewness_shifted)
            kurt_corr = kurtosis.rolling(window=window).corr(kurtosis_shifted)
        
        consistency = (skew_corr + kurt_corr) / 2
        
        return consistency.fillna(0)
    
    def _pad_series(self, series: pd.Series, target_length: int, window: int) -> np.ndarray:
        """Pad series to match target length."""
        if len(series) == target_length:
            return series.fillna(0).values
        
        padded = np.full(target_length, np.nan)
        valid_indices = min(len(series), target_length - window)
        
        if valid_indices > 0:
            padded[window:window + valid_indices] = series[:valid_indices].values
        
        return np.nan_to_num(padded, nan=0.0)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['batch_usage_percentage'] = (
                stats['batch_operations'] / stats['total_operations'] * 100
            )
            stats['chunked_usage_percentage'] = (
                stats['chunked_operations'] / stats['total_operations'] * 100
            )
            stats['gpu_usage_percentage'] = (
                stats['gpu_operations'] / stats['total_operations'] * 100
            )
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / (stats['cache_hits'] + stats['total_operations']) * 100
            )
            stats['average_operation_time'] = (
                stats['total_time'] / stats['total_operations']
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['batch_usage_percentage'] = 0
            stats['chunked_usage_percentage'] = 0
            stats['gpu_usage_percentage'] = 0
            stats['cache_hit_rate'] = 0
            stats['average_operation_time'] = 0
        
        return stats
    
    def optimize_feature_parameters(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Optimize feature parameters using UnifiedVectorizationManager."""
        if not self.unified_optimizer or not UNIFIED_OPTIMIZER_AVAILABLE:
            tprint("UnifiedVectorizationManager not available for parameter optimization")
            return {}
        
        try:
            from ..utils.optimization.unified_optimizer import FeatureOptimizationConfig, OptimizationMethod
            
            # Create optimization configuration
            optimization_config = FeatureOptimizationConfig(
                min_lookback=8,
                max_lookback=128,
                optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS,
                parallel_processing=True,
                enable_validation=True
            )
            
            # Get feature names
            feature_names = list(self.config.parameters.keys())
            
            results = {}
            for feature_name in feature_names:
                try:
                    # Create a simple feature generator function for optimization
                    def feature_generator(data, lookback):
                        if feature_name == "distribution_windows":
                            return self._generate_distribution_features_optimized(
                                pd.Series(np.diff(np.log(data['close'].values)), index=data.index[1:]), 
                                data
                            )
                        # Add other feature types as needed
                        return {}
                    
                    # Optimize the feature
                    result = self.unified_optimizer.optimize_feature_lookback(
                        data, feature_name, target_column, feature_generator
                    )
                    results[feature_name] = result
                    
                except Exception as e:
                    tprint(f"Error optimizing feature {feature_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            tprint(f"Error in feature parameter optimization: {e}")
            return {}
    
    def enable_batch_processing(self, enable: bool = True):
        """Enable or disable batch processing."""
        self.config.parameters['batch_processing'] = enable
        tprint(f"Batch processing {'enabled' if enable else 'disabled'}")
    
    def enable_chunked_processing(self, enable: bool = True):
        """Enable or disable chunked processing."""
        self.config.parameters['chunked_processing'] = enable
        tprint(f"Chunked processing {'enabled' if enable else 'disabled'}")
    
    def enable_gpu_acceleration(self, enable: bool = True):
        """Enable or disable GPU acceleration."""
        self.config.parameters['gpu_acceleration'] = enable
        if hasattr(self, 'enable_gpu'):
            self.enable_gpu = enable
        tprint(f"GPU acceleration {'enabled' if enable else 'disabled'}")
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'chunked_operations': 0,
            'gpu_operations': 0,
            'cache_hits': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
        tprint("Performance statistics reset")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of optimization capabilities and status."""
        summary = {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'rolling_optimizer_available': ROLLING_OPTIMIZER_AVAILABLE,
            'unified_optimizer_available': UNIFIED_OPTIMIZER_AVAILABLE,
            'vectorbt_mixin_available': VECTORBT_MIXIN_AVAILABLE,
            'gpu_available': CUPY_AVAILABLE,
            'batch_processing_enabled': self.config.parameters.get('batch_processing', False),
            'chunked_processing_enabled': self.config.parameters.get('chunked_processing', False),
            'gpu_acceleration_enabled': self.config.parameters.get('gpu_acceleration', False),
            'rolling_optimizer_initialized': self.rolling_optimizer is not None,
            'unified_optimizer_initialized': self.unified_optimizer is not None
        }
        
        return summary
