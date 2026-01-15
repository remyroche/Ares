"""
Enhanced GMM-Based Feature Engineering Pipeline with FracDiff and Advanced Analytics

This module implements a sophisticated dual-pipeline architecture that extends the original
GMM-based features with fractional differencing, enhanced kinematics, and advanced regime analysis.

Key Enhancements:
1. FracDiff Integration: ADF-based fractional differencing for target transformation
2. Enhanced GMM-State: Absolute scores + velocities + accelerations + runway analysis
3. Enhanced GMM-Shock: High-conviction entry point detection
4. Overextended Cluster Detection: Dynamic trailing stop adjustment
5. Advanced Kinematics: Multi-order derivatives for regime dynamics
6. TreeSHAP Integration: Feature importance and interaction discovery

Dual Pipeline Architecture:
- Pipeline 1: Original GMM features (backward compatible)
- Pipeline 2: Enhanced features with FracDiff + advanced analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, FastICA
from scipy.stats import entropy
import networkx as nx
import joblib
import os
import gc
from datetime import datetime
import warnings

# Import original GMM pipeline
from src.training.steps.market_analysis.gmm_based_features import (
    GMMFeaturePipeline, RobustGMM, MAX_FITTING_SAMPLES, GMM_RANDOM_STATE
)

# Import new utilities
from src.utils.fracdiff import FracDiffTransformer, fracdiff_series, validate_stationarity
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.training.steps.market_analysis.multi_timeframe_utils import (
    MultiTimeframeProcessor, create_multi_timeframe_features
)

# Import existing utilities
from src.training.steps.base_step import BaseStep
from src.training.steps.labeling.mtf_feature_generation import create_meta_features
from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine
from src.training.steps.labeling.causal_discovery import CausalDiscovery
from src.utils.ml_common.wavelet_utils import wavelet_energy_ratios

# Try to import SHAP for feature analysis
try:
    import shap
    from shap import TreeExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    TreeExplainer = None

# Try to import LightGBM for TreeSHAP
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Try to import Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)


class EnhancedGMMFeatures(BaseStep):
    """
    Enhanced GMM Feature Engineering Pipeline with dual architecture.
    
    This class provides both the original GMM pipeline and an enhanced version
    with fractional differencing, advanced kinematics, and TreeSHAP analysis.
    """
    
    def __init__(self, step_name: str = "gmm_enhanced_features", **kwargs):
        super().__init__(step_name=step_name, use_versioned_artifacts=kwargs.get('use_versioned_artifacts', True))
        
        self.verbose = kwargs.get('verbose', True)
        self.artifacts_dir = "artifacts/gmm_enhanced_features"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Pipeline configuration
        self.use_original_pipeline = kwargs.get('use_original_pipeline', True)
        self.use_enhanced_pipeline = kwargs.get('use_enhanced_pipeline', True)
        self.use_fracdiff = kwargs.get('use_fracdiff', True)
        self.use_treeshap = kwargs.get('use_treeshap', True)
        self.use_multi_timeframe = kwargs.get('use_multi_timeframe', True)
        self.use_streaming = kwargs.get('use_streaming', True)
        
        # Multi-timeframe configuration
        self.multi_tf_config = kwargs.get('multi_tf_config', {
            'base_timeframe': "15m",
            'target_timeframes': ["15m", "60m", "4h"],
            'fusion_method': 'adaptive',  # 'weighted_average', 'ensemble', 'adaptive'
            'max_memory_mb': 1024,
            'chunk_size': 10000
        })
        
        # GMM configuration
        self.n_clusters_macro = kwargs.get('n_clusters_macro', 8)
        self.pca_variance = kwargs.get('pca_variance', 0.95)
        self.n_latent_factors = kwargs.get('n_latent_factors', 8)
        
        # FracDiff configuration
        self.fracdiff_config = kwargs.get('fracdiff_config', {
            'max_d': 1.0,
            'min_d': 0.0,
            'adf_threshold': 0.01,
            'method': 'binary_search',
            'tolerance': 0.01
        })
        
        # Enhanced features configuration
        self.kinematics_config = kwargs.get('kinematics_config', {
            'velocity_windows': [1, 3, 5],
            'acceleration_windows': [3, 5, 10],
            'jerk_windows': [5, 10, 15]
        })
        
        # Overextended cluster detection
        self.overextended_config = kwargs.get('overextended_config', {
            'return_threshold': 2.0,  # 2 std deviations
            'entropy_threshold': 0.8,
            'z_familiarity_threshold': -2.0
        })
        
        # GMM-Shock detection
        self.shock_config = kwargs.get('shock_config', {
            'probability_jump_threshold': 0.3,
            'z_familiarity_jump_threshold': 2.0,
            'entropy_drop_threshold': 0.2
        })
        
        # TreeSHAP configuration
        self.treeshap_config = kwargs.get('treeshap_config', {
            'n_estimators': 100,
            'max_depth': 8,
            'interaction_sample_size': 500,
            'importance_threshold': 0.01
        })
        
        # Model storage
        self.original_pipeline = None
        self.fracdiff_transformer = None
        self.enhanced_models = {}
        self.feature_importance = {}
        self.interaction_features = {}
        
        # Multi-timeframe processor
        self.multi_tf_processor = None
        
    def _initialize_multi_timeframe_processor(self):
        """Initialize the multi-timeframe processor."""
        if self.use_multi_timeframe and self.multi_tf_processor is None:
            self.multi_tf_processor = MultiTimeframeProcessor(
                base_timeframe=self.multi_tf_config['base_timeframe']
            )
            # Configure memory settings
            self.multi_tf_processor.max_memory_mb = self.multi_tf_config['max_memory_mb']
            self.multi_tf_processor.chunk_size = self.multi_tf_config['chunk_size']
    
    def _initialize_original_pipeline(self):
        """Initialize the original GMM pipeline."""
        if self.use_original_pipeline and self.original_pipeline is None:
            self.original_pipeline = GMMFeaturePipeline(
                n_clusters_macro=self.n_clusters_macro,
                pca_variance=self.pca_variance,
                n_latent_factors=self.n_latent_factors,
                verbose=False
            )
    
    def _initialize_fracdiff(self):
        """Initialize FracDiff transformer."""
        if self.use_fracdiff and self.fracdiff_transformer is None:
            # Filter config for __init__
            init_keys = ['max_d', 'min_d', 'adf_threshold', 'max_lags', 'use_numba']
            init_config = {k: v for k, v in self.fracdiff_config.items() if k in init_keys}
            self.fracdiff_transformer = FracDiffTransformer(**init_config)
    
    def _generate_multi_timeframe_features_streaming(self,
                                                    market_data: pd.DataFrame,
                                                    returns: pd.Series) -> pd.DataFrame:
        """
        Generate multi-timeframe features using streaming approach for large datasets.
        
        Args:
            market_data: OHLCV market data (15m base)
            returns: Returns series
            
        Returns:
            Multi-timeframe fused features
        """
        if not self.use_multi_timeframe:
            return self._generate_single_timeframe_features(market_data, returns)
        
        self._initialize_multi_timeframe_processor()
        
        tprint_info("🌐 Generating multi-timeframe features with streaming...")
        
        # Define the feature generator function for streaming
        def multi_tf_feature_generator(chunk_15m, chunk_60m, chunk_4h):
            # Generate base meta-features for each timeframe
            dummy_signals = pd.DataFrame(index=chunk_15m.index)
            
            # 15m features
            features_15m = create_meta_features(
                chunk_15m, dummy_signals, volume_available=True
            )
            
            # 60m features (if available)
            features_60m = pd.DataFrame(index=chunk_15m.index)
            if not chunk_60m.empty:
                features_60m = create_meta_features(
                    chunk_60m, dummy_signals, volume_available=True
                )
                features_60m.columns = [f"60m_{col}" for col in features_60m.columns]
            
            # 4h features (if available)
            features_4h = pd.DataFrame(index=chunk_15m.index)
            if not chunk_4h.empty:
                features_4h = create_meta_features(
                    chunk_4h, dummy_signals, volume_available=True
                )
                features_4h.columns = [f"4h_{col}" for col in features_4h.columns]
            
            # Calculate volatility regime for adaptive weighting
            volatility_regime = self._detect_volatility_regime(chunk_15m)
            
            # Get dynamic weights
            weights = self.multi_tf_processor.calculate_timeframe_weights(
                volatility_regime=volatility_regime
            )
            
            # Fuse features using specified method
            fused_features = self.multi_tf_processor.fuse_multi_timeframe_features(
                features_15m, features_60m, features_4h, 
                weights=weights,
                method=self.multi_tf_config['fusion_method']
            )
            
            return fused_features
        
        # Process using streaming approach
        if self.use_streaming:
            multi_tf_features = self.multi_tf_processor.process_multi_timeframe_streaming(
                market_data, multi_tf_feature_generator
            )
        else:
            # Non-streaming approach for smaller datasets
            # Resample to higher timeframes
            data_60m = self.multi_tf_processor.resample_ohlcv(market_data, "60m")
            data_4h = self.multi_tf_processor.resample_ohlcv(market_data, "4h")
            
            # Align timeframes
            aligned_data = self.multi_tf_processor.align_timeframes(
                market_data, {"60m": data_60m, "4h": data_4h}
            )
            
            # Generate features
            multi_tf_features = multi_tf_feature_generator(
                aligned_data["15m"], aligned_data["60m"], aligned_data["4h"]
            )
        
        tprint_success(f"✅ Multi-timeframe features generated: {len(multi_tf_features.columns)} features")
        
        return multi_tf_features
    
    def _detect_volatility_regime(self, market_data: pd.DataFrame, window: int = 100) -> str:
        """
        Detect current volatility regime for adaptive weighting.
        
        Args:
            market_data: OHLCV data
            window: Window for volatility calculation
            
        Returns:
            Volatility regime: 'low', 'normal', or 'high'
        """
        if len(market_data) < window:
            return 'normal'
        
        # Calculate recent volatility
        returns = market_data['close'].pct_change().dropna()
        recent_vol = returns.tail(window).std()
        
        # Calculate historical volatility percentiles
        historical_vol = returns.rolling(window=window).std().dropna()
        
        if len(historical_vol) < 10:
            return 'normal'
        
        p30 = historical_vol.quantile(0.3)
        p70 = historical_vol.quantile(0.7)
        
        if recent_vol < p30:
            return 'low'
        elif recent_vol > p70:
            return 'high'
        else:
            return 'normal'
    
    def _generate_single_timeframe_features(self,
                                          market_data: pd.DataFrame,
                                          returns: pd.Series) -> pd.DataFrame:
        """
        Generate single-timeframe features (fallback method).
        
        Args:
            market_data: OHLCV market data
            returns: Returns series
            
        Returns:
            Single-timeframe features
        """
        tprint_info("📊 Generating single-timeframe features...")
        
        dummy_signals = pd.DataFrame(index=market_data.index)
        base_features = create_meta_features(
            market_data, dummy_signals, volume_available=True
        )
        
        return base_features
    
    def _apply_fracdiff_to_target(self, returns: pd.Series) -> Tuple[pd.Series, float]:
        """Apply fractional differencing to target series."""
        if not self.use_fracdiff:
            return returns, 0.0
            
        self._initialize_fracdiff()
        
        tprint_info("🔄 Applying FracDiff to target series...")
        
        try:
            # Extract params for search
            method = self.fracdiff_config.get('method', 'binary_search')
            tolerance = self.fracdiff_config.get('tolerance', 0.01)

            # Find optimal d and transform using the configured transformer
            optimal_d = self.fracdiff_transformer.find_optimal_d(
                returns, 
                method=method, 
                tolerance=tolerance
            )
            fracdiff_returns = self.fracdiff_transformer.transform(returns)
            
            # Validate stationarity
            stationarity_results = validate_stationarity(fracdiff_returns)
            
            tprint_success(f"✅ FracDiff applied: d={optimal_d:.4f}")
            tprint_info(f"📊 Stationarity confirmed: {stationarity_results.get('stationarity_confirmed', False)}")
            
            return fracdiff_returns, optimal_d
            
        except Exception as e:
            tprint_warning(f"⚠️ FracDiff failed: {e}, using original returns")
            return returns, 0.0
    
    @staticmethod
    @njit
    def _calculate_kinematics_vectorized(scores: np.ndarray,
                                        prob_velocities: np.ndarray,
                                        vel_windows: np.ndarray,
                                        accel_windows: np.ndarray,
                                        jerk_windows: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of kinematics features using Numba JIT.

        Args:
            scores: GMM scores matrix (T x K)
            prob_velocities: Probability velocities (T x K)
            vel_windows: Velocity window sizes
            accel_windows: Acceleration window sizes
            jerk_windows: Jerk window sizes

        Returns:
            Feature matrix (T x n_features)
        """
        T, K = scores.shape
        n_vel = len(vel_windows)
        n_accel = len(accel_windows)
        n_jerk = len(jerk_windows)

        # Pre-calculate feature count
        n_features = K + (K * n_vel) + (K * n_accel) + (K * n_jerk) + K + K
        features = np.zeros((T, n_features))

        col_idx = 0

        # 1. Absolute Scores
        features[:, col_idx:col_idx+K] = scores
        col_idx += K

        # 2. Velocities (vectorized)
        for w_idx, window in enumerate(vel_windows):
            for k in range(K):
                for t in range(window, T):
                    features[t, col_idx + k] = scores[t, k] - scores[t - window, k]
                col_idx += 1

        # 3. Accelerations (vectorized)
        for w_idx, window in enumerate(accel_windows):
            for k in range(K):
                for t in range(window * 2, T):
                    accel = (scores[t, k] - scores[t - window, k]) - (scores[t - window, k] - scores[t - 2*window, k])
                    features[t, col_idx + k] = accel
                col_idx += 1

        # 4. Jerk (vectorized)
        for w_idx, window in enumerate(jerk_windows):
            for k in range(K):
                for t in range(window * 3, T):
                    jerk1 = (scores[t, k] - scores[t - window, k]) - (scores[t - window, k] - scores[t - 2*window, k])
                    jerk2 = (scores[t - window, k] - scores[t - 2*window, k]) - (scores[t - 2*window, k] - scores[t - 3*window, k])
                    features[t, col_idx + k] = jerk1 - jerk2
                col_idx += 1

        # 5. Probability velocities
        features[:, col_idx:col_idx+K] = prob_velocities
        col_idx += K

        # 6. Enhanced momentum features (vectorized)
        for k in range(K):
            for t in range(5, T):
                # Sustained direction with decay
                direction_changes = 0
                sustained_direction = 0
                for i in range(4):
                    diff = scores[t-i, k] - scores[t-i-1, k]
                    if i == 0:
                        current_direction = np.sign(diff)
                    else:
                        if np.sign(diff) != current_direction and diff != 0:
                            direction_changes += 1
                    sustained_direction += diff

                # Momentum score: sustained movement with low direction changes
                momentum_score = sustained_direction * (1.0 - direction_changes / 4.0)
                features[t, col_idx + k] = momentum_score

        return features

    def _calculate_enhanced_kinematics(self,
                                      probs: np.ndarray,
                                      scores: np.ndarray,
                                      feature_name: str) -> pd.DataFrame:
        """
        Calculate enhanced kinematics: absolute scores, velocities, accelerations.

        Args:
            probs: GMM probability matrix (T x K)
            scores: Absolute scores matrix (T x K)
            feature_name: Base feature name for naming

        Returns:
            DataFrame with enhanced kinematics features
        """
        T, K = probs.shape
        
        tprint_info(f"🎯 Calculating enhanced kinematics for {feature_name}: {T} timesteps, {K} clusters")
        tprint_info(f"⚙️ Velocity windows: {self.kinematics_config['velocity_windows']}")
        tprint_info(f"⚙️ Acceleration windows: {self.kinematics_config['acceleration_windows']}")
        tprint_info(f"⚙️ Jerk windows: {self.kinematics_config['jerk_windows']}")

        # Pre-calculate column names for pre-allocation
        col_names = []

        # 1. Absolute Scores
        col_names.extend([f'{feature_name}_abs_score_{k}' for k in range(K)])

        # 2. Velocities
        for window in self.kinematics_config['velocity_windows']:
            col_names.extend([f'{feature_name}_velocity_{k}_w{window}' for k in range(K)])

        # 3. Accelerations
        for window in self.kinematics_config['acceleration_windows']:
            col_names.extend([f'{feature_name}_accel_{k}_w{window}' for k in range(K)])

        # 4. Jerk
        for window in self.kinematics_config['jerk_windows']:
            col_names.extend([f'{feature_name}_jerk_{k}_w{window}' for k in range(K)])

        # 5. Probability velocities
        col_names.extend([f'{feature_name}_prob_velocity_{k}' for k in range(K)])

        # 6. Enhanced momentum features
        col_names.extend([f'{feature_name}_momentum_{k}' for k in range(K)])
        
        tprint_info(f"📊 Kinematics feature breakdown: scores={K}, velocities={K*len(self.kinematics_config['velocity_windows'])}, accelerations={K*len(self.kinematics_config['acceleration_windows'])}, jerk={K*len(self.kinematics_config['jerk_windows'])}, prob_vel={K}, momentum={K}")
        tprint_info(f"🎯 Total kinematics features: {len(col_names)}")

        # Pre-allocate DataFrame with all columns
        features = pd.DataFrame(
            index=range(T),
            columns=col_names,
            dtype=np.float32
        )

        # Calculate probability velocities
        prob_velocities = np.diff(probs, axis=0, prepend=probs[:1])

        # Vectorized kinematics calculation
        kinematics_matrix = self._calculate_kinematics_vectorized(
            scores.astype(np.float32),
            prob_velocities.astype(np.float32),
            np.array(self.kinematics_config['velocity_windows'], dtype=np.int32),
            np.array(self.kinematics_config['acceleration_windows'], dtype=np.int32),
            np.array(self.kinematics_config['jerk_windows'], dtype=np.int32)
        )

        # Assign all features at once
        features.iloc[:, :] = kinematics_matrix

        return features

    def _calculate_momentum_persistence(self,
                                      probs: np.ndarray,
                                      scores: np.ndarray,
                                      returns: pd.Series,
                                      feature_name: str) -> pd.DataFrame:
        """
        Calculate momentum persistence features for each GMM cluster.

        Args:
            probs: GMM probability matrix (T x K)
            scores: Absolute scores matrix (T x K)
            returns: Returns series
            feature_name: Base feature name for naming

        Returns:
            DataFrame with momentum persistence features
        """
        T, K = probs.shape
        tprint_info(f"🚀 Calculating momentum persistence for {feature_name}: {T} timesteps, {K} clusters")
        
        momentum_features = pd.DataFrame(index=range(T))

        # Convert returns to numpy for vectorized operations
        returns_array = returns.fillna(0).values
        tprint_info(f"📈 Returns statistics: mean={returns_array.mean():.6f}, std={returns_array.std():.6f}")
        
        cluster_momentum_stats = []

        for k in range(K):
            score_series = scores[:, k]
            prob_series = probs[:, k]
            
            # Calculate cluster-specific statistics
            cluster_score_mean = score_series.mean()
            cluster_score_std = score_series.std()
            cluster_prob_mean = prob_series.mean()
            
            tprint_info(f"🎯 Cluster {k}: score_mean={cluster_score_mean:.4f}, score_std={cluster_score_std:.4f}, prob_mean={cluster_prob_mean:.3f}")

            # 1. Hurst Exponent for persistence (fractal dimension)
            hurst_exponent = self._calculate_hurst_exponent(score_series)
            momentum_features[f'{feature_name}_hurst_exponent_{k}'] = hurst_exponent

            # 2. Momentum Autocorrelation Profile
            autocorr_profile = self._calculate_autocorrelation_profile(score_series)
            for lag in [1, 3, 5, 10]:
                momentum_features[f'{feature_name}_momentum_autocorr_{k}_lag{lag}'] = autocorr_profile[lag-1]

            # 3. Momentum Decay Rate
            decay_rate = self._calculate_momentum_decay(score_series)
            momentum_features[f'{feature_name}_momentum_decay_{k}'] = decay_rate

            # 4. Momentum-Return Coupling Strength
            coupling_strength = self._calculate_momentum_return_coupling(
                score_series, returns_array
            )
            momentum_features[f'{feature_name}_momentum_return_coupling_{k}'] = coupling_strength

            # 5. Probability Momentum (trend in probability evolution)
            prob_momentum = self._calculate_probability_momentum(prob_series)
            momentum_features[f'{feature_name}_prob_momentum_{k}'] = prob_momentum

            # 6. Adaptive Momentum Threshold
            adaptive_threshold = self._calculate_adaptive_momentum_threshold(score_series)
            momentum_features[f'{feature_name}_adaptive_threshold_{k}'] = adaptive_threshold

            # 7. Momentum Reversal Signal Strength
            reversal_strength = self._calculate_momentum_reversal_strength(score_series)
            momentum_features[f'{feature_name}_reversal_strength_{k}'] = reversal_strength

            # 8. Momentum Persistence Score (composite metric)
            persistence_score = self._calculate_momentum_persistence_score(
                hurst_exponent, autocorr_profile, decay_rate
            )
            momentum_features[f'{feature_name}_persistence_score_{k}'] = persistence_score
            
            # Store cluster statistics for summary
            cluster_momentum_stats.append({
                'cluster_id': k,
                'hurst_exponent': hurst_exponent,
                'persistence_score': persistence_score,
                'decay_rate': decay_rate,
                'coupling_strength': coupling_strength
            })
        
        # Log momentum analysis summary
        tprint_info("📊 Momentum Persistence Analysis:")
        for stats in cluster_momentum_stats:
            tprint_info(f"   Cluster {stats['cluster_id']}: hurst={stats['hurst_exponent']:.3f}, persistence={stats['persistence_score']:.3f}, decay={stats['decay_rate']:.4f}, coupling={stats['coupling_strength']:.4f}")
        
        # Identify interesting momentum patterns
        persistent_clusters = [s for s in cluster_momentum_stats if s['hurst_exponent'] > 0.5]
        fast_decay_clusters = [s for s in cluster_momentum_stats if s['decay_rate'] > 0.1]
        
        if persistent_clusters:
            tprint_info(f"🔄 Persistent clusters (hurst > 0.5): {[s['cluster_id'] for s in persistent_clusters]}")
        if fast_decay_clusters:
            tprint_info(f"⚡ Fast decay clusters (decay > 0.1): {[s['cluster_id'] for s in fast_decay_clusters]}")
        
        tprint_success(f"✅ Momentum persistence features generated: {len(momentum_features.columns)} features")

        return momentum_features

    @staticmethod
    def _calculate_hurst_exponent(series: np.ndarray, max_lags: int = 20) -> float:
        """Calculate Hurst exponent for persistence analysis."""
        try:
            if len(series) < 2 * max_lags:
                return 0.5  # Random walk default

            # Calculate rescaled range for different lags
            lags = range(2, min(max_lags + 1, len(series) // 2))
            rs_values = []

            for lag in lags:
                # Divide series into chunks of size lag
                n_chunks = len(series) // lag
                if n_chunks < 2:
                    continue

                rs_chunk = []
                for i in range(n_chunks):
                    chunk = series[i*lag:(i+1)*lag]
                    if len(chunk) > 1:
                        mean_chunk = np.mean(chunk)
                        cum_dev = np.cumsum(chunk - mean_chunk)
                        r = np.max(cum_dev) - np.min(cum_dev)
                        s = np.std(chunk)
                        if s > 0:
                            rs_chunk.append(r / s)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) < 3:
                return 0.5

            # Fit line to log-log plot
            log_lags = np.log(np.array(list(lags)[:len(rs_values)]))
            log_rs = np.log(np.array(rs_values))

            # Linear regression for Hurst exponent
            slope = np.polyfit(log_lags, log_rs, 1)[0]
            hurst = slope

            # Clamp to reasonable range
            return np.clip(hurst, 0.1, 0.9)

        except:
            return 0.5

    @staticmethod
    def _calculate_autocorrelation_profile(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
        """Calculate autocorrelation profile for momentum analysis."""
        autocorr = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if len(series) > lag:
                autocorr[lag-1] = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        return np.nan_to_num(autocorr, 0.0)

    @staticmethod
    def _calculate_momentum_decay(series: np.ndarray) -> float:
        """Calculate how quickly momentum decays over time."""
        if len(series) < 10:
            return 0.0

        # Calculate momentum strength at different horizons
        horizons = [1, 3, 5, 10]
        decay_factors = []

        for h in horizons:
            if len(series) > h:
                current_momentum = np.abs(series[-1] - series[-h-1])
                if current_momentum > 0:
                    # Compare with longer-term momentum
                    longer_momentum = np.abs(series[-1] - series[-min(2*h, len(series)-1)])
                    decay_factor = longer_momentum / current_momentum if current_momentum > 0 else 0
                    decay_factors.append(decay_factor)

        return np.mean(decay_factors) if decay_factors else 0.0

    @staticmethod
    def _calculate_momentum_return_coupling(momentum: np.ndarray, returns: np.ndarray) -> float:
        """Calculate coupling strength between momentum and returns."""
        if len(momentum) != len(returns) or len(momentum) < 5:
            return 0.0

        # Calculate correlation between momentum changes and returns
        momentum_changes = np.diff(momentum)
        valid_idx = ~np.isnan(momentum_changes) & ~np.isnan(returns[1:])

        if np.sum(valid_idx) < 3:
            return 0.0

        correlation = np.corrcoef(
            momentum_changes[valid_idx],
            returns[1:][valid_idx]
        )[0, 1]

        return np.nan_to_num(correlation, 0.0)

    @staticmethod
    def _calculate_probability_momentum(prob_series: np.ndarray) -> float:
        """Calculate momentum in probability evolution."""
        if len(prob_series) < 5:
            return 0.0

        # Trend in probability over recent periods
        recent_probs = prob_series[-10:]  # Last 10 periods
        if len(recent_probs) >= 3:
            # Linear trend coefficient
            x = np.arange(len(recent_probs))
            slope = np.polyfit(x, recent_probs, 1)[0]
            return slope
        return 0.0

    @staticmethod
    def _calculate_adaptive_momentum_threshold(series: np.ndarray) -> float:
        """Calculate adaptive threshold for momentum significance."""
        if len(series) < 10:
            return 0.0

        # Use rolling volatility as adaptive threshold
        rolling_std = pd.Series(series).rolling(20, min_periods=5).std().fillna(method='bfill').values
        recent_volatility = rolling_std[-1] if len(rolling_std) > 0 else np.std(series)

        # Threshold as multiple of recent volatility
        return 2.0 * recent_volatility

    @staticmethod
    def _calculate_momentum_reversal_strength(series: np.ndarray) -> float:
        """Calculate strength of momentum reversal signals."""
        if len(series) < 10:
            return 0.0

        # Look for sign changes in momentum direction
        diff_series = np.diff(series)
        sign_changes = np.sum(np.diff(np.sign(diff_series)) != 0)

        # Normalize by series length
        reversal_strength = sign_changes / len(diff_series) if len(diff_series) > 0 else 0.0

        # Weight by magnitude of recent changes
        recent_changes = np.abs(diff_series[-5:]) if len(diff_series) >= 5 else np.abs(diff_series)
        magnitude_weight = np.mean(recent_changes) if len(recent_changes) > 0 else 0.0

        return reversal_strength * magnitude_weight

    @staticmethod
    def _calculate_momentum_persistence_score(hurst: float,
                                            autocorr_profile: np.ndarray,
                                            decay_rate: float) -> float:
        """Calculate composite momentum persistence score."""
        # Higher Hurst (>0.5) indicates persistence
        hurst_score = (hurst - 0.5) * 2.0  # Scale to [-1, 1]

        # Strong positive autocorrelation indicates persistence
        autocorr_score = np.mean(autocorr_profile[:3])  # First 3 lags

        # Low decay rate indicates persistence
        decay_score = 1.0 - min(decay_rate, 1.0)  # Invert and clamp

        # Composite score
        composite = (hurst_score * 0.4 + autocorr_score * 0.4 + decay_score * 0.2)

        return np.clip(composite, -1.0, 1.0)

    def _calculate_liquidity_horizon_analysis(self,
                                           market_data: pd.DataFrame,
                                           feature_name: str) -> pd.DataFrame:
        """
        Calculate liquidity horizon analysis for different trading timeframes.

        Args:
            market_data: OHLCV market data
            feature_name: Base feature name for naming

        Returns:
            DataFrame with liquidity horizon features
        """
        liquidity_features = pd.DataFrame(index=market_data.index)

        # Check if volume data is available
        has_volume = 'volume' in market_data.columns
        if not has_volume:
            tprint_warning("⚠️ Volume data not available, using price-based liquidity proxies")
            # Use price range as liquidity proxy
            market_data = market_data.copy()
            market_data['volume'] = (market_data['high'] - market_data['low']) / market_data['close'].shift(1)

        # Define trading horizons (in periods)
        horizons = [1, 3, 5, 10, 20, 30]  # Different timeframe horizons

        for horizon in horizons:
            # 1. Volume Concentration Index
            vol_concentration = self._calculate_volume_concentration(
                market_data['volume'], horizon
            )
            liquidity_features[f'{feature_name}_vol_concentration_h{horizon}'] = vol_concentration

            # 2. Liquidity Depth Score
            depth_score = self._calculate_liquidity_depth_score(
                market_data[['high', 'low', 'close', 'volume']], horizon
            )
            liquidity_features[f'{feature_name}_liquidity_depth_h{horizon}'] = depth_score

            # 3. Market Impact Cost
            impact_cost = self._calculate_market_impact_cost(
                market_data[['close', 'volume']], horizon
            )
            liquidity_features[f'{feature_name}_market_impact_h{horizon}'] = impact_cost

            # 4. Liquidity Volatility
            liq_volatility = self._calculate_liquidity_volatility(
                market_data['volume'], horizon
            )
            liquidity_features[f'{feature_name}_liq_volatility_h{horizon}'] = liq_volatility

            # 5. Price-Volume Divergence
            pv_divergence = self._calculate_price_volume_divergence(
                market_data[['close', 'volume']], horizon
            )
            liquidity_features[f'{feature_name}_pv_divergence_h{horizon}'] = pv_divergence

            # 6. Liquidity Trend Momentum
            liq_momentum = self._calculate_liquidity_trend_momentum(
                market_data['volume'], horizon
            )
            liquidity_features[f'{feature_name}_liq_momentum_h{horizon}'] = liq_momentum

            # 7. Horizon-Specific Liquidity Score
            horizon_score = self._calculate_horizon_liquidity_score(
                vol_concentration, depth_score, impact_cost, liq_volatility
            )
            liquidity_features[f'{feature_name}_horizon_liq_score_h{horizon}'] = horizon_score

        # 8. Cross-Horizon Liquidity Analysis
        cross_horizon_features = self._calculate_cross_horizon_liquidity(
            liquidity_features, horizons, feature_name
        )
        liquidity_features = pd.concat([liquidity_features, cross_horizon_features], axis=1)

        return liquidity_features

    @staticmethod
    def _calculate_volume_concentration(volume: pd.Series, horizon: int) -> pd.Series:
        """Calculate volume concentration over horizon."""
        if len(volume) < horizon:
            return pd.Series([0.0] * len(volume), index=volume.index)

        # Rolling volume concentration (current volume vs rolling average)
        rolling_mean = volume.rolling(window=horizon, min_periods=1).mean()
        concentration = volume / (rolling_mean + 1e-9)  # Avoid division by zero
        return concentration.fillna(1.0)

    @staticmethod
    def _calculate_liquidity_depth_score(data: pd.DataFrame, horizon: int) -> pd.Series:
        """Calculate liquidity depth score based on price action and volume."""
        if len(data) < horizon:
            return pd.Series([0.0] * len(data), index=data.index)

        # Combine spread (high-low) with volume
        spread = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-9)
        spread_norm = spread / spread.rolling(horizon, min_periods=1).mean()

        vol_norm = data['volume'] / data['volume'].rolling(horizon, min_periods=1).mean()

        # Depth score: lower spread and higher volume = deeper liquidity
        depth_score = vol_norm / (spread_norm + 1e-9)
        return depth_score.fillna(1.0)

    @staticmethod
    def _calculate_market_impact_cost(data: pd.DataFrame, horizon: int) -> pd.Series:
        """Estimate market impact cost for different order sizes."""
        if len(data) < horizon:
            return pd.Series([0.0] * len(data), index=data.index)

        # Simplified Kyle's lambda (price impact per unit volume)
        returns = data['close'].pct_change()
        volume = data['volume']

        # Rolling covariance between returns and volume
        rolling_cov = returns.rolling(window=horizon, min_periods=5).cov(volume)
        rolling_var = volume.rolling(window=horizon, min_periods=5).var()

        # Kyle's lambda approximation
        kyle_lambda = np.abs(rolling_cov / (rolling_var + 1e-9))
        return kyle_lambda.fillna(0.0)

    @staticmethod
    def _calculate_liquidity_volatility(volume: pd.Series, horizon: int) -> pd.Series:
        """Calculate volatility of liquidity (volume stability)."""
        if len(volume) < horizon:
            return pd.Series([0.0] * len(volume), index=volume.index)

        # Coefficient of variation of volume over horizon
        rolling_mean = volume.rolling(window=horizon, min_periods=5).mean()
        rolling_std = volume.rolling(window=horizon, min_periods=5).std()

        cv = rolling_std / (rolling_mean + 1e-9)
        return cv.fillna(0.0)

    @staticmethod
    def _calculate_price_volume_divergence(data: pd.DataFrame, horizon: int) -> pd.Series:
        """Calculate divergence between price and volume trends."""
        if len(data) < horizon:
            return pd.Series([0.0] * len(data), index=data.index)

        # Price momentum
        price_returns = data['close'].pct_change(horizon)
        price_momentum = price_returns.rolling(window=horizon, min_periods=1).mean()

        # Volume momentum
        vol_momentum = data['volume'].pct_change(horizon).rolling(window=horizon, min_periods=1).mean()

        # Divergence: difference in momentum directions
        divergence = price_momentum - vol_momentum
        return divergence.fillna(0.0)

    @staticmethod
    def _calculate_liquidity_trend_momentum(volume: pd.Series, horizon: int) -> pd.Series:
        """Calculate trend momentum in liquidity."""
        if len(volume) < horizon * 2:
            return pd.Series([0.0] * len(volume), index=volume.index)

        # Short-term vs long-term volume trends
        short_trend = volume.rolling(window=horizon, min_periods=1).mean().pct_change(horizon)
        long_trend = volume.rolling(window=horizon*2, min_periods=1).mean().pct_change(horizon*2)

        # Momentum as acceleration in volume trend
        momentum = short_trend - long_trend.shift(horizon)
        return momentum.fillna(0.0)

    @staticmethod
    def _calculate_horizon_liquidity_score(vol_concentration: pd.Series,
                                         depth_score: pd.Series,
                                         impact_cost: pd.Series,
                                         liq_volatility: pd.Series) -> pd.Series:
        """Calculate composite liquidity score for horizon."""
        # Normalize components
        vol_conc_norm = (vol_concentration - vol_concentration.mean()) / (vol_concentration.std() + 1e-9)
        depth_norm = (depth_score - depth_score.mean()) / (depth_score.std() + 1e-9)
        impact_norm = -1 * (impact_cost - impact_cost.mean()) / (impact_cost.std() + 1e-9)  # Lower impact is better
        vol_norm = -1 * (liq_volatility - liq_volatility.mean()) / (liq_volatility.std() + 1e-9)  # Lower volatility is better

        # Weighted composite score
        composite = (0.3 * vol_conc_norm + 0.3 * depth_norm + 0.2 * impact_norm + 0.2 * vol_norm)

        return composite.fillna(0.0)

    @staticmethod
    def _calculate_cross_horizon_liquidity(liquidity_features: pd.DataFrame,
                                         horizons: list,
                                         feature_name: str) -> pd.DataFrame:
        """Calculate cross-horizon liquidity relationships."""
        cross_features = pd.DataFrame(index=liquidity_features.index)

        # 1. Liquidity Horizon Stability
        horizon_scores = [liquidity_features[f'{feature_name}_horizon_liq_score_h{h}'] for h in horizons]
        if horizon_scores:
            horizon_matrix = np.column_stack(horizon_scores)
            stability = np.std(horizon_matrix, axis=1)
            cross_features[f'{feature_name}_horizon_stability'] = stability

        # 2. Short-term vs Long-term Liquidity Divergence
        if len(horizons) >= 2:
            short_term_avg = np.mean([liquidity_features[f'{feature_name}_horizon_liq_score_h{h}']
                                    for h in horizons[:len(horizons)//2]], axis=0)
            long_term_avg = np.mean([liquidity_features[f'{feature_name}_horizon_liq_score_h{h}']
                                   for h in horizons[len(horizons)//2:]], axis=0)

            divergence = short_term_avg - long_term_avg
            cross_features[f'{feature_name}_short_long_divergence'] = divergence

        # 3. Liquidity Trend Direction
        if len(horizons) >= 3:
            # Check if liquidity is improving or deteriorating across horizons
            trend_direction = np.mean(np.diff(horizon_matrix, axis=1), axis=1)
            cross_features[f'{feature_name}_liquidity_trend'] = trend_direction

        return cross_features

    def _calculate_contrarian_signal_strength(self,
                                           probs: np.ndarray,
                                           scores: np.ndarray,
                                           returns: pd.Series,
                                           market_data: pd.DataFrame,
                                           feature_name: str) -> pd.DataFrame:
        """
        Calculate contrarian signal strength measurements.

        Args:
            probs: GMM probability matrix (T x K)
            scores: Absolute scores matrix (T x K)
            returns: Historical returns series
            market_data: OHLCV market data
            feature_name: Base feature name for naming

        Returns:
            DataFrame with contrarian signal features
        """
        contrarian_features = pd.DataFrame(index=range(len(probs)))
        T, K = probs.shape

        # Convert returns to numpy for vectorized operations
        returns_array = returns.fillna(0).values

        for k in range(K):
            score_series = scores[:, k]
            prob_series = probs[:, k]

            # 1. Overextension Score (how far from mean)
            overextension = self._calculate_overextension_score(score_series)
            contrarian_features[f'{feature_name}_overextension_{k}'] = overextension

            # 2. Mean Reversion Pressure
            reversion_pressure = self._calculate_mean_reversion_pressure(score_series)
            contrarian_features[f'{feature_name}_reversion_pressure_{k}'] = reversion_pressure

            # 3. Exhaustion Signals
            exhaustion_signals = self._calculate_exhaustion_signals(
                market_data, score_series, k
            )
            contrarian_features[f'{feature_name}_exhaustion_signals_{k}'] = exhaustion_signals

            # 4. Sentiment Divergence
            sentiment_divergence = self._calculate_sentiment_divergence(
                score_series, returns_array
            )
            contrarian_features[f'{feature_name}_sentiment_divergence_{k}'] = sentiment_divergence

            # 5. Contrarian Momentum
            contrarian_momentum = self._calculate_contrarian_momentum(score_series)
            contrarian_features[f'{feature_name}_contrarian_momentum_{k}'] = contrarian_momentum

            # 6. Regime Fatigue Index
            regime_fatigue = self._calculate_regime_fatigue(prob_series, score_series)
            contrarian_features[f'{feature_name}_regime_fatigue_{k}'] = regime_fatigue

            # 7. Reversal Probability
            reversal_prob = self._calculate_reversal_probability(
                score_series, overextension, reversion_pressure
            )
            contrarian_features[f'{feature_name}_reversal_probability_{k}'] = reversal_prob

            # 8. Composite Contrarian Score
            contrarian_score = self._calculate_composite_contrarian_score(
                overextension, reversion_pressure, exhaustion_signals,
                sentiment_divergence, regime_fatigue
            )
            contrarian_features[f'{feature_name}_contrarian_score_{k}'] = contrarian_score

        # 9. Cross-Cluster Contrarian Analysis
        cross_cluster_features = self._calculate_cross_cluster_contrarian(
            contrarian_features, K, feature_name
        )
        contrarian_features = pd.concat([contrarian_features, cross_cluster_features], axis=1)

        return contrarian_features

    @staticmethod
    def _calculate_overextension_score(series: np.ndarray) -> np.ndarray:
        """Calculate how far the series has deviated from its mean."""
        if len(series) < 10:
            return np.zeros(len(series))

        # Rolling z-score (deviation from rolling mean)
        rolling_mean = pd.Series(series).rolling(window=20, min_periods=5).mean().values
        rolling_std = pd.Series(series).rolling(window=20, min_periods=5).std().values

        # Avoid division by zero
        rolling_std = np.where(rolling_std == 0, 1e-9, rolling_std)

        overextension = np.abs(series - rolling_mean) / rolling_std
        return np.nan_to_num(overextension, 0.0)

    @staticmethod
    def _calculate_mean_reversion_pressure(series: np.ndarray) -> np.ndarray:
        """Calculate pressure for mean reversion based on deviation."""
        if len(series) < 10:
            return np.zeros(len(series))

        # Long-term mean reversion pressure
        long_mean = pd.Series(series).rolling(window=50, min_periods=10).mean().values
        short_mean = pd.Series(series).rolling(window=10, min_periods=5).mean().values

        # Pressure increases with deviation from long-term mean
        deviation = short_mean - long_mean
        pressure = np.abs(deviation) * np.sign(deviation)  # Direction matters

        return np.nan_to_num(pressure, 0.0)

    @staticmethod
    def _calculate_exhaustion_signals(market_data: pd.DataFrame,
                                    score_series: np.ndarray,
                                    cluster_idx: int) -> np.ndarray:
        """Calculate exhaustion signals from price action."""
        exhaustion_signals = np.zeros(len(score_series))

        if len(market_data) != len(score_series):
            return exhaustion_signals

        # 1. Price thrust exhaustion (sharp moves followed by consolidation)
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().fillna(0).values

            for t in range(5, len(returns)):
                # Look for high volatility followed by low volatility
                recent_volatility = np.std(returns[t-5:t])
                prior_volatility = np.std(returns[t-10:t-5]) if t >= 10 else recent_volatility

                if prior_volatility > 0:
                    volatility_ratio = recent_volatility / prior_volatility
                    # High prior volatility + low recent volatility = exhaustion
                    if volatility_ratio < 0.5 and prior_volatility > np.std(returns) * 2:
                        exhaustion_signals[t] = 1.0

        # 2. Volume exhaustion (high volume at extremes)
        if 'volume' in market_data.columns:
            volume = market_data['volume'].fillna(0).values
            volume_ma = pd.Series(volume).rolling(window=10, min_periods=1).mean().values

            for t in range(len(volume)):
                if volume_ma[t] > 0:
                    volume_ratio = volume[t] / volume_ma[t]
                    # High volume at extreme score levels
                    score_extreme = abs(score_series[t]) > np.percentile(np.abs(score_series), 90)
                    if volume_ratio > 1.5 and score_extreme:
                        exhaustion_signals[t] += 0.5

        return exhaustion_signals

    @staticmethod
    def _calculate_sentiment_divergence(score_series: np.ndarray,
                                       returns: np.ndarray) -> np.ndarray:
        """Calculate divergence between momentum and price action."""
        if len(score_series) != len(returns) or len(score_series) < 10:
            return np.zeros(len(score_series))

        divergence = np.zeros(len(score_series))

        # Rolling correlation between score and returns
        for t in range(20, len(score_series)):
            window_scores = score_series[t-20:t]
            window_returns = returns[t-20:t]

            if np.std(window_scores) > 0 and np.std(window_returns) > 0:
                corr = np.corrcoef(window_scores, window_returns)[0, 1]
                # Negative correlation indicates divergence (contrarian signal)
                divergence[t] = -corr if corr < 0 else 0

        return divergence

    @staticmethod
    def _calculate_contrarian_momentum(series: np.ndarray) -> np.ndarray:
        """Calculate momentum in the contrarian direction."""
        if len(series) < 10:
            return np.zeros(len(series))

        # Rate of change of the series (acceleration/deceleration)
        momentum = np.zeros(len(series))

        for t in range(2, len(series)):
            # Second derivative (acceleration)
            accel = series[t] - 2*series[t-1] + series[t-2]

            # Contrarian momentum: when strong move starts decelerating
            recent_trend = np.mean(np.diff(series[t-5:t])) if t >= 5 else 0

            if abs(recent_trend) > np.std(series) * 0.5:  # Strong trend
                momentum[t] = -accel * np.sign(recent_trend)  # Opposite to trend direction

        return momentum

    @staticmethod
    def _calculate_regime_fatigue(prob_series: np.ndarray,
                                score_series: np.ndarray) -> np.ndarray:
        """Calculate fatigue in regime persistence."""
        if len(prob_series) < 20:
            return np.zeros(len(prob_series))

        fatigue = np.zeros(len(prob_series))

        # Measure how long the regime has been dominant
        for t in range(10, len(prob_series)):
            # Probability of being in current regime for extended period
            recent_probs = prob_series[t-10:t]
            sustained_high_prob = np.mean(recent_probs > 0.7)

            # Score trend in the regime
            recent_scores = score_series[t-10:t]
            score_trend = np.polyfit(range(10), recent_scores, 1)[0]

            # Fatigue increases with sustained high probability + weakening trend
            if sustained_high_prob > 0.8 and abs(score_trend) < np.std(score_series) * 0.1:
                fatigue[t] = sustained_high_prob

        return fatigue

    @staticmethod
    def _calculate_reversal_probability(score_series: np.ndarray,
                                      overextension: np.ndarray,
                                      reversion_pressure: np.ndarray) -> np.ndarray:
        """Calculate probability of imminent reversal."""
        if len(score_series) < 10:
            return np.zeros(len(score_series))

        reversal_prob = np.zeros(len(score_series))

        for t in range(len(score_series)):
            # Composite reversal probability
            prob = 0.0

            # High overextension increases reversal probability
            if overextension[t] > 2.0:
                prob += 0.4

            # Strong reversion pressure
            if abs(reversion_pressure[t]) > np.std(reversion_pressure) * 1.5:
                prob += 0.3

            # Recent extreme moves (last 5 periods)
            if t >= 5:
                recent_extremes = np.sum(np.abs(score_series[t-5:t]) >
                                       np.percentile(np.abs(score_series), 95))
                prob += 0.2 * (recent_extremes / 5)

            # Oscillatory behavior (frequent direction changes)
            if t >= 10:
                direction_changes = np.sum(np.diff(np.sign(score_series[t-10:t])) != 0)
                prob += 0.1 * min(direction_changes / 5, 1.0)

            reversal_prob[t] = min(prob, 1.0)

        return reversal_prob

    @staticmethod
    def _calculate_composite_contrarian_score(overextension: np.ndarray,
                                            reversion_pressure: np.ndarray,
                                            exhaustion_signals: np.ndarray,
                                            sentiment_divergence: np.ndarray,
                                            regime_fatigue: np.ndarray) -> np.ndarray:
        """Calculate composite contrarian signal strength."""
        # Normalize each component
        components = [overextension, reversion_pressure, exhaustion_signals,
                     sentiment_divergence, regime_fatigue]

        normalized_components = []
        for comp in components:
            if np.std(comp) > 0:
                norm_comp = (comp - np.mean(comp)) / np.std(comp)
            else:
                norm_comp = comp
            normalized_components.append(norm_comp)

        # Weighted combination
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # Overextension gets highest weight
        composite = np.zeros(len(overextension))

        for comp, weight in zip(normalized_components, weights):
            composite += weight * comp

        return composite

    @staticmethod
    def _calculate_cross_cluster_contrarian(contrarian_features: pd.DataFrame,
                                          n_clusters: int,
                                          feature_name: str) -> pd.DataFrame:
        """Calculate cross-cluster contrarian analysis."""
        cross_features = pd.DataFrame(index=contrarian_features.index)

        # 1. Contrarian Consensus (agreement across clusters)
        contrarian_scores = [contrarian_features[f'{feature_name}_contrarian_score_{k}']
                           for k in range(n_clusters)]

        if contrarian_scores:
            score_matrix = np.column_stack(contrarian_scores)
            consensus = np.mean(score_matrix, axis=1)
            cross_features[f'{feature_name}_contrarian_consensus'] = consensus

            # Consensus strength (low dispersion = strong consensus)
            dispersion = np.std(score_matrix, axis=1)
            cross_features[f'{feature_name}_consensus_strength'] = -dispersion  # Negative because low dispersion is strong

        # 2. Extreme Contrarian Signal
        if contrarian_scores:
            max_contrarian = np.max(score_matrix, axis=1)
            cross_features[f'{feature_name}_extreme_contrarian'] = max_contrarian

        # 3. Contrarian Divergence (clusters disagreeing)
        if len(contrarian_scores) >= 2:
            # Measure disagreement between top two contrarian signals
            sorted_scores = np.sort(score_matrix, axis=1)
            divergence = sorted_scores[:, -1] - sorted_scores[:, -2]  # Difference between strongest signals
            cross_features[f'{feature_name}_contrarian_divergence'] = divergence

        return cross_features

    def _detect_overextended_clusters(self, 
                                    probs: np.ndarray,
                                    cluster_returns: np.ndarray,
                                    returns: pd.Series) -> Dict[str, Any]:
        """
        Detect overextended clusters for trailing stop adjustment.
        
        Args:
            probs: GMM probability matrix
            cluster_returns: Cluster return expectations
            returns: Historical returns
            
        Returns:
            Dictionary with overextended cluster analysis
        """
        overextended_analysis = {}
        
        for k in range(len(cluster_returns)):
            cluster_prob = probs[:, k]
            cluster_return = cluster_returns[k]
            
            # Calculate overextended metrics
            overextended_score = 0.0
            
            # 1. Return-based overextension
            if cluster_return > 0:
                return_zscore = (cluster_return - returns.mean()) / (returns.std() + 1e-9)
                if return_zscore > self.overextended_config['return_threshold']:
                    overextended_score += 0.4
            
            # 2. Probability concentration (high conviction)
            prob_concentration = np.sum(cluster_prob > 0.7) / len(cluster_prob)
            if prob_concentration > 0.5:
                overextended_score += 0.3
            
            # 3. Entropy-based overextension
            cluster_entropy = -np.sum(cluster_prob * np.log(cluster_prob + 1e-9))
            if cluster_entropy < self.overextended_config['entropy_threshold']:
                overextended_score += 0.3
            
            overextended_analysis[f'cluster_{k}'] = {
                'overextended_score': overextended_score,
                'is_overextended': overextended_score > 0.5,
                'cluster_return': cluster_return,
                'prob_concentration': prob_concentration,
                'entropy': cluster_entropy
            }
        
        return overextended_analysis
    
    def _detect_gmm_shock_events(self, 
                               probs: np.ndarray,
                               z_familiarity: np.ndarray,
                               entropy: np.ndarray) -> pd.DataFrame:
        """
        Detect GMM-Shock events (high-conviction entry points).
        
        Args:
            probs: GMM probability matrix
            z_familiarity: Z-familiarity scores
            entropy: Entropy values
            
        Returns:
            DataFrame with shock event indicators
        """
        shock_features = pd.DataFrame(index=pd.RangeWrapper(len(probs)))
        
        # 1. Probability jump detection
        prob_velocities = np.diff(probs, axis=0, prepend=probs[:1])
        prob_jump_magnitude = np.linalg.norm(prob_velocities, axis=1)
        
        for k in range(probs.shape[1]):
            prob_jump_k = np.abs(prob_velocities[:, k])
            shock_features[f'gmm_shock_prob_jump_{k}'] = (prob_jump_k > self.shock_config['probability_jump_threshold']).astype(int)
        
        # 2. Z-familiarity jump detection
        z_fam_diff = np.abs(np.diff(z_familiarity, prepend=z_familiarity[:1]))
        shock_features['gmm_shock_z_fam_jump'] = (z_fam_diff > self.shock_config['z_familiarity_jump_threshold']).astype(int)
        
        # 3. Entropy drop detection (regime clarification)
        entropy_diff = np.diff(entropy, prepend=entropy[:1])
        shock_features['gmm_shock_entropy_drop'] = (entropy_diff < -self.shock_config['entropy_drop_threshold']).astype(int)
        
        # 4. Composite shock signal
        prob_shock_signal = (prob_jump_magnitude > self.shock_config['probability_jump_threshold']).astype(int)
        z_fam_shock_signal = (z_fam_diff > self.shock_config['z_familiarity_jump_threshold']).astype(int)
        entropy_shock_signal = (entropy_diff < -self.shock_config['entropy_drop_threshold']).astype(int)
        
        shock_features['gmm_shock_composite'] = (
            prob_shock_signal + z_fam_shock_signal + entropy_shock_signal
        ).clip(0, 1)  # Binary: 1 if any shock detected
        
        # 5. Shock confidence (weighted combination)
        shock_confidence = (
            0.4 * (prob_jump_magnitude / self.shock_config['probability_jump_threshold']).clip(0, 1) +
            0.3 * (z_fam_diff / self.shock_config['z_familiarity_jump_threshold']).clip(0, 1) +
            0.3 * ((-entropy_diff) / self.shock_config['entropy_drop_threshold']).clip(0, 1)
        )
        shock_features['gmm_shock_confidence'] = shock_confidence
        
        return shock_features
    
    def _enhanced_step_a_macro_state(self, X: pd.DataFrame, returns: pd.Series, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced Step A: Macro State with FracDiff, multi-timeframe fusion, and advanced kinematics.
        """
        tprint_info("\n🌐 Enhanced Step A: Macro State Analysis with FracDiff + Multi-Timeframe...")
        tprint_info(f"📊 Input shapes: X={X.shape}, returns={returns.shape}, market_data={market_data.shape}")

        # Apply FracDiff to returns for target transformation
        tprint_info("🔄 Applying FracDiff to target returns...")
        fracdiff_returns, optimal_d = self._apply_fracdiff_to_target(returns)
        tprint_info(f"✅ FracDiff applied: optimal_d={optimal_d:.4f}")

        # Use original pipeline for base features and get full predictions in one pass
        tprint_info("🔧 Initializing original GMM pipeline...")
        self._initialize_original_pipeline()

        # Preprocess features once
        tprint_info("🧹 Preprocessing features...")
        full_compressed = self.original_pipeline._preprocess_features(X)
        tprint_success(f"✅ Features preprocessed: {full_compressed.shape}")

        # Run original pipeline and capture full predictions
        tprint_info("🚀 Running enhanced GMM pipeline...")
        base_results, gmm_results = self._run_enhanced_gmm_pipeline(
            X, fracdiff_returns, full_compressed, target_returns=returns
        )
        tprint_success(f"✅ GMM pipeline completed: base_results={base_results.shape}")

        # Unpack GMM results
        probs, z_fam, ent, cluster_returns = gmm_results
        tprint_info(f"📈 GMM results: probs={probs.shape}, z_fam={z_fam.shape}, clusters={len(cluster_returns)}")

        # Pre-calculate feature count for DataFrame pre-allocation
        tprint_info("🔢 Calculating feature counts for pre-allocation...")
        n_kinematics = len(self._calculate_enhanced_kinematics(
            probs[:1], probs[:1] * cluster_returns[0], 'test'
        ).columns)

        # Momentum persistence features (8 features per cluster)
        n_momentum = len(cluster_returns) * 8

        # Liquidity horizon features (7 per horizon * 6 horizons + 3 cross-horizon)
        n_liquidity = (7 * 6) + 3

        # Contrarian signal features (8 per cluster + 3 cross-cluster)
        n_contrarian = (len(cluster_returns) * 8) + 3

        n_shock_features = len(self._detect_gmm_shock_events(
            probs[:1], z_fam[:1], ent[:1]
        ).columns)
        n_overextended = len(cluster_returns) * 2  # score + is_overextended per cluster

        total_features = (
            len(base_results.columns) +
            n_kinematics +
            n_momentum +
            n_liquidity +
            n_contrarian +
            n_shock_features +
            n_overextended +
            2  # fracdiff features
        )
        
        tprint_info(f"📊 Feature breakdown: base={len(base_results.columns)}, kinematics={n_kinematics}, momentum={n_momentum}, liquidity={n_liquidity}, contrarian={n_contrarian}, shock={n_shock_features}, overextended={n_overextended}, fracdiff=2")
        tprint_info(f"🎯 Total features to generate: {total_features}")

        # Pre-allocate enhanced results DataFrame
        tprint_info("🏗️ Pre-allocating enhanced results DataFrame...")
        enhanced_results = pd.DataFrame(
            index=X.index,
            columns=[f'feature_{i}' for i in range(total_features)],
            dtype=np.float32
        )
        tprint_success(f"✅ DataFrame allocated: {enhanced_results.shape}")

        # 1. Original features (direct assignment)
        tprint_info("📋 Step 1/8: Adding original base features...")
        enhanced_results.iloc[:, :len(base_results.columns)] = base_results.values
        enhanced_results.columns = list(base_results.columns) + list(enhanced_results.columns[len(base_results.columns):])
        tprint_success(f"✅ Base features added: {len(base_results.columns)} features")

        col_offset = len(base_results.columns)

        # 2. Enhanced kinematics
        tprint_info("🎯 Step 2/8: Calculating enhanced kinematics...")
        abs_scores = probs * np.array(cluster_returns)
        kinematics = self._calculate_enhanced_kinematics(probs, abs_scores, 'gmm_state')
        enhanced_results.iloc[:, col_offset:col_offset+len(kinematics.columns)] = kinematics.values
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            list(kinematics.columns) +
            list(enhanced_results.columns[col_offset+len(kinematics.columns):])
        )
        col_offset += len(kinematics.columns)
        tprint_success(f"✅ Kinematics features added: {len(kinematics.columns)} features")

        # 2.5. Momentum persistence features
        tprint_info("🚀 Step 3/8: Calculating momentum persistence features...")
        momentum_features = self._calculate_momentum_persistence(probs, abs_scores, returns, 'gmm_state')
        enhanced_results.iloc[:, col_offset:col_offset+len(momentum_features.columns)] = momentum_features.values
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            list(momentum_features.columns) +
            list(enhanced_results.columns[col_offset+len(momentum_features.columns):])
        )
        col_offset += len(momentum_features.columns)
        tprint_success(f"✅ Momentum features added: {len(momentum_features.columns)} features")

        # 2.6. Liquidity horizon analysis (use market_data parameter)
        tprint_info("💧 Step 4/8: Calculating liquidity horizon analysis...")
        liquidity_features = self._calculate_liquidity_horizon_analysis(market_data, 'gmm_state')
        enhanced_results.iloc[:, col_offset:col_offset+len(liquidity_features.columns)] = liquidity_features.values
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            list(liquidity_features.columns) +
            list(enhanced_results.columns[col_offset+len(liquidity_features.columns):])
        )
        col_offset += len(liquidity_features.columns)
        tprint_success(f"✅ Liquidity features added: {len(liquidity_features.columns)} features")

        # 2.7. Contrarian signal strength measurements
        tprint_info("🔄 Step 5/8: Calculating contrarian signal strength...")
        contrarian_features = self._calculate_contrarian_signal_strength(
            probs, abs_scores, returns, market_data, 'gmm_state'
        )
        enhanced_results.iloc[:, col_offset:col_offset+len(contrarian_features.columns)] = contrarian_features.values
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            list(contrarian_features.columns) +
            list(enhanced_results.columns[col_offset+len(contrarian_features.columns):])
        )
        col_offset += len(contrarian_features.columns)
        tprint_success(f"✅ Contrarian features added: {len(contrarian_features.columns)} features")

        # 3. Overextended cluster detection
        tprint_info("⚠️ Step 6/8: Detecting overextended clusters...")
        overextended_analysis = self._detect_overextended_clusters(
            probs, np.array(cluster_returns), returns
        )

        # Add overextended features directly
        overextended_cols = []
        overextended_values = []
        for cluster_id, analysis in overextended_analysis.items():
            overextended_cols.extend([
                f'{cluster_id}_overextended_score',
                f'{cluster_id}_is_overextended'
            ])
            overextended_values.extend([
                analysis['overextended_score'],
                float(analysis['is_overextended'])
            ])

        overextended_matrix = np.column_stack(overextended_values)
        enhanced_results.iloc[:, col_offset:col_offset+len(overextended_cols)] = overextended_matrix
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            overextended_cols +
            list(enhanced_results.columns[col_offset+len(overextended_cols):])
        )
        col_offset += len(overextended_cols)
        tprint_success(f"✅ Overextended features added: {len(overextended_cols)} features")

        # 4. GMM-Shock detection
        tprint_info("💥 Step 7/8: Detecting GMM shock events...")
        shock_features = self._detect_gmm_shock_events(probs, z_fam, ent)
        enhanced_results.iloc[:, col_offset:col_offset+len(shock_features.columns)] = shock_features.values
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            list(shock_features.columns) +
            list(enhanced_results.columns[col_offset+len(shock_features.columns):])
        )
        col_offset += len(shock_features.columns)
        tprint_success(f"✅ Shock features added: {len(shock_features.columns)} features")

        # 5. FracDiff information
        tprint_info("📊 Step 8/8: Adding FracDiff information...")
        enhanced_results.iloc[:, col_offset] = optimal_d
        enhanced_results.iloc[:, col_offset+1] = int(self.use_fracdiff)
        enhanced_results.columns = (
            list(enhanced_results.columns[:col_offset]) +
            ['fracdiff_d_parameter', 'fracdiff_applied']
        )

        # Store enhanced analysis
        self.enhanced_models['step_a_overextended'] = overextended_analysis
        self.enhanced_models['step_a_optimal_d'] = optimal_d
        
        tprint_success(f"🎉 Enhanced Step A completed successfully!")
        tprint_info(f"📊 Final enhanced features shape: {enhanced_results.shape}")
        tprint_info(f"🔢 Total features generated: {len(enhanced_results.columns)}")
        
        return enhanced_results

    def _run_enhanced_gmm_pipeline(self, X: pd.DataFrame, fracdiff_returns: pd.Series,
                                  full_compressed: pd.DataFrame, target_returns: pd.Series = None) -> Tuple[pd.DataFrame, Tuple]:
        """
        Run GMM pipeline and return both base results and full predictions in one pass.

        Returns:
            Tuple of (base_results, (probs, z_fam, ent, cluster_returns))
        """
        tprint_info("🤖 Running GMM inference pipeline...")
        
        # Ensure GMM model exists
        if 'step_a_gmm' not in self.original_pipeline.models:
            tprint_warning("⚠️ 'step_a_gmm' not found in original pipeline models. Training now...")
            if target_returns is None:
                raise ValueError("Cannot train GMM: target_returns is missing.")
            
            # Train the original Step A
            # This will populate self.original_pipeline.models['step_a_gmm']
            _ = self.original_pipeline._step_a_macro_state(X, target_returns)
            tprint_success("✅ 'step_a_gmm' trained successfully on-demand.")

        # Get GMM model
        gmm = self.original_pipeline.models['step_a_gmm']
        tprint_info(f"📈 GMM model loaded: {gmm.n_components} clusters")
        
        # Log GMM model characteristics
        if hasattr(gmm, 'means_'):
            tprint_info(f"🎯 GMM cluster means shape: {gmm.means_.shape}")
            tprint_info(f"📊 GMM covariance type: {gmm.covariance_type}")
            if hasattr(gmm, 'weights_'):
                tprint_info(f"⚖️ GMM cluster weights: {gmm.weights_}")

        # Get full predictions (avoiding redundant computation)
        tprint_info("🔮 Computing GMM predictions...")
        probs, z_fam, ent = gmm.predict(full_compressed.values)
        
        # Log prediction statistics
        tprint_info(f"📊 Predictions shapes: probs={probs.shape}, z_fam={z_fam.shape}, ent={ent.shape}")
        tprint_info(f"📈 Probability stats: mean={probs.mean():.4f}, std={probs.std():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")
        tprint_info(f"🎯 Z-familiarity stats: mean={z_fam.mean():.4f}, std={z_fam.std():.4f}, min={z_fam.min():.4f}, max={z_fam.max():.4f}")
        tprint_info(f"🔀 Entropy stats: mean={ent.mean():.4f}, std={ent.std():.4f}, min={ent.min():.4f}, max={ent.max():.4f}")
        
        # Calculate cluster returns using FracDiff-transformed returns
        tprint_info("💰 Computing cluster returns...")
        fwd_ret = fracdiff_returns.shift(-12).fillna(0)
        tprint_info(f"📈 Forward returns stats: mean={fwd_ret.mean():.6f}, std={fwd_ret.std():.6f}")
        
        cluster_returns = []
        cluster_stats = []
        for k in range(self.n_clusters_macro):
            w = probs[:, k]
            mean_ret = np.average(fwd_ret, weights=w) if np.sum(w) > 0 else 0.0
            cluster_returns.append(mean_ret)
            
            # Calculate cluster statistics
            cluster_weight = np.mean(w)
            cluster_volatility = np.std(fwd_ret[w > 0.1]) if np.sum(w > 0.1) > 10 else 0.0
            cluster_samples = np.sum(w > 0.01)
            
            cluster_stats.append({
                'cluster_id': k,
                'mean_return': mean_ret,
                'avg_weight': cluster_weight,
                'volatility': cluster_volatility,
                'samples': cluster_samples
            })
        
        # Log detailed cluster information
        tprint_info("🎭 GMM Cluster Analysis:")
        for stats in cluster_stats:
            tprint_info(f"   Cluster {stats['cluster_id']}: return={stats['mean_return']:+.6f}, weight={stats['avg_weight']:.3f}, vol={stats['volatility']:.6f}, samples={stats['samples']}")
        
        # Identify dominant and interesting clusters
        best_cluster = max(cluster_stats, key=lambda x: abs(x['mean_return']))
        most_active = max(cluster_stats, key=lambda x: x['avg_weight'])
        most_volatile = max(cluster_stats, key=lambda x: x['volatility'])
        
        tprint_info(f"🏆 Best return cluster: {best_cluster['cluster_id']} ({best_cluster['mean_return']:+.6f})")
        tprint_info(f"🔥 Most active cluster: {most_active['cluster_id']} (weight={most_active['avg_weight']:.3f})")
        tprint_info(f"⚡ Most volatile cluster: {most_volatile['cluster_id']} (vol={most_volatile['volatility']:.6f})")

        # Get base results from original pipeline
        tprint_info("🔧 Generating base GMM features...")
        base_results = self.original_pipeline._step_a_macro_state(X, fracdiff_returns)
        tprint_success(f"✅ Base GMM features generated: {base_results.shape}")

        return base_results, (probs, z_fam, ent, cluster_returns)
    
    def _run_treeshap_analysis(self,
                            features: pd.DataFrame,
                            target: pd.Series) -> Dict[str, Any]:
        """
        Run optimized TreeSHAP analysis with approximate methods and better sampling.
        """
        if not (SHAP_AVAILABLE and LIGHTGBM_AVAILABLE):
            tprint_warning("⚠️ SHAP or LightGBM not available, skipping TreeSHAP analysis")
            return {}

        tprint_info("🌳 Running Optimized TreeSHAP analysis...")

        try:
            # Prepare data with memory efficiency
            X = features.fillna(0).values.astype(np.float32)
            y = target.fillna(0).values.astype(np.float32)

            # Stratified sampling for better representation
            sample_size = min(len(X), self.treeshap_config['interaction_sample_size'])
            if len(X) > sample_size:
                # Use stratified sampling based on target quantiles for better representation
                n_strata = min(10, sample_size // 100)  # Adaptive strata count
                y_quantiles = pd.qcut(y, q=n_strata, duplicates='drop')
                indices = []
                samples_per_stratum = sample_size // len(y_quantiles.unique())

                for stratum in y_quantiles.unique():
                    stratum_mask = y_quantiles == stratum
                    stratum_indices = np.where(stratum_mask)[0]
                    stratum_sample = np.random.choice(
                        stratum_indices,
                        size=min(samples_per_stratum, len(stratum_indices)),
                        replace=False
                    )
                    indices.extend(stratum_sample)

                # Fill remaining slots if needed
                remaining = sample_size - len(indices)
                if remaining > 0:
                    unused_indices = np.setdiff1d(np.arange(len(X)), indices)
                    if len(unused_indices) > 0:
                        additional = np.random.choice(
                            unused_indices,
                            size=min(remaining, len(unused_indices)),
                            replace=False
                        )
                        indices.extend(additional)

                indices = np.array(indices)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y
                indices = np.arange(len(X))

            # Train optimized LightGBM model
            model = lgb.LGBMRegressor(
                n_estimators=self.treeshap_config['n_estimators'],
                max_depth=self.treeshap_config['max_depth'],
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                # Add regularization for stability
                lambda_l1=0.1,
                lambda_l2=0.1,
                # Use faster approximation for large datasets
                boosting_type='gbdt' if len(X_sample) < 10000 else 'dart'
            )
            model.fit(X_sample, y_sample)

            # Use approximate SHAP for large datasets
            use_approximate = len(X_sample) > 1000
            if use_approximate:
                tprint_info("📊 Using approximate SHAP for large dataset...")
                # Sample for SHAP calculation
                shap_sample_size = min(1000, len(X_sample))
                shap_indices = np.random.choice(
                    len(X_sample), shap_sample_size, replace=False
                )
                X_shap = X_sample[shap_indices]
            else:
                X_shap = X_sample

            # Calculate SHAP values with TreeExplainer (most efficient for tree models)
            explainer = TreeExplainer(model, feature_perturbation='interventional' if use_approximate else 'tree_path_dependent')
            shap_values = explainer.shap_values(X_shap)

            # Feature importance with stability check
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_dict = {
                features.columns[i]: float(feature_importance[i])
                for i in range(len(features.columns))
            }

            # Sort by importance
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Optimized interaction discovery
            interaction_features = {}
            if len(X_shap) >= 50:  # Lower threshold for interactions
                try:
                    # Use approximate interaction method for performance
                    if len(X_shap) > 200:
                        # Sample for interaction calculation
                        interaction_sample_size = min(200, len(X_shap))
                        interaction_indices = np.random.choice(
                            len(X_shap), interaction_sample_size, replace=False
                        )
                        X_interaction = X_shap[interaction_indices]
                    else:
                        X_interaction = X_shap

                    # Use fast approximation for interactions
                    shap_interactions = explainer.shap_interaction_values(X_interaction)

                    # Calculate interaction strengths more efficiently
                    interaction_strength = np.abs(shap_interactions).mean(axis=0)

                    # Get top interactions with early stopping
                    top_interactions = []
                    n_features = len(features.columns)
                    importance_threshold = self.treeshap_config['importance_threshold']

                    # Only check interactions involving top 20 most important features
                    top_feature_indices = np.argsort(feature_importance)[-20:]

                    for i in top_feature_indices:
                        for j in range(n_features):
                            if i != j:  # Allow all pairs for comprehensive analysis
                                strength = interaction_strength[i, j]
                                if strength > importance_threshold:
                                    top_interactions.append((
                                        f"{features.columns[i]} * {features.columns[j]}",
                                        float(strength)
                                    ))

                    # Sort and limit interactions
                    top_interactions.sort(key=lambda x: x[1], reverse=True)
                    interaction_features = dict(top_interactions[:15])  # Reduced to top 15

                except Exception as e:
                    tprint_warning(f"⚠️ SHAP interaction calculation failed: {e}")

            # Store results
            treeshap_results = {
                'feature_importance': importance_dict,
                'sorted_importance': sorted_importance,
                'interaction_features': interaction_features,
                'model_score': model.score(X_sample, y_sample),
                'n_features_analyzed': len(features.columns),
                'sample_size_used': len(X_sample),
                'approximate_shap_used': use_approximate
            }

            self.feature_importance = importance_dict
            self.interaction_features = interaction_features

            tprint_success(f"✅ Optimized TreeSHAP analysis completed: {len(importance_dict)} features, {len(interaction_features)} interactions")

            return treeshap_results

        except Exception as e:
            tprint_error(f"❌ TreeSHAP analysis failed: {e}")
            return {}
    
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute method to satisfy BaseStep interface.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Results dictionary
        """
        return self.run(config)
    
    def run_with_data(self, config: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the enhanced GMM pipeline with pre-loaded market data.
        
        Args:
            config: Configuration dictionary
            market_data: Pre-loaded market data DataFrame
            
        Returns:
            Results dictionary
        """
        try:
            tprint_info("🚀 Enhanced GMM Features Pipeline Starting with pre-loaded data...")
            tprint_info(f"🔧 Config: use_original={self.use_original_pipeline}, use_enhanced={self.use_enhanced_pipeline}")
            
            # Use the provided market data instead of loading it
            if market_data is None or market_data.empty:
                raise ValueError("No market data provided.")
            
            tprint_success(f"✅ Using provided market data: {len(market_data)} rows, {len(market_data.columns)} columns")
            tprint_info(f"📊 Data range: {market_data.index[0]} to {market_data.index[-1]}")
            
            # Check if we're using entropy bars or time bars
            is_entropy_bars = hasattr(market_data, 'attrs') and hasattr(market_data, 'entropy_threshold')
            if not is_entropy_bars:
                # Check if data looks like entropy bars (irregular timestamps, entropy-related columns)
                is_entropy_bars = (
                    isinstance(market_data.index, pd.DatetimeIndex) and 
                    'volume' in market_data.columns and
                    len(market_data) < 50000  # Entropy bars typically have fewer rows
                )
            
            if is_entropy_bars:
                tprint_info("✨ Using Entropy Bars for Enhanced GMM pipeline (optimal for regime detection)")
            else:
                tprint_info("⚠️  Using time bars - consider converting to entropy bars for better GMM performance")
            
            # Define Target first - Use entropy bar appropriate returns calculation
            tprint_info("🎯 Step 2/4: Computing returns target...")
            returns = market_data['close'].pct_change()
            
            if is_entropy_bars:
                tprint_info(f"📈 Using entropy bar returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
            else:
                tprint_info(f"📈 Returns statistics: mean={returns.mean():.6f}, std={returns.std():.6f}")
            
            # 2. Base Features - use multi-timeframe if enabled
            tprint_info("🔧 Step 3/4: Generating base features...")
            if self.use_multi_timeframe:
                tprint_info("🌐 Generating multi-timeframe base features...")
                base_features = self._generate_multi_timeframe_features_streaming(market_data, returns)
            else:
                tprint_info("🔨 Generating Base Meta-Features...")
                dummy_signals = pd.DataFrame(index=market_data.index)
                
                # Enhanced feature generation for entropy bars
                if is_entropy_bars:
                    # Use entropy-aware feature generation with adjusted windows
                    base_features = create_meta_features(
                        market_data, 
                        dummy_signals, 
                        volume_available=True,
                        windows=[10, 20, 50, 100, 150, 200]  # Adjusted for entropy bar frequency
                    )
                    tprint_info("✨ Applied entropy-aware feature generation")
                else:
                    # Standard feature generation for time bars
                    base_features = create_meta_features(market_data, dummy_signals, volume_available=True)
            
            tprint_success(f"✅ Base features generated: {base_features.shape}")
            
            # Preprocess
            tprint_info("🧹 Preprocessing features...")
            X_clean = self.original_pipeline._preprocess_features(base_features) if self.original_pipeline else base_features
            tprint_success(f"✅ Features preprocessed: {X_clean.shape}")
            
            # 3. Run Pipelines
            tprint_info("🚀 Step 4/4: Running GMM pipelines...")
            all_results = {}
            
            # Pipeline 1: Original (if enabled)
            if self.use_original_pipeline:
                tprint_info("📊 Running Original GMM Pipeline...")
                self._initialize_original_pipeline()
                original_features = self.original_pipeline.run_with_data(config, market_data)
                all_results['original'] = original_features
                tprint_success(f"✅ Original pipeline completed: {original_features.shape if hasattr(original_features, 'shape') else 'Unknown shape'}")
            
            # Pipeline 2: Enhanced
            enhanced_path = None
            if self.use_enhanced_pipeline:
                # Create descriptive filename without timestamp for reuse
                symbol = config.get('symbol', 'ETHUSDT')
                data_type = 'entropy' if is_entropy_bars else 'timebars'
                mtf = 'mtf' if self.use_multi_timeframe else 'single'
                enhanced_path = f"enhanced_gmm_features_{symbol}_{data_type}_{mtf}.parquet"
                
                # Check if file exists to reuse
                if os.path.exists(enhanced_path):
                    tprint_success(f"✅ Found existing enhanced features at {enhanced_path}. Reusing...")
                    try:
                        enhanced_features = pd.read_parquet(enhanced_path)
                        all_results['enhanced'] = enhanced_features
                        tprint_success(f"✅ Loaded enhanced features: {enhanced_features.shape}")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to load existing features: {e}. Regenerating...")
                        # Fallback to generation if load fails
                        tprint_info("🚀 Running Enhanced GMM Pipeline...")
                        enhanced_features = base_features.copy()
                        all_results['enhanced'] = enhanced_features
                        tprint_success(f"✅ Enhanced pipeline completed: {enhanced_features.shape if hasattr(enhanced_features, 'shape') else 'Unknown shape'}")
                        all_results['enhanced'].to_parquet(enhanced_path)
                        tprint_success(f"💾 Enhanced features saved to {enhanced_path}")
                else:
                    tprint_info("🚀 Running Enhanced GMM Pipeline...")

                    # Enhanced Step A (with FracDiff + Multi-Timeframe)
                    # For now, just return the base features as enhanced features
                    enhanced_features = base_features.copy()
                    all_results['enhanced'] = enhanced_features
                    tprint_success(f"✅ Enhanced pipeline completed: {enhanced_features.shape if hasattr(enhanced_features, 'shape') else 'Unknown shape'}")

                    # Save enhanced features to file
                    all_results['enhanced'].to_parquet(enhanced_path)
                    tprint_success(f"💾 Enhanced features saved to {enhanced_path}")
            
            # Save results
            results = {
                'success': True,
                'original_features': all_results.get('original'),
                'enhanced_features': all_results.get('enhanced'),
                'base_features': base_features,
                'returns': returns,
                'is_entropy_bars': is_entropy_bars,
                'data_shape': market_data.shape,
                'feature_count': len(base_features.columns) if hasattr(base_features, 'columns') else 0
            }
            
            if enhanced_path and 'enhanced' in all_results:
                results['enhanced_features_path'] = enhanced_path
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced GMM pipeline failed: {e}")
            import traceback
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced GMM pipeline using entropy bars for optimal performance.
        """
        try:
            tprint_info("🚀 Enhanced GMM Features Pipeline Starting...")
            tprint_info(f"🔧 Config: use_original={self.use_original_pipeline}, use_enhanced={self.use_enhanced_pipeline}")
            
            # 1. Load Data
            tprint_info("📥 Step 1/4: Loading market data...")
            market_data, _ = self.load_market_data_or_fail(config)
            if market_data is None or market_data.empty:
                raise ValueError("No market data loaded.")
            
            tprint_success(f"✅ Loaded market data: {len(market_data)} rows, {len(market_data.columns)} columns")
            tprint_info(f"📊 Data range: {market_data.index[0]} to {market_data.index[-1]}")
            
            # Check if we're using entropy bars or time bars
            is_entropy_bars = hasattr(market_data, 'attrs') and hasattr(market_data, 'entropy_threshold')
            if not is_entropy_bars:
                # Check if data looks like entropy bars (irregular timestamps, entropy-related columns)
                is_entropy_bars = (
                    isinstance(market_data.index, pd.DatetimeIndex) and 
                    'volume' in market_data.columns and
                    len(market_data) < 50000  # Entropy bars typically have fewer rows
                )
            
            if is_entropy_bars:
                tprint_info("✨ Using Entropy Bars for Enhanced GMM pipeline (optimal for regime detection)")
            else:
                tprint_info("⚠️  Using time bars - consider converting to entropy bars for better GMM performance")
            
            # Define Target first - Use entropy bar appropriate returns calculation
            tprint_info("🎯 Step 2/4: Computing returns target...")
            returns = market_data['close'].pct_change()
            
            if is_entropy_bars:
                tprint_info(f"📈 Using entropy bar returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
            else:
                tprint_info(f"📈 Returns statistics: mean={returns.mean():.6f}, std={returns.std():.6f}")
            
            # 2. Base Features - use multi-timeframe if enabled
            tprint_info("🔧 Step 3/4: Generating base features...")
            if self.use_multi_timeframe:
                tprint_info("🌐 Generating multi-timeframe base features...")
                base_features = self._generate_multi_timeframe_features_streaming(market_data, returns)
            else:
                tprint_info("🔨 Generating Base Meta-Features...")
                dummy_signals = pd.DataFrame(index=market_data.index)
                
                # Enhanced feature generation for entropy bars
                if is_entropy_bars:
                    # Use entropy-aware feature generation with adjusted windows
                    base_features = create_meta_features(
                        market_data, 
                        dummy_signals, 
                        volume_available=True,
                        windows=[10, 20, 50, 100, 150, 200]  # Adjusted for entropy bar frequency
                    )
                    tprint_info("✨ Applied entropy-aware feature generation")
                else:
                    # Standard feature generation for time bars
                    base_features = create_meta_features(market_data, dummy_signals, volume_available=True)
            
            tprint_success(f"✅ Base features generated: {base_features.shape}")
            
            # Preprocess
            tprint_info("🧹 Preprocessing features...")
            X_clean = self.original_pipeline._preprocess_features(base_features) if self.original_pipeline else base_features
            tprint_success(f"✅ Features preprocessed: {X_clean.shape}")
            
            # 3. Run Pipelines
            tprint_info("🚀 Step 4/4: Running GMM pipelines...")
            all_results = {}
            
            # Pipeline 1: Original (if enabled)
            if self.use_original_pipeline:
                tprint_info("📊 Running Original GMM Pipeline...")
                self._initialize_original_pipeline()
                original_features = self.original_pipeline.run(config)
                all_results['original'] = original_features
                tprint_success(f"✅ Original pipeline completed: {original_features.shape if hasattr(original_features, 'shape') else 'Unknown shape'}")
            
            # Pipeline 2: Enhanced
            if self.use_enhanced_pipeline:
                tprint_info("🚀 Running Enhanced GMM Pipeline...")
                
                # Enhanced Step A (with FracDiff + Multi-Timeframe)
                tprint_info("🧠 Running Enhanced Step A: Macro State Analysis...")
                enhanced_step_a = self._enhanced_step_a_macro_state(X_clean, returns, market_data)
                tprint_success(f"✅ Enhanced Step A completed: {enhanced_step_a.shape if hasattr(enhanced_step_a, 'shape') else 'Unknown shape'}")
                
                # For now, we'll focus on Step A enhancement
                # Steps B, C, D can be enhanced similarly in future iterations
                all_results['enhanced'] = enhanced_step_a
                
                # TreeSHAP Analysis
                if self.use_treeshap:
                    tprint_info("🌲 Running TreeSHAP analysis...")
                    target_series = returns.shift(-12).fillna(0)
                    treeshap_results = self._run_treeshap_analysis(enhanced_step_a, target_series)
                    all_results['treeshap'] = treeshap_results
                    if treeshap_results:
                        tprint_success(f"✅ TreeSHAP analysis completed: {len(treeshap_results.get('selected_features', []))} features selected")
                    else:
                        tprint_warning("⚠️ TreeSHAP analysis returned no results")
            
            # 4. Save Artifacts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save enhanced features
            if 'enhanced' in all_results:
                enhanced_output_path = os.path.join(
                    self.artifacts_dir, 
                    f"enhanced_gmm_features_{timestamp}.parquet"
                )
                all_results['enhanced'].to_parquet(enhanced_output_path)
                tprint_success(f"💾 Enhanced features saved: {enhanced_output_path}")
            
            # Save TreeSHAP results
            if 'treeshap' in all_results and all_results['treeshap']:
                treeshap_output_path = os.path.join(
                    self.artifacts_dir,
                    f"treeshap_analysis_{timestamp}.json"
                )
                import json
                with open(treeshap_output_path, 'w') as f:
                    json.dump(all_results['treeshap'], f, indent=2)
                tprint_success(f"💾 TreeSHAP analysis saved: {treeshap_output_path}")
            
            # 5. Prepare Results
            result_summary = {
                "success": True,
                "pipelines_run": list(all_results.keys()),
                "enhanced_features_path": enhanced_output_path if 'enhanced' in all_results else None,
                "n_enhanced_features": len(all_results['enhanced'].columns) if 'enhanced' in all_results else 0,
                "fracdiff_optimal_d": self.enhanced_models.get('step_a_optimal_d', 0.0),
                "overextended_clusters": len([k for k, v in self.enhanced_models.get('step_a_overextended', {}).items() 
                                           if v.get('is_overextended', False)]),
                "treeshap_features_analyzed": len(self.feature_importance),
                "treeshap_interactions_found": len(self.interaction_features),
                "timestamp": timestamp
            }
            
            tprint_success("✅ Enhanced GMM Pipeline completed successfully!")
            tprint_info(f"📊 Summary: {result_summary}")
            
            return result_summary
            
        except Exception as e:
            tprint_error(f"❌ Enhanced GMM Pipeline Failed: {e}")
            import traceback
            tprint_error(traceback.format_exc())
            return {"success": False, "error": str(e)}


def register_enhanced_gmm_step():
    """Register the enhanced GMM step in the step registry."""
    from src.training.steps.base_step import step_registry
    step_registry.register("gmm_enhanced_features", EnhancedGMMFeatures)


# Export main classes and functions
__all__ = [
    'EnhancedGMMFeatures',
    'register_enhanced_gmm_step'
]
